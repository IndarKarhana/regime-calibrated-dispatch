#!/usr/bin/env python3
"""Paper-ready experiments: comprehensive robustness, ablation, and sensitivity analyses.

Runs five experiment blocks (any subset via --parts):
  A: OSRM fleet-adjusted comparison (fix under-provisioning diagnosis)
  B: Fleet sensitivity sweep (show robustness across fleet sizes)
  C: Multi-seed statistical robustness (CIs for headline numbers)
  D: Similarity component ablation (isolate each metric's contribution)
  E: Distributional fairness / tail metrics (P95, P99, Gini)

Usage:
  python scripts/paper_experiments.py               # all parts
  python scripts/paper_experiments.py --parts A B    # specific parts
  python scripts/paper_experiments.py --parts C --seeds 5  # more seeds
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.events import annotate_events
from src.regime.similarity import query_library, compute_similarity
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.engine import SimulationEngine
from src.simulator.routing import HaversineClient
from src.policies.batch import BatchMatchingPolicy
from src.policies.anticipatory import AnticipatoryReposition
from src.evaluation.metrics import compute_kpis


# ── Shared helpers ───────────────────────────────────────────────────────────

NYC_SCENARIOS = {
    "jan_weekday_am":  ("2024-01", "08-12", "weekday"),
    "jan_weekday_pm":  ("2024-01", "16-20", "weekday"),
    "jan_nye_am":      ("2024-01", "08-12", "nye"),
    "jan_nye_pm":      ("2024-01", "16-20", "nye"),
    "jan_weekend_mid": ("2024-01", "12-16", "weekend"),
    "jun_weekday_am":  ("2024-06", "08-12", "weekday"),
    "jun_weekday_pm":  ("2024-06", "16-20", "weekday"),
    "jun_late_night":  ("2024-06", "20-24", "fri_night"),
}

# Subset for time-sensitive experiments
REPRESENTATIVE = ["jan_weekday_pm", "jun_weekday_am", "jun_late_night"]


def _pick_block(blocks, month_prefix, hour_range, day_type):
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        if not bid.startswith(month_prefix):
            continue
        if hour_range not in bid:
            continue
        if blk["request_count"].sum() < 500:
            continue
        date_str = bid.rsplit("_", 1)[0]
        ts = pd.Timestamp(date_str)
        dow = ts.dayofweek
        if day_type == "weekday" and dow < 5 and not (ts.month == 1 and ts.day == 1):
            return blk
        elif day_type == "weekend" and dow >= 5:
            return blk
        elif day_type == "nye" and ts.month == 1 and ts.day == 1:
            return blk
        elif day_type == "fri_night" and dow in (4, 5):
            return blk
    return None


def _get_replay_trips(all_trips, bid):
    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    mask = all_trips["pickup_datetime"].dt.date == target_date
    if end_h > start_h:
        mask &= (all_trips["pickup_datetime"].dt.hour >= start_h)
        mask &= (all_trips["pickup_datetime"].dt.hour < end_h)
    else:
        mask &= ((all_trips["pickup_datetime"].dt.hour >= start_h) |
                  (all_trips["pickup_datetime"].dt.hour < end_h))
    return all_trips[mask]


def _run_one(stream, fleet, horizon, seed, router, repo_policy=None):
    dispatch = BatchMatchingPolicy()
    engine = SimulationEngine(
        demand_stream=stream, policy=dispatch, router=router,
        fleet_size=fleet, horizon_seconds=horizon, seed=seed,
        reposition_policy=repo_policy, reposition_interval_steps=6,
    )
    t0 = time.time()
    state = engine.run()
    kpi = compute_kpis(state, time.time() - t0)
    # Add tail metrics
    waits = np.array(state.metrics.wait_times) if state.metrics.wait_times else np.array([0.0])
    kpi["p99_wait_s"] = float(np.percentile(waits, 99)) if len(waits) > 1 else 0.0
    kpi["max_wait_s"] = float(np.max(waits))
    # Gini coefficient of wait times (fairness)
    if len(waits) > 1 and np.sum(waits) > 0:
        sorted_w = np.sort(waits)
        n = len(sorted_w)
        index = np.arange(1, n + 1)
        kpi["gini_wait"] = float((2 * np.sum(index * sorted_w) - (n + 1) * np.sum(sorted_w)) /
                                  (n * np.sum(sorted_w)))
    else:
        kpi["gini_wait"] = 0.0
    return kpi


def _prepare_scenario(block, all_trips, library):
    """Prepare matching/prior/trips for a given block."""
    bid = block["block_id"].iloc[0]
    q_series = block["request_count"].values.astype(np.float64)
    q_events = annotate_events(q_series)

    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    horizon = float((end_h - start_h) * 3600)

    replay_trips = _get_replay_trips(all_trips, bid)
    n_trips = len(replay_trips)
    fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

    matched = query_library(library, q_series, q_events, q_block_id=bid)
    mrecs = [library[b] for b, _ in matched]
    mscores = [s for _, s in matched]
    prior = build_calibrated_prior(mrecs, mscores, q_series)

    return {
        "bid": bid, "q_series": q_series, "q_events": q_events,
        "horizon": horizon, "replay_trips": replay_trips, "n_trips": n_trips,
        "fleet": fleet, "prior": prior, "matched": matched,
    }


def _run_config(prep, config_name, seed, router, fleet_override=None):
    """Run one configuration (replay or cal+lp) and return KPI dict."""
    fleet = fleet_override or prep["fleet"]
    horizon = prep["horizon"]
    rng = np.random.default_rng(seed)

    if config_name == "replay":
        stream = ReplayDemandStream(prep["replay_trips"])
        repo = None
    elif config_name == "cal_only":
        pr = prior_matched_to_replay_volume(prep["prior"], horizon, prep["n_trips"])
        stream = CalibratedDemandStream(pr, horizon, rng=rng)
        repo = None
    elif config_name == "cal+lp":
        pr = prior_matched_to_replay_volume(prep["prior"], horizon, prep["n_trips"])
        stream = CalibratedDemandStream(pr, horizon, rng=rng)
        repo = AnticipatoryReposition(pr, lookahead_minutes=5.0, max_move_fraction=0.50)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    return _run_one(stream, fleet, horizon, seed, router, repo)


# ── Part A: OSRM fleet-adjusted comparison ───────────────────────────────────

def run_part_a(all_blocks, all_trips, library):
    """OSRM vs Haversine with fleet scaled to match effective utilization."""
    from src.simulator.routing import GridCachedOSRMClient

    def check_osrm():
        import requests
        cfg = get_config()["osrm"]
        try:
            r = requests.get(f"{cfg['host']}:{cfg['port']}/nearest/v1/car/-73.985,40.748", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    print("\n" + "=" * 80)
    print("PART A: OSRM FLEET-ADJUSTED COMPARISON")
    print("=" * 80)

    if not check_osrm():
        print("  [SKIP] OSRM not reachable. Run: docker compose up -d osrm")
        return pd.DataFrame()

    print("  Building grid-cached OSRM router ...")
    osrm_router = GridCachedOSRMClient(grid_size=25)
    haver_router = HaversineClient()

    # Test fleet scaling factors: 1.0x (original, under-provisioned under OSRM),
    # 1.25x, 1.45x (median TT ratio), 1.6x
    fleet_scales = [1.0, 1.25, 1.45, 1.6]
    test_scenarios = ["jan_weekday_pm", "jun_weekday_am"]
    seeds = [42, 123, 456]

    rows = []
    for scen_name in test_scenarios:
        month, hr, dtype = NYC_SCENARIOS[scen_name]
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            print(f"  [SKIP] {scen_name}: no block found")
            continue

        prep = _prepare_scenario(block, all_trips, library)
        base_fleet = prep["fleet"]
        print(f"\n  {scen_name} [{prep['bid']}] base_fleet={base_fleet}")

        for scale in fleet_scales:
            fleet = int(base_fleet * scale)
            for router_name, router in [("haversine", haver_router), ("osrm", osrm_router)]:
                for config in ["replay", "cal+lp"]:
                    seed_kpis = []
                    for seed in seeds:
                        kpi = _run_config(prep, config, seed, router, fleet_override=fleet)
                        seed_kpis.append(kpi)

                    # Average across seeds
                    avg_kpi = {}
                    for k in seed_kpis[0]:
                        vals = [sk[k] for sk in seed_kpis]
                        if isinstance(vals[0], (int, float)):
                            avg_kpi[k] = float(np.mean(vals))
                            avg_kpi[f"{k}_std"] = float(np.std(vals))
                    avg_kpi["scenario"] = scen_name
                    avg_kpi["config"] = config
                    avg_kpi["router"] = router_name
                    avg_kpi["fleet_scale"] = scale
                    avg_kpi["fleet_actual"] = fleet
                    rows.append(avg_kpi)

                    print(f"    {router_name:10s} {config:10s} fleet={fleet:4d} (×{scale:.2f}): "
                          f"wait={avg_kpi['mean_wait_s']:6.1f}s  "
                          f"comp={avg_kpi['completion_rate']:.1%}  "
                          f"idle={avg_kpi['mean_idle_s_per_driver']:.0f}s")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  FLEET-ADJUSTED OSRM SUMMARY:")
        for scale in fleet_scales:
            sub = df[df["fleet_scale"] == scale]
            for router in ["haversine", "osrm"]:
                replay_wait = sub[(sub["router"] == router) & (sub["config"] == "replay")]["mean_wait_s"].mean()
                cal_wait = sub[(sub["router"] == router) & (sub["config"] == "cal+lp")]["mean_wait_s"].mean()
                if replay_wait > 0:
                    imp = (replay_wait - cal_wait) / replay_wait * 100
                    print(f"    ×{scale:.2f} {router:10s}: replay={replay_wait:.1f}s  "
                          f"cal+lp={cal_wait:.1f}s  improvement={imp:+.1f}%")
    return df


# ── Part B: Fleet sensitivity sweep ──────────────────────────────────────────

def run_part_b(all_blocks, all_trips, library):
    """Fleet size sensitivity: show cal+LP improvement is robust across fleet sizes."""
    print("\n" + "=" * 80)
    print("PART B: FLEET SENSITIVITY SWEEP")
    print("=" * 80)

    router = HaversineClient()
    fleet_multipliers = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    seeds = [42, 123, 456]

    rows = []
    for scen_name in REPRESENTATIVE:
        month, hr, dtype = NYC_SCENARIOS[scen_name]
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            continue

        prep = _prepare_scenario(block, all_trips, library)
        base_fleet = prep["fleet"]
        print(f"\n  {scen_name} base_fleet={base_fleet}")

        for mult in fleet_multipliers:
            fleet = max(int(base_fleet * mult), 20)
            for config in ["replay", "cal_only", "cal+lp"]:
                seed_kpis = []
                for seed in seeds:
                    kpi = _run_config(prep, config, seed, router, fleet_override=fleet)
                    seed_kpis.append(kpi)

                avg = {k: float(np.mean([s[k] for s in seed_kpis]))
                       for k in seed_kpis[0] if isinstance(seed_kpis[0][k], (int, float))}
                std = {f"{k}_std": float(np.std([s[k] for s in seed_kpis]))
                       for k in seed_kpis[0] if isinstance(seed_kpis[0][k], (int, float))}
                avg.update(std)
                avg["scenario"] = scen_name
                avg["config"] = config
                avg["fleet_multiplier"] = mult
                avg["fleet_actual"] = fleet
                rows.append(avg)

            # Print row
            rep_w = [r for r in rows if r["scenario"] == scen_name and
                     r["fleet_multiplier"] == mult and r["config"] == "replay"]
            cal_w = [r for r in rows if r["scenario"] == scen_name and
                     r["fleet_multiplier"] == mult and r["config"] == "cal+lp"]
            if rep_w and cal_w:
                rw = rep_w[-1]["mean_wait_s"]
                cw = cal_w[-1]["mean_wait_s"]
                imp = (rw - cw) / rw * 100 if rw > 0 else 0
                print(f"    fleet={fleet:4d} (×{mult:.2f}): replay={rw:.0f}s  "
                      f"cal+lp={cw:.0f}s  Δ={imp:+.1f}%")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  FLEET SENSITIVITY SUMMARY (avg across {len(REPRESENTATIVE)} scenarios):")
        for mult in fleet_multipliers:
            sub = df[df["fleet_multiplier"] == mult]
            rep = sub[sub["config"] == "replay"]["mean_wait_s"].mean()
            cal = sub[sub["config"] == "cal_only"]["mean_wait_s"].mean()
            lp = sub[sub["config"] == "cal+lp"]["mean_wait_s"].mean()
            imp_cal = (rep - cal) / rep * 100 if rep > 0 else 0
            imp_lp = (rep - lp) / rep * 100 if rep > 0 else 0
            print(f"    ×{mult:.2f}: replay={rep:.0f}s  cal_only={cal:.0f}s ({imp_cal:+.1f}%)  "
                  f"cal+lp={lp:.0f}s ({imp_lp:+.1f}%)")
    return df


# ── Part C: Multi-seed statistical robustness ────────────────────────────────

def run_part_c(all_blocks, all_trips, library, n_seeds=5):
    """Full 8-scenario evaluation with multiple seeds for proper confidence intervals."""
    print("\n" + "=" * 80)
    print(f"PART C: MULTI-SEED ROBUSTNESS ({n_seeds} seeds × 8 scenarios × 3 configs)")
    print("=" * 80)

    router = HaversineClient()
    seeds = list(range(42, 42 + n_seeds * 100, 100))[:n_seeds]
    print(f"  Seeds: {seeds}")

    rows = []
    for scen_name, (month, hr, dtype) in NYC_SCENARIOS.items():
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            continue

        prep = _prepare_scenario(block, all_trips, library)
        print(f"\n  {scen_name} fleet={prep['fleet']}")

        for config in ["replay", "cal_only", "cal+lp"]:
            for seed in seeds:
                kpi = _run_config(prep, config, seed, router)
                kpi["scenario"] = scen_name
                kpi["config"] = config
                kpi["seed"] = seed
                rows.append(kpi)

            # Summary for this scenario+config
            waits = [r["mean_wait_s"] for r in rows
                     if r["scenario"] == scen_name and r["config"] == config]
            print(f"    {config:10s}: {np.mean(waits):.1f} ± {np.std(waits):.1f}s "
                  f"(n={len(waits)})")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  HEADLINE NUMBERS (mean ± std across {n_seeds} seeds):")
        for config in ["replay", "cal_only", "cal+lp"]:
            sub = df[df["config"] == config]
            scen_means = sub.groupby("scenario")["mean_wait_s"].mean()
            scen_stds = sub.groupby("scenario")["mean_wait_s"].std()
            grand_mean = scen_means.mean()
            # Propagate uncertainty: std of scenario means
            grand_std = scen_means.std() / np.sqrt(len(scen_means))
            print(f"    {config:10s}: {grand_mean:.1f} ± {grand_std:.1f}s")

        # Per-scenario improvement with CI
        print(f"\n  PER-SCENARIO IMPROVEMENT (cal+lp vs replay, {n_seeds} seeds):")
        for scen_name in NYC_SCENARIOS:
            rep = df[(df["config"] == "replay") & (df["scenario"] == scen_name)]
            cal = df[(df["config"] == "cal+lp") & (df["scenario"] == scen_name)]
            if rep.empty or cal.empty:
                continue
            # Paired comparison: each seed
            imps = []
            for seed in seeds:
                rw = rep[rep["seed"] == seed]["mean_wait_s"].values
                cw = cal[cal["seed"] == seed]["mean_wait_s"].values
                if len(rw) and len(cw):
                    imps.append((rw[0] - cw[0]) / rw[0] * 100)
            if imps:
                mean_imp = np.mean(imps)
                std_imp = np.std(imps)
                ci_lo = mean_imp - 1.96 * std_imp / np.sqrt(len(imps))
                ci_hi = mean_imp + 1.96 * std_imp / np.sqrt(len(imps))
                print(f"    {scen_name:20s}: {mean_imp:+.1f}% "
                      f"[{ci_lo:+.1f}, {ci_hi:+.1f}] 95% CI")
    return df


# ── Part D: Similarity component ablation ────────────────────────────────────

def run_part_d(all_blocks, all_trips, library):
    """Ablate similarity components: which metrics matter most for calibration quality?"""
    print("\n" + "=" * 80)
    print("PART D: SIMILARITY COMPONENT ABLATION")
    print("=" * 80)

    router = HaversineClient()
    seeds = [42, 123, 456]

    # Component ablation configs
    ablation_configs = {
        "full_ensemble": None,  # default weights from config
        "ks_only":       {"ks": 1.0, "w1": 0, "feat": 0, "var": 0, "event": 0, "temporal": 0},
        "w1_only":       {"ks": 0, "w1": 1.0, "feat": 0, "var": 0, "event": 0, "temporal": 0},
        "distributional": {"ks": 0.35, "w1": 0.35, "feat": 0.15, "var": 0.15, "event": 0, "temporal": 0},
        "no_event":      {"ks": 0.25, "w1": 0.25, "feat": 0.20, "var": 0.15, "event": 0, "temporal": 0.15},
        "no_temporal":   {"ks": 0.25, "w1": 0.25, "feat": 0.15, "var": 0.10, "event": 0.25, "temporal": 0},
        "event_only":    {"ks": 0, "w1": 0, "feat": 0, "var": 0, "event": 1.0, "temporal": 0},
        "random_top5":   "random",  # random library matches (shuffled)
    }

    rows = []
    for scen_name in REPRESENTATIVE:
        month, hr, dtype = NYC_SCENARIOS[scen_name]
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            continue

        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].values.astype(np.float64)
        q_events = annotate_events(q_series)
        date_str, hour_range = bid.rsplit("_", 1)
        start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
        horizon = float((end_h - start_h) * 3600)
        replay_trips = _get_replay_trips(all_trips, bid)
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

        print(f"\n  {scen_name} [{bid}] fleet={fleet}")

        # Replay baseline
        for seed in seeds:
            kpi = _run_one(ReplayDemandStream(replay_trips), fleet, horizon, seed, router)
            kpi.update({"scenario": scen_name, "config": "replay", "similarity": "replay",
                        "seed": seed})
            rows.append(kpi)

        replay_avg = np.mean([r["mean_wait_s"] for r in rows
                              if r["scenario"] == scen_name and r["config"] == "replay"])

        for sim_name, sim_weights in ablation_configs.items():
            # Get matched regimes with specific weights
            if sim_weights == "random":
                # Random baseline: pick 5 random regimes (exclude query)
                all_bids = [b for b in library.records.keys() if b != bid]
                rng_lib = np.random.default_rng(42)
                rng_lib.shuffle(all_bids)
                random_bids = all_bids[:5]
                mrecs = [library[b] for b in random_bids]
                mscores = [1.0] * 5
            elif sim_weights is not None:
                # Custom weights — exclude query block (leave-one-out)
                scores = []
                for b, rec in library.records.items():
                    if b == bid:
                        continue
                    s = compute_similarity(q_series, q_events, rec, sim_weights, q_block_id=bid)
                    scores.append((b, s))
                scores.sort(key=lambda x: x[1], reverse=True)
                matched = scores[:5]
                mrecs = [library[b] for b, _ in matched]
                mscores = [s for _, s in matched]
            else:
                # Full ensemble (default config)
                matched = query_library(library, q_series, q_events, q_block_id=bid)
                mrecs = [library[b] for b, _ in matched]
                mscores = [s for _, s in matched]

            if not mrecs:
                continue

            prior = build_calibrated_prior(mrecs, mscores, q_series)

            for seed in seeds:
                rng = np.random.default_rng(seed)
                pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
                stream = CalibratedDemandStream(pr, horizon, rng=rng)
                repo = AnticipatoryReposition(pr, lookahead_minutes=5.0, max_move_fraction=0.50)
                kpi = _run_one(stream, fleet, horizon, seed, router, repo)
                kpi.update({"scenario": scen_name, "config": "cal+lp",
                            "similarity": sim_name, "seed": seed})
                rows.append(kpi)

            avg_wait = np.mean([r["mean_wait_s"] for r in rows
                                if r["scenario"] == scen_name and r["similarity"] == sim_name])
            imp = (replay_avg - avg_wait) / replay_avg * 100
            print(f"    {sim_name:20s}: wait={avg_wait:.1f}s  vs replay={imp:+.1f}%")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  COMPONENT ABLATION SUMMARY (avg across {len(REPRESENTATIVE)} scenarios):")
        for sim_name in ablation_configs:
            sub = df[(df["similarity"] == sim_name) & (df["config"] == "cal+lp")]
            rep_sub = df[df["config"] == "replay"]
            if sub.empty or rep_sub.empty:
                continue
            sim_wait = sub.groupby("scenario")["mean_wait_s"].mean().mean()
            rep_wait = rep_sub.groupby("scenario")["mean_wait_s"].mean().mean()
            imp = (rep_wait - sim_wait) / rep_wait * 100
            print(f"    {sim_name:20s}: {sim_wait:.1f}s ({imp:+.1f}% vs replay)")
    return df


# ── Part E: Distributional fairness & tail metrics ───────────────────────────

def run_part_e(all_blocks, all_trips, library):
    """Compute P50/P95/P99/max wait and Gini coefficient across all 8 scenarios."""
    print("\n" + "=" * 80)
    print("PART E: TAIL WAIT & FAIRNESS METRICS (all 8 scenarios)")
    print("=" * 80)

    router = HaversineClient()
    seeds = [42, 123, 456]

    rows = []
    for scen_name, (month, hr, dtype) in NYC_SCENARIOS.items():
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            continue

        prep = _prepare_scenario(block, all_trips, library)
        print(f"\n  {scen_name} fleet={prep['fleet']}")

        for config in ["replay", "cal+lp"]:
            for seed in seeds:
                kpi = _run_config(prep, config, seed, router)
                kpi["scenario"] = scen_name
                kpi["config"] = config
                kpi["seed"] = seed
                rows.append(kpi)

            sub = [r for r in rows if r["scenario"] == scen_name and r["config"] == config]
            mean_w = np.mean([r["mean_wait_s"] for r in sub])
            p50 = np.mean([r["p50_wait_s"] for r in sub])
            p95 = np.mean([r["p95_wait_s"] for r in sub])
            p99 = np.mean([r["p99_wait_s"] for r in sub])
            gini = np.mean([r["gini_wait"] for r in sub])
            print(f"    {config:10s}: mean={mean_w:.0f}s  P50={p50:.0f}s  P95={p95:.0f}s  "
                  f"P99={p99:.0f}s  Gini={gini:.3f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  AGGREGATE TAIL METRICS:")
        for config in ["replay", "cal+lp"]:
            sub = df[df["config"] == config]
            metrics = {m: sub[m].mean() for m in
                       ["mean_wait_s", "p50_wait_s", "p95_wait_s", "p99_wait_s", "gini_wait"]}
            print(f"    {config:10s}: mean={metrics['mean_wait_s']:.1f}s  "
                  f"P50={metrics['p50_wait_s']:.1f}s  P95={metrics['p95_wait_s']:.1f}s  "
                  f"P99={metrics['p99_wait_s']:.1f}s  Gini={metrics['gini_wait']:.3f}")

        # Improvement at each quantile
        rep = df[df["config"] == "replay"]
        cal = df[df["config"] == "cal+lp"]
        print(f"\n  IMPROVEMENT BY QUANTILE:")
        for m_label, m_col in [("Mean", "mean_wait_s"), ("P50", "p50_wait_s"),
                                ("P95", "p95_wait_s"), ("P99", "p99_wait_s")]:
            r_avg = rep.groupby("scenario")[m_col].mean().mean()
            c_avg = cal.groupby("scenario")[m_col].mean().mean()
            imp = (r_avg - c_avg) / r_avg * 100
            print(f"    {m_label:4s}: replay={r_avg:.1f}s  cal+lp={c_avg:.1f}s  Δ={imp:+.1f}%")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Paper-ready experiments")
    ap.add_argument("--parts", nargs="*", default=["A", "B", "C", "D", "E"],
                    choices=["A", "B", "C", "D", "E"],
                    help="Which experiment parts to run")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds for Part C (default: 5)")
    args = ap.parse_args()
    parts = [p.upper() for p in args.parts]

    cfg = get_config()
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(exist_ok=True)

    t_total = time.time()
    print("Loading NYC data and regime library ...")
    library = RegimeLibrary()
    library.load()
    jan_trips = load_cleaned("2024-01")
    jun_trips = load_cleaned("2024-06")
    all_trips = pd.concat([jan_trips, jun_trips], ignore_index=True)
    jan_profile = build_demand_profile(jan_trips)
    jun_profile = build_demand_profile(jun_trips)
    jan_blocks = split_into_blocks(jan_profile)
    jun_blocks = split_into_blocks(jun_profile)
    all_blocks = jan_blocks + jun_blocks
    print(f"  {len(library)} regimes, {len(all_blocks)} blocks, {len(all_trips)} trips")

    results = {}

    if "A" in parts:
        df_a = run_part_a(all_blocks, all_trips, library)
        if not df_a.empty:
            df_a.to_csv(out_dir / "osrm_fleet_adjusted.csv", index=False)
            results["A"] = df_a

    if "B" in parts:
        df_b = run_part_b(all_blocks, all_trips, library)
        if not df_b.empty:
            df_b.to_csv(out_dir / "fleet_sensitivity.csv", index=False)
            results["B"] = df_b

    if "C" in parts:
        df_c = run_part_c(all_blocks, all_trips, library, n_seeds=args.seeds)
        if not df_c.empty:
            df_c.to_csv(out_dir / "multiseed_robustness.csv", index=False)
            results["C"] = df_c

    if "D" in parts:
        df_d = run_part_d(all_blocks, all_trips, library)
        if not df_d.empty:
            df_d.to_csv(out_dir / "similarity_ablation.csv", index=False)
            results["D"] = df_d

    if "E" in parts:
        df_e = run_part_e(all_blocks, all_trips, library)
        if not df_e.empty:
            df_e.to_csv(out_dir / "tail_fairness_metrics.csv", index=False)
            results["E"] = df_e

    elapsed = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to {out_dir}/")
    for part, df in results.items():
        print(f"  Part {part}: {len(df)} rows → {out_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
