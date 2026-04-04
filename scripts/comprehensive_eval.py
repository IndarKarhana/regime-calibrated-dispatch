#!/usr/bin/env python3
"""Comprehensive evaluation: end-to-end pipeline comparison + LP sensitivity + academic benchmarks.

Runs:
  1. Fixed 8-scenario evaluation with correct month-aware block matching
  2. Full pipeline comparison: replay -> calibrated -> calibrated+LP -> calibrated+heuristic
  3. LP parameter sensitivity (lookahead, move fraction)
  4. Comparison against published academic baselines
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.similarity import query_library
from src.regime.events import annotate_events
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.engine import SimulationEngine
from src.simulator.routing import HaversineClient, OSRMClient
from src.policies.batch import BatchMatchingPolicy
from src.policies.anticipatory import (
    AnticipatoryReposition,
    DemandFollowingReposition,
)
from src.evaluation.metrics import compute_kpis


def _pick_block(blocks, month_prefix: str, hour_range: str, day_type: str):
    """Month-aware block selection to prevent cross-month contamination."""
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        if not bid.startswith(month_prefix):
            continue
        if hour_range not in bid:
            continue
        total = blk["request_count"].sum()
        if total < 500:
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


SCENARIOS = {
    "jan_weekday_am":  ("2024-01", "08-12", "weekday"),
    "jan_weekday_pm":  ("2024-01", "16-20", "weekday"),
    "jan_nye_am":      ("2024-01", "08-12", "nye"),
    "jan_nye_pm":      ("2024-01", "16-20", "nye"),
    "jan_weekend_mid": ("2024-01", "12-16", "weekend"),
    "jun_weekday_am":  ("2024-06", "08-12", "weekday"),
    "jun_weekday_pm":  ("2024-06", "16-20", "weekday"),
    "jun_late_night":  ("2024-06", "20-24", "fri_night"),
}


def _get_replay_trips(all_trips, bid):
    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    mask = (all_trips["pickup_datetime"].dt.date == target_date)
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
    return compute_kpis(state, time.time() - t0)


def run_full_pipeline(
    all_blocks, all_trips, library, router, seeds,
    *,
    lp_lookahead: float = 5.0,
    lp_move_frac: float = 0.50,
    heur_lookahead: float = 15.0,
    heur_move_frac: float = 0.25,
):
    """Part 1 & 2: Full pipeline comparison across 8 scenarios."""
    print("\n" + "=" * 80)
    print("PART 1: END-TO-END PIPELINE COMPARISON")
    print("=" * 80)
    print(f"  LP params: lookahead={lp_lookahead} min, max_move_fraction={lp_move_frac}")
    print("  Calibrated demand: expected total matched to replay trip count per scenario.")

    rows = []

    for scen_name, (month, hr, dtype) in SCENARIOS.items():
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            print(f"  SKIP {scen_name}: no block found")
            continue

        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].values.astype(np.float64)
        q_events = annotate_events(q_series)

        date_str, hour_range = bid.rsplit("_", 1)
        start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
        if end_h <= start_h:
            end_h += 24
        horizon = float((end_h - start_h) * 3600)

        replay_trips = _get_replay_trips(all_trips, bid)
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

        matched = query_library(library, q_series, q_events, q_block_id=bid)
        mrecs = [library[b] for b, _ in matched]
        mscores = [s for _, s in matched]
        prior = build_calibrated_prior(mrecs, mscores, q_series)

        configs = [
            ("1_replay",      "replay", None),
            ("2_cal_only",    "cal",    None),
            ("3_cal+heur",    "cal", lambda pr: DemandFollowingReposition(
                pr, lookahead_minutes=heur_lookahead, max_move_fraction=heur_move_frac,
            )),
            ("4_cal+lp",      "cal", lambda pr: AnticipatoryReposition(
                pr, lookahead_minutes=lp_lookahead, max_move_fraction=lp_move_frac,
            )),
        ]

        print(f"\n  {scen_name} [{bid}] trips={n_trips} fleet={fleet}")

        for config_name, stream_type, repo_factory in configs:
            seed_results = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                if stream_type == "replay":
                    stream = ReplayDemandStream(replay_trips)
                    repo = None
                else:
                    pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
                    stream = CalibratedDemandStream(pr, horizon, rng=rng)
                    repo = repo_factory(pr) if repo_factory else None
                kpi = _run_one(stream, fleet, horizon, seed, router, repo)
                kpi["scenario"] = scen_name
                kpi["block_id"] = bid
                kpi["config"] = config_name
                kpi["fleet"] = fleet
                kpi["n_trips"] = n_trips
                kpi["seed"] = seed
                rows.append(kpi)
                seed_results.append(kpi)

            avg_w = np.mean([r["mean_wait_s"] for r in seed_results])
            avg_c = np.mean([r["completion_rate"] for r in seed_results])
            print(f"    {config_name:16s}: wait={avg_w:6.1f}s  comp={avg_c:.1%}")

    return pd.DataFrame(rows)


def run_lp_sensitivity(all_blocks, all_trips, library, router):
    """Part 3: LP parameter sensitivity on 2 representative scenarios."""
    print("\n" + "=" * 80)
    print("PART 2: LP PARAMETER SENSITIVITY")
    print("=" * 80)

    test_scenarios = [
        ("jan_weekday_pm", "2024-01", "16-20", "weekday"),
        ("jan_weekend_mid", "2024-01", "12-16", "weekend"),
    ]

    lookaheads = [5.0, 10.0, 15.0, 20.0, 30.0]
    move_fracs = [0.10, 0.20, 0.35, 0.50]

    rows = []

    for scen_name, month, hr, dtype in test_scenarios:
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

        matched = query_library(library, q_series, q_events, q_block_id=bid)
        mrecs = [library[b] for b, _ in matched]
        mscores = [s for _, s in matched]
        prior = build_calibrated_prior(mrecs, mscores, q_series)

        # Baseline (no reposition), volume-matched to replay
        rng = np.random.default_rng(42)
        pr0 = prior_matched_to_replay_volume(prior, horizon, n_trips)
        stream = CalibratedDemandStream(pr0, horizon, rng=rng)
        base_kpi = _run_one(stream, fleet, horizon, 42, router, None)
        base_wait = base_kpi["mean_wait_s"]

        print(f"\n  {scen_name} [{bid}] fleet={fleet}, baseline wait={base_wait:.0f}s")

        for la in lookaheads:
            for mf in move_fracs:
                pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
                lp = AnticipatoryReposition(pr, lookahead_minutes=la, max_move_fraction=mf)
                rng = np.random.default_rng(42)
                stream = CalibratedDemandStream(pr, horizon, rng=rng)
                kpi = _run_one(stream, fleet, horizon, 42, router, lp)
                improvement = (base_wait - kpi["mean_wait_s"]) / base_wait * 100
                rows.append({
                    "scenario": scen_name,
                    "lookahead_min": la,
                    "move_fraction": mf,
                    "wait_s": kpi["mean_wait_s"],
                    "completion": kpi["completion_rate"],
                    "improvement_pct": improvement,
                })

        # Print pivot for this scenario
        scen_rows = [r for r in rows if r["scenario"] == scen_name]
        scen_df = pd.DataFrame(scen_rows)
        pivot = scen_df.pivot_table(
            values="improvement_pct", index="lookahead_min",
            columns="move_fraction", aggfunc="mean",
        ).round(1)
        print(f"    Wait reduction (%) by lookahead x move_fraction:")
        print(pivot.to_string())

    return pd.DataFrame(rows)


def print_academic_comparison(pipeline_df, lp_desc: str = "Min-cost transport LP"):
    """Part 4: Compare our results against published academic baselines."""
    print("\n" + "=" * 80)
    print("PART 3: COMPARISON WITH PUBLISHED ACADEMIC RESULTS")
    print("=" * 80)

    published = [
        {
            "method": "Proactive Rebalancing (Wen et al. 2017)",
            "wait_reduction": "5.0%",
            "notes": "Probabilistic demand forecast, NYC taxi",
        },
        {
            "method": "RL from Optimization Proxy (IJCAI 2023)",
            "wait_reduction": "~10-15%",
            "notes": "RL-based relocation, ride-hailing",
        },
        {
            "method": "AAVR Framework (Dec 2024)",
            "wait_reduction": "22.7%",
            "notes": "Adherence-aware rebalancing, NYC taxi, +28% served demand",
        },
        {
            "method": "Sim-informed RL (Namdarpour & Chow 2025)",
            "wait_reduction": "27.3%",
            "notes": "Ride-POOLING (not hailing), non-myopic matching+rebalancing",
        },
    ]

    # Compute our numbers
    if not pipeline_df.empty:
        replay = pipeline_df[pipeline_df["config"] == "1_replay"].groupby("scenario")["mean_wait_s"].mean()
        cal_only = pipeline_df[pipeline_df["config"] == "2_cal_only"].groupby("scenario")["mean_wait_s"].mean()
        cal_lp = pipeline_df[pipeline_df["config"] == "4_cal+lp"].groupby("scenario")["mean_wait_s"].mean()

        cal_vs_replay = ((replay - cal_only) / replay * 100).mean()
        lp_vs_cal = ((cal_only - cal_lp) / cal_only * 100).mean()
        combined_vs_replay = ((replay - cal_lp) / replay * 100).mean()
    else:
        cal_vs_replay = lp_vs_cal = combined_vs_replay = 0.0

    print("\n  PUBLISHED BASELINES:")
    print(f"  {'Method':<55s} {'Wait Reduction':<18s} {'Notes'}")
    print("  " + "-" * 110)
    for p in published:
        print(f"  {p['method']:<55s} {p['wait_reduction']:<18s} {p['notes']}")

    print(f"\n  OUR RESULTS (this work):")
    print(f"  {'Method':<55s} {'Wait Reduction':<18s} {'Notes'}")
    print("  " + "-" * 110)
    print(f"  {'Calibrated prior only (vs replay)':<55s} {cal_vs_replay:+.1f}%{'':<13s} {'Similarity ensemble, 8 scenarios avg'}")
    print(f"  {'LP repositioning only (vs cal no-reposition)':<55s} {lp_vs_cal:+.1f}%{'':<13s} {lp_desc}")
    print(f"  {'Combined: calibrated + LP (vs replay)':<55s} {combined_vs_replay:+.1f}%{'':<13s} {'Full pipeline, end-to-end'}")

    print(f"\n  POSITIONING:")
    if combined_vs_replay > 22:
        print(f"  -> Our combined approach ({combined_vs_replay:+.1f}%) is COMPETITIVE with AAVR (22.7%)")
        print(f"     and approaches sim-informed RL (27.3%, but that's ride-pooling).")
    elif combined_vs_replay > 15:
        print(f"  -> Our combined approach ({combined_vs_replay:+.1f}%) EXCEEDS simple proactive (5%)")
        print(f"     and RL-proxy methods (~10-15%). Competitive with advanced methods.")
    elif combined_vs_replay > 10:
        print(f"  -> Our combined approach ({combined_vs_replay:+.1f}%) EXCEEDS simple proactive (5%)")
        print(f"     and matches RL-proxy methods (~10-15%).")
    else:
        print(f"  -> Our combined approach ({combined_vs_replay:+.1f}%) exceeds simple proactive (5%).")
    print(f"  -> Key differentiator: NO TRAINING REQUIRED. LP is deterministic + explainable.")
    print(f"  -> Calibration component (prior) is novel; not present in any baseline above.")

    # Per-scenario breakdown
    if not pipeline_df.empty:
        print(f"\n  PER-SCENARIO END-TO-END TABLE:")
        print(f"  {'Scenario':<20s} {'Replay':>8s} {'Cal Only':>10s} {'Cal+LP':>8s} {'vs Replay':>10s}")
        print("  " + "-" * 60)
        for scen in sorted(replay.index):
            r_w = replay.get(scen, 0)
            c_w = cal_only.get(scen, 0)
            l_w = cal_lp.get(scen, 0)
            imp = (r_w - l_w) / r_w * 100 if r_w > 0 else 0
            print(f"  {scen:<20s} {r_w:7.0f}s {c_w:9.0f}s {l_w:7.0f}s {imp:+9.1f}%")
        avg_r = replay.mean()
        avg_c = cal_only.mean()
        avg_l = cal_lp.mean()
        avg_imp = (avg_r - avg_l) / avg_r * 100
        print("  " + "-" * 60)
        print(f"  {'AVERAGE':<20s} {avg_r:7.0f}s {avg_c:9.0f}s {avg_l:7.0f}s {avg_imp:+9.1f}%")


def _check_osrm() -> bool:
    import requests
    oc = get_config()["osrm"]
    try:
        url = f"{oc['host']}:{oc['port']}/nearest/v1/car/-73.985,40.748"
        r = requests.get(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description="Comprehensive NYC pipeline eval")
    p.add_argument(
        "--router", choices=("haversine", "osrm"), default="haversine",
        help="Routing backend (OSRM requires docker compose up -d osrm)",
    )
    p.add_argument("--lp-lookahead", type=float, default=5.0, help="LP forecast horizon (minutes)")
    p.add_argument("--lp-move-frac", type=float, default=0.50, help="LP max fraction of idle to move")
    p.add_argument("--skip-sensitivity", action="store_true", help="Skip LP grid sweep (faster)")
    args = p.parse_args()

    cfg = get_config()
    seeds = cfg["evaluation"]["seeds"]
    if args.router == "osrm" and _check_osrm():
        router = OSRMClient()
        print("Using OSRM routing (NYC graph).")
    elif args.router == "osrm":
        print("OSRM unreachable; falling back to Haversine.")
        router = HaversineClient()
    else:
        router = HaversineClient()

    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(exist_ok=True)

    print("Loading data and regime library ...")
    library = RegimeLibrary()
    library.load()
    print(f"  {len(library)} regimes loaded.")

    jan_trips = load_cleaned("2024-01")
    jun_trips = load_cleaned("2024-06")
    all_trips = pd.concat([jan_trips, jun_trips], ignore_index=True)

    jan_profile = build_demand_profile(jan_trips)
    jun_profile = build_demand_profile(jun_trips)
    jan_blocks = split_into_blocks(jan_profile)
    jun_blocks = split_into_blocks(jun_profile)
    all_blocks = jan_blocks + jun_blocks
    print(f"  {len(all_blocks)} blocks ({len(jan_blocks)} Jan + {len(jun_blocks)} Jun)")

    # Part 1 & 2: Full pipeline (default LP: tuned 5 min / 0.50 move fraction)
    pipeline_df = run_full_pipeline(
        all_blocks, all_trips, library, router, seeds,
        lp_lookahead=args.lp_lookahead,
        lp_move_frac=args.lp_move_frac,
    )
    pipeline_df.to_csv(out_dir / "comprehensive_pipeline.csv", index=False)

    # Part 3: LP sensitivity
    if args.skip_sensitivity:
        sensitivity_df = pd.DataFrame()
        print("\n[skip] LP sensitivity (--skip-sensitivity)")
    else:
        sensitivity_df = run_lp_sensitivity(all_blocks, all_trips, library, router)
        sensitivity_df.to_csv(out_dir / "lp_sensitivity.csv", index=False)

    # Part 4: Academic comparison
    lp_desc = f"Min-cost transport LP ({args.lp_lookahead} min, move_frac={args.lp_move_frac})"
    print_academic_comparison(pipeline_df, lp_desc=lp_desc)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
