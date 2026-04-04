#!/usr/bin/env python3
"""Extended baselines and missing ablations for paper completeness.

Runs:
  F: Extended baselines (greedy dispatch, heuristic reposition, no-reposition)
  G: top_k sensitivity sweep (3, 5, 10, 20)
  H: batch_window sensitivity sweep (30, 60, 120s)

Usage:
  python scripts/extended_experiments.py
  python scripts/extended_experiments.py --parts F G H
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
from src.policies.greedy import GreedyNearestPolicy
from src.policies.anticipatory import AnticipatoryReposition, DemandFollowingReposition
from src.evaluation.metrics import compute_kpis


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

REPRESENTATIVE = ["jan_weekday_pm", "jun_weekday_am", "jun_late_night"]

OUT_DIR = Path("results")


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


def _prepare_scenario(block, all_trips, library):
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


def _run_sim(stream, fleet, horizon, seed, router, dispatch_policy, repo_policy=None,
             batch_window=None):
    """Run one simulation with configurable dispatch and reposition policies."""
    if batch_window is not None and hasattr(dispatch_policy, '_window'):
        dispatch_policy._window = batch_window
        dispatch_policy._last_batch_time = -1e9  # reset timer

    engine = SimulationEngine(
        demand_stream=stream, policy=dispatch_policy, router=router,
        fleet_size=fleet, horizon_seconds=horizon, seed=seed,
        reposition_policy=repo_policy, reposition_interval_steps=6,
    )
    t0 = time.time()
    state = engine.run()
    kpi = compute_kpis(state, time.time() - t0)
    # Add tail metrics
    waits = np.array(state.metrics.wait_times) if state.metrics.wait_times else np.array([0.0])
    kpi["p99_wait_s"] = float(np.percentile(waits, 99)) if len(waits) > 1 else 0.0
    if len(waits) > 1 and np.sum(waits) > 0:
        sorted_w = np.sort(waits)
        n = len(sorted_w)
        index = np.arange(1, n + 1)
        kpi["gini_wait"] = float((2 * np.sum(index * sorted_w) - (n + 1) * np.sum(sorted_w)) /
                                  (n * np.sum(sorted_w)))
    else:
        kpi["gini_wait"] = 0.0
    return kpi


# ── Part F: Extended baselines ───────────────────────────────────────────────

def run_part_f(all_blocks, all_trips, library):
    """Extended baselines: greedy dispatch, heuristic reposition, no-reposition cal."""
    print("\n" + "=" * 80)
    print("PART F: EXTENDED BASELINES")
    print("  Configs: replay-greedy, replay-batch, cal+heuristic, cal+lp, cal_only")
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

        configs = {
            "replay_greedy": {
                "stream_type": "replay", "dispatch": "greedy", "repo": None,
            },
            "replay_batch": {
                "stream_type": "replay", "dispatch": "batch", "repo": None,
            },
            "cal_only": {
                "stream_type": "cal", "dispatch": "batch", "repo": None,
            },
            "cal+heuristic": {
                "stream_type": "cal", "dispatch": "batch", "repo": "heuristic",
            },
            "cal+lp": {
                "stream_type": "cal", "dispatch": "batch", "repo": "lp",
            },
        }

        for cfg_name, cfg_spec in configs.items():
            for seed in seeds:
                rng = np.random.default_rng(seed)

                # Build stream
                if cfg_spec["stream_type"] == "replay":
                    stream = ReplayDemandStream(prep["replay_trips"])
                else:
                    pr = prior_matched_to_replay_volume(
                        prep["prior"], prep["horizon"], prep["n_trips"])
                    stream = CalibratedDemandStream(pr, prep["horizon"], rng=rng)

                # Build dispatch
                if cfg_spec["dispatch"] == "greedy":
                    dispatch = GreedyNearestPolicy()
                else:
                    dispatch = BatchMatchingPolicy()

                # Build repo
                if cfg_spec["repo"] == "lp":
                    pr = prior_matched_to_replay_volume(
                        prep["prior"], prep["horizon"], prep["n_trips"])
                    repo = AnticipatoryReposition(pr, lookahead_minutes=5.0, max_move_fraction=0.50)
                elif cfg_spec["repo"] == "heuristic":
                    pr = prior_matched_to_replay_volume(
                        prep["prior"], prep["horizon"], prep["n_trips"])
                    repo = DemandFollowingReposition(pr, lookahead_minutes=5.0, max_move_fraction=0.50)
                else:
                    repo = None

                kpi = _run_sim(stream, prep["fleet"], prep["horizon"], seed,
                               router, dispatch, repo)
                kpi["scenario"] = scen_name
                kpi["config"] = cfg_name
                kpi["seed"] = seed
                rows.append(kpi)

            sub = [r for r in rows if r["scenario"] == scen_name and r["config"] == cfg_name]
            avg_w = np.mean([r["mean_wait_s"] for r in sub])
            print(f"    {cfg_name:20s}: wait={avg_w:.1f}s")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  EXTENDED BASELINES SUMMARY (avg across {len(NYC_SCENARIOS)} scenarios):")
        for cfg in ["replay_greedy", "replay_batch", "cal_only", "cal+heuristic", "cal+lp"]:
            sub = df[df["config"] == cfg]
            if not sub.empty:
                avg = sub.groupby("scenario")["mean_wait_s"].mean().mean()
                print(f"    {cfg:20s}: {avg:.1f}s")

        # Improvement table vs replay_batch
        rep_batch = df[df["config"] == "replay_batch"]
        if not rep_batch.empty:
            rep_avg = rep_batch.groupby("scenario")["mean_wait_s"].mean().mean()
            print(f"\n  vs replay_batch ({rep_avg:.1f}s):")
            for cfg in ["replay_greedy", "cal_only", "cal+heuristic", "cal+lp"]:
                sub = df[df["config"] == cfg]
                if not sub.empty:
                    avg = sub.groupby("scenario")["mean_wait_s"].mean().mean()
                    imp = (rep_avg - avg) / rep_avg * 100
                    print(f"    {cfg:20s}: {avg:.1f}s  ({imp:+.1f}%)")
    return df


# ── Part G: top_k sensitivity ────────────────────────────────────────────────

def run_part_g(all_blocks, all_trips, library):
    """Sweep top_k (number of matched regimes) to check sensitivity."""
    print("\n" + "=" * 80)
    print("PART G: TOP_K SENSITIVITY SWEEP")
    print("=" * 80)

    router = HaversineClient()
    seeds = [42, 123, 456]
    top_k_values = [1, 3, 5, 10, 20]

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
            stream = ReplayDemandStream(replay_trips)
            kpi = _run_sim(stream, fleet, horizon, seed, router, BatchMatchingPolicy())
            kpi.update({"scenario": scen_name, "config": "replay", "top_k": 0, "seed": seed})
            rows.append(kpi)

        for k in top_k_values:
            # Query library with specific top_k
            matched = query_library(library, q_series, q_events, q_block_id=bid, top_k=k)
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
                kpi = _run_sim(stream, fleet, horizon, seed, router,
                               BatchMatchingPolicy(), repo)
                kpi.update({"scenario": scen_name, "config": "cal+lp",
                            "top_k": k, "seed": seed})
                rows.append(kpi)

            sub = [r for r in rows if r["scenario"] == scen_name and
                   r["config"] == "cal+lp" and r["top_k"] == k]
            avg_w = np.mean([r["mean_wait_s"] for r in sub])
            rep_sub = [r for r in rows if r["scenario"] == scen_name and r["config"] == "replay"]
            rep_w = np.mean([r["mean_wait_s"] for r in rep_sub])
            imp = (rep_w - avg_w) / rep_w * 100 if rep_w > 0 else 0
            print(f"    top_k={k:3d}: wait={avg_w:.1f}s  ({imp:+.1f}% vs replay)")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  TOP_K SUMMARY (avg across {len(REPRESENTATIVE)} scenarios):")
        rep_avg = df[df["config"] == "replay"].groupby("scenario")["mean_wait_s"].mean().mean()
        for k in top_k_values:
            sub = df[(df["config"] == "cal+lp") & (df["top_k"] == k)]
            if not sub.empty:
                avg = sub.groupby("scenario")["mean_wait_s"].mean().mean()
                imp = (rep_avg - avg) / rep_avg * 100
                print(f"    top_k={k:3d}: {avg:.1f}s  ({imp:+.1f}%)")
    return df


# ── Part H: batch_window sensitivity ─────────────────────────────────────────

def run_part_h(all_blocks, all_trips, library):
    """Sweep batch matching window to check sensitivity."""
    print("\n" + "=" * 80)
    print("PART H: BATCH WINDOW SENSITIVITY SWEEP")
    print("=" * 80)

    router = HaversineClient()
    seeds = [42, 123, 456]
    windows = [15, 30, 60, 90, 120]

    rows = []
    for scen_name in REPRESENTATIVE:
        month, hr, dtype = NYC_SCENARIOS[scen_name]
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            continue

        prep = _prepare_scenario(block, all_trips, library)
        print(f"\n  {scen_name} fleet={prep['fleet']}")

        for window in windows:
            for cfg_name in ["replay", "cal+lp"]:
                for seed in seeds:
                    rng = np.random.default_rng(seed)

                    if cfg_name == "replay":
                        stream = ReplayDemandStream(prep["replay_trips"])
                        repo = None
                    else:
                        pr = prior_matched_to_replay_volume(
                            prep["prior"], prep["horizon"], prep["n_trips"])
                        stream = CalibratedDemandStream(pr, prep["horizon"], rng=rng)
                        repo = AnticipatoryReposition(
                            pr, lookahead_minutes=5.0, max_move_fraction=0.50)

                    dispatch = BatchMatchingPolicy()
                    kpi = _run_sim(stream, prep["fleet"], prep["horizon"], seed,
                                   router, dispatch, repo, batch_window=window)
                    kpi.update({"scenario": scen_name, "config": cfg_name,
                                "batch_window": window, "seed": seed})
                    rows.append(kpi)

            rep_sub = [r for r in rows if r["scenario"] == scen_name and
                       r["config"] == "replay" and r["batch_window"] == window]
            cal_sub = [r for r in rows if r["scenario"] == scen_name and
                       r["config"] == "cal+lp" and r["batch_window"] == window]
            if rep_sub and cal_sub:
                rw = np.mean([r["mean_wait_s"] for r in rep_sub])
                cw = np.mean([r["mean_wait_s"] for r in cal_sub])
                imp = (rw - cw) / rw * 100 if rw > 0 else 0
                print(f"    window={window:3d}s: replay={rw:.1f}s  cal+lp={cw:.1f}s  Δ={imp:+.1f}%")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  BATCH WINDOW SUMMARY:")
        for window in windows:
            rep = df[(df["config"] == "replay") & (df["batch_window"] == window)]
            cal = df[(df["config"] == "cal+lp") & (df["batch_window"] == window)]
            if not rep.empty and not cal.empty:
                rw = rep.groupby("scenario")["mean_wait_s"].mean().mean()
                cw = cal.groupby("scenario")["mean_wait_s"].mean().mean()
                imp = (rw - cw) / rw * 100 if rw > 0 else 0
                print(f"    {window:3d}s: replay={rw:.1f}s  cal+lp={cw:.1f}s  ({imp:+.1f}%)")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Extended baselines and ablations")
    ap.add_argument("--parts", nargs="*", default=["F", "G", "H"],
                    choices=["F", "G", "H"], help="Parts to run")
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

    if "F" in parts:
        df_f = run_part_f(all_blocks, all_trips, library)
        if not df_f.empty:
            df_f.to_csv(out_dir / "extended_baselines.csv", index=False)
            print(f"  → Saved: {out_dir}/extended_baselines.csv")

    if "G" in parts:
        df_g = run_part_g(all_blocks, all_trips, library)
        if not df_g.empty:
            df_g.to_csv(out_dir / "topk_sensitivity.csv", index=False)
            print(f"  → Saved: {out_dir}/topk_sensitivity.csv")

    if "H" in parts:
        df_h = run_part_h(all_blocks, all_trips, library)
        if not df_h.empty:
            df_h.to_csv(out_dir / "batchwindow_sensitivity.csv", index=False)
            print(f"  → Saved: {out_dir}/batchwindow_sensitivity.csv")

    elapsed = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"EXTENDED EXPERIMENTS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
