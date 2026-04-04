"""Evaluate optimization-based repositioning vs batch-only baseline.

Runs across the same 8 diverse scenarios from cross_scenario_ablation,
comparing:
  1. batch_only:     batch dispatch, no repositioning
  2. batch+heuristic: batch dispatch + demand-following heuristic reposition
  3. batch+lp:       batch dispatch + LP anticipatory reposition

All use calibrated demand from the full similarity ensemble.
"""

from __future__ import annotations

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
from src.simulator.routing import HaversineClient
from src.policies.batch import BatchMatchingPolicy
from src.policies.anticipatory import (
    AnticipatoryReposition,
    DemandFollowingReposition,
)
from src.evaluation.metrics import compute_kpis


SCENARIOS = {
    "jan_weekday_am": {"month": "2024-01", "hour_range": "08-12", "day_type": "weekday"},
    "jan_weekday_pm": {"month": "2024-01", "hour_range": "16-20", "day_type": "weekday"},
    "jan_nye_am":     {"month": "2024-01", "hour_range": "08-12", "day_type": "nye"},
    "jan_nye_pm":     {"month": "2024-01", "hour_range": "16-20", "day_type": "nye"},
    "jan_weekend_mid":{"month": "2024-01", "hour_range": "12-16", "day_type": "weekend"},
    "jun_weekday_am": {"month": "2024-06", "hour_range": "08-12", "day_type": "weekday"},
    "jun_weekday_pm": {"month": "2024-06", "hour_range": "16-20", "day_type": "weekday"},
    "jun_late_night": {"month": "2024-06", "hour_range": "00-04", "day_type": "weekday"},
}


def _find_block(blocks, hour_range: str, day_type: str):
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        if hour_range not in bid:
            continue
        total = blk["request_count"].sum()
        if total < 500:
            continue
        date_str = bid.rsplit("_", 1)[0]
        ts = pd.Timestamp(date_str)
        dow = ts.dayofweek
        if day_type == "weekday" and dow < 5:
            return blk
        elif day_type == "weekend" and dow >= 5:
            return blk
        elif day_type == "nye" and ts.month == 1 and ts.day == 1:
            return blk
    return None


def _run_scenario(
    scenario_name: str,
    library: RegimeLibrary,
    blocks: list,
    all_trips: pd.DataFrame,
    router,
    seeds: list[int],
) -> list[dict]:
    spec = SCENARIOS[scenario_name]
    block = _find_block(blocks, spec["hour_range"], spec["day_type"])
    if block is None:
        print(f"  SKIP {scenario_name}: no matching block found")
        return []

    bid = block["block_id"].iloc[0]
    q_series = block["request_count"].values.astype(np.float64)
    q_events = annotate_events(q_series)
    matched = query_library(library, q_series, q_events, q_block_id=bid)
    mrecs = [library[b] for b, _ in matched]
    mscores = [s for _, s in matched]
    base_prior = build_calibrated_prior(mrecs, mscores, q_series)

    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    if end_h <= start_h:
        end_h += 24
    horizon = float((end_h - start_h) * 3600)

    target_date = pd.Timestamp(date_str).date()
    replay_trips = all_trips[
        (all_trips["pickup_datetime"].dt.date == target_date)
        & (all_trips["pickup_datetime"].dt.hour >= start_h)
        & (all_trips["pickup_datetime"].dt.hour < (end_h % 24))
    ]
    n_trips = len(replay_trips)
    fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

    print(f"\n  {scenario_name}: block={bid}, trips={n_trips}, fleet={fleet}")

    configs = [
        ("batch_only", None),
        ("batch+heuristic", lambda pr: DemandFollowingReposition(
            pr, lookahead_minutes=15.0, max_move_fraction=0.25,
        )),
        ("batch+lp", lambda pr: AnticipatoryReposition(
            pr, lookahead_minutes=15.0, max_move_fraction=0.35,
        )),
    ]

    rows = []
    for config_name, repo_factory in configs:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            pr = prior_matched_to_replay_volume(base_prior, horizon, n_trips)
            stream = CalibratedDemandStream(pr, horizon, rng=rng)
            repo_policy = repo_factory(pr) if repo_factory else None
            dispatch = BatchMatchingPolicy()

            t0 = time.time()
            engine = SimulationEngine(
                demand_stream=stream,
                policy=dispatch,
                router=router,
                fleet_size=fleet,
                horizon_seconds=horizon,
                seed=seed,
                reposition_policy=repo_policy,
                reposition_interval_steps=6,
            )
            state = engine.run()
            wall = time.time() - t0
            kpis = compute_kpis(state, wall)
            kpis["scenario"] = scenario_name
            kpis["config"] = config_name
            kpis["seed"] = seed
            rows.append(kpis)

        avg_wait = np.mean([r["mean_wait_s"] for r in rows if r["config"] == config_name and r["scenario"] == scenario_name])
        avg_comp = np.mean([r["completion_rate"] for r in rows if r["config"] == config_name and r["scenario"] == scenario_name])
        print(f"    {config_name:20s}: wait={avg_wait:.0f}s, comp={avg_comp:.1%}")

    return rows


def main():
    cfg = get_config()
    seeds = cfg["evaluation"]["seeds"]
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

    print(f"  {len(all_blocks)} blocks available ({len(jan_blocks)} Jan + {len(jun_blocks)} Jun)")

    all_rows = []
    for scenario in SCENARIOS:
        rows = _run_scenario(scenario, library, all_blocks, all_trips, router, seeds)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "repositioning_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY: Mean across seeds per scenario")
    print("=" * 80)

    summary = df.groupby(["scenario", "config"]).agg({
        "mean_wait_s": "mean",
        "completion_rate": "mean",
        "throughput_trips_per_hour": "mean",
        "mean_pickup_dist_m": "mean",
    }).round(1)

    pivot_wait = df.pivot_table(
        values="mean_wait_s", index="scenario", columns="config", aggfunc="mean",
    ).round(0)

    pivot_comp = df.pivot_table(
        values="completion_rate", index="scenario", columns="config", aggfunc="mean",
    ).round(3)

    print("\nWait time (seconds):")
    print(pivot_wait.to_string())

    print("\nCompletion rate:")
    print(pivot_comp.to_string())

    if "batch_only" in pivot_wait.columns and "batch+lp" in pivot_wait.columns:
        improvement = (
            (pivot_wait["batch_only"] - pivot_wait["batch+lp"])
            / pivot_wait["batch_only"] * 100
        ).round(1)
        print(f"\nLP vs batch-only wait reduction:")
        for scen, pct in improvement.items():
            print(f"  {scen:25s}: {pct:+.1f}%")
        print(f"  {'AVERAGE':25s}: {improvement.mean():+.1f}%")

    if "batch_only" in pivot_wait.columns and "batch+heuristic" in pivot_wait.columns:
        improvement_h = (
            (pivot_wait["batch_only"] - pivot_wait["batch+heuristic"])
            / pivot_wait["batch_only"] * 100
        ).round(1)
        print(f"\nHeuristic vs batch-only wait reduction:")
        for scen, pct in improvement_h.items():
            print(f"  {scen:25s}: {pct:+.1f}%")
        print(f"  {'AVERAGE':25s}: {improvement_h.mean():+.1f}%")

    summary_path = out_dir / "repositioning_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nDetailed: {csv_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
