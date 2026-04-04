#!/usr/bin/env python3
"""Cross-scenario ablation: test the similarity ensemble across diverse demand regimes.

Goal: determine whether the 5-metric ensemble earns its keep vs flat/w1-only/no-event
when tested on structurally different blocks (winter AM, winter PM, summer AM,
summer PM, New Year's Day, weekend).

For each test block:
  - Exclude that block from the library (leave-one-out)
  - Query top-5 matches using full ensemble, flat, w1-only, no-event
  - Build calibrated prior from each
  - Run greedy + batch with each prior, also replay
  - Collect KPIs
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary, RegimeRecord
from src.regime.events import annotate_events, detect_events
from src.regime.similarity import query_library, compute_similarity
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.engine import SimulationEngine
from src.simulator.routing import HaversineClient
from src.policies.greedy import GreedyNearestPolicy
from src.policies.batch import BatchMatchingPolicy
from src.evaluation.metrics import compute_kpis


def _pick_test_blocks(trips: pd.DataFrame, blocks: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Select structurally diverse test blocks."""
    scenarios = {}
    trips["date"] = trips["pickup_datetime"].dt.date
    trips["hour"] = trips["pickup_datetime"].dt.hour
    trips["dow"] = trips["pickup_datetime"].dt.dayofweek
    trips["month"] = trips["pickup_datetime"].dt.month

    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        total = blk["request_count"].sum()
        n_events = len(detect_events(blk["request_count"].values.astype(np.float64)))

        if "2024-01-01" in bid and "16-20" in bid and "jan_nye_pm" not in scenarios:
            scenarios["jan_nye_pm"] = blk
        elif "2024-01-01" in bid and "08-12" in bid and "jan_nye_am" not in scenarios:
            scenarios["jan_nye_am"] = blk
        elif bid.startswith("2024-01") and "08-12" in bid and total > 3000 and "jan_weekday_am" not in scenarios:
            date_str = bid.split("_")[0]
            dow = pd.Timestamp(date_str).dayofweek
            if dow < 5:
                scenarios["jan_weekday_am"] = blk
        elif bid.startswith("2024-01") and "16-20" in bid and total > 3000 and "jan_weekday_pm" not in scenarios:
            date_str = bid.split("_")[0]
            dow = pd.Timestamp(date_str).dayofweek
            if dow < 5:
                scenarios["jan_weekday_pm"] = blk
        elif bid.startswith("2024-01") and "12-16" in bid and "jan_weekend_mid" not in scenarios:
            date_str = bid.split("_")[0]
            dow = pd.Timestamp(date_str).dayofweek
            if dow >= 5:
                scenarios["jan_weekend_mid"] = blk
        elif bid.startswith("2024-06") and "08-12" in bid and total > 3000 and "jun_weekday_am" not in scenarios:
            date_str = bid.split("_")[0]
            dow = pd.Timestamp(date_str).dayofweek
            if dow < 5:
                scenarios["jun_weekday_am"] = blk
        elif bid.startswith("2024-06") and "16-20" in bid and total > 3000 and "jun_weekday_pm" not in scenarios:
            date_str = bid.split("_")[0]
            dow = pd.Timestamp(date_str).dayofweek
            if dow < 5:
                scenarios["jun_weekday_pm"] = blk
        elif bid.startswith("2024-06") and "20-24" in bid and "jun_late_night" not in scenarios:
            date_str = bid.split("_")[0]
            dow = pd.Timestamp(date_str).dayofweek
            if dow == 4 or dow == 5:  # Fri/Sat night
                scenarios["jun_late_night"] = blk

    return scenarios


def _run_sim(policy, stream, fleet, horizon, seed=42) -> dict:
    router = HaversineClient()
    engine = SimulationEngine(stream, policy, router,
                              fleet_size=fleet, horizon_seconds=horizon, seed=seed)
    t0 = time.time()
    state = engine.run()
    return compute_kpis(state, time.time() - t0)


def _get_replay_trips(trips, block_id, block_hours):
    parts = block_id.rsplit("_", 1)
    date_str, hour_range = parts
    start_h = int(hour_range.split("-")[0])
    end_h = int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    return trips[
        (trips["pickup_datetime"].dt.date == target_date)
        & (trips["pickup_datetime"].dt.hour >= start_h)
        & (trips["pickup_datetime"].dt.hour < end_h)
    ]


def run_cross_scenario():
    cfg = get_config()
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(exist_ok=True)
    block_hours = cfg["regime"]["block_hours"]

    print("Loading all data ...")
    trips = load_cleaned()
    library = RegimeLibrary()
    try:
        library.load()
    except FileNotFoundError:
        library.build_from_trips(trips)
        library.save()

    profile = build_demand_profile(trips)
    blocks = split_into_blocks(profile)

    print(f"Library: {len(library)} regimes")
    scenarios = _pick_test_blocks(trips, blocks)
    print(f"Selected {len(scenarios)} test scenarios: {list(scenarios.keys())}")

    similarity_configs = {
        "full_ensemble": None,  # default weights (includes temporal)
        "flat_weights": "flat",
        "no_temporal": {"ks": 0.25, "w1": 0.25, "feat": 0.15, "var": 0.15, "event": 0.20, "temporal": 0},
        "w1_only": {"ks": 0, "w1": 1, "feat": 0, "var": 0, "event": 0, "temporal": 0},
        "no_event": {"ks": 0.3, "w1": 0.3, "feat": 0.2, "var": 0.2, "event": 0.0, "temporal": 0},
    }

    all_rows = []

    for scenario_name, test_blk in scenarios.items():
        bid = test_blk["block_id"].iloc[0]
        q_series = test_blk["request_count"].values.astype(np.float64)
        q_events = annotate_events(q_series)
        n_events = len(q_events)

        date_str, hour_range = bid.rsplit("_", 1)
        start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
        horizon = float((end_h - start_h) * 3600)

        replay_trips = _get_replay_trips(trips, bid, block_hours)
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"  Block: {bid}, trips={n_trips}, fleet={fleet}, events={n_events}")
        print(f"  Demand: mean={np.mean(q_series):.0f}/bin, std={np.std(q_series):.0f}, "
              f"total={np.sum(q_series):.0f}")
        print(f"{'='*60}")

        # Leave-one-out: exclude the exact test block from matches
        excluded_bid = bid

        # Replay baselines
        for policy_name, PolicyCls in [("greedy", GreedyNearestPolicy), ("batch", BatchMatchingPolicy)]:
            stream = ReplayDemandStream(replay_trips)
            kpi = _run_sim(PolicyCls(), stream, fleet, horizon)
            kpi["scenario"] = scenario_name
            kpi["block_id"] = bid
            kpi["policy"] = policy_name
            kpi["calibration"] = "replay"
            kpi["n_trips"] = n_trips
            kpi["n_events"] = n_events
            all_rows.append(kpi)
            print(f"  {policy_name}_replay: wait={kpi['mean_wait_s']:.0f}s, "
                  f"comp={kpi['completion_rate']:.1%}")

        # Calibrated variants
        for sim_name, sim_weights in similarity_configs.items():
            if sim_weights == "flat":
                matched = query_library(library, q_series, q_events, top_k=6, q_block_id=bid)
                matched = [(b, s) for b, s in matched if b != excluded_bid][:5]
                mrecs = [library[b] for b, _ in matched]
                mscores = [1.0] * len(mrecs)
            elif sim_weights is not None:
                scores = []
                for b, rec in library.records.items():
                    if b == excluded_bid:
                        continue
                    s = compute_similarity(q_series, q_events, rec, sim_weights, q_block_id=bid)
                    scores.append((b, s))
                scores.sort(key=lambda x: x[1], reverse=True)
                matched = scores[:5]
                mrecs = [library[b] for b, _ in matched]
                mscores = [s for _, s in matched]
            else:
                matched = query_library(library, q_series, q_events, top_k=6, q_block_id=bid)
                matched = [(b, s) for b, s in matched if b != excluded_bid][:5]
                mrecs = [library[b] for b, _ in matched]
                mscores = [s for _, s in matched]

            if not mrecs:
                continue

            prior = build_calibrated_prior(mrecs, mscores, q_series)

            for policy_name, PolicyCls in [("greedy", GreedyNearestPolicy), ("batch", BatchMatchingPolicy)]:
                rng = np.random.default_rng(42)
                pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
                stream = CalibratedDemandStream(pr, horizon, rng=rng)
                kpi = _run_sim(PolicyCls(), stream, fleet, horizon)
                kpi["scenario"] = scenario_name
                kpi["block_id"] = bid
                kpi["policy"] = policy_name
                kpi["calibration"] = sim_name
                kpi["n_trips"] = n_trips
                kpi["n_events"] = n_events
                all_rows.append(kpi)
                print(f"  {policy_name}_{sim_name}: wait={kpi['mean_wait_s']:.0f}s, "
                      f"comp={kpi['completion_rate']:.1%}")

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "cross_scenario_ablation.csv"
    df.to_csv(csv_path, index=False)

    # Summary pivot
    print("\n" + "=" * 80)
    print("SUMMARY: Mean wait (s) by scenario x calibration method (greedy policy)")
    print("=" * 80)
    greedy = df[df["policy"] == "greedy"]
    if not greedy.empty:
        pivot = greedy.pivot_table(
            values="mean_wait_s", index="scenario", columns="calibration", aggfunc="mean"
        ).round(0)
        if "replay" in pivot.columns:
            for col in pivot.columns:
                if col != "replay":
                    pivot[f"{col}_vs_replay_%"] = (
                        (pivot["replay"] - pivot[col]) / pivot["replay"] * 100
                    ).round(1)
        print(pivot.to_string())
        pivot.to_csv(out_dir / "cross_scenario_summary_greedy.csv")

    print("\n" + "=" * 80)
    print("SUMMARY: Mean wait (s) by scenario x calibration method (batch policy)")
    print("=" * 80)
    batch = df[df["policy"] == "batch"]
    if not batch.empty:
        pivot_b = batch.pivot_table(
            values="mean_wait_s", index="scenario", columns="calibration", aggfunc="mean"
        ).round(0)
        if "replay" in pivot_b.columns:
            for col in pivot_b.columns:
                if col != "replay":
                    pivot_b[f"{col}_vs_replay_%"] = (
                        (pivot_b["replay"] - pivot_b[col]) / pivot_b["replay"] * 100
                    ).round(1)
        print(pivot_b.to_string())
        pivot_b.to_csv(out_dir / "cross_scenario_summary_batch.csv")

    # Event analysis
    print("\n" + "=" * 80)
    print("EVENT ANALYSIS: Does event similarity help on event-rich blocks?")
    print("=" * 80)
    greedy_full = greedy[greedy["calibration"] == "full_ensemble"].set_index("scenario")
    greedy_noev = greedy[greedy["calibration"] == "no_event"].set_index("scenario")
    for sc in greedy_full.index:
        if sc in greedy_noev.index:
            full_w = greedy_full.loc[sc, "mean_wait_s"]
            noev_w = greedy_noev.loc[sc, "mean_wait_s"]
            n_ev = greedy_full.loc[sc, "n_events"]
            diff = (noev_w - full_w) / noev_w * 100
            tag = "HELPS" if diff > 2 else ("HURTS" if diff < -2 else "NEUTRAL")
            print(f"  {sc}: events={n_ev}, full={full_w:.0f}s, no_event={noev_w:.0f}s, "
                  f"diff={diff:+.1f}% [{tag}]")

    print(f"\nResults saved to {csv_path}")
    return df


if __name__ == "__main__":
    run_cross_scenario()
