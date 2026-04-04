"""Run ablation matrix: multiple policies x seeds, collect KPIs.

Uses a single-day test block for apples-to-apples comparison: replay trips from
that block vs calibrated demand with matching total volume and horizon.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.similarity import query_library, compute_similarity
from src.regime.events import annotate_events
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.engine import SimulationEngine
from src.simulator.routing import HaversineClient
from src.policies.greedy import GreedyNearestPolicy
from src.policies.batch import BatchMatchingPolicy
from src.policies.random_baseline import RandomPolicy
from src.evaluation.metrics import compute_kpis


def _run_one(policy, demand_stream, router, seed, horizon, fleet_size=200) -> dict:
    t0 = time.time()
    engine = SimulationEngine(
        demand_stream=demand_stream,
        policy=policy,
        router=router,
        fleet_size=fleet_size,
        horizon_seconds=horizon,
        seed=seed,
    )
    state = engine.run()
    wall = time.time() - t0
    return compute_kpis(state, wall)


def _select_test_block(trips, blocks):
    """Pick a busy weekday morning block for testing."""
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        total = blk["request_count"].sum()
        if "08-12" in bid and total > 2000:
            return blk
    return blocks[len(blocks) // 2]


def run_ablations() -> pd.DataFrame:
    cfg = get_config()
    seeds = cfg["evaluation"]["seeds"]
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(exist_ok=True)
    router = HaversineClient()

    print("Loading data ...")
    trips = load_cleaned("2024-01")
    library = RegimeLibrary()
    try:
        library.load()
    except FileNotFoundError:
        library.build_from_trips(trips)
        library.save()

    profile = build_demand_profile(trips)
    blocks = split_into_blocks(profile)

    test_block = _select_test_block(trips, blocks)
    bid = test_block["block_id"].iloc[0]
    test_series = test_block["request_count"].values.astype(np.float64)

    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    horizon = float((end_h - start_h) * 3600)

    replay_trips = trips[
        (trips["pickup_datetime"].dt.date == target_date)
        & (trips["pickup_datetime"].dt.hour >= start_h)
        & (trips["pickup_datetime"].dt.hour < end_h)
    ]
    n_trips = len(replay_trips)
    fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

    print(f"Test block: {bid}, {n_trips} trips, fleet={fleet}, horizon={horizon/3600:.0f}h")

    test_events = annotate_events(test_series)
    matched = query_library(library, test_series, test_events)
    mrecs = [library[b] for b, _ in matched]
    mscores = [s for _, s in matched]

    prior_full = build_calibrated_prior(mrecs, mscores, test_series)

    flat_scores = [1.0] * len(mrecs)
    prior_flat = build_calibrated_prior(mrecs, flat_scores, test_series)

    w1_only_weights = {"ks": 0, "w1": 1, "feat": 0, "var": 0, "event": 0}
    w1_scores = [compute_similarity(test_series, test_events, r, w1_only_weights) for r in mrecs]
    prior_w1 = build_calibrated_prior(mrecs, w1_scores, test_series)

    no_event_w = {"ks": 0.3, "w1": 0.3, "feat": 0.2, "var": 0.2, "event": 0.0}
    no_event_scores = [compute_similarity(test_series, test_events, r, no_event_w) for r in mrecs]
    prior_no_event = build_calibrated_prior(mrecs, no_event_scores, test_series)

    tgt = float(n_trips)

    def _make_stream(stream_type, seed):
        rng = np.random.default_rng(seed)
        if stream_type == "replay":
            return ReplayDemandStream(replay_trips)
        elif stream_type == "cal_full":
            pr = prior_matched_to_replay_volume(prior_full, horizon, tgt)
            return CalibratedDemandStream(pr, horizon, rng=rng)
        elif stream_type == "cal_no_event":
            pr = prior_matched_to_replay_volume(prior_no_event, horizon, tgt)
            return CalibratedDemandStream(pr, horizon, rng=rng)
        elif stream_type == "cal_w1":
            pr = prior_matched_to_replay_volume(prior_w1, horizon, tgt)
            return CalibratedDemandStream(pr, horizon, rng=rng)
        elif stream_type == "cal_flat":
            pr = prior_matched_to_replay_volume(prior_flat, horizon, tgt)
            return CalibratedDemandStream(pr, horizon, rng=rng)

    configs = [
        ("greedy_replay", GreedyNearestPolicy, "replay"),
        ("batch_replay", BatchMatchingPolicy, "replay"),
        ("random_replay", RandomPolicy, "replay"),
        ("greedy_cal_full", GreedyNearestPolicy, "cal_full"),
        ("batch_cal_full", BatchMatchingPolicy, "cal_full"),
        ("greedy_cal_no_event", GreedyNearestPolicy, "cal_no_event"),
        ("greedy_cal_w1_only", GreedyNearestPolicy, "cal_w1"),
        ("greedy_cal_flat", GreedyNearestPolicy, "cal_flat"),
    ]

    rows = []
    for name, PolicyCls, stream_type in configs:
        print(f"\n=== {name} ===")
        for seed in seeds:
            policy = PolicyCls(seed=seed) if PolicyCls == RandomPolicy else PolicyCls()
            stream = _make_stream(stream_type, seed)
            kpis = _run_one(policy, stream, router, seed, horizon, fleet)
            kpis["config"] = name
            kpis["seed"] = seed
            rows.append(kpis)
            print(f"  seed={seed}: wait={kpis['mean_wait_s']:.1f}s, "
                  f"complete={kpis['completion_rate']:.2%}, "
                  f"throughput={kpis['throughput_trips_per_hour']:.0f}/h")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)

    summary = df.groupby("config").agg({
        "mean_wait_s": ["mean", "std"],
        "p95_wait_s": ["mean"],
        "completion_rate": ["mean", "std"],
        "throughput_trips_per_hour": ["mean"],
        "mean_pickup_dist_m": ["mean"],
    }).round(2)
    summary_path = out_dir / "ablation_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nResults: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"\n{summary.to_string()}")
    return df


if __name__ == "__main__":
    run_ablations()
