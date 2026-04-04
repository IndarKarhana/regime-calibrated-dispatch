#!/usr/bin/env python3
"""Benchmark checkpoint system: run staged smoke tests against literature reference numbers.

Reference numbers (approximate, from published work):
- Namdarpour & Chow (2025): non-myopic RL improved service rate ~8.4% vs myopic baseline
- Zalesak et al. (2025): myopic algorithms ~comparable on Manhattan; LA-MR-CE best compute/quality
- Feng et al. (2024 OR): 3/4 competitive ratio for two-stage matching
- Our target: regime-calibrated prior should improve over flat/no-regime baselines on wait time
  and completion rate, with ablations showing each component contributes.

This script runs three checkpoint levels:
  Level 1 (FAST, ~30s):  Sanity check -- baselines run, greedy < random, regime library loads
  Level 2 (MEDIUM, ~5m): Ablation preview -- 3 configs x 1 seed, check calibrated < flat
  Level 3 (FULL, ~30m):  Full ablation matrix -- 8 configs x 5 seeds, generate report table
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
from src.regime.store import RegimeLibrary
from src.regime.events import annotate_events
from src.regime.similarity import query_library
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.engine import SimulationEngine
from src.simulator.routing import HaversineClient
from src.policies.greedy import GreedyNearestPolicy
from src.policies.batch import BatchMatchingPolicy
from src.policies.random_baseline import RandomPolicy
from src.evaluation.metrics import compute_kpis


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def _get_test_data():
    """Load data and prepare a busy-hour test slice + regime library."""
    trips = load_cleaned("2024-01")
    trips["hour"] = trips["pickup_datetime"].dt.hour
    trips["dow"] = trips["pickup_datetime"].dt.dayofweek
    busy = trips[(trips["hour"] == 8) & (trips["dow"] < 5)].head(15000)

    library = RegimeLibrary()
    try:
        library.load()
    except FileNotFoundError:
        library.build_from_trips(trips)
        library.save()

    profile = build_demand_profile(trips)
    blocks = split_into_blocks(profile)
    return trips, busy, library, profile, blocks


def _run_sim(policy, stream, fleet=200, horizon=3600, seed=42):
    router = HaversineClient()
    engine = SimulationEngine(stream, policy, router,
                              fleet_size=fleet, horizon_seconds=horizon, seed=seed)
    t0 = time.time()
    state = engine.run()
    return compute_kpis(state, time.time() - t0)


def checkpoint_level_1():
    """FAST sanity checks (~30s)."""
    print("\n" + "=" * 60)
    print("CHECKPOINT LEVEL 1: Sanity checks")
    print("=" * 60)
    passed = 0
    total = 0

    trips, busy, library, profile, blocks = _get_test_data()

    # --- Check 1: Regime library has records ---
    total += 1
    n_regimes = len(library)
    ok = n_regimes > 50
    print(f"\n  [1.1] Regime library size: {n_regimes} blocks ... {PASS if ok else FAIL}")
    if ok:
        passed += 1

    # --- Check 2: Greedy runs and produces reasonable wait ---
    total += 1
    stream = ReplayDemandStream(busy)
    kpi_g = _run_sim(GreedyNearestPolicy(), stream)
    ok = 30 < kpi_g["mean_wait_s"] < 600
    print(f"  [1.2] Greedy mean wait: {kpi_g['mean_wait_s']:.0f}s "
          f"(expect 30-600s) ... {PASS if ok else FAIL}")
    if ok:
        passed += 1

    # --- Check 3: Random is worse than greedy (sanity) ---
    total += 1
    stream = ReplayDemandStream(busy)
    kpi_r = _run_sim(RandomPolicy(), stream)
    ok = kpi_r["mean_wait_s"] > kpi_g["mean_wait_s"] * 1.5
    print(f"  [1.3] Random wait ({kpi_r['mean_wait_s']:.0f}s) > 1.5x greedy "
          f"({kpi_g['mean_wait_s']:.0f}s) ... {PASS if ok else FAIL}")
    if ok:
        passed += 1

    # --- Check 4: Completion rate > 0 for greedy ---
    total += 1
    ok = kpi_g["completion_rate"] > 0.5
    print(f"  [1.4] Greedy completion rate: {kpi_g['completion_rate']:.1%} "
          f"(expect >50%) ... {PASS if ok else FAIL}")
    if ok:
        passed += 1

    # --- Check 5: Batch runs and is competitive with greedy ---
    total += 1
    stream = ReplayDemandStream(busy)
    kpi_b = _run_sim(BatchMatchingPolicy(), stream)
    ok = kpi_b["mean_wait_s"] < kpi_g["mean_wait_s"] * 2.0
    print(f"  [1.5] Batch wait ({kpi_b['mean_wait_s']:.0f}s) within 2x of greedy "
          f"... {PASS if ok else FAIL}")
    if ok:
        passed += 1

    # --- Check 6: Regime similarity returns sensible scores ---
    total += 1
    blk = blocks[10] if len(blocks) > 10 else blocks[0]
    q_series = blk["request_count"].values.astype(np.float64)
    q_events = annotate_events(q_series)
    results = query_library(library, q_series, q_events, top_k=5)
    ok = len(results) == 5 and results[0][1] > 0.8
    print(f"  [1.6] Top match similarity: {results[0][1]:.3f} "
          f"(expect >0.8) ... {PASS if ok else FAIL}")
    if ok:
        passed += 1

    # --- Check 7: Calibrated demand stream generates requests ---
    total += 1
    mrecs = [library[bid] for bid, _ in results[:3]]
    mscores = [s for _, s in results[:3]]
    prior = build_calibrated_prior(mrecs, mscores, q_series)
    rng = np.random.default_rng(42)
    target_vol = float(np.sum(q_series))
    pr = prior_matched_to_replay_volume(prior, 14400.0, target_vol)
    cal_stream = CalibratedDemandStream(pr, 14400, rng=rng)
    ok = cal_stream.total_requests > 100
    print(f"  [1.7] Calibrated stream: {cal_stream.total_requests} requests "
          f"(expect >100) ... {PASS if ok else FAIL}")
    if ok:
        passed += 1

    print(f"\n  Level 1: {passed}/{total} passed")

    ref = {
        "greedy_wait_s": kpi_g["mean_wait_s"],
        "greedy_completion": kpi_g["completion_rate"],
        "batch_wait_s": kpi_b["mean_wait_s"],
        "random_wait_s": kpi_r["mean_wait_s"],
    }
    return passed == total, ref


def checkpoint_level_2(ref: dict | None = None):
    """MEDIUM ablation preview: compare calibrated vs flat vs replay on single-day block."""
    print("\n" + "=" * 60)
    print("CHECKPOINT LEVEL 2: Ablation preview (calibrated vs flat vs replay)")
    print("=" * 60)

    trips, busy, library, profile, blocks = _get_test_data()

    # Pick a single-day 4h block for apples-to-apples comparison
    test_blk = blocks[len(blocks) // 3]
    bid = test_blk["block_id"].iloc[0]
    q_series = test_blk["request_count"].values.astype(np.float64)
    q_events = annotate_events(q_series)
    print(f"\n  Test block: {bid}")
    print(f"  Demand profile: {len(q_series)} bins, "
          f"mean={np.mean(q_series):.1f} trips/bin, total~{np.sum(q_series):.0f}")

    # Extract single-day replay trips for this exact block
    date_str, hour_range = bid.rsplit("_", 1)
    start_h = int(hour_range.split("-")[0])
    end_h = int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()

    replay_trips = trips[
        (trips["pickup_datetime"].dt.date == target_date)
        & (trips["pickup_datetime"].dt.hour >= start_h)
        & (trips["pickup_datetime"].dt.hour < end_h)
    ]
    n_replay = len(replay_trips)
    print(f"  Replay trips for this block: {n_replay}")
    horizon = (end_h - start_h) * 3600.0
    fleet = max(int(n_replay / (horizon / 3600) * 0.15), 50)
    print(f"  Fleet size: {fleet}, horizon: {horizon/3600:.0f}h")

    matched = query_library(library, q_series, q_events, top_k=5)
    mrecs = [library[b] for b, _ in matched]
    mscores = [s for _, s in matched]
    print(f"  Top-5 match scores: {[f'{s:.3f}' for _, s in matched]}")

    prior_full = build_calibrated_prior(mrecs, mscores, q_series)
    prior_flat = build_calibrated_prior(mrecs, [1.0] * len(mrecs), q_series)

    from src.regime.similarity import compute_similarity
    no_event_w = {"ks": 0.3, "w1": 0.3, "feat": 0.2, "var": 0.2, "event": 0.0}
    no_event_scores = [compute_similarity(q_series, q_events, r, no_event_w) for r in mrecs]
    prior_no_event = build_calibrated_prior(mrecs, no_event_scores, q_series)

    print(f"  Calibrated (full) demand: ~{np.sum(prior_full.rate_profile):.0f} expected trips")
    print(f"  Calibrated (flat) demand: ~{np.sum(prior_flat.rate_profile):.0f} expected trips")

    vol = float(n_replay)

    configs = [
        ("greedy_replay", GreedyNearestPolicy(), lambda: ReplayDemandStream(replay_trips)),
        ("batch_replay", BatchMatchingPolicy(), lambda: ReplayDemandStream(replay_trips)),
        ("greedy_cal_full", GreedyNearestPolicy(),
         lambda: CalibratedDemandStream(
             prior_matched_to_replay_volume(prior_full, horizon, vol),
             horizon, rng=np.random.default_rng(42))),
        ("greedy_cal_no_event", GreedyNearestPolicy(),
         lambda: CalibratedDemandStream(
             prior_matched_to_replay_volume(prior_no_event, horizon, vol),
             horizon, rng=np.random.default_rng(42))),
        ("greedy_cal_flat", GreedyNearestPolicy(),
         lambda: CalibratedDemandStream(
             prior_matched_to_replay_volume(prior_flat, horizon, vol),
             horizon, rng=np.random.default_rng(42))),
    ]

    results = {}
    for name, policy, make_stream in configs:
        stream = make_stream()
        kpi = _run_sim(policy, stream, fleet=fleet, horizon=horizon)
        results[name] = kpi
        print(f"\n  {name}:")
        print(f"    mean_wait={kpi['mean_wait_s']:.0f}s  "
              f"p95_wait={kpi['p95_wait_s']:.0f}s  "
              f"completion={kpi['completion_rate']:.1%}  "
              f"throughput={kpi['throughput_trips_per_hour']:.0f}/h  "
              f"requests={kpi['total_requests']}")

    # --- Key comparison: calibrated full vs flat (both use greedy) ---
    full_wait = results["greedy_cal_full"]["mean_wait_s"]
    flat_wait = results["greedy_cal_flat"]["mean_wait_s"]
    noev_wait = results["greedy_cal_no_event"]["mean_wait_s"]
    replay_wait = results["greedy_replay"]["mean_wait_s"]
    diff_full_flat = (flat_wait - full_wait) / (flat_wait + 1e-9) * 100
    diff_full_noev = (noev_wait - full_wait) / (noev_wait + 1e-9) * 100

    full_cr = results["greedy_cal_full"]["completion_rate"]
    flat_cr = results["greedy_cal_flat"]["completion_rate"]
    cr_diff = (full_cr - flat_cr) / (flat_cr + 1e-9) * 100

    print(f"\n  {'='*50}")
    print(f"  KEY COMPARISONS (greedy policy, same fleet)")
    print(f"  {'='*50}")
    print(f"  Replay (ground truth):    wait={replay_wait:.0f}s")
    print(f"  Cal full (ensemble):      wait={full_wait:.0f}s")
    print(f"  Cal no-event:             wait={noev_wait:.0f}s")
    print(f"  Cal flat (uniform wt):    wait={flat_wait:.0f}s")
    print(f"  Full vs Flat wait:        {diff_full_flat:+.1f}%")
    print(f"  Full vs No-event wait:    {diff_full_noev:+.1f}%")
    print(f"  Full vs Flat completion:  {cr_diff:+.1f}%")

    if diff_full_flat > 1:
        print(f"\n  Regime-calibrated prior improves over flat: {PASS}")
    elif abs(diff_full_flat) < 5:
        print(f"\n  Within noise ({diff_full_flat:+.1f}%): {WARN} -- need more seeds")
    else:
        print(f"\n  Calibrated prior underperforms flat: {WARN} -- check weights/calibration")

    print(f"\n  --- Literature reference ---")
    print(f"  Namdarpour & Chow (2025): ~8.4% service rate improvement via sim-informed RL")
    print(f"  Our calibration-only wait improvement (no RL yet): {diff_full_flat:+.1f}%")
    print(f"  Our calibration-only completion improvement: {cr_diff:+.1f}%")

    return results


def checkpoint_level_3():
    """FULL ablation matrix."""
    print("\n" + "=" * 60)
    print("CHECKPOINT LEVEL 3: Full ablation matrix")
    print("=" * 60)
    from src.evaluation.ablation_runner import run_ablations
    df = run_ablations()

    if df.empty:
        print(f"  {FAIL}: No results produced")
        return df

    print("\n  --- Summary (mean across seeds) ---")
    summary = df.groupby("config").agg({
        "mean_wait_s": ["mean", "std"],
        "completion_rate": ["mean", "std"],
        "throughput_trips_per_hour": ["mean", "std"],
    }).round(2)
    print(summary.to_string())

    out = Path(get_config()["evaluation"]["output_dir"])
    summary.to_csv(out / "ablation_summary.csv")
    print(f"\n  Summary saved to {out / 'ablation_summary.csv'}")
    return df


def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    if level >= 1:
        ok, ref = checkpoint_level_1()
        if not ok:
            print(f"\n{FAIL}: Level 1 did not pass. Fix issues before proceeding.")
            if level > 1:
                sys.exit(1)

    if level >= 2:
        checkpoint_level_2(ref)

    if level >= 3:
        checkpoint_level_3()

    print("\n" + "=" * 60)
    print(f"Benchmark checkpoints complete (level {level})")
    print("=" * 60)


if __name__ == "__main__":
    main()
