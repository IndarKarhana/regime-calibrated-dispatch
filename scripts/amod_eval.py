#!/usr/bin/env python3
"""Small centralized AMoD/charging extension experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.external_baselines_eval import (  # noqa: E402
    _get_replay_trips,
    _pick_block,
    _top_matches,
)
from src.calibration.calibrator import (  # noqa: E402
    build_calibrated_prior,
    prior_matched_to_replay_volume,
)
from src.evaluation.metrics import compute_kpis  # noqa: E402
from src.policies.amod import ChargingAwareShareLPReposition  # noqa: E402
from src.policies.batch import BatchMatchingPolicy  # noqa: E402
from src.policies.external_baselines import ForecastShareLPReposition  # noqa: E402
from src.regime.events import annotate_events  # noqa: E402
from src.regime.ingest import build_demand_profile, load_cleaned, split_into_blocks  # noqa: E402
from src.regime.learned_weights import PATH2_SCENARIOS, load_gate  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402
from src.simulator.demand import ReplayDemandStream  # noqa: E402
from src.simulator.engine import SimulationEngine  # noqa: E402
from src.simulator.routing import HaversineClient  # noqa: E402


def _run_one(
    replay_trips: pd.DataFrame,
    fleet: int,
    horizon: float,
    seed: int,
    repo_policy,
) -> dict:
    engine = SimulationEngine(
        demand_stream=ReplayDemandStream(replay_trips),
        policy=BatchMatchingPolicy(),
        router=HaversineClient(),
        fleet_size=fleet,
        horizon_seconds=horizon,
        seed=seed,
        reposition_policy=repo_policy,
        reposition_interval_steps=6,
    )
    start = time.time()
    state = engine.run()
    kpi = compute_kpis(state, time.time() - start)
    if hasattr(repo_policy, "charging_summary"):
        kpi.update(repo_policy.charging_summary())
    else:
        kpi.update({
            "charge_violation_rate": 0.0,
            "forced_charging_moves": 0.0,
            "charger_zone_count": 0.0,
        })
    return kpi


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    library = RegimeLibrary()
    library.load()
    gate = load_gate(args.gate)
    jan = load_cleaned("2024-01")
    jun = load_cleaned("2024-06")
    trips = pd.concat([jan, jun], ignore_index=True)
    blocks = split_into_blocks(build_demand_profile(trips))

    rows = []
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for scenario, (month, hour_range, day_type) in PATH2_SCENARIOS.items():
        block = _pick_block(blocks, month, hour_range, day_type)
        if block is None:
            continue
        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].to_numpy(dtype=np.float64)
        qrec = library[bid] if bid in library.records else None
        q_events = qrec.events if qrec is not None else annotate_events(q_series)
        replay_trips = _get_replay_trips(trips, bid)
        if replay_trips.empty:
            continue

        horizon = 4 * 3600.0
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / 4.0 * args.fleet_ratio), args.min_fleet)
        learned_weights = (
            gate(qrec.demand_series, qrec.events, bid, qrec)
            if qrec is not None else gate(q_series, q_events, bid)
        )
        matches = _top_matches(library, q_series, q_events, bid, learned_weights, args.top_k)
        prior = build_calibrated_prior(
            [library[cid] for cid, _ in matches],
            [score for _, score in matches],
            q_series,
        )
        prior = prior_matched_to_replay_volume(prior, horizon, n_trips)

        print(f"\n{scenario} [{bid}] trips={n_trips} fleet={fleet}")
        factories = {
            "central_share_lp": lambda: ForecastShareLPReposition(
                prior,
                max_move_fraction=0.30,
            ),
            "charging_aware_share_lp": lambda: ChargingAwareShareLPReposition(
                prior,
                max_move_fraction=0.30,
                charge_threshold=args.charge_threshold,
                low_charge_start_fraction=args.low_charge_start_fraction,
            ),
        }
        for method, factory in factories.items():
            waits = []
            violations = []
            for seed in seeds:
                policy = factory()
                kpi = _run_one(replay_trips, fleet, horizon, seed, policy)
                kpi.update({
                    "scenario": scenario,
                    "block_id": bid,
                    "method": method,
                    "seed": seed,
                    "fleet": fleet,
                    "n_trips": n_trips,
                })
                rows.append(kpi)
                waits.append(kpi["mean_wait_s"])
                violations.append(kpi["charge_violation_rate"])
            print(
                f"  {method:24s}: wait={np.mean(waits):6.1f}s "
                f"charge-low/call={np.mean(violations):5.2f}"
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    summary = df.groupby("method", as_index=False).agg(
        mean_wait_s=("mean_wait_s", "mean"),
        std_wait_s=("mean_wait_s", "std"),
        completion_rate=("completion_rate", "mean"),
        charge_violation_rate=("charge_violation_rate", "mean"),
        forced_charging_moves=("forced_charging_moves", "mean"),
        n=("mean_wait_s", "size"),
    )
    summary.to_csv(out_dir / "amod_summary.csv", index=False)
    by_scenario = df.groupby(["scenario", "method"], as_index=False).agg(
        mean_wait_s=("mean_wait_s", "mean"),
        completion_rate=("completion_rate", "mean"),
        charge_violation_rate=("charge_violation_rate", "mean"),
    )
    by_scenario.to_csv(out_dir / "amod_by_scenario.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="results/path2_gate_spatial/similarity_gate.pt")
    parser.add_argument("--out-dir", default="results/path2_amod")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fleet-ratio", type=float, default=0.15)
    parser.add_argument("--min-fleet", type=int, default=50)
    parser.add_argument("--charge-threshold", type=float, default=0.18)
    parser.add_argument("--low-charge-start-fraction", type=float, default=0.12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate(args)
    df.to_csv(out_dir / "amod_eval.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    summarize(df, out_dir)


if __name__ == "__main__":
    main()
