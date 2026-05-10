#!/usr/bin/env python3
"""Evaluate Path 2 similarity gates and static baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config  # noqa: E402
from src.calibration.calibrator import (  # noqa: E402
    build_calibrated_prior,
    prior_matched_to_replay_volume,
)
from src.evaluation.metrics import compute_kpis  # noqa: E402
from src.policies.anticipatory import AnticipatoryReposition  # noqa: E402
from src.policies.batch import BatchMatchingPolicy  # noqa: E402
from src.regime.ingest import build_demand_profile, load_cleaned, split_into_blocks  # noqa: E402
from src.regime.learned_weights import (  # noqa: E402
    PATH2_SCENARIOS,
    build_gate_dataset,
    default_weight_vectors,
    evaluate_dataset,
    leakage_safe_split,
    load_gate,
)
from src.regime.similarity import compute_similarity, weights_to_dict  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402
from src.simulator.demand import CalibratedDemandStream, ReplayDemandStream  # noqa: E402
from src.simulator.engine import SimulationEngine  # noqa: E402
from src.simulator.routing import HaversineClient  # noqa: E402


def _pick_block(blocks, month_prefix: str, hour_range: str, day_type: str):
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        if not bid.startswith(month_prefix) or hour_range not in bid:
            continue
        if blk["request_count"].sum() < 500:
            continue
        date_str = bid.rsplit("_", 1)[0]
        ts = pd.Timestamp(date_str)
        if ts.month == 1 and ts.day == 1:
            dtype = "holiday"
        elif ts.dayofweek >= 5:
            dtype = "weekend"
        else:
            dtype = "weekday"
        if dtype == day_type:
            return blk
    return None


def _get_replay_trips(all_trips: pd.DataFrame, bid: str) -> pd.DataFrame:
    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    mask = all_trips["pickup_datetime"].dt.date == target_date
    mask &= all_trips["pickup_datetime"].dt.hour >= start_h
    mask &= all_trips["pickup_datetime"].dt.hour < end_h
    return all_trips[mask]


def _run_one(stream, fleet: int, horizon: float, seed: int, repo_policy=None) -> dict:
    engine = SimulationEngine(
        demand_stream=stream,
        policy=BatchMatchingPolicy(),
        router=HaversineClient(),
        fleet_size=fleet,
        horizon_seconds=horizon,
        seed=seed,
        reposition_policy=repo_policy,
        reposition_interval_steps=6,
    )
    t0 = time.time()
    state = engine.run()
    return compute_kpis(state, time.time() - t0)


def _causal_prefix_total(
    q_series: np.ndarray,
    horizon_seconds: float,
    *,
    prefix_minutes: float = 30.0,
) -> float:
    """Predict horizon volume using only the first causal prefix of the block.

    This is intentionally simple: after observing the first ``prefix_minutes``,
    extrapolate the prefix average rate to the full horizon. It is deployable
    after the first half hour of a block and provides a leakage-free alternative
    to matching the realized replay volume.
    """
    cfg = get_config()["regime"]
    bin_sec = float(cfg["bin_interval_minutes"] * 60)
    horizon_bins = max(1, int(np.ceil(horizon_seconds / bin_sec)))
    prefix_bins = max(1, int(np.ceil(prefix_minutes * 60.0 / bin_sec)))
    prefix = np.asarray(q_series[: min(prefix_bins, len(q_series))], dtype=np.float64)
    if prefix.size == 0:
        return 0.0
    return float(max(prefix.mean(), 0.0) * horizon_bins)


def _causal_prefix_series(q_series: np.ndarray, *, prefix_minutes: float = 30.0) -> np.ndarray:
    """Return the part of a query series observable in the first prefix window."""
    cfg = get_config()["regime"]
    bin_sec = float(cfg["bin_interval_minutes"] * 60)
    prefix_bins = max(1, int(np.ceil(prefix_minutes * 60.0 / bin_sec)))
    return np.asarray(q_series[: min(prefix_bins, len(q_series))], dtype=np.float64)


def _query_series_for_stats(q_series: np.ndarray, mode: str) -> np.ndarray | None:
    """Choose which query statistics the calibrator may see."""
    if mode == "full":
        return q_series
    if mode == "prefix30":
        return _causal_prefix_series(q_series, prefix_minutes=30.0)
    if mode == "none":
        return None
    raise ValueError(f"Unknown query stats mode: {mode}")


def _apply_volume_mode(prior, mode: str, horizon: float, replay_total: int, q_series: np.ndarray):
    """Return a forked/scaled prior and the expected total implied by ``mode``."""
    if mode == "replay":
        return prior_matched_to_replay_volume(prior, horizon, replay_total), float(replay_total)
    if mode == "prefix30":
        target = _causal_prefix_total(q_series, horizon, prefix_minutes=30.0)
        return prior_matched_to_replay_volume(prior, horizon, target), float(target)
    if mode == "none":
        p = prior.fork()
        return p, float(np.sum(p.rate_profile))
    raise ValueError(f"Unknown volume mode: {mode}")


def retrieval_eval(lib: RegimeLibrary, gate_path: Path, out_dir: Path) -> None:
    split = leakage_safe_split(lib)
    ds = build_gate_dataset(
        lib, split.holdout_ids, split.train_ids, split, normalize_from=split.train_ids
    )
    learned = load_gate(gate_path)

    # Evaluate the loaded callable by querying its weights once per block.
    learned_rows = []
    for i, qid in enumerate(ds.query_ids):
        qrec = lib[qid]
        w = np.array(
            list(learned(qrec.demand_series, qrec.events, qid, qrec).values()),
            dtype=np.float32,
        )
        learned_rows.append(w)

    weights = default_weight_vectors()
    weights["learned_gate_loaded"] = np.vstack(learned_rows)
    df = evaluate_dataset(ds, weights)
    df.to_csv(out_dir / "gate_holdout_retrieval_eval.csv", index=False)
    print("\nHoldout retrieval summary:")
    print(df[df["query_id"] == "__MEAN__"][[
        "method", "spearman_rho", "top5_vs_median_improvement_pct",
    ]].sort_values("spearman_rho", ascending=False).to_string(index=False))


def _load_random_simplex_weights(
    gate_path: Path,
    metadata_path: Path | None,
) -> dict[str, float] | None:
    candidates = [gate_path.parent / "metadata.json"]
    if metadata_path is not None:
        candidates.append(metadata_path)
    candidates.append(Path("results/path2_gate_pairwise/metadata.json"))

    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            meta = json.load(f)
        if "random_simplex_weights" in meta:
            return weights_to_dict(
                np.asarray(meta["random_simplex_weights"], dtype=np.float32)
            )
    return None


def downstream_smoke(
    lib: RegimeLibrary,
    gate_path: Path,
    out_dir: Path,
    seeds: list[int],
    *,
    include_all_static: bool = False,
    random_simplex_metadata: Path | None = None,
    volume_mode: str = "replay",
    query_stats_mode: str = "full",
    top_k: int = 5,
) -> None:
    learned = load_gate(gate_path)
    static = default_weight_vectors()
    random_simplex = _load_random_simplex_weights(gate_path, random_simplex_metadata)
    jan = load_cleaned("2024-01")
    jun = load_cleaned("2024-06")
    trips = pd.concat([jan, jun], ignore_index=True)
    blocks = split_into_blocks(build_demand_profile(trips))

    rows = []
    for scen_name, (month, hour_range, day_type) in PATH2_SCENARIOS.items():
        blk = _pick_block(blocks, month, hour_range, day_type)
        if blk is None:
            continue
        bid = blk["block_id"].iloc[0]
        q_series = blk["request_count"].values.astype(np.float64)
        qrec = lib[bid] if bid in lib.records else None
        q_events = qrec.events if qrec is not None else []
        replay_trips = _get_replay_trips(trips, bid)
        horizon = 4 * 3600.0
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / 4.0 * 0.15), 50)
        learned_weights = (
            learned(qrec.demand_series, qrec.events, bid, qrec)
            if qrec is not None else learned(q_series, q_events, bid)
        )
        methods = {
            "hand_tuned": None,
            "distributional": weights_to_dict(static["distributional"]),
            "learned_gate": learned_weights,
        }
        if include_all_static:
            methods["no_temporal"] = weights_to_dict(static["no_temporal"])
            methods["uniform"] = weights_to_dict(static["uniform"])
        if random_simplex is not None:
            methods["random_simplex"] = random_simplex

        for seed in seeds:
            replay_kpi = _run_one(ReplayDemandStream(replay_trips), fleet, horizon, seed)
            replay_kpi.update({
                "scenario": scen_name,
                "method": "replay",
                "seed": seed,
                "volume_mode": "replay",
                "query_stats_mode": "replay",
                "expected_synthetic_requests": float(n_trips),
                "replay_requests": int(n_trips),
                "top_k": int(top_k),
            })
            rows.append(replay_kpi)

        for method_name, weight_spec in methods.items():
            scores = []
            for cid in lib.records:
                if cid == bid:
                    continue
                rec = lib[cid]
                s = compute_similarity(q_series, q_events, rec, weights=weight_spec, q_block_id=bid)
                scores.append((cid, s))
            scores.sort(key=lambda x: x[1], reverse=True)
            matched = scores[:top_k]
            prior = build_calibrated_prior(
                [lib[cid] for cid, _ in matched],
                [score for _, score in matched],
                _query_series_for_stats(q_series, query_stats_mode),
            )
            for seed in seeds:
                rng = np.random.default_rng(seed)
                pr, expected_total = _apply_volume_mode(
                    prior, volume_mode, horizon, n_trips, q_series
                )
                stream = CalibratedDemandStream(pr, horizon, rng=rng)
                repo = AnticipatoryReposition(pr, lookahead_minutes=5.0, max_move_fraction=0.50)
                kpi = _run_one(stream, fleet, horizon, seed, repo)
                kpi.update({
                    "scenario": scen_name,
                    "method": method_name,
                    "seed": seed,
                    "volume_mode": volume_mode,
                    "query_stats_mode": query_stats_mode,
                    "expected_synthetic_requests": expected_total,
                    "replay_requests": int(n_trips),
                    "top_k": int(top_k),
                })
                rows.append(kpi)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gate_downstream_smoke.csv", index=False)
    replay = df[df["method"] == "replay"].groupby("scenario")["mean_wait_s"].mean()
    print("\nDownstream smoke summary:")
    for method in [m for m in df["method"].unique() if m != "replay"]:
        waits = df[df["method"] == method].groupby("scenario")["mean_wait_s"].mean()
        common = replay.index.intersection(waits.index)
        imp = ((replay.loc[common] - waits.loc[common]) / replay.loc[common] * 100).mean()
        vols = df[df["method"] == method].groupby("scenario")["total_requests"].mean()
        replay_vols = df[df["method"] == "replay"].groupby("scenario")["total_requests"].mean()
        vol_ratio = (vols.loc[common] / replay_vols.loc[common]).mean()
        print(
            f"  {method:16s}: wait={waits.loc[common].mean():.1f}s  "
            f"vs replay={imp:+.1f}%  request_ratio={vol_ratio:.2f}x"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate learned similarity gate")
    ap.add_argument("--gate", default="results/path2_gate/similarity_gate.pt")
    ap.add_argument("--out-dir", default="results/path2_gate")
    ap.add_argument("--downstream-smoke", action="store_true")
    ap.add_argument("--seeds", default="42", help="Comma-separated seeds for downstream smoke")
    ap.add_argument("--include-all-static", action="store_true")
    ap.add_argument("--top-k", type=int, default=5, help="Number of retrieved regimes to blend")
    ap.add_argument(
        "--volume-mode",
        choices=["replay", "none", "prefix30"],
        default="replay",
        help=(
            "Synthetic demand scaling for downstream smoke: replay is oracle "
            "shape-isolation, none is raw calibrated volume, prefix30 uses only "
            "the first 30 minutes of the query block."
        ),
    )
    ap.add_argument(
        "--query-stats-mode",
        choices=["full", "prefix30", "none"],
        default="full",
        help=(
            "Which query demand statistics the calibrator may use: full is the "
            "historical/oracle diagnostic, prefix30 is causal after 30 minutes, "
            "and none uses only retrieved regimes."
        ),
    )
    ap.add_argument(
        "--random-simplex-metadata",
        default="results/path2_gate_pairwise/metadata.json",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = RegimeLibrary()
    lib.load()

    gate_path = Path(args.gate)
    retrieval_eval(lib, gate_path, out_dir)
    if args.downstream_smoke:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        downstream_smoke(
            lib,
            gate_path,
            out_dir,
            seeds,
            include_all_static=args.include_all_static,
            random_simplex_metadata=Path(args.random_simplex_metadata),
            volume_mode=args.volume_mode,
            query_stats_mode=args.query_stats_mode,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
