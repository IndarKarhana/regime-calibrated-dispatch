#!/usr/bin/env python3
"""Train a Phase 4 spatially regularized similarity gate.

This gate uses the Phase 3 theory result directly: instead of optimizing only
rank correlation with temporal demand profiles, it ranks candidate regimes by a
composite operational target containing demand L1, spatial transport mismatch,
and stylized queue-regret.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regime.learned_weights import (  # noqa: E402
    GateDataset,
    build_gate_dataset,
    default_weight_vectors,
    evaluate_dataset,
    export_weight_summary,
    leakage_safe_split,
    model_weight_matrix,
    save_gate,
    train_gate,
)
from src.regime.store import RegimeLibrary  # noqa: E402
from src.theory.spatial_metrics import (  # noqa: E402
    build_zone_index,
    greedy_transport_distance,
    scale_counts_to_total,
    travel_cost_matrix_s,
)
from src.theory.wait_bounds import queue_wait_regret_bound  # noqa: E402


def _parse_hidden(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() in {"", "none", "linear"}:
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _safe_norm(arr: np.ndarray) -> np.ndarray:
    scale = float(np.median(arr[arr > 0])) if np.any(arr > 0) else 1.0
    return arr / max(scale, 1e-9)


def build_operational_targets(
    dataset: GateDataset,
    library: RegimeLibrary,
    counts_by_record: dict[str, np.ndarray],
    cost_s: np.ndarray,
    *,
    demand_weight: float,
    spatial_weight: float,
    queue_weight: float,
    horizon_hours: float,
    fleet_ratio: float,
    min_fleet: int,
    service_rate_per_hour: float,
    rho_max: float,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Build per-query candidate targets used by the Phase 4 gate loss."""
    targets: list[np.ndarray] = []
    rows = []

    for qid, cids in zip(dataset.query_ids, dataset.candidate_ids):
        true_counts = counts_by_record[qid]
        total = max(float(true_counts.sum()), 1e-9)
        true_rate = true_counts / horizon_hours
        fleet = max(int(float(true_rate.sum()) * fleet_ratio), min_fleet)

        demand_l1 = []
        spatial_proxy = []
        queue_gap = []
        queue_bound = []

        for cid in cids:
            est_counts = scale_counts_to_total(counts_by_record[cid], total)
            est_rate = est_counts / horizon_hours

            demand = float(np.abs(est_counts - true_counts).sum() / total)
            spatial = greedy_transport_distance(cost_s, true_counts, est_counts)
            q = queue_wait_regret_bound(
                true_rate,
                est_rate,
                fleet,
                service_rate=service_rate_per_hour,
                rho_max=rho_max,
            )
            gap = q.wait_gap_s if np.isfinite(q.wait_gap_s) else q.bound_s

            demand_l1.append(demand)
            spatial_proxy.append(spatial)
            queue_gap.append(gap)
            queue_bound.append(q.bound_s)

        demand_arr = np.asarray(demand_l1, dtype=np.float32)
        spatial_arr = np.asarray(spatial_proxy, dtype=np.float32)
        queue_arr = np.asarray(queue_gap, dtype=np.float32)
        queue_bound_arr = np.asarray(queue_bound, dtype=np.float32)

        combined = (
            demand_weight * _safe_norm(demand_arr)
            + spatial_weight * _safe_norm(spatial_arr)
            + queue_weight * _safe_norm(queue_arr)
        ).astype(np.float32)
        targets.append(combined)

        for cid, d, s, g, b, target in zip(
            cids,
            demand_arr,
            spatial_arr,
            queue_arr,
            queue_bound_arr,
            combined,
        ):
            rows.append({
                "query_id": qid,
                "candidate_id": cid,
                "demand_l1_norm": float(d),
                "spatial_proxy_s": float(s),
                "queue_gap_s": float(g),
                "queue_bound_s": float(b),
                "operational_target": float(target),
            })

    return targets, pd.DataFrame(rows)


def evaluate_operational_dataset(
    dataset: GateDataset,
    target_errors: list[np.ndarray],
    method_weights: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Evaluate ranking quality against the operational target."""
    rows = []
    for method, weights in method_weights.items():
        rhos = []
        top5 = []
        medians = []
        for i, qid in enumerate(dataset.query_ids):
            w = weights[i] if weights.ndim == 2 else weights
            scores = dataset.component_matrices[i] @ w.astype(np.float32)
            errs = target_errors[i]
            rho, p = pd.Series(scores).corr(pd.Series(-errs), method="spearman"), np.nan
            if not np.isfinite(rho):
                rho = 0.0
            order = np.argsort(-scores)
            top_val = float(np.mean(errs[order[:5]]))
            med_val = float(np.median(errs))
            rhos.append(float(rho))
            top5.append(top_val)
            medians.append(med_val)
            rows.append({
                "method": method,
                "query_id": qid,
                "operational_spearman": float(rho),
                "spearman_p": p,
                "top5_operational_target": top_val,
                "median_operational_target": med_val,
                "top5_vs_median_improvement_pct": (
                    1.0 - top_val / max(med_val, 1e-9)
                ) * 100.0,
            })
        rows.append({
            "method": method,
            "query_id": "__MEAN__",
            "operational_spearman": float(np.mean(rhos)) if rhos else 0.0,
            "spearman_p": np.nan,
            "top5_operational_target": float(np.mean(top5)) if top5 else 0.0,
            "median_operational_target": float(np.mean(medians)) if medians else 0.0,
            "top5_vs_median_improvement_pct": (
                1.0 - float(np.mean(top5)) / max(float(np.mean(medians)), 1e-9)
            ) * 100.0 if top5 else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--loss", choices=["topk", "pairwise"], default="pairwise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden",
        default="64,32",
        help="Comma-separated hidden-layer widths. Use 'linear' or empty for no hidden layer.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--out-dir", default="results/path2_gate_spatial")
    parser.add_argument("--demand-weight", type=float, default=0.25)
    parser.add_argument("--spatial-weight", type=float, default=0.55)
    parser.add_argument("--queue-weight", type=float, default=0.20)
    parser.add_argument("--max-zones", type=int, default=8)
    parser.add_argument("--h3-res", type=int, default=8)
    parser.add_argument("--horizon-hours", type=float, default=4.0)
    parser.add_argument("--fleet-ratio", type=float, default=0.50)
    parser.add_argument("--min-fleet", type=int, default=50)
    parser.add_argument("--service-rate-per-hour", type=float, default=20.0)
    parser.add_argument("--rho-max", type=float, default=0.90)
    args = parser.parse_args()
    hidden = _parse_hidden(args.hidden)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    library = RegimeLibrary()
    library.load()
    split = leakage_safe_split(library)
    train_ds = build_gate_dataset(
        library,
        split.train_ids,
        split.train_ids,
        split,
        normalize_from=split.train_ids,
    )
    holdout_ds = build_gate_dataset(
        library,
        split.holdout_ids,
        split.train_ids,
        split,
        normalize_from=split.train_ids,
    )
    zone_ids, counts_by_record = build_zone_index(
        library,
        h3_res=args.h3_res,
        max_zones=args.max_zones,
    )
    cost_s = travel_cost_matrix_s(zone_ids)

    print(f"Regime library: {len(library)} records")
    print(f"Train blocks:   {len(split.train_ids)}")
    print(f"Holdout blocks: {len(split.holdout_ids)}")
    print(f"Zones:          {len(zone_ids)}")
    print("Building operational targets...")
    target_kwargs = {
        "demand_weight": args.demand_weight,
        "spatial_weight": args.spatial_weight,
        "queue_weight": args.queue_weight,
        "horizon_hours": args.horizon_hours,
        "fleet_ratio": args.fleet_ratio,
        "min_fleet": args.min_fleet,
        "service_rate_per_hour": args.service_rate_per_hour,
        "rho_max": args.rho_max,
    }
    train_targets, train_target_df = build_operational_targets(
        train_ds, library, counts_by_record, cost_s, **target_kwargs
    )
    holdout_targets, holdout_target_df = build_operational_targets(
        holdout_ds, library, counts_by_record, cost_s, **target_kwargs
    )
    train_target_df["split"] = "train"
    holdout_target_df["split"] = "holdout"
    pd.concat([train_target_df, holdout_target_df], ignore_index=True).to_csv(
        out_dir / "operational_targets.csv",
        index=False,
    )

    model, history = train_gate(
        train_ds,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        seed=args.seed,
        top_k=args.top_k,
        loss_name=args.loss,
        target_errors=train_targets,
        hidden=hidden,
        dropout=args.dropout,
    )
    save_gate(model, train_ds, out_dir, history)

    learned_train = model_weight_matrix(model, train_ds)
    learned_holdout = model_weight_matrix(model, holdout_ds)
    train_weights = default_weight_vectors()
    train_weights["spatial_gate"] = learned_train
    holdout_weights = default_weight_vectors()
    holdout_weights["spatial_gate"] = learned_holdout

    demand_train = evaluate_dataset(train_ds, train_weights)
    demand_train["split"] = "train"
    demand_holdout = evaluate_dataset(holdout_ds, holdout_weights)
    demand_holdout["split"] = "holdout"
    demand_eval = pd.concat([demand_train, demand_holdout], ignore_index=True)
    demand_eval.to_csv(out_dir / "gate_retrieval_eval.csv", index=False)

    op_train = evaluate_operational_dataset(train_ds, train_targets, train_weights)
    op_train["split"] = "train"
    op_holdout = evaluate_operational_dataset(holdout_ds, holdout_targets, holdout_weights)
    op_holdout["split"] = "holdout"
    op_eval = pd.concat([op_train, op_holdout], ignore_index=True)
    op_eval.to_csv(out_dir / "gate_operational_eval.csv", index=False)

    export_weight_summary(holdout_ds, learned_holdout, out_dir / "learned_weights_holdout.json")

    meta = {
        **vars(args),
        "train_blocks": len(split.train_ids),
        "holdout_blocks": len(split.holdout_ids),
        "holdout_groups": split.holdout_groups,
        "feature_dim": int(train_ds.query_features.shape[1]),
        "hidden_layers": list(hidden),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "zones": zone_ids,
        "final_history": history[-1] if history else {},
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    demand_summary = demand_eval[demand_eval["query_id"] == "__MEAN__"].copy()
    op_summary = op_eval[op_eval["query_id"] == "__MEAN__"].copy()
    demand_summary.to_csv(out_dir / "gate_summary.csv", index=False)
    op_summary.to_csv(out_dir / "gate_operational_summary.csv", index=False)

    print("\nDemand retrieval summary:")
    print(demand_summary[[
        "split",
        "method",
        "spearman_rho",
        "top5_vs_median_improvement_pct",
    ]].sort_values(["split", "spearman_rho"], ascending=[True, False]).to_string(index=False))
    print("\nOperational target summary:")
    print(op_summary[[
        "split",
        "method",
        "operational_spearman",
        "top5_vs_median_improvement_pct",
    ]].sort_values(
        ["split", "operational_spearman"],
        ascending=[True, False],
    ).to_string(index=False))
    print(f"\nSaved Phase 4 spatial gate artifacts to {out_dir}")


if __name__ == "__main__":
    main()
