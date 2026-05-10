#!/usr/bin/env python3
"""Sweep Phase 4 spatial-gate objective weights using cached target metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_spatial_gate import evaluate_operational_dataset  # noqa: E402
from src.regime.learned_weights import (  # noqa: E402
    GateDataset,
    build_gate_dataset,
    default_weight_vectors,
    evaluate_dataset,
    leakage_safe_split,
    model_weight_matrix,
    save_gate,
    train_gate,
)
from src.regime.store import RegimeLibrary  # noqa: E402


def _safe_norm(arr: np.ndarray) -> np.ndarray:
    scale = float(np.median(arr[arr > 0])) if np.any(arr > 0) else 1.0
    return arr / max(scale, 1e-9)


def _parse_grid(grid: str) -> list[tuple[str, float, float, float]]:
    specs = []
    for raw in grid.split(";"):
        if not raw.strip():
            continue
        name, vals = raw.split("=")
        d, s, q = [float(x) for x in vals.split(",")]
        total = max(d + s + q, 1e-12)
        specs.append((name.strip(), d / total, s / total, q / total))
    return specs


def _targets_from_table(
    dataset: GateDataset,
    table: pd.DataFrame,
    split: str,
    weights: tuple[float, float, float],
) -> list[np.ndarray]:
    d_w, s_w, q_w = weights
    split_table = table[table["split"] == split]
    lookup = split_table.set_index(["query_id", "candidate_id"])
    targets = []
    for qid, cids in zip(dataset.query_ids, dataset.candidate_ids):
        rows = lookup.loc[[(qid, cid) for cid in cids]]
        demand = rows["demand_l1_norm"].to_numpy(dtype=np.float32)
        spatial = rows["spatial_proxy_s"].to_numpy(dtype=np.float32)
        queue = rows["queue_gap_s"].to_numpy(dtype=np.float32)
        combined = (
            d_w * _safe_norm(demand)
            + s_w * _safe_norm(spatial)
            + q_w * _safe_norm(queue)
        ).astype(np.float32)
        targets.append(combined)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-table",
        default="results/path2_gate_spatial/operational_targets.csv",
    )
    parser.add_argument("--out-dir", default="results/path2_gate_sweep")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grid",
        default=(
            "current=0.25,0.55,0.20;"
            "spatial_heavy=0.15,0.70,0.15;"
            "no_queue=0.30,0.70,0.00;"
            "balanced=0.34,0.33,0.33;"
            "demand_heavy=0.50,0.35,0.15;"
            "queue_heavy=0.20,0.40,0.40"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_table = pd.read_csv(args.target_table)
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

    summary_rows = []
    all_demand_eval = []
    all_op_eval = []
    for name, demand_w, spatial_w, queue_w in _parse_grid(args.grid):
        print(
            f"\nTraining {name}: demand={demand_w:.2f}, "
            f"spatial={spatial_w:.2f}, queue={queue_w:.2f}"
        )
        train_targets = _targets_from_table(
            train_ds, target_table, "train", (demand_w, spatial_w, queue_w)
        )
        holdout_targets = _targets_from_table(
            holdout_ds, target_table, "holdout", (demand_w, spatial_w, queue_w)
        )
        model, history = train_gate(
            train_ds,
            epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            seed=args.seed,
            top_k=args.top_k,
            loss_name="pairwise",
            target_errors=train_targets,
        )
        run_dir = out_dir / name
        save_gate(model, train_ds, run_dir, history)

        learned_train = model_weight_matrix(model, train_ds)
        learned_holdout = model_weight_matrix(model, holdout_ds)
        train_weights = default_weight_vectors()
        train_weights["sweep_gate"] = learned_train
        holdout_weights = default_weight_vectors()
        holdout_weights["sweep_gate"] = learned_holdout

        demand_train = evaluate_dataset(train_ds, train_weights)
        demand_train["split"] = "train"
        demand_holdout = evaluate_dataset(holdout_ds, holdout_weights)
        demand_holdout["split"] = "holdout"
        demand_eval = pd.concat([demand_train, demand_holdout], ignore_index=True)
        demand_eval["config"] = name
        all_demand_eval.append(demand_eval)

        op_train = evaluate_operational_dataset(train_ds, train_targets, train_weights)
        op_train["split"] = "train"
        op_holdout = evaluate_operational_dataset(holdout_ds, holdout_targets, holdout_weights)
        op_holdout["split"] = "holdout"
        op_eval = pd.concat([op_train, op_holdout], ignore_index=True)
        op_eval["config"] = name
        all_op_eval.append(op_eval)

        holdout_demand_mean = demand_eval[
            (demand_eval["split"] == "holdout")
            & (demand_eval["method"] == "sweep_gate")
            & (demand_eval["query_id"] == "__MEAN__")
        ].iloc[0]
        holdout_op_mean = op_eval[
            (op_eval["split"] == "holdout")
            & (op_eval["method"] == "sweep_gate")
            & (op_eval["query_id"] == "__MEAN__")
        ].iloc[0]
        summary_rows.append({
            "config": name,
            "demand_weight": demand_w,
            "spatial_weight": spatial_w,
            "queue_weight": queue_w,
            "holdout_demand_spearman": float(holdout_demand_mean["spearman_rho"]),
            "holdout_demand_top5_improvement_pct": float(
                holdout_demand_mean["top5_vs_median_improvement_pct"]
            ),
            "holdout_operational_spearman": float(
                holdout_op_mean["operational_spearman"]
            ),
            "holdout_operational_top5_improvement_pct": float(
                holdout_op_mean["top5_vs_median_improvement_pct"]
            ),
            "final_loss": history[-1]["loss"] if history else np.nan,
        })
        with open(run_dir / "sweep_metadata.json", "w") as f:
            json.dump(summary_rows[-1], f, indent=2)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["holdout_operational_spearman", "holdout_demand_top5_improvement_pct"],
        ascending=[False, False],
    )
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    pd.concat(all_demand_eval, ignore_index=True).to_csv(
        out_dir / "sweep_demand_eval.csv", index=False
    )
    pd.concat(all_op_eval, ignore_index=True).to_csv(
        out_dir / "sweep_operational_eval.csv", index=False
    )
    print("\nSweep summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
