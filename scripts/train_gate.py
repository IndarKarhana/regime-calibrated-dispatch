#!/usr/bin/env python3
"""Train the Path 2 learned similarity-weight gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regime.learned_weights import (  # noqa: E402
    build_gate_dataset,
    default_weight_vectors,
    evaluate_dataset,
    export_weight_summary,
    leakage_safe_split,
    model_weight_matrix,
    optimize_global_weights_random,
    save_gate,
    train_gate,
)
from src.regime.store import RegimeLibrary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train learned similarity weights")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--loss", choices=["topk", "pairwise"], default="topk")
    ap.add_argument("--random-simplex-trials", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/path2_gate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lib = RegimeLibrary()
    lib.load()
    split = leakage_safe_split(lib)
    print(f"Regime library: {len(lib)} records")
    print(f"Train blocks:   {len(split.train_ids)}")
    print(f"Holdout blocks: {len(split.holdout_ids)}")
    print(f"Holdout groups: {split.holdout_groups}")

    train_ds = build_gate_dataset(
        lib,
        split.train_ids,
        split.train_ids,
        split,
        normalize_from=split.train_ids,
    )
    holdout_ds = build_gate_dataset(
        lib,
        split.holdout_ids,
        split.train_ids,
        split,
        normalize_from=split.train_ids,
    )

    model, history = train_gate(
        train_ds,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        seed=args.seed,
        top_k=args.top_k,
        loss_name=args.loss,
    )
    save_gate(model, train_ds, out_dir, history)

    learned_train = model_weight_matrix(model, train_ds)
    learned_holdout = model_weight_matrix(model, holdout_ds)
    random_w, random_trials = optimize_global_weights_random(
        train_ds,
        n_trials=args.random_simplex_trials,
        seed=args.seed,
        objective="spearman",
    )
    random_trials.to_csv(out_dir / "random_simplex_trials.csv", index=False)

    train_weights = default_weight_vectors()
    train_weights["learned_gate"] = learned_train
    train_weights["random_simplex"] = random_w
    holdout_weights = default_weight_vectors()
    holdout_weights["learned_gate"] = learned_holdout
    holdout_weights["random_simplex"] = random_w

    train_eval = evaluate_dataset(train_ds, train_weights)
    train_eval["split"] = "train"
    holdout_eval = evaluate_dataset(holdout_ds, holdout_weights)
    holdout_eval["split"] = "holdout"
    eval_df = pd.concat([train_eval, holdout_eval], ignore_index=True)
    eval_df.to_csv(out_dir / "gate_retrieval_eval.csv", index=False)

    export_weight_summary(holdout_ds, learned_holdout, out_dir / "learned_weights_holdout.json")

    summary = eval_df[eval_df["query_id"] == "__MEAN__"].copy()
    summary_path = out_dir / "gate_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "epochs": args.epochs,
        "lr": args.lr,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "loss": args.loss,
        "seed": args.seed,
        "train_blocks": len(split.train_ids),
        "holdout_blocks": len(split.holdout_ids),
        "holdout_groups": split.holdout_groups,
        "feature_dim": int(train_ds.query_features.shape[1]),
        "random_simplex_trials": args.random_simplex_trials,
        "random_simplex_weights": random_w.tolist(),
        "final_history": history[-1] if history else {},
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSummary:")
    cols = ["split", "method", "spearman_rho", "top5_vs_median_improvement_pct"]
    summary_view = summary[cols].sort_values(
        ["split", "spearman_rho"], ascending=[True, False]
    )
    print(summary_view.to_string(index=False))
    print(f"\nSaved checkpoint and CSVs to {out_dir}")


if __name__ == "__main__":
    main()
