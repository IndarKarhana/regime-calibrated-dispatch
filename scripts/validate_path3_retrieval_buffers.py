#!/usr/bin/env python3
"""Validate Path 3 retrieval ambiguity buffers on leakage-held-out blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_path3_retrieval_buffers import (  # noqa: E402
    _normalize_scores,
    _top_matches,
)
from src.policies.anticipatory import _HexGrid  # noqa: E402
from src.policies.external_baselines import _record_zone_fractions  # noqa: E402
from src.regime.learned_weights import block_group, leakage_safe_split, load_gate  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402


def validate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    library = RegimeLibrary()
    library.load()
    split = leakage_safe_split(library)
    gate = load_gate(args.gate)
    grid = _HexGrid(args.policy_h3_res)
    with open(args.buffers_json) as f:
        buffers = json.load(f)

    rows = []
    for qid in split.holdout_ids:
        qrec = library[qid]
        weights = gate(qrec.demand_series, qrec.events, qid, qrec)
        matches = _top_matches(library, qid, weights, split.train_ids, args.top_k)
        if not matches:
            continue
        true_share = _record_zone_fractions(qrec, grid)
        match_weights = _normalize_scores([score for _, score in matches])
        pred_share = np.zeros(grid.n, dtype=float)
        for w, (cid, _) in zip(match_weights, matches):
            pred_share += w * _record_zone_fractions(library[cid], grid)
        pred_share /= max(float(pred_share.sum()), 1e-12)
        positive = np.maximum(true_share - pred_share, 0.0)
        row = {
            "query_id": qid,
            "group": "|".join(block_group(qid)),
            "l1_residual": float(np.abs(true_share - pred_share).sum()),
            "positive_mass": float(positive.sum()),
            "max_zone_positive": float(positive.max()) if positive.size else 0.0,
        }
        for alpha, vals in buffers["alphas"].items():
            zone_buffer = np.asarray(vals["zone_buffer"], dtype=float)
            row[f"covered_positive_mass_{alpha}"] = (
                row["positive_mass"] <= float(vals["positive_mass_quantile"]) + 1e-12
            )
            row[f"covered_l1_{alpha}"] = (
                row["l1_residual"] <= float(vals["l1_residual_quantile"]) + 1e-12
            )
            row[f"covered_max_zone_{alpha}"] = (
                row["max_zone_positive"] <= float(vals["max_zone_positive_quantile"]) + 1e-12
            )
            row[f"covered_zonewise_{alpha}"] = bool(
                zone_buffer.size == positive.size
                and np.all(positive <= zone_buffer + 1e-12)
            )
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for alpha in buffers["alphas"]:
        summary_rows.append({
            "alpha": alpha,
            "n": int(len(detail)),
            "positive_mass_coverage": float(detail[f"covered_positive_mass_{alpha}"].mean()),
            "l1_coverage": float(detail[f"covered_l1_{alpha}"].mean()),
            "max_zone_coverage": float(detail[f"covered_max_zone_{alpha}"].mean()),
            "zonewise_coverage": float(detail[f"covered_zonewise_{alpha}"].mean()),
            "mean_positive_mass": float(detail["positive_mass"].mean()),
            "p90_positive_mass": float(detail["positive_mass"].quantile(0.90)),
            "mean_l1_residual": float(detail["l1_residual"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="results/path2_gate_spatial/similarity_gate.pt")
    parser.add_argument(
        "--buffers-json",
        default="results/path3_retrieval_buffers_top5/buffers.json",
    )
    parser.add_argument("--out-dir", default="results/path3_retrieval_buffers_top5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--policy-h3-res", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = validate(args)
    detail.to_csv(out_dir / "heldout_buffer_validation.csv", index=False)
    summary.to_csv(out_dir / "heldout_buffer_validation_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved heldout buffer validation to {out_dir}")


if __name__ == "__main__":
    main()
