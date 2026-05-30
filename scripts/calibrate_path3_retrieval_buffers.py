#!/usr/bin/env python3
"""Calibrate retrieval residual buffers for Path 3 robust repositioning.

The calibration set is built from the leakage-safe training pool: each block is
treated as a query, its top-k retrieved regimes are selected from the remaining
training blocks, and the positive spatial residual between the true pickup
share and the retrieved mixture share is recorded by policy zone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.policies.anticipatory import _HexGrid  # noqa: E402
from src.policies.external_baselines import _record_zone_fractions  # noqa: E402
from src.regime.learned_weights import leakage_safe_split, load_gate  # noqa: E402
from src.regime.similarity import compute_similarity  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402


def _top_matches(library: RegimeLibrary, qid: str, weights, candidates: list[str], top_k: int):
    qrec = library[qid]
    rows = []
    for cid in candidates:
        if cid == qid:
            continue
        score = compute_similarity(
            qrec.demand_series,
            qrec.events,
            library[cid],
            weights=weights,
            q_block_id=qid,
        )
        rows.append((cid, score))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:top_k]


def _normalize_scores(scores: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        return np.ones(len(arr), dtype=float) / max(len(arr), 1)
    return arr / total


def _conformal_quantile(
    values: np.ndarray,
    alpha: float,
    axis: int | None = None,
) -> np.ndarray | float:
    """Return the split-conformal ceiling order statistic.

    This intentionally does not interpolate. For n calibration scores it returns
    the ceil((n + 1) * alpha)-th sorted value, using one-based indexing. If that
    order statistic is beyond the sample, the finite-sample conformal convention
    is +infinity.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("inf")
    if axis is None:
        flat = np.sort(arr.reshape(-1))
        idx = int(np.ceil((flat.size + 1) * float(alpha))) - 1
        if idx >= flat.size:
            return float("inf")
        return float(flat[max(idx, 0)])

    sorted_arr = np.sort(arr, axis=axis)
    n = sorted_arr.shape[axis]
    idx = int(np.ceil((n + 1) * float(alpha))) - 1
    if idx >= n:
        shape = list(sorted_arr.shape)
        del shape[axis]
        return np.full(shape, np.inf, dtype=float)
    return np.take(sorted_arr, max(idx, 0), axis=axis)


def calibrate(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    library = RegimeLibrary()
    library.load()
    split = leakage_safe_split(library)
    gate = load_gate(args.gate)
    grid = _HexGrid(args.policy_h3_res)

    rows = []
    residuals = []
    for qid in split.train_ids:
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
        residuals.append(positive)
        rows.append({
            "query_id": qid,
            "top_k": args.top_k,
            "l1_residual": float(np.abs(true_share - pred_share).sum()),
            "positive_mass": float(positive.sum()),
            "max_zone_positive": float(positive.max()) if positive.size else 0.0,
        })

    detail = pd.DataFrame(rows)
    residual_matrix = (
        np.vstack(residuals)
        if residuals
        else np.zeros((1, grid.n), dtype=float)
    )

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    buffers = {}
    for alpha in alphas:
        q = float(np.clip(alpha, 0.50, 0.99))
        zone_buffer = np.quantile(residual_matrix, q, axis=0)
        buffers[f"{q:.2f}"] = {
            "zone_buffer": zone_buffer.tolist(),
            "zone_buffer_mass": float(zone_buffer.sum()),
            "l1_residual_quantile": float(
                _conformal_quantile(detail["l1_residual"].to_numpy(), q)
            ),
            "positive_mass_quantile": float(
                _conformal_quantile(detail["positive_mass"].to_numpy(), q)
            ),
            "max_zone_positive_quantile": float(
                _conformal_quantile(detail["max_zone_positive"].to_numpy(), q)
            ),
        }

    payload = {
        "method": "spatial_gate_positive_residual",
        "gate": args.gate,
        "top_k": args.top_k,
        "policy_h3_res": args.policy_h3_res,
        "train_blocks": len(split.train_ids),
        "heldout_blocks_not_used": len(split.holdout_ids),
        "zone_count": grid.n,
        "scalar_quantile_rule": "ceil((n+1)*alpha) split-conformal order statistic",
        "zone_buffer_rule": (
            "coordinatewise empirical quantile; implementation diagnostic, "
            "not simultaneous conformal coverage"
        ),
        "alphas": buffers,
    }
    return detail, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="results/path2_gate_spatial/similarity_gate.pt")
    parser.add_argument("--out-dir", default="results/path3_retrieval_buffers")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--policy-h3-res", type=int, default=None)
    parser.add_argument("--alphas", default="0.70,0.80,0.90,0.95")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail, payload = calibrate(args)
    detail.to_csv(out_dir / "retrieval_residuals.csv", index=False)
    with open(out_dir / "buffers.json", "w") as f:
        json.dump(payload, f, indent=2)

    summary_rows = []
    for alpha, values in payload["alphas"].items():
        summary_rows.append({
            "alpha": alpha,
            **{k: v for k, v in values.items() if k != "zone_buffer"},
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "buffer_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved Path 3 retrieval buffers to {out_dir}")


if __name__ == "__main__":
    main()
