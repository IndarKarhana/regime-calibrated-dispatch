#!/usr/bin/env python3
"""Verify Phase 3 demand-to-wait theory candidates on the regime library.

The script evaluates four lenses:
  1. Queueing wait-regret certificate (main theorem candidate).
  2. Directional under-forecast shortage index.
  3. Spatial Wasserstein pickup-mismatch distance.
  4. Retrieval-to-demand oracle inequality diagnostics.

Outputs CSV summaries and a wait-bound verification figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regime.learned_weights import (  # noqa: E402
    PATH2_SCENARIOS,
    default_weight_vectors,
    load_gate,
)
from src.regime.similarity import compute_similarity, weights_to_dict  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402
from src.theory.spatial_metrics import (  # noqa: E402
    build_zone_index,
    travel_cost_matrix_s,
)
from src.theory.wait_bounds import (  # noqa: E402
    earth_movers_distance,
    queue_wait_regret_bound,
    shortage_index,
)


def _scenario_ids(library: RegimeLibrary) -> dict[str, str]:
    ids = {}
    for scen, (month, hour_range, day_type) in PATH2_SCENARIOS.items():
        for bid in sorted(library.records):
            if not bid.startswith(month) or hour_range not in bid:
                continue
            ts = pd.Timestamp(bid.rsplit("_", 1)[0])
            if ts.month == 1 and ts.day == 1:
                dtype = "holiday"
            elif ts.dayofweek >= 5:
                dtype = "weekend"
            else:
                dtype = "weekday"
            if dtype == day_type:
                ids[scen] = bid
                break
    return ids


def _top_matches(
    library: RegimeLibrary,
    qid: str,
    weight_spec,
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    qrec = library[qid]
    scores = []
    for cid, record in library.records.items():
        if cid == qid:
            continue
        score = compute_similarity(
            qrec.demand_series,
            qrec.events,
            record,
            weights=weight_spec,
            q_block_id=qid,
        )
        scores.append((cid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def _mixture_counts(
    matches: list[tuple[str, float]],
    counts_by_record: dict[str, np.ndarray],
    target_total: float,
) -> np.ndarray:
    if not matches:
        return np.zeros_like(next(iter(counts_by_record.values())))
    weights = np.array([max(score, 0.0) for _, score in matches], dtype=float)
    if weights.sum() <= 1e-12:
        weights[:] = 1.0
    weights /= weights.sum()
    est = np.zeros_like(counts_by_record[matches[0][0]], dtype=float)
    for (bid, _), weight in zip(matches, weights):
        est += weight * counts_by_record[bid]
    if est.sum() > 1e-12:
        est *= target_total / est.sum()
    return est


def _method_specs(library: RegimeLibrary, gate_path: Path | None) -> dict[str, object]:
    vectors = default_weight_vectors()
    specs: dict[str, object] = {
        "hand_tuned": None,
        "distributional": weights_to_dict(vectors["distributional"]),
        "uniform": weights_to_dict(vectors["uniform"]),
        "no_temporal": weights_to_dict(vectors["no_temporal"]),
    }
    metadata = Path("results/path2_gate_pairwise/metadata.json")
    if metadata.exists():
        with open(metadata) as f:
            meta = json.load(f)
        if "random_simplex_weights" in meta:
            specs["random_simplex"] = weights_to_dict(meta["random_simplex_weights"])
    if gate_path and gate_path.exists():
        learned = load_gate(gate_path)
        # Filled per query in the main loop because the gate is query-dependent.
        specs["learned_gate"] = learned
    return specs


def _downstream_means(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"scenario", "method", "mean_wait_s"}.issubset(df.columns):
        return None
    means = df.groupby(["scenario", "method"], as_index=False)["mean_wait_s"].mean()
    replay = means[means["method"] == "replay"][["scenario", "mean_wait_s"]]
    replay = replay.rename(columns={"mean_wait_s": "replay_wait_s"})
    out = means.merge(replay, on="scenario", how="left")
    out["downstream_improvement_pct"] = (
        (out["replay_wait_s"] - out["mean_wait_s"]) / out["replay_wait_s"] * 100.0
    )
    return out[out["method"] != "replay"].rename(columns={"mean_wait_s": "downstream_wait_s"})


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    library = RegimeLibrary()
    library.load()
    scenario_ids = _scenario_ids(library)
    zone_ids, counts_by_record = build_zone_index(
        library,
        h3_res=args.h3_res,
        max_zones=args.max_zones,
    )
    cost_s = travel_cost_matrix_s(zone_ids)
    specs = _method_specs(library, Path(args.gate) if args.gate else None)

    rows = []
    horizon_h = args.horizon_hours
    for scenario, qid in scenario_ids.items():
        qrec = library[qid]
        true_counts = counts_by_record[qid]
        true_rate = true_counts / horizon_h
        if true_counts.sum() <= 0:
            continue
        fleet = max(int(true_rate.sum() * args.fleet_ratio), args.min_fleet)

        for method, spec in specs.items():
            weight_spec = spec
            if method == "learned_gate":
                weight_spec = spec(qrec.demand_series, qrec.events, qid, qrec)
            matches = _top_matches(library, qid, weight_spec, top_k=args.top_k)
            est_counts = _mixture_counts(matches, counts_by_record, true_counts.sum())
            est_rate = est_counts / horizon_h

            bound = queue_wait_regret_bound(
                true_rate,
                est_rate,
                fleet,
                service_rate=args.service_rate_per_hour,
                rho_max=args.rho_max,
            )
            spatial_w1 = earth_movers_distance(cost_s, true_counts, est_counts)
            demand_l1_count = float(np.abs(est_counts - true_counts).sum())
            top_match = matches[0][0] if matches else ""

            rows.append({
                "scenario": scenario,
                "query_id": qid,
                "method": method,
                "top_match_id": top_match,
                "fleet": fleet,
                "zones": len(zone_ids),
                "true_total_count": float(true_counts.sum()),
                "demand_l1_count": demand_l1_count,
                "demand_l1_norm": bound.demand_l1_norm,
                "directional_shortage_norm": shortage_index(true_rate, est_rate),
                "spatial_w1_s": spatial_w1,
                "capacity_l1_norm": bound.capacity_l1_norm,
                "oracle_wait_s": bound.oracle_policy_wait_s,
                "estimated_policy_wait_s": bound.estimated_policy_wait_s,
                "queue_wait_gap_s": bound.wait_gap_s,
                "queue_bound_s": bound.bound_s,
                "queue_bound_direct_s": bound.bound_without_capacity_s,
                "queue_bound_slack_s": bound.slack_s,
                "queue_bound_tightness": bound.tightness,
                "max_utilization": bound.max_utilization,
            })

    detail = pd.DataFrame(rows)
    downstream = _downstream_means(Path(args.downstream))
    if downstream is not None:
        detail = detail.merge(
            downstream[[
                "scenario",
                "method",
                "downstream_wait_s",
                "replay_wait_s",
                "downstream_improvement_pct",
            ]],
            on=["scenario", "method"],
            how="left",
        )

    summary_rows = []
    for method, group in detail.groupby("method"):
        row = {
            "method": method,
            "n": len(group),
            "mean_demand_l1_norm": group["demand_l1_norm"].mean(),
            "mean_shortage_norm": group["directional_shortage_norm"].mean(),
            "mean_spatial_w1_s": group["spatial_w1_s"].mean(),
            "mean_queue_gap_s": group["queue_wait_gap_s"].mean(),
            "mean_queue_bound_s": group["queue_bound_s"].mean(),
            "bound_coverage": float(
                (group["queue_bound_s"] + 1e-9 >= group["queue_wait_gap_s"]).mean()
            ),
            "median_bound_tightness": (
                group["queue_bound_tightness"].replace([np.inf], np.nan).median()
            ),
        }
        if "downstream_wait_s" in group.columns and group["downstream_wait_s"].notna().any():
            row["mean_downstream_wait_s"] = group["downstream_wait_s"].mean()
            row["mean_downstream_improvement_pct"] = group["downstream_improvement_pct"].mean()
        summary_rows.append(row)

    candidates = [
        "demand_l1_norm",
        "directional_shortage_norm",
        "spatial_w1_s",
        "queue_wait_gap_s",
        "queue_bound_s",
    ]
    if "downstream_wait_s" in detail.columns:
        valid = detail.dropna(subset=["downstream_wait_s"])
        for metric in candidates:
            if len(valid) >= 4 and valid[metric].nunique() > 1:
                rho, p = stats.spearmanr(valid[metric], valid["downstream_wait_s"])
                summary_rows.append({
                    "method": f"CORR::{metric}_vs_downstream_wait",
                    "n": len(valid),
                    "mean_demand_l1_norm": np.nan,
                    "mean_shortage_norm": np.nan,
                    "mean_spatial_w1_s": np.nan,
                    "mean_queue_gap_s": rho,
                    "mean_queue_bound_s": p,
                    "bound_coverage": np.nan,
                    "median_bound_tightness": np.nan,
                })

    summary = pd.DataFrame(summary_rows)
    return detail, summary


def plot_verification(detail: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    methods = sorted(detail["method"].unique())
    cmap = plt.get_cmap("tab10")
    for i, method in enumerate(methods):
        g = detail[detail["method"] == method]
        ax.scatter(
            g["queue_bound_s"],
            g["queue_wait_gap_s"],
            s=35,
            alpha=0.75,
            color=cmap(i % 10),
            label=method,
        )
    lim = max(float(detail["queue_bound_s"].max()), float(detail["queue_wait_gap_s"].max()), 1.0)
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("theoretical bound (s)")
    ax.set_ylabel("stylized queue wait gap (s)")
    ax.set_title("Wait-regret bound verification")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/path2_theory")
    parser.add_argument("--gate", default="results/path2_gate_pairwise/similarity_gate.pt")
    parser.add_argument(
        "--downstream",
        default="results/path2_gate_pairwise/gate_downstream_smoke.csv",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-zones", type=int, default=8)
    parser.add_argument("--h3-res", type=int, default=8)
    parser.add_argument("--horizon-hours", type=float, default=4.0)
    parser.add_argument("--fleet-ratio", type=float, default=0.50)
    parser.add_argument("--min-fleet", type=int, default=50)
    parser.add_argument("--service-rate-per-hour", type=float, default=20.0)
    parser.add_argument("--rho-max", type=float, default=0.90)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = run(args)
    detail_path = out_dir / "wait_bound_verification.csv"
    summary_path = out_dir / "theory_candidate_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_verification(detail, Path("paper/figures/fig_wait_bound_verification.pdf"))

    print("\nPhase 3 theory verification")
    print(f"  rows: {len(detail)}")
    print(f"  detail: {detail_path}")
    print(f"  summary: {summary_path}")
    print("  figure: paper/figures/fig_wait_bound_verification.pdf")
    cols = [
        "method",
        "mean_demand_l1_norm",
        "mean_spatial_w1_s",
        "mean_queue_gap_s",
        "mean_queue_bound_s",
        "bound_coverage",
        "mean_downstream_wait_s",
    ]
    present = [c for c in cols if c in summary.columns]
    print(summary[~summary["method"].str.startswith("CORR::")][present].to_string(index=False))
    corr = summary[summary["method"].str.startswith("CORR::")]
    if len(corr):
        print("\nSpearman diagnostics vs downstream wait "
              "(rho in mean_queue_gap_s, p in mean_queue_bound_s):")
        print(
            corr[["method", "n", "mean_queue_gap_s", "mean_queue_bound_s"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
