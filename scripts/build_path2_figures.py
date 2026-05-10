#!/usr/bin/env python3
"""Build Path 2 figures for the Transportation Science draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.25,
}

COLORS = {
    "gate": "#188038",
    "baseline": "#348ABD",
    "replay": "#8B8B8B",
    "oracle": "#7B3294",
    "warn": "#E24A33",
}

EXTERNAL_BASELINE_DIR = "path2_external_baselines_gpr_mf050_10seed"


def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> None:
    fig.savefig(out_dir / f"{name}.pdf", dpi=dpi)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out_dir / f'{name}.pdf'}")


def fig_method_diagram(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    boxes = [
        (0.2, 2.4, 1.7, 0.8, "Historical\nregimes"),
        (0.2, 0.8, 1.7, 0.8, "Query\nblock"),
        (2.7, 1.6, 1.9, 0.9, "Learned\nsimilarity gate"),
        (5.3, 1.6, 1.8, 0.9, "Calibrated\nspatial prior"),
        (7.8, 1.6, 1.8, 0.9, "Share-target\nLP"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#F4F7FB", edgecolor="#333333")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold")

    arrows = [
        ((1.9, 2.8), (2.7, 2.15)),
        ((1.9, 1.2), (2.7, 2.0)),
        ((4.6, 2.05), (5.3, 2.05)),
        ((7.1, 2.05), (7.8, 2.05)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5})

    ax.text(
        5.0,
        0.25,
        "Operational target: demand error + pickup spatial mismatch + queue shortage risk",
        ha="center",
        va="center",
        style="italic",
    )
    _save(fig, out_dir, "fig_path2_method", dpi)


def fig_gate_results(results_dir: Path, out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(results_dir / "path2_gate_spatial_10seed/analysis/method_summary.csv")
    order = ["learned_gate", "hand_tuned", "distributional", "random_simplex", "replay"]
    labels = ["Spatial gate", "Hand-tuned", "Distributional", "Random simplex", "Replay"]
    vals = [float(df[df["method"] == m]["mean_wait_s"].iloc[0]) for m in order]
    colors = [
        COLORS["gate"],
        COLORS["baseline"],
        COLORS["baseline"],
        COLORS["baseline"],
        COLORS["replay"],
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(order))
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean wait (s)")
    ax.set_title("Learned spatial gate improves downstream wait")
    for xi, val in zip(x, vals):
        ax.text(xi, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.18)
    _save(fig, out_dir, "fig_path2_gate_results", dpi)


def fig_external_baselines(results_dir: Path, out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(
        results_dir / EXTERNAL_BASELINE_DIR / "external_baselines_summary.csv"
    )
    order = [
        "oracle_mpc",
        "scenario_chance_mpc",
        "share_lp_hand_tuned",
        "spatial_gate_share_lp",
        "gpr_chance_mpc_lite",
        "wen2017_rebalancing",
        "batch_replay",
    ]
    labels = [
        "Oracle",
        "Scenario\nchance",
        "Share LP",
        "Spatial\nshare LP",
        "GPR\nchance",
        "Wen",
        "Replay",
    ]
    vals = [float(df[df["method"] == m]["mean_wait_s"].iloc[0]) for m in order]
    colors = [
        COLORS["oracle"],
        COLORS["gate"],
        COLORS["gate"],
        COLORS["gate"],
        COLORS["baseline"],
        COLORS["baseline"],
        COLORS["replay"],
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(order))
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean wait (s)")
    ax.set_title("External baselines on replay demand")
    for xi, val in zip(x, vals):
        ax.text(xi, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.18)
    _save(fig, out_dir, "fig_path2_external_baselines", dpi)


def fig_theory_verification(results_dir: Path, out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(results_dir / "path2_theory_spatial_gate_10seed/wait_bound_verification.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.scatter(
        df["spatial_w1_s"],
        df["downstream_wait_s"],
        c=df["demand_l1_norm"],
        cmap="viridis",
        s=36,
        alpha=0.85,
    )
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("Demand L1 error")
    ax.set_xlabel("Pickup spatial mismatch $W_1$ (s)")
    ax.set_ylabel("Downstream wait (s)")
    ax.set_title("Spatial mismatch is an operational diagnostic")
    _save(fig, out_dir, "fig_path2_theory_diagnostic", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plt.rcParams.update(STYLE)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_method_diagram(out_dir, args.dpi)
    fig_gate_results(results_dir, out_dir, args.dpi)
    fig_external_baselines(results_dir, out_dir, args.dpi)
    fig_theory_verification(results_dir, out_dir, args.dpi)


if __name__ == "__main__":
    main()
