#!/usr/bin/env python3
"""Plot the Path 3 wait/movement frontier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("results/path3_figures")
OUT_PDF = Path("paper/figures/path3_movement_frontier.pdf")
OUT_PNG = OUT_DIR / "path3_movement_frontier.png"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    weighted = pd.read_csv("results/path3_spatial_robust_frontier.csv")
    weighted["family"] = "Weighted DCR"
    weighted["label"] = weighted["move_mult"].map(lambda x: f"m={x:g}")

    constrained = pd.read_csv(
        "results/path3_constrained_dcr_buffer70_sweep/constrained_dcr_sweep_summary.csv"
    )
    constrained["family"] = "Constrained DCR"
    constrained["label"] = constrained["shortage_reduction"].map(lambda x: f"r={x:.2f}")

    baseline = pd.read_csv("results/path3_buffer70_m4_10seed/external_baselines_summary.csv")
    baseline_labels = {
        "share_lp_hand_tuned": "Share-LP",
        "scenario_chance_mpc": "Chance-MPC",
        "gpr_chance_mpc_lite": "GPR-lite",
        "wen2017_rebalancing": "Wen",
        "oracle_mpc": "Oracle",
        "batch_replay": "Replay",
        "spatial_gate_robust_cvar": "Selected DCR",
    }
    baseline = baseline[baseline["method"].isin(baseline_labels)].copy()
    baseline["label"] = baseline["method"].map(baseline_labels)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        weighted["reposition_dist_m_per_trip"],
        weighted["mean_wait_s"],
        marker="o",
        color="#1f77b4",
        label="Weighted DCR sweep",
    )
    weighted_offsets = {
        "m=0.5": (11, 10),
        "m=1": (-18, -12),
        "m=2": (5, -13),
        "m=4": (5, -12),
        "m=8": (5, 5),
    }
    for _, row in weighted.iterrows():
        ax.annotate(
            row["label"],
            (row["reposition_dist_m_per_trip"], row["mean_wait_s"]),
            textcoords="offset points",
            xytext=weighted_offsets.get(row["label"], (4, 4)),
            fontsize=6.5,
        )

    ax.plot(
        constrained["reposition_dist_m_per_trip"],
        constrained["mean_wait_s"],
        marker="s",
        color="#d62728",
        label="Constrained DCR sweep",
    )
    constrained_offsets = {
        "r=0.35": (-22, 7),
        "r=0.40": (-18, -12),
        "r=0.45": (5, -11),
        "r=0.50": (5, 5),
        "r=0.55": (5, 5),
    }
    for _, row in constrained.iterrows():
        ax.annotate(
            row["label"],
            (row["reposition_dist_m_per_trip"], row["mean_wait_s"]),
            textcoords="offset points",
            xytext=constrained_offsets.get(row["label"], (4, -10)),
            fontsize=6.5,
        )

    marker_map = {
        "Selected DCR": ("*", "#2ca02c", 150),
        "Share-LP": ("^", "#111111", 60),
        "Chance-MPC": ("v", "#555555", 60),
        "GPR-lite": ("D", "#9467bd", 55),
        "Wen": ("P", "#8c564b", 60),
        "Oracle": ("X", "#ff7f0e", 70),
        "Replay": ("o", "#7f7f7f", 45),
    }
    label_text = {
        "Selected DCR": "Selected DCR",
        "Share-LP": "Share LP",
        "Chance-MPC": "Chance MPC",
        "GPR-lite": "GPR-lite",
        "Wen": "Wen",
        "Oracle": "Oracle",
        "Replay": "Replay",
    }
    baseline_offsets = {
        "Selected DCR": (-50, -1),
        "Share-LP": (18, 15),
        "Chance-MPC": (24, -14),
        "GPR-lite": (-46, 13),
        "Wen": (6, 6),
        "Oracle": (-18, -20),
        "Replay": (5, 5),
    }
    baseline_ha = {
        "Selected DCR": "right",
        "GPR-lite": "right",
        "Oracle": "right",
    }
    for _, row in baseline.iterrows():
        label = row["label"]
        marker, color, size = marker_map[label]
        ax.scatter(
            row["reposition_dist_m_per_trip"],
            row["mean_wait_s"],
            marker=marker,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.annotate(
            label_text[label],
            (row["reposition_dist_m_per_trip"], row["mean_wait_s"]),
            textcoords="offset points",
            xytext=baseline_offsets[label],
            ha=baseline_ha.get(label, "left"),
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "alpha": 0.55}
            if label in {"Share-LP", "Chance-MPC", "Oracle", "Selected DCR", "GPR-lite"}
            else None,
        )

    ax.set_xlabel("Empty repositioning distance (m/trip)")
    ax.set_ylabel("Mean passenger wait (s)")
    ax.set_title("DCR movement/wait frontier")
    ax.set_xlim(-10, 390)
    ax.set_ylim(88, 125)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
