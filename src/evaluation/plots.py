"""Generate comparison plots from ablation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config


def plot_ablation_results(csv_path: str | Path | None = None) -> None:
    cfg = get_config()
    out_dir = Path(cfg["evaluation"]["output_dir"])
    if csv_path is None:
        csv_path = out_dir / "ablation_results.csv"
    df = pd.read_csv(csv_path)

    metrics = ["mean_wait_s", "completion_rate", "throughput_trips_per_hour",
               "mean_idle_s_per_driver", "mean_pickup_dist_m"]

    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df, x="config", y=metric, errorbar="sd", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    plot_ablation_results()
