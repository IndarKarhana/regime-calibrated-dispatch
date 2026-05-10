#!/usr/bin/env python3
"""Summarize downstream gate evaluation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_tests(df: pd.DataFrame, focal_method: str) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["scenario", "seed"],
        columns="method",
        values="mean_wait_s",
    )
    rows = []
    if focal_method not in wide:
        return pd.DataFrame(rows)

    for baseline in sorted(c for c in wide.columns if c not in {focal_method, "replay"}):
        pair = wide[[focal_method, baseline]].dropna()
        if pair.empty:
            continue
        diff = pair[baseline] - pair[focal_method]
        if np.allclose(diff, 0.0):
            p_value = 1.0
            statistic = 0.0
        else:
            statistic, p_value = stats.wilcoxon(
                diff,
                alternative="greater",
                zero_method="wilcox",
            )
        rows.append({
            "focal_method": focal_method,
            "baseline": baseline,
            "n_pairs": int(len(pair)),
            "mean_delta_s": float(diff.mean()),
            "median_delta_s": float(diff.median()),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p_greater": float(p_value),
        })
    return pd.DataFrame(rows)


def summarize(path: Path, out_dir: Path, focal_method: str, replay_method: str) -> None:
    df = pd.read_csv(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_summary = df.groupby("method", as_index=False).agg(
        mean_wait_s=("mean_wait_s", "mean"),
        std_wait_s=("mean_wait_s", "std"),
        mean_completion_rate=("completion_rate", "mean"),
        n=("mean_wait_s", "size"),
    )
    replay_by_scenario = (
        df[df["method"] == replay_method]
        .groupby("scenario")["mean_wait_s"]
        .mean()
        .rename("replay_wait_s")
    )
    method_scenario = (
        df.groupby(["scenario", "method"], as_index=False)
        .agg(mean_wait_s=("mean_wait_s", "mean"))
        .merge(replay_by_scenario, on="scenario", how="left")
    )
    method_scenario["improvement_vs_replay_pct"] = (
        (method_scenario["replay_wait_s"] - method_scenario["mean_wait_s"])
        / method_scenario["replay_wait_s"]
        * 100.0
    )
    improvement = (
        method_scenario[method_scenario["method"] != replay_method]
        .groupby("method", as_index=False)["improvement_vs_replay_pct"]
        .mean()
    )
    method_summary = method_summary.merge(improvement, on="method", how="left")
    method_summary = method_summary.sort_values("mean_wait_s")

    tests = paired_tests(df, focal_method)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    method_scenario.to_csv(out_dir / "scenario_method_summary.csv", index=False)
    tests.to_csv(out_dir / "paired_tests.csv", index=False)

    print("\nMethod summary:")
    print(method_summary.to_string(index=False))
    if len(tests):
        print(f"\nPaired tests: lower wait for {focal_method}")
        print(tests.sort_values("wilcoxon_p_greater").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--focal-method", default="learned_gate")
    parser.add_argument("--replay-method", default="replay")
    args = parser.parse_args()

    out_dir = args.out_dir or args.csv.parent / "analysis"
    summarize(args.csv, out_dir, args.focal_method, args.replay_method)


if __name__ == "__main__":
    main()
