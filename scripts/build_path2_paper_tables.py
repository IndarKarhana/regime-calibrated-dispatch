#!/usr/bin/env python3
"""Build Path 2 paper table fragments from frozen result CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXTERNAL_BASELINE_DIR = "path2_external_baselines_gpr_mf050_10seed"


def _fmt(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def _completion(row: pd.Series) -> float:
    if "mean_completion_rate" in row.index:
        return float(row["mean_completion_rate"])
    return float(row["completion_rate"])


def _latex_table(
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    *,
    size: str | None = None,
) -> str:
    align = "l" + "r" * (len(columns) - 1)
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    header = " & ".join(columns) + r" \\"
    size_line = f"\\{size}\n" if size else ""
    return rf"""
\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
{size_line}\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{{align}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()


def _esc(text: str) -> str:
    return text.replace("_", "\\_")


def build_gate_table(results_dir: Path) -> tuple[str, dict]:
    df = pd.read_csv(results_dir / "path2_gate_spatial_10seed/analysis/method_summary.csv")
    order = [
        "learned_gate",
        "hand_tuned",
        "distributional",
        "random_simplex",
        "uniform",
        "replay",
    ]
    names = {
        "learned_gate": "Spatial learned gate",
        "hand_tuned": "Hand-tuned similarity",
        "distributional": "Distributional-only",
        "random_simplex": "Random simplex",
        "uniform": "Uniform weights",
        "replay": "Replay demand",
    }
    rows = []
    for method in order:
        row = df[df["method"] == method].iloc[0]
        rows.append([
            names[method],
            _fmt(row["mean_wait_s"], 1),
            _fmt(row["mean_completion_rate"], 3),
            _fmt(row["improvement_vs_replay_pct"], 1),
        ])
    tests = pd.read_csv(results_dir / "path2_gate_spatial_10seed/analysis/paired_tests.csv")
    summary = {
        "spatial_gate_wait_s": float(df[df["method"] == "learned_gate"]["mean_wait_s"].iloc[0]),
        "hand_tuned_wait_s": float(df[df["method"] == "hand_tuned"]["mean_wait_s"].iloc[0]),
        "distributional_wait_s": float(df[df["method"] == "distributional"]["mean_wait_s"].iloc[0]),
        "paired_tests": tests.to_dict(orient="records"),
    }
    return _latex_table(
        "Learned gate validation over 8 scenarios and 10 seeds.",
        "tab:gate-v2",
        ["Method", "Wait (s)", "Completion", "Vs. replay (\\%)"],
        rows,
    ), summary


def build_external_table(results_dir: Path) -> tuple[str, dict]:
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
    names = {
        "oracle_mpc": "Oracle MPC",
        "scenario_chance_mpc": "Scenario chance MPC",
        "share_lp_hand_tuned": "Share-target LP",
        "spatial_gate_share_lp": "Spatial-gate share LP",
        "gpr_chance_mpc_lite": "GPR chance MPC-lite",
        "wen2017_rebalancing": "Wen-style rebalancing",
        "batch_replay": "Batch replay",
    }
    rows = []
    for method in order:
        row = df[df["method"] == method].iloc[0]
        rows.append([
            names[method],
            _fmt(row["mean_wait_s"], 1),
            _fmt(row["completion_rate"], 3),
            _fmt(row["improvement_vs_replay_pct"], 1),
        ])
    summary = {
        "share_lp_wait_s": float(
            df[df["method"] == "share_lp_hand_tuned"]["mean_wait_s"].iloc[0]
        ),
        "scenario_chance_wait_s": float(
            df[df["method"] == "scenario_chance_mpc"]["mean_wait_s"].iloc[0]
        ),
        "wen_wait_s": float(
            df[df["method"] == "wen2017_rebalancing"]["mean_wait_s"].iloc[0]
        ),
        "gpr_chance_wait_s": float(
            df[df["method"] == "gpr_chance_mpc_lite"]["mean_wait_s"].iloc[0]
        ),
        "oracle_wait_s": float(df[df["method"] == "oracle_mpc"]["mean_wait_s"].iloc[0]),
    }
    return _latex_table(
        "External baselines re-run in the simulator on replay demand.",
        "tab:external-v2",
        ["Method", "Wait (s)", "Completion", "Vs. replay (\\%)"],
        rows,
    ), summary


def build_sweep_table(results_dir: Path) -> tuple[str, dict]:
    df = pd.read_csv(results_dir / "path2_gate_sweep/sweep_summary.csv")
    rows = []
    for _, row in df.head(6).iterrows():
        weights = (
            f"{row['demand_weight']:.2f}/"
            f"{row['spatial_weight']:.2f}/"
            f"{row['queue_weight']:.2f}"
        )
        rows.append([
            str(row["config"]).replace("_", "\\_"),
            weights,
            _fmt(row["holdout_demand_spearman"], 3),
            _fmt(row["holdout_operational_spearman"], 3),
        ])
    summary = {
        "best_proxy_config": str(df.iloc[0]["config"]),
        "best_proxy_operational_spearman": float(df.iloc[0]["holdout_operational_spearman"]),
    }
    return _latex_table(
        "Objective-weight sweep. Weights are demand/spatial/queue.",
        "tab:sweep-v2",
        ["Config", "Weights", "Demand $\\rho$", "Operational $\\rho$"],
        rows,
    ), summary


def build_supplement_tables(results_dir: Path) -> tuple[str, dict]:
    gate_scen = pd.read_csv(
        results_dir / "path2_gate_spatial_10seed/analysis/scenario_method_summary.csv"
    )
    ext_scen = pd.read_csv(
        results_dir
        / EXTERNAL_BASELINE_DIR
        / "analysis_share_lp/scenario_method_summary.csv"
    )
    dqn = pd.read_csv(results_dir / "path2_contextual_dqn_3seed/external_baselines_summary.csv")
    amod = pd.read_csv(results_dir / "path2_amod_3seed/amod_summary.csv")
    gate_pivot = gate_scen.pivot(index="scenario", columns="method", values="mean_wait_s")
    gate_rows = []
    for scenario in sorted(gate_pivot.index):
        row = gate_pivot.loc[scenario]
        gate_rows.append([
            _esc(scenario),
            _fmt(row["replay"], 1),
            _fmt(row["learned_gate"], 1),
            _fmt(row["hand_tuned"], 1),
            _fmt(row["distributional"], 1),
            _fmt(row["random_simplex"], 1),
        ])
    gate_scen_table = _latex_table(
        "Per-scenario learned-gate wait times over 10 seeds.",
        "tab:gate-scenario-v2",
        ["Scenario", "Replay", "Learned", "Hand", "Dist.", "Random"],
        gate_rows,
        size="footnotesize",
    )

    ext_pivot = ext_scen.pivot(index="scenario", columns="method", values="mean_wait_s")
    ext_rows = []
    for scenario in sorted(ext_pivot.index):
        row = ext_pivot.loc[scenario]
        ext_rows.append([
            _esc(scenario),
            _fmt(row["batch_replay"], 1),
            _fmt(row["wen2017_rebalancing"], 1),
            _fmt(row["scenario_chance_mpc"], 1),
            _fmt(row["share_lp_hand_tuned"], 1),
            _fmt(row["spatial_gate_share_lp"], 1),
            _fmt(row["oracle_mpc"], 1),
        ])
    ext_scen_table = _latex_table(
        "Per-scenario replay-demand external-baseline wait times over 10 seeds.",
        "tab:external-scenario-v2",
        ["Scenario", "Replay", "Wen", "Chance", "LP", "Spatial LP", "Oracle"],
        ext_rows,
        size="footnotesize",
    )

    dqn_row = dqn[dqn["method"] == "lin2018_contextual_dqn"].iloc[0]
    replay_row = dqn[dqn["method"] == "batch_replay"].iloc[0]
    amod_rows = []
    for method, name in [
        ("central_share_lp", "Central share LP"),
        ("charging_aware_share_lp", "Charging-aware share LP"),
    ]:
        row = amod[amod["method"] == method].iloc[0]
        amod_rows.append([
            name,
            _fmt(row["mean_wait_s"], 1),
            _fmt(row["completion_rate"], 3),
            _fmt(row["charge_violation_rate"], 1),
        ])
    dqn_table = _latex_table(
        "Time-boxed contextual-DQN baseline smoke.",
        "tab:dqn-v2",
        ["Method", "Wait (s)", "Completion", "Vs. replay (\\%)"],
        [
            [
                "Contextual-DQN warm start",
                _fmt(dqn_row["mean_wait_s"], 1),
                _fmt(_completion(dqn_row), 3),
                _fmt(dqn_row["improvement_vs_replay_pct"], 1),
            ],
            [
                "Batch replay",
                _fmt(replay_row["mean_wait_s"], 1),
                _fmt(_completion(replay_row), 3),
                "--",
            ],
        ],
    )
    amod_table = _latex_table(
        "Centralized AMoD charging smoke.",
        "tab:amod-v2",
        ["Method", "Wait (s)", "Completion", "Low-charge/call"],
        amod_rows,
    )
    summary = {
        "gate_scenarios": gate_pivot.reset_index().to_dict(orient="records"),
        "external_scenarios": ext_pivot.reset_index().to_dict(orient="records"),
        "dqn_wait_s": float(dqn_row["mean_wait_s"]),
        "dqn_improvement_pct": float(dqn_row["improvement_vs_replay_pct"]),
        "charging_wait_s": float(
            amod[amod["method"] == "charging_aware_share_lp"]["mean_wait_s"].iloc[0]
        ),
        "central_wait_s": float(amod[amod["method"] == "central_share_lp"]["mean_wait_s"].iloc[0]),
    }
    return (
        gate_scen_table
        + "\n\n"
        + ext_scen_table
        + "\n\n"
        + dqn_table
        + "\n\n"
        + amod_table
    ), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-tex", default="paper/path2_results_tables.tex")
    parser.add_argument("--out-appendix-tex", default="paper/path2_appendix_tables.tex")
    parser.add_argument("--out-json", default="paper/path2_results_summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    main_fragments = []
    summary: dict[str, object] = {}
    for builder in [
        build_gate_table,
        build_external_table,
        build_sweep_table,
    ]:
        fragment, stats = builder(results_dir)
        main_fragments.append(fragment)
        summary.update(stats)
    appendix_fragment, appendix_stats = build_supplement_tables(results_dir)
    summary.update(appendix_stats)

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        "% Auto-generated by scripts/build_path2_paper_tables.py\n\n"
        + "\n\n".join(main_fragments)
        + "\n",
        encoding="utf-8",
    )
    out_appendix_tex = Path(args.out_appendix_tex)
    out_appendix_tex.write_text(
        "% Auto-generated by scripts/build_path2_paper_tables.py\n\n"
        + appendix_fragment
        + "\n",
        encoding="utf-8",
    )
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_tex}")
    print(f"Wrote {out_appendix_tex}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
