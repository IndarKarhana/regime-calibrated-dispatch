#!/usr/bin/env python3
"""Build LaTeX tables for the Path 2 review-response hardening pass."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


OUT = Path("paper/path2_review_response_tables.tex")


def _fmt(x: float, digits: int = 1) -> str:
    return f"{float(x):.{digits}f}"


def _gate_table() -> str:
    df = pd.read_csv("results/path2_review_response/gate_volume_topk_summary.csv")
    keep = [
        ("oracle_full_replay_top5_10seed", "Oracle replay volume, full-block stats", "learned_gate"),
        ("deploy_prefixstats_top5_10seed", "Prefix volume/statistics, top-5", "learned_gate"),
        ("raw_noquerystats_top5_10seed", "Raw retrieved prior, top-5", "learned_gate"),
        ("deploy_prefixstats_top3_3seed", "Prefix volume/statistics, top-3", "hand_tuned"),
        ("deploy_prefixstats_top10_3seed", "Prefix volume/statistics, top-10", "learned_gate"),
    ]
    rows = []
    for run, label, headline in keep:
        sub = df[df["run"] == run].copy()
        best = sub.sort_values("mean_wait_s").iloc[0]
        selected = sub[sub["method"] == headline].iloc[0]
        rows.append(
            f"{label} & {int(selected['seeds'])} & {int(selected['top_k'])} & "
            f"{selected['method'].replace('_', '-')}"
            f" & {_fmt(selected['mean_wait_s'])} & "
            f"{_fmt(selected['mean_improvement_pct'])}\\% & "
            f"{_fmt(selected['request_ratio'], 2)}x & "
            f"{best['method'].replace('_', '-')} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\caption{Deployability and retrieved-regime sensitivity for the learned gate. Prefix variants use only the first 30 minutes of the query block for scale/statistics; the oracle row is retained only as a shape-isolation diagnostic.}
\label{tab:review-gate-deployability}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrlrrll}
\toprule
Diagnostic & Seeds & $k$ & Reported method & Wait (s) & Improv. & Vol. ratio & Best \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _theory_table() -> str:
    s = pd.read_csv("results/path2_theory_calibration/theory_calibration_summary.csv").iloc[0]
    rows = [
        ("Zone count $Z$", f"{int(s['Z'])}"),
        ("Mean service rate $\\mu$", f"{_fmt(s['mu_per_h_mean'], 2)} trips/hour"),
        ("$\\mu$ range", f"{_fmt(s['mu_per_h_min'], 2)}--{_fmt(s['mu_per_h_max'], 2)} trips/hour"),
        ("Median queue scale $L_Q$", f"{_fmt(s['L_Q_seconds_median'], 0)} s"),
        ("95th percentile $L_Q$", f"{_fmt(s['L_Q_seconds_p95'], 0)} s"),
        ("Mean rounding residual", f"{_fmt(s['rounding_residual_l1_frac_mean'], 3)}"),
        ("Allocator basis-flip rate", f"{_fmt(s['allocator_basis_flip_rate'], 3)}"),
        ("95th percentile allocator ratio", f"{_fmt(s['allocator_ratio_p95'], 3)} drivers/(trip/hour)"),
        ("Median spatial $\\beta$", f"{_fmt(s['beta_wait_per_w1_median'], 3)}"),
        ("95th percentile spatial $\\beta$", f"{_fmt(s['beta_wait_per_w1_p95'], 2)}"),
    ]
    body = "\n".join(f"{name} & {value} \\\\" for name, value in rows)
    return r"""
\begin{table}[t]
\centering
\caption{Empirical calibration of surrogate-theory constants. Values are local diagnostics for the eight scenario blocks, not universal constants.}
\label{tab:theory-constant-calibration}
\footnotesize
\begin{tabular}{lr}
\toprule
Quantity & Estimate \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _runtime_table() -> str:
    df = pd.read_csv("results/path2_external_baselines_gpr_mf050_10seed/external_baselines_summary.csv")
    order = [
        "batch_replay",
        "wen2017_rebalancing",
        "gpr_chance_mpc_lite",
        "scenario_chance_mpc",
        "share_lp_hand_tuned",
        "spatial_gate_share_lp",
        "oracle_mpc",
    ]
    rows = []
    for method in order:
        if method not in set(df["method"]):
            continue
        r = df[df["method"] == method].iloc[0]
        rows.append(
            f"{method.replace('_', '-')} & {_fmt(r['mean_wait_s'])} & "
            f"{_fmt(r['completion_rate'], 3)} & {_fmt(r['mean_pickup_dist_m'], 0)} & "
            f"{_fmt(r['reposition_dist_m_per_trip'], 0)} & "
            f"{_fmt(r['zone_wait_gini'], 3)} & "
            f"{_fmt(r['zone_wait_p90_p10_gap_s'], 0)} & "
            f"{_fmt(r['lp_solve_time_p95_ms'], 2)} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\caption{Computational, empty-mileage, and zone-fairness footprint on replay demand, ten-seed external-baseline pass. Repos. m/trip is true repositioning distance per completed trip; zone gap is the p90--p10 gap of mean wait across served pickup zones.}
\label{tab:runtime-footprint}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrr}
\toprule
Method & Wait & Compl. & Pickup m & Repos. m/trip & Zone Gini & Zone gap & LP p95 ms \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _mean_se(df: pd.DataFrame, method: str, col: str = "mean_wait_s") -> tuple[float, float, int]:
    vals = df[df["method"] == method][col].astype(float)
    n = int(vals.shape[0])
    return float(vals.mean()), float(vals.std(ddof=1) / (n**0.5)), n


def _gate_capacity_table() -> str:
    meta = json.loads(Path("results/path2_gate_spatial/metadata.json").read_text())
    feature_dim = int(meta["feature_dim"])
    hidden1 = 64
    hidden2 = 32
    output_dim = 6
    params = (
        feature_dim * hidden1
        + hidden1
        + hidden1 * hidden2
        + hidden2
        + hidden2 * output_dim
        + output_dim
    )

    demand = pd.read_csv("results/path2_gate_spatial/gate_summary.csv")
    operational = pd.read_csv("results/path2_gate_spatial/gate_operational_summary.csv")
    gate_wait = pd.read_csv("results/path2_gate_spatial_10seed/gate_downstream_smoke.csv")
    external_wait = pd.read_csv(
        "results/path2_external_baselines_gpr_mf050_10seed/external_baselines.csv"
    )

    def summary_value(
        frame: pd.DataFrame, method: str, split: str, col: str, digits: int = 3
    ) -> str:
        row = frame[(frame["method"] == method) & (frame["split"] == split)].iloc[0]
        return _fmt(row[col], digits)

    learned_mean, learned_se, learned_n = _mean_se(gate_wait, "learned_gate")
    dist_mean, dist_se, _ = _mean_se(gate_wait, "distributional")
    share_mean, share_se, share_n = _mean_se(external_wait, "share_lp_hand_tuned")
    chance_mean, chance_se, _ = _mean_se(external_wait, "scenario_chance_mpc")
    gpr_mean, gpr_se, _ = _mean_se(external_wait, "gpr_chance_mpc_lite")
    wen_mean, wen_se, _ = _mean_se(external_wait, "wen2017_rebalancing")

    rows = [
        ("Gate capacity", "Trainable parameters", f"{params:,}"),
        (
            "Gate data",
            "Train / held-out blocks",
            f"{int(meta['train_blocks'])} / {int(meta['holdout_blocks'])}",
        ),
        (
            "Demand retrieval",
            "Spatial-gate Spearman, train / held-out",
            f"{summary_value(demand, 'spatial_gate', 'train', 'spearman_rho')} / "
            f"{summary_value(demand, 'spatial_gate', 'holdout', 'spearman_rho')}",
        ),
        (
            "Demand retrieval",
            "Distributional Spearman, train / held-out",
            f"{summary_value(demand, 'distributional', 'train', 'spearman_rho')} / "
            f"{summary_value(demand, 'distributional', 'holdout', 'spearman_rho')}",
        ),
        (
            "Operational target",
            "Spatial-gate Spearman, train / held-out",
            f"{summary_value(operational, 'spatial_gate', 'train', 'operational_spearman')} / "
            f"{summary_value(operational, 'spatial_gate', 'holdout', 'operational_spearman')}",
        ),
        (
            "Operational target",
            "Distributional Spearman, train / held-out",
            f"{summary_value(operational, 'distributional', 'train', 'operational_spearman')} / "
            f"{summary_value(operational, 'distributional', 'holdout', 'operational_spearman')}",
        ),
        (
            "Gate wait",
            f"Learned / distributional mean $\\pm$ SE, n={learned_n}",
            f"{_fmt(learned_mean)} $\\pm$ {_fmt(learned_se)}s / "
            f"{_fmt(dist_mean)} $\\pm$ {_fmt(dist_se)}s",
        ),
        (
            "Replay LP wait",
            f"Chance-MPC / GPR-lite / share-LP / Wen mean $\\pm$ SE, n={share_n}",
            f"{_fmt(chance_mean)} $\\pm$ {_fmt(chance_se)}s / "
            f"{_fmt(gpr_mean)} $\\pm$ {_fmt(gpr_se)}s / "
            f"{_fmt(share_mean)} $\\pm$ {_fmt(share_se)}s / "
            f"{_fmt(wen_mean)} $\\pm$ {_fmt(wen_se)}s",
        ),
    ]
    body = "\n".join(f"{group} & {metric} & {value} \\\\" for group, metric, value in rows)
    return r"""
\begin{table}[t]
\centering
\caption{Gate capacity, train/held-out diagnostics, and uncertainty checks. The learned gate has more parameters than training blocks, so we report train/held-out rank metrics and standard errors rather than treating the MLP result as automatically robust.}
\label{tab:gate-capacity-uncertainty}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lp{0.43\linewidth}p{0.28\linewidth}}
\toprule
Concern & Diagnostic & Value \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _topk_sweep_table() -> str:
    df = pd.read_csv("results/path2_review_response/topk_replay_3seed_summary.csv")
    rows = []
    for top_k in sorted(df["top_k"].unique()):
        sub = df[df["top_k"] == top_k].copy()
        best = sub.sort_values("mean_wait_s").iloc[0]
        learned = sub[sub["method"] == "learned_gate"].iloc[0]
        hand = sub[sub["method"] == "hand_tuned"].iloc[0]
        dist = sub[sub["method"] == "distributional"].iloc[0]
        rows.append(
            f"{int(top_k)} & {best['method'].replace('_', '-')} & "
            f"{_fmt(learned['mean_wait_s'])} & {_fmt(hand['mean_wait_s'])} & "
            f"{_fmt(dist['mean_wait_s'])} & "
            f"{_fmt(learned['improvement_vs_replay_pct'])}\\% \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\caption{Retrieved-regime count sensitivity under replay-volume-matched, full-statistic gate evaluation. This is a three-seed diagnostic, not the headline ten-seed result.}
\label{tab:topk-sweep-review}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
$k$ & Best method & Learned wait & Hand wait & Dist. wait & Learned improv. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _architecture_sweep_table() -> str:
    df = pd.read_csv("results/path2_review_response/gate_architecture_sweep_summary.csv")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{row['hidden_layers']} & {int(row['parameters'])} & "
            f"{_fmt(row['train_operational_spearman'], 3)} & "
            f"{_fmt(row['holdout_operational_spearman'], 3)} & "
            f"{_fmt(row['holdout_demand_spearman'], 3)} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\caption{Gate capacity sweep over smaller architectures. All rows use the same operational target and 60 training epochs, so this is a proxy overfitting diagnostic rather than the final downstream selection run.}
\label{tab:gate-architecture-sweep}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Hidden layers & Params & Train op. $\rho$ & Holdout op. $\rho$ & Holdout demand $\rho$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _beta_stratification_table() -> str:
    df = pd.read_csv("results/path2_theory_calibration/spatial_beta_calibration.csv")
    df = df.copy()
    df["load_bin"] = pd.qcut(
        df["max_utilization"],
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )
    grouped = (
        df.groupby("load_bin", observed=True)
        .agg(
            n=("beta_wait_per_w1", "size"),
            rho_min=("max_utilization", "min"),
            rho_max=("max_utilization", "max"),
            beta_median=("beta_wait_per_w1", "median"),
            beta_p95=("beta_wait_per_w1", lambda x: x.quantile(0.95)),
            positive_gap=("queue_wait_gap_s", lambda x: (x > 1e-9).mean()),
        )
        .reset_index()
    )
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            f"{row['load_bin']} & {int(row['n'])} & "
            f"{_fmt(row['rho_min'], 3)}--{_fmt(row['rho_max'], 3)} & "
            f"{_fmt(row['beta_median'], 3)} & {_fmt(row['beta_p95'], 2)} & "
            f"{_fmt(100.0 * row['positive_gap'], 0)}\\% \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\caption{Spatial Lipschitz diagnostic stratified by induced utilization. The median coefficient and positive-gap frequency grow with load, while upper-tail outliers remain possible even at lower load; $\beta$ is therefore local and regime-dependent.}
\label{tab:beta-load-stratification}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
Load bin & $n$ & Utilization range & Median $\beta$ & $\beta_{95}$ & Positive gap \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _policy_zone_resolution_table() -> str:
    df = pd.read_csv("results/path2_review_response/policy_zone_resolution_summary.csv")
    rows = []
    for setting in df["setting"].drop_duplicates():
        sub = df[df["setting"] == setting]
        chance = sub[sub["method"] == "scenario_chance_mpc"].iloc[0]
        share = sub[sub["method"] == "share_lp_hand_tuned"].iloc[0]
        wen = sub[sub["method"] == "wen2017_rebalancing"].iloc[0]
        rows.append(
            f"{int(chance['zones'])} & "
            f"{_fmt(chance['wait'])} / {_fmt(chance['improvement'])}\\% & "
            f"{_fmt(share['wait'])} / {_fmt(share['improvement'])}\\% & "
            f"{_fmt(wen['wait'])} / {_fmt(wen['improvement'])}\\% & "
            f"{_fmt(chance['repos_m_trip'], 0)} & {_fmt(chance['zone_gini'], 3)} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\caption{Closed-loop controller zone-resolution diagnostic. The main controller uses 183 H3 zones; the 28-zone row is a three-seed coarser-grid sensitivity. Cells report mean wait / improvement versus replay.}
\label{tab:policy-zone-resolution}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
Zones & Chance MPC & Share LP & Wen & Chance repos. m/trip & Chance zone Gini \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _spatial_metric_proxy_table() -> str:
    s = pd.read_csv("results/path2_review_response/spatial_metric_proxy_summary.csv").iloc[0]
    rows = [
        ("Sampled query-candidate pairs", f"{int(s['sample_pairs'])}"),
        ("Zones in diagnostic", f"{int(s['zones'])}"),
        ("Spearman: greedy vs exact OT", _fmt(s["spearman_rho"], 3)),
        ("Pearson: greedy vs exact OT", _fmt(s["pearson_r"], 3)),
        ("Mean top-10 overlap", f"{_fmt(100.0 * s['mean_top10_overlap'], 0)}\\%"),
        ("Median top-10 overlap", f"{_fmt(100.0 * s['median_top10_overlap'], 0)}\\%"),
    ]
    body = "\n".join(f"{name} & {value} \\\\" for name, value in rows)
    return r"""
\begin{table}[t]
\centering
\caption{Alternative spatial-metric diagnostic. Exact OT and the greedy transport proxy are compared on sampled holdout/train retrieval pairs; high agreement reduces concern that the gate is driven by an idiosyncratic greedy proxy.}
\label{tab:spatial-metric-proxy}
\footnotesize
\begin{tabular}{lr}
\toprule
Diagnostic & Value \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _robustness_table() -> str:
    rows = []
    for mf in (0.15, 0.30, 0.50, 0.75):
        path = Path(f"results/path2_robustness/move_{mf:.2f}/external_baselines_summary.csv")
        if not path.exists():
            path = Path(f"results/path2_robustness/move_{mf}/external_baselines_summary.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        share = df[df["method"] == "share_lp_hand_tuned"].iloc[0]
        wen = df[df["method"] == "wen2017_rebalancing"].iloc[0]
        rows.append(
            f"Move fraction & {_fmt(mf, 2)} & "
            f"{_fmt(share['mean_wait_s'])}s / {_fmt(share['improvement_vs_replay_pct'])}\\% & "
            f"{_fmt(wen['improvement_vs_replay_pct'])}\\% & "
            f"{_fmt(share['lp_solve_time_p95_ms'], 2)}ms \\\\"
        )

    for fr in (0.12, 0.15, 0.18):
        path = Path(f"results/path2_robustness/fleet_{fr:.2f}/external_baselines_summary.csv")
        if not path.exists():
            path = Path(f"results/path2_robustness/fleet_{fr}/external_baselines_summary.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        share = df[df["method"] == "share_lp_hand_tuned"].iloc[0]
        wen = df[df["method"] == "wen2017_rebalancing"].iloc[0]
        rows.append(
            f"Fleet ratio & {_fmt(fr, 2)} & "
            f"{_fmt(share['mean_wait_s'])}s / {_fmt(share['improvement_vs_replay_pct'])}\\% & "
            f"{_fmt(wen['improvement_vs_replay_pct'])}\\% & "
            f"{_fmt(share['lp_solve_time_p95_ms'], 2)}ms \\\\"
        )

    for z in (6, 8, 12):
        path = Path(f"results/path2_robustness/zone_{z}/theory_calibration_summary.csv")
        if not path.exists():
            continue
        s = pd.read_csv(path).iloc[0]
        rows.append(
            f"Zone count & {z} & "
            f"$L_Q$={_fmt(s['L_Q_seconds_median'], 0)}s & "
            f"$\\beta_{{95}}$={_fmt(s['beta_wait_per_w1_p95'], 2)} & "
            f"flip={_fmt(s['allocator_basis_flip_rate'], 3)} \\\\"
        )

    return r"""
\begin{table}[t]
\centering
\caption{Path-2-specific robustness diagnostics. Share-LP rows report mean wait / improvement vs replay for the hand-tuned share-target LP; the comparator column gives Wen-style improvement under the same sweep. Zone-count rows report surrogate-theory sensitivity.}
\label{tab:path2-robustness}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrr}
\toprule
Dimension & Setting & Primary & Comparator & Footprint \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main() -> None:
    OUT.write_text(
        "% Auto-generated by scripts/build_path2_review_response_tables.py\n"
        + _gate_table()
        + "\n"
        + _theory_table()
        + "\n"
        + _runtime_table()
        + "\n"
        + _gate_capacity_table()
        + "\n"
        + _topk_sweep_table()
        + "\n"
        + _architecture_sweep_table()
        + "\n"
        + _beta_stratification_table()
        + "\n"
        + _policy_zone_resolution_table()
        + "\n"
        + _spatial_metric_proxy_table()
        + "\n"
        + _robustness_table(),
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
