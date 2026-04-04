#!/usr/bin/env python3
"""Formal statistical tests for paper-ready analysis.

Implements:
  1. Wilcoxon signed-rank tests (paired, per-scenario) with Bonferroni correction
  2. Friedman test (non-parametric ANOVA across scenarios)
  3. Nemenyi post-hoc test with critical difference diagram data
  4. Effect size (Cohen's d) per scenario
  5. Bootstrap confidence intervals for grand mean

Usage:
  python scripts/statistical_tests.py
  python scripts/statistical_tests.py --csv results/multiseed_robustness.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("results")


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load multi-seed robustness data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Configs: {sorted(df['config'].unique())}")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    return df


# ── 1. Wilcoxon signed-rank tests (paired) ──────────────────────────────────

def wilcoxon_paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test: cal+lp vs replay for each scenario.

    Uses paired samples (matched by seed) and applies Bonferroni correction
    for multiple comparisons.
    """
    print("\n" + "=" * 80)
    print("1. WILCOXON SIGNED-RANK TESTS (cal+lp vs replay, paired by seed)")
    print("=" * 80)

    scenarios = sorted(df["scenario"].unique())
    seeds = sorted(df["seed"].unique())
    n_tests = len(scenarios)

    rows = []
    for scen in scenarios:
        replay_waits = []
        callp_waits = []
        for seed in seeds:
            rw = df[(df["scenario"] == scen) & (df["config"] == "replay") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            cw = df[(df["scenario"] == scen) & (df["config"] == "cal+lp") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            if len(rw) == 1 and len(cw) == 1:
                replay_waits.append(rw[0])
                callp_waits.append(cw[0])

        replay_arr = np.array(replay_waits)
        callp_arr = np.array(callp_waits)
        diffs = replay_arr - callp_arr

        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

        # Cohen's d (paired)
        cohens_d = mean_diff / std_diff if std_diff > 0 else float("inf")

        # Relative improvement
        rel_imp = float(np.mean(diffs / replay_arr * 100))

        # Wilcoxon test
        if n >= 5 and np.any(diffs != 0):
            stat, p_raw = stats.wilcoxon(diffs, alternative="greater")
        elif n >= 3 and np.any(diffs != 0):
            # Exact test for small samples
            stat, p_raw = stats.wilcoxon(diffs, alternative="greater", method="exact")
        else:
            stat, p_raw = 0.0, 1.0

        p_bonf = min(p_raw * n_tests, 1.0)

        rows.append({
            "scenario": scen,
            "n_pairs": n,
            "mean_replay_s": float(np.mean(replay_arr)),
            "mean_callp_s": float(np.mean(callp_arr)),
            "mean_diff_s": mean_diff,
            "rel_improvement_pct": rel_imp,
            "cohens_d": cohens_d,
            "wilcoxon_stat": stat,
            "p_raw": p_raw,
            "p_bonferroni": p_bonf,
            "significant_005": p_bonf < 0.05,
            "significant_001": p_bonf < 0.01,
        })

        sig_marker = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "ns"
        print(f"  {scen:20s}: Δ={mean_diff:+6.1f}s ({rel_imp:+5.1f}%)  "
              f"d={cohens_d:5.2f}  p_raw={p_raw:.4g}  p_bonf={p_bonf:.4g}  {sig_marker}")

    result_df = pd.DataFrame(rows)
    n_sig = result_df["significant_005"].sum()
    print(f"\n  {n_sig}/{n_tests} scenarios significant at α=0.05 (Bonferroni-corrected)")
    return result_df


# ── 2. Wilcoxon: cal+lp vs cal_only ─────────────────────────────────────────

def wilcoxon_lp_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test: cal+lp vs cal_only for each scenario.
    Tests whether LP repositioning contributes significantly beyond calibration alone.
    """
    print("\n" + "=" * 80)
    print("2. WILCOXON: LP CONTRIBUTION (cal+lp vs cal_only)")
    print("=" * 80)

    if "cal_only" not in df["config"].values:
        print("  [SKIP] No cal_only config found")
        return pd.DataFrame()

    scenarios = sorted(df["scenario"].unique())
    seeds = sorted(df["seed"].unique())
    n_tests = len(scenarios)

    rows = []
    for scen in scenarios:
        cal_waits = []
        callp_waits = []
        for seed in seeds:
            cw = df[(df["scenario"] == scen) & (df["config"] == "cal_only") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            lw = df[(df["scenario"] == scen) & (df["config"] == "cal+lp") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            if len(cw) == 1 and len(lw) == 1:
                cal_waits.append(cw[0])
                callp_waits.append(lw[0])

        cal_arr = np.array(cal_waits)
        callp_arr = np.array(callp_waits)
        diffs = cal_arr - callp_arr

        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        cohens_d = mean_diff / std_diff if std_diff > 0 else float("inf")
        rel_imp = float(np.mean(diffs / cal_arr * 100))

        if n >= 3 and np.any(diffs != 0):
            stat, p_raw = stats.wilcoxon(diffs, alternative="greater",
                                          method="exact" if n < 5 else "auto")
        else:
            stat, p_raw = 0.0, 1.0

        p_bonf = min(p_raw * n_tests, 1.0)

        rows.append({
            "scenario": scen,
            "n_pairs": n,
            "mean_cal_s": float(np.mean(cal_arr)),
            "mean_callp_s": float(np.mean(callp_arr)),
            "mean_diff_s": mean_diff,
            "rel_improvement_pct": rel_imp,
            "cohens_d": cohens_d,
            "wilcoxon_stat": stat,
            "p_raw": p_raw,
            "p_bonferroni": p_bonf,
            "significant_005": p_bonf < 0.05,
        })

        sig_marker = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "ns"
        print(f"  {scen:20s}: Δ={mean_diff:+6.1f}s ({rel_imp:+5.1f}%)  "
              f"d={cohens_d:5.2f}  p_bonf={p_bonf:.4g}  {sig_marker}")

    return pd.DataFrame(rows)


# ── 3. Friedman test across all scenarios ────────────────────────────────────

def friedman_test(df: pd.DataFrame) -> dict:
    """Friedman test: non-parametric repeated-measures ANOVA.

    Tests whether at least one configuration is significantly different
    from the others, treating scenarios × seeds as blocks.
    """
    print("\n" + "=" * 80)
    print("3. FRIEDMAN TEST (non-parametric ANOVA across configs)")
    print("=" * 80)

    configs = ["replay", "cal_only", "cal+lp"]
    configs_present = [c for c in configs if c in df["config"].values]
    scenarios = sorted(df["scenario"].unique())
    seeds = sorted(df["seed"].unique())

    # Build matrix: each row is one block (scenario × seed),
    # each column is one config
    blocks = []
    for scen in scenarios:
        for seed in seeds:
            row = []
            valid = True
            for cfg in configs_present:
                val = df[(df["scenario"] == scen) & (df["config"] == cfg) &
                         (df["seed"] == seed)]["mean_wait_s"].values
                if len(val) == 1:
                    row.append(val[0])
                else:
                    valid = False
                    break
            if valid:
                blocks.append(row)

    blocks_arr = np.array(blocks)
    n_blocks = len(blocks_arr)
    k = len(configs_present)
    print(f"  Blocks (scenario × seed): {n_blocks}")
    print(f"  Configs: {configs_present}")

    if n_blocks < 3 or k < 2:
        print("  [SKIP] Insufficient data for Friedman test")
        return {}

    stat, p_value = stats.friedmanchisquare(*[blocks_arr[:, i] for i in range(k)])

    # Rank each block
    ranks = np.zeros_like(blocks_arr)
    for i in range(n_blocks):
        ranks[i] = stats.rankdata(blocks_arr[i])

    mean_ranks = ranks.mean(axis=0)

    print(f"\n  Friedman χ² = {stat:.4f}")
    print(f"  p-value = {p_value:.4e}")
    print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} at α=0.05")
    print(f"\n  Mean ranks (lower = better wait time):")
    for i, cfg in enumerate(configs_present):
        print(f"    {cfg:10s}: {mean_ranks[i]:.3f}")

    result = {
        "friedman_chi2": float(stat),
        "friedman_p": float(p_value),
        "n_blocks": n_blocks,
        "k_configs": k,
        "configs": configs_present,
        "mean_ranks": {cfg: float(mean_ranks[i]) for i, cfg in enumerate(configs_present)},
    }

    # Nemenyi post-hoc if significant
    if p_value < 0.05:
        result["nemenyi"] = nemenyi_posthoc(blocks_arr, configs_present)

    return result


def nemenyi_posthoc(blocks_arr: np.ndarray, config_names: list[str]) -> dict:
    """Nemenyi post-hoc test for pairwise comparisons after Friedman.

    Critical difference (CD) at α=0.05 using the q_α / √2 * √(k(k+1)/(6N)) formula.
    """
    print("\n  NEMENYI POST-HOC:")
    n_blocks, k = blocks_arr.shape

    # Rank per block
    ranks = np.zeros_like(blocks_arr)
    for i in range(n_blocks):
        ranks[i] = stats.rankdata(blocks_arr[i])
    mean_ranks = ranks.mean(axis=0)

    # Critical q-values for Nemenyi test at α=0.05
    # q_α values from Nemenyi table: k=2: 1.960, k=3: 2.343, k=4: 2.569
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
    q = q_alpha.get(k, 2.343)

    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    print(f"  Critical difference (CD) at α=0.05: {cd:.4f}")

    pairwise = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            sig = diff > cd
            pair_name = f"{config_names[i]} vs {config_names[j]}"
            pairwise[pair_name] = {
                "rank_diff": float(diff),
                "cd": float(cd),
                "significant": bool(sig),
            }
            sig_str = "SIG" if sig else "ns"
            print(f"    {pair_name:30s}: |Δrank| = {diff:.3f}  "
                  f"{'>' if sig else '<='} CD={cd:.3f}  [{sig_str}]")

    return {"cd": float(cd), "pairwise": pairwise, "mean_ranks": dict(zip(config_names, mean_ranks.tolist()))}


# ── 4. Bootstrap confidence intervals ───────────────────────────────────────

def bootstrap_grand_mean(df: pd.DataFrame, n_boot: int = 10000) -> dict:
    """Bootstrap 95% CI for the grand mean improvement (cal+lp vs replay).

    Resamples scenario-level means (accounts for correlation within scenarios).
    """
    print("\n" + "=" * 80)
    print("4. BOOTSTRAP CONFIDENCE INTERVALS (grand mean improvement)")
    print("=" * 80)

    scenarios = sorted(df["scenario"].unique())
    seeds = sorted(df["seed"].unique())

    # Compute per-scenario, per-seed improvements
    scenario_improvements = {}
    for scen in scenarios:
        imps = []
        for seed in seeds:
            rw = df[(df["scenario"] == scen) & (df["config"] == "replay") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            cw = df[(df["scenario"] == scen) & (df["config"] == "cal+lp") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            if len(rw) == 1 and len(cw) == 1:
                imps.append((rw[0] - cw[0]) / rw[0] * 100)
        scenario_improvements[scen] = imps

    # Grand mean: average across scenario means
    scen_means = [np.mean(v) for v in scenario_improvements.values() if len(v) > 0]
    grand_mean = np.mean(scen_means)

    # Bootstrap: resample scenarios with replacement
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        # Hierarchical bootstrap: resample scenarios, then resample seeds within each
        sampled_scens = rng.choice(len(scen_means), size=len(scen_means), replace=True)
        sampled_vals = []
        for si in sampled_scens:
            scen_key = list(scenario_improvements.keys())[si]
            scen_imps = scenario_improvements[scen_key]
            if len(scen_imps) > 0:
                boot_seeds = rng.choice(scen_imps, size=len(scen_imps), replace=True)
                sampled_vals.append(np.mean(boot_seeds))
        if sampled_vals:
            boot_means.append(np.mean(sampled_vals))

    boot_arr = np.array(boot_means)
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    boot_std = float(np.std(boot_arr))

    print(f"  Grand mean improvement: {grand_mean:.2f}%")
    print(f"  Bootstrap 95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%]")
    print(f"  Bootstrap SE: {boot_std:.2f}%")
    print(f"  Number of bootstrap samples: {n_boot}")

    return {
        "grand_mean_pct": grand_mean,
        "ci_lo_pct": ci_lo,
        "ci_hi_pct": ci_hi,
        "bootstrap_se": boot_std,
        "n_boot": n_boot,
        "n_scenarios": len(scen_means),
    }


# ── 5. Effect size summary ──────────────────────────────────────────────────

def effect_size_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cohen's d effect size for each scenario and overall.

    Interpretation: d < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large.
    """
    print("\n" + "=" * 80)
    print("5. EFFECT SIZE SUMMARY (Cohen's d)")
    print("=" * 80)

    scenarios = sorted(df["scenario"].unique())
    seeds = sorted(df["seed"].unique())

    rows = []
    all_diffs = []
    all_stds = []
    for scen in scenarios:
        replay_vals = []
        callp_vals = []
        for seed in seeds:
            rw = df[(df["scenario"] == scen) & (df["config"] == "replay") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            cw = df[(df["scenario"] == scen) & (df["config"] == "cal+lp") &
                    (df["seed"] == seed)]["mean_wait_s"].values
            if len(rw) == 1 and len(cw) == 1:
                replay_vals.append(rw[0])
                callp_vals.append(cw[0])

        r_arr = np.array(replay_vals)
        c_arr = np.array(callp_vals)
        diffs = r_arr - c_arr

        pooled_std = np.sqrt((np.var(r_arr, ddof=1) + np.var(c_arr, ddof=1)) / 2)
        d = float(np.mean(diffs) / pooled_std) if pooled_std > 0 else float("inf")

        all_diffs.extend(diffs.tolist())
        all_stds.append(pooled_std)

        category = ("large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5
                     else "small" if abs(d) > 0.2 else "negligible")

        rows.append({"scenario": scen, "cohens_d": d, "category": category,
                      "mean_diff_s": float(np.mean(diffs))})
        print(f"  {scen:20s}: d = {d:6.2f}  [{category}]  Δ = {np.mean(diffs):+.1f}s")

    # Overall
    overall_d = np.mean(all_diffs) / np.mean(all_stds) if np.mean(all_stds) > 0 else float("inf")
    overall_cat = ("large" if abs(overall_d) > 0.8 else "medium" if abs(overall_d) > 0.5
                   else "small" if abs(overall_d) > 0.2 else "negligible")
    print(f"\n  Overall: d = {overall_d:.2f}  [{overall_cat}]")

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Formal statistical tests")
    ap.add_argument("--csv", default="results/multiseed_robustness.csv",
                    help="Path to multi-seed results CSV")
    args = ap.parse_args()

    df = load_data(args.csv)

    # Run all tests
    wilcoxon_df = wilcoxon_paired_tests(df)
    wilcoxon_df.to_csv(OUT_DIR / "wilcoxon_results.csv", index=False)
    print(f"  → Saved: {OUT_DIR}/wilcoxon_results.csv")

    lp_df = wilcoxon_lp_contribution(df)
    if not lp_df.empty:
        lp_df.to_csv(OUT_DIR / "wilcoxon_lp_contribution.csv", index=False)
        print(f"  → Saved: {OUT_DIR}/wilcoxon_lp_contribution.csv")

    friedman_result = friedman_test(df)

    bootstrap_result = bootstrap_grand_mean(df)

    effect_df = effect_size_summary(df)
    effect_df.to_csv(OUT_DIR / "effect_sizes.csv", index=False)
    print(f"  → Saved: {OUT_DIR}/effect_sizes.csv")

    # Save combined summary
    summary = {
        "wilcoxon_all_significant_005": bool(wilcoxon_df["significant_005"].all()),
        "n_significant_scenarios": int(wilcoxon_df["significant_005"].sum()),
        "friedman_chi2": friedman_result.get("friedman_chi2"),
        "friedman_p": friedman_result.get("friedman_p"),
        "grand_mean_improvement_pct": bootstrap_result["grand_mean_pct"],
        "bootstrap_ci_lo": bootstrap_result["ci_lo_pct"],
        "bootstrap_ci_hi": bootstrap_result["ci_hi_pct"],
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "statistical_summary.csv", index=False)
    print(f"\n  → Saved: {OUT_DIR}/statistical_summary.csv")

    print("\n" + "=" * 80)
    print("STATISTICAL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
