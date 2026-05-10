#!/usr/bin/env python3
"""Test the consistency assumption: higher-similarity regimes have lower demand error.

For each test scenario, we:
1. Get the true demand series from replay data.
2. Query the regime library for ALL regimes (not just top-k), getting similarity scores.
3. Compute each regime's demand error: ||λ_i - λ*||_2.
4. Test Spearman rank correlation between similarity and -error (should be positive).

This validates Proposition 2's key assumption:
  s_i >= s_j  ⟹  ε_i <= ε_j  (higher similarity → lower error)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.events import annotate_events
from src.regime.similarity import compute_similarity


# ── Scenarios (same as paper_experiments.py) ─────────────────────────────────
NYC_SCENARIOS = {
    "jan_weekday_am":  ("2024-01", "08-12", "weekday"),
    "jan_weekday_pm":  ("2024-01", "16-20", "weekday"),
    "jan_nye_am":      ("2024-01", "08-12", "nye"),
    "jan_nye_pm":      ("2024-01", "16-20", "nye"),
    "jan_weekend_mid": ("2024-01", "12-16", "weekend"),
    "jun_weekday_am":  ("2024-06", "08-12", "weekday"),
    "jun_weekday_pm":  ("2024-06", "16-20", "weekday"),
    "jun_late_night":  ("2024-06", "20-24", "fri_night"),
}


def _pick_block(blocks, month_prefix, hour_range, day_type):
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        if not bid.startswith(month_prefix):
            continue
        if hour_range not in bid:
            continue
        if blk["request_count"].sum() < 500:
            continue
        date_str = bid.rsplit("_", 1)[0]
        ts = pd.Timestamp(date_str)
        dow = ts.dayofweek
        if day_type == "weekday" and dow < 5 and not (ts.month == 1 and ts.day == 1):
            return blk
        elif day_type == "weekend" and dow >= 5:
            return blk
        elif day_type == "nye" and ts.month == 1 and ts.day == 1:
            return blk
        elif day_type == "fri_night" and dow in (4, 5):
            return blk
    return None


def main():
    # Load library (auto-discovers from config)
    lib = RegimeLibrary()
    # Load saved records from disk
    lib.load()
    print(f"Regime library: {len(lib.records)} blocks\n")

    # Load and prepare blocks (both months)
    df = load_cleaned()
    profile = build_demand_profile(df)
    blocks = split_into_blocks(profile)

    results = []

    for scenario_name, (month, hours, dtype) in NYC_SCENARIOS.items():
        blk = _pick_block(blocks, month, hours, dtype)
        if blk is None:
            print(f"  {scenario_name}: block not found, skipping")
            continue

        block_id = blk["block_id"].iloc[0]
        true_demand = blk["request_count"].values.astype(float)
        true_events = annotate_events(true_demand)

        # Compute similarity to every library regime
        similarities = []
        errors = []

        for bid, rec in lib.records.items():
            # Skip self-match (the exact same block)
            if bid == block_id:
                continue

            sim = compute_similarity(
                true_demand, true_events, rec, q_block_id=block_id,
            )

            # Demand error: L2 norm between regime's demand and true demand
            lib_demand = rec.demand_series
            min_len = min(len(true_demand), len(lib_demand))
            err = np.linalg.norm(true_demand[:min_len] - lib_demand[:min_len])

            similarities.append(sim)
            errors.append(err)

        sims = np.array(similarities)
        errs = np.array(errors)

        # Spearman: similarity vs. -error (positive correlation means consistency)
        rho, p_val = stats.spearmanr(sims, -errs)

        # Also check top-k: are top-5 errors lower than bottom-5?
        sorted_idx = np.argsort(-sims)  # highest similarity first
        top5_err = np.mean(errs[sorted_idx[:5]])
        bot5_err = np.mean(errs[sorted_idx[-5:]])
        median_err = np.median(errs)
        top5_mean_err = np.mean(errs[sorted_idx[:5]])

        results.append({
            "scenario": scenario_name,
            "block_id": block_id,
            "n_regimes": len(sims),
            "spearman_rho": rho,
            "spearman_p": p_val,
            "top5_mean_error": top5_err,
            "bottom5_mean_error": bot5_err,
            "median_error": median_err,
            "top5_vs_median": f"{(1 - top5_mean_err / median_err) * 100:.1f}%",
            "consistent": rho > 0 and p_val < 0.05,
        })

        status = "✓" if rho > 0 and p_val < 0.05 else "✗"
        print(f"  {status} {scenario_name:20s}  ρ={rho:+.3f}  p={p_val:.2e}"
              f"  top5_err={top5_err:.1f}  median_err={median_err:.1f}"
              f"  bot5_err={bot5_err:.1f}")

    # Summary
    print("\n" + "=" * 70)
    rhos = [r["spearman_rho"] for r in results]
    n_consistent = sum(1 for r in results if r["consistent"])
    print(f"Mean Spearman ρ:  {np.mean(rhos):+.3f}")
    print(f"Consistent (ρ>0, p<0.05): {n_consistent}/{len(results)} scenarios")
    print(f"All top-5 errors < median: "
          f"{sum(1 for r in results if r['top5_mean_error'] < r['median_error'])}/{len(results)}")

    # Save results
    out = Path("results/consistency_test.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
