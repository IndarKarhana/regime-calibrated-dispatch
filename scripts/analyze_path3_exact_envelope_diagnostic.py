#!/usr/bin/env python3
"""Diagnose the exact directional residual envelope on coarsened retrieval states."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.external_baselines_eval import _pick_block, _top_matches  # noqa: E402
from src.policies.anticipatory import _HexGrid  # noqa: E402
from src.policies.external_baselines import _normalize_scores, _record_zone_fractions  # noqa: E402
from src.regime.events import annotate_events  # noqa: E402
from src.regime.ingest import build_demand_profile, load_cleaned, split_into_blocks  # noqa: E402
from src.regime.learned_weights import PATH2_SCENARIOS, load_gate  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402
from src.theory.exact_envelope import (  # noqa: E402
    greedy_directional_residual_envelope_lower_bound,
    milp_directional_residual_envelope,
    scalar_residual_certificate,
    shortage_loss,
)

BUFFER_SUMMARY = Path("results/path3_retrieval_buffers_top5/buffer_summary.csv")
OUT_DIR = Path("results/path3_exact_envelope_diagnostic")
OUT_TEX = Path("paper/path3_exact_envelope_diagnostic.tex")
OUT_MD = Path("docs/path3-exact-envelope-diagnostic.md")


def _fmt(x: float, digits: int = 1) -> str:
    return f"{float(x):.{digits}f}"


def _rho(alpha: float = 0.70) -> float:
    df = pd.read_csv(BUFFER_SUMMARY)
    row = df.loc[np.isclose(df["alpha"], alpha)].iloc[0]
    return float(row["positive_mass_quantile"])


def _allocations(total_idle: float, mixture: np.ndarray) -> dict[str, np.ndarray]:
    n = mixture.size
    uniform = np.ones(n, dtype=float) * total_idle / n
    mixture_alloc = total_idle * mixture

    scarcity_order = np.argsort(-mixture)
    anti = mixture_alloc.copy()
    top = scarcity_order[: max(1, n // 5)]
    bottom = scarcity_order[-max(1, n // 5):]
    move_mass = min(float(anti[top].sum()) * 0.35, total_idle * 0.20)
    if move_mass > 0:
        anti[top] -= move_mass * anti[top] / max(float(anti[top].sum()), 1e-12)
        anti[bottom] += move_mass / len(bottom)

    concentrated = np.zeros(n, dtype=float)
    concentrated[scarcity_order[: max(1, n // 4)]] = total_idle / max(1, n // 4)

    return {
        "Uniform supply": uniform,
        "Retrieved mixture target": mixture_alloc,
        "Anti-retrieval supply": anti,
        "Top-quartile concentration": concentrated,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    library = RegimeLibrary()
    library.load()
    gate = load_gate("results/path2_gate_spatial/similarity_gate.pt")
    grid = _HexGrid(h3_res=7)
    rho = _rho(0.70)

    jan = load_cleaned("2024-01")
    jun = load_cleaned("2024-06")
    trips = pd.concat([jan, jun], ignore_index=True)
    blocks = split_into_blocks(build_demand_profile(trips))

    rows = []
    for scenario, (month, hour_range, day_type) in PATH2_SCENARIOS.items():
        block = _pick_block(blocks, month, hour_range, day_type)
        if block is None:
            continue
        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].to_numpy(dtype=np.float64)
        qrec = library[bid] if bid in library.records else None
        q_events = qrec.events if qrec is not None else annotate_events(q_series)
        weights = (
            gate(qrec.demand_series, qrec.events, bid, qrec)
            if qrec is not None else gate(q_series, q_events, bid)
        )
        matches = _top_matches(library, q_series, q_events, bid, weights, 5)
        scores = [score for _, score in matches]
        regime_weights = _normalize_scores(scores, len(scores))
        shares = np.vstack([
            _record_zone_fractions(library[cid], grid)
            for cid, _ in matches
        ])
        mixture = np.average(shares, axis=0, weights=regime_weights)
        mixture = np.maximum(mixture, 0.0)
        mixture = mixture / max(float(mixture.sum()), 1e-12)
        total_idle = max(int(block["request_count"].sum() / 4.0 * 0.15), 50)

        for label, allocation in _allocations(float(total_idle), mixture).items():
            exact = milp_directional_residual_envelope(allocation, mixture, rho)
            scalar = scalar_residual_certificate(allocation, mixture, rho)
            greedy = greedy_directional_residual_envelope_lower_bound(
                allocation, mixture, rho
            )
            rows.append({
                "scenario": scenario,
                "block_id": bid,
                "allocation": label,
                "zones": grid.n,
                "rho": rho,
                "mixture_shortage": shortage_loss(allocation, mixture),
                "exact_envelope": exact,
                "greedy_lower_bound": greedy,
                "scalar_certificate": scalar,
                "scalar_gap": scalar - exact,
                "scalar_gap_pct": (scalar - exact) / max(exact, 1e-9) * 100.0,
                "greedy_gap": exact - greedy,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exact_envelope_diagnostic.csv", index=False)
    summary = (
        df.groupby("allocation", as_index=False)
        .agg(
            exact_envelope=("exact_envelope", "mean"),
            scalar_certificate=("scalar_certificate", "mean"),
            scalar_gap_pct=("scalar_gap_pct", "mean"),
            greedy_gap=("greedy_gap", "mean"),
        )
        .sort_values("exact_envelope")
    )
    summary.to_csv(OUT_DIR / "exact_envelope_summary.csv", index=False)

    rows_tex = []
    for _, row in summary.iterrows():
        rows_tex.append(
            f"{row['allocation']} & {_fmt(row['exact_envelope'], 1)} & "
            f"{_fmt(row['scalar_certificate'], 1)} & "
            f"{_fmt(row['scalar_gap_pct'], 1)} & "
            f"{_fmt(row['greedy_gap'], 2)} \\\\"
        )
    tex = (
        "% Auto-generated by scripts/analyze_path3_exact_envelope_diagnostic.py\n\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Exact directional residual envelope diagnostic on a coarsened "
        "28-zone grid. The scalar DCR certificate upper-bounds the exact envelope "
        "but can be conservative, motivating future exact-envelope separation.}\n"
        "\\label{tab:path3-exact-envelope-diagnostic}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Allocation & Exact envelope & Scalar cert. & Gap \\% & Greedy gap \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    OUT_TEX.write_text(tex)

    lines = [
        "# Exact Envelope Diagnostic",
        "",
        "This diagnostic evaluates the exact directional residual envelope on the "
        "coarsened 28-zone grid. It is a formula diagnostic, not a closed-loop "
        "policy comparison.",
        "",
        f"Positive residual radius: `{rho:.3f}` from the 0.70 calibration score.",
        "",
        "| Allocation | Exact envelope | Scalar cert. | Gap % | Greedy gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['allocation']} | {_fmt(row['exact_envelope'], 1)} | "
            f"{_fmt(row['scalar_certificate'], 1)} | "
            f"{_fmt(row['scalar_gap_pct'], 1)} | {_fmt(row['greedy_gap'], 2)} |"
        )
    lines.extend([
        "",
        "Interpretation: the scalar DCR residual certificate is valid and tractable, "
        "but it is not always tight. That supports the paper's claim that the "
        "exact envelope is the sharper mathematical object while the implemented "
        "DCR controller uses a scalable upper certificate.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_DIR / 'exact_envelope_diagnostic.csv'}")
    print(f"Wrote {OUT_DIR / 'exact_envelope_summary.csv'}")
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
