#!/usr/bin/env python3
"""Prototype exact-envelope cut generation on the coarsened Path 3 grid."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_path3_exact_envelope_diagnostic import (  # noqa: E402
    _allocations,
    _rho,
)
from scripts.external_baselines_eval import _pick_block, _top_matches  # noqa: E402
from src.policies.anticipatory import _HexGrid  # noqa: E402
from src.policies.external_baselines import _normalize_scores, _record_zone_fractions  # noqa: E402
from src.regime.events import annotate_events  # noqa: E402
from src.regime.ingest import build_demand_profile, load_cleaned, split_into_blocks  # noqa: E402
from src.regime.learned_weights import PATH2_SCENARIOS, load_gate  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402
from src.simulator.routing import HaversineClient  # noqa: E402
from src.theory.exact_envelope import (  # noqa: E402
    milp_directional_residual_envelope_witness,
    scalar_residual_certificate,
)

OUT_DIR = Path("results/path3_exact_envelope_cutgen")
OUT_TEX = Path("paper/path3_exact_envelope_cutgen.tex")
OUT_MD = Path("docs/path3-exact-envelope-cutgen.md")


@dataclass
class SolveResult:
    status: str
    objective: float
    move_units: float
    move_seconds: float
    final_exact: float
    final_scalar: float
    iterations: int
    cuts: int
    max_violation: float


def _fmt(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{float(x):.{digits}f}"


def _cost_matrix(grid: _HexGrid) -> np.ndarray:
    router = HaversineClient()
    centers = [grid.hex_centers[h] for h in grid.hex_ids]
    return router.distance_matrix(centers, centers)


def _flow_index(n_zones: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_zones) for j in range(n_zones) if i != j]


def _base_flow_constraints(
    initial_allocation: np.ndarray,
    cost: np.ndarray,
    max_move_fraction: float,
    extra_vars: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[float],
    list[tuple[float, float | None]],
    list[tuple[int, int]],
    int,
]:
    initial_allocation = np.asarray(initial_allocation, dtype=float)
    n_zones = initial_allocation.size
    pairs = _flow_index(n_zones)
    n_x = len(pairs)
    offset_a = n_x
    n_vars = n_x + n_zones + extra_vars

    objective = np.zeros(n_vars, dtype=float)
    max_cost = max(float(np.max(cost)), 1.0)
    for k, (i, j) in enumerate(pairs):
        objective[k] = cost[i, j] / max_cost

    a_eq_rows = []
    b_eq_vals = []
    for z in range(n_zones):
        row = np.zeros(n_vars, dtype=float)
        row[offset_a + z] = 1.0
        for k, (i, j) in enumerate(pairs):
            if i == z:
                row[k] += 1.0
            if j == z:
                row[k] -= 1.0
        a_eq_rows.append(row)
        b_eq_vals.append(float(initial_allocation[z]))

    a_ub_rows: list[np.ndarray] = []
    b_ub_vals: list[float] = []
    for z in range(n_zones):
        row = np.zeros(n_vars, dtype=float)
        for k, (i, _) in enumerate(pairs):
            if i == z:
                row[k] = 1.0
        a_ub_rows.append(row)
        b_ub_vals.append(float(initial_allocation[z]))

    move_row = np.zeros(n_vars, dtype=float)
    move_row[:n_x] = 1.0
    a_ub_rows.append(move_row)
    b_ub_vals.append(float(initial_allocation.sum() * max_move_fraction))

    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_vars
    return (
        np.vstack(a_eq_rows),
        np.asarray(b_eq_vals, dtype=float),
        a_ub_rows,
        b_ub_vals,
        bounds,
        pairs,
        offset_a,
    )


def _post_allocation(x: np.ndarray, offset_a: int, n_zones: int) -> np.ndarray:
    return np.asarray(x[offset_a:offset_a + n_zones], dtype=float)


def solve_exact_cut_generation(
    initial_allocation: np.ndarray,
    mixture_share: np.ndarray,
    rho: float,
    gamma: float,
    cost: np.ndarray,
    *,
    max_move_fraction: float = 0.50,
    max_iterations: int = 40,
    tolerance: float = 1e-5,
) -> SolveResult:
    n_zones = initial_allocation.size
    total_idle = float(initial_allocation.sum())
    (
        a_eq,
        b_eq,
        base_a_ub,
        base_b_ub,
        bounds,
        pairs,
        offset_a,
    ) = _base_flow_constraints(initial_allocation, cost, max_move_fraction)
    objective = np.zeros(len(bounds), dtype=float)
    max_cost = max(float(np.max(cost)), 1.0)
    for k, (i, j) in enumerate(pairs):
        objective[k] = cost[i, j] / max_cost

    cuts: list[np.ndarray] = []
    cut_rhs: list[float] = []
    seen_masks: set[tuple[int, ...]] = set()
    last_violation = np.inf
    result = None

    for iteration in range(1, max_iterations + 1):
        a_ub_rows = list(base_a_ub)
        b_ub_vals = list(base_b_ub)
        for mask, rhs in zip(cuts, cut_rhs, strict=True):
            row = np.zeros(len(bounds), dtype=float)
            row[offset_a:offset_a + n_zones][mask] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(-rhs)

        result = linprog(
            objective,
            A_ub=np.vstack(a_ub_rows),
            b_ub=np.asarray(b_ub_vals, dtype=float),
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return SolveResult(
                status=f"linprog_{result.status}",
                objective=np.nan,
                move_units=np.nan,
                move_seconds=np.nan,
                final_exact=np.nan,
                final_scalar=np.nan,
                iterations=iteration,
                cuts=len(cuts),
                max_violation=np.nan,
            )

        allocation = _post_allocation(result.x, offset_a, n_zones)
        envelope, mask = milp_directional_residual_envelope_witness(
            allocation, mixture_share, rho
        )
        last_violation = float(envelope - gamma)
        if last_violation <= tolerance:
            flow = result.x[:len(pairs)]
            move_seconds = sum(
                float(flow[k] * cost[i, j]) for k, (i, j) in enumerate(pairs)
            )
            return SolveResult(
                status="success",
                objective=float(result.fun),
                move_units=float(flow.sum()),
                move_seconds=float(move_seconds),
                final_exact=float(envelope),
                final_scalar=scalar_residual_certificate(allocation, mixture_share, rho),
                iterations=iteration,
                cuts=len(cuts),
                max_violation=max(0.0, last_violation),
            )

        mask_key = tuple(np.flatnonzero(mask).tolist())
        if mask_key in seen_masks:
            return SolveResult(
                status="duplicate_cut",
                objective=float(result.fun),
                move_units=np.nan,
                move_seconds=np.nan,
                final_exact=float(envelope),
                final_scalar=scalar_residual_certificate(allocation, mixture_share, rho),
                iterations=iteration,
                cuts=len(cuts),
                max_violation=last_violation,
            )
        seen_masks.add(mask_key)
        rhs = total_idle * min(1.0, float(mixture_share[mask].sum()) + rho) - gamma
        if rhs > tolerance:
            cuts.append(mask)
            cut_rhs.append(float(rhs))

    allocation = (
        _post_allocation(result.x, offset_a, n_zones)
        if result is not None
        else initial_allocation
    )
    return SolveResult(
        status="iteration_limit",
        objective=float(result.fun) if result is not None and result.success else np.nan,
        move_units=np.nan,
        move_seconds=np.nan,
        final_exact=milp_directional_residual_envelope_witness(
            allocation, mixture_share, rho
        )[0],
        final_scalar=scalar_residual_certificate(allocation, mixture_share, rho),
        iterations=max_iterations,
        cuts=len(cuts),
        max_violation=last_violation,
    )


def solve_scalar_certificate(
    initial_allocation: np.ndarray,
    mixture_share: np.ndarray,
    rho: float,
    gamma: float,
    cost: np.ndarray,
    *,
    max_move_fraction: float = 0.50,
) -> SolveResult:
    n_zones = initial_allocation.size
    total_idle = float(initial_allocation.sum())
    scalar_budget = float(gamma - total_idle * rho)
    if scalar_budget < -1e-8:
        return SolveResult(
            "infeasible_negative_scalar_budget",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            0,
            np.nan,
        )

    (
        a_eq,
        b_eq,
        a_ub_rows,
        b_ub_vals,
        bounds,
        pairs,
        offset_a,
    ) = _base_flow_constraints(initial_allocation, cost, max_move_fraction, extra_vars=n_zones)
    n_x = len(pairs)
    offset_u = offset_a + n_zones
    objective = np.zeros(len(bounds), dtype=float)
    max_cost = max(float(np.max(cost)), 1.0)
    for k, (i, j) in enumerate(pairs):
        objective[k] = cost[i, j] / max_cost

    for z in range(n_zones):
        row = np.zeros(len(bounds), dtype=float)
        row[offset_a + z] = -1.0
        row[offset_u + z] = -1.0
        a_ub_rows.append(row)
        b_ub_vals.append(-float(total_idle * mixture_share[z]))

    row = np.zeros(len(bounds), dtype=float)
    row[offset_u:offset_u + n_zones] = 1.0
    a_ub_rows.append(row)
    b_ub_vals.append(max(0.0, scalar_budget))

    result = linprog(
        objective,
        A_ub=np.vstack(a_ub_rows),
        b_ub=np.asarray(b_ub_vals, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return SolveResult(
            status=f"linprog_{result.status}",
            objective=np.nan,
            move_units=np.nan,
            move_seconds=np.nan,
            final_exact=np.nan,
            final_scalar=np.nan,
            iterations=1,
            cuts=0,
            max_violation=np.nan,
        )
    allocation = _post_allocation(result.x, offset_a, n_zones)
    flow = result.x[:n_x]
    move_seconds = sum(float(flow[k] * cost[i, j]) for k, (i, j) in enumerate(pairs))
    final_exact = milp_directional_residual_envelope_witness(allocation, mixture_share, rho)[0]
    final_scalar = scalar_residual_certificate(allocation, mixture_share, rho)
    return SolveResult(
        status="success",
        objective=float(result.fun),
        move_units=float(flow.sum()),
        move_seconds=float(move_seconds),
        final_exact=float(final_exact),
        final_scalar=float(final_scalar),
        iterations=1,
        cuts=0,
        max_violation=max(0.0, final_scalar - gamma),
    )


def _retrieval_states() -> list[dict]:
    library = RegimeLibrary()
    library.load()
    gate = load_gate("results/path2_gate_spatial/similarity_gate.pt")
    grid = _HexGrid(h3_res=7)

    jan = load_cleaned("2024-01")
    jun = load_cleaned("2024-06")
    trips = pd.concat([jan, jun], ignore_index=True)
    blocks = split_into_blocks(build_demand_profile(trips))

    states = []
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
        states.append({
            "scenario": scenario,
            "block_id": bid,
            "grid": grid,
            "mixture": mixture,
            "total_idle": float(total_idle),
        })
    return states


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rho = _rho(0.70)
    reduction = 0.25
    states = _retrieval_states()
    grid = states[0]["grid"]
    cost = _cost_matrix(grid)

    rows = []
    for state in states:
        mixture = state["mixture"]
        allocations = _allocations(state["total_idle"], mixture)
        for label, initial_allocation in allocations.items():
            baseline_exact, _ = milp_directional_residual_envelope_witness(
                initial_allocation, mixture, rho
            )
            baseline_scalar = scalar_residual_certificate(initial_allocation, mixture, rho)
            gamma = baseline_exact * (1.0 - reduction)
            exact = solve_exact_cut_generation(initial_allocation, mixture, rho, gamma, cost)
            scalar = solve_scalar_certificate(initial_allocation, mixture, rho, gamma, cost)
            rows.append({
                "scenario": state["scenario"],
                "block_id": state["block_id"],
                "allocation": label,
                "rho": rho,
                "target_reduction": reduction,
                "gamma": gamma,
                "baseline_exact": baseline_exact,
                "baseline_scalar": baseline_scalar,
                "exact_status": exact.status,
                "exact_move_units": exact.move_units,
                "exact_move_seconds": exact.move_seconds,
                "exact_final_envelope": exact.final_exact,
                "exact_final_scalar": exact.final_scalar,
                "exact_iterations": exact.iterations,
                "exact_cuts": exact.cuts,
                "scalar_status": scalar.status,
                "scalar_move_units": scalar.move_units,
                "scalar_move_seconds": scalar.move_seconds,
                "scalar_final_envelope": scalar.final_exact,
                "scalar_final_scalar": scalar.final_scalar,
                "scalar_iterations": scalar.iterations,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exact_envelope_cutgen.csv", index=False)
    success = df[(df["exact_status"] == "success") & (df["scalar_status"] == "success")].copy()
    success["move_saving_pct"] = (
        (success["scalar_move_units"] - success["exact_move_units"])
        / success["scalar_move_units"].replace(0.0, np.nan)
        * 100.0
    )
    summary = (
        df.groupby("allocation", as_index=False)
        .agg(
            cases=("scenario", "count"),
            exact_success=("exact_status", lambda s: int((s == "success").sum())),
            scalar_success=("scalar_status", lambda s: int((s == "success").sum())),
            exact_move_units=("exact_move_units", "mean"),
            scalar_move_units=("scalar_move_units", "mean"),
            exact_cuts=("exact_cuts", "mean"),
            exact_iterations=("exact_iterations", "mean"),
            baseline_exact=("baseline_exact", "mean"),
            gamma=("gamma", "mean"),
        )
    )
    saving_summary = (
        success.groupby("allocation", as_index=False)
        .agg(move_saving_pct=("move_saving_pct", "mean"))
    )
    summary = summary.merge(saving_summary, on="allocation", how="left")
    summary.to_csv(OUT_DIR / "exact_envelope_cutgen_summary.csv", index=False)

    rows_tex = []
    for _, row in summary.sort_values("allocation").iterrows():
        rows_tex.append(
            f"{row['allocation']} & "
            f"{int(row['exact_success'])}/{int(row['cases'])} & "
            f"{int(row['scalar_success'])}/{int(row['cases'])} & "
            f"{_fmt(row['exact_move_units'], 1)} & "
            f"{_fmt(row['scalar_move_units'], 1)} & "
            f"{_fmt(row['move_saving_pct'], 1)} & "
            f"{_fmt(row['exact_cuts'], 1)} \\\\"
        )
    tex = (
        "% Auto-generated by scripts/analyze_path3_exact_envelope_cutgen.py\n\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Exact-envelope cut-generation diagnostic on the coarsened "
        "28-zone grid. Each case minimizes empty movement subject to a "
        "25\\% reduction in the exact directional residual envelope from the "
        "initial allocation. Scalar DCR imposes the tractable upper certificate "
        "at the same budget.}\n"
        "\\label{tab:path3-exact-envelope-cutgen}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "Allocation & Exact ok & Scalar ok & Exact moves & Scalar moves & "
        "Save \\% & Cuts \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    OUT_TEX.write_text(tex)

    lines = [
        "# Exact-Envelope Cut Generation",
        "",
        "This is an offline 28-zone prototype that turns the exact residual-envelope "
        "theorem into a separation-based optimization routine.",
        "",
        f"Target: `{reduction:.0%}` reduction in the exact envelope from the "
        "initial allocation.",
        f"Positive residual radius: `{rho:.3f}` from the 0.70 calibration score.",
        "",
        "| Allocation | Exact ok | Scalar ok | Exact moves | Scalar moves | Saving % | Cuts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values("allocation").iterrows():
        lines.append(
            f"| {row['allocation']} | {int(row['exact_success'])}/{int(row['cases'])} | "
            f"{int(row['scalar_success'])}/{int(row['cases'])} | "
            f"{_fmt(row['exact_move_units'], 1)} | {_fmt(row['scalar_move_units'], 1)} | "
            f"{_fmt(row['move_saving_pct'], 1)} | {_fmt(row['exact_cuts'], 1)} |"
        )
    lines.extend([
        "",
        "Interpretation: exact-envelope cuts are a genuine algorithmic extension of "
        "the theorem. The scalar DCR certificate is scalable and valid, but the "
        "cut-generated exact envelope can be less conservative at the same risk "
        "budget on the coarsened grid.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_DIR / 'exact_envelope_cutgen.csv'}")
    print(f"Wrote {OUT_DIR / 'exact_envelope_cutgen_summary.csv'}")
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_MD}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
