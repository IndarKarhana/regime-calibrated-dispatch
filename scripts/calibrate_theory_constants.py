#!/usr/bin/env python3
"""Empirically calibrate Path 2 theory constants and sensitivity diagnostics.

This script is a reviewer-facing bridge between the stylized theorem and the
closed-loop simulator. It estimates constants on the exact eight Path 2
scenarios under the same Haversine service-time convention used by the
simulator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regime.ingest import build_demand_profile, load_cleaned, split_into_blocks  # noqa: E402
from src.regime.learned_weights import PATH2_SCENARIOS  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402
from src.simulator.routing import HaversineClient  # noqa: E402
from src.theory.spatial_metrics import build_zone_index, travel_cost_matrix_s  # noqa: E402
from src.theory.wait_bounds import (  # noqa: E402
    earth_movers_distance,
    maximum_utilization,
    proportional_capacity_allocation,
    queue_wait_regret_bound,
)


def _pick_block(blocks, month_prefix: str, hour_range: str, day_type: str):
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        if not bid.startswith(month_prefix) or hour_range not in bid:
            continue
        if blk["request_count"].sum() < 500:
            continue
        date_str = bid.rsplit("_", 1)[0]
        ts = pd.Timestamp(date_str)
        if ts.month == 1 and ts.day == 1:
            dtype = "holiday"
        elif ts.dayofweek >= 5:
            dtype = "weekend"
        else:
            dtype = "weekday"
        if dtype == day_type:
            return blk
    return None


def _get_replay_trips(all_trips: pd.DataFrame, bid: str) -> pd.DataFrame:
    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    mask = all_trips["pickup_datetime"].dt.date == target_date
    mask &= all_trips["pickup_datetime"].dt.hour >= start_h
    mask &= all_trips["pickup_datetime"].dt.hour < end_h
    return all_trips[mask].copy()


def _haversine_service_rate_per_hour(trips: pd.DataFrame, router: HaversineClient) -> tuple[float, float]:
    pairs = [
        ((float(r.pickup_longitude), float(r.pickup_latitude)),
         (float(r.dropoff_longitude), float(r.dropoff_latitude)))
        for r in trips.itertuples()
    ]
    if not pairs:
        return 20.0, 180.0
    travel_s = np.asarray(router.batch_travel_times(pairs), dtype=float)
    travel_s = travel_s[np.isfinite(travel_s) & (travel_s > 0)]
    if travel_s.size == 0:
        return 20.0, 180.0
    mean_service_s = float(np.mean(travel_s))
    return float(3600.0 / mean_service_s), mean_service_s


def _scenario_ids_from_blocks(library: RegimeLibrary) -> dict[str, str]:
    ids = {}
    for scen, (month, hour_range, day_type) in PATH2_SCENARIOS.items():
        for bid in sorted(library.records):
            if not bid.startswith(month) or hour_range not in bid:
                continue
            ts = pd.Timestamp(bid.rsplit("_", 1)[0])
            if ts.month == 1 and ts.day == 1:
                dtype = "holiday"
            elif ts.dayofweek >= 5:
                dtype = "weekend"
            else:
                dtype = "weekday"
            if dtype == day_type:
                ids[scen] = bid
                break
    return ids


def _allocator_sensitivity_rows(
    scenario: str,
    true_rate: np.ndarray,
    fleet: int,
    mu: float,
    rng: np.random.Generator,
    *,
    n_draws: int,
) -> list[dict]:
    base_cap = proportional_capacity_allocation(true_rate, fleet, mu)
    rows = []
    active = np.flatnonzero(true_rate > 1e-9)
    if active.size == 0:
        return rows

    for draw in range(n_draws):
        noise = rng.normal(0.0, 0.08, size=true_rate.shape)
        perturbed = np.maximum(true_rate * (1.0 + noise), 0.0)
        if perturbed.sum() > 1e-12 and true_rate.sum() > 1e-12:
            perturbed *= true_rate.sum() / perturbed.sum()
        pert_cap = proportional_capacity_allocation(perturbed, fleet, mu)
        demand_l1 = float(np.abs(perturbed - true_rate).sum())
        cap_l1 = float(np.abs(pert_cap - base_cap).sum())
        changed = base_cap != pert_cap
        rows.append({
            "scenario": scenario,
            "draw": draw,
            "demand_l1_rate_per_h": demand_l1,
            "capacity_l1_drivers": cap_l1,
            "allocator_ratio_drivers_per_rate": cap_l1 / max(demand_l1, 1e-12),
            "basis_flip": bool(np.any(changed)),
            "changed_zone_fraction": float(np.mean(changed)),
        })
    return rows


def _rounding_residual(true_rate: np.ndarray, cap: np.ndarray, fleet: int) -> float:
    if true_rate.sum() <= 1e-12 or fleet <= 0:
        return 0.0
    continuous = true_rate / true_rate.sum() * fleet
    return float(np.abs(cap - continuous).sum() / fleet)


def _spatial_shift_rows(
    scenario: str,
    true_counts: np.ndarray,
    cost_s: np.ndarray,
    fleet: int,
    mu: float,
    horizon_h: float,
) -> list[dict]:
    rows = []
    active = np.flatnonzero(true_counts > 1e-9)
    if active.size < 2:
        return rows
    source_order = active[np.argsort(-true_counts[active])]
    source_candidates = source_order[: min(3, len(source_order))]
    for src in source_candidates:
        far_order = np.argsort(-cost_s[src])
        dst_candidates = [int(dst) for dst in far_order if dst != src][:3]
        for dst in dst_candidates:
            for frac in (0.02, 0.05, 0.10, 0.15):
                shifted = true_counts.astype(float).copy()
                mass = min(float(true_counts[src]) * frac, float(true_counts[src]))
                shifted[src] -= mass
                shifted[dst] += mass
                w1_s = earth_movers_distance(cost_s, true_counts, shifted)
                bound = queue_wait_regret_bound(
                    true_counts / horizon_h,
                    shifted / horizon_h,
                    fleet,
                    service_rate=mu,
                    rho_max=0.90,
                )
                rows.append({
                    "scenario": scenario,
                    "src_zone_ranked": int(src),
                    "dst_zone_ranked": int(dst),
                    "shift_fraction_of_src": frac,
                    "shifted_count": mass,
                    "spatial_w1_s": w1_s,
                    "queue_wait_gap_s": bound.wait_gap_s,
                    "beta_wait_per_w1": bound.wait_gap_s / max(w1_s, 1e-12),
                    "max_utilization": bound.max_utilization,
                })
    return rows


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("paper/figures").mkdir(parents=True, exist_ok=True)

    library = RegimeLibrary()
    library.load()
    zone_ids, counts_by_record = build_zone_index(
        library,
        h3_res=args.h3_res,
        max_zones=args.max_zones,
    )
    cost_s = travel_cost_matrix_s(zone_ids)
    scenario_ids = _scenario_ids_from_blocks(library)

    jan = load_cleaned("2024-01")
    jun = load_cleaned("2024-06")
    trips = pd.concat([jan, jun], ignore_index=True)
    blocks = split_into_blocks(build_demand_profile(trips))
    router = HaversineClient()
    rng = np.random.default_rng(args.seed)

    constants = []
    sensitivity = []
    spatial = []
    for scen, (month, hour_range, day_type) in PATH2_SCENARIOS.items():
        qid = scenario_ids.get(scen)
        blk = _pick_block(blocks, month, hour_range, day_type)
        if qid is None or blk is None:
            continue
        bid = blk["block_id"].iloc[0]
        replay_trips = _get_replay_trips(trips, bid)
        mu, mean_service_s = _haversine_service_rate_per_hour(replay_trips, router)

        counts = counts_by_record[qid].astype(float)
        true_rate = counts / args.horizon_hours
        fleet = max(int(true_rate.sum() * args.fleet_ratio), args.min_fleet)
        cap = proportional_capacity_allocation(true_rate, fleet, mu, rho_target=0.82)
        rho = maximum_utilization(true_rate, cap, mu)
        effective_rho = min(max(rho, 0.0), args.rho_max)
        lq_hours = 1.0 / (mu * max(1.0 - effective_rho, 1e-6) ** 2)

        constants.append({
            "scenario": scen,
            "query_id": qid,
            "Z": len(zone_ids),
            "fleet": fleet,
            "total_demand_count": float(counts.sum()),
            "lambda_total_per_h": float(true_rate.sum()),
            "mean_haversine_service_s": mean_service_s,
            "mu_per_h": mu,
            "max_rho": rho,
            "L_Q_hours": lq_hours,
            "L_Q_seconds": lq_hours * 3600.0,
            "rounding_residual_l1_frac": _rounding_residual(true_rate, cap, fleet),
        })
        sensitivity.extend(_allocator_sensitivity_rows(
            scen, true_rate, fleet, mu, rng, n_draws=args.allocator_draws
        ))
        spatial.extend(_spatial_shift_rows(
            scen, counts, cost_s, fleet, mu, args.horizon_hours
        ))

    constants_df = pd.DataFrame(constants)
    sensitivity_df = pd.DataFrame(sensitivity)
    spatial_df = pd.DataFrame(spatial)

    constants_df.to_csv(out_dir / "theory_constant_calibration.csv", index=False)
    sensitivity_df.to_csv(out_dir / "allocator_sensitivity_diagnostics.csv", index=False)
    spatial_df.to_csv(out_dir / "spatial_beta_calibration.csv", index=False)

    summary = {
        "Z": len(zone_ids),
        "mu_per_h_mean": constants_df["mu_per_h"].mean(),
        "mu_per_h_min": constants_df["mu_per_h"].min(),
        "mu_per_h_max": constants_df["mu_per_h"].max(),
        "L_Q_seconds_median": constants_df["L_Q_seconds"].median(),
        "L_Q_seconds_p95": constants_df["L_Q_seconds"].quantile(0.95),
        "rounding_residual_l1_frac_mean": constants_df["rounding_residual_l1_frac"].mean(),
        "allocator_basis_flip_rate": sensitivity_df["basis_flip"].mean(),
        "allocator_ratio_p95": sensitivity_df["allocator_ratio_drivers_per_rate"].quantile(0.95),
        "beta_wait_per_w1_median": spatial_df["beta_wait_per_w1"].median(),
        "beta_wait_per_w1_p95": spatial_df["beta_wait_per_w1"].quantile(0.95),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "theory_calibration_summary.csv", index=False)

    if not spatial_df.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.scatter(
            spatial_df["spatial_w1_s"],
            spatial_df["queue_wait_gap_s"],
            s=24,
            alpha=0.7,
        )
        ax.set_xlabel("controlled spatial shift W1^d (s)")
        ax.set_ylabel("stylized queue wait gap (s)")
        ax.set_title("Spatial Lipschitz diagnostic")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig("paper/figures/fig_spatial_beta_calibration.pdf")
        plt.close(fig)

    print("\nTheory constant calibration")
    print(f"  constants: {out_dir / 'theory_constant_calibration.csv'}")
    print(f"  allocator: {out_dir / 'allocator_sensitivity_diagnostics.csv'}")
    print(f"  spatial beta: {out_dir / 'spatial_beta_calibration.csv'}")
    print(f"  summary: {out_dir / 'theory_calibration_summary.csv'}")
    print(pd.DataFrame([summary]).round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/path2_theory_calibration")
    parser.add_argument("--max-zones", type=int, default=8)
    parser.add_argument("--h3-res", type=int, default=8)
    parser.add_argument("--horizon-hours", type=float, default=4.0)
    parser.add_argument("--fleet-ratio", type=float, default=0.50)
    parser.add_argument("--min-fleet", type=int, default=50)
    parser.add_argument("--rho-max", type=float, default=0.90)
    parser.add_argument("--allocator-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260508)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
