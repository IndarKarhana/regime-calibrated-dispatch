"""Retrieval-robust fleet repositioning controllers."""

from __future__ import annotations

import time
from statistics import NormalDist

import numpy as np
from scipy.optimize import linprog

from src.config import get_config
from src.policies.anticipatory import _HexGrid
from src.policies.external_baselines import (
    _idle_state_by_zone,
    _normalize_scores,
    _record_zone_fractions,
)
from src.regime.store import RegimeRecord
from src.simulator.entities import RepositionInstruction, SimState
from src.simulator.routing import RoutingClient


class RetrievalRobustCVaRReposition:
    """Retrieval-conditioned CVaR shortage repositioning.

    Unlike share-target controllers that collapse uncertainty into one target
    share, this controller keeps retrieved regimes as separate scenarios and
    solves a CVaR-shortage LP over the post-reposition idle allocation.
    """

    def __init__(
        self,
        matched_records: list[RegimeRecord],
        match_scores: list[float],
        *,
        lookahead_minutes: float = 15.0,
        alpha: float = 0.80,
        risk_weight: float = 1.0,
        mean_weight: float = 0.25,
        move_cost_multiplier: float = 1.0,
        zone_buffer: np.ndarray | None = None,
        max_move_fraction: float = 0.50,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._records = matched_records
        self._weights = _normalize_scores(match_scores, len(matched_records))
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._scenario_shares = (
            np.vstack([
                _record_zone_fractions(record, self._grid)
                for record in matched_records
            ])
            if matched_records
            else np.ones((1, self._grid.n), dtype=float) / max(self._grid.n, 1)
        )
        self._lookahead_s = lookahead_minutes * 60.0
        self._alpha = float(np.clip(alpha, 0.50, 0.98))
        self._risk_weight = max(float(risk_weight), 0.0)
        self._mean_weight = max(float(mean_weight), 0.0)
        self._move_cost_multiplier = max(float(move_cost_multiplier), 0.0)
        self._zone_buffer = self._coerce_zone_buffer(zone_buffer)
        self._max_move_fraction = max_move_fraction
        self._cost_cache: dict[tuple[int, int], float] = {}
        self.solve_times_s: list[float] = []
        self.solve_status_counts: dict[str, int] = {}
        self.dominance_delta_history: list[float] = []
        self.dominance_holds_history: list[bool] = []
        self.mixture_shortage_history: list[float] = []
        self.dcr_shortage_term_history: list[float] = []

        cfg = get_config()["regime"]
        self._bin_sec = cfg["bin_interval_minutes"] * 60

    def _mark_status(self, status: str) -> None:
        self.solve_status_counts[status] = self.solve_status_counts.get(status, 0) + 1

    def _scenario_zone_demand(self, sim_time: float) -> np.ndarray:
        current_bin = int(sim_time / self._bin_sec)
        lookahead_bins = max(1, int(self._lookahead_s / self._bin_sec))
        end_bin = current_bin + lookahead_bins

        demands = np.zeros_like(self._scenario_shares, dtype=float)
        for ri, record in enumerate(self._records):
            series = record.demand_series
            if len(series) == 0 or current_bin >= len(series):
                total = 0.0
            else:
                total = float(np.sum(series[current_bin:min(end_bin, len(series))]))
            demands[ri, :] = total * self._scenario_shares[ri, :]
        return demands

    def _scenario_targets(self, sim_time: float, total_idle: int) -> np.ndarray:
        demand = self._scenario_zone_demand(sim_time)
        if demand.size == 0 or float(demand.sum()) <= 1e-9:
            shares = self._scenario_shares
        else:
            totals = np.maximum(demand.sum(axis=1, keepdims=True), 1e-9)
            shares = demand / totals
        return float(total_idle) * shares

    def _coerce_zone_buffer(self, zone_buffer: np.ndarray | None) -> np.ndarray:
        if zone_buffer is None:
            return np.zeros(self._grid.n, dtype=float)
        arr = np.asarray(zone_buffer, dtype=float)
        if arr.size != self._grid.n:
            return np.zeros(self._grid.n, dtype=float)
        return np.maximum(arr, 0.0)

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            self._mark_status("no_idle_supply")
            return []

        targets = self._scenario_targets(state.time, total_idle)
        if float(self._zone_buffer.sum()) > 0:
            targets = targets + float(total_idle) * self._zone_buffer[None, :]
        max_moves = int(total_idle * self._max_move_fraction)
        if max_moves <= 0:
            self._mark_status("zero_move_budget")
            return []

        t0 = time.perf_counter()
        instructions = self._solve_and_round(
            idle_per_zone,
            idle_drivers,
            targets,
            max_moves,
            router,
            np.random.default_rng(int(state.time) + 223),
        )
        self.solve_times_s.append(time.perf_counter() - t0)
        return instructions

    def _solve_and_round(
        self,
        idle_per_zone: np.ndarray,
        idle_drivers: dict[int, list[int]],
        targets: np.ndarray,
        max_moves: int,
        router: RoutingClient,
        rng: np.random.Generator,
    ) -> list[RepositionInstruction]:
        grid = self._grid
        n_zones = grid.n
        n_scenarios = targets.shape[0]
        if n_zones <= 0 or n_scenarios <= 0:
            self._mark_status("empty_zone_or_scenario_set")
            return []

        mean_target = np.average(targets, axis=0, weights=self._weights)
        z_value = NormalDist().inv_cdf(self._alpha)
        scenario_var = np.average(
            (targets - mean_target) ** 2,
            axis=0,
            weights=self._weights,
        )
        screening_target = mean_target + z_value * np.sqrt(np.maximum(scenario_var, 0.0))

        src_zones = [int(i) for i in range(n_zones) if idle_per_zone[i] >= 1.0]
        dst_zones = [
            int(i)
            for i in np.argsort(-(screening_target - idle_per_zone))
            if screening_target[i] - idle_per_zone[i] >= 0.25
        ]
        if not src_zones or not dst_zones:
            self._mark_status("no_source_or_destination_zones")
            return []

        s_count = len(src_zones)
        d_count = len(dst_zones)
        n_x = s_count * d_count
        offset_a = n_x
        offset_u = offset_a + n_zones
        offset_ell = offset_u + n_scenarios * n_zones
        offset_tau = offset_ell + n_scenarios
        offset_v = offset_tau + 1
        n_vars = offset_v + n_scenarios

        costs = np.zeros(n_x, dtype=float)
        max_travel = 0.0
        for si, src in enumerate(src_zones):
            for di, dst in enumerate(dst_zones):
                if src == dst:
                    tt = 0.0
                else:
                    key = (src, dst)
                    if key not in self._cost_cache:
                        self._cost_cache[key] = router.travel_time(
                            grid.hex_centers[grid.hex_ids[src]],
                            grid.hex_centers[grid.hex_ids[dst]],
                        )
                    tt = self._cost_cache[key]
                costs[si * d_count + di] = tt
                max_travel = max(max_travel, tt)

        objective = np.zeros(n_vars, dtype=float)
        move_weight = 1.0 / max(max_travel * 2.0, 1.0)
        objective[:n_x] = self._move_cost_multiplier * move_weight * costs
        objective[offset_ell:offset_ell + n_scenarios] = (
            self._mean_weight * self._weights
        )
        objective[offset_tau] = self._risk_weight
        cvar_scale = self._risk_weight / max(1.0 - self._alpha, 1e-6)
        objective[offset_v:offset_v + n_scenarios] = cvar_scale * self._weights

        a_eq_rows = []
        b_eq_vals = []
        for z in range(n_zones):
            row = np.zeros(n_vars, dtype=float)
            row[offset_a + z] = 1.0
            for si, src in enumerate(src_zones):
                for di, dst in enumerate(dst_zones):
                    x_idx = si * d_count + di
                    if src == z:
                        row[x_idx] += 1.0
                    if dst == z:
                        row[x_idx] -= 1.0
            a_eq_rows.append(row)
            b_eq_vals.append(float(idle_per_zone[z]))

        a_ub_rows = []
        b_ub_vals = []
        for si, src in enumerate(src_zones):
            row = np.zeros(n_vars, dtype=float)
            row[si * d_count:(si + 1) * d_count] = 1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(idle_per_zone[src]))

        move_row = np.zeros(n_vars, dtype=float)
        move_row[:n_x] = 1.0
        a_ub_rows.append(move_row)
        b_ub_vals.append(float(max_moves))

        for ri in range(n_scenarios):
            for z in range(n_zones):
                row = np.zeros(n_vars, dtype=float)
                row[offset_a + z] = -1.0
                row[offset_u + ri * n_zones + z] = -1.0
                a_ub_rows.append(row)
                b_ub_vals.append(-float(targets[ri, z]))

        for ri in range(n_scenarios):
            row = np.zeros(n_vars, dtype=float)
            start = offset_u + ri * n_zones
            row[start:start + n_zones] = 1.0
            row[offset_ell + ri] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(0.0)

        for ri in range(n_scenarios):
            row = np.zeros(n_vars, dtype=float)
            row[offset_ell + ri] = 1.0
            row[offset_tau] = -1.0
            row[offset_v + ri] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(0.0)

        result = linprog(
            objective,
            A_ub=np.vstack(a_ub_rows),
            b_ub=np.asarray(b_ub_vals, dtype=float),
            A_eq=np.vstack(a_eq_rows),
            b_eq=np.asarray(b_eq_vals, dtype=float),
            bounds=[(0.0, None)] * n_vars,
            method="highs",
        )
        if not result.success:
            self._mark_status(f"linprog_{result.status}")
            return []

        allocation = np.asarray(result.x[offset_a:offset_a + n_zones], dtype=float)
        buffer_counts = float(idle_per_zone.sum()) * self._zone_buffer[None, :]
        unbuffered_targets = np.maximum(targets - buffer_counts, 0.0)
        mixture_target = np.average(unbuffered_targets, axis=0, weights=self._weights)
        mixture_shortage = float(np.maximum(mixture_target - allocation, 0.0).sum())
        scenario_shortages = np.maximum(targets - allocation[None, :], 0.0).sum(axis=1)
        mean_shortage = float(np.dot(self._weights, scenario_shortages))
        cvar_shortage = _weighted_cvar(scenario_shortages, self._weights, self._alpha)
        dcr_shortage_term = (
            self._mean_weight * mean_shortage
            + self._risk_weight * cvar_shortage
        )
        dominance_delta = mixture_shortage - dcr_shortage_term
        self.mixture_shortage_history.append(mixture_shortage)
        self.dcr_shortage_term_history.append(float(dcr_shortage_term))
        self.dominance_delta_history.append(float(dominance_delta))
        self.dominance_holds_history.append(bool(dominance_delta <= 1e-9))

        instructions = _round_flow_to_instructions(
            grid,
            idle_drivers,
            src_zones,
            dst_zones,
            result.x[:n_x],
            d_count,
            rng,
        )
        self._mark_status("success_with_moves" if instructions else "success_zero_rounded_moves")
        return instructions


class RetrievalRobustBudgetReposition(RetrievalRobustCVaRReposition):
    """Minimum movement subject to a robust shortage-risk reduction."""

    def __init__(
        self,
        matched_records: list[RegimeRecord],
        match_scores: list[float],
        *,
        lookahead_minutes: float = 15.0,
        alpha: float = 0.80,
        shortage_reduction: float = 0.25,
        mean_slack_weight: float = 1e-3,
        zone_buffer: np.ndarray | None = None,
        max_move_fraction: float = 0.50,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        super().__init__(
            matched_records,
            match_scores,
            lookahead_minutes=lookahead_minutes,
            alpha=alpha,
            risk_weight=1.0,
            mean_weight=0.0,
            move_cost_multiplier=1.0,
            zone_buffer=zone_buffer,
            max_move_fraction=max_move_fraction,
            h3_res=h3_res,
            bbox=bbox,
        )
        self._shortage_reduction = float(np.clip(shortage_reduction, 0.0, 0.95))
        self._mean_slack_weight = max(float(mean_slack_weight), 0.0)
        self.cvar_budget_history: list[float] = []
        self.baseline_cvar_history: list[float] = []

    def _solve_and_round(
        self,
        idle_per_zone: np.ndarray,
        idle_drivers: dict[int, list[int]],
        targets: np.ndarray,
        max_moves: int,
        router: RoutingClient,
        rng: np.random.Generator,
    ) -> list[RepositionInstruction]:
        grid = self._grid
        n_zones = grid.n
        n_scenarios = targets.shape[0]
        if n_zones <= 0 or n_scenarios <= 0:
            self._mark_status("empty_zone_or_scenario_set")
            return []

        baseline_shortage = np.maximum(
            targets - idle_per_zone[None, :],
            0.0,
        ).sum(axis=1)
        baseline_cvar = _weighted_cvar(baseline_shortage, self._weights, self._alpha)
        self.baseline_cvar_history.append(float(baseline_cvar))
        if baseline_cvar <= 1e-9:
            self._mark_status("zero_baseline_cvar")
            return []
        cvar_budget = (1.0 - self._shortage_reduction) * baseline_cvar
        self.cvar_budget_history.append(float(cvar_budget))

        mean_target = np.average(targets, axis=0, weights=self._weights)
        src_zones = [int(i) for i in range(n_zones) if idle_per_zone[i] >= 1.0]
        dst_zones = [
            int(i)
            for i in np.argsort(-(mean_target - idle_per_zone))
            if mean_target[i] - idle_per_zone[i] >= 0.25
        ]
        if not src_zones or not dst_zones:
            self._mark_status("no_source_or_destination_zones")
            return []

        s_count = len(src_zones)
        d_count = len(dst_zones)
        n_x = s_count * d_count
        offset_a = n_x
        offset_u = offset_a + n_zones
        offset_ell = offset_u + n_scenarios * n_zones
        offset_tau = offset_ell + n_scenarios
        offset_v = offset_tau + 1
        n_vars = offset_v + n_scenarios

        costs = np.zeros(n_x, dtype=float)
        max_travel = 0.0
        for si, src in enumerate(src_zones):
            for di, dst in enumerate(dst_zones):
                key = (src, dst)
                if key not in self._cost_cache:
                    self._cost_cache[key] = router.travel_time(
                        grid.hex_centers[grid.hex_ids[src]],
                        grid.hex_centers[grid.hex_ids[dst]],
                    )
                tt = self._cost_cache[key]
                costs[si * d_count + di] = tt
                max_travel = max(max_travel, tt)

        objective = np.zeros(n_vars, dtype=float)
        objective[:n_x] = costs / max(max_travel * 2.0, 1.0)
        objective[offset_ell:offset_ell + n_scenarios] = (
            self._mean_slack_weight * self._weights
        )

        a_eq_rows = []
        b_eq_vals = []
        for z in range(n_zones):
            row = np.zeros(n_vars, dtype=float)
            row[offset_a + z] = 1.0
            for si, src in enumerate(src_zones):
                for di, dst in enumerate(dst_zones):
                    x_idx = si * d_count + di
                    if src == z:
                        row[x_idx] += 1.0
                    if dst == z:
                        row[x_idx] -= 1.0
            a_eq_rows.append(row)
            b_eq_vals.append(float(idle_per_zone[z]))

        a_ub_rows = []
        b_ub_vals = []
        for si, src in enumerate(src_zones):
            row = np.zeros(n_vars, dtype=float)
            row[si * d_count:(si + 1) * d_count] = 1.0
            a_ub_rows.append(row)
            b_ub_vals.append(float(idle_per_zone[src]))

        move_row = np.zeros(n_vars, dtype=float)
        move_row[:n_x] = 1.0
        a_ub_rows.append(move_row)
        b_ub_vals.append(float(max_moves))

        for ri in range(n_scenarios):
            for z in range(n_zones):
                row = np.zeros(n_vars, dtype=float)
                row[offset_a + z] = -1.0
                row[offset_u + ri * n_zones + z] = -1.0
                a_ub_rows.append(row)
                b_ub_vals.append(-float(targets[ri, z]))

        for ri in range(n_scenarios):
            row = np.zeros(n_vars, dtype=float)
            start = offset_u + ri * n_zones
            row[start:start + n_zones] = 1.0
            row[offset_ell + ri] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(0.0)

        for ri in range(n_scenarios):
            row = np.zeros(n_vars, dtype=float)
            row[offset_ell + ri] = 1.0
            row[offset_tau] = -1.0
            row[offset_v + ri] = -1.0
            a_ub_rows.append(row)
            b_ub_vals.append(0.0)

        cvar_row = np.zeros(n_vars, dtype=float)
        cvar_row[offset_tau] = 1.0
        cvar_scale = 1.0 / max(1.0 - self._alpha, 1e-6)
        cvar_row[offset_v:offset_v + n_scenarios] = cvar_scale * self._weights
        a_ub_rows.append(cvar_row)
        b_ub_vals.append(float(cvar_budget))

        result = linprog(
            objective,
            A_ub=np.vstack(a_ub_rows),
            b_ub=np.asarray(b_ub_vals, dtype=float),
            A_eq=np.vstack(a_eq_rows),
            b_eq=np.asarray(b_eq_vals, dtype=float),
            bounds=[(0.0, None)] * n_vars,
            method="highs",
        )
        if not result.success:
            self._mark_status(f"linprog_{result.status}")
            return []

        instructions = _round_flow_to_instructions(
            grid,
            idle_drivers,
            src_zones,
            dst_zones,
            result.x[:n_x],
            d_count,
            rng,
        )
        self._mark_status("success_with_moves" if instructions else "success_zero_rounded_moves")
        return instructions


def _round_flow_to_instructions(
    grid: _HexGrid,
    idle_drivers: dict[int, list[int]],
    src_zones: list[int],
    dst_zones: list[int],
    flow: np.ndarray,
    d_count: int,
    rng: np.random.Generator,
) -> list[RepositionInstruction]:
    instructions: list[RepositionInstruction] = []
    for si, src in enumerate(src_zones):
        drivers_here = idle_drivers.get(src, [])
        moved_from_src = 0
        for di, dst in enumerate(dst_zones):
            n_move = int(round(flow[si * d_count + di]))
            if n_move <= 0 or src == dst:
                continue
            dst_lon, dst_lat = grid.hex_centers[grid.hex_ids[dst]]
            available = len(drivers_here) - moved_from_src
            for _ in range(min(n_move, available)):
                driver_id = drivers_here[moved_from_src]
                instructions.append(RepositionInstruction(
                    driver_id=driver_id,
                    target_lon=dst_lon + rng.normal(0.0, 0.001),
                    target_lat=dst_lat + rng.normal(0.0, 0.001),
                ))
                moved_from_src += 1
    return instructions


def _weighted_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.0
    if weights.size != values.size or float(weights.sum()) <= 1e-12:
        weights = np.ones(values.size, dtype=float) / values.size
    else:
        weights = weights / float(weights.sum())
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cum = np.cumsum(sorted_weights)
    var_idx = int(np.searchsorted(cum, alpha, side="left"))
    var_idx = min(max(var_idx, 0), len(sorted_values) - 1)
    var = sorted_values[var_idx]
    excess = np.maximum(values - var, 0.0)
    return float(var + np.dot(weights, excess) / max(1.0 - alpha, 1e-6))
