"""Optimization-based anticipatory fleet repositioning.

Uses the calibrated demand prior to forecast where demand will appear,
then solves a min-cost transportation LP to optimally redistribute idle
drivers toward predicted demand hotspots.

Two variants:
  - AnticipatoryReposition: LP-based (principled, optimal)
  - DemandFollowingReposition: greedy heuristic (fast, simple baseline)
"""

from __future__ import annotations

import numpy as np
import h3
from scipy.optimize import linprog

from src.config import get_config
from src.simulator.entities import (
    DriverStatus, RepositionInstruction, SimState,
)
from src.simulator.routing import RoutingClient, _haversine_m


class _HexGrid:
    """Shared H3 hex grid for spatial bucketing."""

    def __init__(self, h3_res: int | None = None, bbox: dict | None = None):
        cfg = get_config()
        bbox = bbox or cfg["data"]["manhattan_bbox"]
        self._h3_res = h3_res or cfg["rl"]["h3_resolution"]
        center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
        center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
        center_hex = h3.latlng_to_cell(center_lat, center_lon, self._h3_res)
        hexes = list(h3.grid_disk(center_hex, 12))
        self.hex_ids: list[str] = []
        self.hex_centers: dict[str, tuple[float, float]] = {}
        for hx in sorted(hexes):
            lat, lon = h3.cell_to_latlng(hx)
            if (bbox["min_lat"] <= lat <= bbox["max_lat"]
                    and bbox["min_lon"] <= lon <= bbox["max_lon"]):
                self.hex_ids.append(hx)
                self.hex_centers[hx] = (lon, lat)
        self.n = len(self.hex_ids)
        self.hex_to_idx = {h: i for i, h in enumerate(self.hex_ids)}

    def locate(self, lat: float, lon: float) -> int | None:
        hx = h3.latlng_to_cell(lat, lon, self._h3_res)
        return self.hex_to_idx.get(hx)


def _build_zone_fractions(prior, grid: _HexGrid) -> np.ndarray:
    """Pre-compute the fraction of demand falling in each hex zone
    based on the prior's spatial OD pool."""
    counts = np.zeros(grid.n, dtype=np.float64)
    if not prior._has_spatial:
        counts[:] = 1.0 / max(grid.n, 1)
        return counts
    for lat, lon in zip(prior._od_lats_pu_arr, prior._od_lons_pu_arr):
        idx = grid.locate(lat, lon)
        if idx is not None:
            counts[idx] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts[:] = 1.0 / max(grid.n, 1)
    return counts


class AnticipatoryReposition:
    """LP-based anticipatory fleet repositioning.

    At each reposition call:
      1. Forecast demand per zone for the next `lookahead_minutes` using
         the calibrated prior's rate profile and spatial distribution.
      2. Count idle drivers per zone.
      3. Identify surplus zones (idle > demand) and deficit zones (demand > idle).
      4. Solve a min-cost transportation LP to redistribute surplus drivers
         toward deficit zones, minimizing total travel time.
    """

    def __init__(
        self,
        prior,
        lookahead_minutes: float = 5.0,
        max_move_fraction: float = 0.50,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._prior = prior
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._zone_fractions = _build_zone_fractions(prior, self._grid)
        self._lookahead_s = lookahead_minutes * 60.0
        self._max_move_frac = max_move_fraction

        cfg = get_config()["regime"]
        self._bin_sec = cfg["bin_interval_minutes"] * 60

        self._cost_cache: dict[tuple[int, int], float] = {}

    def _forecast_demand(self, sim_time: float) -> np.ndarray:
        """Predict per-zone demand for the next lookahead window."""
        rate = self._prior.rate_profile
        current_bin = int(sim_time / self._bin_sec)
        lookahead_bins = max(1, int(self._lookahead_s / self._bin_sec))
        end_bin = min(current_bin + lookahead_bins, len(rate))
        total_expected = float(np.sum(rate[current_bin:end_bin]))
        return total_expected * self._zone_fractions

    def _zone_travel_time(
        self, src_idx: int, dst_idx: int, router: RoutingClient,
    ) -> float:
        key = (src_idx, dst_idx)
        if key not in self._cost_cache:
            s_lon, s_lat = self._grid.hex_centers[self._grid.hex_ids[src_idx]]
            d_lon, d_lat = self._grid.hex_centers[self._grid.hex_ids[dst_idx]]
            self._cost_cache[key] = router.travel_time(
                (s_lon, s_lat), (d_lon, d_lat)
            )
        return self._cost_cache[key]

    def reposition(
        self, state: SimState, router: RoutingClient,
    ) -> list[RepositionInstruction]:
        grid = self._grid
        demand_forecast = self._forecast_demand(state.time)

        idle_per_zone = np.zeros(grid.n, dtype=np.float64)
        idle_drivers_by_zone: dict[int, list[int]] = {}
        for d in state.drivers:
            if d.status != DriverStatus.IDLE:
                continue
            idx = grid.locate(d.lat, d.lon)
            if idx is not None:
                idle_per_zone[idx] += 1
                idle_drivers_by_zone.setdefault(idx, []).append(d.id)

        total_idle = int(idle_per_zone.sum())
        if total_idle == 0:
            return []

        surplus = np.maximum(idle_per_zone - demand_forecast, 0)
        deficit = np.maximum(demand_forecast - idle_per_zone, 0)

        src_zones = [i for i in range(grid.n) if surplus[i] >= 1.0]
        dst_zones = [i for i in range(grid.n) if deficit[i] >= 0.5]

        if not src_zones or not dst_zones:
            return []

        S, D = len(src_zones), len(dst_zones)

        # LP variables: x_{s,d} (flow) and y_d (demand served in zone d)
        # Minimize: alpha * sum c_{s,d} * x_{s,d} - sum y_d
        # (equivalently: maximize demand served minus weighted travel cost)
        n_x = S * D
        n_y = D
        n_vars = n_x + n_y

        max_travel = 0.0
        travel_costs = np.zeros(n_x)
        for si, s in enumerate(src_zones):
            for di, d in enumerate(dst_zones):
                tt = self._zone_travel_time(s, d, router)
                travel_costs[si * D + di] = tt
                max_travel = max(max_travel, tt)

        # alpha: trade-off weight. Each unit of demand served is worth
        # `max_travel` seconds of repositioning cost, so we always prefer
        # filling demand over saving travel time.
        alpha = 1.0 / max(max_travel * 2.0, 1.0)

        c_obj = np.zeros(n_vars)
        c_obj[:n_x] = alpha * travel_costs
        c_obj[n_x:] = -1.0  # reward for demand served

        A_ub_rows = []
        b_ub_vals = []

        # Supply constraints: sum_d x_{s,d} <= surplus_s
        for si, s in enumerate(src_zones):
            row = np.zeros(n_vars)
            for di in range(D):
                row[si * D + di] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(surplus[s])

        # Demand cap: y_d <= deficit_d
        for di, d in enumerate(dst_zones):
            row = np.zeros(n_vars)
            row[n_x + di] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(deficit[d])

        # Linking: y_d <= sum_s x_{s,d}
        for di in range(D):
            row = np.zeros(n_vars)
            for si in range(S):
                row[si * D + di] = -1.0
            row[n_x + di] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

        # Total move budget
        total_row = np.zeros(n_vars)
        total_row[:n_x] = 1.0
        A_ub_rows.append(total_row)
        max_moves = int(total_idle * self._max_move_frac)
        b_ub_vals.append(float(max_moves))

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds = [(0, None)] * n_vars

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            return []

        instructions: list[RepositionInstruction] = []
        x = result.x[:n_x]

        for si, s in enumerate(src_zones):
            drivers_here = list(idle_drivers_by_zone.get(s, []))
            moved_from_zone = 0
            for di, d in enumerate(dst_zones):
                n_move = int(round(x[si * D + di]))
                if n_move <= 0:
                    continue
                t_lon, t_lat = grid.hex_centers[grid.hex_ids[d]]
                for _ in range(min(n_move, len(drivers_here) - moved_from_zone)):
                    did = drivers_here[moved_from_zone]
                    instructions.append(RepositionInstruction(
                        driver_id=did,
                        target_lon=t_lon + np.random.normal(0, 0.001),
                        target_lat=t_lat + np.random.normal(0, 0.001),
                    ))
                    moved_from_zone += 1

        return instructions


class DemandFollowingReposition:
    """Greedy heuristic: move idle drivers from oversupplied zones to the
    nearest undersupplied zones, ranked by deficit magnitude."""

    def __init__(
        self,
        prior,
        lookahead_minutes: float = 15.0,
        max_move_fraction: float = 0.25,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._prior = prior
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._zone_fractions = _build_zone_fractions(prior, self._grid)
        self._lookahead_s = lookahead_minutes * 60.0
        self._max_move_frac = max_move_fraction

        cfg = get_config()["regime"]
        self._bin_sec = cfg["bin_interval_minutes"] * 60

    def _forecast_demand(self, sim_time: float) -> np.ndarray:
        rate = self._prior.rate_profile
        current_bin = int(sim_time / self._bin_sec)
        lookahead_bins = max(1, int(self._lookahead_s / self._bin_sec))
        end_bin = min(current_bin + lookahead_bins, len(rate))
        total_expected = float(np.sum(rate[current_bin:end_bin]))
        return total_expected * self._zone_fractions

    def reposition(
        self, state: SimState, router: RoutingClient,
    ) -> list[RepositionInstruction]:
        grid = self._grid
        demand_forecast = self._forecast_demand(state.time)

        idle_per_zone = np.zeros(grid.n, dtype=np.float64)
        idle_drivers_by_zone: dict[int, list] = {}
        driver_map = {d.id: d for d in state.drivers}
        for d in state.drivers:
            if d.status != DriverStatus.IDLE:
                continue
            idx = grid.locate(d.lat, d.lon)
            if idx is not None:
                idle_per_zone[idx] += 1
                idle_drivers_by_zone.setdefault(idx, []).append(d.id)

        total_idle = int(idle_per_zone.sum())
        if total_idle == 0:
            return []

        surplus = np.maximum(idle_per_zone - demand_forecast, 0)
        deficit = demand_forecast - idle_per_zone

        deficit_zones = sorted(
            [(i, deficit[i]) for i in range(grid.n) if deficit[i] > 0.5],
            key=lambda x: -x[1],
        )
        if not deficit_zones:
            return []

        instructions: list[RepositionInstruction] = []
        max_moves = int(total_idle * self._max_move_frac)
        moved = 0

        for dz_idx, _ in deficit_zones:
            if moved >= max_moves:
                break
            t_lon, t_lat = grid.hex_centers[grid.hex_ids[dz_idx]]

            surplus_zones = sorted(
                [(i, surplus[i]) for i in range(grid.n) if surplus[i] >= 1.0],
                key=lambda x: _haversine_m(
                    grid.hex_centers[grid.hex_ids[x[0]]],
                    (t_lon, t_lat),
                ),
            )

            for sz_idx, sz_surplus in surplus_zones:
                if moved >= max_moves or surplus[sz_idx] < 1.0:
                    break
                drivers_here = idle_drivers_by_zone.get(sz_idx, [])
                n_move = min(
                    int(surplus[sz_idx]),
                    max(1, int(deficit[dz_idx])),
                    len(drivers_here),
                )
                for k in range(n_move):
                    if moved >= max_moves:
                        break
                    did = drivers_here.pop(0)
                    instructions.append(RepositionInstruction(
                        driver_id=did,
                        target_lon=t_lon + np.random.normal(0, 0.001),
                        target_lat=t_lat + np.random.normal(0, 0.001),
                    ))
                    surplus[sz_idx] -= 1
                    deficit[dz_idx] -= 1
                    moved += 1
                if deficit[dz_idx] <= 0:
                    break

        return instructions
