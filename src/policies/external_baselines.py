"""External-style repositioning baselines re-run inside our simulator.

These are lightweight, documented implementations of common ideas from the
fleet-repositioning literature rather than exact reproduction packages.
They share the simulator's ``reposition(state, router)`` interface.
"""

from __future__ import annotations

from statistics import NormalDist
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from src.policies.anticipatory import _build_zone_fractions, _HexGrid
from src.regime.store import RegimeRecord
from src.simulator.entities import DriverStatus, RepositionInstruction, SimState
from src.simulator.routing import RoutingClient, _haversine_m


class WenStyleHistoricalRebalancing:
    """Fluid rebalancing toward historical demand shares.

    Inspired by Wen et al.-style proactive rebalancing: periodically move idle
    vehicles so the idle distribution tracks historical demand shares. It uses
    no current/future demand oracle and no learned gate.
    """

    def __init__(
        self,
        prior,
        max_move_fraction: float = 0.30,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._zone_fractions = _build_zone_fractions(prior, self._grid)
        self._max_move_fraction = max_move_fraction

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            return []

        target_idle = total_idle * self._zone_fractions
        surplus = np.maximum(idle_per_zone - target_idle, 0.0)
        deficit = np.maximum(target_idle - idle_per_zone, 0.0)
        return _greedy_move_surplus_to_deficit(
            self._grid,
            idle_drivers,
            surplus,
            deficit,
            max_moves=int(total_idle * self._max_move_fraction),
            rng=np.random.default_rng(int(state.time) + 17),
        )


class ForecastShareLPReposition:
    """LP controller targeting a forecast demand-share distribution.

    This is the bridge between Wen-style share balancing and our LP controller:
    first convert the forecast into a stable target distribution for currently
    movable idle supply, then solve a travel-cost transportation LP from
    surplus zones to deficit zones.
    """

    def __init__(
        self,
        prior,
        max_move_fraction: float = 0.30,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._zone_fractions = _build_zone_fractions(prior, self._grid)
        self._max_move_fraction = max_move_fraction
        self._cost_cache: dict[tuple[int, int], float] = {}
        self.solve_times_s: list[float] = []

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            return []

        target_idle = total_idle * self._zone_fractions
        surplus = np.maximum(idle_per_zone - target_idle, 0.0)
        deficit = np.maximum(target_idle - idle_per_zone, 0.0)
        t0 = time.perf_counter()
        instructions = _transport_lp_reposition(
            self._grid,
            idle_drivers,
            surplus,
            deficit,
            max_moves=int(total_idle * self._max_move_fraction),
            router=router,
            cost_cache=self._cost_cache,
            rng=np.random.default_rng(int(state.time) + 43),
        )
        self.solve_times_s.append(time.perf_counter() - t0)
        return instructions


class ScenarioChanceMPCReposition:
    """Scenario-based uncertainty-aware MPC baseline.

    The controller treats retrieved historical regimes as empirical demand
    scenarios. At each reposition epoch it computes per-zone demand over the
    lookahead window for every retrieved regime, takes a weighted upper
    quantile by zone, and solves the same transportation LP used by the
    share-target controller. This is a simulator-native analog of
    chance-constrained MPC: it propagates forecast uncertainty into the control
    target without peeking at future replay requests.
    """

    def __init__(
        self,
        matched_records: list[RegimeRecord],
        match_scores: list[float],
        *,
        lookahead_minutes: float = 15.0,
        quantile: float = 0.80,
        risk_weight: float = 0.70,
        max_move_fraction: float = 0.50,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._records = matched_records
        self._weights = _normalize_scores(match_scores, len(matched_records))
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._scenario_shares = np.vstack([
            _record_zone_fractions(record, self._grid)
            for record in matched_records
        ]) if matched_records else np.ones((1, self._grid.n), dtype=float) / max(self._grid.n, 1)
        self._lookahead_s = lookahead_minutes * 60.0
        self._quantile = float(np.clip(quantile, 0.50, 0.99))
        self._risk_weight = float(np.clip(risk_weight, 0.0, 1.0))
        self._max_move_fraction = max_move_fraction
        self._cost_cache: dict[tuple[int, int], float] = {}
        self.solve_times_s: list[float] = []

        from src.config import get_config

        cfg = get_config()["regime"]
        self._bin_sec = cfg["bin_interval_minutes"] * 60

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

    def _target_share(self, sim_time: float) -> np.ndarray:
        scenario_demand = self._scenario_zone_demand(sim_time)
        if scenario_demand.size == 0 or float(scenario_demand.sum()) <= 1e-9:
            target = np.average(self._scenario_shares, axis=0, weights=self._weights)
        else:
            mean_target = np.average(scenario_demand, axis=0, weights=self._weights)
            quantile_target = np.array([
                _weighted_quantile(scenario_demand[:, z], self._weights, self._quantile)
                for z in range(self._grid.n)
            ])
            target = (
                (1.0 - self._risk_weight) * mean_target
                + self._risk_weight * quantile_target
            )
        total = float(target.sum())
        if total <= 1e-9:
            return np.ones(self._grid.n, dtype=float) / max(self._grid.n, 1)
        return target / total

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            return []

        target_idle = total_idle * self._target_share(state.time)
        surplus = np.maximum(idle_per_zone - target_idle, 0.0)
        deficit = np.maximum(target_idle - idle_per_zone, 0.0)
        t0 = time.perf_counter()
        instructions = _transport_lp_reposition(
            self._grid,
            idle_drivers,
            surplus,
            deficit,
            max_moves=int(total_idle * self._max_move_fraction),
            router=router,
            cost_cache=self._cost_cache,
            rng=np.random.default_rng(int(state.time) + 137),
        )
        self.solve_times_s.append(time.perf_counter() - t0)
        return instructions


class GPRChanceMPCReposition:
    """Reduced GPR-style chance-constrained MPC comparator.

    This baseline is intentionally lighter than a full reproduction of
    GPR-CCMPC papers. It uses retrieved regimes to form per-zone lookahead
    demand scenarios, estimates a spatially smoothed uncertainty surface with a
    Gaussian process over zone centers, and solves the same share-target
    transportation LP against a mean-plus-risk-buffer target.
    """

    def __init__(
        self,
        matched_records: list[RegimeRecord],
        match_scores: list[float],
        *,
        lookahead_minutes: float = 15.0,
        quantile: float = 0.90,
        risk_weight: float = 0.80,
        max_move_fraction: float = 0.50,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._records = matched_records
        self._weights = _normalize_scores(match_scores, len(matched_records))
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._scenario_shares = np.vstack([
            _record_zone_fractions(record, self._grid)
            for record in matched_records
        ]) if matched_records else np.ones((1, self._grid.n), dtype=float) / max(self._grid.n, 1)
        self._lookahead_s = lookahead_minutes * 60.0
        self._quantile = float(np.clip(quantile, 0.50, 0.99))
        self._risk_weight = float(np.clip(risk_weight, 0.0, 1.0))
        self._z_value = NormalDist().inv_cdf(self._quantile)
        self._max_move_fraction = max_move_fraction
        self._cost_cache: dict[tuple[int, int], float] = {}
        self.solve_times_s: list[float] = []

        from src.config import get_config

        cfg = get_config()["regime"]
        self._bin_sec = cfg["bin_interval_minutes"] * 60
        self._zone_sigma = self._fit_spatial_uncertainty_surface()

    def _scenario_zone_demand_for_bin(self, current_bin: int) -> np.ndarray:
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

    def _fit_spatial_uncertainty_surface(self) -> np.ndarray:
        if self._grid.n <= 0:
            return np.ones(1, dtype=float)
        if not self._records:
            return np.ones(self._grid.n, dtype=float)

        max_bins = max((len(r.demand_series) for r in self._records), default=0)
        if max_bins <= 0:
            return np.ones(self._grid.n, dtype=float)

        variances = []
        for current_bin in range(max_bins):
            demands = self._scenario_zone_demand_for_bin(current_bin)
            if float(demands.sum()) <= 1e-9:
                continue
            mean = np.average(demands, axis=0, weights=self._weights)
            var = np.average((demands - mean) ** 2, axis=0, weights=self._weights)
            variances.append(var)

        if not variances:
            return np.ones(self._grid.n, dtype=float)
        empirical_sigma = np.sqrt(np.mean(np.vstack(variances), axis=0))
        empirical_sigma = np.maximum(empirical_sigma, 0.05)

        coords = np.array([
            self._grid.hex_centers[hex_id]
            for hex_id in self._grid.hex_ids
        ], dtype=float)
        coords = _standardize_columns(coords)
        y = np.log1p(empirical_sigma)
        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(length_scale=1.0, length_scale_bounds=(0.25, 4.0))
            + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 0.50))
        )
        try:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-4,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=0,
            )
            gp.fit(coords, y)
            pred = np.expm1(gp.predict(coords))
            return np.maximum(pred, 0.05)
        except Exception:
            return empirical_sigma

    def _target_share(self, sim_time: float) -> np.ndarray:
        current_bin = int(sim_time / self._bin_sec)
        scenario_demand = self._scenario_zone_demand_for_bin(current_bin)
        if scenario_demand.size == 0 or float(scenario_demand.sum()) <= 1e-9:
            target = np.average(self._scenario_shares, axis=0, weights=self._weights)
        else:
            mean_target = np.average(scenario_demand, axis=0, weights=self._weights)
            scenario_var = np.average(
                (scenario_demand - mean_target) ** 2,
                axis=0,
                weights=self._weights,
            )
            sigma = np.sqrt(np.maximum(scenario_var, 0.0) + self._zone_sigma**2)
            chance_target = np.maximum(mean_target + self._z_value * sigma, 0.0)
            target = (
                (1.0 - self._risk_weight) * mean_target
                + self._risk_weight * chance_target
            )
        total = float(target.sum())
        if total <= 1e-9:
            return np.ones(self._grid.n, dtype=float) / max(self._grid.n, 1)
        return target / total

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            return []

        target_idle = total_idle * self._target_share(state.time)
        surplus = np.maximum(idle_per_zone - target_idle, 0.0)
        deficit = np.maximum(target_idle - idle_per_zone, 0.0)
        t0 = time.perf_counter()
        instructions = _transport_lp_reposition(
            self._grid,
            idle_drivers,
            surplus,
            deficit,
            max_moves=int(total_idle * self._max_move_fraction),
            router=router,
            cost_cache=self._cost_cache,
            rng=np.random.default_rng(int(state.time) + 181),
        )
        self.solve_times_s.append(time.perf_counter() - t0)
        return instructions


class ContextualDQNReposition:
    """Lightweight contextual-DQN style rebalancing baseline.

    Lin et al.-style DQN policies score local move actions from contextual
    features. Reproducing their full training stack is intentionally time-boxed
    in Path 2, so this class provides the simulator-facing policy hook with a
    small linear Q head. A checkpoint can override the default head; without one
    the policy is a conservative shortage-following baseline.
    """

    def __init__(
        self,
        prior,
        checkpoint: str | None = None,
        max_move_fraction: float = 0.25,
        temperature: float = 0.20,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._zone_fractions = _build_zone_fractions(prior, self._grid)
        self._max_move_fraction = max_move_fraction
        self._temperature = max(temperature, 1e-3)
        # Features: forecast share, idle share, pending share, contextual
        # shortage, hour sin, hour cos. Positive weights favor zones with
        # expected and observed shortage.
        self._q_weights = np.array([1.6, -1.1, 0.7, 2.2, 0.05, 0.05], dtype=float)
        if checkpoint is not None:
            with np.load(checkpoint) as data:
                self._q_weights = data["q_weights"].astype(float)

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            return []

        idle_share = idle_per_zone / max(float(total_idle), 1.0)
        pending = np.zeros(self._grid.n, dtype=float)
        for request in state.pending_requests:
            idx = self._grid.locate(request.pickup_lat, request.pickup_lon)
            if idx is not None:
                pending[idx] += 1.0
        pending_share = pending / max(float(pending.sum()), 1.0)

        hour = (state.time / 3600.0) % 24.0
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        shortage = self._zone_fractions + pending_share - idle_share
        features = np.column_stack([
            self._zone_fractions,
            idle_share,
            pending_share,
            shortage,
            np.full(self._grid.n, hour_sin),
            np.full(self._grid.n, hour_cos),
        ])
        q_values = features @ self._q_weights
        q_values -= float(np.max(q_values))
        target_share = np.exp(q_values / self._temperature)
        target_share /= max(float(target_share.sum()), 1e-9)

        target_idle = total_idle * target_share
        surplus = np.maximum(idle_per_zone - target_idle, 0.0)
        deficit = np.maximum(target_idle - idle_per_zone, 0.0)
        return _greedy_move_surplus_to_deficit(
            self._grid,
            idle_drivers,
            surplus,
            deficit,
            max_moves=int(total_idle * self._max_move_fraction),
            rng=np.random.default_rng(int(state.time) + 71),
        )


class OracleMPCReposition:
    """Rolling-horizon upper-bound baseline using realized future pickups.

    At each reposition epoch, this baseline peeks at replay requests within the
    lookahead window and moves idle drivers toward zones where actual future
    demand exceeds current idle supply. It is intentionally optimistic.
    """

    def __init__(
        self,
        replay_trips: pd.DataFrame,
        *,
        start_time: pd.Timestamp,
        lookahead_minutes: float = 5.0,
        max_move_fraction: float = 0.50,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._lookahead_s = lookahead_minutes * 60.0
        self._max_move_fraction = max_move_fraction
        df = replay_trips.sort_values("pickup_datetime").copy()
        self._times = (
            df["pickup_datetime"] - start_time
        ).dt.total_seconds().to_numpy(dtype=float)
        self._pickup_lons = df["pickup_longitude"].to_numpy(dtype=float)
        self._pickup_lats = df["pickup_latitude"].to_numpy(dtype=float)
        self._cost_cache: dict[tuple[int, int], float] = {}
        self.solve_times_s: list[float] = []

    def _future_demand(self, sim_time: float) -> np.ndarray:
        end_time = sim_time + self._lookahead_s
        mask = (self._times > sim_time) & (self._times <= end_time)
        demand = np.zeros(self._grid.n, dtype=float)
        for lon, lat in zip(self._pickup_lons[mask], self._pickup_lats[mask]):
            idx = self._grid.locate(float(lat), float(lon))
            if idx is not None:
                demand[idx] += 1.0
        return demand

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            return []

        demand = self._future_demand(state.time)
        if float(demand.sum()) <= 0:
            return []
        target_idle = total_idle * demand / float(demand.sum())
        surplus = np.maximum(idle_per_zone - target_idle, 0.0)
        deficit = np.maximum(target_idle - idle_per_zone, 0.0)
        t0 = time.perf_counter()
        instructions = _transport_lp_reposition(
            self._grid,
            idle_drivers,
            surplus,
            deficit,
            max_moves=int(total_idle * self._max_move_fraction),
            router=router,
            cost_cache=self._cost_cache,
            rng=np.random.default_rng(int(state.time) + 101),
        )
        self.solve_times_s.append(time.perf_counter() - t0)
        return instructions


def _idle_state_by_zone(
    state: SimState,
    grid: _HexGrid,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    idle_per_zone = np.zeros(grid.n, dtype=float)
    idle_drivers: dict[int, list[int]] = {}
    for driver in state.drivers:
        if driver.status != DriverStatus.IDLE:
            continue
        idx = grid.locate(driver.lat, driver.lon)
        if idx is None:
            continue
        idle_per_zone[idx] += 1.0
        idle_drivers.setdefault(idx, []).append(driver.id)
    return idle_per_zone, idle_drivers


def _normalize_scores(scores: list[float], n: int) -> np.ndarray:
    if n <= 0:
        return np.ones(1, dtype=float)
    arr = np.asarray(scores, dtype=float)
    if arr.size != n:
        arr = np.ones(n, dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        return np.ones(n, dtype=float) / n
    return arr / total


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cum = np.cumsum(sorted_weights)
    if float(cum[-1]) <= 1e-12:
        return float(np.quantile(values, quantile))
    cum /= cum[-1]
    idx = int(np.searchsorted(cum, quantile, side="left"))
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.size == 0:
        return x
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-9)


def _record_zone_fractions(record: RegimeRecord, grid: _HexGrid) -> np.ndarray:
    counts = np.zeros(grid.n, dtype=float)
    meta = record.metadata or {}
    pickup_lats = meta.get("pickup_lats", [])
    pickup_lons = meta.get("pickup_lons", [])
    for lat, lon in zip(pickup_lats, pickup_lons):
        idx = grid.locate(float(lat), float(lon))
        if idx is not None:
            counts[idx] += 1.0
    total = float(counts.sum())
    if total <= 0.0:
        counts[:] = 1.0 / max(grid.n, 1)
        return counts
    return counts / total


def _greedy_move_surplus_to_deficit(
    grid: _HexGrid,
    idle_drivers: dict[int, list[int]],
    surplus: np.ndarray,
    deficit: np.ndarray,
    *,
    max_moves: int,
    rng: np.random.Generator,
) -> list[RepositionInstruction]:
    if max_moves <= 0 or float(deficit.sum()) <= 0 or float(surplus.sum()) <= 0:
        return []

    instructions: list[RepositionInstruction] = []
    moved = 0
    deficit_zones = [
        int(i) for i in np.argsort(-deficit)
        if deficit[i] >= 0.5
    ]

    for dst in deficit_zones:
        if moved >= max_moves:
            break
        dst_lon, dst_lat = grid.hex_centers[grid.hex_ids[dst]]
        surplus_zones = [
            int(i) for i in np.argsort([
                _haversine_m(grid.hex_centers[grid.hex_ids[src]], (dst_lon, dst_lat))
                if surplus[src] >= 1.0 else np.inf
                for src in range(grid.n)
            ])
            if surplus[i] >= 1.0
        ]
        for src in surplus_zones:
            if moved >= max_moves or deficit[dst] < 0.5:
                break
            drivers_here = idle_drivers.get(src, [])
            while drivers_here and surplus[src] >= 1.0 and deficit[dst] >= 0.5:
                if moved >= max_moves:
                    break
                driver_id = drivers_here.pop(0)
                instructions.append(RepositionInstruction(
                    driver_id=driver_id,
                    target_lon=dst_lon + rng.normal(0.0, 0.001),
                    target_lat=dst_lat + rng.normal(0.0, 0.001),
                ))
                surplus[src] -= 1.0
                deficit[dst] -= 1.0
                moved += 1
    return instructions


def _transport_lp_reposition(
    grid: _HexGrid,
    idle_drivers: dict[int, list[int]],
    surplus: np.ndarray,
    deficit: np.ndarray,
    *,
    max_moves: int,
    router: RoutingClient,
    cost_cache: dict[tuple[int, int], float],
    rng: np.random.Generator,
) -> list[RepositionInstruction]:
    """LP transportation repositioning used by the oracle MPC baseline."""
    src_zones = [int(i) for i in range(grid.n) if surplus[i] >= 1.0]
    dst_zones = [int(i) for i in range(grid.n) if deficit[i] >= 0.5]
    if max_moves <= 0 or not src_zones or not dst_zones:
        return []

    s_count = len(src_zones)
    d_count = len(dst_zones)
    n_x = s_count * d_count
    n_y = d_count
    n_vars = n_x + n_y

    costs = np.zeros(n_x, dtype=float)
    max_travel = 0.0
    for si, src in enumerate(src_zones):
        for di, dst in enumerate(dst_zones):
            key = (src, dst)
            if key not in cost_cache:
                cost_cache[key] = router.travel_time(
                    grid.hex_centers[grid.hex_ids[src]],
                    grid.hex_centers[grid.hex_ids[dst]],
                )
            tt = cost_cache[key]
            costs[si * d_count + di] = tt
            max_travel = max(max_travel, tt)

    objective = np.zeros(n_vars, dtype=float)
    alpha = 1.0 / max(max_travel * 2.0, 1.0)
    objective[:n_x] = alpha * costs
    objective[n_x:] = -1.0

    rows = []
    rhs = []
    for si, src in enumerate(src_zones):
        row = np.zeros(n_vars, dtype=float)
        row[si * d_count:(si + 1) * d_count] = 1.0
        rows.append(row)
        rhs.append(float(surplus[src]))

    for di, dst in enumerate(dst_zones):
        row = np.zeros(n_vars, dtype=float)
        row[n_x + di] = 1.0
        rows.append(row)
        rhs.append(float(deficit[dst]))

    for di in range(d_count):
        row = np.zeros(n_vars, dtype=float)
        row[di:n_x:d_count] = -1.0
        row[n_x + di] = 1.0
        rows.append(row)
        rhs.append(0.0)

    row = np.zeros(n_vars, dtype=float)
    row[:n_x] = 1.0
    rows.append(row)
    rhs.append(float(max_moves))

    result = linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs),
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not result.success:
        return []

    instructions: list[RepositionInstruction] = []
    x = result.x[:n_x]
    for si, src in enumerate(src_zones):
        drivers_here = idle_drivers.get(src, [])
        moved_from_src = 0
        for di, dst in enumerate(dst_zones):
            n_move = int(round(x[si * d_count + di]))
            if n_move <= 0:
                continue
            dst_lon, dst_lat = grid.hex_centers[grid.hex_ids[dst]]
            for _ in range(min(n_move, len(drivers_here) - moved_from_src)):
                driver_id = drivers_here[moved_from_src]
                instructions.append(RepositionInstruction(
                    driver_id=driver_id,
                    target_lon=dst_lon + rng.normal(0.0, 0.001),
                    target_lat=dst_lat + rng.normal(0.0, 0.001),
                ))
                moved_from_src += 1
    return instructions
