"""Gymnasium environment: batch matching handles dispatch, RL controls repositioning.

Architecture rationale (standard in fleet management RL literature):
- Dispatch = batch matching (best OR baseline, runs every step automatically)
- RL action = select a target hex zone to reposition excess idle drivers toward
- This separates what's already solved (myopic matching) from what RL can improve
  (proactive fleet balancing based on predicted demand patterns + regime context)
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import h3
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import get_config
from src.simulator.entities import (
    Assignment, Driver, DriverStatus, MetricsAccumulator, RideRequest, SimState,
)
from src.simulator.routing import RoutingClient, HaversineClient, _haversine_m

DEMAND_TREND_WINDOW = 5


class RideHailEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_stream,
        router: RoutingClient | None = None,
        fleet_size: int | None = None,
        step_seconds: float | None = None,
        horizon_seconds: float = 14400.0,
        seed: int = 42,
        regime_features: np.ndarray | None = None,
        reposition_fraction: float = 0.15,
        batch_window_steps: int = 2,
    ):
        super().__init__()
        cfg = get_config()
        sim_cfg = cfg["simulator"]
        rl_cfg = cfg["rl"]

        self._demand = demand_stream
        self._router = router or HaversineClient()
        self._fleet_size = fleet_size or sim_cfg["fleet_size"]
        self._dt = step_seconds or sim_cfg["step_seconds"]
        self._max_wait = sim_cfg["max_wait_seconds"]
        self._horizon = horizon_seconds
        self._h3_res = rl_cfg["h3_resolution"]
        self._seed = seed
        self._regime_feats = regime_features if regime_features is not None else np.zeros(8)
        self._reposition_frac = reposition_fraction
        self._batch_interval = batch_window_steps
        self._trend_window = DEMAND_TREND_WINDOW

        bbox = cfg["data"]["manhattan_bbox"]
        self._bbox = bbox
        self._hex_ids = self._build_hex_grid(bbox)
        self._n_hexes = len(self._hex_ids)
        self._hex_to_idx = {h: i for i, h in enumerate(self._hex_ids)}
        self._hex_centers = {}
        for hx in self._hex_ids:
            lat, lon = h3.cell_to_latlng(hx)
            self._hex_centers[hx] = (lon, lat)

        n_regime = len(self._regime_feats)
        # demand_map + driver_map + supply_gap + demand_trend + 4 global + regime
        obs_dim = self._n_hexes * 4 + 4 + n_regime
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self._n_hexes + 1)  # last = no reposition

        self._state: SimState | None = None
        self._step_count = 0
        self._prev_completed = 0
        self._prev_wait_sum = 0.0
        self._demand_history: deque[np.ndarray] = deque(maxlen=self._trend_window)

    def _build_hex_grid(self, bbox: dict) -> list[str]:
        center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
        center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
        center_hex = h3.latlng_to_cell(center_lat, center_lon, self._h3_res)
        hexes = list(h3.grid_disk(center_hex, 12))
        filtered = []
        for hx in hexes:
            lat, lon = h3.cell_to_latlng(hx)
            if (bbox["min_lat"] <= lat <= bbox["max_lat"]
                    and bbox["min_lon"] <= lon <= bbox["max_lon"]):
                filtered.append(hx)
        return sorted(filtered) if filtered else [center_hex]

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._demand.reset()
        rng = np.random.default_rng(seed or self._seed)
        drivers = []
        for i in range(self._fleet_size):
            lon = rng.uniform(self._bbox["min_lon"], self._bbox["max_lon"])
            lat = rng.uniform(self._bbox["min_lat"], self._bbox["max_lat"])
            drivers.append(Driver(id=i, lon=lon, lat=lat))
        self._state = SimState(time=0.0, drivers=drivers, metrics=MetricsAccumulator())
        self._step_count = 0
        self._prev_completed = 0
        self._prev_wait_sum = 0.0
        self._demand_history.clear()
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        state = self._state
        self._step_count += 1

        # 1. Inject new demand
        new_reqs = self._demand.get_requests_until(state.time)
        for r in new_reqs:
            state.pending_requests.append(r)
            state.metrics.total_requests += 1

        # 2. Batch matching dispatch (automatic, every batch_interval steps)
        if self._step_count % self._batch_interval == 0:
            assignments = self._batch_dispatch(state)
            self._apply_assignments(state, assignments)

        # 3. RL-controlled repositioning (only every few steps, and only if enough idle drivers)
        idle_count = sum(1 for d in state.drivers if d.status == DriverStatus.IDLE)
        idle_frac = idle_count / max(self._fleet_size, 1)
        if action < self._n_hexes and self._step_count % 3 == 0 and idle_frac > 0.1:
            self._reposition(state, action)

        # 4. Physics: move drivers, expire requests
        self._step_drivers(state)
        self._expire_requests(state)

        # 5. Reward: bounded, normalized signal
        new_completed = state.metrics.completed_trips - self._prev_completed
        new_wait = state.metrics.total_wait_seconds - self._prev_wait_sum
        self._prev_completed = state.metrics.completed_trips
        self._prev_wait_sum = state.metrics.total_wait_seconds

        n_pending = len(state.pending_requests)
        frac_completed = new_completed / max(self._fleet_size * 0.1, 1)
        frac_pending = n_pending / max(self._fleet_size, 1)
        avg_new_wait = (new_wait / max(new_completed, 1)) / self._max_wait if new_completed > 0 else 0

        reward = frac_completed - 0.5 * avg_new_wait - 0.1 * frac_pending
        reward = np.clip(reward, -2.0, 2.0)

        state.time += self._dt
        done = state.time >= self._horizon
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self) -> np.ndarray:
        state = self._state
        demand_map = np.zeros(self._n_hexes, dtype=np.float32)
        driver_map = np.zeros(self._n_hexes, dtype=np.float32)

        for r in state.pending_requests:
            hx = h3.latlng_to_cell(r.pickup_lat, r.pickup_lon, self._h3_res)
            idx = self._hex_to_idx.get(hx)
            if idx is not None:
                demand_map[idx] += 1

        for d in state.drivers:
            if d.status == DriverStatus.IDLE:
                hx = h3.latlng_to_cell(d.lat, d.lon, self._h3_res)
                idx = self._hex_to_idx.get(hx)
                if idx is not None:
                    driver_map[idx] += 1

        total_demand = max(demand_map.sum(), 1.0)
        total_idle = max(driver_map.sum(), 1.0)
        supply_gap = driver_map / total_idle - demand_map / total_demand

        self._demand_history.append(demand_map.copy())
        if len(self._demand_history) >= self._trend_window:
            oldest = self._demand_history[0]
            demand_trend = (demand_map - oldest) / max(total_demand, 1.0)
        else:
            demand_trend = np.zeros(self._n_hexes, dtype=np.float32)

        time_feats = np.array([
            state.time / self._horizon,
            len(state.pending_requests) / max(self._fleet_size, 1),
            state.metrics.completed_trips / max(state.metrics.total_requests, 1),
            state.metrics.expired_requests / max(state.metrics.total_requests, 1),
        ], dtype=np.float32)

        return np.concatenate([
            demand_map / max(total_demand, 1),
            driver_map / max(total_idle, 1),
            supply_gap,
            demand_trend,
            time_feats,
            self._regime_feats.astype(np.float32),
        ])

    def _batch_dispatch(self, state: SimState) -> list[Assignment]:
        """Min-cost bipartite matching of idle drivers to pending requests."""
        idle = [d for d in state.drivers if d.status == DriverStatus.IDLE]
        reqs = list(state.pending_requests)
        if not idle or not reqs:
            return []

        n_drv, n_req = len(idle), len(reqs)
        n = max(n_drv, n_req)
        cost = np.full((n, n), 1e9)
        for i, d in enumerate(idle):
            for j, r in enumerate(reqs):
                cost[i, j] = self._router.travel_time(
                    (d.lon, d.lat), (r.pickup_lon, r.pickup_lat)
                )

        row_ind, col_ind = linear_sum_assignment(cost)
        assignments = []
        for r_i, c_i in zip(row_ind, col_ind):
            if r_i < n_drv and c_i < n_req and cost[r_i, c_i] < 1e8:
                assignments.append(Assignment(
                    driver_id=idle[r_i].id,
                    request_id=reqs[c_i].id,
                    pickup_time_est=float(cost[r_i, c_i]),
                ))
        return assignments

    def _reposition(self, state: SimState, zone_idx: int) -> None:
        """Move a fraction of truly idle drivers toward the chosen hex zone."""
        idle = [d for d in state.drivers
                if d.status == DriverStatus.IDLE and d.idle_seconds > self._dt * 2]
        if not idle:
            return

        n_to_move = max(1, int(len(idle) * self._reposition_frac))
        target_hex = self._hex_ids[zone_idx]
        target_lon, target_lat = self._hex_centers[target_hex]

        dists = []
        for d in idle:
            dist = _haversine_m((d.lon, d.lat), (target_lon, target_lat))
            dists.append((d, dist))
        dists.sort(key=lambda x: x[1], reverse=True)

        moved = 0
        for d, dist in dists:
            if moved >= n_to_move:
                break
            if dist < 200:
                continue
            jitter_lon = target_lon + np.random.normal(0, 0.002)
            jitter_lat = target_lat + np.random.normal(0, 0.002)
            tt = self._router.travel_time((d.lon, d.lat), (jitter_lon, jitter_lat))
            d.status = DriverStatus.EN_ROUTE_PICKUP
            d.current_request = None
            d.remaining_seconds = tt
            d.dest_lon = jitter_lon
            d.dest_lat = jitter_lat
            moved += 1

    def _apply_assignments(self, state: SimState, assignments: list[Assignment]) -> None:
        driver_map = {d.id: d for d in state.drivers}
        request_map = {r.id: r for r in state.pending_requests}
        for a in assignments:
            drv = driver_map.get(a.driver_id)
            req = request_map.get(a.request_id)
            if drv is None or req is None or drv.status != DriverStatus.IDLE or req.assigned:
                continue
            tt = self._router.travel_time((drv.lon, drv.lat), (req.pickup_lon, req.pickup_lat))
            pickup_dist = _haversine_m((drv.lon, drv.lat), (req.pickup_lon, req.pickup_lat))
            drv.status = DriverStatus.EN_ROUTE_PICKUP
            drv.current_request = req
            drv.remaining_seconds = tt
            drv.dest_lon = req.pickup_lon
            drv.dest_lat = req.pickup_lat
            req.assigned = True
            state.metrics.total_pickup_distance_m += pickup_dist
        state.pending_requests = [r for r in state.pending_requests if not r.assigned]

    def _step_drivers(self, state: SimState) -> None:
        for drv in state.drivers:
            if drv.status == DriverStatus.IDLE:
                drv.idle_seconds += self._dt
                state.metrics.total_idle_seconds += self._dt
                continue
            drv.remaining_seconds -= self._dt
            if drv.remaining_seconds <= 0:
                req = drv.current_request
                if req is None:
                    # Repositioning complete
                    drv.lon = drv.dest_lon
                    drv.lat = drv.dest_lat
                    drv.status = DriverStatus.IDLE
                    drv.remaining_seconds = 0.0
                elif drv.status == DriverStatus.EN_ROUTE_PICKUP:
                    drv.lon, drv.lat = req.pickup_lon, req.pickup_lat
                    req.picked_up = True
                    req.wait_time = state.time - req.time + abs(drv.remaining_seconds)
                    state.metrics.wait_times.append(req.wait_time)
                    state.metrics.total_wait_seconds += req.wait_time
                    trip_tt = self._router.travel_time(
                        (req.pickup_lon, req.pickup_lat), (req.dropoff_lon, req.dropoff_lat)
                    )
                    drv.status = DriverStatus.IN_TRIP
                    drv.remaining_seconds = trip_tt
                    drv.dest_lon, drv.dest_lat = req.dropoff_lon, req.dropoff_lat
                    state.active_trips.append(req)
                elif drv.status == DriverStatus.IN_TRIP:
                    drv.lon, drv.lat = req.dropoff_lon, req.dropoff_lat
                    req.completed = True
                    state.metrics.completed_trips += 1
                    state.active_trips = [r for r in state.active_trips if r.id != req.id]
                    drv.status = DriverStatus.IDLE
                    drv.current_request = None
                    drv.remaining_seconds = 0.0
            else:
                frac = self._dt / (drv.remaining_seconds + self._dt)
                drv.lon += (drv.dest_lon - drv.lon) * frac
                drv.lat += (drv.dest_lat - drv.lat) * frac

    def _expire_requests(self, state: SimState) -> None:
        still = []
        for r in state.pending_requests:
            if state.time - r.time > self._max_wait:
                r.expired = True
                state.metrics.expired_requests += 1
            else:
                still.append(r)
        state.pending_requests = still
