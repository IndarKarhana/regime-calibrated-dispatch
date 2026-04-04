"""Discrete-time ride-hailing simulation engine."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from src.config import get_config
from src.simulator.entities import (
    Assignment, Driver, DriverStatus, MetricsAccumulator,
    RideRequest, RepositionInstruction, SimState,
)
from src.simulator.routing import RoutingClient, HaversineClient, _haversine_m


class DemandStreamProtocol(Protocol):
    def get_requests_until(self, sim_time: float) -> list[RideRequest]: ...
    def reset(self) -> None: ...


class PolicyProtocol(Protocol):
    def assign(self, state: SimState, router: RoutingClient) -> list[Assignment]: ...


class RepositionProtocol(Protocol):
    def reposition(self, state: SimState, router: RoutingClient) -> list[RepositionInstruction]: ...


class SimulationEngine:
    """Runs one episode of the discrete-time simulator.

    Optionally accepts a reposition_policy that redistributes idle drivers
    toward predicted demand hotspots at a configurable interval.
    """

    def __init__(
        self,
        demand_stream: DemandStreamProtocol,
        policy: PolicyProtocol,
        router: RoutingClient | None = None,
        fleet_size: int | None = None,
        step_seconds: float | None = None,
        max_wait_seconds: float | None = None,
        horizon_seconds: float = 14400.0,
        seed: int = 42,
        reposition_policy: RepositionProtocol | None = None,
        reposition_interval_steps: int = 6,
        bbox: dict | None = None,
    ):
        cfg = get_config()["simulator"]
        self._demand = demand_stream
        self._policy = policy
        self._router = router or HaversineClient()
        self._fleet_size = fleet_size or cfg["fleet_size"]
        self._dt = step_seconds or cfg["step_seconds"]
        self._max_wait = max_wait_seconds or cfg["max_wait_seconds"]
        self._horizon = horizon_seconds
        self._rng = np.random.default_rng(seed)
        self._reposition_policy = reposition_policy
        self._reposition_interval = reposition_interval_steps
        self._bbox = bbox

    def run(self) -> SimState:
        state = self._init_state()
        step = 0

        while state.time < self._horizon:
            new_reqs = self._demand.get_requests_until(state.time)
            for r in new_reqs:
                state.pending_requests.append(r)
                state.metrics.total_requests += 1

            assignments = self._policy.assign(state, self._router)
            self._apply_assignments(state, assignments)

            if (self._reposition_policy is not None
                    and step % self._reposition_interval == 0):
                instructions = self._reposition_policy.reposition(
                    state, self._router
                )
                self._apply_repositions(state, instructions)

            self._step_drivers(state)
            self._expire_requests(state)

            state.time += self._dt
            step += 1

        return state

    def _init_state(self) -> SimState:
        bbox = self._bbox or get_config()["data"]["manhattan_bbox"]
        drivers = []
        for i in range(self._fleet_size):
            lon = self._rng.uniform(bbox["min_lon"], bbox["max_lon"])
            lat = self._rng.uniform(bbox["min_lat"], bbox["max_lat"])
            drivers.append(Driver(id=i, lon=lon, lat=lat))
        return SimState(time=0.0, drivers=drivers)

    def _apply_assignments(self, state: SimState, assignments: list[Assignment]) -> None:
        driver_map = {d.id: d for d in state.drivers}
        request_map = {r.id: r for r in state.pending_requests}

        # Filter to valid assignments first
        valid = []
        for a in assignments:
            drv = driver_map.get(a.driver_id)
            req = request_map.get(a.request_id)
            if drv is None or req is None or drv.status != DriverStatus.IDLE or req.assigned:
                continue
            valid.append((drv, req))

        if not valid:
            state.pending_requests = [r for r in state.pending_requests if not r.assigned]
            return

        # Batch routing: get all pickup travel times in one call
        pairs = [((d.lon, d.lat), (r.pickup_lon, r.pickup_lat)) for d, r in valid]
        travel_times = self._router.batch_travel_times(pairs)

        for (drv, req), tt in zip(valid, travel_times):
            pickup_dist = _haversine_m((drv.lon, drv.lat), (req.pickup_lon, req.pickup_lat))

            drv.status = DriverStatus.EN_ROUTE_PICKUP
            drv.current_request = req
            drv.remaining_seconds = tt
            drv.dest_lon = req.pickup_lon
            drv.dest_lat = req.pickup_lat

            req.assigned = True
            state.metrics.total_pickup_distance_m += pickup_dist

        state.pending_requests = [r for r in state.pending_requests if not r.assigned]

    def _apply_repositions(
        self, state: SimState, instructions: list[RepositionInstruction],
    ) -> None:
        driver_map = {d.id: d for d in state.drivers}

        # Filter to valid repositions first
        valid = []
        for inst in instructions:
            drv = driver_map.get(inst.driver_id)
            if drv is None or drv.status != DriverStatus.IDLE:
                continue
            dist = _haversine_m(
                (drv.lon, drv.lat), (inst.target_lon, inst.target_lat)
            )
            if dist < 100:
                continue
            valid.append((drv, inst))

        if not valid:
            return

        # Batch routing
        pairs = [((d.lon, d.lat), (i.target_lon, i.target_lat)) for d, i in valid]
        travel_times = self._router.batch_travel_times(pairs)

        for (drv, inst), tt in zip(valid, travel_times):
            drv.status = DriverStatus.EN_ROUTE_PICKUP
            drv.current_request = None
            drv.remaining_seconds = tt
            drv.dest_lon = inst.target_lon
            drv.dest_lat = inst.target_lat

    def _step_drivers(self, state: SimState) -> None:
        # First pass: identify drivers arriving at pickup that need trip routing
        need_trip_tt: list[tuple[Driver, RideRequest]] = []

        for drv in state.drivers:
            if drv.status == DriverStatus.IDLE:
                drv.idle_seconds += self._dt
                state.metrics.total_idle_seconds += self._dt
                continue

            drv.remaining_seconds -= self._dt

            if drv.remaining_seconds <= 0:
                req = drv.current_request

                if req is None:
                    drv.lon = drv.dest_lon
                    drv.lat = drv.dest_lat
                    drv.status = DriverStatus.IDLE
                    drv.remaining_seconds = 0.0

                elif drv.status == DriverStatus.EN_ROUTE_PICKUP:
                    drv.lon = req.pickup_lon
                    drv.lat = req.pickup_lat
                    req.picked_up = True
                    req.wait_time = state.time - req.time + abs(drv.remaining_seconds)
                    state.metrics.wait_times.append(req.wait_time)
                    state.metrics.total_wait_seconds += req.wait_time
                    # Defer trip routing to batch call
                    need_trip_tt.append((drv, req))

                elif drv.status == DriverStatus.IN_TRIP:
                    drv.lon = req.dropoff_lon
                    drv.lat = req.dropoff_lat
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

        # Batch trip travel times for all drivers that just completed pickup
        if need_trip_tt:
            pairs = [
                ((req.pickup_lon, req.pickup_lat), (req.dropoff_lon, req.dropoff_lat))
                for _, req in need_trip_tt
            ]
            trip_tts = self._router.batch_travel_times(pairs)
            for (drv, req), trip_tt in zip(need_trip_tt, trip_tts):
                drv.status = DriverStatus.IN_TRIP
                drv.remaining_seconds = trip_tt
                drv.dest_lon = req.dropoff_lon
                drv.dest_lat = req.dropoff_lat
                state.active_trips.append(req)

    def _expire_requests(self, state: SimState) -> None:
        still_pending = []
        for r in state.pending_requests:
            if state.time - r.time > self._max_wait:
                r.expired = True
                state.metrics.expired_requests += 1
            else:
                still_pending.append(r)
        state.pending_requests = still_pending
