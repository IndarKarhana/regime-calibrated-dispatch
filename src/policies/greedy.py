"""Greedy nearest-driver dispatch: assign each pending request to the closest idle driver."""

from __future__ import annotations

from src.simulator.entities import Assignment, DriverStatus, SimState
from src.simulator.routing import RoutingClient


class GreedyNearestPolicy:
    def assign(self, state: SimState, router: RoutingClient) -> list[Assignment]:
        idle = [d for d in state.drivers if d.status == DriverStatus.IDLE]
        if not idle or not state.pending_requests:
            return []

        assignments: list[Assignment] = []
        used_drivers: set[int] = set()

        for req in state.pending_requests:
            best_drv = None
            best_tt = float("inf")
            for drv in idle:
                if drv.id in used_drivers:
                    continue
                tt = router.travel_time(
                    (drv.lon, drv.lat), (req.pickup_lon, req.pickup_lat)
                )
                if tt < best_tt:
                    best_tt = tt
                    best_drv = drv
            if best_drv is not None:
                assignments.append(Assignment(
                    driver_id=best_drv.id,
                    request_id=req.id,
                    pickup_time_est=best_tt,
                ))
                used_drivers.add(best_drv.id)

        return assignments
