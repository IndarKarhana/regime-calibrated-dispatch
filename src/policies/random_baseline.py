"""Random baseline: assign pending requests to random idle drivers."""

from __future__ import annotations

import random

from src.simulator.entities import Assignment, DriverStatus, SimState
from src.simulator.routing import RoutingClient


class RandomPolicy:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def assign(self, state: SimState, router: RoutingClient) -> list[Assignment]:
        idle = [d for d in state.drivers if d.status == DriverStatus.IDLE]
        reqs = list(state.pending_requests)
        if not idle or not reqs:
            return []

        self._rng.shuffle(idle)
        assignments = []
        for req, drv in zip(reqs, idle):
            tt = router.travel_time(
                (drv.lon, drv.lat), (req.pickup_lon, req.pickup_lat)
            )
            assignments.append(Assignment(
                driver_id=drv.id,
                request_id=req.id,
                pickup_time_est=tt,
            ))
        return assignments
