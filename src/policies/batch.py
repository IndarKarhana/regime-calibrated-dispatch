"""Batch matching: accumulate requests over a window, then solve min-cost bipartite matching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import get_config
from src.simulator.entities import Assignment, DriverStatus, SimState
from src.simulator.routing import RoutingClient


class BatchMatchingPolicy:
    def __init__(self, window_seconds: float | None = None):
        cfg = get_config()["policies"]
        self._window = window_seconds or cfg["batch_window_seconds"]
        self._last_batch_time: float = -1e9

    def assign(self, state: SimState, router: RoutingClient) -> list[Assignment]:
        if state.time - self._last_batch_time < self._window:
            return []

        self._last_batch_time = state.time
        idle = [d for d in state.drivers if d.status == DriverStatus.IDLE]
        reqs = list(state.pending_requests)
        if not idle or not reqs:
            return []

        n_drv = len(idle)
        n_req = len(reqs)
        n = max(n_drv, n_req)
        cost = np.full((n, n), 1e9)

        origins = [(d.lon, d.lat) for d in idle]
        dests = [(r.pickup_lon, r.pickup_lat) for r in reqs]

        mat = router.distance_matrix(origins, dests)
        cost[:n_drv, :n_req] = mat

        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = []
        for r, c in zip(row_ind, col_ind):
            if r < n_drv and c < n_req and cost[r, c] < 1e8:
                assignments.append(Assignment(
                    driver_id=idle[r].id,
                    request_id=reqs[c].id,
                    pickup_time_est=float(cost[r, c]),
                ))
        return assignments
