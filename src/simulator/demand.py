"""Demand stream: replay historical trips or sample from a calibrated prior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.entities import RideRequest


class ReplayDemandStream:
    """Replays cleaned TLC trips as a stream of RideRequests, sorted by pickup time."""

    def __init__(self, trips_df: pd.DataFrame, start_time: pd.Timestamp | None = None):
        df = trips_df.sort_values("pickup_datetime").reset_index(drop=True)
        if start_time is None:
            start_time = df["pickup_datetime"].iloc[0]
        self._start = start_time
        self._requests: list[RideRequest] = []
        for i, row in df.iterrows():
            t = (row["pickup_datetime"] - self._start).total_seconds()
            if t < 0:
                continue
            self._requests.append(RideRequest(
                id=int(i),
                time=t,
                pickup_lon=row["pickup_longitude"],
                pickup_lat=row["pickup_latitude"],
                dropoff_lon=row["dropoff_longitude"],
                dropoff_lat=row["dropoff_latitude"],
            ))
        self._idx = 0

    def get_requests_until(self, sim_time: float) -> list[RideRequest]:
        """Return all requests with time <= sim_time that haven't been emitted yet."""
        out = []
        while self._idx < len(self._requests) and self._requests[self._idx].time <= sim_time:
            out.append(self._requests[self._idx])
            self._idx += 1
        return out

    def reset(self) -> None:
        self._idx = 0

    @property
    def total_requests(self) -> int:
        return len(self._requests)


class CalibratedDemandStream:
    """Sample synthetic requests from a calibrated prior (mixture of regime distributions).

    Expects a CalibratedPrior object (from src.calibration.calibrator).
    """

    def __init__(self, prior, horizon_seconds: float, rng: np.random.Generator | None = None):
        self._prior = prior
        self._horizon = horizon_seconds
        self._rng = rng or np.random.default_rng(42)
        self._requests: list[RideRequest] = []
        self._idx = 0
        self._generate()

    def _generate(self) -> None:
        """Pre-generate all requests for the episode from the calibrated prior."""
        self._requests = self._prior.sample_requests(self._horizon, self._rng)
        self._requests.sort(key=lambda r: r.time)
        self._idx = 0

    def get_requests_until(self, sim_time: float) -> list[RideRequest]:
        out = []
        while self._idx < len(self._requests) and self._requests[self._idx].time <= sim_time:
            out.append(self._requests[self._idx])
            self._idx += 1
        return out

    def reset(self) -> None:
        self._generate()

    @property
    def total_requests(self) -> int:
        return len(self._requests)
