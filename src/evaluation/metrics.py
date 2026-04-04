"""Compute KPIs from a completed simulation state."""

from __future__ import annotations

import numpy as np

from src.simulator.entities import SimState


def compute_kpis(state: SimState, wall_time_seconds: float = 0.0) -> dict[str, float]:
    m = state.metrics
    waits = np.array(m.wait_times) if m.wait_times else np.array([0.0])
    n_completed = max(m.completed_trips, 1)
    n_total = max(m.total_requests, 1)
    n_drivers = max(len(state.drivers), 1)
    horizon_h = max(state.time / 3600.0, 1e-6)

    return {
        "mean_wait_s": float(np.mean(waits)),
        "p50_wait_s": float(np.median(waits)),
        "p95_wait_s": float(np.percentile(waits, 95)) if len(waits) > 1 else float(waits[0]),
        "mean_idle_s_per_driver": m.total_idle_seconds / n_drivers,
        "mean_pickup_dist_m": m.total_pickup_distance_m / n_completed,
        "completion_rate": m.completed_trips / n_total,
        "expired_rate": m.expired_requests / n_total,
        "throughput_trips_per_hour": m.completed_trips / horizon_h,
        "total_requests": m.total_requests,
        "completed_trips": m.completed_trips,
        "wall_time_s": wall_time_seconds,
    }
