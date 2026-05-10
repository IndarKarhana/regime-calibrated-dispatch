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
    zone_stats = _zone_wait_stats(m.wait_times, m.wait_zone_ids)

    return {
        "mean_wait_s": float(np.mean(waits)),
        "p50_wait_s": float(np.median(waits)),
        "p95_wait_s": float(np.percentile(waits, 95)) if len(waits) > 1 else float(waits[0]),
        "mean_idle_s_per_driver": m.total_idle_seconds / n_drivers,
        "mean_pickup_dist_m": m.total_pickup_distance_m / n_completed,
        "mean_reposition_dist_m": m.total_reposition_distance_m / max(m.reposition_legs, 1),
        "reposition_dist_m_per_driver": m.total_reposition_distance_m / n_drivers,
        "reposition_dist_m_per_trip": m.total_reposition_distance_m / n_completed,
        "reposition_seconds_per_driver": m.total_reposition_seconds / n_drivers,
        "reposition_legs": m.reposition_legs,
        "completion_rate": m.completed_trips / n_total,
        "expired_rate": m.expired_requests / n_total,
        "throughput_trips_per_hour": m.completed_trips / horizon_h,
        **zone_stats,
        "total_requests": m.total_requests,
        "completed_trips": m.completed_trips,
        "wall_time_s": wall_time_seconds,
    }


def _zone_wait_stats(wait_times: list[float], zone_ids: list[str]) -> dict[str, float]:
    if not wait_times or not zone_ids or len(wait_times) != len(zone_ids):
        return {
            "zone_wait_gini": 0.0,
            "zone_wait_p90_p10_gap_s": 0.0,
            "zone_wait_max_mean_s": 0.0,
            "served_zones": 0,
        }

    waits_by_zone: dict[str, list[float]] = {}
    for wait, zone_id in zip(wait_times, zone_ids):
        waits_by_zone.setdefault(zone_id, []).append(float(wait))
    zone_means = np.asarray([np.mean(vals) for vals in waits_by_zone.values()], dtype=float)
    if zone_means.size == 0:
        return {
            "zone_wait_gini": 0.0,
            "zone_wait_p90_p10_gap_s": 0.0,
            "zone_wait_max_mean_s": 0.0,
            "served_zones": 0,
        }
    return {
        "zone_wait_gini": _gini(zone_means),
        "zone_wait_p90_p10_gap_s": float(
            np.percentile(zone_means, 90) - np.percentile(zone_means, 10)
        ) if zone_means.size > 1 else 0.0,
        "zone_wait_max_mean_s": float(np.max(zone_means)),
        "served_zones": int(zone_means.size),
    }


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.sort(np.maximum(arr, 0.0))
    total = float(arr.sum())
    if total <= 1e-12:
        return 0.0
    n = arr.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * arr) / (n * total)) - ((n + 1.0) / n))
