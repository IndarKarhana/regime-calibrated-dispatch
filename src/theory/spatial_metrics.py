"""Spatial demand-mismatch helpers for Path 2 theory and gate training."""

from __future__ import annotations

import h3
import numpy as np

from src.config import get_config
from src.regime.store import RegimeLibrary, RegimeRecord
from src.simulator.routing import _haversine_m

OTHER_ZONE = "__OTHER__"


def record_pickup_cells(record: RegimeRecord, h3_res: int) -> list[str]:
    """Map a regime's stored pickup coordinates to H3 cells."""
    lats = record.metadata.get("pickup_lats", [])
    lons = record.metadata.get("pickup_lons", [])
    cells = []
    for lat, lon in zip(lats, lons):
        if np.isfinite(lat) and np.isfinite(lon):
            cells.append(h3.latlng_to_cell(float(lat), float(lon), h3_res))
    return cells


def build_zone_index(
    library: RegimeLibrary,
    *,
    h3_res: int = 8,
    max_zones: int = 8,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Build a coarse spatial count vector for every regime record."""
    global_counts: dict[str, int] = {}
    record_cells: dict[str, list[str]] = {}
    for bid, record in library.records.items():
        cells = record_pickup_cells(record, h3_res)
        record_cells[bid] = cells
        for cell in cells:
            global_counts[cell] = global_counts.get(cell, 0) + 1

    top_n = max(1, max_zones - 1)
    top_cells = [
        cell for cell, _ in sorted(global_counts.items(), key=lambda kv: -kv[1])[:top_n]
    ]
    zone_ids = top_cells + [OTHER_ZONE]
    zone_set = set(top_cells)
    zone_pos = {cell: i for i, cell in enumerate(top_cells)}

    counts_by_record = {}
    for bid, cells in record_cells.items():
        counts = np.zeros(len(zone_ids), dtype=float)
        for cell in cells:
            idx = zone_pos[cell] if cell in zone_set else len(zone_ids) - 1
            counts[idx] += 1.0
        counts_by_record[bid] = counts
    return zone_ids, counts_by_record


def zone_centers(zone_ids: list[str]) -> list[tuple[float, float]]:
    """Return `(lon, lat)` center for each zone."""
    bbox = get_config()["data"]["manhattan_bbox"]
    other = (
        (bbox["min_lon"] + bbox["max_lon"]) / 2.0,
        (bbox["min_lat"] + bbox["max_lat"]) / 2.0,
    )
    centers = []
    for zone in zone_ids:
        if zone == OTHER_ZONE:
            centers.append(other)
        else:
            lat, lon = h3.cell_to_latlng(zone)
            centers.append((lon, lat))
    return centers


def travel_cost_matrix_s(zone_ids: list[str]) -> np.ndarray:
    """Haversine travel-time matrix between zone centers."""
    speed_ms = get_config()["simulator"]["speed_kmh_fallback"] / 3.6
    centers = zone_centers(zone_ids)
    out = np.zeros((len(zone_ids), len(zone_ids)), dtype=float)
    for i, src in enumerate(centers):
        for j, dst in enumerate(centers):
            out[i, j] = _haversine_m(src, dst) / speed_ms
    return out


def scale_counts_to_total(counts: np.ndarray, target_total: float) -> np.ndarray:
    """Scale a nonnegative spatial count vector to a desired total."""
    arr = np.asarray(counts, dtype=float).copy()
    total = float(arr.sum())
    if target_total <= 0:
        return np.zeros_like(arr)
    if total <= 1e-12:
        return np.ones_like(arr) * (target_total / max(len(arr), 1))
    return arr * (target_total / total)


def greedy_transport_distance(cost_matrix: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Fast min-cost-flow proxy between two spatial distributions.

    This greedily ships surplus estimated mass to true shortage zones using the
    nearest remaining surplus. It is not used as the formal verification
    certificate; exact OT remains in `earth_movers_distance`.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    shortage = np.maximum(p - q, 0.0)
    surplus = np.maximum(q - p, 0.0)
    total_cost = 0.0

    for dst in np.argsort(-shortage):
        need = float(shortage[dst])
        if need <= 1e-12:
            continue
        for src in np.argsort(cost_matrix[:, dst]):
            have = float(surplus[src])
            if have <= 1e-12:
                continue
            flow = min(need, have)
            total_cost += flow * float(cost_matrix[src, dst])
            surplus[src] -= flow
            need -= flow
            if need <= 1e-12:
                break
    return float(total_cost)

