"""Routing backends: OSRM (HTTP) and Haversine (fallback)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import requests

from src.config import get_config


class RoutingClient(ABC):
    @abstractmethod
    def travel_time(self, origin: tuple[float, float], dest: tuple[float, float]) -> float:
        """Return travel time in seconds between (lon, lat) pairs."""

    @abstractmethod
    def distance_matrix(
        self,
        origins: list[tuple[float, float]],
        dests: list[tuple[float, float]],
    ) -> np.ndarray:
        """Return |origins| x |dests| matrix of travel times in seconds."""

    def batch_travel_times(
        self,
        pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[float]:
        """Return travel times for a list of (origin, dest) pairs.

        Default uses individual calls; OSRM overrides with /table diagonal.
        """
        return [self.travel_time(o, d) for o, d in pairs]


class HaversineClient(RoutingClient):
    """Straight-line distance / constant speed — deterministic, no network needed."""

    def __init__(self, speed_kmh: float | None = None):
        cfg = get_config()
        self._speed_ms = (speed_kmh or cfg["simulator"]["speed_kmh_fallback"]) / 3.6

    def travel_time(self, origin: tuple[float, float], dest: tuple[float, float]) -> float:
        return _haversine_m(origin, dest) / self._speed_ms

    def distance_matrix(
        self,
        origins: list[tuple[float, float]],
        dests: list[tuple[float, float]],
    ) -> np.ndarray:
        mat = np.empty((len(origins), len(dests)), dtype=np.float64)
        for i, o in enumerate(origins):
            for j, d in enumerate(dests):
                mat[i, j] = _haversine_m(o, d) / self._speed_ms
        return mat


class OSRMClient(RoutingClient):
    """OSRM backend via HTTP — requires a running OSRM Docker container."""

    # OSRM /table/v1 can handle large coord sets but performance degrades.
    # Chunk to keep each request ≤ MAX_TABLE_COORDS total points.
    MAX_TABLE_COORDS = 250

    def __init__(self, host: str | None = None, port: int | None = None):
        cfg = get_config()["osrm"]
        self._base = f"{host or cfg['host']}:{port or cfg['port']}"
        self._profile = cfg.get("profile", "car")
        self._fallback = HaversineClient()
        # Persistent HTTP session for connection reuse
        self._session = requests.Session()
        # LRU cache for single travel_time calls (rounded to ~10m precision)
        self._tt_cache: dict[tuple[int, int, int, int], float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _round_key(
        origin: tuple[float, float], dest: tuple[float, float],
    ) -> tuple[int, int, int, int]:
        """Round coords to ~11m grid for cache key (4 decimal places)."""
        return (
            round(origin[0], 4), round(origin[1], 4),
            round(dest[0], 4), round(dest[1], 4),
        )

    def travel_time(self, origin: tuple[float, float], dest: tuple[float, float]) -> float:
        key = self._round_key(origin, dest)
        if key in self._tt_cache:
            self._cache_hits += 1
            return self._tt_cache[key]
        self._cache_misses += 1

        coords = f"{origin[0]},{origin[1]};{dest[0]},{dest[1]}"
        url = f"{self._base}/route/v1/{self._profile}/{coords}"
        try:
            r = self._session.get(url, params={"overview": "false"}, timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get("code") == "Ok":
                tt = float(data["routes"][0]["duration"])
                self._tt_cache[key] = tt
                return tt
        except Exception:
            pass
        tt = self._fallback.travel_time(origin, dest)
        self._tt_cache[key] = tt
        return tt

    def distance_matrix(
        self,
        origins: list[tuple[float, float]],
        dests: list[tuple[float, float]],
    ) -> np.ndarray:
        n_src = len(origins)
        n_dst = len(dests)

        # If small enough, single OSRM /table call
        if n_src + n_dst <= self.MAX_TABLE_COORDS:
            return self._table_call(origins, dests)

        # Chunk origins to stay within coord limit
        result = np.full((n_src, n_dst), 1e9, dtype=np.float64)
        chunk_size = max(1, self.MAX_TABLE_COORDS - n_dst)

        for i_start in range(0, n_src, chunk_size):
            i_end = min(i_start + chunk_size, n_src)
            chunk_origins = origins[i_start:i_end]

            # If even a single origin + all dests exceeds limit, chunk dests too
            if len(chunk_origins) + n_dst > self.MAX_TABLE_COORDS:
                d_chunk = max(1, self.MAX_TABLE_COORDS - len(chunk_origins))
                for j_start in range(0, n_dst, d_chunk):
                    j_end = min(j_start + d_chunk, n_dst)
                    sub = self._table_call(chunk_origins, dests[j_start:j_end])
                    result[i_start:i_end, j_start:j_end] = sub
            else:
                sub = self._table_call(chunk_origins, dests)
                result[i_start:i_end, :] = sub

        return result

    def _table_call(
        self,
        origins: list[tuple[float, float]],
        dests: list[tuple[float, float]],
    ) -> np.ndarray:
        """Single OSRM /table/v1 request. Falls back to Haversine on error."""
        all_pts = origins + dests
        n_src = len(origins)
        coords = ";".join(f"{p[0]},{p[1]}" for p in all_pts)
        sources = ";".join(str(i) for i in range(n_src))
        destinations = ";".join(str(i) for i in range(n_src, len(all_pts)))
        url = f"{self._base}/table/v1/{self._profile}/{coords}"
        try:
            r = self._session.get(
                url,
                params={"sources": sources, "destinations": destinations},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("code") == "Ok":
                mat = np.array(data["durations"], dtype=np.float64)
                mat = np.where(np.isnan(mat) | (mat < 0), 1e9, mat)
                return mat
        except Exception:
            pass
        return self._fallback.distance_matrix(origins, dests)

    def batch_travel_times(
        self,
        pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[float]:
        """Batch OD-pair travel times using OSRM /table with diagonal extraction.

        For N pairs we send 2N coordinates (N sources + N dests) and read
        the diagonal of the resulting N×N matrix — one API call instead of N.
        """
        if not pairs:
            return []
        n = len(pairs)
        origins = [p[0] for p in pairs]
        dests = [p[1] for p in pairs]

        # Chunk if needed (2N coords total)
        if 2 * n <= self.MAX_TABLE_COORDS:
            mat = self._table_call(origins, dests)
            return [float(mat[i, i]) for i in range(n)]

        # Chunk into sub-batches
        results: list[float] = []
        chunk = self.MAX_TABLE_COORDS // 2
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sub_origins = origins[start:end]
            sub_dests = dests[start:end]
            mat = self._table_call(sub_origins, sub_dests)
            for i in range(end - start):
                results.append(float(mat[i, i]))
        return results


class GridCachedOSRMClient(RoutingClient):
    """OSRM travel times pre-cached on a spatial grid.

    At initialization, queries OSRM for an NxN grid of travel times covering
    the bounding box, then uses nearest-grid-cell lookup for O(1) queries.
    Construction takes ~30-60s; simulation then runs at Haversine speed.
    """

    def __init__(
        self,
        grid_size: int = 30,
        bbox: dict | None = None,
        osrm_host: str | None = None,
        osrm_port: int | None = None,
    ):
        cfg = get_config()
        bbox = bbox or cfg["data"]["manhattan_bbox"]
        self._min_lon = bbox["min_lon"]
        self._max_lon = bbox["max_lon"]
        self._min_lat = bbox["min_lat"]
        self._max_lat = bbox["max_lat"]
        self._n = grid_size
        self._fallback = HaversineClient()

        # Build grid
        self._lons = np.linspace(self._min_lon, self._max_lon, grid_size)
        self._lats = np.linspace(self._min_lat, self._max_lat, grid_size)
        self._lon_step = self._lons[1] - self._lons[0] if grid_size > 1 else 1.0
        self._lat_step = self._lats[1] - self._lats[0] if grid_size > 1 else 1.0

        # Pre-compute full grid using OSRM
        osrm = OSRMClient(host=osrm_host, port=osrm_port)
        grid_points = [(lon, lat) for lat in self._lats for lon in self._lons]
        n_pts = len(grid_points)
        print(f"  Pre-computing OSRM grid: {grid_size}x{grid_size} = "
              f"{n_pts} points, {n_pts}x{n_pts} = {n_pts**2:,} travel times ...")

        # Get full N²×N² matrix via chunked OSRM table calls
        self._grid_tt = osrm.distance_matrix(grid_points, grid_points)
        # Replace unreachable with Haversine fallback
        bad = self._grid_tt >= 1e8
        if bad.any():
            fb = self._fallback.distance_matrix(grid_points, grid_points)
            self._grid_tt[bad] = fb[bad]
        print(f"  Grid cached: shape={self._grid_tt.shape}, "
              f"median={np.median(self._grid_tt):.0f}s, "
              f"unreachable filled={bad.sum()}")

    def _to_grid_idx(self, lon: float, lat: float) -> int:
        """Map (lon, lat) to nearest grid cell index (row-major: lat*n + lon)."""
        ci = int(round((lon - self._min_lon) / self._lon_step))
        ri = int(round((lat - self._min_lat) / self._lat_step))
        ci = max(0, min(ci, self._n - 1))
        ri = max(0, min(ri, self._n - 1))
        return ri * self._n + ci

    def travel_time(self, origin: tuple[float, float], dest: tuple[float, float]) -> float:
        i = self._to_grid_idx(origin[0], origin[1])
        j = self._to_grid_idx(dest[0], dest[1])
        return float(self._grid_tt[i, j])

    def distance_matrix(
        self,
        origins: list[tuple[float, float]],
        dests: list[tuple[float, float]],
    ) -> np.ndarray:
        oi = np.array([self._to_grid_idx(o[0], o[1]) for o in origins])
        di = np.array([self._to_grid_idx(d[0], d[1]) for d in dests])
        return self._grid_tt[np.ix_(oi, di)]

    def batch_travel_times(
        self,
        pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[float]:
        return [self.travel_time(o, d) for o, d in pairs]


def get_routing_client(backend: str = "osrm") -> RoutingClient:
    if backend == "osrm":
        return OSRMClient()
    return HaversineClient()


def _haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    R = 6_371_000.0
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
