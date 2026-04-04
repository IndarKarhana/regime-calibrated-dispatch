"""Prior calibrator: blend matched regimes into a demand prior for simulator rollouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config
from src.simulator.entities import RideRequest
from src.regime.store import RegimeRecord


class CalibratedPrior:
    """A calibrated demand model built from a weighted mixture of matched regimes.

    Supports sampling synthetic RideRequests for the simulator.
    """

    def __init__(
        self,
        matched_records: list[RegimeRecord],
        match_scores: list[float],
        query_stats: dict | None = None,
    ):
        total = sum(match_scores) + 1e-12
        self._weights = [s / total for s in match_scores]
        self._records = matched_records
        self._query_stats = query_stats or {}

        self._build_rate_model()
        self._build_spatial_pool()

    def _build_rate_model(self) -> None:
        """Combine demand rate series from matched regimes into a weighted mixture."""
        max_len = max(len(r.demand_series) for r in self._records) if self._records else 0
        blended = np.zeros(max_len)
        for rec, w in zip(self._records, self._weights):
            s = rec.demand_series
            padded = np.pad(s, (0, max_len - len(s)), constant_values=s[-1] if len(s) else 0)
            blended += w * padded

        q_mean = self._query_stats.get("mean_rate")
        q_std = self._query_stats.get("std_rate")
        if q_mean is not None and np.mean(blended) > 0:
            scale = q_mean / (np.mean(blended) + 1e-12)
            blended *= scale
        if q_std is not None and np.std(blended) > 0:
            current_std = np.std(blended)
            blended = np.mean(blended) + (blended - np.mean(blended)) * (q_std / (current_std + 1e-12))

        self._rate_profile = np.maximum(blended, 0.0)

    def fork(self) -> CalibratedPrior:
        """Shallow copy with an independent ``_rate_profile`` (records / OD arrays shared read-only)."""
        p = object.__new__(CalibratedPrior)
        p._weights = self._weights
        p._records = self._records
        p._query_stats = self._query_stats
        p._rate_profile = self._rate_profile.copy()
        p._has_spatial = self._has_spatial
        if self._has_spatial:
            p._od_lons_pu_arr = self._od_lons_pu_arr
            p._od_lats_pu_arr = self._od_lats_pu_arr
            p._od_lons_do_arr = self._od_lons_do_arr
            p._od_lats_do_arr = self._od_lats_do_arr
        return p

    def match_expected_total_for_horizon(
        self, target_total_requests: float, horizon_seconds: float,
    ) -> None:
        """Scale rates for bins overlapping ``[0, horizon)`` so Poisson means sum to ``target``.

        Keeps relative within-window shape; used so calibrated episodes match replay trip volume.
        """
        cfg = get_config()["regime"]
        bin_sec = cfg["bin_interval_minutes"] * 60
        n_bins = len(self._rate_profile)
        end_b = 0
        for b in range(n_bins):
            if b * bin_sec >= horizon_seconds:
                break
            end_b = b + 1
        if end_b <= 0:
            return
        tgt = max(float(target_total_requests), 0.0)
        prefix = self._rate_profile[:end_b]
        s = float(np.sum(prefix))
        if tgt <= 0:
            self._rate_profile[:end_b] = 0.0
            return
        if s <= 1e-12:
            self._rate_profile[:end_b] = tgt / end_b
        else:
            self._rate_profile[:end_b] *= tgt / s

    def _build_spatial_pool(self) -> None:
        """Pool all raw trip coordinates from matched regime days as spatial OD distribution.

        In production this would be per-regime spatial data; here we store zone-centroid
        pairs derived from the TLC data used to build each regime.
        """
        self._od_lons_pu: list[float] = []
        self._od_lats_pu: list[float] = []
        self._od_lons_do: list[float] = []
        self._od_lats_do: list[float] = []

        for rec in self._records:
            meta = rec.metadata
            if "pickup_lons" in meta:
                self._od_lons_pu.extend(meta["pickup_lons"])
                self._od_lats_pu.extend(meta["pickup_lats"])
                self._od_lons_do.extend(meta["dropoff_lons"])
                self._od_lats_do.extend(meta["dropoff_lats"])

        self._has_spatial = len(self._od_lons_pu) > 0
        if self._has_spatial:
            self._od_lons_pu_arr = np.array(self._od_lons_pu)
            self._od_lats_pu_arr = np.array(self._od_lats_pu)
            self._od_lons_do_arr = np.array(self._od_lons_do)
            self._od_lats_do_arr = np.array(self._od_lats_do)

    def sample_requests(
        self, horizon_seconds: float, rng: np.random.Generator,
    ) -> list[RideRequest]:
        """Generate synthetic requests over the horizon."""
        from src.config import get_config
        cfg = get_config()["regime"]
        bin_sec = cfg["bin_interval_minutes"] * 60
        n_bins = len(self._rate_profile)

        requests: list[RideRequest] = []
        req_id = 0

        for b in range(n_bins):
            bin_start_sec = b * bin_sec
            if bin_start_sec >= horizon_seconds:
                break
            rate = self._rate_profile[b]
            n_trips = rng.poisson(max(rate, 0))

            for _ in range(n_trips):
                t = bin_start_sec + rng.uniform(0, bin_sec)
                if t >= horizon_seconds:
                    continue
                pu_lon, pu_lat, do_lon, do_lat = self._sample_od(rng)
                requests.append(RideRequest(
                    id=req_id, time=t,
                    pickup_lon=pu_lon, pickup_lat=pu_lat,
                    dropoff_lon=do_lon, dropoff_lat=do_lat,
                ))
                req_id += 1

        return requests

    def _sample_od(self, rng: np.random.Generator) -> tuple[float, float, float, float]:
        """Sample an OD pair from the pooled spatial distribution (with jitter)."""
        if self._has_spatial:
            idx = rng.integers(0, len(self._od_lons_pu_arr))
            jitter = 0.001
            return (
                self._od_lons_pu_arr[idx] + rng.normal(0, jitter),
                self._od_lats_pu_arr[idx] + rng.normal(0, jitter),
                self._od_lons_do_arr[idx] + rng.normal(0, jitter),
                self._od_lats_do_arr[idx] + rng.normal(0, jitter),
            )
        from src.config import get_config
        bbox = get_config()["data"]["manhattan_bbox"]
        return (
            rng.uniform(bbox["min_lon"], bbox["max_lon"]),
            rng.uniform(bbox["min_lat"], bbox["max_lat"]),
            rng.uniform(bbox["min_lon"], bbox["max_lon"]),
            rng.uniform(bbox["min_lat"], bbox["max_lat"]),
        )

    def override_spatial_pool(self, od_pairs: list[tuple[float, float, float, float]]) -> None:
        """Replace the spatial OD pool (for cross-city transfer where spatial doesn't apply)."""
        if not od_pairs:
            return
        self._od_lons_pu_arr = np.array([p[0] for p in od_pairs])
        self._od_lats_pu_arr = np.array([p[1] for p in od_pairs])
        self._od_lons_do_arr = np.array([p[2] for p in od_pairs])
        self._od_lats_do_arr = np.array([p[3] for p in od_pairs])
        self._has_spatial = True

    @property
    def rate_profile(self) -> np.ndarray:
        return self._rate_profile


def build_calibrated_prior(
    matched_records: list[RegimeRecord],
    match_scores: list[float],
    query_series: np.ndarray | None = None,
) -> CalibratedPrior:
    """Convenience: build a CalibratedPrior from regime query results."""
    stats: dict = {}
    if query_series is not None and len(query_series) > 0:
        stats["mean_rate"] = float(np.mean(query_series))
        stats["std_rate"] = float(np.std(query_series))
    return CalibratedPrior(matched_records, match_scores, query_stats=stats)


def prior_matched_to_replay_volume(
    prior: CalibratedPrior,
    horizon_seconds: float,
    target_total_requests: float,
) -> CalibratedPrior:
    """Return a prior whose in-horizon expected request count matches ``target_total_requests``."""
    p = prior.fork()
    p.match_expected_total_for_horizon(target_total_requests, horizon_seconds)
    return p
