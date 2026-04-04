"""Regime library: stores indexed regimes (demand profiles + event metadata) on disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.regime.events import SurgeEvent, annotate_events
from src.regime.ingest import build_demand_profile, split_into_blocks


class RegimeRecord:
    """One stored regime (a sub-day block of demand)."""

    __slots__ = (
        "block_id", "demand_series", "ecdf_values", "summary_features",
        "events", "bin_starts", "metadata",
    )

    def __init__(
        self,
        block_id: str,
        demand_series: np.ndarray,
        ecdf_values: np.ndarray,
        summary_features: np.ndarray,
        events: list[SurgeEvent],
        bin_starts: list[str],
        metadata: dict | None = None,
    ):
        self.block_id = block_id
        self.demand_series = demand_series
        self.ecdf_values = ecdf_values
        self.summary_features = summary_features
        self.events = events
        self.bin_starts = bin_starts
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        def _event_to_dict(e: SurgeEvent) -> dict:
            d = asdict(e)
            if d.get("bin_start") is not None:
                d["bin_start"] = str(d["bin_start"])
            return d

        return {
            "block_id": self.block_id,
            "demand_series": self.demand_series.tolist(),
            "ecdf_values": self.ecdf_values.tolist(),
            "summary_features": self.summary_features.tolist(),
            "events": [_event_to_dict(e) for e in self.events],
            "bin_starts": self.bin_starts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeRecord:
        events = []
        for ed in d.get("events", []):
            events.append(SurgeEvent(**ed))
        return cls(
            block_id=d["block_id"],
            demand_series=np.array(d["demand_series"], dtype=np.float64),
            ecdf_values=np.array(d["ecdf_values"], dtype=np.float64),
            summary_features=np.array(d["summary_features"], dtype=np.float64),
            events=events,
            bin_starts=d.get("bin_starts", []),
            metadata=d.get("metadata", {}),
        )


def _compute_summary_features(series: np.ndarray) -> np.ndarray:
    """Low-dimensional summary: mean, std, skew, kurtosis, min, max, iqr, slope."""
    n = len(series)
    if n == 0:
        return np.zeros(8)
    m = np.mean(series)
    s = np.std(series)
    sk = float(pd.Series(series).skew()) if n > 2 else 0.0
    ku = float(pd.Series(series).kurtosis()) if n > 3 else 0.0
    q25, q75 = np.percentile(series, [25, 75])
    slope = np.polyfit(range(n), series, 1)[0] if n > 1 else 0.0
    return np.array([m, s, sk, ku, float(np.min(series)), float(np.max(series)),
                     q75 - q25, slope], dtype=np.float64)


def _compute_ecdf(series: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Evaluate ECDF at n_points equally spaced quantiles."""
    sorted_vals = np.sort(series)
    n = len(sorted_vals)
    if n == 0:
        return np.zeros(n_points)
    quantiles = np.linspace(0, 1, n_points)
    indices = np.clip((quantiles * (n - 1)).astype(int), 0, n - 1)
    return sorted_vals[indices]


class RegimeLibrary:
    """Collection of RegimeRecords, serialized as JSON on disk."""

    def __init__(self, store_dir: Path | str | None = None):
        cfg = get_config()["data"]
        self._dir = Path(store_dir) if store_dir else Path(cfg["processed_dir"]) / "regime_library"
        self._dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, RegimeRecord] = {}

    def build_from_trips(self, trips_df: pd.DataFrame, max_od_per_block: int = 2000) -> None:
        """Build library from cleaned trips DataFrame, including spatial OD samples."""
        trips_df = trips_df.copy()
        trips_df["date"] = trips_df["pickup_datetime"].dt.date
        trips_df["hour"] = trips_df["pickup_datetime"].dt.hour

        profile = build_demand_profile(trips_df)
        blocks = split_into_blocks(profile)

        cfg = get_config()["regime"]
        block_hours = cfg["block_hours"]

        for block_df in blocks:
            bid = block_df["block_id"].iloc[0]
            series = block_df["request_count"].values.astype(np.float64)
            bin_starts_raw = block_df["bin_start"]
            bin_starts_str = [str(b) for b in bin_starts_raw]

            events = annotate_events(
                series,
                bin_starts=bin_starts_raw.reset_index(drop=True),
            )

            metadata = self._extract_spatial_od(
                trips_df, bid, block_hours, max_od_per_block
            )

            rec = RegimeRecord(
                block_id=bid,
                demand_series=series,
                ecdf_values=_compute_ecdf(series),
                summary_features=_compute_summary_features(series),
                events=events,
                bin_starts=bin_starts_str,
                metadata=metadata,
            )
            self.records[bid] = rec

    @staticmethod
    def _extract_spatial_od(
        trips_df: pd.DataFrame, block_id: str, block_hours: int, max_samples: int,
    ) -> dict:
        """Extract sampled pickup/dropoff coordinates for a block."""
        parts = block_id.rsplit("_", 1)
        if len(parts) != 2:
            return {}
        date_str, hour_range = parts
        try:
            target_date = pd.Timestamp(date_str).date()
            start_h = int(hour_range.split("-")[0])
        except (ValueError, IndexError):
            return {}

        end_h = start_h + block_hours
        mask = (
            (trips_df["date"] == target_date)
            & (trips_df["hour"] >= start_h)
            & (trips_df["hour"] < end_h)
        )
        sub = trips_df.loc[mask]

        coord_cols = ["pickup_longitude", "pickup_latitude",
                      "dropoff_longitude", "dropoff_latitude"]
        if not all(c in sub.columns for c in coord_cols):
            return {}

        sub = sub.dropna(subset=coord_cols)
        if len(sub) > max_samples:
            sub = sub.sample(max_samples, random_state=42)

        return {
            "pickup_lons": sub["pickup_longitude"].tolist(),
            "pickup_lats": sub["pickup_latitude"].tolist(),
            "dropoff_lons": sub["dropoff_longitude"].tolist(),
            "dropoff_lats": sub["dropoff_latitude"].tolist(),
        }

    def save(self) -> None:
        out = self._dir / "library.json"
        data = {k: v.to_dict() for k, v in self.records.items()}
        with open(out, "w") as f:
            json.dump(data, f)

    def load(self) -> None:
        p = self._dir / "library.json"
        if not p.exists():
            raise FileNotFoundError(f"No regime library at {p}")
        with open(p) as f:
            data = json.load(f)
        self.records = {k: RegimeRecord.from_dict(v) for k, v in data.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: str) -> RegimeRecord:
        return self.records[key]
