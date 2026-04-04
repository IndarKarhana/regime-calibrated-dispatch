"""Load cleaned TLC parquet and produce per-day, per-interval demand profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config


def load_cleaned(month_tag: str | None = None, exclude_chicago: bool = True) -> pd.DataFrame:
    """Return cleaned trip DataFrame; optionally filter to one month tag like '2024-01'.

    Chicago TNP parquets use ``clean_chicago_tripdata_*.parquet``; they are excluded by
    default so NYC loads are not silently mixed with cross-city data.
    """
    cfg = get_config()["data"]
    proc = Path(cfg["processed_dir"])
    files = sorted(proc.glob("clean_*.parquet"))
    if exclude_chicago:
        files = [f for f in files if "chicago" not in f.name.lower()]
    if month_tag:
        files = [f for f in files if month_tag in f.name]
    if not files:
        raise FileNotFoundError(f"No cleaned parquet in {proc}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    return df


def build_demand_profile(df: pd.DataFrame, bin_minutes: int | None = None) -> pd.DataFrame:
    """Bin trips into fixed intervals; return DataFrame with demand features per bin.

    Columns: date, bin_start, request_count, mean_trip_distance, mean_fare.
    """
    cfg = get_config()["regime"]
    bm = bin_minutes or cfg["bin_interval_minutes"]

    df = df.copy()
    df["date"] = df["pickup_datetime"].dt.date
    df["bin_start"] = df["pickup_datetime"].dt.floor(f"{bm}min")

    agg = {"pickup_datetime": "count"}
    if "trip_distance" in df.columns:
        agg["trip_distance"] = "mean"
    if "total_amount" in df.columns:
        agg["total_amount"] = "mean"

    profile = df.groupby(["date", "bin_start"]).agg(**{
        "request_count": pd.NamedAgg(column="pickup_datetime", aggfunc="count"),
        **({
            "mean_trip_distance": pd.NamedAgg(column="trip_distance", aggfunc="mean"),
        } if "trip_distance" in df.columns else {}),
        **({
            "mean_fare": pd.NamedAgg(column="total_amount", aggfunc="mean"),
        } if "total_amount" in df.columns else {}),
    }).reset_index()

    profile["hour"] = pd.to_datetime(profile["bin_start"]).dt.hour
    profile["minute"] = pd.to_datetime(profile["bin_start"]).dt.minute
    return profile


def split_into_blocks(profile: pd.DataFrame, block_hours: int | None = None) -> list[pd.DataFrame]:
    """Split a demand profile into sub-day blocks (e.g. 4-hour windows)."""
    cfg = get_config()["regime"]
    bh = block_hours or cfg["block_hours"]
    blocks = []
    for date, day_df in profile.groupby("date"):
        for start_h in range(0, 24, bh):
            end_h = start_h + bh
            mask = day_df["hour"].between(start_h, end_h - 1)
            block = day_df[mask].copy()
            if len(block) > 0:
                block["block_id"] = f"{date}_{start_h:02d}-{end_h:02d}"
                blocks.append(block)
    return blocks
