#!/usr/bin/env python3
"""Download and clean Chicago TNP (rideshare) trip data via Socrata API.

Chicago TNP data uses Census Tract centroid lat/lon (already provided by the API).
We download Jan 2024 and Jun 2024 to match our NYC months.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET_ID = "n26f-ihde"
BASE_URL = f"https://data.cityofchicago.org/resource/{DATASET_ID}.csv"

MONTHS = [
    ("2024-01", "2024-01-01T00:00:00", "2024-01-31T23:59:59"),
    ("2024-06", "2024-06-01T00:00:00", "2024-06-30T23:59:59"),
]

CHICAGO_CORE_BBOX = {
    "min_lon": -87.78,
    "max_lon": -87.58,
    "min_lat": 41.82,
    "max_lat": 41.96,
}

PAGE_SIZE = 50_000
OUT_DIR = Path("data/processed")


def download_month(tag: str, start: str, end: str) -> pd.DataFrame:
    """Page through Socrata API with spatial+temporal filter to get core-area trips."""
    out_path = OUT_DIR / f"clean_chicago_tripdata_{tag}.parquet"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return pd.read_parquet(out_path)

    all_chunks = []
    offset = 0
    bbox = CHICAGO_CORE_BBOX
    where = (
        f"trip_start_timestamp between '{start}' and '{end}'"
        f" AND pickup_centroid_latitude > {bbox['min_lat']}"
        f" AND pickup_centroid_latitude < {bbox['max_lat']}"
        f" AND pickup_centroid_longitude > {bbox['min_lon']}"
        f" AND pickup_centroid_longitude < {bbox['max_lon']}"
    )

    print(f"  Downloading Chicago {tag} (core bbox filtered) ...")
    max_retries = 3
    with tqdm(desc=tag, unit=" rows") as bar:
        while True:
            params = {
                "$limit": PAGE_SIZE,
                "$offset": offset,
                "$where": where,
                "$order": "trip_start_timestamp",
                "$select": (
                    "trip_start_timestamp,trip_end_timestamp,"
                    "trip_seconds,trip_miles,fare,tip,trip_total,"
                    "pickup_centroid_latitude,pickup_centroid_longitude,"
                    "dropoff_centroid_latitude,dropoff_centroid_longitude"
                ),
            }
            for attempt in range(max_retries):
                try:
                    resp = requests.get(BASE_URL, params=params, timeout=300)
                    resp.raise_for_status()
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        import time
                        wait = 10 * (attempt + 1)
                        print(f"\n  Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        print(f"\n  [WARN] Giving up after {max_retries} retries at offset={offset}")
                        break
            else:
                break

            from io import StringIO
            chunk = pd.read_csv(StringIO(resp.text))
            if chunk.empty:
                break
            all_chunks.append(chunk)
            bar.update(len(chunk))
            offset += PAGE_SIZE
            if len(chunk) < PAGE_SIZE:
                break

    if not all_chunks:
        print(f"  [WARN] No data for {tag}")
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    print(f"  Downloaded: {len(df):,} rows (core bbox)")
    return df


def clean_chicago(df: pd.DataFrame, tag: str, bbox: dict) -> pd.DataFrame:
    out_path = OUT_DIR / f"clean_chicago_tripdata_{tag}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    df = df.rename(columns={
        "trip_start_timestamp": "pickup_datetime",
        "trip_end_timestamp": "dropoff_datetime",
        "pickup_centroid_longitude": "pickup_longitude",
        "pickup_centroid_latitude": "pickup_latitude",
        "dropoff_centroid_longitude": "dropoff_longitude",
        "dropoff_centroid_latitude": "dropoff_latitude",
        "trip_miles": "trip_distance",
        "trip_total": "total_amount",
    })

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    coord_cols = ["pickup_longitude", "pickup_latitude",
                  "dropoff_longitude", "dropoff_latitude"]
    for c in coord_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=coord_cols + ["pickup_datetime", "dropoff_datetime"])

    df = df[
        (df["pickup_longitude"].between(bbox["min_lon"], bbox["max_lon"]))
        & (df["pickup_latitude"].between(bbox["min_lat"], bbox["max_lat"]))
        & (df["dropoff_longitude"].between(bbox["min_lon"], bbox["max_lon"]))
        & (df["dropoff_latitude"].between(bbox["min_lat"], bbox["max_lat"]))
    ]

    keep = [c for c in [
        "pickup_datetime", "dropoff_datetime",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "trip_distance", "total_amount",
    ] if c in df.columns]

    df = df[keep].reset_index(drop=True)
    after = len(df)
    print(f"  {before:,} -> {after:,} rows ({after / max(before, 1) * 100:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Wrote {out_path}")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tag, start, end in MONTHS:
        print(f"\n=== Chicago {tag} ===")
        raw = download_month(tag, start, end)
        if raw.empty:
            continue
        clean_chicago(raw, tag, CHICAGO_CORE_BBOX)

    print("\nDone.")


if __name__ == "__main__":
    main()
