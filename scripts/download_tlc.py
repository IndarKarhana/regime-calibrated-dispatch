#!/usr/bin/env python3
"""Download and clean NYC TLC Yellow Taxi parquet files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def clean_month(raw_path: Path, out_path: Path, bbox: dict) -> None:
    """Filter to Manhattan bbox, drop nulls, write cleaned parquet."""
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    print(f"  Cleaning {raw_path.name} ...")
    df = pd.read_parquet(raw_path)

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "pickup" in cl and ("lon" in cl or "long" in cl):
            col_map[c] = "pickup_longitude"
        elif "pickup" in cl and "lat" in cl:
            col_map[c] = "pickup_latitude"
        elif "dropoff" in cl and ("lon" in cl or "long" in cl):
            col_map[c] = "dropoff_longitude"
        elif "dropoff" in cl and "lat" in cl:
            col_map[c] = "dropoff_latitude"
        elif "pickup" in cl and ("time" in cl or "date" in cl):
            col_map[c] = "pickup_datetime"
        elif "dropoff" in cl and ("time" in cl or "date" in cl):
            col_map[c] = "dropoff_datetime"
        elif "distance" in cl:
            col_map[c] = "trip_distance"
        elif "amount" in cl and "total" in cl:
            col_map[c] = "total_amount"
        elif "passenger" in cl:
            col_map[c] = "passenger_count"

    df = df.rename(columns=col_map)

    needed = ["pickup_datetime", "dropoff_datetime"]
    has_coords = "pickup_longitude" in df.columns and "pickup_latitude" in df.columns

    if not has_coords and "PULocationID" in df.columns:
        print("  Raw file uses LocationID, not lat/lon — applying zone centroid lookup")
        df = _join_zone_centroids(df)
        has_coords = "pickup_longitude" in df.columns

    if not has_coords:
        print(f"  [WARN] Cannot find coordinate columns in {raw_path.name}; skipping")
        return

    coord_cols = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ]
    for c in needed + coord_cols:
        if c not in df.columns:
            print(f"  [WARN] Missing column {c}; skipping")
            return

    before = len(df)
    df = df.dropna(subset=coord_cols + needed)

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
        "trip_distance", "total_amount", "passenger_count",
    ] if c in df.columns]

    df = df[keep].reset_index(drop=True)
    after = len(df)
    print(f"  {before:,} → {after:,} rows ({after / max(before, 1) * 100:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Wrote {out_path}")


_ZONE_CENTROIDS: pd.DataFrame | None = None


def _get_zone_centroids() -> pd.DataFrame:
    """Download and cache taxi zone centroid lookup."""
    global _ZONE_CENTROIDS
    if _ZONE_CENTROIDS is not None:
        return _ZONE_CENTROIDS
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    try:
        zones = pd.read_csv(url)
    except Exception:
        url2 = "https://raw.githubusercontent.com/nyc-taxi-data/nyc-taxi-data/master/taxi_zone_lookup.csv"
        zones = pd.read_csv(url2)

    centroid_url = "https://data.cityofnewyork.us/api/views/755u-8jsi/rows.csv?accessType=DOWNLOAD"
    try:
        geo = pd.read_csv(centroid_url)
        geo = geo.rename(columns=lambda c: c.strip())
        if "the_geom" in geo.columns:
            import re
            coords = geo["the_geom"].str.extract(
                r"POINT \((?P<lon>-?\d+\.?\d*) (?P<lat>\d+\.?\d*)\)"
            )
            geo["longitude"] = coords["lon"].astype(float)
            geo["latitude"] = coords["lat"].astype(float)
            geo["LocationID"] = geo.get("OBJECTID", geo.get("objectid", range(1, len(geo) + 1)))
            _ZONE_CENTROIDS = geo[["LocationID", "longitude", "latitude"]]
            return _ZONE_CENTROIDS
    except Exception:
        pass

    _ZONE_CENTROIDS = _hardcoded_manhattan_centroids()
    return _ZONE_CENTROIDS


def _hardcoded_manhattan_centroids() -> pd.DataFrame:
    """Fallback: approximate centroids for common Manhattan zone IDs."""
    data = {
        4: (-73.9549, 40.8030), 12: (-74.0007, 40.7268), 13: (-73.9776, 40.7454),
        24: (-73.9814, 40.7692), 41: (-73.9939, 40.7509), 42: (-73.9945, 40.7551),
        43: (-73.9776, 40.7580), 45: (-73.9712, 40.7583), 48: (-73.9863, 40.7236),
        50: (-73.9665, 40.7637), 68: (-73.9857, 40.7484), 74: (-73.9712, 40.7948),
        75: (-73.9665, 40.7741), 79: (-73.9600, 40.7763), 87: (-73.9700, 40.7531),
        88: (-73.9818, 40.7489), 90: (-73.9483, 40.7829), 100: (-73.9838, 40.7316),
        107: (-73.9561, 40.7807), 113: (-73.9862, 40.7271), 114: (-73.9770, 40.7455),
        125: (-73.9553, 40.7823), 137: (-73.9569, 40.7808), 140: (-73.9556, 40.7818),
        141: (-73.9562, 40.7818), 142: (-73.9704, 40.7538), 143: (-73.9817, 40.7616),
        144: (-73.9713, 40.7555), 148: (-73.9712, 40.7580), 151: (-73.9757, 40.7498),
        152: (-73.9718, 40.7577), 158: (-73.9557, 40.7809), 161: (-73.9680, 40.7622),
        162: (-73.9839, 40.7539), 163: (-73.9869, 40.7435), 164: (-73.9702, 40.7916),
        166: (-73.9793, 40.7444), 170: (-73.9550, 40.7777), 186: (-73.9803, 40.7330),
        194: (-73.9545, 40.7840), 202: (-73.9640, 40.8007), 209: (-73.9560, 40.7825),
        211: (-73.9680, 40.7575), 224: (-73.9632, 40.7575), 229: (-73.9827, 40.7518),
        230: (-73.9702, 40.7506), 231: (-73.9862, 40.7307), 232: (-73.9857, 40.7397),
        233: (-73.9862, 40.7527), 234: (-73.9876, 40.7541), 236: (-73.9702, 40.7862),
        237: (-73.9767, 40.7576), 238: (-73.9702, 40.7625), 239: (-73.9820, 40.7661),
        243: (-73.9834, 40.7588), 244: (-73.9823, 40.7538), 246: (-73.9786, 40.7468),
        249: (-73.9675, 40.7541), 261: (-73.9652, 40.7601), 262: (-73.9595, 40.7782),
        263: (-73.9857, 40.7593),
    }
    rows = [{"LocationID": k, "longitude": v[0], "latitude": v[1]} for k, v in data.items()]
    return pd.DataFrame(rows)


def _join_zone_centroids(df: pd.DataFrame) -> pd.DataFrame:
    centroids = _get_zone_centroids()

    pu = centroids.rename(columns={"longitude": "pickup_longitude", "latitude": "pickup_latitude"})
    df = df.merge(pu, left_on="PULocationID", right_on="LocationID", how="left")
    if "LocationID" in df.columns:
        df = df.drop(columns=["LocationID"])

    do = centroids.rename(columns={"longitude": "dropoff_longitude", "latitude": "dropoff_latitude"})
    df = df.merge(do, left_on="DOLocationID", right_on="LocationID", how="left")
    if "LocationID" in df.columns:
        df = df.drop(columns=["LocationID"])

    return df


def main() -> None:
    cfg = get_config()["data"]
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    bbox = cfg["manhattan_bbox"]

    for month in cfg["months"]:
        fname = f"{cfg['taxi_type']}_tripdata_{month}.parquet"
        url = f"{cfg['tlc_base_url']}/{fname}"
        raw_path = raw_dir / fname
        out_path = processed_dir / f"clean_{fname}"
        download_file(url, raw_path)
        clean_month(raw_path, out_path, bbox)

    print("\nDone.")


if __name__ == "__main__":
    main()
