#!/usr/bin/env python3
"""Standalone Part C: OSRM vs Haversine comparison.

Runs only the OSRM comparison from cross_city_and_tuned_eval.py
without re-running Parts A (tuned LP) or B (cross-city).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.events import annotate_events
from src.regime.similarity import query_library
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.routing import HaversineClient, OSRMClient, GridCachedOSRMClient

# Import shared helpers from the main eval script
from cross_city_and_tuned_eval import (
    _pick_block, _get_replay_trips, _run_one, check_osrm,
    run_osrm_comparison,
)


def main():
    cfg = get_config()
    seeds = cfg["evaluation"]["seeds"]
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(exist_ok=True)

    print("Checking OSRM ...")
    if not check_osrm():
        print("ERROR: OSRM is not reachable. Run: docker compose up -d osrm")
        sys.exit(1)
    print("  OSRM is up.\n")

    print("Loading NYC data and regime library ...")
    library = RegimeLibrary()
    library.load()
    print(f"  {len(library)} NYC regimes loaded.")

    jan_trips = load_cleaned("2024-01")
    jun_trips = load_cleaned("2024-06")
    all_trips = pd.concat([jan_trips, jun_trips], ignore_index=True)

    jan_profile = build_demand_profile(jan_trips)
    jun_profile = build_demand_profile(jun_trips)
    jan_blocks = split_into_blocks(jan_profile)
    jun_blocks = split_into_blocks(jun_profile)
    all_blocks = jan_blocks + jun_blocks
    print(f"  {len(all_blocks)} blocks\n")

    osrm_df = run_osrm_comparison(all_blocks, all_trips, library, seeds)
    if not osrm_df.empty:
        osrm_df.to_csv(out_dir / "osrm_comparison.csv", index=False)
        print(f"\nResults saved to {out_dir / 'osrm_comparison.csv'}")
    else:
        print("\nNo results produced.")


if __name__ == "__main__":
    main()
