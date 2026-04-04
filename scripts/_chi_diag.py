#!/usr/bin/env python3
"""One-shot diagnostic: why does NYC-library Chicago calibration produce 3x trips?"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.regime.ingest import build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.events import annotate_events
from src.regime.similarity import query_library
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume

cfg = get_config()
bin_sec = cfg["regime"]["bin_interval_minutes"] * 60
print(f"bin_sec={bin_sec}")

# NYC library
lib = RegimeLibrary()
lib.load()
print(f"NYC lib: {len(lib)} records")

# Chicago trips
proc = Path(cfg["data"]["processed_dir"])
chi_files = sorted(proc.glob("clean_chicago*.parquet"))
if not chi_files:
    print("NO CHICAGO DATA")
    sys.exit(0)
chi = pd.concat([pd.read_parquet(f) for f in chi_files], ignore_index=True)
chi["pickup_datetime"] = pd.to_datetime(chi["pickup_datetime"])
chi["dropoff_datetime"] = pd.to_datetime(chi["dropoff_datetime"])
print(f"Chi trips: {len(chi)}")

chi_prof = build_demand_profile(chi)
chi_blocks = split_into_blocks(chi_prof)

# Find chi_jan_weekday_am
block = None
for blk in chi_blocks:
    bid = blk["block_id"].iloc[0]
    if "2024-01" not in bid or "08-12" not in bid:
        continue
    ts = pd.Timestamp(bid.rsplit("_", 1)[0])
    if ts.dayofweek < 5 and not (ts.month == 1 and ts.day == 1):
        block = blk
        break

bid = block["block_id"].iloc[0]
q_series = block["request_count"].values.astype(np.float64)
q_events = annotate_events(q_series)

date_str, hour_range = bid.rsplit("_", 1)
sh, eh = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
horizon = float((eh - sh) * 3600)

mask = chi["pickup_datetime"].dt.date == pd.Timestamp(date_str).date()
mask &= (chi["pickup_datetime"].dt.hour >= sh) & (chi["pickup_datetime"].dt.hour < eh)
n_trips = int(mask.sum())

print(f"\n=== Block: {bid} ===")
print(f"horizon={horizon}s ({horizon/3600:.0f}h), replay_trips={n_trips}")
print(f"q_series: len={len(q_series)}, sum={q_series.sum():.0f}, mean={q_series.mean():.1f}")

# NYC match
matched = query_library(lib, q_series, q_events, q_block_id=bid)
mrecs = [lib[b] for b, _ in matched]
mscores = [s for _, s in matched]
print(f"\nNYC matched: {len(mrecs)} records")
for r in mrecs[:5]:
    print(f"  {r.block_id}: series len={len(r.demand_series)}, sum={r.demand_series.sum():.0f}")

# Build prior (BEFORE volume matching)
prior = build_calibrated_prior(mrecs, mscores, q_series)
rp = prior.rate_profile
end_b = int(horizon / bin_sec)
print(f"\nrate_profile (raw prior): len={len(rp)}, end_b={end_b}")
print(f"  sum[:end_b]={rp[:end_b].sum():.1f}")
print(f"  sum[end_b:]={rp[end_b:].sum():.1f}")
print(f"  total={rp.sum():.1f}")

# Generate WITHOUT volume matching
rng = np.random.default_rng(42)
reqs_raw = prior.sample_requests(horizon, rng)
print(f"  Generated (NO volume match): {len(reqs_raw)}")

# Volume match
pr = prior.fork()
pr.match_expected_total_for_horizon(n_trips, horizon)
rp2 = pr.rate_profile
print(f"\nrate_profile (after match, target={n_trips}):")
print(f"  sum[:end_b]={rp2[:end_b].sum():.1f}")
print(f"  sum[end_b:]={rp2[end_b:].sum():.1f}")
print(f"  total={rp2.sum():.1f}")

rng2 = np.random.default_rng(42)
reqs_matched = pr.sample_requests(horizon, rng2)
print(f"  Generated (WITH volume match): {len(reqs_matched)}")

# Also check: does the CalibratedDemandStream use total_requests from somewhere else?
from src.simulator.demand import CalibratedDemandStream
rng3 = np.random.default_rng(42)
stream = CalibratedDemandStream(pr, horizon, rng=rng3)
print(f"  CalibratedDemandStream.total_requests: {stream.total_requests}")

print(f"\n=== DIAGNOSIS ===")
if len(reqs_matched) > n_trips * 1.5:
    print(f"BUG: volume matching NOT working. Got {len(reqs_matched)}, expected ~{n_trips}")
    print(f"  Likely cause: rate_profile sum[:end_b] = {rp2[:end_b].sum():.1f} vs target {n_trips}")
elif abs(len(reqs_matched) - n_trips) < n_trips * 0.2:
    print(f"Volume matching WORKS here. Got {len(reqs_matched)}, expected ~{n_trips}")
    print("If CSV shows 44K, the bug is in the eval script (not calling match, or calling on wrong prior)")
else:
    print(f"Partial match: {len(reqs_matched)} vs {n_trips} — investigate further")
