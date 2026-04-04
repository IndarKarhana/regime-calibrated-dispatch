#!/usr/bin/env python3
"""Quick 1-scenario test: chi_jan_weekday_am with NYC library, volume-matched."""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.regime.ingest import build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.events import annotate_events
from src.regime.similarity import query_library
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.engine import SimulationEngine
from src.simulator.routing import HaversineClient
from src.policies.batch import BatchMatchingPolicy
from src.policies.anticipatory import AnticipatoryReposition
from src.evaluation.metrics import compute_kpis

cfg = get_config()

# Load NYC library
lib = RegimeLibrary()
lib.load()

# Load Chicago
proc = Path(cfg["data"]["processed_dir"])
chi_files = sorted(proc.glob("clean_chicago*.parquet"))
chi = pd.concat([pd.read_parquet(f) for f in chi_files], ignore_index=True)
chi["pickup_datetime"] = pd.to_datetime(chi["pickup_datetime"])
chi["dropoff_datetime"] = pd.to_datetime(chi["dropoff_datetime"])

chi_prof = build_demand_profile(chi)
chi_blocks = split_into_blocks(chi_prof)
chi_bbox = cfg["data"]["chicago_core_bbox"]

# Find chi_jan_weekday_am
block = None
for blk in chi_blocks:
    bid = blk["block_id"].iloc[0]
    if "2024-01" not in bid or "08-12" not in bid: continue
    ts = pd.Timestamp(bid.rsplit("_", 1)[0])
    if ts.dayofweek < 5 and not (ts.month == 1 and ts.day == 1):
        block = blk; break

bid = block["block_id"].iloc[0]
q_series = block["request_count"].values.astype(np.float64)
q_events = annotate_events(q_series)
date_str, hour_range = bid.rsplit("_", 1)
sh, eh = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
horizon = float((eh - sh) * 3600)

# Get replay trips
target_date = pd.Timestamp(date_str).date()
mask = (chi["pickup_datetime"].dt.date == target_date)
mask &= (chi["pickup_datetime"].dt.hour >= sh) & (chi["pickup_datetime"].dt.hour < eh)
replay_trips = chi[mask].copy()
n_trips = len(replay_trips)
fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

print(f"Block: {bid}, trips={n_trips}, fleet={fleet}")

# NYC library match
matched = query_library(lib, q_series, q_events, q_block_id=bid)
mrecs = [lib[b] for b, _ in matched]
mscores = [s for _, s in matched]
prior = build_calibrated_prior(mrecs, mscores, q_series)

# Override spatial pool with Chicago OD
n_sample = min(2000, len(replay_trips))
od_sample = replay_trips.sample(n=n_sample, random_state=42)
chi_od_pairs = list(zip(
    od_sample["pickup_longitude"].values.tolist(),
    od_sample["pickup_latitude"].values.tolist(),
    od_sample["dropoff_longitude"].values.tolist(),
    od_sample["dropoff_latitude"].values.tolist(),
))
prior.override_spatial_pool(chi_od_pairs)

router = HaversineClient()
seed = 42

# 1) Replay baseline
jittered = replay_trips.copy()
rng_j = np.random.default_rng(seed)
jitter = pd.to_timedelta(rng_j.uniform(0, 900, size=len(jittered)), unit="s")
jittered["pickup_datetime"] = jittered["pickup_datetime"] + jitter
stream = ReplayDemandStream(jittered)
engine = SimulationEngine(
    demand_stream=stream, policy=BatchMatchingPolicy(), router=router,
    fleet_size=fleet, horizon_seconds=horizon, seed=seed,
    bbox=chi_bbox,
)
t0 = time.time()
state = engine.run()
kpi_replay = compute_kpis(state, time.time() - t0)
print(f"REPLAY:   wait={kpi_replay['mean_wait_s']:.1f}s  comp={kpi_replay['completion_rate']:.1%}  total_req={kpi_replay['total_requests']}")

# 2) NYC cal (volume-matched)
rng = np.random.default_rng(seed)
pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
stream = CalibratedDemandStream(pr, horizon, rng=rng)
engine = SimulationEngine(
    demand_stream=stream, policy=BatchMatchingPolicy(), router=router,
    fleet_size=fleet, horizon_seconds=horizon, seed=seed,
    bbox=chi_bbox,
)
t0 = time.time()
state = engine.run()
kpi_cal = compute_kpis(state, time.time() - t0)
print(f"NYC_CAL:  wait={kpi_cal['mean_wait_s']:.1f}s  comp={kpi_cal['completion_rate']:.1%}  total_req={kpi_cal['total_requests']}")

# 3) NYC cal + LP (volume-matched)
rng = np.random.default_rng(seed)
pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
lp = AnticipatoryReposition(pr, lookahead_minutes=5.0, max_move_fraction=0.50, bbox=chi_bbox)
stream = CalibratedDemandStream(pr, horizon, rng=rng)
engine = SimulationEngine(
    demand_stream=stream, policy=BatchMatchingPolicy(), router=router,
    fleet_size=fleet, horizon_seconds=horizon, seed=seed,
    reposition_policy=lp, reposition_interval_steps=6,
    bbox=chi_bbox,
)
t0 = time.time()
state = engine.run()
kpi_lp = compute_kpis(state, time.time() - t0)
print(f"NYC_LP:   wait={kpi_lp['mean_wait_s']:.1f}s  comp={kpi_lp['completion_rate']:.1%}  total_req={kpi_lp['total_requests']}")

# Summary
print(f"\n=== SUMMARY (chi_jan_weekday_am, single seed) ===")
print(f"Replay:  {kpi_replay['mean_wait_s']:.1f}s wait, {kpi_replay['completion_rate']:.1%} comp, {kpi_replay['total_requests']} reqs")
print(f"Cal:     {kpi_cal['mean_wait_s']:.1f}s wait, {kpi_cal['completion_rate']:.1%} comp, {kpi_cal['total_requests']} reqs")
print(f"Cal+LP:  {kpi_lp['mean_wait_s']:.1f}s wait, {kpi_lp['completion_rate']:.1%} comp, {kpi_lp['total_requests']} reqs")
cal_vs = (kpi_replay['mean_wait_s'] - kpi_lp['mean_wait_s']) / kpi_replay['mean_wait_s'] * 100
print(f"Cal+LP vs Replay: {cal_vs:+.1f}% wait improvement")
