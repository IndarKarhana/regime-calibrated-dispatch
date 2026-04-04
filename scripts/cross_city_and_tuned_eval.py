#!/usr/bin/env python3
"""Cross-city generalization + tuned LP + OSRM evaluation.

Part A: Re-run NYC pipeline with tuned LP params (move_frac=0.50, lookahead=5min)
Part B: Cross-city generalization (NYC library -> Chicago test blocks)
Part C: OSRM vs Haversine comparison (if OSRM is available)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary, RegimeRecord, _compute_summary_features, _compute_ecdf
from src.regime.events import annotate_events, detect_events
from src.regime.similarity import query_library
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import ReplayDemandStream, CalibratedDemandStream
from src.simulator.engine import SimulationEngine
from src.simulator.routing import HaversineClient, OSRMClient, GridCachedOSRMClient
from src.policies.batch import BatchMatchingPolicy
from src.policies.anticipatory import AnticipatoryReposition
from src.evaluation.metrics import compute_kpis


def _pick_block(blocks, month_prefix: str, hour_range: str, day_type: str):
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        if not bid.startswith(month_prefix):
            continue
        if hour_range not in bid:
            continue
        total = blk["request_count"].sum()
        if total < 500:
            continue
        date_str = bid.rsplit("_", 1)[0]
        ts = pd.Timestamp(date_str)
        dow = ts.dayofweek
        if day_type == "weekday" and dow < 5 and not (ts.month == 1 and ts.day == 1):
            return blk
        elif day_type == "weekend" and dow >= 5:
            return blk
        elif day_type == "nye" and ts.month == 1 and ts.day == 1:
            return blk
        elif day_type == "fri_night" and dow in (4, 5):
            return blk
    return None


NYC_SCENARIOS = {
    "jan_weekday_am":  ("2024-01", "08-12", "weekday"),
    "jan_weekday_pm":  ("2024-01", "16-20", "weekday"),
    "jan_nye_am":      ("2024-01", "08-12", "nye"),
    "jan_nye_pm":      ("2024-01", "16-20", "nye"),
    "jan_weekend_mid": ("2024-01", "12-16", "weekend"),
    "jun_weekday_am":  ("2024-06", "08-12", "weekday"),
    "jun_weekday_pm":  ("2024-06", "16-20", "weekday"),
    "jun_late_night":  ("2024-06", "20-24", "fri_night"),
}


def _get_replay_trips(all_trips, bid):
    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    mask = (all_trips["pickup_datetime"].dt.date == target_date)
    if end_h > start_h:
        mask &= (all_trips["pickup_datetime"].dt.hour >= start_h)
        mask &= (all_trips["pickup_datetime"].dt.hour < end_h)
    else:
        mask &= ((all_trips["pickup_datetime"].dt.hour >= start_h) |
                  (all_trips["pickup_datetime"].dt.hour < end_h))
    return all_trips[mask]


def _run_one(stream, fleet, horizon, seed, router, repo_policy=None, bbox=None):
    dispatch = BatchMatchingPolicy()
    engine = SimulationEngine(
        demand_stream=stream, policy=dispatch, router=router,
        fleet_size=fleet, horizon_seconds=horizon, seed=seed,
        reposition_policy=repo_policy, reposition_interval_steps=6,
        bbox=bbox,
    )
    t0 = time.time()
    state = engine.run()
    return compute_kpis(state, time.time() - t0)


def check_osrm() -> bool:
    """Check if OSRM is reachable (host/port from config; docker-compose maps 5050->5000)."""
    import requests
    cfg = get_config()["osrm"]
    try:
        url = f"{cfg['host']}:{cfg['port']}/nearest/v1/car/-73.985,40.748"
        r = requests.get(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Part A: Tuned LP on NYC ─────────────────────────────────────────────────

def run_tuned_lp(all_blocks, all_trips, library, router, seeds):
    print("\n" + "=" * 80)
    print("PART A: TUNED LP (move_frac=0.50, lookahead=5min) ON NYC")
    print("=" * 80)

    rows = []
    for scen_name, (month, hr, dtype) in NYC_SCENARIOS.items():
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            continue

        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].values.astype(np.float64)
        q_events = annotate_events(q_series)

        date_str, hour_range = bid.rsplit("_", 1)
        start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
        if end_h <= start_h:
            end_h += 24
        horizon = float((end_h - start_h) * 3600)

        replay_trips = _get_replay_trips(all_trips, bid)
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

        matched = query_library(library, q_series, q_events, q_block_id=bid)
        mrecs = [library[b] for b, _ in matched]
        mscores = [s for _, s in matched]
        prior = build_calibrated_prior(mrecs, mscores, q_series)

        configs = [
            ("replay",         "replay", None),
            ("cal_only",       "cal",    None),
            ("cal+lp_tuned",   "cal", lambda pr: AnticipatoryReposition(
                pr, lookahead_minutes=5.0, max_move_fraction=0.50,
            )),
        ]

        print(f"\n  {scen_name} [{bid}] fleet={fleet}")
        for config_name, stream_type, repo_factory in configs:
            seed_results = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                if stream_type == "replay":
                    stream = ReplayDemandStream(replay_trips)
                    repo = None
                else:
                    pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
                    stream = CalibratedDemandStream(pr, horizon, rng=rng)
                    repo = repo_factory(pr) if repo_factory else None
                kpi = _run_one(stream, fleet, horizon, seed, router, repo)
                kpi["scenario"] = scen_name
                kpi["config"] = config_name
                kpi["fleet"] = fleet
                kpi["seed"] = seed
                rows.append(kpi)
                seed_results.append(kpi)
            avg_w = np.mean([r["mean_wait_s"] for r in seed_results])
            avg_c = np.mean([r["completion_rate"] for r in seed_results])
            print(f"    {config_name:18s}: wait={avg_w:6.1f}s  comp={avg_c:.1%}")

    df = pd.DataFrame(rows)
    if not df.empty:
        replay = df[df["config"] == "replay"].groupby("scenario")["mean_wait_s"].mean()
        cal_only = df[df["config"] == "cal_only"].groupby("scenario")["mean_wait_s"].mean()
        cal_lp = df[df["config"] == "cal+lp_tuned"].groupby("scenario")["mean_wait_s"].mean()
        combined = ((replay - cal_lp) / replay * 100).mean()
        cal_imp = ((replay - cal_only) / replay * 100).mean()
        lp_imp = ((cal_only - cal_lp) / cal_only * 100).mean()
        print(f"\n  TUNED LP SUMMARY:")
        print(f"    Calibration alone:  {cal_imp:+.1f}% vs replay")
        print(f"    Tuned LP alone:     {lp_imp:+.1f}% vs cal-only")
        print(f"    Combined:           {combined:+.1f}% vs replay")
    return df


# ── Part B: Cross-city generalization ────────────────────────────────────────

def load_chicago(month_tag: str | None = None):
    """Load cleaned Chicago TNP data (``clean_chicago_tripdata_*.parquet``)."""
    proc = Path(get_config()["data"]["processed_dir"])
    files = sorted(proc.glob("clean_chicago*.parquet"))
    if month_tag:
        files = [f for f in files if month_tag in f.name]
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    return df


def build_chicago_library(chi_trips):
    """Build a regime library from Chicago data."""
    profile = build_demand_profile(chi_trips)
    blocks = split_into_blocks(profile)
    lib = RegimeLibrary()
    # Index by date once — avoid O(n_trips) scan per block
    chi_trips = chi_trips.copy()
    chi_trips["_d"] = chi_trips["pickup_datetime"].dt.date
    by_date = {d: g for d, g in chi_trips.groupby("_d", sort=False)}

    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        q_series = blk["request_count"].values.astype(np.float64)
        if len(q_series) < 10 or q_series.sum() < 100:
            continue
        events = detect_events(q_series)
        date_str = bid.rsplit("_", 1)[0]
        ts = pd.Timestamp(date_str)

        ecdf_values = _compute_ecdf(q_series)
        summary_features = _compute_summary_features(q_series)
        bin_starts = [str(b) for b in blk["bin_start"].values]

        hour_range = bid.rsplit("_", 1)[1]
        sh, eh = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])

        day_df = by_date.get(ts.date())
        if day_df is None or len(day_df) == 0:
            continue
        trip_mask = (day_df["pickup_datetime"].dt.hour >= sh) & (
            day_df["pickup_datetime"].dt.hour < eh
        )
        block_trips = day_df.loc[trip_mask]

        n_od = min(2000, len(block_trips))
        meta = {
            "date": date_str,
            "hour_range": hour_range,
            "total_demand": float(q_series.sum()),
        }
        if n_od > 0:
            sample = block_trips.sample(n=n_od, random_state=42)
            meta["pickup_lons"] = sample["pickup_longitude"].tolist()
            meta["pickup_lats"] = sample["pickup_latitude"].tolist()
            meta["dropoff_lons"] = sample["dropoff_longitude"].tolist()
            meta["dropoff_lats"] = sample["dropoff_latitude"].tolist()

        rec = RegimeRecord(
            block_id=bid,
            demand_series=q_series,
            ecdf_values=ecdf_values,
            summary_features=summary_features,
            events=events,
            bin_starts=bin_starts,
            metadata=meta,
        )
        lib.records[bid] = rec

    return lib


def run_cross_city(nyc_library, router, seeds):
    """Test NYC library matching on Chicago test blocks."""
    print("\n" + "=" * 80)
    print("PART B: CROSS-CITY GENERALIZATION (NYC library -> Chicago)")
    print("=" * 80)
    print("  Calibrated streams: expected demand matched to replay trip count per block.")

    chi_bbox = get_config()["data"]["chicago_core_bbox"]
    chi_trips = load_chicago()
    if chi_trips is None:
        print("  [SKIP] Chicago data not found. Run: python scripts/download_chicago.py")
        return pd.DataFrame()

    print(f"  Chicago trips: {len(chi_trips):,}")

    chi_profile = build_demand_profile(chi_trips)
    chi_blocks = split_into_blocks(chi_profile)
    print(f"  Chicago blocks: {len(chi_blocks)}")

    chi_scenarios = {
        "chi_jan_weekday_am": ("2024-01", "08-12", "weekday"),
        "chi_jan_weekday_pm": ("2024-01", "16-20", "weekday"),
        "chi_jan_weekend_mid": ("2024-01", "12-16", "weekend"),
    }

    chi_lib = build_chicago_library(chi_trips)
    print(f"  Chicago library: {len(chi_lib)} regimes")

    rows = []
    for scen_name, (month, hr, dtype) in chi_scenarios.items():
        block = _pick_block(chi_blocks, month, hr, dtype)
        if block is None:
            print(f"  SKIP {scen_name}: no block found")
            continue

        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].values.astype(np.float64)
        q_events = annotate_events(q_series)

        date_str, hour_range = bid.rsplit("_", 1)
        start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
        horizon = float((end_h - start_h) * 3600)

        replay_trips = _get_replay_trips(chi_trips, bid)
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

        print(f"\n  {scen_name} [{bid}] trips={n_trips} fleet={fleet}")

        # NYC library matching (cross-city)
        matched_nyc = query_library(nyc_library, q_series, q_events, q_block_id=bid)
        # Chicago self-library matching
        matched_chi = query_library(chi_lib, q_series, q_events, q_block_id=bid)

        # Collect Chicago OD pairs from test block for spatial override
        chi_block_trips = _get_replay_trips(chi_trips, bid)
        n_sample = min(2000, len(chi_block_trips))
        if n_sample > 0:
            od_sample = chi_block_trips.sample(n=n_sample, random_state=42)
            chi_od_pairs = list(zip(
                od_sample["pickup_longitude"].values.tolist(),
                od_sample["pickup_latitude"].values.tolist(),
                od_sample["dropoff_longitude"].values.tolist(),
                od_sample["dropoff_latitude"].values.tolist(),
            ))
        else:
            chi_od_pairs = []

        for lib_name, matched, lib_ref in [
            ("nyc_library", matched_nyc, nyc_library),
            ("chi_library", matched_chi, chi_lib),
        ]:
            mrecs = [lib_ref[b] for b, _ in matched]
            mscores = [s for _, s in matched]
            if not mrecs:
                continue
            prior = build_calibrated_prior(mrecs, mscores, q_series)
            if lib_name == "nyc_library" and chi_od_pairs:
                prior.override_spatial_pool(chi_od_pairs)

            for config_name, stream_type, use_lp in [
                (f"{lib_name}_cal",    "cal", False),
                (f"{lib_name}_cal+lp", "cal", True),
            ]:
                seed_results = []
                for seed in seeds[:3]:
                    rng = np.random.default_rng(seed)
                    pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
                    stream = CalibratedDemandStream(pr, horizon, rng=rng)
                    repo = (
                        AnticipatoryReposition(
                            pr, lookahead_minutes=5.0, max_move_fraction=0.50,
                            bbox=chi_bbox,
                        )
                        if use_lp
                        else None
                    )
                    kpi = _run_one(stream, fleet, horizon, seed, router, repo, bbox=chi_bbox)
                    kpi["scenario"] = scen_name
                    kpi["config"] = config_name
                    kpi["seed"] = seed
                    rows.append(kpi)
                    seed_results.append(kpi)
                avg_w = np.mean([r["mean_wait_s"] for r in seed_results])
                print(f"    {config_name:25s}: wait={avg_w:6.1f}s")

        # Replay baseline (add jitter for Chicago's 15-min-rounded timestamps)
        for seed in seeds[:3]:
            jittered = replay_trips.copy()
            rng_j = np.random.default_rng(seed)
            jitter = pd.to_timedelta(rng_j.uniform(0, 900, size=len(jittered)), unit="s")
            jittered["pickup_datetime"] = jittered["pickup_datetime"] + jitter
            stream = ReplayDemandStream(jittered)
            kpi = _run_one(stream, fleet, horizon, seed, router, None, bbox=chi_bbox)
            kpi["scenario"] = scen_name
            kpi["config"] = "replay"
            kpi["seed"] = seed
            rows.append(kpi)

        replay_w = np.mean([r["mean_wait_s"] for r in rows
                           if r["scenario"] == scen_name and r["config"] == "replay"])
        print(f"    {'replay':25s}: wait={replay_w:6.1f}s")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  CROSS-CITY SUMMARY:")
        for cfg in df["config"].unique():
            subset = df[df["config"] == cfg]
            avg = subset["mean_wait_s"].mean()
            print(f"    {cfg:25s}: avg wait={avg:.1f}s")
    return df


# ── Part C: OSRM vs Haversine ───────────────────────────────────────────────

def run_osrm_comparison(all_blocks, all_trips, library, seeds):
    """Compare OSRM vs Haversine on 2 representative NYC scenarios."""
    print("\n" + "=" * 80)
    print("PART C: OSRM vs HAVERSINE COMPARISON")
    print("=" * 80)

    oc = get_config()["osrm"]
    if not check_osrm():
        print(f"  [SKIP] OSRM not reachable at {oc['host']}:{oc['port']} "
              f"(run: make osrm-prepare && docker compose up -d osrm)")
        return pd.DataFrame()

    print("  Building grid-cached OSRM router (one-time cost) ...")
    osrm_router = GridCachedOSRMClient(grid_size=25)
    haver_router = HaversineClient()

    test_scenarios = [
        ("jan_weekday_pm", "2024-01", "16-20", "weekday"),
        ("jun_weekday_am", "2024-06", "08-12", "weekday"),
    ]

    rows = []
    for scen_name, month, hr, dtype in test_scenarios:
        block = _pick_block(all_blocks, month, hr, dtype)
        if block is None:
            continue

        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].values.astype(np.float64)
        q_events = annotate_events(q_series)

        date_str, hour_range = bid.rsplit("_", 1)
        start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
        horizon = float((end_h - start_h) * 3600)

        replay_trips = _get_replay_trips(all_trips, bid)
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / (horizon / 3600) * 0.15), 50)

        matched = query_library(library, q_series, q_events, q_block_id=bid)
        mrecs = [library[b] for b, _ in matched]
        mscores = [s for _, s in matched]
        prior = build_calibrated_prior(mrecs, mscores, q_series)

        print(f"\n  {scen_name} [{bid}] fleet={fleet}")
        for router_name, router in [("haversine", haver_router), ("osrm", osrm_router)]:
            for config_name, stream_type, use_lp in [
                ("replay", "replay", False),
                ("cal+lp", "cal", True),
            ]:
                rng = np.random.default_rng(42)
                if stream_type == "replay":
                    stream = ReplayDemandStream(replay_trips)
                    repo = None
                else:
                    pr = prior_matched_to_replay_volume(prior, horizon, n_trips)
                    stream = CalibratedDemandStream(pr, horizon, rng=rng)
                    repo = (
                        AnticipatoryReposition(
                            pr, lookahead_minutes=5.0, max_move_fraction=0.50,
                        )
                        if use_lp
                        else None
                    )
                kpi = _run_one(stream, fleet, horizon, 42, router, repo)
                kpi["scenario"] = scen_name
                kpi["config"] = config_name
                kpi["router"] = router_name
                rows.append(kpi)
                print(f"    {router_name:10s} {config_name:10s}: wait={kpi['mean_wait_s']:6.1f}s  "
                      f"comp={kpi['completion_rate']:.1%}")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  OSRM vs HAVERSINE IMPACT:")
        for router_name in ["haversine", "osrm"]:
            sub = df[(df["router"] == router_name) & (df["config"] == "cal+lp")]
            avg = sub["mean_wait_s"].mean()
            print(f"    {router_name:10s} cal+lp avg wait: {avg:.1f}s")
    return df


def main():
    ap = argparse.ArgumentParser(description="Tuned LP, cross-city Chicago, optional OSRM compare")
    ap.add_argument(
        "--router", choices=("haversine", "osrm"), default="haversine",
        help="Routing for NYC + Chicago sims (OSRM is NYC road graph only; Chi still Haversine if OSRM)",
    )
    args = ap.parse_args()

    cfg = get_config()
    seeds = cfg["evaluation"]["seeds"]
    if args.router == "osrm" and check_osrm():
        router = OSRMClient()
        print("Using OSRM for routing (NYC graph).")
    elif args.router == "osrm":
        print("OSRM not reachable; using Haversine.")
        router = HaversineClient()
    else:
        router = HaversineClient()
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(exist_ok=True)

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
    print(f"  {len(all_blocks)} blocks")

    # Part A
    tuned_df = run_tuned_lp(all_blocks, all_trips, library, router, seeds)
    if not tuned_df.empty:
        tuned_df.to_csv(out_dir / "tuned_lp_results.csv", index=False)

    # Part B
    cross_df = run_cross_city(library, router, seeds)
    if not cross_df.empty:
        cross_df.to_csv(out_dir / "cross_city_results.csv", index=False)

    # Part C
    osrm_df = run_osrm_comparison(all_blocks, all_trips, library, seeds)
    if not osrm_df.empty:
        osrm_df.to_csv(out_dir / "osrm_comparison.csv", index=False)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    if not tuned_df.empty:
        replay = tuned_df[tuned_df["config"] == "replay"].groupby("scenario")["mean_wait_s"].mean()
        cal_lp = tuned_df[tuned_df["config"] == "cal+lp_tuned"].groupby("scenario")["mean_wait_s"].mean()
        if not replay.empty and not cal_lp.empty:
            combined = ((replay - cal_lp) / replay * 100).mean()
            print(f"  NYC tuned pipeline:    {combined:+.1f}% vs replay (target: >30%)")
    if not cross_df.empty:
        replay_chi = cross_df[cross_df["config"] == "replay"].groupby("scenario")["mean_wait_s"].mean()
        nyc_chi = cross_df[cross_df["config"] == "nyc_library_cal+lp"].groupby("scenario")["mean_wait_s"].mean()
        chi_chi = cross_df[cross_df["config"] == "chi_library_cal+lp"].groupby("scenario")["mean_wait_s"].mean()
        if not nyc_chi.empty and not replay_chi.empty:
            cross_imp = ((replay_chi - nyc_chi) / replay_chi * 100).mean()
            self_imp = ((replay_chi - chi_chi) / replay_chi * 100).mean()
            print(f"  Cross-city (NYC->CHI): {cross_imp:+.1f}% vs replay")
            print(f"  Same-city (CHI->CHI):  {self_imp:+.1f}% vs replay")
            transfer_ratio = cross_imp / self_imp * 100 if self_imp != 0 else 0
            print(f"  Transfer efficiency:   {transfer_ratio:.0f}% of same-city performance")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
