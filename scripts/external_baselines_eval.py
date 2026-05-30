#!/usr/bin/env python3
"""Run external-style fleet repositioning baselines in our simulator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.calibrator import (  # noqa: E402
    build_calibrated_prior,
    prior_matched_to_replay_volume,
)
from src.config import get_config  # noqa: E402
from src.evaluation.metrics import compute_kpis  # noqa: E402
from src.policies.anticipatory import AnticipatoryReposition  # noqa: E402
from src.policies.batch import BatchMatchingPolicy  # noqa: E402
from src.policies.external_baselines import (  # noqa: E402
    ContextualDQNReposition,
    ForecastShareLPReposition,
    GPRChanceMPCReposition,
    OracleMPCReposition,
    ScenarioChanceMPCReposition,
    WenStyleHistoricalRebalancing,
)
from src.policies.robust_reposition import (  # noqa: E402
    RetrievalRobustBudgetReposition,
    RetrievalRobustCVaRReposition,
)
from src.regime.events import annotate_events  # noqa: E402
from src.regime.ingest import build_demand_profile, load_cleaned, split_into_blocks  # noqa: E402
from src.regime.learned_weights import PATH2_SCENARIOS, load_gate  # noqa: E402
from src.regime.similarity import compute_similarity  # noqa: E402
from src.regime.store import RegimeLibrary  # noqa: E402
from src.simulator.demand import ReplayDemandStream  # noqa: E402
from src.simulator.engine import SimulationEngine  # noqa: E402
from src.simulator.routing import GridCachedOSRMClient, HaversineClient, OSRMClient  # noqa: E402


def _load_zone_buffer(path: str, alpha: str) -> np.ndarray | None:
    if not path:
        return None
    with open(path) as f:
        payload = json.load(f)
    alphas = payload.get("alphas", {})
    key = f"{float(alpha):.2f}"
    if key not in alphas:
        available = ", ".join(sorted(alphas))
        raise KeyError(f"alpha {key} not in {path}; available: {available}")
    return np.asarray(alphas[key]["zone_buffer"], dtype=float)


def _pick_block(blocks, month_prefix: str, hour_range: str, day_type: str):
    for block in blocks:
        bid = block["block_id"].iloc[0]
        if not bid.startswith(month_prefix) or hour_range not in bid:
            continue
        if block["request_count"].sum() < 500:
            continue
        ts = pd.Timestamp(bid.rsplit("_", 1)[0])
        if ts.month == 1 and ts.day == 1:
            dtype = "holiday"
        elif ts.dayofweek >= 5:
            dtype = "weekend"
        else:
            dtype = "weekday"
        if dtype == day_type:
            return block
    return None


def _get_replay_trips(all_trips: pd.DataFrame, bid: str) -> pd.DataFrame:
    date_str, hour_range = bid.rsplit("_", 1)
    start_h, end_h = int(hour_range.split("-")[0]), int(hour_range.split("-")[1])
    target_date = pd.Timestamp(date_str).date()
    mask = all_trips["pickup_datetime"].dt.date == target_date
    mask &= all_trips["pickup_datetime"].dt.hour >= start_h
    mask &= all_trips["pickup_datetime"].dt.hour < end_h
    return all_trips[mask].copy()


def _top_matches(library: RegimeLibrary, q_series, q_events, bid: str, weights, top_k: int):
    scores = []
    for cid, record in library.records.items():
        if cid == bid:
            continue
        score = compute_similarity(
            q_series,
            q_events,
            record,
            weights=weights,
            q_block_id=bid,
        )
        scores.append((cid, score))
    scores.sort(key=lambda row: row[1], reverse=True)
    return scores[:top_k]


def _check_osrm() -> bool:
    cfg = get_config()["osrm"]
    url = f"{cfg['host']}:{cfg['port']}/nearest/v1/{cfg.get('profile', 'car')}/-73.985,40.748"
    try:
        response = requests.get(url, timeout=3)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _make_router(args: argparse.Namespace):
    if args.router == "haversine":
        args.router_actual = "haversine"
        return HaversineClient()
    if not _check_osrm():
        print("OSRM unreachable; falling back to Haversine routing.")
        args.router_actual = "haversine_fallback"
        return HaversineClient()
    if args.router == "osrm-grid":
        args.router_actual = f"osrm-grid-{args.osrm_grid_size}"
        return GridCachedOSRMClient(grid_size=args.osrm_grid_size)
    args.router_actual = "osrm"
    return OSRMClient()


def _run_one(
    replay_trips: pd.DataFrame,
    fleet: int,
    horizon: float,
    seed: int,
    repo_policy,
    reposition_interval_steps: int,
    router,
) -> dict:
    engine = SimulationEngine(
        demand_stream=ReplayDemandStream(replay_trips),
        policy=BatchMatchingPolicy(),
        router=router,
        fleet_size=fleet,
        horizon_seconds=horizon,
        seed=seed,
        reposition_policy=repo_policy,
        reposition_interval_steps=reposition_interval_steps,
    )
    start = time.time()
    state = engine.run()
    kpi = compute_kpis(state, time.time() - start)
    solve_times = getattr(repo_policy, "solve_times_s", None)
    if solve_times:
        arr = np.asarray(solve_times, dtype=float)
        kpi.update({
            "lp_solve_calls": int(arr.size),
            "lp_solve_time_mean_ms": float(arr.mean() * 1000.0),
            "lp_solve_time_p95_ms": float(np.percentile(arr, 95) * 1000.0),
        })
    else:
        kpi.update({
            "lp_solve_calls": 0,
            "lp_solve_time_mean_ms": 0.0,
            "lp_solve_time_p95_ms": 0.0,
        })
    status_counts = getattr(repo_policy, "solve_status_counts", None)
    if status_counts:
        kpi["solve_status_counts_json"] = json.dumps(status_counts, sort_keys=True)
        for status, count in status_counts.items():
            kpi[f"solve_status_{status}"] = int(count)
    cvar_budgets = getattr(repo_policy, "cvar_budget_history", None)
    if cvar_budgets:
        arr = np.asarray(cvar_budgets, dtype=float)
        kpi["dcr_cvar_budget_mean"] = float(arr.mean())
        kpi["dcr_cvar_budget_p95"] = float(np.percentile(arr, 95))
    baseline_cvars = getattr(repo_policy, "baseline_cvar_history", None)
    if baseline_cvars:
        arr = np.asarray(baseline_cvars, dtype=float)
        kpi["dcr_baseline_cvar_mean"] = float(arr.mean())
        kpi["dcr_baseline_cvar_p95"] = float(np.percentile(arr, 95))
    dominance_deltas = getattr(repo_policy, "dominance_delta_history", None)
    if dominance_deltas:
        arr = np.asarray(dominance_deltas, dtype=float)
        kpi["dcr_dominance_delta_mean"] = float(arr.mean())
        kpi["dcr_dominance_delta_p95"] = float(np.percentile(arr, 95))
        kpi["dcr_dominance_delta_max"] = float(arr.max())
        kpi["dcr_dominance_n"] = int(arr.size)
    dominance_holds = getattr(repo_policy, "dominance_holds_history", None)
    if dominance_holds:
        arr = np.asarray(dominance_holds, dtype=bool)
        kpi["dcr_dominance_hold_rate"] = float(arr.mean())
    mixture_shortages = getattr(repo_policy, "mixture_shortage_history", None)
    if mixture_shortages:
        arr = np.asarray(mixture_shortages, dtype=float)
        kpi["dcr_mixture_shortage_mean"] = float(arr.mean())
    dcr_shortage_terms = getattr(repo_policy, "dcr_shortage_term_history", None)
    if dcr_shortage_terms:
        arr = np.asarray(dcr_shortage_terms, dtype=float)
        kpi["dcr_shortage_term_mean"] = float(arr.mean())
    return kpi


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    library = RegimeLibrary()
    library.load()
    gate = load_gate(args.gate)
    zone_buffer = _load_zone_buffer(args.robust_buffer_json, args.robust_buffer_alpha)

    jan = load_cleaned("2024-01")
    jun = load_cleaned("2024-06")
    trips = pd.concat([jan, jun], ignore_index=True)
    blocks = split_into_blocks(build_demand_profile(trips))
    router = _make_router(args)

    rows = []
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    selected_scenarios = (
        {s.strip() for s in args.scenarios.split(",") if s.strip()}
        if args.scenarios else set(PATH2_SCENARIOS)
    )
    for scenario, (month, hour_range, day_type) in PATH2_SCENARIOS.items():
        if scenario not in selected_scenarios:
            continue
        block = _pick_block(blocks, month, hour_range, day_type)
        if block is None:
            print(f"SKIP {scenario}: no matching block")
            continue
        bid = block["block_id"].iloc[0]
        q_series = block["request_count"].to_numpy(dtype=np.float64)
        qrec = library[bid] if bid in library.records else None
        q_events = qrec.events if qrec is not None else annotate_events(q_series)
        replay_trips = _get_replay_trips(trips, bid)
        if replay_trips.empty:
            print(f"SKIP {scenario}: no replay trips")
            continue

        start_time = replay_trips["pickup_datetime"].min()
        horizon = 4 * 3600.0
        n_trips = len(replay_trips)
        fleet = max(int(n_trips / 4.0 * args.fleet_ratio), args.min_fleet)

        learned_weights = (
            gate(qrec.demand_series, qrec.events, bid, qrec)
            if qrec is not None else gate(q_series, q_events, bid)
        )
        spatial_matches = _top_matches(
            library, q_series, q_events, bid, learned_weights, args.top_k
        )
        hand_matches = _top_matches(library, q_series, q_events, bid, None, args.top_k)
        spatial_prior = build_calibrated_prior(
            [library[cid] for cid, _ in spatial_matches],
            [score for _, score in spatial_matches],
            q_series,
        )
        hand_prior = build_calibrated_prior(
            [library[cid] for cid, _ in hand_matches],
            [score for _, score in hand_matches],
            q_series,
        )
        spatial_prior = prior_matched_to_replay_volume(spatial_prior, horizon, n_trips)
        hand_prior = prior_matched_to_replay_volume(hand_prior, horizon, n_trips)

        print(f"\n{scenario} [{bid}] trips={n_trips} fleet={fleet}")
        factories = {
            "batch_replay": lambda: None,
            "wen2017_rebalancing": lambda: WenStyleHistoricalRebalancing(
                hand_prior,
                max_move_fraction=args.wen_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "spatial_gate_share": lambda: WenStyleHistoricalRebalancing(
                spatial_prior,
                max_move_fraction=args.wen_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "share_lp_hand_tuned": lambda: ForecastShareLPReposition(
                hand_prior,
                max_move_fraction=args.share_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "spatial_gate_share_lp": lambda: ForecastShareLPReposition(
                spatial_prior,
                max_move_fraction=args.share_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "scenario_chance_mpc": lambda: ScenarioChanceMPCReposition(
                [library[cid] for cid, _ in hand_matches],
                [score for _, score in hand_matches],
                lookahead_minutes=args.chance_lookahead,
                quantile=args.chance_quantile,
                risk_weight=args.chance_risk_weight,
                max_move_fraction=args.chance_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "gpr_chance_mpc_lite": lambda: GPRChanceMPCReposition(
                [library[cid] for cid, _ in hand_matches],
                [score for _, score in hand_matches],
                lookahead_minutes=args.gpr_chance_lookahead,
                quantile=args.gpr_chance_quantile,
                risk_weight=args.gpr_chance_risk_weight,
                max_move_fraction=args.gpr_chance_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "retrieval_robust_cvar": lambda: RetrievalRobustCVaRReposition(
                [library[cid] for cid, _ in hand_matches],
                [score for _, score in hand_matches],
                lookahead_minutes=args.robust_lookahead,
                alpha=args.robust_alpha,
                risk_weight=args.robust_risk_weight,
                mean_weight=args.robust_mean_weight,
                move_cost_multiplier=args.robust_move_cost_multiplier,
                zone_buffer=zone_buffer,
                max_move_fraction=args.robust_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "spatial_gate_robust_cvar": lambda: RetrievalRobustCVaRReposition(
                [library[cid] for cid, _ in spatial_matches],
                [score for _, score in spatial_matches],
                lookahead_minutes=args.robust_lookahead,
                alpha=args.robust_alpha,
                risk_weight=args.robust_risk_weight,
                mean_weight=args.robust_mean_weight,
                move_cost_multiplier=args.robust_move_cost_multiplier,
                zone_buffer=zone_buffer,
                max_move_fraction=args.robust_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "spatial_gate_robust_budget": lambda: RetrievalRobustBudgetReposition(
                [library[cid] for cid, _ in spatial_matches],
                [score for _, score in spatial_matches],
                lookahead_minutes=args.robust_lookahead,
                alpha=args.robust_alpha,
                shortage_reduction=args.robust_shortage_reduction,
                mean_slack_weight=args.robust_mean_slack_weight,
                zone_buffer=zone_buffer,
                max_move_fraction=args.robust_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "lin2018_contextual_dqn": lambda: ContextualDQNReposition(
                hand_prior,
                checkpoint=args.dqn_checkpoint,
                max_move_fraction=args.dqn_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "cal_lp_hand_tuned": lambda: AnticipatoryReposition(
                hand_prior,
                lookahead_minutes=5.0,
                max_move_fraction=args.raw_lp_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "spatial_gate_lp": lambda: AnticipatoryReposition(
                spatial_prior,
                lookahead_minutes=5.0,
                max_move_fraction=args.raw_lp_move_fraction,
                h3_res=args.policy_h3_res,
            ),
            "oracle_mpc": lambda: OracleMPCReposition(
                replay_trips,
                start_time=start_time,
                lookahead_minutes=args.oracle_lookahead,
                max_move_fraction=args.oracle_move_fraction,
                h3_res=args.policy_h3_res,
            ),
        }
        selected_methods = (
            {m.strip() for m in args.methods.split(",") if m.strip()}
            if args.methods else set(factories)
        )

        for method, factory in factories.items():
            if method not in selected_methods:
                continue
            waits = []
            for seed in seeds:
                kpi = _run_one(
                    replay_trips,
                    fleet,
                    horizon,
                    seed,
                    factory(),
                    args.reposition_interval_steps,
                    router,
                )
                kpi.update({
                    "scenario": scenario,
                    "block_id": bid,
                    "method": method,
                    "seed": seed,
                    "fleet": fleet,
                    "n_trips": n_trips,
                    "fleet_ratio": args.fleet_ratio,
                    "share_move_fraction": args.share_move_fraction,
                    "wen_move_fraction": args.wen_move_fraction,
                    "oracle_move_fraction": args.oracle_move_fraction,
                    "chance_move_fraction": args.chance_move_fraction,
                    "chance_quantile": args.chance_quantile,
                    "chance_risk_weight": args.chance_risk_weight,
                    "gpr_chance_move_fraction": args.gpr_chance_move_fraction,
                    "gpr_chance_quantile": args.gpr_chance_quantile,
                    "gpr_chance_risk_weight": args.gpr_chance_risk_weight,
                    "robust_move_fraction": args.robust_move_fraction,
                    "robust_alpha": args.robust_alpha,
                    "robust_risk_weight": args.robust_risk_weight,
                    "robust_mean_weight": args.robust_mean_weight,
                    "robust_move_cost_multiplier": args.robust_move_cost_multiplier,
                    "robust_shortage_reduction": args.robust_shortage_reduction,
                    "robust_mean_slack_weight": args.robust_mean_slack_weight,
                    "robust_buffer_json": args.robust_buffer_json,
                    "robust_buffer_alpha": args.robust_buffer_alpha,
                    "policy_h3_res": args.policy_h3_res,
                    "reposition_interval_steps": args.reposition_interval_steps,
                    "router": args.router,
                    "router_actual": args.router_actual,
                })
                rows.append(kpi)
                waits.append(kpi["mean_wait_s"])
            print(f"  {method:22s}: wait={np.mean(waits):6.1f}s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    summary = df.groupby("method", as_index=False).agg(
        mean_wait_s=("mean_wait_s", "mean"),
        std_wait_s=("mean_wait_s", "std"),
        completion_rate=("completion_rate", "mean"),
        mean_pickup_dist_m=("mean_pickup_dist_m", "mean"),
        reposition_dist_m_per_trip=("reposition_dist_m_per_trip", "mean"),
        reposition_dist_m_per_driver=("reposition_dist_m_per_driver", "mean"),
        zone_wait_gini=("zone_wait_gini", "mean"),
        zone_wait_p90_p10_gap_s=("zone_wait_p90_p10_gap_s", "mean"),
        mean_idle_s_per_driver=("mean_idle_s_per_driver", "mean"),
        wall_time_s=("wall_time_s", "mean"),
        lp_solve_time_mean_ms=("lp_solve_time_mean_ms", "mean"),
        lp_solve_time_p95_ms=("lp_solve_time_p95_ms", "mean"),
        n=("mean_wait_s", "size"),
    )
    replay = (
        df[df["method"] == "batch_replay"]
        .groupby("scenario")["mean_wait_s"]
        .mean()
        .rename("replay_wait_s")
    )
    by_scenario = (
        df.groupby(["scenario", "method"], as_index=False)["mean_wait_s"]
        .mean()
        .merge(replay, on="scenario", how="left")
    )
    by_scenario["improvement_vs_replay_pct"] = (
        (by_scenario["replay_wait_s"] - by_scenario["mean_wait_s"])
        / by_scenario["replay_wait_s"]
        * 100.0
    )
    method_imp = (
        by_scenario[by_scenario["method"] != "batch_replay"]
        .groupby("method", as_index=False)["improvement_vs_replay_pct"]
        .mean()
    )
    summary = summary.merge(method_imp, on="method", how="left")
    summary = summary.sort_values("mean_wait_s")
    summary.to_csv(out_dir / "external_baselines_summary.csv", index=False)
    by_scenario.to_csv(out_dir / "external_baselines_by_scenario.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="results/path2_gate_spatial/similarity_gate.pt")
    parser.add_argument("--out-dir", default="results/path2_external_baselines")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated subset of PATH2_SCENARIOS; default runs all.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fleet-ratio", type=float, default=0.15)
    parser.add_argument("--min-fleet", type=int, default=50)
    parser.add_argument("--policy-h3-res", type=int, default=None)
    parser.add_argument("--oracle-lookahead", type=float, default=15.0)
    parser.add_argument("--share-move-fraction", type=float, default=0.30)
    parser.add_argument("--wen-move-fraction", type=float, default=0.30)
    parser.add_argument("--oracle-move-fraction", type=float, default=0.50)
    parser.add_argument("--chance-lookahead", type=float, default=15.0)
    parser.add_argument("--chance-quantile", type=float, default=0.80)
    parser.add_argument("--chance-risk-weight", type=float, default=0.70)
    parser.add_argument("--chance-move-fraction", type=float, default=0.50)
    parser.add_argument("--gpr-chance-lookahead", type=float, default=15.0)
    parser.add_argument("--gpr-chance-quantile", type=float, default=0.90)
    parser.add_argument("--gpr-chance-risk-weight", type=float, default=0.80)
    parser.add_argument("--gpr-chance-move-fraction", type=float, default=0.50)
    parser.add_argument("--robust-lookahead", type=float, default=15.0)
    parser.add_argument("--robust-alpha", type=float, default=0.80)
    parser.add_argument("--robust-risk-weight", type=float, default=1.0)
    parser.add_argument("--robust-mean-weight", type=float, default=0.25)
    parser.add_argument("--robust-move-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--robust-shortage-reduction", type=float, default=0.25)
    parser.add_argument("--robust-mean-slack-weight", type=float, default=1e-3)
    parser.add_argument("--robust-buffer-json", default="")
    parser.add_argument("--robust-buffer-alpha", default="0.80")
    parser.add_argument("--robust-move-fraction", type=float, default=0.50)
    parser.add_argument("--raw-lp-move-fraction", type=float, default=0.50)
    parser.add_argument("--dqn-move-fraction", type=float, default=0.25)
    parser.add_argument("--reposition-interval-steps", type=int, default=6)
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument(
        "--router",
        choices=("haversine", "osrm-grid", "osrm"),
        default="haversine",
        help="Routing backend. OSRM options require a running local OSRM service.",
    )
    parser.add_argument("--osrm-grid-size", type=int, default=25)
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated subset of methods; default runs all.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate(args)
    df.to_csv(out_dir / "external_baselines.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    summarize(df, out_dir)


if __name__ == "__main__":
    main()
