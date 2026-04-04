"""Training loop: regime query -> calibrate -> rollout -> PPO update."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from tqdm import trange

from src.config import get_config
from src.regime.ingest import load_cleaned, build_demand_profile, split_into_blocks
from src.regime.store import RegimeLibrary
from src.regime.similarity import query_library
from src.regime.events import annotate_events
from src.calibration.calibrator import build_calibrated_prior, prior_matched_to_replay_volume
from src.simulator.demand import CalibratedDemandStream
from src.simulator.routing import HaversineClient
from src.rl.env import RideHailEnv
from src.rl.agent import PPOAgent


def _calibrate_from_block(library, block_df):
    """Query library with a demand block, return (prior, regime_feats, block_id)."""
    bid = block_df["block_id"].iloc[0]
    q_series = block_df["request_count"].values.astype(np.float64)
    q_events = annotate_events(q_series)
    matched = query_library(library, q_series, q_events, q_block_id=bid)
    mrecs = [library[b] for b, _ in matched]
    mscores = [s for _, s in matched]
    prior = build_calibrated_prior(mrecs, mscores, q_series)
    feats = mrecs[0].summary_features if mrecs else np.zeros(8)
    return prior, feats, bid


def _estimate_fleet(n_trips: int, horizon_h: float) -> int:
    return max(int(n_trips / horizon_h * 0.15), 50)


def train(max_episodes: int | None = None, fleet_size: int | None = None) -> dict:
    cfg = get_config()
    rl_cfg = cfg["rl"]
    sim_cfg = cfg["simulator"]

    print("Loading regime library ...")
    library = RegimeLibrary()
    try:
        library.load()
    except FileNotFoundError:
        print("  Building regime library from cleaned trips ...")
        trips = load_cleaned()
        library.build_from_trips(trips)
        library.save()
    print(f"  {len(library)} regimes loaded.")

    router = HaversineClient()

    trips = load_cleaned()
    profile = build_demand_profile(trips)
    blocks = split_into_blocks(profile)
    rng = np.random.default_rng(42)

    high_demand_blocks = [b for b in blocks if b["request_count"].sum() > 3000]
    if not high_demand_blocks:
        high_demand_blocks = blocks
    print(f"  {len(high_demand_blocks)} high-demand training blocks selected from {len(blocks)} total")

    horizon = cfg["regime"]["block_hours"] * 3600.0
    horizon_h = horizon / 3600.0
    max_eps = max_episodes or rl_cfg["max_episodes"]
    eps_per_update = rl_cfg["episodes_per_update"]

    # Fixed fleet size: median across training blocks for stable observations.
    # Changing fleet mid-training was causing massive observation distribution shifts.
    if fleet_size:
        fleet = fleet_size
    else:
        fleet_estimates = [
            _estimate_fleet(int(b["request_count"].sum()), horizon_h)
            for b in high_demand_blocks
        ]
        fleet = int(np.median(fleet_estimates))
    print(f"  Fixed fleet size: {fleet} (median across {len(high_demand_blocks)} blocks)")

    sample_block = high_demand_blocks[0]
    prior, regime_feats, _ = _calibrate_from_block(library, sample_block)
    n_req0 = int(sample_block["request_count"].sum())
    pr0 = prior_matched_to_replay_volume(prior, horizon, n_req0)
    stream = CalibratedDemandStream(pr0, horizon, rng=rng)

    env = RideHailEnv(
        demand_stream=stream,
        router=router,
        fleet_size=fleet,
        step_seconds=sim_cfg["step_seconds"],
        horizon_seconds=horizon,
        regime_features=regime_feats,
    )

    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    agent.clip_eps = 0.1

    lr_start = 3e-4
    lr_end = 1e-5
    entropy_start = 0.03
    entropy_end = 0.005

    for pg in agent.optimizer.param_groups:
        pg["lr"] = lr_start

    print(f"Training PPO: {max_eps} episodes, fleet={fleet}, horizon={horizon_h:.0f}h, "
          f"step={sim_cfg['step_seconds']}s ({int(horizon / sim_cfg['step_seconds'])} steps/ep)")
    print(f"  obs_dim={env.observation_space.shape[0]}, act_dim={env.action_space.n}")
    print(f"  LR: {lr_start} -> {lr_end}, entropy: {entropy_start} -> {entropy_end}")

    ep_rewards = []
    ep_completions = []
    ep_waits = []
    t0 = time.time()

    warm_up = min(150, max_eps // 5)
    curriculum_phases = [
        (warm_up, 1),                    # phase 0: single regime
        (warm_up + max_eps // 4, 5),     # phase 1: top-5 similar blocks
        (warm_up + max_eps // 2, 20),    # phase 2: top-20 blocks
        (max_eps, len(high_demand_blocks)),  # phase 3: full diversity
    ]
    regime_switch_interval = 15

    for ep in trange(max_eps, desc="Episodes"):
        # Cosine schedule for LR and entropy
        progress = ep / max(max_eps - 1, 1)
        current_lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * progress))
        current_entropy = entropy_end + 0.5 * (entropy_start - entropy_end) * (1 + math.cos(math.pi * progress))
        for pg in agent.optimizer.param_groups:
            pg["lr"] = current_lr

        # Curriculum: gradually expand regime diversity
        if ep >= warm_up and ep % regime_switch_interval == 0:
            n_pool = 1
            for phase_end, pool_size in curriculum_phases:
                if ep < phase_end:
                    n_pool = pool_size
                    break
            n_pool = min(n_pool, len(high_demand_blocks))
            idx = rng.integers(0, n_pool)
            blk = high_demand_blocks[idx]
            prior, regime_feats, _ = _calibrate_from_block(library, blk)
            n_req = int(blk["request_count"].sum())
            pr = prior_matched_to_replay_volume(prior, horizon, n_req)
            stream = CalibratedDemandStream(pr, horizon, rng=rng)
            env._demand = stream
            env._regime_feats = regime_feats
            # Fleet stays fixed -- no more env._fleet_size changes

        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        ep_reward = 0.0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.buffer.store(obs, action, reward, value, log_prob, done)
            obs = next_obs
            ep_reward += reward

        state = env._state
        m = state.metrics
        cr = m.completed_trips / max(m.total_requests, 1)
        avg_wait = m.total_wait_seconds / max(m.completed_trips, 1)

        ep_rewards.append(ep_reward)
        ep_completions.append(cr)
        ep_waits.append(avg_wait)

        if (ep + 1) % eps_per_update == 0:
            stats = agent.update(entropy_coeff=current_entropy)
            recent_r = np.mean(ep_rewards[-eps_per_update:])
            recent_cr = np.mean(ep_completions[-eps_per_update:])
            recent_w = np.mean(ep_waits[-eps_per_update:])
            print(f"  ep {ep+1}: reward={recent_r:.2f}, "
                  f"wait={recent_w:.0f}s, completion={recent_cr:.1%}, "
                  f"lr={current_lr:.1e}, ent={current_entropy:.3f}, "
                  f"loss={stats['loss']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/max_eps:.1f}s/ep)")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    agent.save(str(out_dir / "ppo_dispatch.pt"))
    np.save(str(out_dir / "train_rewards.npy"), np.array(ep_rewards))
    np.save(str(out_dir / "train_completions.npy"), np.array(ep_completions))
    np.save(str(out_dir / "train_waits.npy"), np.array(ep_waits))
    print(f"Model saved to {out_dir / 'ppo_dispatch.pt'}")

    summary = {
        "episodes": max_eps,
        "elapsed_s": elapsed,
        "final_100_mean_reward": float(np.mean(ep_rewards[-100:])),
        "final_100_mean_wait_s": float(np.mean(ep_waits[-100:])),
        "final_100_mean_completion": float(np.mean(ep_completions[-100:])),
        "first_100_mean_reward": float(np.mean(ep_rewards[:100])),
        "first_100_mean_wait_s": float(np.mean(ep_waits[:100])),
        "first_100_mean_completion": float(np.mean(ep_completions[:100])),
    }
    return summary


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train(max_episodes=n)
