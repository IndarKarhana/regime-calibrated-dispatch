# Regime-Calibrated Dispatch Optimization for Ride-Hailing

**31.1% average wait time reduction** over replay baselines — no training required.

A research-grade system that matches current demand conditions to a library of historical "regimes" via distributional similarity, calibrates a generative demand prior, and uses LP-based anticipatory repositioning to proactively balance fleet supply. Evaluated on NYC TLC 2024 data across 8 diverse scenarios (winter/summer, weekday/weekend/holiday/night).

---

## Headline Results

| Scenario | Replay Wait | Calibrated Only | Cal + LP | vs Replay |
|----------|-------------|-----------------|----------|-----------|
| jan_nye_am | 189.6s | 164.8s | 147.8s | **-22.1%** |
| jan_nye_pm | 152.1s | 108.9s | 93.0s | **-38.9%** |
| jan_weekday_am | 106.4s | 90.8s | 73.5s | **-30.9%** |
| jan_weekday_pm | 107.2s | 99.3s | 81.8s | **-23.7%** |
| jan_weekend_mid | 96.2s | 84.7s | 66.6s | **-30.7%** |
| jun_late_night | 127.8s | 85.0s | 69.5s | **-45.6%** |
| jun_weekday_am | 107.2s | 94.0s | 78.0s | **-27.3%** |
| jun_weekday_pm | 99.8s | 88.7s | 69.8s | **-30.0%** |
| **AVERAGE** | **123.3s** | **101.8s** | **85.0s** | **-31.1%** |

The improvement decomposes into **~16.9% from calibration** plus a further **~15.5% from repositioning** in the current paper analysis. Extended 3-seed baselines show LP and the simpler heuristic are statistically close in aggregate; Path 2 will rework this into a stronger tier-1 transportation submission.

### Comparison with Published Research

| Method | Wait Reduction | Training? |
|--------|---------------|-----------|
| Proactive Rebalancing (Wen et al. 2017) | 5.0% | No |
| RL from Optimization Proxy (IJCAI 2023) | ~10-15% | Yes (RL) |
| AAVR Framework (Dec 2024) | 22.7% | Yes (ML) |
| Sim-informed RL (Namdarpour & Chow 2025) | 27.3% | Yes (RL, ride-pooling) |
| **Ours: Calibrated Prior + LP** | **31.1%** | **No** |

Key advantages: (1) no training for the current default method, (2) interpretable regime-similarity calibration, (3) separate calibration and repositioning ablations, and (4) robustness across seasons/holidays/day types without retraining. The next research phase replaces cited-number comparisons with re-implemented external baselines.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  TLC / Chicago TNP parquets → cleaned → demand profiles → blocks │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                    Regime Library                                 │
│  373 blocks (Jan+Jun 2024) with demand series, ECDFs, events,    │
│  summary features, and 2000 raw OD lat/lon pairs per block       │
└──────────────────────┬───────────────────────────────────────────┘
                       │ similarity ensemble (KS + W1 + feat +
                       │ var + event + temporal, adaptive gate)
┌──────────────────────▼───────────────────────────────────────────┐
│               Calibrated Demand Prior                            │
│  Weighted mixture of top-K matched regimes. Quantile mapping.    │
│  Volume-matched to replay trip count for fair comparison.         │
│  Spatial pool from historical OD pairs.                          │
└───────────┬────────────────────────────┬─────────────────────────┘
            │                            │
┌───────────▼──────────┐   ┌─────────────▼─────────────────────────┐
│  Demand Stream       │   │  LP Anticipatory Repositioning        │
│  (Poisson sampling   │   │  Min-cost transportation LP:          │
│   from rate profile) │   │  forecast zone demand from prior →    │
│                      │   │  identify supply gaps → redistribute  │
│                      │   │  idle drivers (move_frac ≤ 0.50)      │
└───────────┬──────────┘   └─────────────┬─────────────────────────┘
            │                            │
┌───────────▼────────────────────────────▼─────────────────────────┐
│                     Simulation Engine                             │
│  Discrete-time (30s steps). Batch dispatch (60s window).         │
│  Haversine or OSRM routing. Fleet of N drivers.                  │
│  Reposition every K steps via LP/heuristic protocol.             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                      Evaluation                                   │
│  KPIs: wait, p95_wait, completion, throughput, pickup distance   │
│  3-level benchmark checkpoints. Cross-scenario ablation.         │
│  LP sensitivity sweep. Academic baseline comparison.             │
└──────────────────────────────────────────────────────────────────┘
```

## Module Layout

```
src/
├── config.py                    # YAML config loader (config/default.yaml)
├── regime/                      # Regime library & similarity
│   ├── ingest.py                # TLC/Chicago data ingest, demand profiles, blocks
│   ├── store.py                 # RegimeLibrary, RegimeRecord, ECDF, features
│   ├── events.py                # Surge/dip event detection (rolling MAD)
│   └── similarity.py            # Ensemble similarity (KS, W1, feat, var, event, temporal)
├── calibration/
│   └── calibrator.py            # CalibratedPrior, volume matching, spatial OD pool
├── simulator/
│   ├── entities.py              # RideRequest, Driver, SimulationState
│   ├── demand.py                # ReplayDemandStream, CalibratedDemandStream
│   ├── routing.py               # HaversineClient, OSRMClient
│   └── engine.py                # SimulationEngine (step loop, dispatch, reposition)
├── policies/
│   ├── greedy.py                # Greedy nearest-feasible dispatch
│   ├── batch.py                 # Batch matching (Hungarian/min-cost)
│   ├── random_baseline.py       # Random assignment baseline
│   └── anticipatory.py          # LP repositioning + demand-following heuristic
├── rl/                          # PPO agent (negative result — documented)
│   ├── env.py                   # Gym environment (H3 hex obs, reposition action)
│   ├── agent.py                 # PPO with cosine LR/entropy schedules
│   └── train.py                 # Training loop with curriculum
└── evaluation/
    ├── metrics.py               # KPI computation (wait, completion, throughput, etc.)
    ├── ablation_runner.py       # Full ablation matrix (8 configs × 5 seeds)
    └── plots.py                 # Visualization utilities
```

---

## Quick Start

### Prerequisites
- Python ≥ 3.11 (Anaconda recommended)
- Docker (for OSRM routing — optional, Haversine fallback works)

### Installation

```bash
make install                     # pip install -e ".[dev]"
make data                        # Download NYC TLC Jan+Jun 2024 (~5M trips)
python scripts/download_chicago.py  # Optional: Chicago TNP data
```

### OSRM Routing (Optional)

```bash
make osrm-prepare                # Download OSM + extract/partition/customize (~20 min)
make osrm-up                     # Start OSRM Docker container (port 5050)
make osrm-check                  # Verify OSRM is responding
```

### Run Evaluations

```bash
# Sanity check (30s) — run after any change
python scripts/benchmark_checkpoints.py 1

# Ablation preview (5 min)
python scripts/benchmark_checkpoints.py 2

# Full ablation matrix (30 min)
python scripts/benchmark_checkpoints.py 3

# Cross-scenario ablation (8 scenarios, ~10 min)
python scripts/cross_scenario_ablation.py

# End-to-end pipeline with tuned LP
python scripts/comprehensive_eval.py --router osrm --lp-lookahead 5.0 --lp-move-frac 0.50

# Cross-city Chicago + OSRM comparison
python scripts/cross_city_and_tuned_eval.py --router osrm
```

---

## Key Contributions

### 1. Regime-Calibrated Demand Prior (~16.9% wait avg)

A **regime library** indexes historical demand into sub-day blocks (4h each, 373 blocks from Jan+Jun 2024). Each block stores the demand time series, ECDF, summary features, detected events (surges/dips), and 2000 raw OD coordinate pairs.

At query time, a **6-component similarity ensemble** (KS distance, Wasserstein-1, feature similarity, variance ratio, event pattern, temporal proximity) finds the top-K most similar historical regimes. An **adaptive event gate** redistributes event weight when one side has no events, preventing noise injection.

The matched regimes form a **calibrated prior**: a weighted mixture of demand rate profiles with quantile mapping and a spatial OD pool. Volume is matched to replay trip counts for fair comparison.

### 2. LP Anticipatory Repositioning (~15.5% wait avg on top of calibration)

A rolling-horizon **min-cost transportation LP** uses the calibrated prior's per-zone demand forecast to proactively redistribute idle drivers:

1. Forecast per-zone demand for next 5 minutes from the prior's rate profile
2. Identify supply-demand gaps across H3 hex zones
3. Solve LP: maximize demand served − α·travel cost, subject to supply/demand/budget constraints
4. Move up to 50% of idle drivers toward underserved zones

Also includes a simpler **demand-following heuristic** baseline (greedy move toward highest-demand zone). Current extended baselines show the heuristic and LP are close in aggregate; LP is most useful when enough idle fleet exists for strategic reallocation.

### 3. RL Repositioning (Negative Result)

2000-episode PPO training with stability fixes (fixed fleet, cosine schedules, 4-phase curriculum) **did not converge**. Five structural root causes documented: (1) 184-action space too large, (2) weak repositioning lever, (3) regime diversity too high for single MLP, (4) delayed credit assignment, (5) 744-dim observation underfitted. The LP deterministically achieves what RL could not learn.

---

## LP Parameter Sensitivity

Higher move fractions dominate; short lookaheads are marginally best. Optimal: **move_fraction=0.50, lookahead=5min → 19.4% wait reduction** from LP alone.

| Lookahead \ Move Fraction | 0.10 | 0.20 | 0.35 | 0.50 |
|---------------------------|------|------|------|------|
| 5 min | 8.1% | 11.5% | 16.0% | 17.8% |
| 10 min | 6.2% | 8.5% | 13.1% | 15.9% |
| 15 min | 5.4% | 8.4% | 13.2% | 15.6% |
| 20 min | 5.8% | 8.1% | 12.8% | 15.0% |
| 30 min | 5.6% | 8.6% | 12.7% | 15.1% |

---

## Cross-Scenario Robustness

Calibrated + LP beats replay in **8/8 scenarios** for batch dispatch:

| Scenario | Improvement | Notes |
|----------|-------------|-------|
| jun_late_night | **-45.6%** | Best: demand patterns very distinctive |
| jan_nye_pm | **-38.9%** | Holiday PM well-matched |
| jan_weekday_am | **-30.9%** | Standard commute |
| jan_weekend_mid | **-30.7%** | Fixed by temporal proximity metric |
| jun_weekday_pm | **-30.0%** | Summer PM |
| jun_weekday_am | **-27.3%** | Summer weekday |
| jan_weekday_pm | **-23.7%** | Winter PM |
| jan_nye_am | **-22.1%** | Holiday morning |

Key ablation insights:
- **Temporal proximity** fixed cross-seasonal contamination (jan_weekend_mid: was -5%, now -26%)
- **Adaptive event gate** fixed jun_late_night where event noise was hurting
- **Spatial OD enrichment** (raw lat/lon pairs) was the breakthrough for calibration fidelity

---

## Data

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| NYC TLC Yellow Taxi | [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | ~5M trips (Jan+Jun 2024) | Primary eval |
| Chicago TNP | [Chicago Data Portal](https://data.cityofchicago.org/) | ~3.4M trips (Jan 2024) | Cross-city generalization |
| NY OSM | [Geofabrik](https://download.geofabrik.de/) | ~1 GB PBF | OSRM road routing |

---

## Configuration

All hyperparameters live in [`config/default.yaml`](config/default.yaml):

- **Regime**: bin interval, block hours, event detection, similarity weights, top-K
- **Simulator**: step size, fleet, max wait, fallback speed
- **Policies**: batch window
- **RL**: algorithm, network, LR, GAE, clip, episodes
- **Evaluation**: seeds, output directory

---

## Reproducibility

The public repository keeps code, scripts, and Makefile targets under version
control. Large data, generated results, local agent notes, and manuscript
source packages are intentionally omitted from Git and regenerated locally.
Use the Makefile targets below for the current experiment pipeline.

---

## Benchmark System

Three-level checkpoint system ensures no silent regressions:

| Level | Time | What it checks |
|-------|------|----------------|
| L1 | ~30s | Library size, greedy/random/batch ranges, similarity scores, calibrated stream |
| L2 | ~5 min | Calibrated vs flat vs replay on wait/completion |
| L3 | ~30 min | Full 8-config × 5-seed ablation matrix |

**Rule:** Run L1 after every change. L2 before major commits. L3 for paper-ready results.

---

## Path 2: Tier-1 Transportation Rework

The active next phase targets Transportation Science or Transportation Research
Part C, with IEEE T-ITS as fallback.

Main planned upgrades:

- learned similarity weights with leakage-safe train/holdout splits,
- a queueing-based wait-bound theorem replacing the shallow LP-feasibility theorem,
- re-implemented external baselines in this simulator,
- preregistered 10-seed main experiments,
- optional AMoD/charging experiment for TR-C framing,
- an arXiv v2 only after new results or claims change.

---

## Citation

If you use this work, please cite:

```
@misc{regime_calibrated_dispatch_2026,
  title={Regime-Calibrated Demand Prior with LP Repositioning for Ride-Hailing Dispatch},
  year={2026},
  note={Research code: https://github.com/...}
}
```

---

## License

Research code. See project documentation for usage terms.
