# Copilot Instructions — Routing_research

This repo targets **research-grade** ride-hailing **routing, dispatch, and demand-aware optimization**. The goal is to **beat strong baselines** under clear assumptions and **relate claims to published methods**.

---

## Before writing code

1. Read **`docs/progress.md`** end-to-end for current focus, facts, and what already failed/succeeded.
2. Skim **`docs/design-decisions.md`** for ADRs that bound the work.
3. After substantive changes, **append `docs/progress.md`** (and add an ADR if the decision is non-obvious).

---

## Core execution

- **Real environment:** Assume shell and network work. Run commands to verify installs, tests, and experiments.
- **Persistence:** Retry with a different approach after failures; diagnose with logs and minimal repros.
- **Output:** Prefer diffs, runnable scripts, and measured numbers over long prose.
- **Scope:** Change only what the task needs; preserve unrelated code and comments.
- **Repo docs:** Do not add markdown files the user did not request — except mandated updates under `docs/` (`progress.md`, `design-decisions.md`, `agents.md`).
- **Freshness:** When suggesting a paper or method, check freshness (e.g. search arXiv/conference proceedings) if the claim depends on "latest" work — the field moves quickly.

---

## Research stance (PhD-level)

### Mental model

Operate as someone trained in **optimization**, **probability/ML**, and **software design**: precise definitions before algorithms. Industrial ride platforms optimize **system-level** KPIs (wait, idle, service rate, ETA reliability) — not myopic shortest path or single-trip greed unless that is the defined baseline.

### Literature-first habit

Before coding a "new" idea:

1. Map the problem to a **known formulation** (online/stochastic matching, VRP variants, MPC, MDP/RL, robust optimization, joint prediction + optimization).
2. Identify **2–3 strong recent references** (peer-reviewed or widely cited preprints) and how this repo will **differ** (assumptions, data, scale, objective, uncertainty).
3. Propose **ablations** that isolate whether gains come from forecasting, matching policy, rebalancing, or routing fidelity.

### What "better than current practice" means

- **Not** copying Uber (unknown). **Yes:** beating **documented baselines** under **stated** information and fairness constraints.
- Prefer **reproducible** comparisons: fixed seeds, frozen splits, reported compute.

### Experimental hygiene

- Objective = scalar or explicit Pareto tradeoffs (say which).
- State **information lag** (e.g. routing query latency, forecast refresh).
- If using ML: calibration and leakage checks (no future labels in features unless explicitly allowed).

---

## Software design

- **Separation:** Simulation core (state transition), policies (dispatch/rebalance), routing adapter (OSRM/GH client), data ingest — thin boundaries, explicit interfaces.
- **Determinism:** Seeded RNGs; log config + git hash for runs meant for comparison.
- **Config:** Hyperparameters and scenario knobs in one place (YAML/TOML), not scattered literals.
- **Performance:** Profile before micro-optimizing; batch routing/matrix requests when the engine allows.
- **Types & tests:** Prefer typed Python for numerical code; add tests for state transitions and one end-to-end tiny scenario.
- **No speculative enterprise patterns** unless scale demands them — favor readable research code.

---

## Source of truth before coding

Before **substantive** edits (new modules, behavior changes, experiments, refactors beyond typos):

1. **Read** `docs/progress.md` in full. Treat "Current focus," "Environment & facts," and the session log as ground truth for what exists, what was tried, and what failed.
2. **Check** `docs/design-decisions.md` for ADRs that apply to the files or subsystems you touch. Do not contradict an **Accepted** decision without flagging it and proposing an update to that file.
3. **Do not invent:** data paths, CLI commands, implemented features, or experimental results not recorded there — if missing, state "unknown" and update `docs/progress.md` after discovery.

After meaningful work:

- **Append** a row to the session table in `docs/progress.md` (newest first).
- Add an ADR to `docs/design-decisions.md` when the **why** matters for future readers.

---

## Smoke test gate

After **every** substantive change (new module, behavior change, hyperparameter tweak, refactor that touches simulation or RL):

1. **Run Level 1 benchmark** (`python scripts/benchmark_checkpoints.py 1`). Takes ~30 s.
   - All 7 checks must **PASS**. If any fail, fix before moving on.
2. **Log the result** in your response: state which checks passed/failed and any KPI drift.
3. If the change is **architecture-level** (env redesign, new policy, new similarity metric, calibrator change):
   - Also run **Level 2** (`python scripts/benchmark_checkpoints.py 2`, ~5 min).
   - Compare calibrated-vs-flat and calibrated-vs-replay wait/completion against the reference numbers in `docs/progress.md` → "Ablation results" table.
   - Flag any regression > 5% on wait or > 2pp on completion rate.

**Do not skip the smoke test.** A 30-second sanity check is cheaper than debugging a silent regression three sessions later.

Reference baselines (ablation 2026-04-02, Jan 2024 08-12h block, fleet=204):

| Metric | batch_cal_full | greedy_replay | random_replay |
|--------|---------------|--------------|--------------|
| mean_wait_s | 150 | 402 | 693 |
| completion | 84% | 74% | 62% |
| throughput/h | 1129 | 1007 | 840 |

---

## Routing research skill / workflow

### When to use

- Designing or changing the **simulator**, **dispatch policy**, **demand model**, or **evaluation**.
- Deciding **what to build next** to maximize research credibility.
- Writing experiment sections or interpreting **wait/idle/throughput** tradeoffs.

### Problem card (short, mandatory before major implementation)

- **Decisions:** What gets decided, when, and with what information?
- **Objective:** Primary + secondary (e.g. minimize mean wait subject to fleet capacity).
- **Constraints:** SLA, fairness, max detour, rebalancing budget, horizon.
- **Uncertainty:** Demand process, travel time noise, no-shows/cancels if modeled.
- **Oracles:** Exact travel times vs noisy forecasts vs API routing.

### Literature snapshot

- Find the **closest 3–5 papers** (dispatch + rideshare / online matching / repositioning / joint learning-and-optimization).
- For each: problem class, assumptions, reported metrics, limitations.
- **Gap:** One sentence on what this repo tests that they do not.

### Baselines (non-negotiable)

Implement or retain at least:

- **Greedy nearest feasible** driver–request matching.
- **Batch** matching at fixed intervals.
- Optional: **reposition-to-hotspot** or **idle penalty** only if the paper claims spatial efficiency.

Advanced methods must **lose gracefully** on some scenarios or explain why (no free lunch).

### Implementation order

1. Data ingest + small smoke slice.
2. Routing adapter stub (straight-line) then real matrix/table if available.
3. Simulator loop with logging hooks.
4. Baselines → metrics dashboard (CSV/JSON) → plots only when numbers stable.
5. "Idea" policy last, with ablations (e.g. prediction off vs on).

### Reporting norms

- Tables: mean ± std or CI over seeds and/or days.
- **Compute:** wall time, routing calls, hardware class if relevant.
- Claim checklist: every sentence of "improves" must point to **which baseline**, **which metric**, **which setting**.

### Out of scope claims

Do not assert beating Uber, Lyft, or any closed production system. Assert beating **named baselines** in this simulator and data regime.

### Quick reference metrics

Rider **wait** (pickup), driver **idle**, **pickup distance**, **completion rate**, **throughput**, **P95** waits where applicable.

---

## Project documentation map

| Doc | Role |
|-----|------|
| `docs/agents.md` | Full agent brief |
| `docs/progress.md` | What we tried, results, environment facts |
| `docs/design-decisions.md` | ADRs: why / what / when |
| `AGENTS.md` | Root pointer to all docs |

## Key commands

| Command | Purpose |
|---------|---------|
| `make install` | Install dependencies |
| `make data` | Download/process data |
| `make osrm-prepare && make osrm-up` | Start OSRM routing engine |
| `python scripts/benchmark_checkpoints.py 1` | L1 sanity (30s) |
| `python scripts/benchmark_checkpoints.py 2` | L2 ablation preview (5min) |
| `python scripts/benchmark_checkpoints.py 3` | L3 full matrix (30min) |
| `python scripts/cross_scenario_ablation.py` | Cross-scenario eval (~4min) |
| `python scripts/comprehensive_eval.py` | Full end-to-end pipeline eval |
