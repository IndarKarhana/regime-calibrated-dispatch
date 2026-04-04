---
name: routing-research
description: >-
  End-to-end workflow for ride-hailing routing/dispatch research in this repo:
  formulate problem, survey literature, implement baselines, run simulation
  experiments, and report vs SOTA-style references. Use for dispatch papers,
  simulator design, benchmarking, ablations, and optimization-heavy features.
---

# Routing research (project skill)

## When to use

- Designing or changing the **simulator**, **dispatch policy**, **demand model**, or **evaluation**.
- Deciding **what to build next** to maximize research credibility.
- Writing experiment sections or interpreting **wait/idle/throughput** tradeoffs.

## Workflow

### 1. Problem card (short, mandatory)

Write (in chat or a scratch file) before major implementation:

- **Decisions:** What gets decided, when, and with what information?
- **Objective:** Primary + secondary (e.g. minimize mean wait subject to fleet capacity).
- **Constraints:** SLA, fairness, max detour, rebalancing budget, horizon.
- **Uncertainty:** Demand process, travel time noise, no-shows/cancels if modeled.
- **Oracles:** Exact travel times vs noisy forecasts vs API routing.

### 2. Literature snapshot

- Find the **closest 3–5 papers** ((dispatch + rideshare / online matching / repositioning / joint learning-and-optimization).
- For each: problem class, assumptions, reported metrics, limitations.
- **Gap:** One sentence on what this repo tests that they do not.

### 3. Baselines (non-negotiable)

Implement or retain at least:

- **Greedy nearest feasible** driver–request matching.
- **Batch** matching at fixed intervals (if real-time batching is in scope).
- Optional: **reposition-to-hotspot** or **idle penalty** only if the paper claims spatial efficiency.

Advanced methods must **lose gracefully** on some scenarios or explain why (no free lunch).

### 4. Implementation order

1. Data ingest + small **smoke** slice.
2. Routing adapter stub (straight-line) then **real** matrix/table if available.
3. Simulator loop with logging hooks.
4. Baselines → metrics dashboard (CSV/JSON) → plots only when numbers stable.
5. “Idea” policy last, with **ablations** (e.g. prediction off vs on).

### 5. Reporting norms

- Tables: mean ± std or CI over **seeds** and/or days.
- **Compute:** wall time, routing calls, hardware class if relevant.
- Claim checklist: every sentence of “improves” must point to **which baseline**, **which metric**, **which setting**.

## Out of scope claims

Do not assert beating **Uber**, **Lyft**, or any closed production system. Assert beating **named baselines** in **this** simulator and data regime.

## Quick reference metrics

- Rider **wait** (pickup), driver **idle**, **pickup distance**, **completion rate**, **throughput**, **P95** waits where applicable.

## Related project files

- `docs/agents.md` — full agent brief; root `AGENTS.md` is the pointer.
- `docs/progress.md` — read before coding; update after substantive work.
- `docs/design-decisions.md` — ADRs for non-obvious choices.
- `.cursor/rules/*.mdc` — always-on behavior and language-specific design hints.
