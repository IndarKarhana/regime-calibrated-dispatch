# Agent instructions — Routing_research

This repo targets **research-grade** ride-hailing **routing, dispatch, and demand-aware optimization**—inspired by large platforms, but **not** claiming parity with any proprietary system. The goal is to **beat strong baselines** under clear assumptions and **relate claims to published methods**.

## Before writing code

1. Read **`docs/progress.md`** end-to-end for current focus, facts, and what already failed/succeeded.
2. Skim **`docs/design-decisions.md`** for ADRs that bound the work.
3. After substantive changes, **append `docs/progress.md`** (and add an ADR if the decision is non-obvious).

## Operating mode

1. **Execute.** Use the terminal, run experiments, fix errors. Do not end turns with “you could run X” unless execution is blocked (missing secrets, destructive action).
2. **Ship artifacts.** Code, configs, reproducible scripts, and tables/figures paths—not prose-only plans.
3. **Default to clarity.** One concern per change; avoid drive-by refactors unrelated to the task.
4. **Ask once when blocked.** Missing data paths, API keys, or a hard fork in experimental design.

## Research stance (treat as PhD-level)

- **Formalize** the decision problem: decision variables, objective(s), constraints, information structure (what is known when), horizon, and uncertainty.
- **Position against literature.** Name problem classes (e.g. online bipartite matching, batching, vehicle routing, MDP/RL, stochastic/robust optimization, predict-then-optimize vs end-to-end). Prefer **verifiable** references (DOI/arXiv) when claiming “state of the art.”
- **Baselines first.** Greedy / nearest-driver / fixed-batch matching / simple forecasting must run before “fancy” methods.
- **Evaluation contract.** Same simulator, same data split, same seeds; report wait time, idle time, pickup distance, completion rate, throughput, and runtime—plus sensitivity where claims depend on scale or hyperparameters.
- **Honesty.** Uber’s production stack is opaque; contributions are **method + evidence vs baselines**, not “we beat Uber.”

## Stack hints (non-binding)

Public trip data, OSRM/GraphHopper-class routing, discrete-event or discrete-time simulation, optional hex grids (H3) for spatial aggregation—see rules in `.cursor/rules/` for detail.

## Where project docs live

| Doc | Role |
|-----|------|
| `docs/progress.md` | What we tried, results, environment facts |
| `docs/design-decisions.md` | ADRs: why / what / when |
| `docs/agents.md` | This file (full agent brief) |

## Editor-specific agent config

| Editor | Config location | Notes |
|--------|----------------|-------|
| **GitHub Copilot** (VS Code) | `.github/copilot-instructions.md` | Unified workspace instructions — all rules + skill in one file |
| **Cursor** | `.cursor/rules/*.mdc`, `.cursor/skills/routing-research/SKILL.md` | Split across rule files and skill file |

Both contain the same content — keep them in sync when updating agent behavior.
