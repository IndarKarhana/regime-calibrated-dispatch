"""Queueing and spatial mismatch bounds for Path 2 theory experiments.

The functions here are intentionally small and dependency-light. They are
not a replacement for the simulator; they provide an auditable numerical
check for the paper's queueing-style wait-regret theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class WaitBoundResult:
    """Computed wait gap and certificate for one estimated-demand vector."""

    true_wait_s: float
    estimated_policy_wait_s: float
    oracle_policy_wait_s: float
    wait_gap_s: float
    demand_l1_norm: float
    capacity_l1_norm: float
    bound_s: float
    bound_without_capacity_s: float
    max_utilization: float

    @property
    def slack_s(self) -> float:
        return self.bound_s - self.wait_gap_s

    @property
    def tightness(self) -> float:
        if self.wait_gap_s <= 0:
            return np.nan
        return self.bound_s / self.wait_gap_s


def erlang_c_probability_wait(arrival_rate: float, servers: int, service_rate: float) -> float:
    """Return Erlang-C probability that an arrival waits.

    Rates must use the same time unit. Returns 1.0 at or above instability so
    callers can still compute conservative diagnostics without crashing.
    """
    lam = float(arrival_rate)
    mu = float(service_rate)
    c = int(servers)
    if lam <= 0:
        return 0.0
    if c <= 0 or mu <= 0:
        return 1.0

    offered = lam / mu
    rho = offered / c
    if rho >= 1.0:
        return 1.0

    # Stable recurrence for sum_{n=0}^{c-1} a^n/n!.
    term = 1.0
    partial = 1.0
    for n in range(1, c):
        term *= offered / n
        partial += term

    final_term = term * offered / c
    wait_term = final_term / (1.0 - rho)
    return float(wait_term / (partial + wait_term))


def mmc_mean_wait(arrival_rate: float, servers: int, service_rate: float) -> float:
    """Mean queue wait in an M/M/c system in the same time unit as the rates."""
    lam = float(arrival_rate)
    mu = float(service_rate)
    c = int(servers)
    if lam <= 0:
        return 0.0
    if c <= 0 or mu <= 0:
        return np.inf

    spare_capacity = c * mu - lam
    if spare_capacity <= 0:
        return np.inf
    return erlang_c_probability_wait(lam, c, mu) / spare_capacity


def weighted_network_wait(
    arrival_rates: np.ndarray,
    capacities: np.ndarray,
    service_rate: float,
) -> float:
    """Arrival-weighted mean M/M/c queue wait across zones."""
    lam = np.asarray(arrival_rates, dtype=float)
    cap = np.asarray(capacities, dtype=int)
    total = float(lam.sum())
    if total <= 0:
        return 0.0

    waits = np.array([
        mmc_mean_wait(float(lz), int(cz), service_rate)
        for lz, cz in zip(lam, cap)
    ])
    if np.any(~np.isfinite(waits)):
        return np.inf
    return float(np.dot(lam / total, waits))


def proportional_capacity_allocation(
    demand_rates: np.ndarray,
    fleet_size: int,
    service_rate: float,
    rho_target: float = 0.82,
) -> np.ndarray:
    """Integer capacity allocation with stability-aware rounding.

    It first allocates the minimum number of servers needed to keep active
    zones below ``rho_target`` when possible, then distributes remaining
    servers by largest fractional residual. If the requested stability target
    is infeasible, it falls back to proportional rounding.
    """
    lam = np.asarray(demand_rates, dtype=float)
    if fleet_size <= 0:
        return np.zeros_like(lam, dtype=int)
    if lam.sum() <= 0:
        out = np.zeros_like(lam, dtype=int)
        out[: min(fleet_size, len(out))] = 1
        return out

    active = lam > 1e-9
    min_cap = np.zeros_like(lam, dtype=int)
    min_cap[active] = np.maximum(
        1,
        np.ceil(lam[active] / max(service_rate * rho_target, 1e-12)).astype(int),
    )

    if int(min_cap.sum()) <= fleet_size:
        cap = min_cap.copy()
        remaining = int(fleet_size - cap.sum())
        desired = lam / lam.sum() * fleet_size
        residual = desired - cap
        order = np.argsort(-residual)
        for idx in order[:remaining]:
            cap[idx] += 1
        return cap

    desired = lam / lam.sum() * fleet_size
    cap = np.floor(desired).astype(int)
    cap[active & (cap == 0)] = 1
    while int(cap.sum()) > fleet_size:
        removable = np.where(cap > 1)[0]
        if len(removable) == 0:
            break
        idx = removable[np.argmin(lam[removable] / cap[removable])]
        cap[idx] -= 1
    while int(cap.sum()) < fleet_size:
        idx = int(np.argmax(desired - cap))
        cap[idx] += 1
    return cap


def maximum_utilization(
    arrival_rates: np.ndarray,
    capacities: np.ndarray,
    service_rate: float,
) -> float:
    lam = np.asarray(arrival_rates, dtype=float)
    cap = np.asarray(capacities, dtype=float)
    active = (lam > 1e-9) & (cap > 0)
    if not np.any(active):
        return 0.0
    return float(np.max(lam[active] / (cap[active] * service_rate)))


def queue_wait_regret_bound(
    true_rates: np.ndarray,
    estimated_rates: np.ndarray,
    fleet_size: int,
    service_rate: float,
    rho_max: float = 0.90,
) -> WaitBoundResult:
    """Compute the Phase 3 wait-regret certificate.

    The certificate has two terms: demand misspecification and allocator
    instability. The second term is what makes the bound survive integer
    reallocations and is useful diagnostically when high retrieval quality
    does not translate into lower downstream wait.
    """
    true_rates = np.asarray(true_rates, dtype=float)
    estimated_rates = np.asarray(estimated_rates, dtype=float)
    if true_rates.shape != estimated_rates.shape:
        raise ValueError("true_rates and estimated_rates must have same shape")

    cap_star = proportional_capacity_allocation(
        true_rates, fleet_size, service_rate, rho_target=min(rho_max, 0.82)
    )
    cap_hat = proportional_capacity_allocation(
        estimated_rates, fleet_size, service_rate, rho_target=min(rho_max, 0.82)
    )

    oracle_wait = weighted_network_wait(true_rates, cap_star, service_rate)
    policy_wait = weighted_network_wait(true_rates, cap_hat, service_rate)
    if np.isfinite(policy_wait) and np.isfinite(oracle_wait):
        wait_gap = max(0.0, policy_wait - oracle_wait)
    elif np.isfinite(oracle_wait):
        wait_gap = np.inf
    else:
        wait_gap = 0.0
    total = max(float(true_rates.sum()), 1e-12)

    demand_l1 = float(np.abs(estimated_rates - true_rates).sum())
    capacity_l1 = float(np.abs(cap_hat - cap_star).sum())
    utilization = max(
        maximum_utilization(true_rates, cap_star, service_rate),
        maximum_utilization(true_rates, cap_hat, service_rate),
    )
    effective_rho = min(max(utilization, 0.0), rho_max)
    lipschitz = 1.0 / (service_rate * max(1.0 - effective_rho, 1e-6) ** 2)
    bound = lipschitz * (demand_l1 + service_rate * capacity_l1) / total
    direct_bound = lipschitz * demand_l1 / total

    return WaitBoundResult(
        true_wait_s=oracle_wait * 3600.0,
        estimated_policy_wait_s=policy_wait * 3600.0,
        oracle_policy_wait_s=oracle_wait * 3600.0,
        wait_gap_s=wait_gap * 3600.0,
        demand_l1_norm=demand_l1 / total,
        capacity_l1_norm=capacity_l1 / max(float(fleet_size), 1.0),
        bound_s=bound * 3600.0,
        bound_without_capacity_s=direct_bound * 3600.0,
        max_utilization=utilization,
    )


def shortage_index(true_rates: np.ndarray, estimated_rates: np.ndarray) -> float:
    """Directional under-forecast mass as a fraction of total demand."""
    true_rates = np.asarray(true_rates, dtype=float)
    estimated_rates = np.asarray(estimated_rates, dtype=float)
    total = max(float(true_rates.sum()), 1e-12)
    return float(np.maximum(true_rates - estimated_rates, 0.0).sum() / total)


def earth_movers_distance(cost_matrix: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Discrete optimal-transport distance between probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    n, m = len(p), len(q)
    c = np.asarray(cost_matrix, dtype=float).reshape(-1)

    rows = []
    rhs = []
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1.0
        rows.append(row)
        rhs.append(p[i])
    for j in range(m):
        row = np.zeros(n * m)
        row[j::m] = 1.0
        rows.append(row)
        rhs.append(q[j])

    result = linprog(
        c,
        A_eq=np.vstack(rows),
        b_eq=np.asarray(rhs),
        bounds=[(0.0, None)] * (n * m),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"OT solve failed: {result.message}")
    return float(result.fun)
