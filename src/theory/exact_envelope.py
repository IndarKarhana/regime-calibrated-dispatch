"""Exact directional residual shortage envelope utilities."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def shortage_loss(allocation: np.ndarray, demand_share: np.ndarray) -> float:
    """Return normalized one-period shortage loss L(a, p)."""
    allocation = np.asarray(allocation, dtype=float)
    demand_share = np.asarray(demand_share, dtype=float)
    total_idle = float(allocation.sum())
    return float(np.maximum(total_idle * demand_share - allocation, 0.0).sum())


def scalar_residual_certificate(
    allocation: np.ndarray,
    mixture_share: np.ndarray,
    rho: float,
) -> float:
    """Return the tractable DCR scalar envelope upper certificate."""
    allocation = np.asarray(allocation, dtype=float)
    return shortage_loss(allocation, mixture_share) + float(allocation.sum()) * float(rho)


def exact_directional_residual_envelope(
    allocation: np.ndarray,
    mixture_share: np.ndarray,
    rho: float,
    *,
    max_exact_zones: int = 28,
) -> float:
    """Evaluate the exact positive-residual shortage envelope.

    The exact envelope requires subset separation. We intentionally cap exact
    enumeration to keep accidental 183-zone closed-loop use from exploding.
    Use this for diagnostics on coarsened zone systems or small examples.
    """
    allocation = np.asarray(allocation, dtype=float)
    mixture_share = np.asarray(mixture_share, dtype=float)
    if allocation.shape != mixture_share.shape:
        raise ValueError("allocation and mixture_share must have the same shape")
    n_zones = allocation.size
    if n_zones > max_exact_zones:
        raise ValueError(
            f"exact envelope enumeration requested for {n_zones} zones; "
            f"max_exact_zones={max_exact_zones}"
        )
    total_idle = float(allocation.sum())
    best = 0.0
    zones = range(n_zones)
    for size in range(1, n_zones + 1):
        for subset in combinations(zones, size):
            idx = np.fromiter(subset, dtype=int)
            share_mass = float(mixture_share[idx].sum())
            alloc_mass = float(allocation[idx].sum())
            value = total_idle * min(1.0, share_mass + float(rho)) - alloc_mass
            if value > best:
                best = value
    return float(best)


def milp_directional_residual_envelope(
    allocation: np.ndarray,
    mixture_share: np.ndarray,
    rho: float,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Evaluate the exact envelope by solving the two subset MILPs.

    This is intended for diagnostics and possible future cut generation. It is
    much faster than brute-force enumeration on the 28-zone coarsened grid.
    """
    allocation = np.asarray(allocation, dtype=float)
    mixture_share = np.asarray(mixture_share, dtype=float)
    if allocation.shape != mixture_share.shape:
        raise ValueError("allocation and mixture_share must have the same shape")
    n_zones = allocation.size
    if n_zones == 0:
        return 0.0
    total_idle = float(allocation.sum())
    rho = float(rho)
    capacity = 1.0 - rho
    integrality = np.ones(n_zones, dtype=int)
    bounds = Bounds(np.zeros(n_zones), np.ones(n_zones))
    nonempty = LinearConstraint(np.ones((1, n_zones)), [1.0], [np.inf])

    values = [0.0]
    if capacity >= 0.0:
        d = total_idle * mixture_share - allocation
        constraints = [
            LinearConstraint(mixture_share[None, :], [-np.inf], [capacity]),
            nonempty,
        ]
        result = milp(
            c=-d,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": 10.0},
        )
        if result.success:
            values.append(total_idle * rho - float(result.fun))

    if capacity < 1.0:
        constraints = [
            LinearConstraint(mixture_share[None, :], [capacity + epsilon], [np.inf]),
            nonempty,
        ]
        result = milp(
            c=allocation,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": 10.0},
        )
        if result.success:
            values.append(total_idle - float(result.fun))
    return float(max(values))


def milp_directional_residual_envelope_witness(
    allocation: np.ndarray,
    mixture_share: np.ndarray,
    rho: float,
    *,
    epsilon: float = 1e-9,
) -> tuple[float, np.ndarray]:
    """Return the exact envelope value and a maximizing subset witness.

    The subset is encoded as a boolean mask over zones. This is the separation
    oracle needed by exact-envelope cut generation: if the returned value
    exceeds a target budget Gamma, the corresponding subset yields the violated
    linear cut ``a(A) >= I min(1, phat(A)+rho) - Gamma``.
    """
    allocation = np.asarray(allocation, dtype=float)
    mixture_share = np.asarray(mixture_share, dtype=float)
    if allocation.shape != mixture_share.shape:
        raise ValueError("allocation and mixture_share must have the same shape")
    n_zones = allocation.size
    if n_zones == 0:
        return 0.0, np.zeros(0, dtype=bool)
    total_idle = float(allocation.sum())
    rho = float(rho)
    capacity = 1.0 - rho
    integrality = np.ones(n_zones, dtype=int)
    bounds = Bounds(np.zeros(n_zones), np.ones(n_zones))
    nonempty = LinearConstraint(np.ones((1, n_zones)), [1.0], [np.inf])

    best_value = 0.0
    best_mask = np.zeros(n_zones, dtype=bool)

    if capacity >= 0.0:
        d = total_idle * mixture_share - allocation
        constraints = [
            LinearConstraint(mixture_share[None, :], [-np.inf], [capacity]),
            nonempty,
        ]
        result = milp(
            c=-d,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": 10.0},
        )
        if result.success:
            value = total_idle * rho - float(result.fun)
            if value > best_value:
                best_value = value
                best_mask = np.asarray(result.x > 0.5, dtype=bool)

    if capacity < 1.0:
        constraints = [
            LinearConstraint(mixture_share[None, :], [capacity + epsilon], [np.inf]),
            nonempty,
        ]
        result = milp(
            c=allocation,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": 10.0},
        )
        if result.success:
            value = total_idle - float(result.fun)
            if value > best_value:
                best_value = value
                best_mask = np.asarray(result.x > 0.5, dtype=bool)

    return float(best_value), best_mask


def greedy_directional_residual_envelope_lower_bound(
    allocation: np.ndarray,
    mixture_share: np.ndarray,
    rho: float,
) -> float:
    """Fast lower bound from greedy subset candidates.

    This is not a certificate; it is a diagnostic witness for how tight the
    scalar upper certificate may be.
    """
    allocation = np.asarray(allocation, dtype=float)
    mixture_share = np.asarray(mixture_share, dtype=float)
    total_idle = float(allocation.sum())
    if total_idle <= 0 or allocation.size == 0:
        return 0.0
    deficits = total_idle * mixture_share - allocation
    candidates: list[np.ndarray] = []
    candidates.append(np.where(deficits > 0)[0])
    candidates.append(np.argsort(-deficits))
    ratio = deficits / np.maximum(mixture_share, 1e-12)
    candidates.append(np.argsort(-ratio))
    candidates.append(np.argsort(allocation))

    best = 0.0
    for order in candidates:
        subset: list[int] = []
        for z in order:
            subset.append(int(z))
            idx = np.asarray(subset, dtype=int)
            share_mass = float(mixture_share[idx].sum())
            alloc_mass = float(allocation[idx].sum())
            value = total_idle * min(1.0, share_mass + float(rho)) - alloc_mass
            best = max(best, value)
    return float(best)
