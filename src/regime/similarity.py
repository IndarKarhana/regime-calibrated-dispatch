"""Query–regime similarity: ensemble of KS, W1, feature, variance, and event-pattern metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

from src.config import get_config
from src.regime.events import SurgeEvent, annotate_events
from src.regime.store import RegimeLibrary, RegimeRecord, _compute_summary_features

COMPONENT_KEYS = ("ks", "w1", "feat", "var", "event", "temporal")
WeightSpec = dict[str, float] | Callable[
    [np.ndarray, list[SurgeEvent], str | None], dict[str, float]
]


def sim_ks(q_series: np.ndarray, s_series: np.ndarray) -> float:
    """1 - two-sample KS statistic."""
    if len(q_series) < 2 or len(s_series) < 2:
        return 0.0
    d, _ = stats.ks_2samp(q_series, s_series)
    return 1.0 - d


def sim_w1(q_series: np.ndarray, s_series: np.ndarray, eps: float = 1e-9) -> float:
    """Normalised Wasserstein-1 similarity."""
    if len(q_series) < 1 or len(s_series) < 1:
        return 0.0
    w = stats.wasserstein_distance(q_series, s_series)
    ptp = max(np.ptp(q_series), np.ptp(s_series), eps)
    return 1.0 / (1.0 + w / ptp)


def sim_feat(q_feats: np.ndarray, s_feats: np.ndarray, eps: float = 1e-9) -> float:
    """Normalised Euclidean distance in feature space."""
    d = np.linalg.norm(q_feats - s_feats)
    avg_norm = (np.linalg.norm(q_feats) + np.linalg.norm(s_feats)) / 2.0 + eps
    return 1.0 / (1.0 + d / avg_norm)


def sim_var(q_series: np.ndarray, s_series: np.ndarray, eps: float = 1e-9) -> float:
    """Variance-ratio similarity."""
    sq = float(np.std(q_series))
    ss = float(np.std(s_series))
    return min(sq, ss) / (max(sq, ss) + eps)


def sim_event(q_events: list[SurgeEvent], s_events: list[SurgeEvent]) -> float:
    """Event-pattern similarity: compares intensity distributions and prefix features."""
    if not q_events and not s_events:
        return 1.0
    if not q_events or not s_events:
        return 0.0

    q_int = np.array([e.intensity_zscore for e in q_events])
    s_int = np.array([e.intensity_zscore for e in s_events])
    int_sim = 1.0 / (1.0 + stats.wasserstein_distance(q_int, s_int))

    q_dur = np.array([e.duration_bins for e in q_events], dtype=float)
    s_dur = np.array([e.duration_bins for e in s_events], dtype=float)
    dur_sim = 1.0 / (1.0 + stats.wasserstein_distance(q_dur, s_dur))

    prefix_sims = []
    for qe in q_events:
        best = 0.0
        qp = qe.prefix_features
        if not qp:
            continue
        qv = _prefix_to_vec(qp)
        for se in s_events:
            sp = se.prefix_features
            if not sp:
                continue
            sv = _prefix_to_vec(sp)
            d = np.linalg.norm(qv - sv)
            norm = (np.linalg.norm(qv) + np.linalg.norm(sv)) / 2.0 + 1e-9
            best = max(best, 1.0 / (1.0 + d / norm))
        prefix_sims.append(best)
    prefix_sim = float(np.mean(prefix_sims)) if prefix_sims else 0.5

    count_ratio = min(len(q_events), len(s_events)) / (max(len(q_events), len(s_events)) + 1e-9)

    return 0.3 * int_sim + 0.25 * dur_sim + 0.25 * prefix_sim + 0.2 * count_ratio


def _prefix_to_vec(pf: dict) -> np.ndarray:
    keys = ["prefix_mean", "prefix_std", "prefix_slope", "prefix_entropy",
            "prefix_autocorr_lag1", "prefix_autocorr_lag2", "prefix_autocorr_lag3"]
    return np.array([pf.get(k, 0.0) for k in keys], dtype=np.float64)


def sim_temporal(q_block_id: str, s_block_id: str) -> float:
    """Temporal proximity: same month, same day-type (weekday/weekend), same hour block."""
    try:
        import pandas as pd
        q_parts = q_block_id.rsplit("_", 1)
        s_parts = s_block_id.rsplit("_", 1)
        q_date = pd.Timestamp(q_parts[0])
        s_date = pd.Timestamp(s_parts[0])
        q_hours = q_parts[1]
        s_hours = s_parts[1]
    except (ValueError, IndexError):
        return 0.5

    score = 0.0

    month_diff = abs(q_date.month - s_date.month)
    month_diff = min(month_diff, 12 - month_diff)
    score += 0.4 * (1.0 - month_diff / 6.0)

    q_weekend = q_date.dayofweek >= 5
    s_weekend = s_date.dayofweek >= 5
    score += 0.35 if q_weekend == s_weekend else 0.0

    score += 0.25 if q_hours == s_hours else 0.0

    return score


def _adaptive_weights(
    base_weights: dict[str, float],
    q_events: list[SurgeEvent],
    s_events: list[SurgeEvent],
) -> dict[str, float]:
    """Redistribute event weight when event comparison is unreliable.

    Rules:
    - Both sides have >= 1 event: keep full event weight (event matching is meaningful)
    - Both sides have 0 events: keep full weight (sim_event returns 1.0, harmless)
    - Exactly one side has 0 events: redistribute event weight to KS/W1/feat
      (comparing events to no-events is noise)
    """
    w = dict(base_weights)
    event_w = w.get("event", 0.0)
    if event_w == 0.0:
        return w

    n_q = len(q_events)
    n_s = len(s_events)

    both_have = n_q > 0 and n_s > 0
    neither_has = n_q == 0 and n_s == 0

    if both_have or neither_has:
        return w

    w["event"] = 0.0
    dist_keys = ["ks", "w1", "feat"]
    dist_total = sum(w.get(k, 0) for k in dist_keys)
    if dist_total > 0:
        for k in dist_keys:
            w[k] = w[k] + event_w * (w[k] / dist_total)
    else:
        w["w1"] = w.get("w1", 0) + event_w

    return w


def weights_to_dict(weights: np.ndarray | list[float] | tuple[float, ...]) -> dict[str, float]:
    """Convert a six-vector to a component-weight dict."""
    arr = np.asarray(weights, dtype=float)
    if arr.shape[0] != len(COMPONENT_KEYS):
        raise ValueError(f"Expected {len(COMPONENT_KEYS)} weights, got {arr.shape[0]}")
    total = float(np.sum(arr))
    if total <= 0:
        arr = np.ones(len(COMPONENT_KEYS), dtype=float) / len(COMPONENT_KEYS)
    else:
        arr = arr / total
    return {k: float(v) for k, v in zip(COMPONENT_KEYS, arr)}


def component_scores(
    q_series: np.ndarray,
    q_events: list[SurgeEvent],
    record: RegimeRecord,
    q_block_id: str | None = None,
) -> np.ndarray:
    """Return raw component similarities in canonical order.

    This helper supports learned weighting and diagnostics while preserving
    ``compute_similarity`` as the public scoring API.
    """
    q_feats = _compute_summary_features(q_series)
    return np.array([
        sim_ks(q_series, record.demand_series),
        sim_w1(q_series, record.demand_series),
        sim_feat(q_feats, record.summary_features),
        sim_var(q_series, record.demand_series),
        sim_event(q_events, record.events),
        sim_temporal(q_block_id, record.block_id) if q_block_id and record.block_id else 0.0,
    ], dtype=np.float64)


def compute_similarity(
    q_series: np.ndarray,
    q_events: list[SurgeEvent],
    record: RegimeRecord,
    weights: WeightSpec | None = None,
    q_block_id: str | None = None,
) -> float:
    """Weighted ensemble similarity between a query batch and a stored regime.

    Uses adaptive weighting: event component is downweighted when events are
    sparse in either query or candidate, redistributing mass to distributional
    metrics (KS, W1, feat). Temporal proximity boosts same-season/day-type matches.
    """
    cfg = get_config()["regime"]["similarity_weights"]
    if callable(weights):
        base_w = weights(q_series, q_events, q_block_id)
    else:
        base_w = weights or cfg

    w = _adaptive_weights(base_w, q_events, record.events)
    comps = component_scores(q_series, q_events, record, q_block_id=q_block_id)
    return float(sum(w.get(k, 0.0) * comps[i] for i, k in enumerate(COMPONENT_KEYS)))


def query_library(
    library: RegimeLibrary,
    q_series: np.ndarray,
    q_events: list[SurgeEvent] | None = None,
    top_k: int | None = None,
    q_block_id: str | None = None,
    weights: WeightSpec | None = None,
) -> list[tuple[str, float]]:
    """Return top-K (block_id, similarity) pairs sorted descending."""
    cfg = get_config()["regime"]
    k = top_k or cfg["top_k"]

    if q_events is None:
        q_events = annotate_events(q_series)

    scores = []
    for bid, rec in library.records.items():
        s = compute_similarity(q_series, q_events, rec, weights=weights, q_block_id=q_block_id)
        scores.append((bid, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
