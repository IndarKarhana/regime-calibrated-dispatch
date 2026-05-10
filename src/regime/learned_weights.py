"""Learned similarity weights for Path 2 regime retrieval.

The gate maps query-block features to a six-component similarity weight vector:
KS, W1, feature, variance, event, temporal. Training uses a leakage-safe
block-group split and a differentiable top-k weighted demand-error objective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn

from src.config import get_config
from src.regime.events import SurgeEvent
from src.regime.similarity import (
    COMPONENT_KEYS,
    component_scores,
    weights_to_dict,
)
from src.regime.store import RegimeLibrary, RegimeRecord

PATH2_SCENARIOS: dict[str, tuple[str, str, str]] = {
    "jan_weekday_am": ("2024-01", "08-12", "weekday"),
    "jan_weekday_pm": ("2024-01", "16-20", "weekday"),
    "jan_nye_am": ("2024-01", "08-12", "holiday"),
    "jan_nye_pm": ("2024-01", "16-20", "holiday"),
    "jan_weekend_mid": ("2024-01", "12-16", "weekend"),
    "jun_weekday_am": ("2024-06", "08-12", "weekday"),
    "jun_weekday_pm": ("2024-06", "16-20", "weekday"),
    "jun_late_night": ("2024-06", "20-24", "weekend"),
}


@dataclass(frozen=True)
class GateSplit:
    """Leakage-safe train/holdout split for Path 2 gate training."""

    train_ids: list[str]
    holdout_ids: list[str]
    holdout_groups: list[tuple[str, str, str]]


@dataclass
class GateDataset:
    """Precomputed leave-one-block-out retrieval matrices."""

    query_ids: list[str]
    candidate_ids: list[list[str]]
    query_features: np.ndarray
    component_matrices: list[np.ndarray]
    demand_errors: list[np.ndarray]
    split: GateSplit
    feature_mean: np.ndarray
    feature_std: np.ndarray


class SimilarityGate(nn.Module):
    """MLP gate g_theta(x) -> simplex over six similarity components."""

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
        init_weights: np.ndarray | None = None,
    ):
        super().__init__()
        self._hidden = tuple(hidden)
        self._dropout = float(dropout)
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.extend([
                nn.Linear(prev, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = width
        layers.append(nn.Linear(prev, len(COMPONENT_KEYS)))
        self.net = nn.Sequential(*layers)
        if init_weights is not None:
            arr = np.asarray(init_weights, dtype=np.float32)
            arr = arr / max(float(arr.sum()), 1e-9)
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                with torch.no_grad():
                    final.bias.copy_(torch.log(torch.tensor(arr) + 1e-9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class LearnedWeightFunction:
    """Callable weight provider compatible with ``compute_similarity``."""

    def __init__(self, model: SimilarityGate, feature_mean: np.ndarray, feature_std: np.ndarray):
        self.model = model.eval()
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)

    def __call__(
        self,
        q_series: np.ndarray,
        q_events: list[SurgeEvent],
        q_block_id: str | None = None,
        record: RegimeRecord | None = None,
    ) -> dict[str, float]:
        feats = query_feature_vector(q_series, q_events, q_block_id, record)
        x = ((feats - self.feature_mean) / np.maximum(self.feature_std, 1e-6)).astype(np.float32)
        with torch.no_grad():
            w = self.model(torch.tensor(x).unsqueeze(0)).squeeze(0).cpu().numpy()
        return weights_to_dict(w)


def parse_block_id(block_id: str) -> tuple[pd.Timestamp, str]:
    date_str, hour_range = block_id.rsplit("_", 1)
    return pd.Timestamp(date_str), hour_range


def day_type_for_timestamp(ts: pd.Timestamp) -> str:
    if ts.month == 1 and ts.day == 1:
        return "holiday"
    return "weekend" if ts.dayofweek >= 5 else "weekday"


def block_group(block_id: str) -> tuple[str, str, str]:
    ts, hour_range = parse_block_id(block_id)
    return f"{ts.year:04d}-{ts.month:02d}", day_type_for_timestamp(ts), hour_range


def valid_record(record: RegimeRecord, min_bins: int = 10, min_demand: float = 100.0) -> bool:
    return (
        len(record.demand_series) >= min_bins
        and float(np.sum(record.demand_series)) >= min_demand
    )


def leakage_safe_split(library: RegimeLibrary) -> GateSplit:
    """Hold out every group matching one of the eight preregistered scenarios."""
    holdout_groups = sorted({
        (month, day_type, hour_range)
        for month, hour_range, day_type in PATH2_SCENARIOS.values()
    })
    holdout_set = set(holdout_groups)
    train_ids: list[str] = []
    holdout_ids: list[str] = []

    for bid, rec in sorted(library.records.items()):
        if not valid_record(rec):
            continue
        grp = block_group(bid)
        if grp in holdout_set:
            holdout_ids.append(bid)
        else:
            train_ids.append(bid)

    return GateSplit(train_ids=train_ids, holdout_ids=holdout_ids, holdout_groups=holdout_groups)


def query_feature_vector(
    series: np.ndarray,
    events: list[SurgeEvent],
    block_id: str | None,
    record: RegimeRecord | None = None,
) -> np.ndarray:
    """Feature vector for the gate.

    Uses demand summaries, autocorrelation, event summaries, contextual one-hots,
    and compact spatial moments when a record is available.
    """
    x = np.asarray(series, dtype=np.float64)
    if len(x) == 0:
        x = np.zeros(1, dtype=np.float64)

    base = [
        float(np.mean(x)),
        float(np.std(x)),
        float(pd.Series(x).skew()) if len(x) > 2 else 0.0,
        float(pd.Series(x).kurtosis()) if len(x) > 3 else 0.0,
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 75) - np.percentile(x, 25)),
        float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 1 else 0.0,
    ]
    for lag in (1, 2, 3):
        if len(x) > lag and np.std(x[:-lag]) > 0 and np.std(x[lag:]) > 0:
            ac = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
            base.append(ac if np.isfinite(ac) else 0.0)
        else:
            base.append(0.0)

    intensities = np.array([e.intensity_zscore for e in events], dtype=np.float64)
    durations = np.array([e.duration_bins for e in events], dtype=np.float64)
    event_feats = [
        float(len(events)),
        float(np.mean(intensities)) if len(intensities) else 0.0,
        float(np.max(intensities)) if len(intensities) else 0.0,
        float(np.mean(durations)) if len(durations) else 0.0,
        float(sum(1 for e in events if e.sign > 0)),
        float(sum(1 for e in events if e.sign < 0)),
    ]

    hour_onehot = np.zeros(6, dtype=np.float64)
    day_onehot = np.zeros(3, dtype=np.float64)
    month_onehot = np.zeros(12, dtype=np.float64)
    if block_id:
        try:
            ts, hour_range = parse_block_id(block_id)
            hour_idx = int(hour_range.split("-")[0]) // 4
            hour_onehot[max(0, min(hour_idx, 5))] = 1.0
            day_map = {"weekday": 0, "weekend": 1, "holiday": 2}
            day_onehot[day_map[day_type_for_timestamp(ts)]] = 1.0
            month_onehot[ts.month - 1] = 1.0
        except (ValueError, IndexError):
            pass

    spatial = spatial_feature_vector(record)
    return np.concatenate([
        np.array(base + event_feats, dtype=np.float64),
        hour_onehot,
        day_onehot,
        month_onehot,
        spatial,
    ])


def spatial_feature_vector(record: RegimeRecord | None) -> np.ndarray:
    """Compact OD distribution proxy from stored coordinate pools."""
    if record is None:
        return np.zeros(6, dtype=np.float64)
    meta = record.metadata or {}
    required = ["pickup_lons", "pickup_lats", "dropoff_lons", "dropoff_lats"]
    if not all(k in meta and len(meta[k]) for k in required):
        return np.zeros(6, dtype=np.float64)
    pu_lon = np.asarray(meta["pickup_lons"], dtype=np.float64)
    pu_lat = np.asarray(meta["pickup_lats"], dtype=np.float64)
    do_lon = np.asarray(meta["dropoff_lons"], dtype=np.float64)
    do_lat = np.asarray(meta["dropoff_lats"], dtype=np.float64)
    return np.array([
        float(np.mean(pu_lon)),
        float(np.mean(pu_lat)),
        float(np.mean(do_lon)),
        float(np.mean(do_lat)),
        float(np.std(pu_lon) + np.std(pu_lat)),
        float(np.std(do_lon) + np.std(do_lat)),
    ], dtype=np.float64)


def demand_error(a: np.ndarray, b: np.ndarray) -> float:
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    aa = np.pad(a, (0, n - len(a)), constant_values=a[-1] if len(a) else 0)
    bb = np.pad(b, (0, n - len(b)), constant_values=b[-1] if len(b) else 0)
    return float(np.linalg.norm(aa - bb))


def build_gate_dataset(
    library: RegimeLibrary,
    query_ids: Iterable[str],
    candidate_pool_ids: Iterable[str],
    split: GateSplit,
    *,
    normalize_from: Iterable[str] | None = None,
) -> GateDataset:
    """Precompute component score matrices and demand errors."""
    qids = [bid for bid in query_ids if bid in library.records and valid_record(library[bid])]
    pool = [
        bid for bid in candidate_pool_ids
        if bid in library.records and valid_record(library[bid])
    ]

    features = []
    candidate_ids: list[list[str]] = []
    matrices: list[np.ndarray] = []
    errors: list[np.ndarray] = []

    for qid in qids:
        qrec = library[qid]
        features.append(query_feature_vector(qrec.demand_series, qrec.events, qid, qrec))
        cids = [cid for cid in pool if cid != qid]
        candidate_ids.append(cids)

        comp_rows = []
        err_rows = []
        for cid in cids:
            crec = library[cid]
            comp_rows.append(
                component_scores(qrec.demand_series, qrec.events, crec, q_block_id=qid)
            )
            err_rows.append(demand_error(qrec.demand_series, crec.demand_series))
        matrices.append(np.vstack(comp_rows).astype(np.float32))
        errors.append(np.asarray(err_rows, dtype=np.float32))

    feature_arr = (
        np.vstack(features).astype(np.float32)
        if features else np.zeros((0, 1), dtype=np.float32)
    )
    norm_ids = set(normalize_from or qids)
    norm_features = [
        query_feature_vector(library[bid].demand_series, library[bid].events, bid, library[bid])
        for bid in norm_ids
        if bid in library.records and valid_record(library[bid])
    ]
    norm_arr = np.vstack(norm_features).astype(np.float32) if norm_features else feature_arr
    mean = norm_arr.mean(axis=0)
    std = norm_arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return GateDataset(
        query_ids=qids,
        candidate_ids=candidate_ids,
        query_features=feature_arr,
        component_matrices=matrices,
        demand_errors=errors,
        split=split,
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
    )


def train_gate(
    dataset: GateDataset,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    temperature: float = 0.05,
    seed: int = 42,
    top_k: int = 20,
    loss_name: str = "topk",
    target_errors: list[np.ndarray] | None = None,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
) -> tuple[SimilarityGate, list[dict[str, float]]]:
    """Train gate with top-k/ranking loss over demand or custom target errors."""
    torch.manual_seed(seed)
    x = ((dataset.query_features - dataset.feature_mean) / dataset.feature_std).astype(np.float32)
    init = default_weight_vectors()["distributional"]
    model = SimilarityGate(
        input_dim=x.shape[1],
        hidden=hidden,
        dropout=dropout,
        init_weights=init,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []

    x_t = torch.tensor(x, dtype=torch.float32)
    comp_t = [torch.tensor(m, dtype=torch.float32) for m in dataset.component_matrices]
    err_t = []
    objective_errors = target_errors or dataset.demand_errors
    for e in objective_errors:
        scale = float(np.median(e)) if len(e) else 1.0
        err_t.append(torch.tensor(e / max(scale, 1e-6), dtype=torch.float32))

    for ep in range(epochs):
        model.train()
        weights = model(x_t)
        losses = []
        entropies = []
        for i, (comp, err) in enumerate(zip(comp_t, err_t)):
            sims = comp @ weights[i]
            k = min(top_k, sims.numel())
            if loss_name == "pairwise":
                good_idx = torch.topk(-err, k=k).indices
                bad_idx = torch.topk(err, k=k).indices
                margin = sims[good_idx].unsqueeze(1) - sims[bad_idx].unsqueeze(0)
                losses.append(torch.nn.functional.softplus(-margin / temperature).mean())
            else:
                vals, idx = torch.topk(sims, k=k)
                alpha = torch.softmax(vals / temperature, dim=0)
                losses.append(torch.sum(alpha * err[idx]))
            entropies.append(-torch.sum(weights[i] * torch.log(weights[i] + 1e-9)))
        loss = torch.stack(losses).mean() - 0.005 * torch.stack(entropies).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            history.append({"epoch": ep + 1, "loss": float(loss.detach().cpu())})

    return model.eval(), history


def score_with_weights(component_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return component_matrix @ weights.astype(np.float32)


def evaluate_dataset(
    dataset: GateDataset,
    method_weights: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Evaluate Spearman and top-k demand error for static or per-query weights."""
    rows = []
    for method, weights in method_weights.items():
        rhos = []
        top5_errors = []
        median_errors = []
        for i, qid in enumerate(dataset.query_ids):
            w = weights[i] if weights.ndim == 2 else weights
            scores = score_with_weights(dataset.component_matrices[i], w)
            errs = dataset.demand_errors[i]
            rho, p = stats.spearmanr(scores, -errs)
            if not np.isfinite(rho):
                rho, p = 0.0, 1.0
            order = np.argsort(-scores)
            rhos.append(float(rho))
            top5_errors.append(float(np.mean(errs[order[:5]])))
            median_errors.append(float(np.median(errs)))
            rows.append({
                "method": method,
                "query_id": qid,
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "top5_mean_error": top5_errors[-1],
                "median_error": median_errors[-1],
                "top5_vs_median_improvement_pct": (
                    1.0 - top5_errors[-1] / median_errors[-1]
                ) * 100.0,
            })
        rows.append({
            "method": method,
            "query_id": "__MEAN__",
            "spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
            "spearman_p": np.nan,
            "top5_mean_error": float(np.mean(top5_errors)) if top5_errors else 0.0,
            "median_error": float(np.mean(median_errors)) if median_errors else 0.0,
            "top5_vs_median_improvement_pct": (
                1.0 - float(np.mean(top5_errors)) / max(float(np.mean(median_errors)), 1e-9)
            ) * 100.0 if top5_errors else 0.0,
        })
    return pd.DataFrame(rows)


def optimize_global_weights_random(
    dataset: GateDataset,
    *,
    n_trials: int = 512,
    seed: int = 42,
    objective: str = "spearman",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Dependency-free random-simplex search for a static weight baseline.

    This is a lightweight stand-in for BO in Phase 2 until an optimizer
    dependency is added. It searches Dirichlet samples over the same frozen
    train split and reports the best static six-vector.
    """
    rng = np.random.default_rng(seed)
    candidates = [default_weight_vectors()["distributional"]]
    candidates.extend(rng.dirichlet(np.ones(len(COMPONENT_KEYS)), size=n_trials))
    rows = []
    best_score = -np.inf
    best_w = np.asarray(candidates[0], dtype=np.float32)

    for trial, w in enumerate(candidates):
        eval_df = evaluate_dataset(dataset, {"candidate": np.asarray(w, dtype=np.float32)})
        mean_row = eval_df[eval_df["query_id"] == "__MEAN__"].iloc[0]
        score = (
            float(mean_row["spearman_rho"])
            if objective == "spearman"
            else float(mean_row["top5_vs_median_improvement_pct"])
        )
        rows.append({
            "trial": trial,
            "score": score,
            "spearman_rho": float(mean_row["spearman_rho"]),
            "top5_vs_median_improvement_pct": float(mean_row["top5_vs_median_improvement_pct"]),
            **weights_to_dict(w),
        })
        if score > best_score:
            best_score = score
            best_w = np.asarray(w, dtype=np.float32)

    return best_w / best_w.sum(), pd.DataFrame(rows)


def default_weight_vectors() -> dict[str, np.ndarray]:
    cfg = get_config()["regime"]["similarity_weights"]
    hand = np.array([cfg.get(k, 0.0) for k in COMPONENT_KEYS], dtype=np.float32)
    uniform = np.ones(len(COMPONENT_KEYS), dtype=np.float32) / len(COMPONENT_KEYS)
    dist = np.array([0.35, 0.35, 0.15, 0.15, 0.0, 0.0], dtype=np.float32)
    no_temporal = np.array([0.25, 0.25, 0.15, 0.10, 0.25, 0.0], dtype=np.float32)
    return {
        "hand_tuned": hand / hand.sum(),
        "uniform": uniform,
        "distributional": dist / dist.sum(),
        "no_temporal": no_temporal / no_temporal.sum(),
    }


def model_weight_matrix(model: SimilarityGate, dataset: GateDataset) -> np.ndarray:
    x = ((dataset.query_features - dataset.feature_mean) / dataset.feature_std).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(x)).cpu().numpy().astype(np.float32)


def save_gate(
    model: SimilarityGate,
    dataset: GateDataset,
    out_dir: Path,
    history: list[dict[str, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(dataset.query_features.shape[1]),
        "feature_mean": dataset.feature_mean.tolist(),
        "feature_std": dataset.feature_std.tolist(),
        "component_keys": COMPONENT_KEYS,
        "hidden": list(getattr(model, "_hidden", (64, 32))),
        "dropout": float(getattr(model, "_dropout", 0.1)),
    }, out_dir / "similarity_gate.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


def load_gate(path: Path | str) -> LearnedWeightFunction:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SimilarityGate(
        input_dim=int(ckpt["input_dim"]),
        hidden=tuple(ckpt.get("hidden", (64, 32))),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    return LearnedWeightFunction(
        model,
        np.asarray(ckpt["feature_mean"], dtype=np.float32),
        np.asarray(ckpt["feature_std"], dtype=np.float32),
    )


def export_weight_summary(
    dataset: GateDataset,
    learned_weights: np.ndarray,
    out_path: Path,
) -> None:
    """Write per-query and per-group learned weights to JSON."""
    rows = []
    for qid, w in zip(dataset.query_ids, learned_weights):
        rows.append({
            "query_id": qid,
            "group": list(block_group(qid)),
            "weights": weights_to_dict(w),
        })
    grouped: dict[str, list[np.ndarray]] = {}
    for qid, w in zip(dataset.query_ids, learned_weights):
        grouped.setdefault("|".join(block_group(qid)), []).append(w)
    group_rows = {
        grp: weights_to_dict(np.mean(np.vstack(vals), axis=0))
        for grp, vals in grouped.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_query": rows, "per_group": group_rows}, f, indent=2)
