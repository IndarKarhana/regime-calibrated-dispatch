"""Surge / dip event detection and prefix-window feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import get_config


@dataclass
class SurgeEvent:
    time_idx: int
    bin_start: pd.Timestamp | None
    sign: int  # +1 = surge, -1 = dip
    intensity_zscore: float
    duration_bins: int
    sharpness: float  # max |first-difference| in the event window
    auc_above_baseline: float
    prefix_features: dict = field(default_factory=dict)


def detect_events_rolling_mad(
    series: np.ndarray,
    threshold: float | None = None,
    min_duration: int | None = None,
    window: int = 12,
) -> list[SurgeEvent]:
    """Detect surges/dips using rolling median + MAD. Returns list of SurgeEvent."""
    cfg = get_config()["regime"]["event_detection"]
    threshold = threshold or cfg["mad_threshold"]
    min_duration = min_duration or cfg["min_duration_bins"]

    n = len(series)
    if n < window:
        return []

    med = pd.Series(series).rolling(window, center=True, min_periods=max(1, window // 2)).median()
    mad = pd.Series(np.abs(series - med)).rolling(
        window, center=True, min_periods=max(1, window // 2)
    ).median()
    mad = np.maximum(mad.values, 1e-9)
    z = (series - med.values) / (1.4826 * mad)

    events: list[SurgeEvent] = []
    i = 0
    while i < n:
        if abs(z[i]) >= threshold:
            sign = 1 if z[i] > 0 else -1
            start = i
            while i < n and z[i] * sign >= threshold * 0.5:
                i += 1
            end = i
            dur = end - start
            if dur >= min_duration:
                seg = series[start:end]
                baseline = med.values[start:end]
                diff = np.diff(seg)
                events.append(SurgeEvent(
                    time_idx=start,
                    bin_start=None,
                    sign=sign,
                    intensity_zscore=float(np.max(np.abs(z[start:end]))),
                    duration_bins=dur,
                    sharpness=float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0,
                    auc_above_baseline=float(np.sum(np.abs(seg - baseline))),
                ))
        else:
            i += 1
    return events


def detect_events_pelt(series: np.ndarray) -> list[SurgeEvent]:
    """Detect regime changepoints via PELT (ruptures) and derive events."""
    import ruptures as rpt

    n = len(series)
    if n < 10:
        return []
    algo = rpt.Pelt(model="rbf", min_size=3).fit(series.reshape(-1, 1))
    bkps = algo.predict(pen=np.std(series) * 2)

    boundaries = [0] + bkps
    events: list[SurgeEvent] = []
    global_med = np.median(series)
    global_mad = max(np.median(np.abs(series - global_med)) * 1.4826, 1e-9)

    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k + 1]
        seg = series[s:e]
        seg_mean = np.mean(seg)
        z = (seg_mean - global_med) / global_mad
        if abs(z) < 1.5:
            continue
        sign = 1 if z > 0 else -1
        diff = np.diff(seg)
        events.append(SurgeEvent(
            time_idx=s,
            bin_start=None,
            sign=sign,
            intensity_zscore=float(abs(z)),
            duration_bins=e - s,
            sharpness=float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0,
            auc_above_baseline=float(np.sum(np.abs(seg - global_med))),
        ))
    return events


def detect_events(series: np.ndarray, method: str | None = None) -> list[SurgeEvent]:
    cfg = get_config()["regime"]["event_detection"]
    method = method or cfg["method"]
    if method == "pelt":
        return detect_events_pelt(series)
    return detect_events_rolling_mad(series)


def extract_prefix_features(
    series: np.ndarray,
    event: SurgeEvent,
    bin_minutes: int = 5,
    prefix_minutes: int | None = None,
) -> dict:
    """Extract features from the window preceding an event."""
    cfg = get_config()["regime"]
    pm = prefix_minutes or cfg["prefix_window_minutes"]
    prefix_bins = pm // bin_minutes

    start = max(0, event.time_idx - prefix_bins)
    prefix = series[start: event.time_idx]
    if len(prefix) < 3:
        return {"prefix_len": len(prefix)}

    feats: dict = {
        "prefix_len": len(prefix),
        "prefix_mean": float(np.mean(prefix)),
        "prefix_std": float(np.std(prefix)),
        "prefix_slope": float(np.polyfit(range(len(prefix)), prefix, 1)[0]),
        "prefix_entropy": float(_entropy(prefix)),
    }

    for lag in (1, 2, 3):
        if len(prefix) > lag:
            ac = np.corrcoef(prefix[:-lag], prefix[lag:])[0, 1]
            feats[f"prefix_autocorr_lag{lag}"] = float(ac) if np.isfinite(ac) else 0.0
        else:
            feats[f"prefix_autocorr_lag{lag}"] = 0.0

    return feats


def _entropy(x: np.ndarray, bins: int = 10) -> float:
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def annotate_events(
    profile_series: np.ndarray,
    bin_starts: pd.Series | None = None,
    method: str | None = None,
    bin_minutes: int = 5,
) -> list[SurgeEvent]:
    """Full pipeline: detect events, attach prefix features and optional timestamps."""
    events = detect_events(profile_series, method=method)
    for ev in events:
        ev.prefix_features = extract_prefix_features(profile_series, ev, bin_minutes=bin_minutes)
        if bin_starts is not None and ev.time_idx < len(bin_starts):
            ev.bin_start = bin_starts.iloc[ev.time_idx]
    return events
