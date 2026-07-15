"""Paired prequential metrics for the sequential-versus-anchor comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def paired_economic_contribution(frame: pd.DataFrame) -> pd.Series:
    """Return the row-level economic increment used by the paired endpoint."""

    required = {
        "target_remaining_net_bps",
        "anchor_probability",
        "sequential_probability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing paired contribution columns: {sorted(missing)}")
    target = pd.to_numeric(frame["target_remaining_net_bps"], errors="coerce")
    anchor = pd.to_numeric(frame["anchor_probability"], errors="coerce")
    sequential = pd.to_numeric(frame["sequential_probability"], errors="coerce")
    return target * (sequential - anchor)


def paired_predictive_metrics(
    frame: pd.DataFrame,
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 20260715,
) -> dict[str, float | int]:
    """Evaluate sequence probabilities against the same target and population."""

    required = {
        "session_date",
        "target_positive",
        "target_remaining_net_bps",
        "anchor_probability",
        "sequential_probability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing paired metric columns: {sorted(missing)}")
    paired = frame.dropna(subset=list(required)).copy()
    if paired.empty:
        return {
            "paired_rows": 0,
            "brier_improvement": math.nan,
            "log_loss_improvement": math.nan,
            "paired_economic_increment_bps": math.nan,
            "brier_interval_lower": math.nan,
            "brier_interval_upper": math.nan,
        }
    y = pd.to_numeric(paired["target_positive"], errors="raise").to_numpy(float)
    anchor = pd.to_numeric(paired["anchor_probability"], errors="raise").to_numpy(float)
    sequential = pd.to_numeric(paired["sequential_probability"], errors="raise").to_numpy(float)
    row_improvement = (anchor - y) ** 2 - (sequential - y) ** 2
    brier_improvement = float(np.mean(row_improvement))
    log_improvement = _log_loss(y, anchor) - _log_loss(y, sequential)
    economic_increment = float(paired_economic_contribution(paired).sum(min_count=1))

    sessions = sorted(paired["session_date"].astype(str).unique())
    by_session = {
        session: row_improvement[paired["session_date"].astype(str).to_numpy() == session]
        for session in sessions
    }
    values = np.full(bootstrap_resamples, brier_improvement, dtype=float)
    if bootstrap_resamples > 0 and sessions:
        rng = np.random.default_rng(seed)
        block = min(5, len(sessions))
        blocks = int(math.ceil(len(sessions) / block))
        for draw in range(bootstrap_resamples):
            starts = rng.integers(0, len(sessions), size=blocks)
            sampled_sessions = [
                sessions[(int(start) + offset) % len(sessions)]
                for start in starts
                for offset in range(block)
            ][: len(sessions)]
            sampled = np.concatenate([by_session[session] for session in sampled_sessions])
            values[draw] = float(np.mean(sampled))
    return {
        "paired_rows": int(len(paired)),
        "anchor_brier": float(np.mean((anchor - y) ** 2)),
        "sequential_brier": float(np.mean((sequential - y) ** 2)),
        "brier_improvement": brier_improvement,
        "anchor_log_loss": _log_loss(y, anchor),
        "sequential_log_loss": _log_loss(y, sequential),
        "log_loss_improvement": log_improvement,
        "paired_economic_increment_bps": economic_increment,
        "brier_interval_lower": float(np.quantile(values, 0.025)),
        "brier_interval_upper": float(np.quantile(values, 0.975)),
    }
