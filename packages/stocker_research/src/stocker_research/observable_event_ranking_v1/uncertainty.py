"""Paired session-block uncertainty estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def session_block_sample_indices(
    slate_metrics: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> list[np.ndarray]:
    """Return bootstrap row indices, always preserving complete session blocks."""

    sessions = np.asarray(sorted(slate_metrics["session"].unique()), dtype=object)
    if len(sessions) == 0:
        raise ValueError("session bootstrap requires at least one session")
    session_indices = {
        session: slate_metrics.index[slate_metrics["session"].eq(session)].to_numpy(dtype="int64")
        for session in sessions
    }
    generator = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for _ in range(draws):
        sampled = generator.choice(sessions, size=len(sessions), replace=True)
        output.append(np.concatenate([session_indices[session] for session in sampled]))
    return output


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and paired percentile confidence interval."""

    estimate: float
    lower: float
    upper: float
    draws: int
    seed: int


def paired_session_block_bootstrap(
    slate_metrics: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    draws: int = 2_000,
    seed: int = 20260719,
) -> BootstrapResult:
    """Bootstrap paired candidate-minus-baseline means by complete session blocks."""

    paired = slate_metrics[["session", candidate_column, baseline_column]].dropna().copy()
    differences = paired[candidate_column] - paired[baseline_column]
    sampled_indices = session_block_sample_indices(paired, draws=draws, seed=seed)
    estimates = np.asarray(
        [float(differences.loc[indices].mean()) for indices in sampled_indices],
        dtype="float64",
    )
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return BootstrapResult(
        estimate=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        draws=draws,
        seed=seed,
    )
