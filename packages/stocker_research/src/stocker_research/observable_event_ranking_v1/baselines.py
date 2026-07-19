"""Frozen observable ranking baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

_DETERMINISTIC_COLUMNS: dict[str, str] = {
    "B1_EVENT_STRENGTH": "event_strength",
    "B2_PREVIOUS_5M_MARKET_RELATIVE_RETURN": "market_relative_return_5m",
    "B3_15M_MARKET_RELATIVE_STRENGTH": "market_relative_return_15m",
    "B4_30M_MARKET_RELATIVE_STRENGTH": "market_relative_return_30m",
    "B5_15M_SECTOR_RELATIVE_STRENGTH": "sector_relative_return_15m",
    "B6_ACTIVITY_SHOCK": "activity_shock_z",
    "B7_REALIZED_VOLATILITY": "realized_volatility_30m",
}


def deterministic_baseline_scores(frame: pd.DataFrame, baseline_id: str) -> pd.Series:
    """Return one frozen simple observable score without target access."""

    if baseline_id not in _DETERMINISTIC_COLUMNS:
        raise ValueError(f"not a deterministic simple baseline: {baseline_id}")
    column = _DETERMINISTIC_COLUMNS[baseline_id]
    if column not in frame:
        raise ValueError(f"baseline input missing: {column}")
    return frame[column].astype("float64").copy()


@dataclass(frozen=True)
class TrainingOnlyStockClockPrior:
    """Shrunk stock-by-clock baseline fitted only on prior training rows."""

    kind: Literal["event_frequency", "mean_target"]
    global_prior: float
    cell_scores: dict[tuple[str, str], float]
    shrinkage: float

    @classmethod
    def fit(
        cls,
        training: pd.DataFrame,
        *,
        kind: Literal["event_frequency", "mean_target"],
        shrinkage: float = 20.0,
    ) -> TrainingOnlyStockClockPrior:
        """Fit a prior with a deterministic global fallback."""

        required = {"symbol", "decision_clock", "session"}
        if kind == "mean_target":
            required.add("target_rank_60m")
        missing = sorted(required.difference(training.columns))
        if missing:
            raise ValueError(f"prior training rows missing: {missing}")
        cells: dict[tuple[str, str], float] = {}
        if kind == "mean_target":
            global_prior = float(training["target_rank_60m"].mean())
            for key, group in training.groupby(["symbol", "decision_clock"], sort=True):
                count = len(group)
                cells[(str(key[0]), str(key[1]))] = float(
                    (group["target_rank_60m"].sum() + shrinkage * global_prior)
                    / (count + shrinkage)
                )
        else:
            sessions = max(1, training["session"].nunique())
            distinct_cells = max(
                1, training[["symbol", "decision_clock"]].drop_duplicates().shape[0]
            )
            global_prior = float(len(training) / (sessions * distinct_cells))
            for key, group in training.groupby(["symbol", "decision_clock"], sort=True):
                cells[(str(key[0]), str(key[1]))] = float(
                    (len(group) + shrinkage * global_prior) / (sessions + shrinkage)
                )
        return cls(
            kind=kind,
            global_prior=global_prior,
            cell_scores=cells,
            shrinkage=shrinkage,
        )

    def score(self, symbol: str, decision_clock: str) -> float:
        """Score a cell or use the recorded global fallback when unseen."""

        return self.cell_scores.get((symbol, decision_clock), self.global_prior)
