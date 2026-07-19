"""Expanding chronological development folds."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalFold:
    """One immutable expanding training/evaluation split."""

    fold_id: str
    training_months: tuple[str, ...]
    evaluation_months: tuple[str, ...]
    train_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]


def build_expanding_folds(
    target_ledger: pd.DataFrame,
    *,
    initial_training_months: int = 6,
    evaluation_months: int = 3,
) -> list[ChronologicalFold]:
    """Build non-overlapping three-month evaluations after at least six training months."""

    frame = target_ledger.copy()
    frame["session"] = pd.to_datetime(frame["session"], utc=True)
    eligible = frame["slate_evaluable"].astype(bool) & frame["target_rank_60m"].notna()
    month_series = frame["session"].dt.tz_localize(None).dt.to_period("M").astype(str)
    observed_months = month_series.loc[eligible]
    if observed_months.empty:
        return []
    months = [
        str(month)
        for month in pd.period_range(
            start=observed_months.min(),
            end=observed_months.max(),
            freq="M",
        )
    ]
    folds: list[ChronologicalFold] = []
    start = initial_training_months
    fold_number = 1
    while start + evaluation_months <= len(months):
        train_months = tuple(months[:start])
        eval_months = tuple(months[start : start + evaluation_months])
        train_indices = tuple(frame.index[eligible & month_series.isin(train_months)])
        eval_indices = tuple(frame.index[eligible & month_series.isin(eval_months)])
        if train_indices and eval_indices:
            latest_train = frame.loc[list(train_indices), "session"].max()
            earliest_eval = frame.loc[list(eval_indices), "session"].min()
            if latest_train >= earliest_eval:
                raise ValueError("chronological fold overlap")
            train_events = set(frame.loc[list(train_indices), "event_id"])
            eval_events = set(frame.loc[list(eval_indices), "event_id"])
            if not train_events.isdisjoint(eval_events):
                raise ValueError("event appears in both training and evaluation")
            folds.append(
                ChronologicalFold(
                    fold_id=f"fold_{fold_number:02d}",
                    training_months=train_months,
                    evaluation_months=eval_months,
                    train_indices=train_indices,
                    evaluation_indices=eval_indices,
                )
            )
            fold_number += 1
        start += evaluation_months
    return folds
