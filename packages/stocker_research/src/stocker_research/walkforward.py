"""Walk-forward split helpers."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class WalkForwardSplit(BaseModel, frozen=True):
    """Index ranges for one train/test walk-forward split."""

    split_id: str = "split_001"
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_timestamp: str | None = None
    train_end_timestamp: str | None = None
    test_start_timestamp: str | None = None
    test_end_timestamp: str | None = None


class WalkForwardConfig(BaseModel, frozen=True):
    """Configuration for time-ordered walk-forward splits."""

    mode: Literal["rolling", "expanding", "fixed"] = "rolling"
    train_size: int = Field(gt=0)
    test_size: int = Field(gt=0)
    step_size: int | None = Field(default=None, gt=0)
    embargo_bars: int = Field(default=0, ge=0)
    min_rows: int = Field(default=0, ge=0)


def _timestamp_at(frame: pd.DataFrame, index: int) -> str | None:
    if "timestamp" not in frame or frame.empty:
        return None
    safe_index = min(max(index, 0), len(frame) - 1)
    return str(frame.iloc[safe_index]["timestamp"])


def generate_walk_forward_splits(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
) -> list[WalkForwardSplit]:
    """Generate deterministic ordered walk-forward splits."""

    n_rows = len(frame)
    if n_rows < max(config.min_rows, config.train_size + config.test_size):
        return []

    step = config.step_size or config.test_size
    splits: list[WalkForwardSplit] = []
    split_no = 1
    offset = 0
    while True:
        if config.mode == "expanding":
            train_start = 0
            train_end = config.train_size + offset
        elif config.mode == "fixed":
            train_start = 0
            train_end = config.train_size
            if splits:
                break
        else:
            train_start = offset
            train_end = train_start + config.train_size

        test_start = train_end + config.embargo_bars
        test_end = test_start + config.test_size
        if test_end > n_rows:
            break
        splits.append(
            WalkForwardSplit(
                split_id=f"split_{split_no:03d}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_start_timestamp=_timestamp_at(frame, train_start),
                train_end_timestamp=_timestamp_at(frame, train_end - 1),
                test_start_timestamp=_timestamp_at(frame, test_start),
                test_end_timestamp=_timestamp_at(frame, test_end - 1),
            )
        )
        split_no += 1
        offset += step
    return splits


def rolling_walkforward_splits(
    n_rows: int, *, train_size: int, test_size: int, step_size: int | None = None
) -> list[WalkForwardSplit]:
    """Create simple rolling index splits for future research experiments."""

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    splits: list[WalkForwardSplit] = []
    train_start = 0
    while True:
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n_rows:
            break
        splits.append(
            WalkForwardSplit(
                split_id=f"split_{len(splits) + 1:03d}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start += step
    return splits
