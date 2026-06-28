"""Walk-forward split helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    """Index ranges for one train/test walk-forward split."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


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
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start += step
    return splits
