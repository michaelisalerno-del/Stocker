"""Research leakage and lookahead-bias checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from stocker_research.walkforward import WalkForwardSplit


@dataclass(frozen=True)
class LeakageIssue:
    """One suspicious research-design issue."""

    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable issue data."""

        return asdict(self)


def check_same_bar_close_signal(*, uses_close: bool, execution_lag_bars: int) -> list[LeakageIssue]:
    """Detect signals that use same-bar close for same-bar execution."""

    if uses_close and execution_lag_bars <= 0:
        return [
            LeakageIssue(
                code="same_bar_close",
                message="Signal uses same-bar close without execution lag",
            )
        ]
    return []


def check_feature_target_overlap(
    feature_columns: list[str],
    target_columns: list[str],
) -> list[LeakageIssue]:
    """Detect target columns included in feature columns."""

    overlap = sorted(set(feature_columns).intersection(target_columns))
    issues: list[LeakageIssue] = []
    if overlap:
        issues.append(
            LeakageIssue(
                code="target_in_features",
                message=f"Target columns are present in features: {', '.join(overlap)}",
            )
        )
    suspicious = [column for column in feature_columns if "future" in column.lower()]
    if suspicious:
        issues.append(
            LeakageIssue(
                code="future_feature_name",
                message=f"Feature names suggest future leakage: {', '.join(sorted(suspicious))}",
                severity="warning",
            )
        )
    return issues


def check_train_test_overlap(split: WalkForwardSplit) -> list[LeakageIssue]:
    """Detect train/test overlap or reversed split order."""

    if split.train_end > split.test_start:
        return [
            LeakageIssue(
                code="train_test_overlap",
                message=f"{split.split_id} train rows overlap test rows",
            )
        ]
    if split.train_end == split.test_start:
        return []
    return []


def check_future_timestamps(
    feature_timestamps: list[Any],
    prediction_timestamps: list[Any],
) -> list[LeakageIssue]:
    """Detect features timestamped after their corresponding prediction time."""

    for feature_time, prediction_time in zip(
        feature_timestamps, prediction_timestamps, strict=False
    ):
        if feature_time > prediction_time:
            return [
                LeakageIssue(
                    code="future_timestamp",
                    message="Feature timestamp is after prediction timestamp",
                )
            ]
    return []
