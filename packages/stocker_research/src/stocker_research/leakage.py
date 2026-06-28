"""Research leakage and lookahead-bias checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

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


def check_embargo_violation(split: WalkForwardSplit, *, embargo_bars: int) -> list[LeakageIssue]:
    """Detect a split that does not leave the required train/test embargo gap."""

    actual_gap = split.test_start - split.train_end
    if actual_gap < embargo_bars:
        return [
            LeakageIssue(
                code="embargo_violation",
                message=(f"{split.split_id} has embargo gap {actual_gap}, required {embargo_bars}"),
            )
        ]
    return []


def check_timestamp_integrity(
    frame: pd.DataFrame, *, column: str = "timestamp"
) -> list[LeakageIssue]:
    """Detect duplicate or non-monotonic research timestamps."""

    if column not in frame:
        return [
            LeakageIssue(
                code="missing_timestamp",
                message=f"Missing timestamp column: {column}",
            )
        ]
    timestamps = pd.to_datetime(frame[column], errors="coerce")
    issues: list[LeakageIssue] = []
    if timestamps.duplicated().any():
        issues.append(
            LeakageIssue(
                code="duplicate_timestamps",
                message="Research data contains duplicate timestamps",
            )
        )
    if not timestamps.is_monotonic_increasing:
        issues.append(
            LeakageIssue(
                code="non_monotonic_timestamps",
                message="Research timestamps are not monotonic increasing",
            )
        )
    return issues


def check_signal_quality(
    signal: pd.Series, *, max_nan_fraction: float = 0.25
) -> list[LeakageIssue]:
    """Detect signal output that is too sparse or suspiciously NaN-heavy."""

    if signal.empty:
        return [LeakageIssue(code="empty_signal", message="Signal output is empty")]
    nan_fraction = float(signal.isna().mean())
    if nan_fraction > max_nan_fraction:
        return [
            LeakageIssue(
                code="nan_heavy_signal",
                message=f"Signal NaN fraction {nan_fraction:.3f} exceeds {max_nan_fraction:.3f}",
            )
        ]
    return []


def check_suspicious_perfect_prediction(
    signal: pd.Series,
    future_returns: pd.Series,
    *,
    threshold: float = 0.995,
) -> list[LeakageIssue]:
    """Flag extremely high signal/target correlation as suspicious."""

    aligned = pd.concat(
        [
            pd.to_numeric(signal, errors="coerce").rename("signal"),
            pd.to_numeric(future_returns, errors="coerce").rename("future_returns"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 3:
        return []
    if (
        aligned["signal"].nunique(dropna=True) < 2
        or aligned["future_returns"].nunique(dropna=True) < 2
    ):
        return []
    correlation = aligned["signal"].corr(aligned["future_returns"])
    if correlation is not None and abs(float(correlation)) >= threshold:
        return [
            LeakageIssue(
                code="suspicious_perfect_prediction",
                message=f"Signal/target correlation {correlation:.4f} is suspiciously high",
                severity="warning",
            )
        ]
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


def _numeric_signal(signals: pd.DataFrame, column: str) -> pd.Series:
    if column not in signals:
        return pd.Series(dtype=float)
    return pd.to_numeric(signals[column], errors="coerce").reset_index(drop=True)


def _deduplicate_issues(issues: list[LeakageIssue]) -> list[LeakageIssue]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[LeakageIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.severity)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def collect_research_leakage_issues(
    *,
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    signals: pd.DataFrame,
    embargo_bars: int,
    timestamp_column: str = "timestamp",
) -> list[LeakageIssue]:
    """Collect basic leakage checks used by the research runner."""

    issues: list[LeakageIssue] = []
    issues.extend(check_timestamp_integrity(frame, column=timestamp_column))
    for split in splits:
        issues.extend(check_train_test_overlap(split))
        issues.extend(check_embargo_violation(split, embargo_bars=embargo_bars))

    signal = _numeric_signal(signals, "signal")
    target_position = _numeric_signal(signals, "target_position")
    issues.extend(check_signal_quality(signal))
    if target_position.empty:
        issues.append(
            LeakageIssue(code="empty_target_position", message="Target position is empty")
        )
    else:
        issues.extend(check_signal_quality(target_position))

    if "close" in frame and not frame.empty:
        close = pd.to_numeric(frame["close"], errors="coerce").reset_index(drop=True)
        future_returns = close.pct_change().shift(-1)
        issues.extend(check_suspicious_perfect_prediction(signal, future_returns))
        issues.extend(check_suspicious_perfect_prediction(target_position, future_returns))

    return _deduplicate_issues(issues)
