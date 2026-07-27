"""Twenty-session EODHD-to-IBKR transfer metrics for frozen M1C V0."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final, Literal

from stocker_prospective.contract import (
    M1C_BOTTOM_5_THRESHOLD,
    M1C_BOTTOM_10_THRESHOLD,
    M1C_BOTTOM_20_THRESHOLD,
    M1C_FROZEN_THRESHOLD,
    ORIGINAL_LOW_MOVEMENT_DECISION,
    claims_boundary,
)

ProviderName = Literal["ibkr", "eodhd"]
FROZEN_CHECKPOINTS: Final[tuple[int, ...]] = tuple(range(6, 36, 2))


@dataclass(frozen=True)
class TransferBar:
    identity: str
    start_utc: datetime
    end_utc: datetime
    open: float
    high: float
    low: float
    close: float
    complete: bool

    def __post_init__(self) -> None:
        if (
            self.start_utc.tzinfo is None
            or self.start_utc.utcoffset() is None
            or self.end_utc.tzinfo is None
            or self.end_utc.utcoffset() is None
        ):
            raise ValueError("transfer bars require timezone-aware timestamps")
        if not self.identity:
            raise ValueError("transfer bar identity is required")
        if not all(math.isfinite(value) for value in (self.open, self.high, self.low, self.close)):
            raise ValueError("transfer bar OHLC must be finite")
        if self.high < max(self.open, self.close) or self.low > min(
            self.open,
            self.close,
        ):
            raise ValueError("transfer bar OHLC ordering is invalid")


@dataclass(frozen=True)
class ProviderM1CObservation:
    provider: ProviderName
    symbol: str
    session: date
    checkpoint: int
    bar: TransferBar
    features: dict[str, float]
    probability: float
    quiet_episode: bool
    high_tail_episode: bool

    def __post_init__(self) -> None:
        if self.checkpoint not in FROZEN_CHECKPOINTS:
            raise ValueError("transfer observation checkpoint is not frozen")
        if not self.symbol:
            raise ValueError("transfer observation symbol is required")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("M1C probability must lie in [0, 1]")
        if any(not math.isfinite(value) for value in self.features.values()):
            raise ValueError("M1C transfer features must be finite")

    @property
    def key(self) -> tuple[str, date, int]:
        return self.symbol, self.session, self.checkpoint


@dataclass(frozen=True)
class BarComparison:
    symbol: str
    session: date
    checkpoint: int
    ibkr_bar_identity: str
    eodhd_bar_identity: str
    timestamp_aligned: bool
    open_absolute_difference: float
    high_absolute_difference: float
    low_absolute_difference: float
    close_absolute_difference: float


@dataclass(frozen=True)
class FeatureComparison:
    symbol: str
    session: date
    checkpoint: int
    feature: str
    absolute_difference: float
    robust_scaled_difference: float
    probability_contribution_difference: float


@dataclass(frozen=True)
class BarSemanticsMetrics:
    matched_bar_count: int
    ibkr_only_bar_count: int
    eodhd_only_bar_count: int
    duplicate_ibkr_key_count: int
    duplicate_eodhd_key_count: int
    timestamp_mismatch_count: int
    duration_mismatch_count: int
    incomplete_bar_count: int
    non_equal_ohlc_count: int
    bar_interval_convention: str
    auction_print_diagnostics: str
    corporate_action_policy: str


@dataclass(frozen=True)
class ProbabilityComparison:
    symbol: str
    session: date
    checkpoint: int
    ibkr_probability: float
    eodhd_probability: float
    absolute_difference: float
    signed_difference: float


@dataclass(frozen=True)
class ProbabilityMetrics:
    count: int
    pearson: float
    spearman: float
    mean_absolute_difference: float
    median_absolute_difference: float
    p95_absolute_difference: float
    mean_signed_bias: float
    ibkr_mean: float
    eodhd_mean: float
    ibkr_standard_deviation: float
    eodhd_standard_deviation: float
    distribution_shift_detected: bool


@dataclass(frozen=True)
class TailMetrics:
    bottom_5_agreement: float
    bottom_10_agreement: float
    bottom_20_agreement: float
    high_tail_agreement: float
    far_from_threshold_agreement: float
    near_threshold_disagreement_count: int
    ibkr_frequencies: dict[str, float]
    eodhd_frequencies: dict[str, float]


@dataclass(frozen=True)
class EpisodeMetrics:
    quiet_exact_checkpoint_matches: int
    quiet_matches_within_one_checkpoint: int
    quiet_ibkr_only: int
    quiet_eodhd_only: int
    quiet_frequency_ratio: float | None
    high_exact_checkpoint_matches: int
    high_matches_within_one_checkpoint: int
    high_ibkr_only: int
    high_eodhd_only: int
    high_frequency_ratio: float | None


@dataclass(frozen=True)
class TransferReport:
    decision: str
    valid_session_count: int
    bar_semantics_passed: bool
    runtime_parity_passed: bool
    exact_vendor_bar_equality_required: bool
    bar_semantics_metrics: BarSemanticsMetrics
    probability_metrics: ProbabilityMetrics
    tail_metrics: TailMetrics
    episode_metrics: EpisodeMetrics
    bar_comparisons: tuple[BarComparison, ...]
    feature_comparisons: tuple[FeatureComparison, ...]
    probability_comparisons: tuple[ProbabilityComparison, ...]
    results_by_stock: dict[str, ProbabilityMetrics]
    results_by_checkpoint: dict[str, ProbabilityMetrics]
    results_by_time_of_day: dict[str, ProbabilityMetrics]
    largest_feature_contributors: tuple[FeatureComparison, ...]
    claims_boundary: dict[str, bool | float | str]
    historical_decision: str


@dataclass(frozen=True)
class IBKRCalibrationCandidate:
    candidate_id: str
    source: str
    thresholds: dict[str, float]
    original_v0_thresholds_continue_in_parallel: bool
    outcome_fields_used: tuple[str, ...]
    option_pnl_used: bool
    claims_boundary: dict[str, bool | float | str]


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index, _value in indexed[cursor:end]:
            result[index] = average_rank
        cursor = end
    return tuple(result)


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum * right_sum)
    if denominator == 0.0:
        return 1.0 if left == right else 0.0
    value = numerator / denominator
    if abs(value - 1.0) <= 1e-15:
        return 1.0
    if abs(value + 1.0) <= 1e-15:
        return -1.0
    return value


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _probability_metrics(
    comparisons: Iterable[ProbabilityComparison],
) -> ProbabilityMetrics:
    rows = tuple(comparisons)
    ibkr = tuple(row.ibkr_probability for row in rows)
    eodhd = tuple(row.eodhd_probability for row in rows)
    absolute = tuple(row.absolute_difference for row in rows)
    signed = tuple(row.signed_difference for row in rows)
    if not rows:
        return ProbabilityMetrics(
            count=0,
            pearson=0.0,
            spearman=0.0,
            mean_absolute_difference=0.0,
            median_absolute_difference=0.0,
            p95_absolute_difference=0.0,
            mean_signed_bias=0.0,
            ibkr_mean=0.0,
            eodhd_mean=0.0,
            ibkr_standard_deviation=0.0,
            eodhd_standard_deviation=0.0,
            distribution_shift_detected=False,
        )
    mean_absolute = statistics.fmean(absolute)
    signed_bias = statistics.fmean(signed)
    ibkr_std = statistics.pstdev(ibkr)
    eodhd_std = statistics.pstdev(eodhd)
    return ProbabilityMetrics(
        count=len(rows),
        pearson=_correlation(ibkr, eodhd),
        spearman=_correlation(_ranks(ibkr), _ranks(eodhd)),
        mean_absolute_difference=mean_absolute,
        median_absolute_difference=statistics.median(absolute),
        p95_absolute_difference=_quantile(absolute, 0.95),
        mean_signed_bias=signed_bias,
        ibkr_mean=statistics.fmean(ibkr),
        eodhd_mean=statistics.fmean(eodhd),
        ibkr_standard_deviation=ibkr_std,
        eodhd_standard_deviation=eodhd_std,
        distribution_shift_detected=(
            abs(signed_bias) >= 0.03 or mean_absolute >= 0.04 or abs(ibkr_std - eodhd_std) >= 0.04
        ),
    )


def _agreement(
    comparisons: tuple[ProbabilityComparison, ...],
    threshold: float,
) -> float:
    if not comparisons:
        return 0.0
    return statistics.fmean(
        (row.ibkr_probability <= threshold) == (row.eodhd_probability <= threshold)
        for row in comparisons
    )


def _frequency(
    comparisons: tuple[ProbabilityComparison, ...],
    *,
    provider: ProviderName,
    threshold: float,
    high: bool = False,
) -> float:
    if not comparisons:
        return 0.0
    values = (
        row.ibkr_probability if provider == "ibkr" else row.eodhd_probability for row in comparisons
    )
    if high:
        return statistics.fmean(value >= threshold for value in values)
    return statistics.fmean(value <= threshold for value in values)


def _tail_metrics(
    comparisons: tuple[ProbabilityComparison, ...],
) -> TailMetrics:
    thresholds = (
        M1C_BOTTOM_5_THRESHOLD,
        M1C_BOTTOM_10_THRESHOLD,
        M1C_BOTTOM_20_THRESHOLD,
        M1C_FROZEN_THRESHOLD,
    )
    far_rows: list[tuple[ProbabilityComparison, float]] = []
    near_disagreements = 0
    for row in comparisons:
        for threshold in thresholds:
            left = (
                row.ibkr_probability >= threshold
                if threshold == M1C_FROZEN_THRESHOLD
                else row.ibkr_probability <= threshold
            )
            right = (
                row.eodhd_probability >= threshold
                if threshold == M1C_FROZEN_THRESHOLD
                else row.eodhd_probability <= threshold
            )
            if (
                abs(row.ibkr_probability - threshold) > 0.02
                and abs(row.eodhd_probability - threshold) > 0.02
            ):
                far_rows.append((row, threshold))
            elif left != right:
                near_disagreements += 1
    far_agreement = (
        statistics.fmean(
            (
                row.ibkr_probability >= threshold
                if threshold == M1C_FROZEN_THRESHOLD
                else row.ibkr_probability <= threshold
            )
            == (
                row.eodhd_probability >= threshold
                if threshold == M1C_FROZEN_THRESHOLD
                else row.eodhd_probability <= threshold
            )
            for row, threshold in far_rows
        )
        if far_rows
        else 1.0
    )
    labels = {
        "bottom_5": (M1C_BOTTOM_5_THRESHOLD, False),
        "bottom_10": (M1C_BOTTOM_10_THRESHOLD, False),
        "bottom_20": (M1C_BOTTOM_20_THRESHOLD, False),
        "high_tail": (M1C_FROZEN_THRESHOLD, True),
    }
    provider_names: tuple[ProviderName, ...] = ("ibkr", "eodhd")
    frequencies = {
        provider: {
            label: _frequency(
                comparisons,
                provider=provider,
                threshold=threshold,
                high=high,
            )
            for label, (threshold, high) in labels.items()
        }
        for provider in provider_names
    }
    high_agreement = (
        statistics.fmean(
            (row.ibkr_probability >= M1C_FROZEN_THRESHOLD)
            == (row.eodhd_probability >= M1C_FROZEN_THRESHOLD)
            for row in comparisons
        )
        if comparisons
        else 0.0
    )
    return TailMetrics(
        bottom_5_agreement=_agreement(comparisons, M1C_BOTTOM_5_THRESHOLD),
        bottom_10_agreement=_agreement(comparisons, M1C_BOTTOM_10_THRESHOLD),
        bottom_20_agreement=_agreement(comparisons, M1C_BOTTOM_20_THRESHOLD),
        high_tail_agreement=high_agreement,
        far_from_threshold_agreement=far_agreement,
        near_threshold_disagreement_count=near_disagreements,
        ibkr_frequencies=frequencies["ibkr"],
        eodhd_frequencies=frequencies["eodhd"],
    )


def _event_matches(
    ibkr_events: set[tuple[str, date, int]],
    eodhd_events: set[tuple[str, date, int]],
) -> tuple[int, int, int, int]:
    exact_events = ibkr_events.intersection(eodhd_events)
    remaining_ibkr = sorted(ibkr_events.difference(exact_events))
    remaining_eodhd = set(eodhd_events.difference(exact_events))
    within = 0
    for event in remaining_ibkr:
        symbol, session, checkpoint = event
        candidate = min(
            (
                other
                for other in remaining_eodhd
                if other[0] == symbol and other[1] == session and abs(other[2] - checkpoint) <= 2
            ),
            key=lambda other: (abs(other[2] - checkpoint), other),
            default=None,
        )
        if candidate is not None:
            within += 1
            remaining_eodhd.remove(candidate)
    unmatched_ibkr = len(remaining_ibkr) - within
    unmatched_eodhd = len(remaining_eodhd)
    return len(exact_events), within, unmatched_ibkr, unmatched_eodhd


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    return float(numerator) / float(denominator)


def _episode_metrics(
    ibkr: tuple[ProviderM1CObservation, ...],
    eodhd: tuple[ProviderM1CObservation, ...],
) -> EpisodeMetrics:
    quiet_ibkr = {row.key for row in ibkr if row.quiet_episode}
    quiet_eodhd = {row.key for row in eodhd if row.quiet_episode}
    high_ibkr = {row.key for row in ibkr if row.high_tail_episode}
    high_eodhd = {row.key for row in eodhd if row.high_tail_episode}
    quiet = _event_matches(quiet_ibkr, quiet_eodhd)
    high = _event_matches(high_ibkr, high_eodhd)
    return EpisodeMetrics(
        quiet_exact_checkpoint_matches=quiet[0],
        quiet_matches_within_one_checkpoint=quiet[1],
        quiet_ibkr_only=quiet[2],
        quiet_eodhd_only=quiet[3],
        quiet_frequency_ratio=_ratio(len(quiet_ibkr), len(quiet_eodhd)),
        high_exact_checkpoint_matches=high[0],
        high_matches_within_one_checkpoint=high[1],
        high_ibkr_only=high[2],
        high_eodhd_only=high[3],
        high_frequency_ratio=_ratio(len(high_ibkr), len(high_eodhd)),
    )


def _ratio_broadly_comparable(value: float | None) -> bool:
    return value is not None and 0.5 <= value <= 2.0


class M1CTransferMonitor:
    """Compare aligned provider rows without demanding equal OHLC values."""

    def __init__(
        self,
        *,
        robust_feature_scales: dict[str, float] | None = None,
        feature_coefficients: dict[str, float] | None = None,
    ) -> None:
        self.robust_feature_scales = dict(robust_feature_scales or {})
        self.feature_coefficients = dict(feature_coefficients or {})
        if any(value <= 0.0 for value in self.robust_feature_scales.values()):
            raise ValueError("robust feature scales must be positive")

    def evaluate(
        self,
        *,
        ibkr: tuple[ProviderM1CObservation, ...],
        eodhd: tuple[ProviderM1CObservation, ...],
        runtime_parity_passed: bool,
    ) -> TransferReport:
        if any(row.provider != "ibkr" for row in ibkr):
            raise ValueError("IBKR transfer input contains another provider")
        if any(row.provider != "eodhd" for row in eodhd):
            raise ValueError("EODHD transfer input contains another provider")
        ibkr_by_key = {row.key: row for row in ibkr}
        eodhd_by_key = {row.key: row for row in eodhd}
        duplicate_keys = len(ibkr_by_key) != len(ibkr) or len(eodhd_by_key) != len(eodhd)
        duplicate_ibkr_count = len(ibkr) - len(ibkr_by_key)
        duplicate_eodhd_count = len(eodhd) - len(eodhd_by_key)
        common_keys = sorted(set(ibkr_by_key).intersection(eodhd_by_key))
        missing_keys = set(ibkr_by_key).symmetric_difference(eodhd_by_key)
        bars: list[BarComparison] = []
        features: list[FeatureComparison] = []
        probabilities: list[ProbabilityComparison] = []
        invalid_sessions: set[date] = set()
        for key in common_keys:
            ibkr_row = ibkr_by_key[key]
            eodhd_row = eodhd_by_key[key]
            ibkr_bar = ibkr_row.bar
            eodhd_bar = eodhd_row.bar
            timestamps_aligned = ibkr_bar.start_utc.astimezone(
                UTC
            ) == eodhd_bar.start_utc.astimezone(UTC) and ibkr_bar.end_utc.astimezone(
                UTC
            ) == eodhd_bar.end_utc.astimezone(UTC)
            valid_bar_semantics = (
                ibkr_bar.complete
                and eodhd_bar.complete
                and ibkr_bar.end_utc - ibkr_bar.start_utc == timedelta(minutes=5)
                and eodhd_bar.end_utc - eodhd_bar.start_utc == timedelta(minutes=5)
                and timestamps_aligned
            )
            if not valid_bar_semantics:
                invalid_sessions.add(key[1])
            bars.append(
                BarComparison(
                    symbol=key[0],
                    session=key[1],
                    checkpoint=key[2],
                    ibkr_bar_identity=ibkr_bar.identity,
                    eodhd_bar_identity=eodhd_bar.identity,
                    timestamp_aligned=timestamps_aligned,
                    open_absolute_difference=abs(ibkr_bar.open - eodhd_bar.open),
                    high_absolute_difference=abs(ibkr_bar.high - eodhd_bar.high),
                    low_absolute_difference=abs(ibkr_bar.low - eodhd_bar.low),
                    close_absolute_difference=abs(ibkr_bar.close - eodhd_bar.close),
                )
            )
            feature_names = sorted(set(ibkr_row.features).union(eodhd_row.features))
            for feature in feature_names:
                if feature not in ibkr_row.features or feature not in eodhd_row.features:
                    invalid_sessions.add(key[1])
                    continue
                difference = abs(ibkr_row.features[feature] - eodhd_row.features[feature])
                scale = self.robust_feature_scales.get(feature, 1.0)
                scaled = difference / scale
                features.append(
                    FeatureComparison(
                        symbol=key[0],
                        session=key[1],
                        checkpoint=key[2],
                        feature=feature,
                        absolute_difference=difference,
                        robust_scaled_difference=scaled,
                        probability_contribution_difference=abs(
                            self.feature_coefficients.get(feature, 0.0) * scaled
                        ),
                    )
                )
            signed = ibkr_row.probability - eodhd_row.probability
            probabilities.append(
                ProbabilityComparison(
                    symbol=key[0],
                    session=key[1],
                    checkpoint=key[2],
                    ibkr_probability=ibkr_row.probability,
                    eodhd_probability=eodhd_row.probability,
                    absolute_difference=abs(signed),
                    signed_difference=signed,
                )
            )
        invalid_sessions.update(key[1] for key in missing_keys)
        all_sessions = {row.session for row in (*ibkr, *eodhd)}
        valid_sessions = all_sessions.difference(invalid_sessions)
        bar_semantics_passed = (
            not duplicate_keys and not missing_keys and not invalid_sessions and bool(common_keys)
        )
        bar_semantics_metrics = BarSemanticsMetrics(
            matched_bar_count=len(common_keys),
            ibkr_only_bar_count=len(set(ibkr_by_key).difference(eodhd_by_key)),
            eodhd_only_bar_count=len(set(eodhd_by_key).difference(ibkr_by_key)),
            duplicate_ibkr_key_count=duplicate_ibkr_count,
            duplicate_eodhd_key_count=duplicate_eodhd_count,
            timestamp_mismatch_count=sum(not row.timestamp_aligned for row in bars),
            duration_mismatch_count=sum(
                (
                    ibkr_by_key[key].bar.end_utc - ibkr_by_key[key].bar.start_utc
                    != timedelta(minutes=5)
                )
                or (
                    eodhd_by_key[key].bar.end_utc - eodhd_by_key[key].bar.start_utc
                    != timedelta(minutes=5)
                )
                for key in common_keys
            ),
            incomplete_bar_count=sum(
                not ibkr_by_key[key].bar.complete or not eodhd_by_key[key].bar.complete
                for key in common_keys
            ),
            non_equal_ohlc_count=sum(
                any(
                    difference > 0.0
                    for difference in (
                        row.open_absolute_difference,
                        row.high_absolute_difference,
                        row.low_absolute_difference,
                        row.close_absolute_difference,
                    )
                )
                for row in bars
            ),
            bar_interval_convention="[bar_start_utc,bar_end_utc)",
            auction_print_diagnostics=(
                "provider_auction_print_effects_remain_visible_in_ohlc_differences"
            ),
            corporate_action_policy=(
                "compare_provider_raw_intraday_semantics_and_flag_adjustment_mismatch"
            ),
        )
        probability_rows = tuple(probabilities)
        probability_metrics = _probability_metrics(probability_rows)
        tail_metrics = _tail_metrics(probability_rows)
        episode_metrics = _episode_metrics(ibkr, eodhd)
        by_stock = self._groups(probability_rows, lambda row: row.symbol)
        by_checkpoint = self._groups(
            probability_rows,
            lambda row: str(row.checkpoint),
        )
        by_time = self._groups(
            probability_rows,
            lambda row: f"checkpoint_{row.checkpoint:02d}",
        )
        decision = self._decision(
            valid_session_count=len(valid_sessions),
            bar_semantics_passed=bar_semantics_passed,
            runtime_parity_passed=runtime_parity_passed,
            probability=probability_metrics,
            tails=tail_metrics,
            episodes=episode_metrics,
            by_stock=by_stock,
            by_checkpoint=by_checkpoint,
        )
        return TransferReport(
            decision=decision,
            valid_session_count=len(valid_sessions),
            bar_semantics_passed=bar_semantics_passed,
            runtime_parity_passed=runtime_parity_passed,
            exact_vendor_bar_equality_required=False,
            bar_semantics_metrics=bar_semantics_metrics,
            probability_metrics=probability_metrics,
            tail_metrics=tail_metrics,
            episode_metrics=episode_metrics,
            bar_comparisons=tuple(bars),
            feature_comparisons=tuple(features),
            probability_comparisons=probability_rows,
            results_by_stock=by_stock,
            results_by_checkpoint=by_checkpoint,
            results_by_time_of_day=by_time,
            largest_feature_contributors=tuple(
                sorted(
                    features,
                    key=lambda row: (
                        -row.probability_contribution_difference,
                        row.symbol,
                        row.session,
                        row.checkpoint,
                        row.feature,
                    ),
                )[:25]
            ),
            claims_boundary=claims_boundary(),
            historical_decision=ORIGINAL_LOW_MOVEMENT_DECISION,
        )

    @staticmethod
    def _groups(
        rows: tuple[ProbabilityComparison, ...],
        key_function: Callable[[ProbabilityComparison], str],
    ) -> dict[str, ProbabilityMetrics]:
        grouped: dict[str, list[ProbabilityComparison]] = {}
        for row in rows:
            key = key_function(row)
            grouped.setdefault(str(key), []).append(row)
        return {
            key: _probability_metrics(group_rows) for key, group_rows in sorted(grouped.items())
        }

    @staticmethod
    def _decision(
        *,
        valid_session_count: int,
        bar_semantics_passed: bool,
        runtime_parity_passed: bool,
        probability: ProbabilityMetrics,
        tails: TailMetrics,
        episodes: EpisodeMetrics,
        by_stock: dict[str, ProbabilityMetrics],
        by_checkpoint: dict[str, ProbabilityMetrics],
    ) -> str:
        if not runtime_parity_passed:
            return "blocked_m1c_runtime_parity_failure"
        if not bar_semantics_passed:
            return "blocked_bar_semantics_failure"
        if valid_session_count < 20:
            return "blocked_insufficient_valid_sessions"
        ranking_strong = probability.spearman >= 0.90 and probability.pearson >= 0.80
        tail_agreement_high = (
            tails.bottom_10_agreement >= 0.85
            and tails.high_tail_agreement >= 0.80
            and tails.far_from_threshold_agreement >= 0.95
        )
        bottom_eodhd = tails.eodhd_frequencies["bottom_10"]
        bottom_ibkr = tails.ibkr_frequencies["bottom_10"]
        high_eodhd = tails.eodhd_frequencies["high_tail"]
        high_ibkr = tails.ibkr_frequencies["high_tail"]
        frequency_comparable = _ratio_broadly_comparable(
            _ratio(bottom_ibkr, bottom_eodhd)
        ) and _ratio_broadly_comparable(_ratio(high_ibkr, high_eodhd))
        episodes_comparable = _ratio_broadly_comparable(
            episodes.quiet_frequency_ratio
        ) and _ratio_broadly_comparable(episodes.high_frequency_ratio)
        if (
            ranking_strong
            and tail_agreement_high
            and frequency_comparable
            and episodes_comparable
            and probability.mean_absolute_difference <= 0.05
            and abs(probability.mean_signed_bias) <= 0.03
        ):
            return "ibkr_transfer_supported_without_recalibration"
        scale_shift = (
            ranking_strong
            and tails.far_from_threshold_agreement >= 0.80
            and (probability.distribution_shift_detected or not frequency_comparable)
        )
        if scale_shift:
            return "ibkr_ranking_supported_probability_scale_shifted"
        group_failures = any(
            metrics.count >= 3 and metrics.spearman < 0.70
            for metrics in (*by_stock.values(), *by_checkpoint.values())
        )
        if ranking_strong or group_failures:
            return "ibkr_transfer_mixed_stock_or_checkpoint_failures"
        return "ibkr_transfer_not_supported"


def create_ibkr_calibration_candidate(
    *,
    report: TransferReport,
    ibkr: tuple[ProviderM1CObservation, ...],
) -> IBKRCalibrationCandidate:
    """Freeze IBKR percentiles only; outcome and option inputs are impossible."""

    if report.decision != "ibkr_ranking_supported_probability_scale_shifted":
        raise ValueError("IBKR calibration candidate requires rank transfer with scale shift")
    probabilities = tuple(row.probability for row in ibkr)
    if not probabilities:
        raise ValueError("IBKR calibration requires probability observations")
    return IBKRCalibrationCandidate(
        candidate_id="M1C_IBKR_CALIBRATION_V1_CANDIDATE",
        source="ibkr_probability_distribution_only",
        thresholds={
            "bottom_5": _quantile(probabilities, 0.05),
            "bottom_10": _quantile(probabilities, 0.10),
            "bottom_20": _quantile(probabilities, 0.20),
            "high_tail_95": _quantile(probabilities, 0.95),
        },
        original_v0_thresholds_continue_in_parallel=True,
        outcome_fields_used=(),
        option_pnl_used=False,
        claims_boundary=claims_boundary(),
    )


__all__ = [
    "BarComparison",
    "BarSemanticsMetrics",
    "EpisodeMetrics",
    "FeatureComparison",
    "FROZEN_CHECKPOINTS",
    "IBKRCalibrationCandidate",
    "M1CTransferMonitor",
    "ProbabilityComparison",
    "ProbabilityMetrics",
    "ProviderM1CObservation",
    "TailMetrics",
    "TransferBar",
    "TransferReport",
    "create_ibkr_calibration_candidate",
]
