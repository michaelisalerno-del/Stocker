"""Logging-only mechanics for M1C Opening Market Transition V1."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stocker_prospective.signed_market_shock_v1 import MarketShockBarV1

OPENING_TRANSITION_CHECKPOINT_V1: Final[int] = 6
EXPECTED_OPENING_BAR_COUNT_V1: Final[int] = 6
OPENING_MARKET_PROXY_V1: Final[str] = "VTI"
FIVE_MINUTES_V1: Final[timedelta] = timedelta(minutes=5)

GapOpenAlignmentV1 = Literal[
    "ALIGNED_POSITIVE",
    "ALIGNED_NEGATIVE",
    "GAP_UP_OPENING_DOWN",
    "GAP_DOWN_OPENING_UP",
    "ZERO_OR_NEUTRAL",
    "UNKNOWN_INCOMPLETE",
]
OpeningMarketTransitionStateV1 = Literal[
    "NEGATIVE_SEVERE_OPENING_TRANSITION",
    "POSITIVE_SEVERE_OPENING_TRANSITION",
    "ELEVATED_OPENING_RANGE_NONDIRECTIONAL",
    "NORMAL_OPENING",
    "UNKNOWN_INCOMPLETE",
]
StockOpeningResponseClassV1 = Literal[
    "AMPLIFYING",
    "RESISTING",
    "NEUTRAL_EXACT",
    "NOT_SEVERE_OPENING_TRANSITION",
    "UNKNOWN_INCOMPLETE",
]
StockOpeningResistingSubtypeV1 = Literal[
    "RESISTING_BUT_STILL_ALIGNED",
    "ABSOLUTELY_OPPOSING",
]


class OpeningPreEntryWindowV1(BaseModel):
    """The complete same-session opening window visible at checkpoint 6."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    market_proxy_v1: str
    session: date
    previous_session_v1: date
    checkpoint_v1: Literal[6]
    session_open_timestamp_v1: datetime
    signal_timestamp_v1: datetime
    entry_timestamp_v1: datetime
    opening_bar_ordinals_v1: tuple[int, ...]
    expected_opening_bar_count_v1: int
    observed_opening_bar_count_v1: int
    final_complete_pre_entry_bar_start_v1: datetime | None
    entry_bar_ordinal_v1: int
    entry_bar_included_v1: Literal[False]
    market_session_open_v1: float | None
    market_prior_regular_session_close_v1: float | None
    market_last_complete_pre_entry_close_v1: float | None
    market_opening_return_v1: float | None
    market_opening_range_v1: float | None
    market_overnight_gap_v1: float | None
    market_total_transition_v1: float | None
    market_gap_open_alignment_v1: GapOpenAlignmentV1
    maximum_market_timestamp_v1: datetime | None
    complete_v1: bool
    missing_reasons_v1: tuple[str, ...]


class OpeningTransitionThresholdsV1(BaseModel):
    """Predictor-only 2024 thresholds for checkpoint-6 opening transitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_v1: Literal[6] = 6
    market_opening_return_q10_v1: float | None
    market_opening_return_q90_v1: float | None
    market_opening_range_q75_v1: float | None
    market_overnight_gap_q10_v1: float | None
    market_overnight_gap_q90_v1: float | None
    market_total_transition_q10_v1: float | None
    market_total_transition_q90_v1: float | None
    market_opening_return_support_v1: int
    market_opening_range_support_v1: int
    market_overnight_gap_support_v1: int
    market_total_transition_support_v1: int
    calibration_complete_v1: bool
    calibration_missing_reason_v1: str | None

    @model_validator(mode="after")
    def _valid_thresholds(self) -> OpeningTransitionThresholdsV1:
        supports = (
            self.market_opening_return_support_v1,
            self.market_opening_range_support_v1,
            self.market_overnight_gap_support_v1,
            self.market_total_transition_support_v1,
        )
        if any(value < 0 for value in supports):
            raise ValueError("opening threshold support cannot be negative")
        values = (
            self.market_opening_return_q10_v1,
            self.market_opening_return_q90_v1,
            self.market_opening_range_q75_v1,
            self.market_overnight_gap_q10_v1,
            self.market_overnight_gap_q90_v1,
            self.market_total_transition_q10_v1,
            self.market_total_transition_q90_v1,
        )
        if self.calibration_complete_v1:
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(
                    "complete opening calibration requires finite thresholds"
                )
            assert self.market_opening_return_q10_v1 is not None
            assert self.market_opening_return_q90_v1 is not None
            assert self.market_opening_range_q75_v1 is not None
            assert self.market_overnight_gap_q10_v1 is not None
            assert self.market_overnight_gap_q90_v1 is not None
            assert self.market_total_transition_q10_v1 is not None
            assert self.market_total_transition_q90_v1 is not None
            if (
                self.market_opening_return_q10_v1
                > self.market_opening_return_q90_v1
                or self.market_opening_range_q75_v1 < 0.0
                or self.market_overnight_gap_q10_v1
                > self.market_overnight_gap_q90_v1
                or self.market_total_transition_q10_v1
                > self.market_total_transition_q90_v1
            ):
                raise ValueError("opening calibration thresholds are invalid")
            if self.calibration_missing_reason_v1 is not None:
                raise ValueError(
                    "complete opening calibration cannot have a missing reason"
                )
        elif self.calibration_missing_reason_v1 is None:
            raise ValueError("incomplete opening calibration requires a missing reason")
        return self


class OpeningCalibrationPeriodV1(BaseModel):
    """Frozen predictor-only chronology for opening threshold calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: Literal["2024-01-01"]
    end: Literal["2024-12-31"]
    predictors_only: Literal[True]
    future_stock_outcomes_accessed_for_thresholds: Literal[False]
    option_outcomes_accessed_for_thresholds: Literal[False]


class OpeningCalibrationQuantilesV1(BaseModel):
    """The only quantiles permitted by the opening-transition V1 contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signed_return_lower: float
    signed_return_upper: float
    range: float
    method: Literal["numpy_linear"]

    @model_validator(mode="after")
    def _frozen_quantiles(self) -> OpeningCalibrationQuantilesV1:
        if (
            self.signed_return_lower,
            self.signed_return_upper,
            self.range,
        ) != (0.10, 0.90, 0.75):
            raise ValueError("opening calibration quantiles differ from V1")
        return self


class OpeningTransitionThresholdManifestV1(BaseModel):
    """Exact frozen threshold artifact consumed by logging-only code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m1c-opening-market-transition-thresholds-v1"]
    market_proxy_v1: Literal["VTI"]
    checkpoint_v1: Literal[6]
    expected_opening_bar_count_v1: Literal[6]
    calibration_period: OpeningCalibrationPeriodV1
    quantiles: OpeningCalibrationQuantilesV1
    minimum_predictor_support_v1: int
    pooling_fallback_used: Literal[False]
    configuration_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    thresholds: OpeningTransitionThresholdsV1
    stock_relative_response_quintile_boundaries_v1: tuple[
        float, float, float, float
    ]
    stock_relative_response_quintile_support_v1: int

    @model_validator(mode="after")
    def _frozen_identity(self) -> OpeningTransitionThresholdManifestV1:
        values = self.stock_relative_response_quintile_boundaries_v1
        if any(not math.isfinite(value) for value in values):
            raise ValueError("opening response quintiles must be finite")
        if tuple(sorted(values)) != values:
            raise ValueError("opening response quintiles must be ordered")
        if self.minimum_predictor_support_v1 < 1:
            raise ValueError("minimum predictor support must be positive")
        if self.stock_relative_response_quintile_support_v1 < 0:
            raise ValueError("response quintile support cannot be negative")
        return self


class OpeningMarketTransitionStateResultV1(BaseModel):
    """One deterministic opening-transition state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opening_market_transition_state_v1: OpeningMarketTransitionStateV1
    opening_transition_sign_v1: Literal[-1, 1] | None
    opening_transition_event_id_v1: str | None
    complete_v1: bool
    missing_reasons_v1: tuple[str, ...]


class StockOpeningResponseResultV1(BaseModel):
    """One stock's causal response over the fixed opening window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stock_opening_return_v1: float | None
    stock_opening_range_v1: float | None
    stock_opening_alignment_v1: float | None
    stock_relative_opening_response_v1: float | None
    stock_opening_response_class_v1: StockOpeningResponseClassV1
    resisting_subtype_v1: StockOpeningResistingSubtypeV1 | None
    expected_opening_bar_count_v1: int
    observed_opening_bar_count_v1: int
    maximum_stock_timestamp_v1: datetime | None
    complete_v1: bool
    missing_reasons_v1: tuple[str, ...]


def _valid_bar(
    bar: MarketShockBarV1,
    *,
    expected_start: datetime,
    signal_timestamp: datetime,
) -> bool:
    prices = (bar.open, bar.high, bar.low, bar.close)
    return (
        bar.finalised
        and all(math.isfinite(value) and value > 0.0 for value in prices)
        and bar.high >= max(bar.open, bar.close, bar.low)
        and bar.low <= min(bar.open, bar.close, bar.high)
        and bar.bar_start_timestamp == expected_start
        and bar.bar_complete_timestamp == expected_start + FIVE_MINUTES_V1
        and bar.bar_complete_timestamp <= signal_timestamp
    )


def _collect_opening_bars(
    *,
    symbol: str,
    session: date,
    session_open_timestamp: datetime,
    signal_timestamp: datetime,
    completed_bars: tuple[MarketShockBarV1, ...],
    reason_scope: Literal["market", "stock"],
) -> tuple[
    tuple[MarketShockBarV1, ...],
    int,
    datetime | None,
    tuple[str, ...],
]:
    required = tuple(range(EXPECTED_OPENING_BAR_COUNT_V1))
    relevant = [
        bar
        for bar in completed_bars
        if bar.symbol == symbol
        and bar.session == session
        and bar.bar_ordinal in required
    ]
    by_ordinal: dict[int, MarketShockBarV1] = {}
    reasons: list[str] = []
    for bar in relevant:
        if bar.bar_ordinal in by_ordinal:
            reasons.append(f"duplicate_{reason_scope}_bar:{bar.bar_ordinal}")
            continue
        by_ordinal[bar.bar_ordinal] = bar
    for ordinal in required:
        if ordinal not in by_ordinal:
            reasons.append(f"missing_{reason_scope}_bar:{ordinal}")
    for ordinal, bar in sorted(by_ordinal.items()):
        expected_start = session_open_timestamp + FIVE_MINUTES_V1 * ordinal
        if not _valid_bar(
            bar,
            expected_start=expected_start,
            signal_timestamp=signal_timestamp,
        ):
            reasons.append(f"invalid_{reason_scope}_bar:{ordinal}")
    ordered = tuple(
        by_ordinal[ordinal]
        for ordinal in required
        if ordinal in by_ordinal
        and _valid_bar(
            by_ordinal[ordinal],
            expected_start=session_open_timestamp + FIVE_MINUTES_V1 * ordinal,
            signal_timestamp=signal_timestamp,
        )
    )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if (
            current.bar_ordinal != previous.bar_ordinal + 1
            or current.bar_start_timestamp != previous.bar_complete_timestamp
        ):
            reasons.append(
                f"non_contiguous_{reason_scope}_timestamps:"
                f"{previous.bar_ordinal}-{current.bar_ordinal}"
            )
    final = by_ordinal.get(EXPECTED_OPENING_BAR_COUNT_V1 - 1)
    if (
        final is None
        or not _valid_bar(
            final,
            expected_start=session_open_timestamp
            + FIVE_MINUTES_V1 * (EXPECTED_OPENING_BAR_COUNT_V1 - 1),
            signal_timestamp=signal_timestamp,
        )
        or final.bar_complete_timestamp != signal_timestamp
    ):
        reasons.append(f"{reason_scope}_signal_timestamp_mismatch")
    maximum = max((bar.bar_complete_timestamp for bar in ordered), default=None)
    return ordered, len(relevant), maximum, tuple(dict.fromkeys(reasons))


def _gap_open_alignment(
    *,
    overnight_gap: float,
    opening_return: float,
) -> GapOpenAlignmentV1:
    if overnight_gap > 0.0 and opening_return > 0.0:
        return "ALIGNED_POSITIVE"
    if overnight_gap < 0.0 and opening_return < 0.0:
        return "ALIGNED_NEGATIVE"
    if overnight_gap > 0.0 and opening_return < 0.0:
        return "GAP_UP_OPENING_DOWN"
    if overnight_gap < 0.0 and opening_return > 0.0:
        return "GAP_DOWN_OPENING_UP"
    return "ZERO_OR_NEUTRAL"


def calculate_opening_preentry_window_v1(
    *,
    market_proxy: str,
    session: date,
    previous_session: date,
    session_open_timestamp: datetime,
    signal_timestamp: datetime,
    entry_timestamp: datetime,
    completed_bars: tuple[MarketShockBarV1, ...],
    prior_regular_session_close: float | None,
) -> OpeningPreEntryWindowV1:
    """Measure the six complete opening bars available before checkpoint-6 entry."""

    if market_proxy != OPENING_MARKET_PROXY_V1:
        raise ValueError("market proxy differs from frozen VTI")
    for name, value in (
        ("session_open_timestamp", session_open_timestamp),
        ("signal_timestamp", signal_timestamp),
        ("entry_timestamp", entry_timestamp),
    ):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")

    reasons: list[str] = []
    expected_signal = (
        session_open_timestamp + FIVE_MINUTES_V1 * EXPECTED_OPENING_BAR_COUNT_V1
    )
    if signal_timestamp != expected_signal:
        reasons.append("checkpoint_6_signal_timestamp_mismatch")
    if entry_timestamp != signal_timestamp:
        reasons.append("checkpoint_6_entry_timestamp_mismatch")
    if previous_session >= session:
        reasons.append("prior_regular_session_invalid")
    prior_close = (
        None
        if prior_regular_session_close is None
        else float(prior_regular_session_close)
    )
    if prior_close is None or not math.isfinite(prior_close) or prior_close <= 0.0:
        reasons.append("prior_regular_session_close_invalid")

    ordered, observed, maximum, bar_reasons = _collect_opening_bars(
        symbol=market_proxy,
        session=session,
        session_open_timestamp=session_open_timestamp,
        signal_timestamp=signal_timestamp,
        completed_bars=completed_bars,
        reason_scope="market",
    )
    reasons.extend(bar_reasons)
    unique = tuple(dict.fromkeys(reasons))
    complete = not unique and len(ordered) == EXPECTED_OPENING_BAR_COUNT_V1

    session_open: float | None = None
    last_close: float | None = None
    opening_return: float | None = None
    opening_range: float | None = None
    overnight_gap: float | None = None
    total_transition: float | None = None
    alignment: GapOpenAlignmentV1 = "UNKNOWN_INCOMPLETE"
    if complete:
        assert prior_close is not None
        session_open = ordered[0].open
        last_close = ordered[-1].close
        opening_return = math.log(last_close / session_open)
        opening_range = math.log(
            max(bar.high for bar in ordered) / min(bar.low for bar in ordered)
        )
        overnight_gap = math.log(session_open / prior_close)
        total_transition = math.log(last_close / prior_close)
        if not math.isclose(
            overnight_gap + opening_return,
            total_transition,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("opening transition return identity failed")
        alignment = _gap_open_alignment(
            overnight_gap=overnight_gap,
            opening_return=opening_return,
        )

    return OpeningPreEntryWindowV1(
        market_proxy_v1=market_proxy,
        session=session,
        previous_session_v1=previous_session,
        checkpoint_v1=6,
        session_open_timestamp_v1=session_open_timestamp,
        signal_timestamp_v1=signal_timestamp,
        entry_timestamp_v1=entry_timestamp,
        opening_bar_ordinals_v1=tuple(range(EXPECTED_OPENING_BAR_COUNT_V1)),
        expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
        observed_opening_bar_count_v1=observed,
        final_complete_pre_entry_bar_start_v1=(
            ordered[-1].bar_start_timestamp
            if len(ordered) == EXPECTED_OPENING_BAR_COUNT_V1
            else None
        ),
        entry_bar_ordinal_v1=EXPECTED_OPENING_BAR_COUNT_V1,
        entry_bar_included_v1=False,
        market_session_open_v1=session_open,
        market_prior_regular_session_close_v1=prior_close if complete else None,
        market_last_complete_pre_entry_close_v1=last_close,
        market_opening_return_v1=opening_return,
        market_opening_range_v1=opening_range,
        market_overnight_gap_v1=overnight_gap,
        market_total_transition_v1=total_transition,
        market_gap_open_alignment_v1=alignment,
        maximum_market_timestamp_v1=maximum,
        complete_v1=complete,
        missing_reasons_v1=unique,
    )


def _opening_event_id(
    *,
    session: date,
    market_proxy: str,
    transition_sign: Literal[-1, 1],
) -> str:
    raw = "|".join(
        (
            "M1C_OPENING_MARKET_TRANSITION_V1",
            session.isoformat(),
            str(OPENING_TRANSITION_CHECKPOINT_V1),
            market_proxy,
            str(int(transition_sign)),
        )
    )
    return f"opening-transition-v1-{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def classify_opening_market_transition_v1(
    *,
    window: OpeningPreEntryWindowV1,
    thresholds: OpeningTransitionThresholdsV1 | None,
) -> OpeningMarketTransitionStateResultV1:
    """Apply the frozen severe-opening state definition."""

    reasons = list(window.missing_reasons_v1)
    if thresholds is None:
        reasons.append("opening_calibration_thresholds_unavailable")
    elif not thresholds.calibration_complete_v1:
        reasons.append(
            thresholds.calibration_missing_reason_v1
            or "opening_calibration_thresholds_incomplete"
        )
    values = (
        window.market_opening_return_v1,
        window.market_opening_range_v1,
        window.market_overnight_gap_v1,
        window.market_total_transition_v1,
    )
    if any(value is None or not math.isfinite(value) for value in values):
        reasons.append("opening_market_measurement_incomplete")
    if reasons:
        return OpeningMarketTransitionStateResultV1(
            opening_market_transition_state_v1="UNKNOWN_INCOMPLETE",
            opening_transition_sign_v1=None,
            opening_transition_event_id_v1=None,
            complete_v1=False,
            missing_reasons_v1=tuple(dict.fromkeys(reasons)),
        )

    assert thresholds is not None
    assert window.market_opening_return_v1 is not None
    assert window.market_opening_range_v1 is not None
    assert thresholds.market_opening_return_q10_v1 is not None
    assert thresholds.market_opening_return_q90_v1 is not None
    assert thresholds.market_opening_range_q75_v1 is not None
    negative = (
        window.market_opening_return_v1
        <= thresholds.market_opening_return_q10_v1
        and window.market_opening_range_v1
        >= thresholds.market_opening_range_q75_v1
    )
    positive = (
        window.market_opening_return_v1
        >= thresholds.market_opening_return_q90_v1
        and window.market_opening_range_v1
        >= thresholds.market_opening_range_q75_v1
    )
    state: OpeningMarketTransitionStateV1
    sign: Literal[-1, 1] | None = None
    if negative:
        state = "NEGATIVE_SEVERE_OPENING_TRANSITION"
        sign = -1
    elif positive:
        state = "POSITIVE_SEVERE_OPENING_TRANSITION"
        sign = 1
    elif (
        window.market_opening_range_v1
        >= thresholds.market_opening_range_q75_v1
    ):
        state = "ELEVATED_OPENING_RANGE_NONDIRECTIONAL"
    else:
        state = "NORMAL_OPENING"
    return OpeningMarketTransitionStateResultV1(
        opening_market_transition_state_v1=state,
        opening_transition_sign_v1=sign,
        opening_transition_event_id_v1=(
            None
            if sign is None
            else _opening_event_id(
                session=window.session,
                market_proxy=window.market_proxy_v1,
                transition_sign=sign,
            )
        ),
        complete_v1=True,
        missing_reasons_v1=(),
    )


def calculate_stock_opening_response_v1(
    *,
    symbol: str,
    session: date,
    session_open_timestamp: datetime,
    signal_timestamp: datetime,
    completed_stock_bars: tuple[MarketShockBarV1, ...],
    market_opening_return_v1: float | None,
    opening_transition_state_v1: OpeningMarketTransitionStateResultV1,
    threshold_15m: float | None,
) -> StockOpeningResponseResultV1:
    """Measure a stock locally over the same six-bar opening window."""

    ordered, observed, maximum, reasons = _collect_opening_bars(
        symbol=symbol,
        session=session,
        session_open_timestamp=session_open_timestamp,
        signal_timestamp=signal_timestamp,
        completed_bars=completed_stock_bars,
        reason_scope="stock",
    )
    if reasons or len(ordered) != EXPECTED_OPENING_BAR_COUNT_V1:
        return StockOpeningResponseResultV1(
            stock_opening_return_v1=None,
            stock_opening_range_v1=None,
            stock_opening_alignment_v1=None,
            stock_relative_opening_response_v1=None,
            stock_opening_response_class_v1="UNKNOWN_INCOMPLETE",
            resisting_subtype_v1=None,
            expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
            observed_opening_bar_count_v1=observed,
            maximum_stock_timestamp_v1=maximum,
            complete_v1=False,
            missing_reasons_v1=reasons,
        )
    stock_return = math.log(ordered[-1].close / ordered[0].open)
    stock_range = math.log(
        max(bar.high for bar in ordered) / min(bar.low for bar in ordered)
    )
    state = opening_transition_state_v1.opening_market_transition_state_v1
    if state == "UNKNOWN_INCOMPLETE":
        return StockOpeningResponseResultV1(
            stock_opening_return_v1=stock_return,
            stock_opening_range_v1=stock_range,
            stock_opening_alignment_v1=None,
            stock_relative_opening_response_v1=None,
            stock_opening_response_class_v1="UNKNOWN_INCOMPLETE",
            resisting_subtype_v1=None,
            expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
            observed_opening_bar_count_v1=observed,
            maximum_stock_timestamp_v1=maximum,
            complete_v1=False,
            missing_reasons_v1=(
                opening_transition_state_v1.missing_reasons_v1
                or ("opening_transition_state_incomplete",)
            ),
        )
    if state not in {
        "NEGATIVE_SEVERE_OPENING_TRANSITION",
        "POSITIVE_SEVERE_OPENING_TRANSITION",
    }:
        return StockOpeningResponseResultV1(
            stock_opening_return_v1=stock_return,
            stock_opening_range_v1=stock_range,
            stock_opening_alignment_v1=None,
            stock_relative_opening_response_v1=None,
            stock_opening_response_class_v1="NOT_SEVERE_OPENING_TRANSITION",
            resisting_subtype_v1=None,
            expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
            observed_opening_bar_count_v1=observed,
            maximum_stock_timestamp_v1=maximum,
            complete_v1=True,
            missing_reasons_v1=(),
        )

    scale = None if threshold_15m is None else float(threshold_15m)
    if scale is None or not math.isfinite(scale) or scale <= 0.0:
        return StockOpeningResponseResultV1(
            stock_opening_return_v1=stock_return,
            stock_opening_range_v1=stock_range,
            stock_opening_alignment_v1=None,
            stock_relative_opening_response_v1=None,
            stock_opening_response_class_v1="UNKNOWN_INCOMPLETE",
            resisting_subtype_v1=None,
            expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
            observed_opening_bar_count_v1=observed,
            maximum_stock_timestamp_v1=maximum,
            complete_v1=False,
            missing_reasons_v1=("threshold_15m_invalid",),
        )
    market_return = (
        None
        if market_opening_return_v1 is None
        else float(market_opening_return_v1)
    )
    if market_return is None or not math.isfinite(market_return):
        return StockOpeningResponseResultV1(
            stock_opening_return_v1=stock_return,
            stock_opening_range_v1=stock_range,
            stock_opening_alignment_v1=None,
            stock_relative_opening_response_v1=None,
            stock_opening_response_class_v1="UNKNOWN_INCOMPLETE",
            resisting_subtype_v1=None,
            expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
            observed_opening_bar_count_v1=observed,
            maximum_stock_timestamp_v1=maximum,
            complete_v1=False,
            missing_reasons_v1=("market_opening_return_invalid",),
        )
    sign = opening_transition_state_v1.opening_transition_sign_v1
    if sign not in {-1, 1}:
        raise ValueError("severe opening transition lacks its deterministic sign")
    alignment = sign * stock_return / scale
    relative = sign * (stock_return - market_return) / scale
    response_class: StockOpeningResponseClassV1
    subtype: StockOpeningResistingSubtypeV1 | None = None
    if relative > 0.0:
        response_class = "AMPLIFYING"
    elif relative < 0.0:
        response_class = "RESISTING"
        subtype = (
            "RESISTING_BUT_STILL_ALIGNED"
            if alignment > 0.0
            else "ABSOLUTELY_OPPOSING"
        )
    else:
        response_class = "NEUTRAL_EXACT"
    return StockOpeningResponseResultV1(
        stock_opening_return_v1=stock_return,
        stock_opening_range_v1=stock_range,
        stock_opening_alignment_v1=alignment,
        stock_relative_opening_response_v1=relative,
        stock_opening_response_class_v1=response_class,
        resisting_subtype_v1=subtype,
        expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
        observed_opening_bar_count_v1=observed,
        maximum_stock_timestamp_v1=maximum,
        complete_v1=True,
        missing_reasons_v1=(),
    )


def load_opening_transition_threshold_manifest_v1(
    path: str | Path,
) -> OpeningTransitionThresholdManifestV1:
    """Load one explicitly configured manifest; never discover newest output."""

    source = Path(path)
    if not source.is_file():
        raise ValueError("opening-transition threshold manifest is absent")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return OpeningTransitionThresholdManifestV1.model_validate(payload)


__all__ = [
    "EXPECTED_OPENING_BAR_COUNT_V1",
    "OPENING_MARKET_PROXY_V1",
    "OPENING_TRANSITION_CHECKPOINT_V1",
    "GapOpenAlignmentV1",
    "OpeningCalibrationPeriodV1",
    "OpeningCalibrationQuantilesV1",
    "OpeningMarketTransitionStateResultV1",
    "OpeningMarketTransitionStateV1",
    "OpeningPreEntryWindowV1",
    "OpeningTransitionThresholdManifestV1",
    "OpeningTransitionThresholdsV1",
    "StockOpeningResponseClassV1",
    "StockOpeningResponseResultV1",
    "StockOpeningResistingSubtypeV1",
    "calculate_opening_preentry_window_v1",
    "calculate_stock_opening_response_v1",
    "classify_opening_market_transition_v1",
    "load_opening_transition_threshold_manifest_v1",
]
