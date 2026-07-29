"""Logging-only mechanics for M1C Signed Market Shock Transition V1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FROZEN_SHOCK_CHECKPOINTS_V1: Final[tuple[int, ...]] = tuple(range(6, 35, 2))
FIVE_MINUTES_V1: Final[timedelta] = timedelta(minutes=5)
MARKET_SHOCK_PROXY_V1: Final[str] = "VTI"
MarketShockStateV1 = Literal[
    "NEGATIVE_SHOCK_ONSET",
    "POSITIVE_SHOCK_ONSET",
    "ONGOING_NEGATIVE_SHOCK",
    "ONGOING_POSITIVE_SHOCK",
    "ELEVATED_RANGE_NONDIRECTIONAL",
    "NORMAL_OTHER",
    "UNKNOWN_INCOMPLETE",
]
ShockResponseClassV1 = Literal[
    "AMPLIFYING",
    "RESISTING",
    "NEUTRAL_EXACT",
    "NOT_SHOCK_ONSET",
    "UNKNOWN_INCOMPLETE",
]
ResistingSubtypeV1 = Literal[
    "RESISTING_BUT_STILL_WITH_SHOCK",
    "ABSOLUTELY_OPPOSING_SHOCK",
]
MaterialEndpointStateV1 = Literal[
    "MATERIAL_UP",
    "MATERIAL_DOWN",
    "NO_MATERIAL_MOVE",
]


class MarketShockBarV1(BaseModel):
    """One finalised five-minute OHLC bar available at its completion time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    bar_ordinal: int
    bar_start_timestamp: datetime
    bar_complete_timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    finalised: bool


class PreentryMarketWindowsV1(BaseModel):
    """The two fixed adjacent market windows and their explicit completeness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    market_proxy_v1: str
    session: date
    checkpoint: int
    signal_timestamp: datetime
    w0_bar_ordinals_v1: tuple[int, int, int]
    w1_bar_ordinals_v1: tuple[int, int, int]
    market_return_w0_v1: float | None
    market_range_w0_v1: float | None
    market_return_w1_v1: float | None
    market_range_w1_v1: float | None
    maximum_market_timestamp_v1: datetime | None
    complete_v1: bool
    missing_reasons_v1: tuple[str, ...]


class CheckpointShockThresholdsV1(BaseModel):
    """One checkpoint's predictor-only 2024 market calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: int
    market_return_w0_q10_v1: float | None
    market_return_w0_q90_v1: float | None
    market_range_w0_q75_v1: float | None
    market_return_w1_q10_v1: float | None
    market_return_w1_q90_v1: float | None
    market_range_w1_q75_v1: float | None
    market_return_w0_support_v1: int
    market_range_w0_support_v1: int
    market_return_w1_support_v1: int
    market_range_w1_support_v1: int
    calibration_complete_v1: bool
    calibration_missing_reason_v1: str | None

    @model_validator(mode="after")
    def _valid_thresholds(self) -> CheckpointShockThresholdsV1:
        if self.checkpoint not in FROZEN_SHOCK_CHECKPOINTS_V1:
            raise ValueError("shock threshold checkpoint differs from frozen M1C grid")
        supports = (
            self.market_return_w0_support_v1,
            self.market_range_w0_support_v1,
            self.market_return_w1_support_v1,
            self.market_range_w1_support_v1,
        )
        if any(value < 0 for value in supports):
            raise ValueError("shock threshold support cannot be negative")
        values = (
            self.market_return_w0_q10_v1,
            self.market_return_w0_q90_v1,
            self.market_range_w0_q75_v1,
            self.market_return_w1_q10_v1,
            self.market_return_w1_q90_v1,
            self.market_range_w1_q75_v1,
        )
        if self.calibration_complete_v1:
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("complete shock calibration requires finite thresholds")
            assert self.market_return_w0_q10_v1 is not None
            assert self.market_return_w0_q90_v1 is not None
            assert self.market_return_w1_q10_v1 is not None
            assert self.market_return_w1_q90_v1 is not None
            assert self.market_range_w0_q75_v1 is not None
            assert self.market_range_w1_q75_v1 is not None
            if (
                self.market_return_w0_q10_v1 > self.market_return_w0_q90_v1
                or self.market_return_w1_q10_v1 > self.market_return_w1_q90_v1
                or self.market_range_w0_q75_v1 < 0.0
                or self.market_range_w1_q75_v1 < 0.0
            ):
                raise ValueError("shock calibration thresholds are invalid")
            if self.calibration_missing_reason_v1 is not None:
                raise ValueError("complete shock calibration cannot have a missing reason")
        elif self.calibration_missing_reason_v1 is None:
            raise ValueError("incomplete shock calibration requires a missing reason")
        return self


class ShockCalibrationPeriodV1(BaseModel):
    """Frozen predictor-only chronology for the V1 threshold calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: Literal["2024-01-01"]
    end: Literal["2024-12-31"]
    predictors_only: Literal[True]
    future_stock_outcomes_accessed_for_thresholds: Literal[False]
    option_outcomes_accessed_for_thresholds: Literal[False]


class ShockCalibrationQuantilesV1(BaseModel):
    """The only quantiles permitted by the signed-shock V1 contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signed_return_lower: float
    signed_return_upper: float
    range: float
    method: Literal["numpy_linear"]

    @model_validator(mode="after")
    def _frozen_quantiles(self) -> ShockCalibrationQuantilesV1:
        if (
            self.signed_return_lower,
            self.signed_return_upper,
            self.range,
        ) != (0.10, 0.90, 0.75):
            raise ValueError("shock calibration quantiles differ from V1")
        return self


class SignedMarketShockThresholdManifestV1(BaseModel):
    """Exact predictor-only threshold artifact consumed by logging-only code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m1c-signed-market-shock-thresholds-v1"]
    market_proxy_v1: Literal["VTI"]
    calibration_period: ShockCalibrationPeriodV1
    quantiles: ShockCalibrationQuantilesV1
    minimum_predictor_support_v1: int
    pooling_fallback_used: bool
    configuration_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    checkpoints: tuple[CheckpointShockThresholdsV1, ...]

    @model_validator(mode="after")
    def _frozen_identity(self) -> SignedMarketShockThresholdManifestV1:
        observed = tuple(item.checkpoint for item in self.checkpoints)
        if observed != FROZEN_SHOCK_CHECKPOINTS_V1:
            raise ValueError("shock threshold manifest differs from frozen checkpoint grid")
        if self.pooling_fallback_used:
            raise ValueError("shock threshold manifest used prohibited checkpoint pooling")
        return self

    def threshold_for_checkpoint(
        self,
        checkpoint: int,
    ) -> CheckpointShockThresholdsV1 | None:
        matches = [item for item in self.checkpoints if item.checkpoint == checkpoint]
        return matches[0] if len(matches) == 1 else None


class MarketShockStateResultV1(BaseModel):
    """One logging-only market state at an M1C checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    market_shock_state_v1: MarketShockStateV1
    market_shock_event_id_v1: str | None
    shock_sign_v1: Literal[-1, 1] | None
    complete_v1: bool
    missing_reasons_v1: tuple[str, ...]


class StockShockResponseResultV1(BaseModel):
    """One stock's causal response to an already-classified market onset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stock_return_w0_v1: float | None
    stock_absolute_alignment_v1: float | None
    shock_relative_response_v1: float | None
    shock_response_class_v1: ShockResponseClassV1
    resisting_subtype_v1: ResistingSubtypeV1 | None
    maximum_stock_timestamp_v1: datetime | None
    complete_v1: bool
    missing_reasons_v1: tuple[str, ...]


def _valid_bar(bar: MarketShockBarV1, *, signal_timestamp: datetime) -> bool:
    prices = (bar.open, bar.high, bar.low, bar.close)
    return (
        bar.finalised
        and all(math.isfinite(value) and value > 0.0 for value in prices)
        and bar.high >= max(bar.open, bar.close, bar.low)
        and bar.low <= min(bar.open, bar.close, bar.high)
        and bar.bar_complete_timestamp - bar.bar_start_timestamp == FIVE_MINUTES_V1
        and bar.bar_complete_timestamp <= signal_timestamp
    )


def _window_values(
    by_ordinal: dict[int, MarketShockBarV1],
    *,
    reference_ordinal: int,
    window_ordinals: tuple[int, int, int],
) -> tuple[float | None, float | None]:
    reference = by_ordinal.get(reference_ordinal)
    window = [by_ordinal.get(ordinal) for ordinal in window_ordinals]
    if reference is None or any(bar is None for bar in window):
        return None, None
    present = [bar for bar in window if bar is not None]
    terminal = present[-1]
    return (
        math.log(terminal.close / reference.close),
        math.log(max(bar.high for bar in present) / min(bar.low for bar in present)),
    )


def calculate_preentry_windows_v1(
    *,
    market_proxy: str,
    session: date,
    checkpoint: int,
    signal_timestamp: datetime,
    completed_bars: tuple[MarketShockBarV1, ...],
) -> PreentryMarketWindowsV1:
    """Calculate only W0 and W1, ending before the checkpoint entry bar."""

    value = int(checkpoint)
    if value not in FROZEN_SHOCK_CHECKPOINTS_V1:
        raise ValueError(f"checkpoint outside frozen M1C grid: {value}")
    if market_proxy != MARKET_SHOCK_PROXY_V1:
        raise ValueError("market proxy differs from frozen VTI")
    if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() is None:
        raise ValueError("signal timestamp must be timezone-aware")

    w0_ordinals = (value - 3, value - 2, value - 1)
    w1_ordinals = (value - 6, value - 5, value - 4)
    required = set(range(max(0, value - 7), value))
    relevant = [
        bar
        for bar in completed_bars
        if bar.symbol == market_proxy
        and bar.session == session
        and bar.bar_ordinal in required
    ]
    by_ordinal: dict[int, MarketShockBarV1] = {}
    reasons: list[str] = []
    for bar in relevant:
        if bar.bar_ordinal in by_ordinal:
            reasons.append(f"duplicate_market_bar:{bar.bar_ordinal}")
            continue
        by_ordinal[bar.bar_ordinal] = bar

    expected = set(range(value - 7, value))
    if value == FROZEN_SHOCK_CHECKPOINTS_V1[0]:
        expected.discard(-1)
        reasons.append("w1_reference_would_cross_session")
    for ordinal in sorted(expected):
        if ordinal not in by_ordinal:
            reasons.append(f"missing_market_bar:{ordinal}")
    for ordinal, bar in sorted(by_ordinal.items()):
        if not _valid_bar(bar, signal_timestamp=signal_timestamp):
            reasons.append(f"invalid_market_bar:{ordinal}")

    ordered = [
        by_ordinal[ordinal]
        for ordinal in sorted(expected)
        if ordinal in by_ordinal
        and _valid_bar(by_ordinal[ordinal], signal_timestamp=signal_timestamp)
    ]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if (
            current.bar_ordinal != previous.bar_ordinal + 1
            or current.bar_start_timestamp != previous.bar_complete_timestamp
        ):
            reasons.append(
                "non_contiguous_market_timestamps:"
                f"{previous.bar_ordinal}-{current.bar_ordinal}"
            )
    final = by_ordinal.get(value - 1)
    if (
        final is None
        or not _valid_bar(final, signal_timestamp=signal_timestamp)
        or final.bar_complete_timestamp != signal_timestamp
    ):
        reasons.append("market_signal_timestamp_mismatch")

    invalid_ordinals = {
        int(reason.rsplit(":", maxsplit=1)[-1])
        for reason in reasons
        if reason.startswith("invalid_market_bar:")
        or reason.startswith("missing_market_bar:")
        or reason.startswith("duplicate_market_bar:")
    }
    w0_required = {value - 4, *w0_ordinals}
    w1_required = {value - 7, *w1_ordinals}
    w0_return: float | None = None
    w0_range: float | None = None
    if not invalid_ordinals.intersection(w0_required):
        w0_return, w0_range = _window_values(
            by_ordinal,
            reference_ordinal=value - 4,
            window_ordinals=w0_ordinals,
        )
    w1_return: float | None = None
    w1_range: float | None = None
    if value > FROZEN_SHOCK_CHECKPOINTS_V1[0] and not invalid_ordinals.intersection(
        w1_required
    ):
        w1_return, w1_range = _window_values(
            by_ordinal,
            reference_ordinal=value - 7,
            window_ordinals=w1_ordinals,
        )

    unique_reasons = tuple(dict.fromkeys(reasons))
    complete = (
        not unique_reasons
        and w0_return is not None
        and w0_range is not None
        and w1_return is not None
        and w1_range is not None
    )
    maximum_timestamp = (
        max((bar.bar_complete_timestamp for bar in ordered), default=None)
        if ordered
        else None
    )
    return PreentryMarketWindowsV1(
        market_proxy_v1=market_proxy,
        session=session,
        checkpoint=value,
        signal_timestamp=signal_timestamp,
        w0_bar_ordinals_v1=w0_ordinals,
        w1_bar_ordinals_v1=w1_ordinals,
        market_return_w0_v1=w0_return,
        market_range_w0_v1=w0_range,
        market_return_w1_v1=w1_return,
        market_range_w1_v1=w1_range,
        maximum_market_timestamp_v1=maximum_timestamp,
        complete_v1=complete,
        missing_reasons_v1=unique_reasons,
    )


def _shock_event_id(
    *,
    session: date,
    checkpoint: int,
    market_proxy: str,
    shock_sign: Literal[-1, 1],
) -> str:
    raw = "|".join(
        (
            "M1C_SIGNED_MARKET_SHOCK_V1",
            session.isoformat(),
            str(int(checkpoint)),
            market_proxy,
            str(int(shock_sign)),
        )
    )
    return f"market-shock-v1-{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def classify_market_shock_state_v1(
    *,
    windows: PreentryMarketWindowsV1,
    thresholds: CheckpointShockThresholdsV1 | None,
) -> MarketShockStateResultV1:
    """Apply the frozen signed-shock state machine without fitted coefficients."""

    reasons = list(windows.missing_reasons_v1)
    if thresholds is None:
        reasons.append("shock_calibration_thresholds_unavailable")
    elif thresholds.checkpoint != windows.checkpoint:
        reasons.append("shock_calibration_checkpoint_mismatch")
    elif not thresholds.calibration_complete_v1:
        reasons.append(
            thresholds.calibration_missing_reason_v1
            or "shock_calibration_thresholds_incomplete"
        )
    required_values = (
        windows.market_return_w0_v1,
        windows.market_range_w0_v1,
        windows.market_return_w1_v1,
        windows.market_range_w1_v1,
    )
    if any(value is None or not math.isfinite(value) for value in required_values):
        reasons.append("market_window_measurement_incomplete")
    if reasons:
        return MarketShockStateResultV1(
            market_shock_state_v1="UNKNOWN_INCOMPLETE",
            market_shock_event_id_v1=None,
            shock_sign_v1=None,
            complete_v1=False,
            missing_reasons_v1=tuple(dict.fromkeys(reasons)),
        )

    assert thresholds is not None
    assert windows.market_return_w0_v1 is not None
    assert windows.market_range_w0_v1 is not None
    assert windows.market_return_w1_v1 is not None
    assert windows.market_range_w1_v1 is not None
    assert thresholds.market_return_w0_q10_v1 is not None
    assert thresholds.market_return_w0_q90_v1 is not None
    assert thresholds.market_range_w0_q75_v1 is not None
    assert thresholds.market_return_w1_q10_v1 is not None
    assert thresholds.market_return_w1_q90_v1 is not None
    assert thresholds.market_range_w1_q75_v1 is not None

    current_negative = (
        windows.market_return_w0_v1 <= thresholds.market_return_w0_q10_v1
        and windows.market_range_w0_v1 >= thresholds.market_range_w0_q75_v1
    )
    current_positive = (
        windows.market_return_w0_v1 >= thresholds.market_return_w0_q90_v1
        and windows.market_range_w0_v1 >= thresholds.market_range_w0_q75_v1
    )
    previous_negative = (
        windows.market_return_w1_v1 <= thresholds.market_return_w1_q10_v1
        and windows.market_range_w1_v1 >= thresholds.market_range_w1_q75_v1
    )
    previous_positive = (
        windows.market_return_w1_v1 >= thresholds.market_return_w1_q90_v1
        and windows.market_range_w1_v1 >= thresholds.market_range_w1_q75_v1
    )
    state: MarketShockStateV1
    sign: Literal[-1, 1] | None = None
    if current_negative and not previous_negative:
        state = "NEGATIVE_SHOCK_ONSET"
        sign = -1
    elif current_positive and not previous_positive:
        state = "POSITIVE_SHOCK_ONSET"
        sign = 1
    elif current_negative and previous_negative:
        state = "ONGOING_NEGATIVE_SHOCK"
    elif current_positive and previous_positive:
        state = "ONGOING_POSITIVE_SHOCK"
    elif windows.market_range_w0_v1 >= thresholds.market_range_w0_q75_v1:
        state = "ELEVATED_RANGE_NONDIRECTIONAL"
    else:
        state = "NORMAL_OTHER"
    return MarketShockStateResultV1(
        market_shock_state_v1=state,
        market_shock_event_id_v1=(
            None
            if sign is None
            else _shock_event_id(
                session=windows.session,
                checkpoint=windows.checkpoint,
                market_proxy=windows.market_proxy_v1,
                shock_sign=sign,
            )
        ),
        shock_sign_v1=sign,
        complete_v1=True,
        missing_reasons_v1=(),
    )


def _stock_w0_return(
    *,
    symbol: str,
    session: date,
    checkpoint: int,
    signal_timestamp: datetime,
    completed_stock_bars: tuple[MarketShockBarV1, ...],
) -> tuple[float | None, datetime | None, tuple[str, ...]]:
    required = tuple(range(checkpoint - 4, checkpoint))
    relevant = [
        bar
        for bar in completed_stock_bars
        if bar.symbol == symbol
        and bar.session == session
        and bar.bar_ordinal in required
    ]
    by_ordinal: dict[int, MarketShockBarV1] = {}
    reasons: list[str] = []
    for bar in relevant:
        if bar.bar_ordinal in by_ordinal:
            reasons.append(f"duplicate_stock_bar:{bar.bar_ordinal}")
            continue
        by_ordinal[bar.bar_ordinal] = bar
    for ordinal in required:
        if ordinal not in by_ordinal:
            reasons.append(f"missing_stock_bar:{ordinal}")
    for ordinal, bar in sorted(by_ordinal.items()):
        if not _valid_bar(bar, signal_timestamp=signal_timestamp):
            reasons.append(f"invalid_stock_bar:{ordinal}")
    ordered = [
        by_ordinal[ordinal]
        for ordinal in required
        if ordinal in by_ordinal
        and _valid_bar(by_ordinal[ordinal], signal_timestamp=signal_timestamp)
    ]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.bar_start_timestamp != previous.bar_complete_timestamp:
            reasons.append(
                "non_contiguous_stock_timestamps:"
                f"{previous.bar_ordinal}-{current.bar_ordinal}"
            )
    final = by_ordinal.get(checkpoint - 1)
    if (
        final is None
        or not _valid_bar(final, signal_timestamp=signal_timestamp)
        or final.bar_complete_timestamp != signal_timestamp
    ):
        reasons.append("stock_signal_timestamp_mismatch")
    unique = tuple(dict.fromkeys(reasons))
    maximum = max((bar.bar_complete_timestamp for bar in ordered), default=None)
    if unique or len(ordered) != len(required):
        return None, maximum, unique
    return math.log(ordered[-1].close / ordered[0].close), maximum, ()


def calculate_stock_shock_response_v1(
    *,
    symbol: str,
    session: date,
    checkpoint: int,
    signal_timestamp: datetime,
    completed_stock_bars: tuple[MarketShockBarV1, ...],
    market_return_w0_v1: float | None,
    market_shock_state_v1: MarketShockStateResultV1,
    threshold_15m: float | None,
) -> StockShockResponseResultV1:
    """Measure one stock locally; never consult peers, outcomes, or future bars."""

    if checkpoint not in FROZEN_SHOCK_CHECKPOINTS_V1:
        raise ValueError(f"checkpoint outside frozen M1C grid: {checkpoint}")
    stock_return, maximum_timestamp, stock_reasons = _stock_w0_return(
        symbol=symbol,
        session=session,
        checkpoint=checkpoint,
        signal_timestamp=signal_timestamp,
        completed_stock_bars=completed_stock_bars,
    )
    if stock_reasons:
        return StockShockResponseResultV1(
            stock_return_w0_v1=None,
            stock_absolute_alignment_v1=None,
            shock_relative_response_v1=None,
            shock_response_class_v1="UNKNOWN_INCOMPLETE",
            resisting_subtype_v1=None,
            maximum_stock_timestamp_v1=maximum_timestamp,
            complete_v1=False,
            missing_reasons_v1=stock_reasons,
        )
    assert stock_return is not None
    if market_shock_state_v1.market_shock_state_v1 not in {
        "NEGATIVE_SHOCK_ONSET",
        "POSITIVE_SHOCK_ONSET",
    }:
        if market_shock_state_v1.market_shock_state_v1 == "UNKNOWN_INCOMPLETE":
            reasons = (
                market_shock_state_v1.missing_reasons_v1
                or ("market_shock_state_incomplete",)
            )
            return StockShockResponseResultV1(
                stock_return_w0_v1=stock_return,
                stock_absolute_alignment_v1=None,
                shock_relative_response_v1=None,
                shock_response_class_v1="UNKNOWN_INCOMPLETE",
                resisting_subtype_v1=None,
                maximum_stock_timestamp_v1=maximum_timestamp,
                complete_v1=False,
                missing_reasons_v1=reasons,
            )
        return StockShockResponseResultV1(
            stock_return_w0_v1=stock_return,
            stock_absolute_alignment_v1=None,
            shock_relative_response_v1=None,
            shock_response_class_v1="NOT_SHOCK_ONSET",
            resisting_subtype_v1=None,
            maximum_stock_timestamp_v1=maximum_timestamp,
            complete_v1=True,
            missing_reasons_v1=(),
        )

    scale = None if threshold_15m is None else float(threshold_15m)
    if scale is None or not math.isfinite(scale) or scale <= 0.0:
        return StockShockResponseResultV1(
            stock_return_w0_v1=stock_return,
            stock_absolute_alignment_v1=None,
            shock_relative_response_v1=None,
            shock_response_class_v1="UNKNOWN_INCOMPLETE",
            resisting_subtype_v1=None,
            maximum_stock_timestamp_v1=maximum_timestamp,
            complete_v1=False,
            missing_reasons_v1=("threshold_15m_invalid",),
        )
    if market_return_w0_v1 is None or not math.isfinite(market_return_w0_v1):
        return StockShockResponseResultV1(
            stock_return_w0_v1=stock_return,
            stock_absolute_alignment_v1=None,
            shock_relative_response_v1=None,
            shock_response_class_v1="UNKNOWN_INCOMPLETE",
            resisting_subtype_v1=None,
            maximum_stock_timestamp_v1=maximum_timestamp,
            complete_v1=False,
            missing_reasons_v1=("market_return_w0_invalid",),
        )
    sign = market_shock_state_v1.shock_sign_v1
    if sign not in {-1, 1}:
        raise ValueError("signed shock onset lacks its deterministic sign")
    alignment = sign * stock_return / scale
    relative = sign * (stock_return - market_return_w0_v1) / scale
    response_class: ShockResponseClassV1
    subtype: ResistingSubtypeV1 | None = None
    if relative > 0.0:
        response_class = "AMPLIFYING"
    elif relative < 0.0:
        response_class = "RESISTING"
        subtype = (
            "RESISTING_BUT_STILL_WITH_SHOCK"
            if alignment > 0.0
            else "ABSOLUTELY_OPPOSING_SHOCK"
        )
    else:
        response_class = "NEUTRAL_EXACT"
    return StockShockResponseResultV1(
        stock_return_w0_v1=stock_return,
        stock_absolute_alignment_v1=alignment,
        shock_relative_response_v1=relative,
        shock_response_class_v1=response_class,
        resisting_subtype_v1=subtype,
        maximum_stock_timestamp_v1=maximum_timestamp,
        complete_v1=True,
        missing_reasons_v1=(),
    )


def partition_material_endpoint_v1(
    *,
    signed_return: float,
    threshold_15m: float,
) -> MaterialEndpointStateV1:
    """Partition the endpoint exactly around the strict frozen M1C event."""

    observed = float(signed_return)
    threshold = float(threshold_15m)
    if (
        not math.isfinite(observed)
        or not math.isfinite(threshold)
        or threshold <= 0.0
    ):
        raise ValueError("material endpoint inputs must be finite with a positive threshold")
    if observed > threshold:
        return "MATERIAL_UP"
    if observed < -threshold:
        return "MATERIAL_DOWN"
    return "NO_MATERIAL_MOVE"


def frozen_material_move_v1(
    *,
    signed_return: float,
    threshold_15m: float,
) -> bool:
    """Return the unchanged strict symmetric material-move event."""

    observed = float(signed_return)
    threshold = float(threshold_15m)
    if (
        not math.isfinite(observed)
        or not math.isfinite(threshold)
        or threshold <= 0.0
    ):
        raise ValueError("material endpoint inputs must be finite with a positive threshold")
    return abs(observed) > threshold


def assert_unprotected_sessions_v1(sessions: Iterable[object]) -> None:
    """Fail before calculation when a protected historical session is present."""

    for raw in sessions:
        if isinstance(raw, datetime):
            observed = raw.date()
        elif isinstance(raw, date):
            observed = raw
        else:
            observed = date.fromisoformat(str(raw)[:10])
        if observed >= date(2026, 1, 1):
            raise ValueError("protected 2026 historical outcomes must not be accessed")


def load_signed_market_shock_threshold_manifest_v1(
    path: str | Path,
) -> SignedMarketShockThresholdManifestV1:
    """Load one explicitly configured manifest; never discover a newest artifact."""

    source = Path(path)
    if not source.is_file():
        raise ValueError("signed market-shock threshold manifest is absent")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return SignedMarketShockThresholdManifestV1.model_validate(payload)


__all__ = [
    "FROZEN_SHOCK_CHECKPOINTS_V1",
    "MARKET_SHOCK_PROXY_V1",
    "CheckpointShockThresholdsV1",
    "MaterialEndpointStateV1",
    "MarketShockBarV1",
    "MarketShockStateResultV1",
    "MarketShockStateV1",
    "PreentryMarketWindowsV1",
    "ResistingSubtypeV1",
    "SignedMarketShockThresholdManifestV1",
    "ShockResponseClassV1",
    "StockShockResponseResultV1",
    "assert_unprotected_sessions_v1",
    "calculate_preentry_windows_v1",
    "calculate_stock_shock_response_v1",
    "classify_market_shock_state_v1",
    "frozen_material_move_v1",
    "load_signed_market_shock_threshold_manifest_v1",
    "partition_material_endpoint_v1",
]
