from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_prospective.signed_market_shock_v1 import (
    CheckpointShockThresholdsV1,
    MarketShockBarV1,
    MarketShockStateResultV1,
    assert_unprotected_sessions_v1,
    calculate_preentry_windows_v1,
    calculate_stock_shock_response_v1,
    classify_market_shock_state_v1,
    frozen_material_move_v1,
    load_signed_market_shock_threshold_manifest_v1,
    partition_material_endpoint_v1,
)
from stocker_research.m1c_signed_market_shock_transition_v1 import (
    assign_response_quintile_v1,
    freeze_checkpoint_thresholds_v1,
    freeze_response_quintiles_v1,
)

ROOT = Path(__file__).resolve().parents[1]


def _bars(
    *,
    symbol: str = "VTI",
    session: date = date(2025, 1, 2),
    count: int = 10,
) -> tuple[MarketShockBarV1, ...]:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    return tuple(
        MarketShockBarV1(
            symbol=symbol,
            session=session,
            bar_ordinal=ordinal,
            bar_start_timestamp=start + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=start + timedelta(minutes=5 * (ordinal + 1)),
            open=100.0 + ordinal,
            high=101.0 + ordinal,
            low=99.0 + ordinal,
            close=100.5 + ordinal,
            finalised=True,
        )
        for ordinal in range(count)
    )


def test_market_windows_use_exact_adjacent_nonoverlapping_preentry_bars() -> None:
    bars = _bars()
    signal_timestamp = bars[7].bar_complete_timestamp

    result = calculate_preentry_windows_v1(
        market_proxy="VTI",
        session=date(2025, 1, 2),
        checkpoint=8,
        signal_timestamp=signal_timestamp,
        completed_bars=bars,
    )

    assert result.complete_v1
    assert result.w0_bar_ordinals_v1 == (5, 6, 7)
    assert result.w1_bar_ordinals_v1 == (2, 3, 4)
    assert set(result.w0_bar_ordinals_v1).isdisjoint(result.w1_bar_ordinals_v1)
    assert result.market_return_w0_v1 == pytest.approx(
        math.log(bars[7].close / bars[4].close)
    )
    assert result.market_range_w0_v1 == pytest.approx(
        math.log(
            max(bar.high for bar in bars[5:8]) / min(bar.low for bar in bars[5:8])
        )
    )
    assert result.market_return_w1_v1 == pytest.approx(
        math.log(bars[4].close / bars[1].close)
    )
    assert result.market_range_w1_v1 == pytest.approx(
        math.log(
            max(bar.high for bar in bars[2:5]) / min(bar.low for bar in bars[2:5])
        )
    )
    assert result.maximum_market_timestamp_v1 == signal_timestamp
    assert 8 not in (*result.w0_bar_ordinals_v1, *result.w1_bar_ordinals_v1)


def test_market_windows_ignore_entry_and_future_bars() -> None:
    bars = _bars()
    changed = tuple(
        bar.model_copy(
            update={
                "open": 1_000_000.0,
                "high": 1_000_001.0,
                "low": 999_999.0,
                "close": 1_000_000.0,
            }
        )
        if bar.bar_ordinal >= 8
        else bar
        for bar in bars
    )
    arguments = {
        "market_proxy": "VTI",
        "session": date(2025, 1, 2),
        "checkpoint": 8,
        "signal_timestamp": bars[7].bar_complete_timestamp,
    }

    expected = calculate_preentry_windows_v1(completed_bars=bars, **arguments)
    actual = calculate_preentry_windows_v1(completed_bars=changed, **arguments)

    assert actual == expected


def test_checkpoint_six_is_incomplete_instead_of_crossing_sessions() -> None:
    bars = _bars(count=6)

    result = calculate_preentry_windows_v1(
        market_proxy="VTI",
        session=date(2025, 1, 2),
        checkpoint=6,
        signal_timestamp=bars[5].bar_complete_timestamp,
        completed_bars=bars,
    )

    assert not result.complete_v1
    assert result.market_return_w0_v1 is not None
    assert result.market_range_w0_v1 is not None
    assert result.market_return_w1_v1 is None
    assert result.market_range_w1_v1 is None
    assert result.missing_reasons_v1 == ("w1_reference_would_cross_session",)


def test_missing_or_partial_market_bars_fail_closed() -> None:
    bars = _bars()
    missing = tuple(bar for bar in bars if bar.bar_ordinal != 3)
    partial = tuple(
        bar.model_copy(update={"finalised": False}) if bar.bar_ordinal == 7 else bar
        for bar in bars
    )
    arguments = {
        "market_proxy": "VTI",
        "session": date(2025, 1, 2),
        "checkpoint": 8,
        "signal_timestamp": bars[7].bar_complete_timestamp,
    }

    missing_result = calculate_preentry_windows_v1(
        completed_bars=missing,
        **arguments,
    )
    partial_result = calculate_preentry_windows_v1(
        completed_bars=partial,
        **arguments,
    )

    assert not missing_result.complete_v1
    assert "missing_market_bar:3" in missing_result.missing_reasons_v1
    assert not partial_result.complete_v1
    assert "invalid_market_bar:7" in partial_result.missing_reasons_v1


def _thresholds() -> CheckpointShockThresholdsV1:
    return CheckpointShockThresholdsV1(
        checkpoint=8,
        market_return_w0_q10_v1=-0.02,
        market_return_w0_q90_v1=0.02,
        market_range_w0_q75_v1=0.03,
        market_return_w1_q10_v1=-0.01,
        market_return_w1_q90_v1=0.01,
        market_range_w1_q75_v1=0.02,
        market_return_w0_support_v1=250,
        market_range_w0_support_v1=250,
        market_return_w1_support_v1=250,
        market_range_w1_support_v1=250,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )


@pytest.mark.parametrize(
    (
        "w0_return",
        "w0_range",
        "w1_return",
        "w1_range",
        "expected",
        "shock_sign",
    ),
    [
        (-0.02, 0.03, 0.0, 0.02, "NEGATIVE_SHOCK_ONSET", -1),
        (0.02, 0.03, 0.0, 0.02, "POSITIVE_SHOCK_ONSET", 1),
        (-0.03, 0.04, -0.01, 0.02, "ONGOING_NEGATIVE_SHOCK", None),
        (0.03, 0.04, 0.01, 0.02, "ONGOING_POSITIVE_SHOCK", None),
        (0.0, 0.03, 0.0, 0.02, "ELEVATED_RANGE_NONDIRECTIONAL", None),
        (0.0, 0.01, 0.0, 0.01, "NORMAL_OTHER", None),
    ],
)
def test_market_shock_state_definitions_and_inclusive_boundaries(
    w0_return: float,
    w0_range: float,
    w1_return: float,
    w1_range: float,
    expected: str,
    shock_sign: int | None,
) -> None:
    windows = calculate_preentry_windows_v1(
        market_proxy="VTI",
        session=date(2025, 1, 2),
        checkpoint=8,
        signal_timestamp=_bars()[7].bar_complete_timestamp,
        completed_bars=_bars(),
    ).model_copy(
        update={
            "market_return_w0_v1": w0_return,
            "market_range_w0_v1": w0_range,
            "market_return_w1_v1": w1_return,
            "market_range_w1_v1": w1_range,
        }
    )

    result = classify_market_shock_state_v1(
        windows=windows,
        thresholds=_thresholds(),
    )

    assert result.market_shock_state_v1 == expected
    assert result.shock_sign_v1 == shock_sign
    assert result.complete_v1
    if shock_sign is None:
        assert result.market_shock_event_id_v1 is None
    else:
        assert result.market_shock_event_id_v1 is not None


def test_incomplete_window_or_calibration_produces_unknown_state() -> None:
    incomplete_windows = calculate_preentry_windows_v1(
        market_proxy="VTI",
        session=date(2025, 1, 2),
        checkpoint=6,
        signal_timestamp=_bars(count=6)[5].bar_complete_timestamp,
        completed_bars=_bars(count=6),
    )
    incomplete_thresholds = _thresholds().model_copy(
        update={
            "calibration_complete_v1": False,
            "calibration_missing_reason_v1": "insufficient_predictor_support",
        }
    )

    window_result = classify_market_shock_state_v1(
        windows=incomplete_windows,
        thresholds=None,
    )
    calibration_result = classify_market_shock_state_v1(
        windows=calculate_preentry_windows_v1(
            market_proxy="VTI",
            session=date(2025, 1, 2),
            checkpoint=8,
            signal_timestamp=_bars()[7].bar_complete_timestamp,
            completed_bars=_bars(),
        ),
        thresholds=incomplete_thresholds,
    )

    assert window_result.market_shock_state_v1 == "UNKNOWN_INCOMPLETE"
    assert "w1_reference_would_cross_session" in window_result.missing_reasons_v1
    assert calibration_result.market_shock_state_v1 == "UNKNOWN_INCOMPLETE"
    assert "insufficient_predictor_support" in calibration_result.missing_reasons_v1


def _shock_state(sign: int) -> MarketShockStateResultV1:
    return MarketShockStateResultV1(
        market_shock_state_v1=(
            "POSITIVE_SHOCK_ONSET" if sign == 1 else "NEGATIVE_SHOCK_ONSET"
        ),
        market_shock_event_id_v1=f"event-{sign}",
        shock_sign_v1=sign,
        complete_v1=True,
        missing_reasons_v1=(),
    )


def _with_w0_return(
    bars: tuple[MarketShockBarV1, ...],
    signed_return: float,
) -> tuple[MarketShockBarV1, ...]:
    terminal = bars[4].close * math.exp(signed_return)
    return tuple(
        bar.model_copy(
            update={
                "open": terminal,
                "high": terminal + 1.0,
                "low": terminal - 1.0,
                "close": terminal,
            }
        )
        if bar.bar_ordinal == 7
        else bar
        for bar in bars
    )


@pytest.mark.parametrize(
    ("shock_sign", "market_return", "stock_return", "expected_response", "expected_class"),
    [
        (1, 0.02, 0.03, 1.0, "AMPLIFYING"),
        (-1, -0.02, -0.03, 1.0, "AMPLIFYING"),
        (1, 0.02, 0.01, -1.0, "RESISTING"),
        (-1, -0.02, -0.01, -1.0, "RESISTING"),
        (1, 0.02, 0.02, 0.0, "NEUTRAL_EXACT"),
    ],
)
def test_stock_relative_response_and_fixed_classes(
    shock_sign: int,
    market_return: float,
    stock_return: float,
    expected_response: float,
    expected_class: str,
) -> None:
    bars = _bars(symbol="AAL")
    changed = _with_w0_return(bars, stock_return)
    exact_stock_return = math.log(changed[7].close / changed[4].close)
    effective_market_return = (
        exact_stock_return if expected_class == "NEUTRAL_EXACT" else market_return
    )

    result = calculate_stock_shock_response_v1(
        symbol="AAL",
        session=date(2025, 1, 2),
        checkpoint=8,
        signal_timestamp=bars[7].bar_complete_timestamp,
        completed_stock_bars=changed,
        market_return_w0_v1=effective_market_return,
        market_shock_state_v1=_shock_state(shock_sign),
        threshold_15m=0.01,
    )

    assert result.complete_v1
    assert result.stock_return_w0_v1 == pytest.approx(stock_return)
    assert result.shock_relative_response_v1 == pytest.approx(expected_response)
    assert result.stock_absolute_alignment_v1 == pytest.approx(
        shock_sign * stock_return / 0.01
    )
    assert result.shock_response_class_v1 == expected_class


def test_future_and_peer_stock_bars_cannot_change_one_stock_response() -> None:
    bars = _with_w0_return(_bars(symbol="AAL", count=10), 0.03)
    future_changed = tuple(
        bar.model_copy(
            update={
                "open": 1_000_000.0,
                "high": 1_000_001.0,
                "low": 999_999.0,
                "close": 1_000_000.0,
            }
        )
        if bar.bar_ordinal >= 8
        else bar
        for bar in bars
    )
    peers = _bars(symbol="MSFT", count=10)
    arguments = {
        "symbol": "AAL",
        "session": date(2025, 1, 2),
        "checkpoint": 8,
        "signal_timestamp": bars[7].bar_complete_timestamp,
        "market_return_w0_v1": 0.02,
        "market_shock_state_v1": _shock_state(1),
        "threshold_15m": 0.01,
    }

    expected = calculate_stock_shock_response_v1(
        completed_stock_bars=bars,
        **arguments,
    )
    changed = calculate_stock_shock_response_v1(
        completed_stock_bars=(*future_changed, *peers),
        **arguments,
    )

    assert changed == expected


def test_response_calculator_has_no_contaminated_or_peer_slate_inputs() -> None:
    parameters = set(signature(calculate_stock_shock_response_v1).parameters)

    assert parameters.isdisjoint(
        {
            "archived_signed_pressure",
            "archived_tension",
            "peer_slate",
            "peer_normalisation",
            "future_stock_bars",
            "future_market_bars",
            "stock_beta",
        }
    )


def test_resisting_subtypes_are_descriptive_only() -> None:
    bars = _bars(symbol="AAL")
    with_shock = _with_w0_return(bars, 0.01)
    opposing = _with_w0_return(bars, -0.01)
    arguments = {
        "symbol": "AAL",
        "session": date(2025, 1, 2),
        "checkpoint": 8,
        "signal_timestamp": bars[7].bar_complete_timestamp,
        "market_return_w0_v1": 0.02,
        "market_shock_state_v1": _shock_state(1),
        "threshold_15m": 0.01,
    }

    lagging = calculate_stock_shock_response_v1(
        completed_stock_bars=with_shock,
        **arguments,
    )
    absolute_opposition = calculate_stock_shock_response_v1(
        completed_stock_bars=opposing,
        **arguments,
    )

    assert lagging.shock_response_class_v1 == "RESISTING"
    assert lagging.resisting_subtype_v1 == "RESISTING_BUT_STILL_WITH_SHOCK"
    assert absolute_opposition.shock_response_class_v1 == "RESISTING"
    assert absolute_opposition.resisting_subtype_v1 == "ABSOLUTELY_OPPOSING_SHOCK"


def test_invalid_iv_or_nonshock_state_fails_closed() -> None:
    bars = _bars(symbol="AAL")
    normal = MarketShockStateResultV1(
        market_shock_state_v1="NORMAL_OTHER",
        market_shock_event_id_v1=None,
        shock_sign_v1=None,
        complete_v1=True,
        missing_reasons_v1=(),
    )
    arguments = {
        "symbol": "AAL",
        "session": date(2025, 1, 2),
        "checkpoint": 8,
        "signal_timestamp": bars[7].bar_complete_timestamp,
        "completed_stock_bars": bars,
        "market_return_w0_v1": 0.02,
    }

    invalid = calculate_stock_shock_response_v1(
        market_shock_state_v1=_shock_state(1),
        threshold_15m=0.0,
        **arguments,
    )
    nonshock = calculate_stock_shock_response_v1(
        market_shock_state_v1=normal,
        threshold_15m=0.01,
        **arguments,
    )

    assert not invalid.complete_v1
    assert invalid.shock_response_class_v1 == "UNKNOWN_INCOMPLETE"
    assert invalid.missing_reasons_v1 == ("threshold_15m_invalid",)
    assert nonshock.complete_v1
    assert nonshock.shock_response_class_v1 == "NOT_SHOCK_ONSET"
    assert nonshock.shock_relative_response_v1 is None


@pytest.mark.parametrize(
    ("signed_return", "expected"),
    [
        (0.0200001, "MATERIAL_UP"),
        (-0.0200001, "MATERIAL_DOWN"),
        (0.02, "NO_MATERIAL_MOVE"),
        (-0.02, "NO_MATERIAL_MOVE"),
        (0.0, "NO_MATERIAL_MOVE"),
    ],
)
def test_strict_material_endpoint_partition(
    signed_return: float,
    expected: str,
) -> None:
    state = partition_material_endpoint_v1(
        signed_return=signed_return,
        threshold_15m=0.02,
    )

    assert state == expected
    assert (state in {"MATERIAL_UP", "MATERIAL_DOWN"}) == frozen_material_move_v1(
        signed_return=signed_return,
        threshold_15m=0.02,
    )


def test_protected_sessions_remain_fail_closed() -> None:
    assert_unprotected_sessions_v1([date(2025, 12, 31)])

    with pytest.raises(ValueError, match="protected"):
        assert_unprotected_sessions_v1([date(2026, 1, 1)])


def test_checkpoint_thresholds_use_only_2024_market_predictors() -> None:
    development = pd.DataFrame(
        {
            "session": [f"2024-01-{day:02d}" for day in range(1, 26)],
            "checkpoint": [8] * 25,
            "complete_v1": [True] * 25,
            "market_return_w0_v1": np.linspace(-0.05, 0.05, 25),
            "market_range_w0_v1": np.linspace(0.01, 0.06, 25),
            "market_return_w1_v1": np.linspace(-0.04, 0.04, 25),
            "market_range_w1_v1": np.linspace(0.02, 0.07, 25),
            "future_signed_return_15m": np.linspace(-999.0, 999.0, 25),
        }
    )
    changed_outcomes = development.assign(future_signed_return_15m=-1_000_000.0)
    with_2025 = pd.concat(
        [
            development,
            development.assign(
                session="2025-01-02",
                market_return_w0_v1=999.0,
                market_range_w0_v1=999.0,
                market_return_w1_v1=999.0,
                market_range_w1_v1=999.0,
            ),
        ],
        ignore_index=True,
    )

    expected = freeze_checkpoint_thresholds_v1(development)[8]
    changed = freeze_checkpoint_thresholds_v1(changed_outcomes)[8]
    future_period = freeze_checkpoint_thresholds_v1(with_2025)[8]

    assert changed == expected
    assert future_period == expected
    assert expected.calibration_complete_v1
    assert expected.market_return_w0_q10_v1 == pytest.approx(
        np.quantile(development["market_return_w0_v1"], 0.1, method="linear")
    )
    assert expected.market_return_w0_q90_v1 == pytest.approx(
        np.quantile(development["market_return_w0_v1"], 0.9, method="linear")
    )
    assert expected.market_range_w0_q75_v1 == pytest.approx(
        np.quantile(development["market_range_w0_v1"], 0.75, method="linear")
    )


def test_checkpoint_six_calibration_stays_incomplete_without_a_fallback() -> None:
    frame = pd.DataFrame(
        {
            "session": [f"2024-02-{day:02d}" for day in range(1, 26)],
            "checkpoint": [6] * 25,
            "complete_v1": [False] * 25,
            "market_return_w0_v1": np.linspace(-0.05, 0.05, 25),
            "market_range_w0_v1": np.linspace(0.01, 0.06, 25),
            "market_return_w1_v1": [math.nan] * 25,
            "market_range_w1_v1": [math.nan] * 25,
        }
    )

    frozen = freeze_checkpoint_thresholds_v1(frame)[6]

    assert not frozen.calibration_complete_v1
    assert frozen.calibration_missing_reason_v1 == (
        "insufficient_predictor_support:"
        "market_return_w0_v1=0,market_range_w0_v1=0,"
        "market_return_w1_v1=0,market_range_w1_v1=0"
    )
    assert frozen.market_return_w0_q10_v1 is None
    assert frozen.market_return_w1_q10_v1 is None


def test_calibration_excludes_partially_complete_market_rows() -> None:
    frame = pd.DataFrame(
        {
            "session": [f"2024-04-{day:02d}" for day in range(1, 26)],
            "checkpoint": [8] * 25,
            "complete_v1": [True] * 24 + [False],
            "market_return_w0_v1": [0.0] * 24 + [999.0],
            "market_range_w0_v1": [0.01] * 24 + [999.0],
            "market_return_w1_v1": [0.0] * 24 + [999.0],
            "market_range_w1_v1": [0.01] * 24 + [999.0],
        }
    )

    frozen = freeze_checkpoint_thresholds_v1(frame)[8]

    assert frozen.calibration_complete_v1
    assert frozen.market_return_w0_support_v1 == 24
    assert frozen.market_return_w0_q90_v1 == 0.0
    assert frozen.market_range_w1_q75_v1 == 0.01


def test_response_quintiles_use_only_valid_2024_predictors() -> None:
    development = pd.DataFrame(
        {
            "session": [f"2024-03-{day:02d}" for day in range(1, 26)],
            "M1C_probability": [0.9] * 25,
            "market_shock_state_v1": ["NEGATIVE_SHOCK_ONSET"] * 25,
            "shock_response_complete_v1": [True] * 25,
            "shock_relative_response_v1": np.arange(25, dtype=float),
            "future_signed_return_15m": np.linspace(-10.0, 10.0, 25),
        }
    )
    changed = development.assign(future_signed_return_15m=999_999.0)
    future = pd.concat(
        [
            development,
            development.assign(
                session="2025-03-01",
                shock_relative_response_v1=-999_999.0,
            ),
        ],
        ignore_index=True,
    )

    expected = freeze_response_quintiles_v1(development)

    assert freeze_response_quintiles_v1(changed) == expected
    assert freeze_response_quintiles_v1(future) == expected
    assert expected.calibration_complete_v1
    assert expected.support_v1 == 25
    assert expected.q20_v1 == pytest.approx(4.8)
    assert expected.q80_v1 == pytest.approx(19.2)
    assert assign_response_quintile_v1(expected.q20_v1, expected) == "Q1"
    assert assign_response_quintile_v1(expected.q20_v1 + 0.01, expected) == "Q2"


def test_response_quintiles_do_not_invent_an_unstated_minimum_support() -> None:
    predictors = pd.DataFrame(
        {
            "session": [f"2024-04-{day:02d}" for day in range(1, 9)],
            "M1C_probability": [0.9] * 8,
            "market_shock_state_v1": ["POSITIVE_SHOCK_ONSET"] * 8,
            "shock_response_complete_v1": [True] * 8,
            "shock_relative_response_v1": np.arange(8, dtype=float),
        }
    )

    frozen = freeze_response_quintiles_v1(predictors)

    assert frozen.calibration_complete_v1
    assert frozen.support_v1 == 8
    assert frozen.q20_v1 == pytest.approx(1.4)
    assert frozen.q80_v1 == pytest.approx(5.6)


def test_frozen_threshold_manifest_loader_accepts_only_the_v1_identity(
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "research"
        / "directional-readiness"
        / "20260728-m1c-signed-market-shock-transition-v1"
        / "artifacts"
        / "primary"
        / "checkpoint_shock_threshold_manifest_v1.json"
    )
    manifest = load_signed_market_shock_threshold_manifest_v1(source)

    assert manifest.market_proxy_v1 == "VTI"
    assert manifest.threshold_for_checkpoint(6) is not None
    assert not manifest.threshold_for_checkpoint(6).calibration_complete_v1
    assert manifest.threshold_for_checkpoint(8).calibration_complete_v1

    drifted = source.read_text(encoding="utf-8").replace(
        '"signed_return_lower": 0.1',
        '"signed_return_lower": 0.2',
    )
    drifted_path = tmp_path / "drifted.json"
    drifted_path.write_text(drifted, encoding="utf-8")

    with pytest.raises(ValueError, match="quantiles differ"):
        load_signed_market_shock_threshold_manifest_v1(drifted_path)
