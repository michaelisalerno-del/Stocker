from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def _bars(symbol: str) -> list[dict[str, object]]:
    prices = [
        (99.0, 101.0, 98.0, 100.0),
        (100.0, 103.0, 99.0, 102.0),
        (102.0, 104.0, 100.0, 101.0),
        (101.0, 106.0, 100.5, 105.0),
    ]
    return [
        {
            "symbol": symbol,
            "session": "2025-01-06",
            "bar_ordinal": ordinal,
            "open": values[0],
            "high": values[1],
            "low": values[2],
            "close": values[3],
            "bar_start_timestamp": pd.Timestamp("2025-01-06T14:30:00Z")
            + pd.Timedelta(minutes=5 * ordinal),
            "bar_complete_timestamp": pd.Timestamp("2025-01-06T14:35:00Z")
            + pd.Timedelta(minutes=5 * ordinal),
        }
        for ordinal, values in enumerate(prices)
    ]


def test_build_movement_panels_keeps_underlying_rows_but_requires_valid_pair_for_iv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(EXPERIMENT_DIR))
    from run_movement_outcomes import build_movement_panels

    structural = pd.DataFrame(
        [
            {
                "row_id": f"{symbol}|2025-01-06|1",
                "symbol": symbol,
                "session": "2025-01-06",
                "checkpoint": 1,
                "checkpoint_bar_ordinal_zero_based": 0,
                "period": "assessment",
                "route_resolution_state": state,
                "row_weight": 0.5,
                "first_completion_lead": None,
            }
            for symbol, state in (("AAL", "BROAD_CONFLICT"), ("MSTR", "LOW_ROUTE_SUPPORT"))
        ]
    )
    bars = pd.DataFrame([*_bars("AAL"), *_bars("MSTR")])
    pairs = pd.DataFrame(
        [
            {
                "symbol": "AAL",
                "signal_date": "2025-01-06",
                "required_options_date": "2025-01-03",
                "pair_available": True,
                "atm_iv": 0.4,
                "front_dte": 14,
            },
            {
                "symbol": "MSTR",
                "signal_date": "2025-01-06",
                "required_options_date": "2025-01-03",
                "pair_available": False,
                "atm_iv": None,
                "front_dte": None,
            },
        ]
    )

    movement, iv_relative = build_movement_panels(structural, bars, pairs)

    assert len(movement) == 2
    assert set(movement["symbol"]) == {"AAL", "MSTR"}
    assert len(iv_relative) == 1
    assert iv_relative.iloc[0]["symbol"] == "AAL"
    assert iv_relative.iloc[0]["absolute_log_return_15m"] == pytest.approx(abs(math.log(1.05)))
    expected_sigma = 0.4 * math.sqrt(15 / (252 * 390))
    assert iv_relative.iloc[0]["iv_sigma_15m"] == pytest.approx(expected_sigma)
    assert iv_relative.iloc[0]["iv_expected_absolute_15m"] == pytest.approx(
        expected_sigma * math.sqrt(2 / math.pi)
    )


def test_build_movement_panels_rejects_same_day_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(EXPERIMENT_DIR))
    from run_movement_outcomes import build_movement_panels

    structural = pd.DataFrame(
        [
            {
                "row_id": "AAL|2025-01-06|1",
                "symbol": "AAL",
                "session": "2025-01-06",
                "checkpoint": 1,
                "checkpoint_bar_ordinal_zero_based": 0,
                "period": "assessment",
                "route_resolution_state": "BROAD_CONFLICT",
                "row_weight": 1.0,
                "first_completion_lead": None,
            }
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "symbol": "AAL",
                "signal_date": "2025-01-06",
                "required_options_date": "2025-01-06",
                "pair_available": True,
                "atm_iv": 0.4,
            }
        ]
    )

    with pytest.raises(ValueError, match="exact previous trading session"):
        build_movement_panels(structural, pd.DataFrame(_bars("AAL")), pairs)


def test_route_state_summary_uses_frozen_row_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(EXPERIMENT_DIR))
    from run_movement_outcomes import summarize_route_states

    panel = pd.DataFrame(
        [
            {
                "row_id": "b1",
                "symbol": "AAL",
                "session": "2025-01-06",
                "period": "assessment",
                "route_resolution_state": "BROAD_CONFLICT",
                "row_weight": 1.0,
                "absolute_log_return_15m": 0.02,
                "absolute_log_return_10m": 0.01,
                "absolute_log_return_30m": 0.03,
                "absolute_log_return_60m": 0.05,
                "iv_expected_absolute_15m": 0.01,
                "iv_absolute_residual_15m": 0.01,
                "iv_sigma_ratio_15m": 1.0,
                "movement_exceeds_iv_expected_absolute": 1,
                "movement_exceeds_one_iv_sigma": 0,
                "realised_range_15m": 0.03,
                "maximum_absolute_excursion_15m": 0.02,
                "realised_variance_15m": 0.0004,
                "registered_completion_in_bars_2_or_3": 1,
                "movement_before_completion": 0.005,
                "movement_from_completion_to_horizon_end": 0.002,
            },
            {
                "row_id": "b2",
                "symbol": "MSTR",
                "session": "2025-01-07",
                "period": "assessment",
                "route_resolution_state": "BROAD_CONFLICT",
                "row_weight": 3.0,
                "absolute_log_return_15m": 0.04,
                "absolute_log_return_10m": 0.03,
                "absolute_log_return_30m": 0.05,
                "absolute_log_return_60m": float("nan"),
                "iv_expected_absolute_15m": 0.01,
                "iv_absolute_residual_15m": 0.03,
                "iv_sigma_ratio_15m": 2.0,
                "movement_exceeds_iv_expected_absolute": 1,
                "movement_exceeds_one_iv_sigma": 1,
                "realised_range_15m": 0.05,
                "maximum_absolute_excursion_15m": 0.04,
                "realised_variance_15m": 0.0008,
                "registered_completion_in_bars_2_or_3": 0,
                "movement_before_completion": float("nan"),
                "movement_from_completion_to_horizon_end": float("nan"),
            },
            {
                "row_id": "l1",
                "symbol": "WULF",
                "session": "2025-01-08",
                "period": "assessment",
                "route_resolution_state": "LOW_ROUTE_SUPPORT",
                "row_weight": 2.0,
                "absolute_log_return_15m": 0.0,
                "absolute_log_return_10m": 0.0,
                "absolute_log_return_30m": 0.01,
                "absolute_log_return_60m": 0.02,
                "iv_expected_absolute_15m": 0.01,
                "iv_absolute_residual_15m": -0.01,
                "iv_sigma_ratio_15m": 0.0,
                "movement_exceeds_iv_expected_absolute": 0,
                "movement_exceeds_one_iv_sigma": 0,
                "realised_range_15m": 0.01,
                "maximum_absolute_excursion_15m": 0.01,
                "realised_variance_15m": 0.0001,
                "registered_completion_in_bars_2_or_3": 0,
                "movement_before_completion": float("nan"),
                "movement_from_completion_to_horizon_end": float("nan"),
            },
        ]
    )

    states, contrasts = summarize_route_states(panel)
    broad = states.loc[
        states["scope"].eq("assessment") & states["route_resolution_state"].eq("BROAD_CONFLICT")
    ].iloc[0]
    contrast = contrasts.loc[contrasts["scope"].eq("assessment")].iloc[0]

    assert broad["mean_iv_absolute_residual_15m"] == pytest.approx(0.025)
    assert broad["median_iv_absolute_residual_15m"] == pytest.approx(0.03)
    assert broad["exceed_iv_expected_rate"] == pytest.approx(1.0)
    assert broad["absolute_log_return_60m_available_rows"] == 1
    assert broad["mean_absolute_log_return_10m"] == pytest.approx(0.025)
    assert broad["registered_completion_in_bars_2_or_3_rate"] == pytest.approx(0.25)
    assert broad["mean_movement_before_completion"] == pytest.approx(0.005)
    assert contrast["iv_absolute_residual_15m_difference"] == pytest.approx(0.035)


def test_stock_date_summary_reads_iv_metrics_only_from_valid_pair_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(EXPERIMENT_DIR))
    from run_movement_outcomes import summarize_stock_dates

    movement = pd.DataFrame(
        [
            {
                "row_id": "a",
                "symbol": "AAL",
                "session": "2025-01-06",
                "required_options_date": "2025-01-03",
                "period": "assessment",
                "pair_available": True,
                "row_weight": 1.0,
                "absolute_log_return_15m": 0.02,
                "realised_range_15m": 0.03,
                "maximum_absolute_excursion_15m": 0.025,
            },
            {
                "row_id": "m",
                "symbol": "MSTR",
                "session": "2025-01-06",
                "required_options_date": "2025-01-03",
                "period": "assessment",
                "pair_available": False,
                "row_weight": 1.0,
                "absolute_log_return_15m": 0.01,
                "realised_range_15m": 0.02,
                "maximum_absolute_excursion_15m": 0.015,
            },
        ]
    )
    iv_relative = movement.iloc[[0]].assign(
        front_dte=14,
        atm_iv=0.4,
        iv_expected_absolute_15m=0.01,
        iv_absolute_residual_15m=0.01,
        iv_sigma_ratio_15m=1.25,
        movement_exceeds_iv_expected_absolute=1,
    )

    summary = summarize_stock_dates(movement, iv_relative)
    aal = summary.loc[summary["symbol"].eq("AAL")].iloc[0]
    mstr = summary.loc[summary["symbol"].eq("MSTR")].iloc[0]

    assert aal["mean_iv_absolute_residual_15m"] == pytest.approx(0.01)
    assert pd.isna(mstr["mean_iv_absolute_residual_15m"])


def test_required_field_comparison_detects_join_and_movement_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(EXPERIMENT_DIR))
    from run_movement_outcomes import compare_required_fields

    expected = pd.DataFrame(
        [
            {
                "row_id": "x",
                "call_contract_id": "CALL-A",
                "entry_bar_start_timestamp": pd.Timestamp("2025-01-06T14:35:00Z"),
                "absolute_log_return_15m": 0.02,
                "movement_before_completion": float("nan"),
            }
        ]
    )
    changed = expected.assign(call_contract_id="CALL-B", absolute_log_return_15m=0.03)

    identical_result = compare_required_fields(expected, expected.copy(), list(expected.columns))
    changed_result = compare_required_fields(expected, changed, list(expected.columns))

    assert identical_result["field_mismatches"] == 0
    assert identical_result["maximum_numeric_difference"] == 0.0
    assert changed_result["field_mismatches"] == 2
    assert changed_result["maximum_numeric_difference"] == pytest.approx(0.01)
