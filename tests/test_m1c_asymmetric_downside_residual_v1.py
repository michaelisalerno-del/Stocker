from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.m1c_asymmetric_downside_residual_v1 import (
    DOWNSIDE_FEATURES,
    apply_asymmetric_policy,
    assert_unprotected_sessions,
    build_downside_features,
    expanding_time_ordered_oof,
    freeze_action_thresholds,
    joint_probabilities,
    partition_endpoint_return,
    partition_first_breach_ohlc,
)


@pytest.mark.parametrize(
    ("signed_return", "expected"),
    [
        (0.021, "UP_MOVE"),
        (-0.021, "DOWN_MOVE"),
        (0.005, "NO_MOVE"),
        (0.02, "UP_MOVE"),
        (-0.02, "DOWN_MOVE"),
    ],
)
def test_endpoint_partition_includes_exact_directional_thresholds(
    signed_return: float,
    expected: str,
) -> None:
    assert partition_endpoint_return(signed_return, implied_movement=0.02) == expected


def test_first_breach_partition_does_not_invent_an_intrabar_path() -> None:
    both_in_first_bar = pd.DataFrame(
        {
            "high": [103.0, 104.0],
            "low": [97.0, 96.0],
        }
    )
    up_then_down = pd.DataFrame(
        {
            "high": [103.0, 104.0],
            "low": [99.0, 96.0],
        }
    )
    no_breach = pd.DataFrame(
        {
            "high": [101.0, 101.5],
            "low": [99.0, 98.5],
        }
    )

    assert (
        partition_first_breach_ohlc(
            both_in_first_bar,
            entry_price=100.0,
            implied_log_movement=0.02,
        )
        == "AMBIGUOUS_BOTH_WITHIN_BAR"
    )
    assert (
        partition_first_breach_ohlc(
            up_then_down,
            entry_price=100.0,
            implied_log_movement=0.02,
        )
        == "UP_FIRST"
    )
    assert (
        partition_first_breach_ohlc(
            no_breach,
            entry_price=100.0,
            implied_log_movement=0.02,
        )
        == "NO_BREACH"
    )


def test_fixed_downside_features_match_a_worked_same_session_example() -> None:
    start = pd.Timestamp("2024-01-02T14:30:00Z")
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    bars = pd.DataFrame(
        [
            {
                "stock": "AAL",
                "session": "2024-01-02",
                "bar_ordinal": ordinal,
                "bar_complete_timestamp": start + pd.Timedelta(minutes=(ordinal + 1) * 5),
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000.0,
            }
            for ordinal, close in enumerate(closes)
        ]
    )
    checkpoints = pd.DataFrame(
        [
            {
                "stock": "AAL",
                "session": "2024-01-02",
                "checkpoint": 6,
                "feature_available_timestamp_utc": bars.iloc[-1]["bar_complete_timestamp"],
                "implied_movement_15m_price": 2.0,
            }
        ]
    )

    result = build_downside_features(checkpoints, bars).iloc[0]

    assert result["D1_signed_return_5m"] == pytest.approx(0.009569451016150672)
    assert result["D2_signed_return_15m"] == pytest.approx(0.028987536873252187)
    assert result["D3_close_location_15m"] == pytest.approx(0.5)
    assert result["D4_distance_from_session_vwap_iv"] == pytest.approx(1.25)
    assert result["maximum_predictor_bar_ordinal"] == 5


def test_downside_features_ignore_future_bars_and_other_stocks() -> None:
    start = pd.Timestamp("2024-01-02T14:30:00Z")
    base = pd.DataFrame(
        [
            {
                "stock": "AAL",
                "session": "2024-01-02",
                "bar_ordinal": ordinal,
                "bar_complete_timestamp": start + pd.Timedelta(minutes=(ordinal + 1) * 5),
                "high": 101.0 + ordinal,
                "low": 99.0 + ordinal,
                "close": 100.0 + ordinal,
                "volume": 1_000.0,
            }
            for ordinal in range(8)
        ]
    )
    checkpoints = pd.DataFrame(
        [
            {
                "stock": "AAL",
                "session": "2024-01-02",
                "checkpoint": 6,
                "feature_available_timestamp_utc": start + pd.Timedelta(minutes=30),
                "implied_movement_15m_price": 2.0,
            }
        ]
    )
    peers = base.assign(stock="MSFT", close=10_000.0, high=10_001.0, low=9_999.0)
    prior_session = base.assign(
        session="2024-01-01",
        close=50_000.0,
        high=50_001.0,
        low=49_999.0,
    )
    expanded = pd.concat([base, peers, prior_session], ignore_index=True)
    changed = expanded.copy()
    changed.loc[
        changed["stock"].eq("AAL")
        & changed["session"].eq("2024-01-02")
        & changed["bar_ordinal"].ge(6),
        ["high", "low", "close", "volume"],
    ] = [1_000_001.0, 999_999.0, 1_000_000.0, 9_000_000.0]
    changed.loc[changed["stock"].eq("MSFT"), "close"] = -999.0

    columns = [
        "D1_signed_return_5m",
        "D2_signed_return_15m",
        "D3_close_location_15m",
        "D4_distance_from_session_vwap_iv",
        "maximum_predictor_bar_ordinal",
        "maximum_predictor_timestamp",
    ]
    expected = build_downside_features(checkpoints, base)[columns]
    actual = build_downside_features(checkpoints, changed)[columns]
    without_peer = build_downside_features(
        checkpoints,
        changed.loc[changed["stock"].eq("AAL")],
    )[columns]

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(without_peer, expected)


def test_zero_range_close_location_is_missing_without_imputation() -> None:
    start = pd.Timestamp("2024-01-02T14:30:00Z")
    bars = pd.DataFrame(
        [
            {
                "stock": "AAL",
                "session": "2024-01-02",
                "bar_ordinal": ordinal,
                "bar_complete_timestamp": start + pd.Timedelta(minutes=(ordinal + 1) * 5),
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1_000.0,
            }
            for ordinal in range(6)
        ]
    )
    checkpoints = pd.DataFrame(
        [
            {
                "stock": "AAL",
                "session": "2024-01-02",
                "checkpoint": 6,
                "feature_available_timestamp_utc": start + pd.Timedelta(minutes=30),
                "implied_movement_15m_price": 2.0,
            }
        ]
    )

    row = build_downside_features(checkpoints, bars).iloc[0]

    assert pd.isna(row["D3_close_location_15m"])
    assert not bool(row["downside_features_complete"])
    assert row["downside_feature_missing_reason"] == "zero_trailing_15m_range"


def test_protected_sessions_fail_closed() -> None:
    assert_unprotected_sessions(pd.Series(["2024-01-02", "2025-12-31"]))

    with pytest.raises(ValueError, match="protected"):
        assert_unprotected_sessions(pd.Series(["2025-12-31", "2026-01-01"]))


def test_expanding_oof_scalers_and_models_never_train_on_future_sessions() -> None:
    sessions = pd.to_datetime(
        [f"2024-{month:02d}-15" for month in range(1, 13)],
        utc=True,
    )
    values = [0.0, 1.0, 2.0, *([1_000.0] * 9)]
    development = pd.DataFrame(
        {
            "session": sessions,
            "is_down_move": [0, 1] * 6,
            **{
                feature: [value + feature_index for value in values]
                for feature_index, feature in enumerate(DOWNSIDE_FEATURES)
            },
        }
    )

    predictions, audits = expanding_time_ordered_oof(
        development,
        target_column="is_down_move",
    )

    assert len(predictions) == 9
    assert predictions["session"].min() == pd.Timestamp("2024-04-15", tz="UTC")
    assert [audit["fold"] for audit in audits] == [1, 2, 3]
    assert all(audit["train_end"] < audit["predict_start"] for audit in audits)
    assert audits[0]["standardisation"]["means"][DOWNSIDE_FEATURES[0]] == pytest.approx(1.0)
    assert audits[0]["train_rows"] == 3
    assert audits[0]["predict_rows"] == 3


def test_action_thresholds_use_only_the_2024_oof_score_distribution() -> None:
    oof = pd.DataFrame(
        {
            "q_down_oof": [0.1, 0.2, 0.3, 0.4, 0.5],
            "future_15m_signed_return": [99.0, -99.0, 5.0, -5.0, 0.0],
        }
    )
    changed_outcomes = oof.assign(future_15m_signed_return=-1_000_000.0)

    thresholds = freeze_action_thresholds(oof["q_down_oof"])
    changed = freeze_action_thresholds(changed_outcomes["q_down_oof"])

    assert thresholds == changed
    assert thresholds["low"] == pytest.approx(0.18)
    assert thresholds["high"] == pytest.approx(0.42)


def test_low_downside_is_a_call_only_at_the_frozen_low_threshold() -> None:
    actions = apply_asymmetric_policy(
        pd.Series([0.1, 0.2, 0.5, 0.8, 0.9, float("nan")]),
        low_threshold=0.2,
        high_threshold=0.8,
    )

    assert actions.tolist() == [
        "CALL",
        "CALL",
        "ABSTAIN",
        "PUT",
        "PUT",
        "ABSTAIN",
    ]


def test_joint_probabilities_are_blocked_for_the_audited_target_mismatch() -> None:
    with pytest.raises(ValueError, match="target mismatch"):
        joint_probabilities(
            pd.Series([0.6]),
            pd.Series([0.25]),
            exact_target_compatibility=False,
        )

    coherent = joint_probabilities(
        pd.Series([0.6]),
        pd.Series([0.25]),
        exact_target_compatibility=True,
    )
    assert coherent.iloc[0]["p_down_joint_v1"] == pytest.approx(0.15)
    assert coherent.iloc[0]["p_up_joint_v1"] == pytest.approx(0.45)
    assert coherent.iloc[0]["p_no_move_joint_v1"] == pytest.approx(0.4)
    assert coherent.iloc[0].sum() == pytest.approx(1.0)
