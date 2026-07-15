from __future__ import annotations

import math

import pandas as pd

from stocker_research.sequential_loop_competitor_veto import (
    build_registered_checkpoints,
    remaining_payoff,
)


def _state_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_ordinal": [10, 13, 17, 30],
            "state": [4, 2, 4, 6],
            "start_timestamp": pd.to_datetime(
                [
                    "2025-04-03T14:20:00Z",
                    "2025-04-03T14:35:00Z",
                    "2025-04-03T14:55:00Z",
                    "2025-04-03T16:00:00Z",
                ],
                utc=True,
            ),
        }
    )


def _bars() -> pd.DataFrame:
    timestamps = pd.date_range("2025-04-03T14:20:00Z", periods=40, freq="5min")
    opens = [100.0 + index for index in range(40)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": [value + 1.0 for value in opens],
            "low": [value - 1.0 for value in opens],
            "close": [value + 0.5 for value in opens],
            "session_date": "2025-04-03",
        }
    )


def test_registered_checkpoints_use_only_completed_observable_transitions() -> None:
    checkpoints = build_registered_checkpoints(
        _state_runs(),
        anchor_ordinal=10,
        terminal_ordinal=34,
        target_cycle="2->4->2",
        anchor_state=4,
    )

    first = checkpoints.loc[checkpoints["checkpoint_type"].eq("first_completed_transition")].iloc[0]
    completion = checkpoints.loc[checkpoints["checkpoint_type"].eq("exact_parent_completion")].iloc[
        0
    ]
    assert first["observed_transitions_json"] == "[2]"
    assert first["checkpoint_timestamp"] == pd.Timestamp("2025-04-03T14:40:00Z")
    assert completion["observed_transitions_json"] == "[2,4]"
    assert completion["bars_since_anchor"] == 7
    assert (
        checkpoints["feature_max_availability_timestamp"]
        .le(checkpoints["checkpoint_timestamp"])
        .all()
    )


def test_appending_future_state_runs_does_not_change_frozen_checkpoints() -> None:
    original = _state_runs().iloc[:3].copy()
    appended = _state_runs().copy()

    expected = build_registered_checkpoints(
        original, anchor_ordinal=10, terminal_ordinal=34, target_cycle="2->4->2", anchor_state=4
    )
    actual = build_registered_checkpoints(
        appended, anchor_ordinal=10, terminal_ordinal=34, target_cycle="2->4->2", anchor_state=4
    )

    pd.testing.assert_frame_equal(expected, actual)


def test_remaining_payoff_starts_at_next_open_and_excludes_prior_profit() -> None:
    bars = _bars()
    result = remaining_payoff(
        bars,
        direction=1,
        checkpoint_timestamp=pd.Timestamp("2025-04-03T14:40:00Z"),
        terminal_timestamp=pd.Timestamp("2025-04-03T16:25:00Z"),
    )

    assert result.status == "available"
    assert result.entry_timestamp == pd.Timestamp("2025-04-03T14:40:00Z")
    assert result.entry_price == 104.0
    assert result.constant_terminal_exit_timestamp == pd.Timestamp("2025-04-03T16:25:00Z")
    expected_gross = 10_000.0 * (124.5 / 104.0 - 1.0)
    assert math.isclose(result.constant_terminal_gross_bps, expected_gross)
    assert math.isclose(result.constant_terminal_net_bps, expected_gross - 10.0)


def test_constant_terminal_and_restarted_horizon_are_separate() -> None:
    result = remaining_payoff(
        _bars(),
        direction=-1,
        checkpoint_timestamp=pd.Timestamp("2025-04-03T14:30:00Z"),
        terminal_timestamp=pd.Timestamp("2025-04-03T15:30:00Z"),
        restarted_horizon_bars=6,
    )

    assert result.constant_terminal_exit_timestamp == pd.Timestamp("2025-04-03T15:30:00Z")
    assert result.restarted_exit_timestamp == pd.Timestamp("2025-04-03T15:00:00Z")
    assert result.constant_terminal_net_bps != result.restarted_net_bps


def test_checkpoint_after_terminal_is_unavailable_not_zero() -> None:
    result = remaining_payoff(
        _bars(),
        direction=1,
        checkpoint_timestamp=pd.Timestamp("2025-04-03T16:30:00Z"),
        terminal_timestamp=pd.Timestamp("2025-04-03T16:25:00Z"),
    )

    assert result.status == "too_late"
    assert result.constant_terminal_net_bps is None
    assert result.restarted_net_bps is None


def test_missing_price_path_remains_missing_and_costs_are_charged() -> None:
    missing = remaining_payoff(
        pd.DataFrame(),
        direction=1,
        checkpoint_timestamp=pd.Timestamp("2023-04-03T14:40:00Z"),
        terminal_timestamp=pd.Timestamp("2023-04-03T16:25:00Z"),
    )
    stressed = remaining_payoff(
        _bars(),
        direction=1,
        checkpoint_timestamp=pd.Timestamp("2025-04-03T14:40:00Z"),
        terminal_timestamp=pd.Timestamp("2025-04-03T16:25:00Z"),
        cost_bps_per_side=10.0,
    )

    assert missing.status == "missing_source_data"
    assert missing.constant_terminal_net_bps is None
    assert math.isclose(
        stressed.constant_terminal_net_bps,
        stressed.constant_terminal_gross_bps - 20.0,
    )
