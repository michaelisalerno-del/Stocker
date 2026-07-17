from __future__ import annotations

import math

import pandas as pd
import pytest

from stocker_research.frozen_named_loop_t0_execution import (
    FILL_STRESSES_BPS,
    PRIMARY_FILL_MODEL,
    FillEvidence,
    TriggerType,
    apply_adverse_entry_slippage,
    family_spec,
    gross_payoff_bps,
    reconstruct_frozen_oco_trigger,
    score_fill_envelope,
)

ANCHOR = pd.Timestamp("2026-07-17T13:30:00Z")


def bars(*rows: tuple[str, float, float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])


def test_named_and_control_family_mappings_are_exact_and_frozen() -> None:
    expected = {
        "cycle_04|state_4": ("named", "2->4->2", 4, "named_candidate"),
        "cycle_04|state_2": ("control", "2->4->2", 2, "neutral_control"),
        "cycle_07|state_5": ("named", "5->6->5", 5, "named_candidate"),
        "cycle_07|state_6": ("control", "5->6->5", 6, "negative_control"),
    }

    assert {
        key: (value.classification, value.cycle, value.current_state, value.role)
        for key in expected
        if (value := family_spec(key))
    } == expected


def test_failed_family_cannot_be_replaced() -> None:
    with pytest.raises(ValueError, match="not frozen"):
        family_spec("cycle_08|state_1")


def test_original_intrabar_threshold_fill_is_reconstructed_exactly_but_bounded() -> None:
    result = reconstruct_frozen_oco_trigger(
        bars(("2026-07-17T13:35:00Z", 100.0, 102.0, 99.0, 101.5)),
        anchor_timestamp=ANCHOR,
        long_threshold=102.0,
        short_threshold=98.0,
        horizon_bars=1,
    )

    assert result.direction == 1
    assert result.entry_step == 1
    assert result.trigger_type is TriggerType.INTRABAR_THRESHOLD_CROSS
    assert result.reference_entry_price == 102.0
    assert result.reference_entry_timestamp == pd.Timestamp("2026-07-17T13:35:00Z")
    assert result.fill_evidence is FillEvidence.BOUNDED_BUT_NOT_EXACT
    assert result.signal_known_timestamp is None
    assert result.market_data_availability_timestamp == pd.Timestamp("2026-07-17T13:40:00Z")
    assert result.signal_fill_time_status == "SIGNAL_OR_FILL_TIME_AMBIGUOUS"


def test_opening_gap_long_uses_observed_open_and_is_causally_timestamped() -> None:
    result = reconstruct_frozen_oco_trigger(
        bars(("2026-07-17T13:35:00Z", 103.0, 104.0, 100.0, 101.0)),
        anchor_timestamp=ANCHOR,
        long_threshold=102.0,
        short_threshold=98.0,
        horizon_bars=1,
    )

    assert result.direction == 1
    assert result.reference_entry_price == 103.0
    assert result.trigger_type is TriggerType.OPENING_GAP_THROUGH_THRESHOLD
    assert result.fill_evidence is FillEvidence.GAP_FILL_OBSERVABLE
    assert result.signal_known_timestamp == result.reference_entry_timestamp


def test_opening_gap_short_uses_observed_open() -> None:
    result = reconstruct_frozen_oco_trigger(
        bars(("2026-07-17T13:35:00Z", 97.0, 100.0, 96.0, 99.0)),
        anchor_timestamp=ANCHOR,
        long_threshold=102.0,
        short_threshold=98.0,
        horizon_bars=1,
    )

    assert result.direction == -1
    assert result.reference_entry_price == 97.0
    assert result.fill_evidence is FillEvidence.GAP_FILL_OBSERVABLE


def test_ambiguous_oco_cross_is_explicit_and_never_selects_future_direction() -> None:
    result = reconstruct_frozen_oco_trigger(
        bars(("2026-07-17T13:35:00Z", 100.0, 103.0, 97.0, 101.0)),
        anchor_timestamp=ANCHOR,
        long_threshold=102.0,
        short_threshold=98.0,
        horizon_bars=1,
    )

    assert result.direction is None
    assert result.reference_entry_price is None
    assert result.fill_evidence is FillEvidence.AMBIGUOUS_WITHIN_BAR_ORDER
    assert result.trigger_type is TriggerType.AMBIGUOUS_DUAL_OCO_CROSS


def test_missing_exact_trigger_bar_remains_unavailable_not_zero() -> None:
    result = reconstruct_frozen_oco_trigger(
        bars(("2026-07-17T13:40:00Z", 100.0, 101.0, 99.0, 100.0)),
        anchor_timestamp=ANCHOR,
        long_threshold=102.0,
        short_threshold=98.0,
        horizon_bars=1,
    )

    assert result.fill_evidence is FillEvidence.MISSING_MARKET_DATA
    assert result.reference_entry_price is None


@pytest.mark.parametrize(
    ("direction", "stress_bps", "expected"),
    [(1, 5.0, 100.05), (1, 20.0, 100.2), (-1, 5.0, 99.95), (-1, 20.0, 99.8)],
)
def test_adverse_entry_slippage_worsens_long_and_short_correctly(
    direction: int, stress_bps: float, expected: float
) -> None:
    assert apply_adverse_entry_slippage(100.0, direction, stress_bps) == pytest.approx(expected)


def test_fill_envelope_is_non_cumulative_and_keeps_identity_direction_and_terminal() -> None:
    rows = score_fill_envelope(
        opportunity_id="opp-1",
        direction=1,
        reference_entry_price=100.0,
        terminal_timestamp=pd.Timestamp("2026-07-17T15:35:00Z"),
        terminal_price=101.0,
        cost_bps=10.0,
    )

    assert [row.fill_model for row in rows] == ["F0", "F5", "F10", "F15", "F20"]
    assert [row.stressed_entry_price for row in rows] == pytest.approx(
        [100.0, 100.05, 100.1, 100.15, 100.2]
    )
    assert {row.opportunity_id for row in rows} == {"opp-1"}
    assert {row.direction for row in rows} == {1}
    assert {row.terminal_timestamp for row in rows} == {pd.Timestamp("2026-07-17T15:35:00Z")}


def test_existing_fixed_cost_is_not_double_counted_by_entry_stress() -> None:
    row = score_fill_envelope(
        opportunity_id="opp-1",
        direction=1,
        reference_entry_price=100.0,
        terminal_timestamp=pd.Timestamp("2026-07-17T15:35:00Z"),
        terminal_price=101.0,
        cost_bps=10.0,
    )[2]

    expected_gross = 10_000.0 * (101.0 / 100.1 - 1.0)
    assert row.fill_model == "F10"
    assert row.gross_payoff_bps == pytest.approx(expected_gross)
    assert row.cost_bps == 10.0
    assert row.net_payoff_bps == pytest.approx(expected_gross - 10.0)


def test_long_and_short_return_conventions_are_exact() -> None:
    assert gross_payoff_bps(1, 100.0, 101.0) == pytest.approx(100.0)
    assert gross_payoff_bps(-1, 100.0, 99.0) == pytest.approx(100.0)
    assert math.isfinite(gross_payoff_bps(-1, 99.9, 99.0))


def test_f10_is_fixed_as_primary_execution_stress() -> None:
    assert PRIMARY_FILL_MODEL == "F10"
    assert FILL_STRESSES_BPS == {"F0": 0.0, "F5": 5.0, "F10": 10.0, "F15": 15.0, "F20": 20.0}
