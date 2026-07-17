from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from stocker_research.directional_signature_atlas.features import (
    add_cross_sectional_features,
    assert_causal_feature_ledger,
    assert_outcome_free_feature_names,
    fit_training_quantile_bins,
    reconstruct_state_motifs,
)
from stocker_research.directional_signature_atlas.historical import _state_anchor_rows
from stocker_research.directional_signature_atlas.outcomes import (
    build_economic_outcome,
    classify_terminal_move,
    movement_permission,
)
from stocker_research.directional_signature_atlas.signatures import (
    Condition,
    Signature,
    complexity_penalty,
    validate_signature,
)


def _session(rows: int = 61) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-02 14:30:00+00:00", periods=rows, freq="5min")
    close = 100.0 + np.arange(rows) * 0.01
    return pd.DataFrame(
        {
            "bar_ordinal": np.arange(rows),
            "timestamp": timestamps,
            "open": close - 0.005,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
        }
    )


def _auditor_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "research/slrno-v2/20260714-regime-loop-handoff/work"
        / "audit_directional_signature_atlas_v1.py"
    )
    specification = importlib.util.spec_from_file_location("atlas_auditor_test_module", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_every_feature_timestamp_is_no_later_than_decision() -> None:
    decision = pd.Timestamp("2025-01-02 15:30:00+00:00")
    frame = pd.DataFrame(
        {
            "decision_timestamp": [decision],
            "return_1": [0.1],
            "return_1__available_at": [decision],
        }
    )
    assert_causal_feature_ledger(frame, ["return_1"])
    frame.loc[0, "return_1__available_at"] = decision + pd.Timedelta(minutes=5)
    with pytest.raises(AssertionError, match="future availability"):
        assert_causal_feature_ledger(frame, ["return_1"])


def test_future_returns_cannot_enter_feature_ledger() -> None:
    with pytest.raises(ValueError, match="forbidden causal feature"):
        assert_outcome_free_feature_names(["return_1", "future_return_24"])


def test_future_loop_or_route_identity_cannot_enter_features() -> None:
    for forbidden in (
        "realised_child_route_identity",
        "future_loop_identity",
        "realised_morph_identity",
    ):
        with pytest.raises(ValueError, match="forbidden causal feature"):
            assert_outcome_free_feature_names([forbidden])


def test_mfe_and_mae_are_absent_from_causal_features() -> None:
    with pytest.raises(ValueError, match="forbidden causal feature"):
        assert_outcome_free_feature_names(["mfe_long_bps", "mae_long_bps"])


def test_targets_and_payoffs_are_absent_from_causal_features() -> None:
    for forbidden in (
        "target",
        "payoff",
        "outcome_label",
        "first_touch_target",
        "net_long_return_bps",
        "gross_short_return_bps",
        "gross_long_payoff_bps",
        "primary_target",
        "target_label",
        "economic_outcome",
    ):
        with pytest.raises(ValueError, match="forbidden causal feature"):
            assert_outcome_free_feature_names([forbidden])


def test_exact_next_provider_open_entry_is_used() -> None:
    result = build_economic_outcome(_session(), decision_ordinal=12, round_trip_cost_bps=10.0)
    assert result["entry_timestamp"] == _session().iloc[13]["timestamp"]
    assert result["entry_open"] == pytest.approx(_session().iloc[13]["open"])


def test_fixed_24_bar_terminal_is_used() -> None:
    result = build_economic_outcome(_session(), decision_ordinal=12, round_trip_cost_bps=10.0)
    assert result["terminal_timestamp"] == _session().iloc[36]["timestamp"]
    assert result["terminal_close"] == pytest.approx(_session().iloc[36]["close"])


def test_secondary_first_touch_uses_separate_frozen_causal_barrier() -> None:
    primary_barrier = build_economic_outcome(
        _session(), decision_ordinal=12, round_trip_cost_bps=10.0
    )
    predecessor_barrier = build_economic_outcome(
        _session(),
        decision_ordinal=12,
        round_trip_cost_bps=10.0,
        first_touch_barrier_bps=100.0,
    )
    assert primary_barrier["first_touch_target"] == "UPPER_FIRST"
    assert predecessor_barrier["first_touch_target"] == "NEITHER"
    assert predecessor_barrier["first_touch_barrier_bps"] == 100.0


def test_secondary_first_touch_checks_gap_open_before_dual_touch() -> None:
    session = _session()
    entry_open = float(session.loc[session["bar_ordinal"].eq(13), "open"].iloc[0])
    upper = entry_open * 1.004
    gap_index = session.index[session["bar_ordinal"].eq(14)][0]
    session.loc[gap_index, "open"] = upper + 0.01
    session.loc[gap_index, "high"] = upper + 0.02
    session.loc[gap_index, "low"] = entry_open * 0.995
    result = build_economic_outcome(
        session,
        decision_ordinal=12,
        round_trip_cost_bps=10.0,
        first_touch_barrier_bps=40.0,
    )
    assert result["first_touch_target"] == "UPPER_FIRST"
    assert result["first_touch_step"] == 2


def test_missing_entry_or_terminal_remains_unavailable() -> None:
    missing_entry = _session().query("bar_ordinal != 13")
    missing_terminal = _session().query("bar_ordinal != 36")
    assert build_economic_outcome(missing_entry, 12, 10.0)["target"] == "UNAVAILABLE"
    unavailable = build_economic_outcome(missing_terminal, 12, 10.0, first_touch_barrier_bps=80.0)
    assert unavailable["target"] == "UNAVAILABLE"
    assert unavailable["first_touch_target"] == "UNAVAILABLE"
    assert unavailable["first_touch_barrier_bps"] == 80.0


def test_row_cannot_be_both_long_and_short() -> None:
    labels = [classify_terminal_move(value, 10.0, 2.0) for value in (-25.0, 0.0, 25.0)]
    assert labels == ["SHORT", "NEUTRAL", "LONG"]
    assert all(not (label == "LONG" and label == "SHORT") for label in labels)


def test_neutral_dead_band_uses_exact_cost_model() -> None:
    assert classify_terminal_move(20.0, 10.0, 2.0) == "NEUTRAL"
    assert classify_terminal_move(20.0001, 10.0, 2.0) == "LONG"
    assert classify_terminal_move(-20.0001, 10.0, 2.0) == "SHORT"


def test_one_two_and_three_times_cost_dead_bands_are_separate() -> None:
    move = 15.0
    assert classify_terminal_move(move, 10.0, 1.0) == "LONG"
    assert classify_terminal_move(move, 10.0, 2.0) == "NEUTRAL"
    assert classify_terminal_move(move, 10.0, 3.0) == "NEUTRAL"


def test_current_active_movement_is_not_future_direction() -> None:
    assert_outcome_free_feature_names(["predicted_remaining_range_bps", "current_bar_return"])
    with pytest.raises(ValueError, match="forbidden causal feature"):
        assert_outcome_free_feature_names(["predicted_future_direction"])


def test_movement_permission_is_direction_neutral() -> None:
    assert movement_permission(30.0001, 10.0)
    assert not movement_permission(30.0, 10.0)


def _states() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA"] * 8,
            "session": ["2025-01-02"] * 8,
            "timestamp": pd.date_range("2025-01-02 14:30:00+00:00", periods=8, freq="5min"),
            "state": [1, 1, 3, 3, 1, 2, 2, 1],
        }
    )


def test_state_motifs_use_only_states_observed_by_current_bar() -> None:
    result = reconstruct_state_motifs(_states())
    assert pd.isna(result.iloc[3]["state_motif_2"])
    assert result.iloc[3]["previous_state"] == 1
    assert "2" not in str(result.iloc[3]["state_motif_4"])


def test_motif_lengths_two_three_and_four_are_reconstructed() -> None:
    result = reconstruct_state_motifs(_states())
    last = result.iloc[-1]
    assert last["state_motif_2"] == "1>2"
    assert last["state_motif_3"] == "3>1>2"
    assert last["state_motif_4"] == "1>3>1>2"
    assert not str(last["state_motif_4"]).endswith(f">{last['state']}")


def test_future_state_cannot_enter_current_motif() -> None:
    full = reconstruct_state_motifs(_states())
    prefix = reconstruct_state_motifs(_states().iloc[:4])
    pd.testing.assert_series_equal(
        full.iloc[:4]["state_motif_3"].reset_index(drop=True),
        prefix["state_motif_3"].reset_index(drop=True),
    )


def test_ambiguous_transition_duration_is_not_aliased_from_state_dwell() -> None:
    panel = pd.DataFrame(
        {
            "symbol_norm": ["AAA"] * 4,
            "session_date": ["2025-01-02"] * 4,
            "state": [1, 1, 2, 2],
            "bar_index_in_session": [0, 1, 2, 3],
            "age": [1, 2, 1, 2],
        }
    )
    anchor = _state_anchor_rows(panel, {3}).iloc[0]
    assert anchor["prior_completed_state_dwell_bars"] == 2
    assert pd.isna(anchor["prior_completed_transition_duration_bars"])


def test_full_raw_loop_score_vectors_are_rejected() -> None:
    signature = Signature("bad", "LONG", (Condition("loop_score_01", ">", 0.5, "loop"),))
    with pytest.raises(ValueError, match="raw loop-score"):
        validate_signature(signature)


def test_candidate_signatures_have_at_most_three_conditions() -> None:
    conditions = tuple(Condition(f"f{i}", ">", 0.0, f"family{i}") for i in range(4))
    with pytest.raises(ValueError, match="three conditions"):
        validate_signature(Signature("bad", "LONG", conditions))


def test_stock_identity_cannot_be_signature_condition() -> None:
    signature = Signature("bad", "LONG", (Condition("symbol", "==", "AAA", "identity"),))
    with pytest.raises(ValueError, match="identity"):
        validate_signature(signature)


def test_month_identity_cannot_be_signature_condition() -> None:
    signature = Signature("bad", "LONG", (Condition("month", "==", "2025-01", "clock"),))
    with pytest.raises(ValueError, match="identity"):
        validate_signature(signature)


def test_outcome_episode_identity_cannot_be_signature_condition() -> None:
    signature = Signature("bad", "LONG", (Condition("hindsight_episode", "==", 1, "outcome"),))
    with pytest.raises(ValueError, match="outcome"):
        validate_signature(signature)


def test_training_quantile_bins_cannot_use_future_periods() -> None:
    frame = pd.DataFrame({"period": [2024, 2024, 2025], "value": [1.0, 2.0, 100.0]})
    edges = fit_training_quantile_bins(frame, "value", discovery_periods={2024}, bins=2)
    assert edges[-1] == pytest.approx(2.0)


def test_complexity_penalty_increases_with_conditions() -> None:
    assert complexity_penalty(1, 0.15) < complexity_penalty(2, 0.15)
    assert complexity_penalty(2, 0.15) < complexity_penalty(3, 0.15)


def test_cross_sectional_ranks_use_same_timestamp_only() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-02T15:30Z"] * 2 + ["2025-01-03T15:30Z"] * 2),
            "symbol": ["A", "B", "A", "B"],
            "return_6": [1.0, 2.0, 100.0, 0.0],
        }
    )
    result = add_cross_sectional_features(frame, ["return_6"], min_peers=2)
    assert result.loc[0, "return_6_cross_sectional_rank"] == pytest.approx(0.0)
    assert result.loc[2, "return_6_cross_sectional_rank"] == pytest.approx(1.0)


def test_missing_peer_data_does_not_become_zero_rank() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-02T15:30Z"] * 3),
            "symbol": ["A", "B", "C"],
            "return_6": [1.0, np.nan, 2.0],
        }
    )
    result = add_cross_sectional_features(frame, ["return_6"], min_peers=2)
    assert pd.isna(result.loc[1, "return_6_cross_sectional_rank"])


def test_leave_one_stock_out_recomputes_cross_sectional_features() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-02T15:30Z"] * 3),
            "symbol": ["A", "B", "C"],
            "return_6": [1.0, 2.0, 3.0],
        }
    )
    full = add_cross_sectional_features(frame, ["return_6"], min_peers=2)
    dropped = add_cross_sectional_features(
        frame.loc[frame["symbol"] != "C"].copy(), ["return_6"], min_peers=2
    )
    assert full.loc[full["symbol"] == "B", "return_6_cross_sectional_rank"].item() == 0.5
    assert dropped.loc[dropped["symbol"] == "B", "return_6_cross_sectional_rank"].item() == 1.0


def test_track_b_auditor_rejects_summary_comparison_tamper() -> None:
    auditor = _auditor_module()
    relative = pd.DataFrame(
        {
            "opportunity_id": ["v1", "v2", "f1", "f2"],
            "chronology_stage": [
                "validation",
                "validation",
                "final_opened_holdout",
                "final_opened_holdout",
            ],
            "long_net_bps": [10.0, 10.0, 8.0, 8.0],
            "short_net_bps": [-10.0, -10.0, -8.0, -8.0],
        }
    )
    atlas = pd.DataFrame(
        {
            "opportunity_id": relative["opportunity_id"],
            "predicted_state": "LONG",
        }
    )
    baseline = pd.DataFrame(
        {
            "opportunity_id": relative["opportunity_id"],
            "predicted_state": "NEUTRAL",
        }
    )
    valid, _ = auditor.verify_relative_atlas_baseline_comparison(
        relative,
        atlas,
        baseline,
        {"atlas_beats_relative_strength_validation_and_final": True},
    )
    tampered, detail = auditor.verify_relative_atlas_baseline_comparison(
        relative,
        atlas,
        baseline,
        {"atlas_beats_relative_strength_validation_and_final": False},
    )
    assert valid
    assert not tampered
    assert "summary_comparison_tamper" in detail
