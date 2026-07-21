from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.hidden_loop_economics_registered_bridge_v0 import (
    benjamini_hochberg,
    bridge_feature_sets,
    choose_primary_decision,
    cohort_relative_signed_return_bps,
    completion_momentum_direction,
    deduplicate_hidden_events,
    eligible_matched_controls,
    expanding_logistic_crossfit,
    net_after_friction_bps,
    opening_pressure_direction,
    opposite_opening_pressure_direction,
    permute_feature_within_slates,
    registered_completion_targets,
    registered_loop_bridge_target,
    reject_protected_dates,
    score_event_horizons,
    session_block_bootstrap_indices,
    stock_clock_session_permutation,
)


def test_deduplication_uses_latest_strictly_eligible_source_checkpoint() -> None:
    completion = pd.Timestamp("2025-01-02T15:45:00Z")
    events = pd.DataFrame(
        {
            "symbol": ["AAL", "AAL", "AAL"],
            "session": ["2025-01-02"] * 3,
            "event_timestamp_utc": [completion] * 3,
            "family_id": ["unregistered_primitive_like__5-6-5"] * 3,
            "decision_ordinal": [6, 12, 24],
            "decision_timestamp_utc": [
                pd.Timestamp("2025-01-02T15:00:00Z"),
                pd.Timestamp("2025-01-02T15:30:00Z"),
                completion,
            ],
        }
    )

    deduplicated = deduplicate_hidden_events(events)

    assert len(deduplicated) == 1
    assert int(deduplicated.iloc[0]["decision_ordinal"]) == 12


def test_predecessor_completion_availability_is_the_causal_dedup_cutoff() -> None:
    completion_bar_start = pd.Timestamp("2025-01-02T15:30:00Z")
    events = pd.DataFrame(
        {
            "symbol": ["AAL", "AAL"],
            "session": ["2025-01-02", "2025-01-02"],
            "event_timestamp_utc": [completion_bar_start, completion_bar_start],
            "event_available_timestamp_utc": [completion_bar_start + pd.Timedelta(minutes=5)] * 2,
            "family_id": ["unregistered_primitive_like__5-6-5"] * 2,
            "decision_ordinal": [6, 12],
            "decision_timestamp_utc": [
                pd.Timestamp("2025-01-02T15:00:00Z"),
                completion_bar_start,
            ],
        }
    )

    deduplicated = deduplicate_hidden_events(events)

    assert int(deduplicated.iloc[0]["decision_ordinal"]) == 12


def test_economic_timing_uses_next_bar_open_and_sixth_and_twelfth_closes() -> None:
    timestamps = pd.date_range("2025-01-02T14:30:00Z", periods=16, freq="5min")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + index for index in range(16)],
            "high": [101.0 + index for index in range(16)],
            "low": [99.0 + index for index in range(16)],
            "close": [100.5 + index for index in range(16)],
        }
    )

    scored = score_event_horizons(
        bars,
        completion_timestamp=timestamps[1],
        direction=1,
        horizons=(6, 12),
    ).set_index("horizon_bars")

    assert scored.loc[6, "entry_timestamp_utc"] == timestamps[2]
    assert scored.loc[6, "entry_price"] == 102.0
    assert scored.loc[6, "exit_timestamp_utc"] == timestamps[7] + pd.Timedelta(minutes=5)
    assert scored.loc[6, "exit_price"] == 107.5
    assert scored.loc[12, "exit_timestamp_utc"] == timestamps[13] + pd.Timedelta(minutes=5)
    assert scored.loc[12, "exit_price"] == 113.5


def test_fixed_direction_conventions_do_not_use_post_completion_returns() -> None:
    assert opening_pressure_direction(0.25) == 1
    assert opening_pressure_direction(-0.25) == -1
    assert opening_pressure_direction(0.0) is None
    assert opposite_opening_pressure_direction(1) == -1
    assert completion_momentum_direction(30.0, [10.0, 20.0]) == 1
    assert completion_momentum_direction(10.0, [10.0, 20.0]) == -1


def test_cohort_relative_return_and_twenty_basis_point_friction() -> None:
    relative = cohort_relative_signed_return_bps(
        stock_raw_return_bps=40.0,
        other_stock_raw_returns_bps=[10.0, 20.0, 30.0],
        direction=-1,
    )

    assert relative == -20.0
    assert net_after_friction_bps(35.0, friction_bps=20.0) == 15.0


def test_same_time_matched_controls_use_only_causal_eligible_stocks() -> None:
    decision = pd.Timestamp("2025-01-02T15:00:00Z")
    completion = pd.Timestamp("2025-01-02T15:15:00Z")
    candidates = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F", "G"],
            "signed_pressure": [1.0, -1.0, 0.0, 1.0, 1.0, 1.0, -1.0],
            "entry_price": [10.0, 10.0, 10.0, 10.0, 0.0, 10.0, 10.0],
            "exit_price": [11.0, 9.0, 11.0, 11.0, 11.0, 12.0, 9.5],
        }
    )
    hidden = pd.DataFrame(
        {
            "symbol": ["D", "G"],
            "event_timestamp_utc": [
                pd.Timestamp("2025-01-02T15:10:00Z"),
                pd.Timestamp("2025-01-02T15:20:00Z"),
            ],
        }
    )

    controls = eligible_matched_controls(
        candidates,
        hidden,
        focal_symbol="A",
        decision_timestamp=decision,
        completion_timestamp=completion,
    )

    assert controls["symbol"].tolist() == ["B", "F", "G"]
    assert controls.set_index("symbol").loc["B", "direction"] == -1


def test_matched_control_excludes_event_completing_after_decision() -> None:
    decision = pd.Timestamp("2025-01-02T15:00:00Z")
    candidates = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "signed_pressure": [1.0, 1.0, -1.0],
            "entry_price": [10.0, 10.0, 10.0],
            "exit_price": [11.0, 11.0, 9.0],
        }
    )
    hidden = pd.DataFrame(
        {
            "symbol": ["B"],
            "event_timestamp_utc": [decision],
            "event_available_timestamp_utc": [decision + pd.Timedelta(minutes=5)],
        }
    )

    controls = eligible_matched_controls(
        candidates,
        hidden,
        focal_symbol="A",
        decision_timestamp=decision,
        completion_timestamp=decision + pd.Timedelta(minutes=10),
    )

    assert controls["symbol"].tolist() == ["C"]


def test_matched_control_excludes_event_completed_before_decision() -> None:
    decision = pd.Timestamp("2025-01-02T15:00:00Z")
    candidates = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "signed_pressure": [1.0, 1.0, -1.0],
            "entry_price": [10.0, 10.0, 10.0],
            "exit_price": [11.0, 11.0, 9.0],
        }
    )
    hidden = pd.DataFrame(
        {
            "symbol": ["B"],
            "event_timestamp_utc": [decision - pd.Timedelta(minutes=10)],
            "event_available_timestamp_utc": [decision - pd.Timedelta(minutes=5)],
        }
    )

    controls = eligible_matched_controls(
        candidates,
        hidden,
        focal_symbol="A",
        decision_timestamp=decision,
        completion_timestamp=decision + pd.Timedelta(minutes=10),
    )

    assert controls["symbol"].tolist() == ["C"]


def test_hidden_to_registered_targets_are_strict_and_use_six_and_twelve_bars() -> None:
    completions = pd.DataFrame(
        {
            "completion_bar_ordinal": [10, 16, 22],
            "semantic_loop_id": ["same_bar", "loop_p_0-1-0", "loop_r2_0-1-0"],
            "motif_type": ["primitive", "primitive", "repeat"],
        }
    )

    targets = registered_completion_targets(10, completions)

    assert targets["registered_within_6_bars"] is True
    assert targets["registered_within_12_bars"] is True
    assert targets["bars_to_first_registered_completion"] == 6
    assert targets["first_registered_semantic_loop_id"] == "loop_p_0-1-0"


def test_registered_loop_bridge_target_allows_hidden_then_registered_sequence() -> None:
    completions = pd.DataFrame(
        {
            "completion_bar_ordinal": [11, 23, 24],
            "semantic_loop_id": ["same_bar", "within_twelve", "too_late"],
            "motif_type": ["primitive"] * 3,
        }
    )

    assert registered_loop_bridge_target(11, completions) == 1
    assert registered_loop_bridge_target(24, completions) == 0


def test_session_bootstrap_retains_every_row_from_sampled_sessions() -> None:
    frame = pd.DataFrame({"session": ["A", "A", "B", "C", "C", "C"], "value": range(6)})

    draws = session_block_bootstrap_indices(frame, draws=3, seed=7)

    assert len(draws) == 3
    for indices in draws:
        sampled = frame.iloc[indices]
        for session, count in sampled["session"].value_counts().items():
            assert count % int(frame["session"].eq(session).sum()) == 0


def test_benjamini_hochberg_adjusts_exactly_four_family_hypotheses() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])

    assert adjusted == [0.04, 0.05333333333333334, 0.05333333333333334, 0.20]


def test_bridge_null_permutation_stays_within_each_slate() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["A", "A", "A", "B", "B"],
            "hidden_probability": [0.1, 0.2, 0.3, 0.7, 0.8],
            "target": [0, 1, 0, 1, 1],
        }
    )

    permuted = permute_feature_within_slates(frame, feature="hidden_probability", seed=11)

    assert permuted["target"].tolist() == frame["target"].tolist()
    for slate in ("A", "B"):
        before = sorted(frame.loc[frame["slate_id"].eq(slate), "hidden_probability"])
        after = sorted(permuted.loc[permuted["slate_id"].eq(slate), "hidden_probability"])
        assert before == after


def test_protected_date_is_rejected_before_analysis() -> None:
    with pytest.raises(ValueError, match="protected"):
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))


def test_expanding_u1_crossfit_trains_only_on_earlier_sessions() -> None:
    sessions = [f"2024-01-{day:02d}" for day in range(1, 11) for _ in range(2)]
    frame = pd.DataFrame(
        {
            "session": sessions,
            "x": [float(index % 4) for index in range(20)],
            "target": [index % 2 for index in range(20)],
            "row_weight": [0.5] * 20,
        }
    )

    predictions, manifest = expanding_logistic_crossfit(
        frame,
        features=("x",),
        target="target",
        folds=4,
        warmup_fraction=0.2,
    )

    assert predictions.iloc[:4].isna().all()
    assert predictions.iloc[4:].notna().all()
    assert len(manifest) == 4
    assert all(manifest["train_session_end"] < manifest["prediction_session_start"])


def test_bridge_feature_construction_adds_only_frozen_hidden_probability() -> None:
    b0, b1 = bridge_feature_sets(("state_p_0", "transition_probability", "checkpoint_12"))

    assert b0 == ("state_p_0", "transition_probability", "checkpoint_12")
    assert b1 == (*b0, "p_unregistered_within_6_bars")


def test_stock_and_clock_lead_null_preserves_family_counts_and_clock() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "session": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "clock_bin": ["10:00", "10:00", "10:00"],
            "hidden_family_class": ["F1", "F1", "F2"],
            "completion_bar_ordinal": [8, 9, 10],
        }
    )
    eligible = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "session": [f"2025-01-0{day}" for day in range(1, 6)],
        }
    )

    permuted = stock_clock_session_permutation(events, eligible, seed=19)

    assert permuted["clock_bin"].tolist() == events["clock_bin"].tolist()
    assert permuted["hidden_family_class"].value_counts().to_dict() == {"F1": 2, "F2": 1}
    assert set(permuted["session"]).issubset(set(eligible["session"]))


def test_decision_logic_keeps_economic_and_bridge_results_independent() -> None:
    assert (
        choose_primary_decision(
            economic_status="supported",
            registered_lead_status="not_supported",
            predictive_bridge_status="not_supported",
        )
        == "hidden_loop_economic_consequence_only"
    )
    assert (
        choose_primary_decision(
            economic_status="not_supported",
            registered_lead_status="not_supported",
            predictive_bridge_status="supported",
        )
        == "hidden_loop_registered_bridge_only"
    )
