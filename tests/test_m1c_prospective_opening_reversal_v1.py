from __future__ import annotations

import hashlib
import inspect
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import stocker_prospective.m1c_prospective_opening_reversal_v1 as experiment_module
from stocker_prospective.direction_features import DirectionFeatureBar
from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID,
    M1C_PROSPECTIVE_OPENING_REVERSAL_V1_VERSION,
    NEGATIVE_OPENING_RETURN_THRESHOLD_V1,
    OPENING_RANGE_THRESHOLD_V1,
    POSITIVE_OPENING_RETURN_THRESHOLD_V1,
    FrozenOpeningReversalRuleV1,
    OpeningEventAccountingRowV1,
    OpeningReversalPredictionInputV1,
    OpeningTransferBarV1,
    OpeningTransferOperationalEvidenceV1,
    OpeningTransferSessionResultV1,
    PostEntryBarV1,
    build_frozen_experiment_config_v1,
    build_incomplete_opening_reversal_outcome_v1,
    build_opening_reversal_outcome_v1,
    build_opening_transfer_decision_receipt_v1,
    build_prediction_receipt_v1,
    evaluate_opening_transfer_session_v1,
    load_frozen_experiment_config_v1,
    partition_material_outcome_v1,
    reconcile_opening_event_accounting_v1,
    select_promoted_prediction_v1,
)
from stocker_prospective.recorder_v0 import (
    _existing_clean_market_direction_baseline_sign_v1,
)

SESSION = date(2026, 7, 30)
SESSION_OPEN = datetime(2026, 7, 30, 13, 30, tzinfo=UTC)
SIGNAL_AND_ENTRY = SESSION_OPEN + timedelta(minutes=30)
ACTIVATION = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _direction_bar(ordinal: int, market_log_return: float) -> DirectionFeatureBar:
    start = SESSION_OPEN + timedelta(minutes=5 * ordinal)
    return DirectionFeatureBar(
        symbol="AAL",
        session=SESSION,
        bar_ordinal=ordinal,
        bar_start_timestamp=start,
        bar_complete_timestamp=start + timedelta(minutes=5),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1_000.0,
        historical_relative_activity=1.0,
        stock_log_return=0.0,
        market_log_return=market_log_return,
        finalised=True,
    )


def _passing_operational_evidence() -> OpeningTransferOperationalEvidenceV1:
    return OpeningTransferOperationalEvidenceV1(
        prediction_receipt_count=20,
        prediction_receipt_timing_pass=True,
        prediction_receipt_immutability_pass=True,
        capacity_snapshot_count=20,
        capacity_snapshots_complete=True,
        reserved_twelve_lines_pass=True,
        promoted_episode_count=1,
        promoted_underlying_level1_pass=True,
        contract_discovery_audit_count=1,
        contract_discovery_complete=True,
        primary_option_pair_available_count=1,
        primary_option_pair_recording_pass=True,
        graceful_degradation_pass=True,
        cancellation_recovery_pass=True,
        m1c_universe_uninterrupted=True,
        recorder_reliability_pass=True,
        no_order_guard_pass=True,
        critical_checks_pass=True,
        missing_reasons=(),
    )


def _hash_without(payload: dict[str, object], field: str) -> str:
    value = {key: item for key, item in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_clean_market_baseline_preserves_inherited_t_minus_one_window() -> None:
    bars = (
        _direction_bar(3, -0.003),
        _direction_bar(4, 0.001),
        # Ordinal 5 is intentionally excluded by the inherited definition.
        _direction_bar(5, 0.100),
    )

    assert _existing_clean_market_direction_baseline_sign_v1(bars) == -1


def _prediction_input(
    *,
    stock: str = "AAL",
    probability: float = 0.70,
    state: str = "NEGATIVE_SEVERE_OPENING_TRANSITION",
    sign: int | None = -1,
    receipt_created_at: datetime | None = None,
    phase: str = "engineering_transfer",
    comparisons: dict[str, object] | None = None,
) -> OpeningReversalPredictionInputV1:
    opening_return = (
        0.004
        if state == "POSITIVE_SEVERE_OPENING_TRANSITION"
        else -0.004
        if state == "NEGATIVE_SEVERE_OPENING_TRANSITION"
        else 0.0
    )
    opening_range = (
        0.002 if state == "NORMAL_OPENING" else 0.006
    )
    return OpeningReversalPredictionInputV1(
        activation_timestamp_utc=ACTIVATION,
        cohort_phase=phase,
        transfer_status="opening_transfer_session_valid",
        session=SESSION,
        stock=stock,
        checkpoint=6,
        signal_timestamp_utc=SIGNAL_AND_ENTRY,
        entry_timestamp_utc=SIGNAL_AND_ENTRY,
        receipt_created_at_utc=(
            SIGNAL_AND_ENTRY - timedelta(microseconds=1)
            if receipt_created_at is None
            else receipt_created_at
        ),
        m1c_probability=probability,
        m1c_probability_valid=True,
        high_tail_membership=True,
        fresh_episode_id=f"fresh-{stock}",
        canonical_fresh_episode=True,
        tail_phase_v1="FIRST_ENTRY",
        market_opening_return_v1=opening_return,
        market_opening_range_v1=opening_range,
        opening_market_transition_state_v1=state,
        opening_transition_sign_v1=sign,
        opening_transition_event_id_v1="opening-event-1",
        vti_opening_transition_complete=True,
        stock_causal_data_complete=True,
        previous_close_atm_iv_scale_15m=0.01,
        previous_close_atm_iv_scale_valid=True,
        data_source="ibkr",
        capacity_snapshot_id="capacity-1",
        frozen_comparisons={} if comparisons is None else comparisons,
    )


def test_frozen_identity_and_thresholds_are_exact() -> None:
    rule = FrozenOpeningReversalRuleV1()

    assert M1C_PROSPECTIVE_OPENING_REVERSAL_V1_ID == (
        "m1c-prospective-opening-reversal-v1"
    )
    assert M1C_PROSPECTIVE_OPENING_REVERSAL_V1_VERSION == "1"
    assert rule.checkpoint == 6
    assert rule.m1c_probability_threshold == 0.488333710794033
    assert rule.market_proxy == "VTI"
    assert rule.opening_return_q10 == -0.00288963733897
    assert rule.opening_return_q90 == 0.00225522676046
    assert rule.opening_range_q75 == 0.00384818171835
    assert NEGATIVE_OPENING_RETURN_THRESHOLD_V1 == -0.00288963733897
    assert POSITIVE_OPENING_RETURN_THRESHOLD_V1 == 0.00225522676046
    assert OPENING_RANGE_THRESHOLD_V1 == 0.00384818171835
    assert rule.reserved_market_data_lines == 12
    assert not rule.order_routing_enabled
    assert not rule.order_methods_available


def test_experiment_module_has_no_order_or_contaminated_feature_path() -> None:
    source = inspect.getsource(experiment_module)

    for forbidden in (
        "placeOrder(",
        ".place_order(",
        ".submit_order(",
        "contaminated_pressure",
        "contaminated_tension",
        "peer_slate_normalization",
        "peer_slate_normalisation",
    ):
        assert forbidden not in source


def test_frozen_configuration_hash_survives_json_round_trip(
    tmp_path: Path,
) -> None:
    config = build_frozen_experiment_config_v1()
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_frozen_experiment_config_v1(str(path))

    assert loaded == config
    assert config.configuration_hash == _hash_without(
        config.model_dump(mode="json"),
        "configuration_hash",
    )


@pytest.mark.parametrize(
    ("state", "sign", "prediction", "prediction_sign"),
    [
        ("NEGATIVE_SEVERE_OPENING_TRANSITION", -1, "CALL", 1),
        ("POSITIVE_SEVERE_OPENING_TRANSITION", 1, "PUT", -1),
        ("NORMAL_OPENING", None, "ABSTAIN", 0),
        ("ELEVATED_OPENING_RANGE_NONDIRECTIONAL", None, "ABSTAIN", 0),
        ("UNKNOWN_INCOMPLETE", None, "ABSTAIN", 0),
    ],
)
def test_prediction_is_exact_opposition_or_abstain(
    state: str,
    sign: int | None,
    prediction: str,
    prediction_sign: int,
) -> None:
    receipt = build_prediction_receipt_v1(
        _prediction_input(state=state, sign=sign)
    )

    assert receipt.prediction_v1 == prediction
    assert receipt.prediction_sign_v1 == prediction_sign
    if sign is not None and receipt.eligibility_v1:
        assert receipt.prediction_sign_v1 == -sign


@pytest.mark.parametrize(
    ("state", "opening_range"),
    [
        ("NORMAL_OPENING", 0.002),
        ("ELEVATED_OPENING_RANGE_NONDIRECTIONAL", 0.006),
    ],
)
def test_complete_nondirectional_opening_abstains_without_claiming_missingness(
    state: str,
    opening_range: float,
) -> None:
    item = _prediction_input(state=state, sign=None).model_copy(
        update={"market_opening_range_v1": opening_range}
    )

    receipt = build_prediction_receipt_v1(item)

    assert not receipt.eligibility_v1
    assert receipt.prediction_v1 == "ABSTAIN"
    assert receipt.completeness_status_v1 == "complete"
    assert receipt.ineligibility_reasons_v1 == ("opening_state_not_severe",)


def test_non_frozen_comparisons_cannot_change_prediction_or_promotion() -> None:
    baseline = build_prediction_receipt_v1(_prediction_input())
    changed = build_prediction_receipt_v1(
        _prediction_input(
            comparisons={
                "a1_action_v1": "PUT",
                "rsi": 99.9,
                "stock_opening_response_class_v1": "AMPLIFYING",
                "recent_momentum": -1.0,
                "option_spread": 100.0,
                "apparent_chart_quality": "perfect",
            }
        )
    )

    assert baseline.prediction_v1 == changed.prediction_v1 == "CALL"
    assert baseline.prediction_sign_v1 == changed.prediction_sign_v1 == 1
    assert baseline.eligibility_v1 == changed.eligibility_v1


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"checkpoint": 8}, "checkpoint_not_6"),
        ({"m1c_probability": 0.48}, "m1c_below_frozen_high_tail"),
        ({"m1c_probability_valid": False}, "m1c_probability_invalid"),
        ({"canonical_fresh_episode": False}, "canonical_fresh_episode_required"),
        ({"tail_phase_v1": "PERSISTENT"}, "tail_phase_not_first_entry"),
        (
            {"vti_opening_transition_complete": False},
            "vti_opening_transition_incomplete",
        ),
        ({"stock_causal_data_complete": False}, "stock_causal_data_incomplete"),
        (
            {"previous_close_atm_iv_scale_valid": False},
            "previous_close_atm_iv_scale_invalid",
        ),
        ({"transfer_status": "UNKNOWN_INCOMPLETE"}, "transfer_status_incomplete"),
    ],
)
def test_every_frozen_eligibility_guard_fails_closed(
    update: dict[str, object],
    reason: str,
) -> None:
    item = _prediction_input().model_copy(update=update)
    receipt = build_prediction_receipt_v1(item)

    assert not receipt.eligibility_v1
    assert receipt.prediction_v1 == "ABSTAIN"
    assert reason in receipt.ineligibility_reasons_v1


def test_receipt_must_be_completed_strictly_before_entry() -> None:
    receipt = build_prediction_receipt_v1(
        _prediction_input(receipt_created_at=SIGNAL_AND_ENTRY)
    )

    assert not receipt.eligibility_v1
    assert receipt.prediction_v1 == "ABSTAIN"
    assert "receipt_not_completed_before_entry" in receipt.ineligibility_reasons_v1


def test_pre_activation_session_is_rejected() -> None:
    item = _prediction_input().model_copy(
        update={
            "session": ACTIVATION.date(),
            "signal_timestamp_utc": ACTIVATION - timedelta(minutes=1),
            "entry_timestamp_utc": ACTIVATION,
            "receipt_created_at_utc": ACTIVATION - timedelta(microseconds=1),
        }
    )

    receipt = build_prediction_receipt_v1(item)

    assert not receipt.eligibility_v1
    assert "session_not_after_activation" in receipt.ineligibility_reasons_v1


def test_engineering_prediction_is_recorded_but_not_scientific_outcome_evidence() -> None:
    receipt = build_prediction_receipt_v1(_prediction_input())

    assert receipt.eligibility_v1
    assert receipt.prediction_v1 == "CALL"
    assert not receipt.scientific_outcome_eligible_v1
    assert receipt.scientific_exclusion_reason_v1 == "engineering_transfer"


def test_prediction_receipt_is_frozen_and_self_hashing() -> None:
    comparisons: dict[str, object] = {"a1_action_v1": "CALL"}
    item = _prediction_input(comparisons=comparisons)
    receipt = build_prediction_receipt_v1(item)
    comparisons["a1_action_v1"] = "PUT"

    with pytest.raises((ValidationError, TypeError)):
        receipt.prediction_v1 = "PUT"  # type: ignore[misc]
    with pytest.raises(TypeError):
        receipt.frozen_comparisons[0][1] = "PUT"  # type: ignore[index]
    assert dict(receipt.frozen_comparisons)["a1_action_v1"] == "CALL"
    assert len(receipt.receipt_hash_v1) == 64
    assert build_prediction_receipt_v1(item) == receipt
    tampered = receipt.model_dump(mode="python")
    tampered["stock"] = "MSFT"
    with pytest.raises(ValidationError, match="receipt hash mismatch"):
        type(receipt).model_validate(tampered)


def test_promotion_is_probability_then_receipt_time_then_ticker() -> None:
    a = build_prediction_receipt_v1(_prediction_input(stock="AAL", probability=0.80))
    b = build_prediction_receipt_v1(_prediction_input(stock="AAOI", probability=0.90))
    c = build_prediction_receipt_v1(_prediction_input(stock="APLD", probability=0.90))
    selection = select_promoted_prediction_v1((c, a, b))

    assert selection.promoted is not None
    assert selection.promoted.stock == "AAOI"
    assert [row.stock for row in selection.non_promoted] == ["APLD", "AAL"]
    assert {row.winning_promoted_stock for row in selection.non_promoted} == {"AAOI"}
    assert {
        row.reason_not_promoted_v1 for row in selection.non_promoted
    } == {"lower_frozen_promotion_rank"}


def _post_entry_bars(final_close: float) -> tuple[PostEntryBarV1, ...]:
    values = ((100.0, 101.0, 99.0, 100.2), (100.2, 101.2, 99.8, 100.5))
    output = [
        PostEntryBarV1(
            ordinal=index,
            bar_start_timestamp_utc=SIGNAL_AND_ENTRY
            + timedelta(minutes=5 * index),
            bar_complete_timestamp_utc=SIGNAL_AND_ENTRY
            + timedelta(minutes=5 * (index + 1)),
            open=open_,
            high=high,
            low=low,
            close=close,
            finalised=True,
        )
        for index, (open_, high, low, close) in enumerate(values)
    ]
    output.append(
        PostEntryBarV1(
            ordinal=2,
            bar_start_timestamp_utc=SIGNAL_AND_ENTRY + timedelta(minutes=10),
            bar_complete_timestamp_utc=SIGNAL_AND_ENTRY + timedelta(minutes=15),
            open=100.5,
            high=max(101.5, final_close),
            low=min(99.5, final_close),
            close=final_close,
            finalised=True,
        )
    )
    return tuple(output)


@pytest.mark.parametrize(
    ("signed_return", "expected"),
    [
        (0.01, "NO_MATERIAL_MOVE"),
        (-0.01, "NO_MATERIAL_MOVE"),
        (math.nextafter(0.01, math.inf), "MATERIAL_UP"),
        (math.nextafter(-0.01, -math.inf), "MATERIAL_DOWN"),
    ],
)
def test_material_partition_keeps_exact_threshold_equality_as_no_move(
    signed_return: float,
    expected: str,
) -> None:
    assert partition_material_outcome_v1(signed_return, 0.01) == expected


def test_outcome_is_opposition_aligned_and_does_not_mutate_prediction() -> None:
    receipt = build_prediction_receipt_v1(
        _prediction_input(phase="prospective_development")
    )
    receipt_dump = receipt.model_dump(mode="json")
    outcome = build_opening_reversal_outcome_v1(
        prediction_receipt=receipt,
        completed_post_entry_bars=_post_entry_bars(98.0),
        threshold_15m=0.01,
        outcome_created_at_utc=SIGNAL_AND_ENTRY + timedelta(minutes=16),
    )

    assert outcome.r_15m < 0.0
    assert outcome.opening_reversal_aligned_return_v1 < 0.0
    assert outcome.correct_predicted_material_direction_v1 is False
    assert receipt.model_dump(mode="json") == receipt_dump
    tampered = outcome.model_dump(mode="python")
    tampered["maximum_favourable_excursion_v1"] = 9.0
    with pytest.raises(ValidationError, match="outcome receipt hash mismatch"):
        type(outcome).model_validate(tampered)


def test_incomplete_outcome_records_missingness_without_inventing_a_return() -> None:
    receipt = build_prediction_receipt_v1(
        _prediction_input(phase="prospective_development")
    )

    outcome = build_incomplete_opening_reversal_outcome_v1(
        prediction_receipt=receipt,
        missing_reason_v1="post_entry_bar_missing",
        outcome_created_at_utc=SIGNAL_AND_ENTRY + timedelta(minutes=16),
    )

    assert outcome.outcome_completeness_v1 == "incomplete"
    assert outcome.missing_reason_v1 == "post_entry_bar_missing"
    assert outcome.r_15m is None
    assert outcome.opening_reversal_aligned_return_v1 is None
    assert outcome.correct_predicted_material_direction_v1 is None
    assert outcome.exceed_iv_v1 is None


def test_event_accounting_has_one_sign_per_event_and_distinguishes_populations() -> None:
    rows = (
        OpeningEventAccountingRowV1(
            period="assessment",
            session=date(2025, 1, 2),
            checkpoint=6,
            vti_proxy="VTI",
            opening_state="NEGATIVE_SEVERE_OPENING_TRANSITION",
            opening_sign=-1,
            event_id="event-1",
            eligible_stock_count=2,
            acted_stock_count=2,
            included_in_total_event_count=True,
            included_in_positive_count=False,
            included_in_negative_count=True,
            exact_explanation="eligible severe event",
        ),
        OpeningEventAccountingRowV1(
            period="assessment",
            session=date(2025, 1, 3),
            checkpoint=6,
            vti_proxy="VTI",
            opening_state="POSITIVE_SEVERE_OPENING_TRANSITION",
            opening_sign=1,
            event_id="event-2",
            eligible_stock_count=1,
            acted_stock_count=1,
            included_in_total_event_count=True,
            included_in_positive_count=True,
            included_in_negative_count=False,
            exact_explanation="eligible severe event",
        ),
    )

    result = reconcile_opening_event_accounting_v1(rows)

    assert result.outcome == "event_accounting_fully_reconciled"
    assert result.unique_signed_event_count == 2
    assert result.negative_unique_event_count == 1
    assert result.positive_unique_event_count == 1
    assert result.negative_unique_event_count + result.positive_unique_event_count == 2


def test_event_accounting_rejects_a_single_event_with_two_signs() -> None:
    base = OpeningEventAccountingRowV1(
        period="assessment",
        session=date(2025, 1, 2),
        checkpoint=6,
        vti_proxy="VTI",
        opening_state="NEGATIVE_SEVERE_OPENING_TRANSITION",
        opening_sign=-1,
        event_id="event-1",
        eligible_stock_count=1,
        acted_stock_count=1,
        included_in_total_event_count=True,
        included_in_positive_count=False,
        included_in_negative_count=True,
        exact_explanation="eligible severe event",
    )
    contradictory = base.model_copy(
        update={
            "opening_state": "POSITIVE_SEVERE_OPENING_TRANSITION",
            "opening_sign": 1,
            "included_in_positive_count": True,
            "included_in_negative_count": False,
        }
    )

    result = reconcile_opening_event_accounting_v1((base, contradictory))

    assert result.outcome == "blocked_event_accounting"
    assert "event_has_multiple_transition_signs:event-1" in result.errors


def test_transfer_compares_predictors_without_requiring_exact_ohlc() -> None:
    def bars(source_shift: float) -> tuple[OpeningTransferBarV1, ...]:
        return tuple(
            OpeningTransferBarV1(
                ordinal=ordinal,
                bar_start_timestamp_utc=SESSION_OPEN
                + timedelta(minutes=5 * ordinal),
                bar_complete_timestamp_utc=SESSION_OPEN
                + timedelta(minutes=5 * (ordinal + 1)),
                open=100.0 - 0.1 * ordinal + source_shift,
                high=100.4 - 0.1 * ordinal + source_shift,
                low=99.6 - 0.1 * ordinal + source_shift,
                close=99.9 - 0.1 * ordinal + source_shift,
                complete=True,
            )
            for ordinal in range(6)
        )

    result = evaluate_opening_transfer_session_v1(
        session=SESSION,
        ibkr_bars=bars(0.0),
        eodhd_bars=bars(0.01),
        checkpoint_6_episode_identity_agreement=True,
        stock_probability_rank_comparison_available=True,
        operational_evidence=_passing_operational_evidence(),
    )

    assert result.decision == "opening_transfer_supported_without_recalibration"
    assert result.valid
    assert result.exact_ohlc_equality_required is False
    assert result.future_stock_returns_accessed is False
    assert result.direction_outcomes_accessed is False
    assert result.m1c_outcomes_accessed is False
    assert result.option_pnl_accessed is False


def test_transfer_rejects_six_bars_from_the_wrong_session_window() -> None:
    bars = tuple(
        OpeningTransferBarV1(
            ordinal=ordinal,
            bar_start_timestamp_utc=SESSION_OPEN
            + timedelta(minutes=5 * (ordinal + 1)),
            bar_complete_timestamp_utc=SESSION_OPEN
            + timedelta(minutes=5 * (ordinal + 2)),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.1,
            complete=True,
        )
        for ordinal in range(6)
    )

    result = evaluate_opening_transfer_session_v1(
        session=SESSION,
        ibkr_bars=bars,
        eodhd_bars=bars,
        checkpoint_6_episode_identity_agreement=True,
        stock_probability_rank_comparison_available=True,
        operational_evidence=_passing_operational_evidence(),
    )

    assert not result.valid
    assert result.decision == "opening_transfer_operational_failure"
    assert "opening_transfer_wrong_session_window" in result.missing_reasons


def test_transfer_disagreement_counts_in_fixed_engineering_cohort() -> None:
    def bars(close_step: float) -> tuple[OpeningTransferBarV1, ...]:
        return tuple(
            OpeningTransferBarV1(
                ordinal=ordinal,
                bar_start_timestamp_utc=SESSION_OPEN
                + timedelta(minutes=5 * ordinal),
                bar_complete_timestamp_utc=SESSION_OPEN
                + timedelta(minutes=5 * (ordinal + 1)),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0 + close_step * (ordinal + 1),
                complete=True,
            )
            for ordinal in range(6)
        )

    result = evaluate_opening_transfer_session_v1(
        session=SESSION,
        ibkr_bars=bars(0.10),
        eodhd_bars=bars(-0.10),
        checkpoint_6_episode_identity_agreement=True,
        stock_probability_rank_comparison_available=True,
        operational_evidence=_passing_operational_evidence(),
    )

    assert result.valid
    assert result.decision == "opening_transfer_mixed"


def test_aggregate_transfer_receipt_uses_exactly_first_twenty_valid_sessions() -> None:
    def one_result(session: date) -> OpeningTransferSessionResultV1:
        session_open, _ = xnys_session_bounds(session)
        bars = tuple(
            OpeningTransferBarV1(
                ordinal=index,
                bar_start_timestamp_utc=session_open
                + timedelta(minutes=5 * index),
                bar_complete_timestamp_utc=session_open
                + timedelta(minutes=5 * (index + 1)),
                open=100.0,
                high=100.5,
                low=99.5,
                close=99.9,
                complete=True,
            )
            for index in range(6)
        )
        return evaluate_opening_transfer_session_v1(
            session=session,
            ibkr_bars=bars,
            eodhd_bars=bars,
            checkpoint_6_episode_identity_agreement=True,
            stock_probability_rank_comparison_available=True,
            operational_evidence=_passing_operational_evidence(),
        )

    trading_sessions: list[date] = []
    candidate = SESSION
    while len(trading_sessions) < 20:
        try:
            xnys_session_bounds(candidate)
        except ValueError:
            pass
        else:
            trading_sessions.append(candidate)
        candidate += timedelta(days=1)
    sessions = tuple(one_result(session) for session in trading_sessions)
    receipt = build_opening_transfer_decision_receipt_v1(
        sessions=sessions,
        boundary_timestamp_utc=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert receipt.receipt_kind == "transfer"
    assert receipt.decision == "opening_transfer_supported_without_recalibration"
    assert dict(receipt.support_counts)["valid_sessions"] == 20
    assert not receipt.protected_outcome_fields_accessed


def test_transfer_without_operational_guard_evidence_fails_closed() -> None:
    bars = tuple(
        OpeningTransferBarV1(
            ordinal=index,
            bar_start_timestamp_utc=SESSION_OPEN
            + timedelta(minutes=5 * index),
            bar_complete_timestamp_utc=SESSION_OPEN
            + timedelta(minutes=5 * (index + 1)),
            open=100.0,
            high=100.5,
            low=99.5,
            close=99.9,
            complete=True,
        )
        for index in range(6)
    )

    result = evaluate_opening_transfer_session_v1(
        session=SESSION,
        ibkr_bars=bars,
        eodhd_bars=bars,
        checkpoint_6_episode_identity_agreement=True,
        stock_probability_rank_comparison_available=True,
    )

    assert not result.valid
    assert result.decision == "opening_transfer_operational_failure"
    assert "engineering_operational_evidence_missing" in result.missing_reasons
