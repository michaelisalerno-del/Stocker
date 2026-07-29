from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalPredictionInputV1,
    build_activation_receipt_v1,
    build_frozen_experiment_config_v1,
    build_prediction_receipt_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    OpeningReversalDecisionDataGateV1_1,
    OpeningReversalPredictionTimingEvidenceV1_1,
    build_activation_receipt_v1_1,
    build_causal_barrier_audit_v1_1,
    build_frozen_timing_addendum_config_v1_1,
)

BASE_ACTIVATION = datetime(2026, 7, 29, 6, 39, tzinfo=UTC)
ADDENDUM_ACTIVATION = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
SESSION = date(2026, 7, 30)
ENTRY = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)


def _base_activation():
    config = build_frozen_experiment_config_v1()
    return build_activation_receipt_v1(
        activation_timestamp_utc=BASE_ACTIVATION,
        new_york_trading_date_at_activation=BASE_ACTIVATION.date(),
        branch="codex/m1c-prospective-opening-reversal-v1",
        commit="a" * 40,
        dirty_working_tree_status="clean",
        configuration_hash=config.configuration_hash,
        m1c_version="frozen-m1c-v0",
        tail_phase_version="m1c-tail-phase-v1",
        a1_version="frozen-a1-v0",
    )


def test_v1_1_accepts_post_boundary_receipt_only_behind_causal_barrier() -> None:
    base = _base_activation()
    config = build_frozen_timing_addendum_config_v1_1(
        superseded_activation_receipt_hash_v1=base.activation_receipt_hash,
        frozen_rule_hash_v1=base.frozen_rule_hash,
        frozen_configuration_hash_v1=base.configuration_hash,
    )
    activation = build_activation_receipt_v1_1(
        activation_timestamp_utc=ADDENDUM_ACTIVATION,
        new_york_trading_date_at_activation=ADDENDUM_ACTIVATION.date(),
        branch="codex/m1c-prospective-opening-reversal-v1",
        commit="b" * 40,
        dirty_working_tree_status="clean",
        timing_addendum_config=config,
        superseded_activation_receipt=base,
        m1c_version=base.m1c_version,
        tail_phase_version=base.tail_phase_version,
        a1_version=base.a1_version,
    )
    created = ENTRY + timedelta(milliseconds=250)
    timing = OpeningReversalPredictionTimingEvidenceV1_1(
        timing_addendum_activation_receipt_hash_v1_1=(activation.activation_receipt_hash_v1_1),
        rule_committed_at_utc=activation.activation_timestamp_utc,
        causal_barrier_armed_at_utc=activation.activation_timestamp_utc,
        predictor_window_completed_at_utc=ENTRY,
        first_entry_or_post_entry_event_buffered_at_utc=(ENTRY + timedelta(milliseconds=1)),
        entry_or_post_entry_data_admitted_before_receipt=False,
        raw_event_archive_write_before_receipt=True,
        decision_surface_release_requires_durable_receipt=True,
        nominal_entry_actionable=False,
        receipt_latency_after_nominal_entry_seconds=0.25,
    )
    source = OpeningReversalPredictionInputV1(
            experiment_version="1.1",
            activation_timestamp_utc=activation.activation_timestamp_utc,
            cohort_phase="engineering_transfer",
            transfer_status="engineering_transfer_pending",
            session=SESSION,
            stock="AAL",
            checkpoint=6,
            signal_timestamp_utc=ENTRY,
            entry_timestamp_utc=ENTRY,
            receipt_created_at_utc=created,
            m1c_probability=0.70,
            m1c_probability_valid=True,
            high_tail_membership=True,
            fresh_episode_id="fresh-aal",
            canonical_fresh_episode=True,
            tail_phase_v1="FIRST_ENTRY",
            market_opening_return_v1=-0.004,
            market_opening_range_v1=0.006,
            opening_market_transition_state_v1=("NEGATIVE_SEVERE_OPENING_TRANSITION"),
            opening_transition_sign_v1=-1,
            opening_transition_event_id_v1="opening-event-1",
            vti_opening_transition_complete=True,
            stock_causal_data_complete=True,
            previous_close_atm_iv_scale_15m=0.01,
            previous_close_atm_iv_scale_valid=True,
            data_source="ibkr",
            capacity_snapshot_id="capacity-1",
            timing_evidence_v1_1=timing,
        )
    receipt = build_prediction_receipt_v1(source)

    assert receipt.experiment_version == "1.1"
    assert receipt.eligibility_v1
    assert receipt.prediction_v1 == "CALL"
    assert receipt.prediction_sign_v1 == 1
    assert receipt.timing_evidence_v1_1 == timing
    assert "receipt_not_completed_before_entry" not in (receipt.ineligibility_reasons_v1)
    late_buffer_receipt = build_prediction_receipt_v1(
        source.model_copy(
            update={
                "timing_evidence_v1_1": timing.model_copy(
                    update={
                        "first_entry_or_post_entry_event_buffered_at_utc": (
                            created + timedelta(microseconds=1)
                        )
                    }
                )
            }
        )
    )
    assert late_buffer_receipt.prediction_v1 == "ABSTAIN"
    assert "entry_buffer_timestamp_after_receipt" in (
        late_buffer_receipt.ineligibility_reasons_v1
    )


def test_decision_data_gate_buffers_entry_events_until_durable_audit() -> None:
    gate = OpeningReversalDecisionDataGateV1_1(
        protected_symbols=frozenset({"AAL", "VTI"}),
    )

    assert (
        gate.observe(
            session=SESSION,
            symbol="AAL",
            nominal_entry_timestamp_utc=ENTRY,
            event_ordering_timestamp_utc=ENTRY - timedelta(microseconds=1),
            event_received_timestamp_utc=ENTRY,
        )
        == "admit"
    )
    assert (
        gate.observe(
            session=SESSION,
            symbol="AAL",
            nominal_entry_timestamp_utc=ENTRY,
            event_ordering_timestamp_utc=ENTRY,
            event_received_timestamp_utc=ENTRY + timedelta(milliseconds=1),
            event_id="entry-event-1",
        )
        == "buffer"
    )
    assert (
        gate.observe(
            session=SESSION,
            symbol="AAL",
            nominal_entry_timestamp_utc=ENTRY,
            event_ordering_timestamp_utc=ENTRY,
            event_received_timestamp_utc=ENTRY + timedelta(milliseconds=1),
            event_id="entry-event-1",
        )
        == "buffer"
    )
    assert gate.deferred_event_count(SESSION) == 1
    assert gate.first_deferred_event_received_at(SESSION) == (ENTRY + timedelta(milliseconds=1))

    gate.authorize_release_after_durable_audit(
        session=SESSION,
        audit_hash_v1_1="a" * 64,
    )

    assert (
        gate.observe(
            session=SESSION,
            symbol="VTI",
            nominal_entry_timestamp_utc=ENTRY,
            event_ordering_timestamp_utc=ENTRY + timedelta(seconds=1),
            event_received_timestamp_utc=ENTRY + timedelta(seconds=1),
        )
        == "admit"
    )
    assert not gate.scientific_barrier_compromised(SESSION)


def test_incomplete_receipt_barrier_fails_science_but_releases_core_recording() -> None:
    base = _base_activation()
    config = build_frozen_timing_addendum_config_v1_1(
        superseded_activation_receipt_hash_v1=base.activation_receipt_hash,
        frozen_rule_hash_v1=base.frozen_rule_hash,
        frozen_configuration_hash_v1=base.configuration_hash,
    )
    activation = build_activation_receipt_v1_1(
        activation_timestamp_utc=ADDENDUM_ACTIVATION,
        new_york_trading_date_at_activation=ADDENDUM_ACTIVATION.date(),
        branch="codex/m1c-prospective-opening-reversal-v1",
        commit="b" * 40,
        dirty_working_tree_status="clean",
        timing_addendum_config=config,
        superseded_activation_receipt=base,
        m1c_version=base.m1c_version,
        tail_phase_version=base.tail_phase_version,
        a1_version=base.a1_version,
    )
    audit = build_causal_barrier_audit_v1_1(
        activation_receipt_hash_v1_1=activation.activation_receipt_hash_v1_1,
        session=SESSION,
        nominal_entry_timestamp_utc=ENTRY,
        prediction_receipts=(),
        deferred_event_received_timestamps=(ENTRY + timedelta(milliseconds=1),),
        entry_or_post_entry_data_admitted_before_receipts=False,
        release_authorized_at_utc=ENTRY + timedelta(milliseconds=250),
    )
    gate = OpeningReversalDecisionDataGateV1_1(
        protected_symbols=frozenset({"AAL", "VTI"}),
    )
    gate.fail_closed_for_science_and_continue_core(
        session=SESSION,
        reason=audit.failure_reason or "",
    )

    assert audit.barrier_status == "failed_closed"
    assert audit.failure_reason == "prediction_receipt_set_incomplete_before_release"
    assert audit.core_recorder_continued
    assert gate.scientific_barrier_compromised(SESSION)
    assert (
        gate.observe(
            session=SESSION,
            symbol="AAL",
            nominal_entry_timestamp_utc=ENTRY,
            event_ordering_timestamp_utc=ENTRY + timedelta(seconds=1),
            event_received_timestamp_utc=ENTRY + timedelta(seconds=1),
        )
        == "admit"
    )
