from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalPredictionInputV1,
    build_activation_receipt_v1,
    build_frozen_experiment_config_v1,
    build_incomplete_opening_reversal_outcome_v1,
    build_opening_reversal_confirmation_start_receipt_v1,
    build_opening_reversal_decision_receipt_v1,
    build_prediction_receipt_v1,
)
from stocker_prospective.recorder_repository import FrozenRecorderRepository

ACTIVATION = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
SESSION = date(2026, 7, 30)
ENTRY = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)


def _metadata(run_id: str, *, observed: datetime = ENTRY) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=ACTIVATION,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[observed.isoformat()],
        recorded_at_utc=observed,
    )


def _activation_receipt():
    config = build_frozen_experiment_config_v1()
    return build_activation_receipt_v1(
        activation_timestamp_utc=ACTIVATION,
        new_york_trading_date_at_activation=ACTIVATION.date(),
        branch="codex/m1c-prospective-opening-reversal-v1",
        commit="a" * 40,
        dirty_working_tree_status="clean",
        configuration_hash=config.configuration_hash,
        m1c_version="frozen-m1c-v0",
        tail_phase_version="m1c-tail-phase-v1",
        a1_version="frozen-a1-v0",
    )


def _seed_supported_transfer(
    *,
    database: ProspectiveRepository,
    repository: FrozenRecorderRepository,
    metadata: EvidenceMetadata,
) -> None:
    sessions = tuple(date(2026, 7, 1) + timedelta(days=index) for index in range(20))
    hashes = tuple(f"{index + 50_000:064x}" for index in range(20))
    with database._connect() as connection:
        for ordinal, (session, report_hash) in enumerate(
            zip(sessions, hashes, strict=True),
            start=1,
        ):
            envelope_id = database._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO opening_reversal_transfer_session_v1(
                    envelope_id, run_id, session_date, valid,
                    valid_session_ordinal, decision, ibkr_opening_return,
                    eodhd_opening_return, ibkr_opening_range,
                    eodhd_opening_range, severe_state_agreement,
                    sign_agreement, timestamp_alignment,
                    checkpoint_6_episode_identity_agreement,
                    operational_checks_pass, operational_evidence_json,
                    outcome_fields_accessed, report_json, report_hash_v1
                ) VALUES (?, ?, ?, 1, ?,
                          'opening_transfer_supported_without_recalibration',
                          0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1,
                          '{}', 0, '{}', ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    session.isoformat(),
                    ordinal,
                    report_hash,
                ),
            )
    transfer = build_opening_reversal_decision_receipt_v1(
        receipt_kind="transfer",
        boundary_timestamp_utc=ENTRY - timedelta(days=1),
        decision="opening_transfer_supported_without_recalibration",
        cohort_first_session=sessions[0],
        cohort_last_session=sessions[-1],
        source_receipt_hashes=hashes,
        support_counts={
            "operational_sessions_passed": 20,
            "valid_sessions": 20,
        },
        protected_outcome_fields_accessed=False,
    )
    repository.record_opening_reversal_decision_receipt_v1(metadata, transfer)


def _receipt(*, probability: float = 0.70):
    return build_prediction_receipt_v1(
        OpeningReversalPredictionInputV1(
            activation_timestamp_utc=ACTIVATION,
            cohort_phase="prospective_development",
            transfer_status="opening_transfer_supported_without_recalibration",
            session=SESSION,
            stock="AAL",
            checkpoint=6,
            signal_timestamp_utc=ENTRY,
            entry_timestamp_utc=ENTRY,
            receipt_created_at_utc=ENTRY - timedelta(microseconds=1),
            m1c_probability=probability,
            m1c_probability_valid=True,
            high_tail_membership=True,
            fresh_episode_id="fresh-aal",
            canonical_fresh_episode=True,
            tail_phase_v1="FIRST_ENTRY",
            market_opening_return_v1=-0.004,
            market_opening_range_v1=0.006,
            opening_market_transition_state_v1=("NEGATIVE_SEVERE_OPENING_TRANSITION"),
            opening_transition_sign_v1=-1,
            opening_transition_event_id_v1="event-1",
            vti_opening_transition_complete=True,
            stock_causal_data_complete=True,
            previous_close_atm_iv_scale_15m=0.01,
            previous_close_atm_iv_scale_valid=True,
            data_source="ibkr",
            capacity_snapshot_id="snapshot-1",
        )
    )


def test_migration_and_prediction_receipt_are_append_immutable(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    activation = _activation_receipt()
    assert {
        "VTI_5m",
        "frozen_20_stock_m1c_5m",
        "one_promoted_underlying_level1",
        "primary_1dte_atm_call",
        "primary_1dte_atm_put",
    }.issubset(activation.mandatory_feed_manifest)
    receipt = _receipt()

    activation_id = repository.record_opening_reversal_activation_v1(
        metadata,
        activation,
    )
    _seed_supported_transfer(
        database=database,
        repository=repository,
        metadata=metadata,
    )
    prediction_id = repository.record_opening_reversal_prediction_v1(
        metadata,
        receipt,
    )

    assert (
        repository.record_opening_reversal_activation_v1(
            metadata,
            activation,
        )
        == activation_id
    )
    assert (
        repository.record_opening_reversal_prediction_v1(
            metadata,
            receipt,
        )
        == prediction_id
    )
    with database._connect() as connection:
        stored = connection.execute(
            """
            SELECT prediction_v1, prediction_sign_v1, eligibility_v1,
                   scientific_outcome_eligible_v1, capacity_snapshot_id,
                   receipt_hash_v1
            FROM opening_reversal_prediction_v1
            """
        ).fetchone()
        migration = connection.execute(
            """
            SELECT version FROM schema_migrations
            WHERE version = '0014_m1c_prospective_opening_reversal_v1.sql'
            """
        ).fetchone()

    assert migration is not None
    assert stored is not None
    assert stored["prediction_v1"] == "CALL"
    assert stored["prediction_sign_v1"] == 1
    assert bool(stored["eligibility_v1"])
    assert bool(stored["scientific_outcome_eligible_v1"])
    assert stored["capacity_snapshot_id"] == "snapshot-1"
    assert stored["receipt_hash_v1"] == receipt.receipt_hash_v1


def test_changed_prediction_cannot_overwrite_an_existing_receipt(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    repository.record_opening_reversal_activation_v1(
        metadata,
        _activation_receipt(),
    )
    _seed_supported_transfer(
        database=database,
        repository=repository,
        metadata=metadata,
    )
    repository.record_opening_reversal_prediction_v1(metadata, _receipt())

    with pytest.raises(ValueError, match="immutable"):
        repository.record_opening_reversal_prediction_v1(
            metadata,
            _receipt(probability=0.80),
        )


def test_incomplete_underlying_outcome_is_persisted_as_missing_evidence(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    repository.record_opening_reversal_activation_v1(
        metadata,
        _activation_receipt(),
    )
    _seed_supported_transfer(
        database=database,
        repository=repository,
        metadata=metadata,
    )
    receipt = _receipt()
    repository.record_opening_reversal_prediction_v1(metadata, receipt)
    outcome = build_incomplete_opening_reversal_outcome_v1(
        prediction_receipt=receipt,
        missing_reason_v1="post_entry_bar_missing",
        outcome_created_at_utc=ENTRY + timedelta(minutes=16),
    )

    repository.record_opening_reversal_underlying_outcome_v1(
        _metadata(
            "opening-reversal-v1",
            observed=ENTRY + timedelta(minutes=16),
        ),
        outcome,
    )

    with database._connect() as connection:
        stored = connection.execute(
            """
            SELECT r_15m, outcome_completeness_v1, missing_reason_v1
            FROM opening_reversal_underlying_outcome_v1
            """
        ).fetchone()
    assert stored is not None
    assert stored["r_15m"] is None
    assert stored["outcome_completeness_v1"] == "incomplete"
    assert stored["missing_reason_v1"] == "post_entry_bar_missing"


def test_contract_discovery_failure_is_persisted_without_live_chain_lines(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    repository.record_opening_reversal_activation_v1(
        metadata,
        _activation_receipt(),
    )
    _seed_supported_transfer(
        database=database,
        repository=repository,
        metadata=metadata,
    )
    repository.record_opening_reversal_prediction_v1(metadata, _receipt())

    repository.record_opening_reversal_contract_discovery_failure_v1(
        metadata,
        episode_id="fresh-aal",
        discovery_timestamp_utc=ENTRY,
        contract_source="ibkr_secdef_metadata",
        cache_hit=False,
        candidates_inspected=4,
        missing_reason="primary_1dte_option_pair_unavailable",
    )

    with database._connect() as connection:
        stored = connection.execute(
            """
            SELECT status, candidates_inspected,
                   live_market_data_lines_consumed,
                   metadata_request_ended,
                   full_chain_live_subscription_created,
                   missing_reason
            FROM opening_reversal_contract_discovery_v1
            """
        ).fetchone()

    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["candidates_inspected"] == 4
    assert stored["live_market_data_lines_consumed"] == 0
    assert bool(stored["metadata_request_ended"])
    assert not bool(stored["full_chain_live_subscription_created"])
    assert stored["missing_reason"] == "primary_1dte_option_pair_unavailable"


def test_phase_resolution_does_not_require_transfer_sessions(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)

    phase, transfer = repository.opening_reversal_phase_for_session(
        run_id=metadata.run_id,
        session=SESSION,
    )

    assert phase == "prospective_development"
    assert transfer == "cross_vendor_validation_not_configured"


def test_invalid_transfer_receipt_does_not_block_prospective_development(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    transfer = build_opening_reversal_decision_receipt_v1(
        receipt_kind="transfer",
        boundary_timestamp_utc=ENTRY,
        decision="opening_transfer_supported_without_recalibration",
        cohort_first_session=date(2026, 7, 1),
        cohort_last_session=SESSION,
        source_receipt_hashes=tuple(f"{index + 1:064x}" for index in range(20)),
        support_counts={
            "operational_sessions_passed": 20,
            "valid_sessions": 20,
        },
        protected_outcome_fields_accessed=False,
    )
    with pytest.raises(ValueError, match="first 20 valid sessions"):
        repository.record_opening_reversal_decision_receipt_v1(
            metadata,
            transfer,
        )

    phase, decision = repository.opening_reversal_phase_for_session(
        run_id=metadata.run_id,
        session=SESSION + timedelta(days=1),
    )

    assert phase == "prospective_development"
    assert decision == "cross_vendor_validation_not_configured"


def test_transfer_boundary_receipt_rejects_protected_outcome_access() -> None:
    with pytest.raises(ValueError, match="cannot access protected outcomes"):
        build_opening_reversal_decision_receipt_v1(
            receipt_kind="transfer",
            boundary_timestamp_utc=ENTRY,
            decision="opening_transfer_supported_without_recalibration",
            cohort_first_session=date(2026, 7, 1),
            cohort_last_session=SESSION,
            source_receipt_hashes=tuple(f"{index + 1:064x}" for index in range(20)),
            support_counts={"valid_sessions": 20},
            protected_outcome_fields_accessed=True,
        )


def test_decision_receipt_rejects_hash_tampering() -> None:
    receipt = build_opening_reversal_decision_receipt_v1(
        receipt_kind="transfer",
        boundary_timestamp_utc=ENTRY,
        decision="opening_transfer_supported_without_recalibration",
        cohort_first_session=date(2026, 7, 1),
        cohort_last_session=SESSION,
        source_receipt_hashes=tuple(f"{index + 1:064x}" for index in range(20)),
        support_counts={
            "operational_sessions_passed": 20,
            "valid_sessions": 20,
        },
        protected_outcome_fields_accessed=False,
    )
    payload = receipt.model_dump(mode="python")
    payload["decision"] = "opening_transfer_mixed"

    with pytest.raises(ValueError, match="hash mismatch"):
        type(receipt).model_validate(payload)


def test_confirmation_phase_requires_development_and_confirmation_start_receipts(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    development_end = SESSION + timedelta(days=60)
    development = build_opening_reversal_decision_receipt_v1(
        receipt_kind="development",
        boundary_timestamp_utc=ENTRY + timedelta(days=60),
        decision="prospective_opening_reversal_development_supported",
        cohort_first_session=SESSION + timedelta(days=1),
        cohort_last_session=development_end,
        source_receipt_hashes=tuple(f"{index + 1_000:064x}" for index in range(150)),
        support_counts={
            "complete_eligible_stock_episodes": 150,
            "maximum_event_episode_count": 4,
            "maximum_stock_episode_count": 10,
            "negative_transition_events": 20,
            "positive_transition_events": 20,
            "represented_stocks": 15,
            "sessions": 40,
            "unique_severe_opening_events": 40,
        },
        protected_outcome_fields_accessed=True,
    )
    confirmation_start = build_opening_reversal_confirmation_start_receipt_v1(
        development_receipt=development,
        boundary_timestamp_utc=ENTRY + timedelta(days=61),
    )
    with pytest.raises(ValueError, match="persisted outcomes"):
        repository.record_opening_reversal_decision_receipt_v1(
            metadata,
            development,
        )
    with pytest.raises(ValueError, match="stored supported development"):
        repository.record_opening_reversal_decision_receipt_v1(
            metadata,
            confirmation_start,
        )
