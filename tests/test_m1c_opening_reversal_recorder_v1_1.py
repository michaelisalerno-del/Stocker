from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.events import OptionQuoteEvent
from stocker_prospective.frozen_m1c import FrozenM1CScore
from stocker_prospective.group_o import build_group_o_context
from stocker_prospective.m1c_features import LiveFeatureBar
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalPredictionInputV1,
    OpeningReversalPredictionTimingEvidenceV1_1,
    OptionContractCandidateV1,
    OptionTopOfBookV1,
    build_activation_receipt_v1,
    build_frozen_experiment_config_v1,
    build_prediction_receipt_v1,
    build_primary_option_bid_ask_outcome_v1,
    select_primary_option_pair_v1,
    select_promoted_prediction_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    build_activation_receipt_v1_1,
    build_causal_barrier_audit_v1_1,
    build_frozen_timing_addendum_config_v1_1,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.opening_market_transition_v1 import (
    OpeningTransitionThresholdsV1,
)
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.options import DteBucket
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointInput,
)
from stocker_prospective.signed_market_shock_v1 import MarketShockBarV1

BASE_ACTIVATION = datetime(2026, 7, 29, 6, 39, tzinfo=UTC)
ADDENDUM_ACTIVATION = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
SESSION = date(2026, 7, 30)
ENTRY = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)
RECEIPT_CREATED = ENTRY + timedelta(milliseconds=250)


def _metadata(run_id: str, observed: datetime) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=BASE_ACTIVATION,
        app_version="test",
        git_commit="c" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[observed.isoformat()],
        recorded_at_utc=observed,
    )


def _activation_pair():
    frozen = build_frozen_experiment_config_v1()
    base = build_activation_receipt_v1(
        activation_timestamp_utc=BASE_ACTIVATION,
        new_york_trading_date_at_activation=BASE_ACTIVATION.date(),
        branch="codex/m1c-prospective-opening-reversal-v1",
        commit="a" * 40,
        dirty_working_tree_status="clean",
        configuration_hash=frozen.configuration_hash,
        m1c_version="frozen-m1c-v0",
        tail_phase_version="m1c-tail-phase-v1",
        a1_version="frozen-a1-v0",
    )
    addendum = build_frozen_timing_addendum_config_v1_1(
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
        timing_addendum_config=addendum,
        superseded_activation_receipt=base,
        m1c_version=base.m1c_version,
        tail_phase_version=base.tail_phase_version,
        a1_version=base.a1_version,
    )
    return base, activation


def _prediction(
    addendum_activation_hash: str,
    *,
    stock: str = "AAL",
    cohort_phase: Literal[
        "engineering_transfer",
        "prospective_development",
        "untouched_confirmation",
    ] = "prospective_development",
    transfer_status: str = "cross_vendor_validation_not_configured",
):
    timing = OpeningReversalPredictionTimingEvidenceV1_1(
        timing_addendum_activation_receipt_hash_v1_1=(addendum_activation_hash),
        rule_committed_at_utc=ADDENDUM_ACTIVATION,
        causal_barrier_armed_at_utc=ADDENDUM_ACTIVATION,
        predictor_window_completed_at_utc=ENTRY,
        first_entry_or_post_entry_event_buffered_at_utc=(ENTRY + timedelta(milliseconds=1)),
        entry_or_post_entry_data_admitted_before_receipt=False,
        raw_event_archive_write_before_receipt=True,
        decision_surface_release_requires_durable_receipt=True,
        nominal_entry_actionable=False,
        receipt_latency_after_nominal_entry_seconds=0.25,
    )
    return build_prediction_receipt_v1(
        OpeningReversalPredictionInputV1(
            experiment_version="1.1",
            activation_timestamp_utc=ADDENDUM_ACTIVATION,
            cohort_phase=cohort_phase,
            transfer_status=transfer_status,
            session=SESSION,
            stock=stock,
            checkpoint=6,
            signal_timestamp_utc=ENTRY,
            entry_timestamp_utc=ENTRY,
            receipt_created_at_utc=RECEIPT_CREATED,
            m1c_probability=0.70,
            m1c_probability_valid=True,
            high_tail_membership=True,
            fresh_episode_id=f"fresh-{stock.lower()}",
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
    )


def _record_fresh_episode(
    database: ProspectiveRepository,
    metadata: EvidenceMetadata,
    *,
    quiet_state_alias: bool = False,
    scientific_recording_valid: bool = True,
) -> None:
    with database._connect() as connection:
        checkpoint_envelope = database._insert_envelope(connection, metadata)
        checkpoint = connection.execute(
            """
            INSERT INTO m1c_checkpoint_v0(
                envelope_id, run_id, symbol, session_date, checkpoint,
                bar_start_utc, bar_end_utc, feature_as_of_utc, model_id,
                model_version, model_hash, feature_hash,
                session_context_hash, feature_values_json, probability,
                threshold, threshold_passed, eligible, feature_freshness,
                missing_feature_count, rejection_reasons_json, claims_json
            ) VALUES (?, ?, 'AAL', ?, 6, ?, ?, ?, 'M1C', 'frozen-m1c-v0',
                      ?, ?, ?, '{}', 0.70, 0.488333710794033, 1, ?,
                      'complete', 0, '[]', '{}')
            """,
            (
                checkpoint_envelope,
                metadata.run_id,
                SESSION.isoformat(),
                (ENTRY - timedelta(minutes=5)).isoformat(),
                ENTRY.isoformat(),
                ENTRY.isoformat(),
                "a" * 64,
                "b" * 64,
                "c" * 64,
                int(scientific_recording_valid),
            ),
        )
        assert checkpoint.lastrowid is not None
        episode_envelope = database._insert_envelope(connection, metadata)
        connection.execute(
            """
            INSERT INTO m1c_episode_v0(
                episode_id, envelope_id, checkpoint_id, run_id, symbol,
                session_date, trigger_checkpoint, trigger_bar_end_utc,
                prospective_entry_timestamp_utc, m1c_probability,
                previous_m1c_probability, episode_number,
                minutes_since_previous_episode, scientific_recording_valid,
                rejection_reasons_json, phase, completion_status,
                completed_at_utc, claims_json
            ) VALUES ('fresh-aal', ?, ?, ?, 'AAL', ?, 6, ?, ?, 0.70,
                      0.40, 1, NULL, ?, '[]', 'pending_completion',
                      'active', NULL, '{}')
            """,
            (
                episode_envelope,
                checkpoint.lastrowid,
                metadata.run_id,
                SESSION.isoformat(),
                ENTRY.isoformat(),
                ENTRY.isoformat(),
                int(scientific_recording_valid),
            ),
        )
        if not quiet_state_alias:
            return
        quiet_checkpoint_envelope = database._insert_envelope(connection, metadata)
        quiet_checkpoint = connection.execute(
            """
            INSERT INTO quiet_state_checkpoint_v0(
                envelope_id, checkpoint_id, run_id, symbol, session_date,
                checkpoint, m1c_probability, previous_m1c_probability,
                bottom_5, bottom_10, bottom_20, high_tail,
                distance_from_bottom_10, model_hash, feature_hash, eligible,
                data_quality_status, data_quality_flags_json, claims_json
            ) VALUES (?, ?, ?, 'AAL', ?, 6, 0.70, 0.40, 0, 0, 0, 1,
                      0.564103034304374, ?, ?, 1, 'valid', '[]', '{}')
            """,
            (
                quiet_checkpoint_envelope,
                checkpoint.lastrowid,
                metadata.run_id,
                SESSION.isoformat(),
                "a" * 64,
                "b" * 64,
            ),
        )
        assert quiet_checkpoint.lastrowid is not None
        quiet_episode_envelope = database._insert_envelope(connection, metadata)
        connection.execute(
            """
            INSERT INTO quiet_state_observation_v0(
                observation_id, envelope_id, quiet_checkpoint_id, run_id,
                observation_kind, symbol, session_date, trigger_checkpoint,
                trigger_timestamp_utc, prospective_entry_timestamp_utc,
                m1c_probability, previous_m1c_probability, bottom_5,
                bottom_10, bottom_20, high_tail, episode_number,
                minutes_since_previous_quiet_episode,
                previous_high_tail_within_60_minutes,
                following_high_tail_within_60_minutes,
                scientific_recording_valid, data_quality_flags_json, phase,
                completion_status, completed_at_utc, claims_json
            ) VALUES ('fresh-aal', ?, ?, ?, 'high_tail_control', 'AAL', ?,
                      6, ?, ?, 0.70, 0.40, 0, 0, 0, 1, 1, NULL, 0, 0, 1,
                      '[]', 'pending_completion', 'active', NULL, '{}')
            """,
            (
                quiet_episode_envelope,
                quiet_checkpoint.lastrowid,
                metadata.run_id,
                SESSION.isoformat(),
                ENTRY.isoformat(),
                ENTRY.isoformat(),
            ),
        )


def _seed_eligible_v1_1_episode(
    database: ProspectiveRepository,
    repository: FrozenRecorderRepository,
    metadata: EvidenceMetadata,
    *,
    quiet_state_alias: bool = False,
    scientific_recording_valid: bool = True,
):
    base, activation = _activation_pair()
    repository.record_opening_reversal_activation_v1(metadata, base)
    repository.record_opening_reversal_activation_v1_1(metadata, activation)
    with database._connect() as connection:
        envelope_id = database._insert_envelope(connection, metadata)
        connection.execute(
            """
            INSERT INTO opening_reversal_decision_receipt_v1(
                envelope_id, run_id, receipt_kind, boundary_timestamp_utc,
                decision, cohort_first_session, cohort_last_session,
                receipt_hash_v1, receipt_json
            ) VALUES (?, ?, 'transfer', ?,
                      'opening_transfer_supported_without_recalibration',
                      '2026-07-01', '2026-07-29', ?, '{}')
            """,
            (
                envelope_id,
                metadata.run_id,
                ADDENDUM_ACTIVATION.isoformat(),
                "d" * 64,
            ),
        )
    _record_fresh_episode(
        database,
        metadata,
        quiet_state_alias=quiet_state_alias,
        scientific_recording_valid=scientific_recording_valid,
    )
    stocks = ("AAL", *(f"S{index:02d}" for index in range(1, 20)))
    receipts = tuple(
        _prediction(
            activation.activation_receipt_hash_v1_1,
            stock=stock,
            cohort_phase="prospective_development",
            transfer_status="opening_transfer_supported_without_recalibration",
        )
        for stock in stocks
    )
    for receipt in receipts:
        repository.record_opening_reversal_prediction_v1(metadata, receipt)
    audit = build_causal_barrier_audit_v1_1(
        activation_receipt_hash_v1_1=activation.activation_receipt_hash_v1_1,
        session=SESSION,
        nominal_entry_timestamp_utc=ENTRY,
        prediction_receipts=receipts,
        deferred_event_received_timestamps=(ENTRY + timedelta(milliseconds=1),),
        entry_or_post_entry_data_admitted_before_receipts=False,
        release_authorized_at_utc=RECEIPT_CREATED,
    )
    repository.record_opening_reversal_causal_barrier_audit_v1_1(metadata, audit)
    repository.record_opening_reversal_promotion_v1(
        metadata,
        select_promoted_prediction_v1(receipts),
    )
    return receipts[0]


def _primary_selection(*, expiry_offset: int = 1):
    expiry = SESSION + timedelta(days=expiry_offset)
    candidates = (
        OptionContractCandidateV1(
            con_id=1001,
            underlying="AAL",
            expiry=expiry,
            strike=100.0,
            right="C",
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        ),
        OptionContractCandidateV1(
            con_id=1002,
            underlying="AAL",
            expiry=expiry,
            strike=100.0,
            right="P",
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        ),
    )
    return select_primary_option_pair_v1(
        session=SESSION + timedelta(days=expiry_offset - 1),
        underlying_reference=100.0,
        candidates=candidates,
        discovery_timestamp_utc=ENTRY + timedelta(seconds=1),
        contract_source="synthetic_contract_metadata",
        cache_hit=False,
    )


def _primary_outcome(
    *,
    receipt_hash: str,
    contract: OptionContractCandidateV1,
    role: Literal["predicted_leg", "opposite_leg"],
):
    return build_primary_option_bid_ask_outcome_v1(
        prediction_receipt_hash_v1=receipt_hash,
        contract=contract,
        role=role,
        entry_timestamp_utc=ENTRY,
        subscription_start_utc=ENTRY,
        subscription_end_utc=ENTRY + timedelta(minutes=16),
        capacity_line_owner="opening-reversal-primary-pair",
        entry_quote=OptionTopOfBookV1(
            timestamp_utc=ENTRY + timedelta(seconds=1),
            bid=1.0,
            ask=1.1,
            quote_age_seconds=1.0,
            locked_or_crossed=False,
            stale=False,
            missing_reason=None,
        ),
        exit_quote=OptionTopOfBookV1(
            timestamp_utc=ENTRY + timedelta(minutes=15, seconds=1),
            bid=1.2,
            ask=1.3,
            quote_age_seconds=1.0,
            locked_or_crossed=False,
            stale=False,
            missing_reason=None,
        ),
    )


def test_v1_1_activation_and_prediction_are_durably_bound(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()

    repository.record_opening_reversal_activation_v1(metadata, base)
    repository.record_opening_reversal_activation_v1_1(
        metadata,
        activation,
    )
    receipt = _prediction(activation.activation_receipt_hash_v1_1)
    repository.record_opening_reversal_prediction_v1(metadata, receipt)

    with database._connect() as connection:
        stored_activation = connection.execute(
            """
            SELECT experiment_version, activation_receipt_hash
            FROM opening_reversal_activation_v1
            WHERE run_id = ? AND experiment_version = '1.1'
            """,
            (metadata.run_id,),
        ).fetchone()
        stored_prediction = connection.execute(
            """
            SELECT experiment_version, receipt_hash_v1, receipt_json
            FROM opening_reversal_prediction_v1
            WHERE run_id = ?
            """,
            (metadata.run_id,),
        ).fetchone()
        migration = connection.execute(
            """
            SELECT version FROM schema_migrations
            WHERE version =
                '0015_m1c_prospective_opening_reversal_v1_1.sql'
            """
        ).fetchone()

    assert migration is not None
    assert stored_activation is not None
    assert stored_activation["experiment_version"] == "1.1"
    assert stored_activation["activation_receipt_hash"] == activation.activation_receipt_hash_v1_1
    assert stored_prediction is not None
    assert stored_prediction["experiment_version"] == "1.1"
    assert stored_prediction["receipt_hash_v1"] == receipt.receipt_hash_v1
    assert "timing_evidence_v1_1" in stored_prediction["receipt_json"]


def test_v1_1_promotion_requires_persisted_passing_barrier_audit(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()
    repository.record_opening_reversal_activation_v1(metadata, base)
    repository.record_opening_reversal_activation_v1_1(metadata, activation)
    stocks = ("AAL", *(f"S{index:02d}" for index in range(1, 20)))
    receipts = tuple(
        _prediction(
            activation.activation_receipt_hash_v1_1,
            stock=stock,
        )
        for stock in stocks
    )
    for receipt in receipts:
        repository.record_opening_reversal_prediction_v1(metadata, receipt)
    selection = select_promoted_prediction_v1(receipts)

    with pytest.raises(ValueError, match="causal barrier"):
        repository.record_opening_reversal_promotion_v1(metadata, selection)

    audit = build_causal_barrier_audit_v1_1(
        activation_receipt_hash_v1_1=(activation.activation_receipt_hash_v1_1),
        session=SESSION,
        nominal_entry_timestamp_utc=ENTRY,
        prediction_receipts=receipts,
        deferred_event_received_timestamps=(ENTRY + timedelta(milliseconds=1),),
        entry_or_post_entry_data_admitted_before_receipts=False,
        release_authorized_at_utc=RECEIPT_CREATED,
    )
    repository.record_opening_reversal_causal_barrier_audit_v1_1(
        metadata,
        audit,
    )
    operational = repository.opening_reversal_engineering_operational_evidence_v1(
        run_id=metadata.run_id,
        session=SESSION,
    )

    assert (
        repository.record_opening_reversal_promotion_v1(
            metadata,
            selection,
        )
        > 0
    )
    assert operational.prediction_receipt_count == 20
    assert operational.prediction_receipt_timing_pass


def test_v1_1_eligible_episode_accepts_only_its_selected_two_line_outcomes(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "eligible-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("eligible-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    receipt = _seed_eligible_v1_1_episode(database, repository, metadata)
    selection = _primary_selection()

    repository.record_opening_reversal_contract_discovery_v1(
        metadata,
        episode_id="fresh-aal",
        selection=selection,
    )
    predicted_contract = selection.call if receipt.prediction_v1 == "CALL" else selection.put
    opposite_contract = selection.put if receipt.prediction_v1 == "CALL" else selection.call
    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=predicted_contract,
            role="predicted_leg",
        ),
    )
    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=opposite_contract,
            role="opposite_leg",
        ),
    )

    with database._connect() as connection:
        eligible = connection.execute(
            """
            SELECT activation_receipt_identity,
                   causal_barrier_audit_identity, episode_id
            FROM opening_reversal_v1_1_eligible_episode
            WHERE run_id = ?
            """,
            (metadata.run_id,),
        ).fetchall()
        legs = connection.execute(
            """
            SELECT right, role, expiry, strike
            FROM opening_reversal_primary_option_outcome_v1
            WHERE run_id = ?
            ORDER BY role
            """,
            (metadata.run_id,),
        ).fetchall()

    assert len(eligible) == 1
    assert eligible[0]["episode_id"] == "fresh-aal"
    assert eligible[0]["activation_receipt_identity"]
    assert eligible[0]["causal_barrier_audit_identity"]
    assert len(legs) == 2
    assert {str(row["right"]) for row in legs} == {"C", "P"}
    assert {str(row["role"]) for row in legs} == {
        "predicted_leg",
        "opposite_leg",
    }
    assert {str(row["expiry"]) for row in legs} == {(SESSION + timedelta(days=1)).isoformat()}
    assert {float(row["strike"]) for row in legs} == {100.0}


def test_v1_1_shadow_episode_captures_pair_without_becoming_scientific(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "shadow-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("shadow-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    receipt = _seed_eligible_v1_1_episode(
        database,
        repository,
        metadata,
        scientific_recording_valid=False,
    )
    selection = _primary_selection()

    repository.record_opening_reversal_contract_discovery_v1(
        metadata,
        episode_id="fresh-aal",
        selection=selection,
    )
    predicted = selection.call if receipt.prediction_v1 == "CALL" else selection.put
    opposite = selection.put if receipt.prediction_v1 == "CALL" else selection.call
    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=predicted,
            role="predicted_leg",
        ),
    )
    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=opposite,
            role="opposite_leg",
        ),
    )

    with database._connect() as connection:
        capture_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM opening_reversal_v1_1_capture_eligible_episode
            WHERE run_id = ?
            """,
            (metadata.run_id,),
        ).fetchone()
        scientific_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM opening_reversal_v1_1_eligible_episode
            WHERE run_id = ?
            """,
            (metadata.run_id,),
        ).fetchone()
        ledger = connection.execute(
            """
            SELECT lifecycle_state, pair_outcome_count, scientific_eligible
            FROM opening_reversal_virtual_position_v1
            WHERE run_id = ?
            """,
            (metadata.run_id,),
        ).fetchone()

    assert capture_count[0] == 1
    assert scientific_count[0] == 0
    assert ledger is not None
    assert ledger["lifecycle_state"] == "CLOSED"
    assert ledger["pair_outcome_count"] == 2
    assert ledger["scientific_eligible"] == 0
    projected = ProspectiveReadStore(
        database.database_path,
        run_id=metadata.run_id,
    ).opening_reversal_virtual_positions_v1()
    assert len(projected) == 1
    assert projected[0]["scientific_eligible"] is False
    assert projected[0]["execution_claimed"] is False
    assert projected[0]["paper_fill_claimed"] is False


def test_v1_1_repository_rejects_secondary_dte_selection(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "secondary-dte.sqlite3")
    database.migrate()
    metadata = _metadata("secondary-dte", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    _seed_eligible_v1_1_episode(database, repository, metadata)

    with pytest.raises(ValueError, match="primary_expiry_must_be_1dte"):
        repository.record_opening_reversal_contract_discovery_v1(
            metadata,
            episode_id="fresh-aal",
            selection=_primary_selection(expiry_offset=2),
        )


def test_v1_1_discovery_failure_is_valid_segregated_evidence(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "failed-discovery.sqlite3")
    database.migrate()
    metadata = _metadata("failed-discovery", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    _seed_eligible_v1_1_episode(database, repository, metadata)

    identifier = repository.record_opening_reversal_contract_discovery_failure_v1(
        metadata,
        episode_id="fresh-aal",
        discovery_timestamp_utc=RECEIPT_CREATED,
        contract_source="ibkr_secdef",
        cache_hit=False,
        candidates_inspected=0,
        missing_reason="no_valid_common_1dte_pair",
    )

    with database._connect() as connection:
        row = connection.execute(
            """
            SELECT status, call_con_id, put_con_id,
                   planned_live_market_data_lines, missing_reason
            FROM opening_reversal_contract_discovery_v1
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
    assert tuple(row) == (
        "failed",
        None,
        None,
        0,
        "no_valid_common_1dte_pair",
    )


def test_database_guard_rejects_generic_episode_discovery_identity(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "generic-discovery.sqlite3")
    database.migrate()
    metadata = _metadata("generic-discovery", RECEIPT_CREATED)
    database.create_run(metadata)

    with database._connect() as connection:
        envelope_id = database._insert_envelope(connection, metadata)
        with pytest.raises(
            sqlite3.IntegrityError,
            match="blocked_opening_reversal_episode_identity_missing",
        ):
            connection.execute(
                """
                INSERT INTO opening_reversal_contract_discovery_v1(
                    envelope_id, run_id, episode_id, discovery_timestamp_utc,
                    contract_source, cache_hit, candidates_inspected,
                    call_con_id, put_con_id, expiry, strike, tie_break_rule,
                    live_market_data_lines_consumed,
                    planned_live_market_data_lines, metadata_request_ended,
                    full_chain_live_subscription_created, status,
                    missing_reason, audit_hash_v1, audit_json
                ) VALUES (?, ?, 'quiet-or-generic-episode', ?, 'fixture',
                          0, 0, NULL, NULL, NULL, NULL, 'frozen', 0, 0,
                          1, 0, 'failed', 'unavailable', ?, '{}')
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    RECEIPT_CREATED.isoformat(),
                    "d" * 64,
                ),
            )


def test_database_guard_rejects_two_calls_even_with_distinct_roles(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "duplicate-right.sqlite3")
    database.migrate()
    metadata = _metadata("duplicate-right", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    receipt = _seed_eligible_v1_1_episode(database, repository, metadata)
    selection = _primary_selection()
    repository.record_opening_reversal_contract_discovery_v1(
        metadata,
        episode_id="fresh-aal",
        selection=selection,
    )
    predicted = selection.call if receipt.prediction_v1 == "CALL" else selection.put
    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=predicted,
            role="predicted_leg",
        ),
    )

    with database._connect() as connection:
        row = dict(
            connection.execute(
                """
                SELECT *
                FROM opening_reversal_primary_option_outcome_v1
                WHERE run_id = ?
                """,
                (metadata.run_id,),
            ).fetchone()
        )
        row.pop("id")
        row["role"] = "opposite_leg"
        row["outcome_hash_v1"] = "e" * 64
        columns = tuple(row)
        placeholders = ",".join("?" for _ in columns)
        with pytest.raises(
            sqlite3.IntegrityError,
            match="blocked_v1_1_outcome_duplicate_option_right",
        ):
            connection.execute(
                f"""
                INSERT INTO opening_reversal_primary_option_outcome_v1(
                    {",".join(columns)}
                ) VALUES ({placeholders})
                """,
                tuple(row[column] for column in columns),
            )


def test_database_guards_reject_identity_changing_evidence_updates(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "identity-update.sqlite3")
    database.migrate()
    metadata = _metadata("identity-update", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    receipt = _seed_eligible_v1_1_episode(database, repository, metadata)
    selection = _primary_selection()
    repository.record_opening_reversal_contract_discovery_v1(
        metadata,
        episode_id="fresh-aal",
        selection=selection,
    )
    predicted = selection.call if receipt.prediction_v1 == "CALL" else selection.put
    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=predicted,
            role="predicted_leg",
        ),
    )

    with database._connect() as connection:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="blocked_opening_reversal_contract_discovery_is_immutable",
        ):
            connection.execute(
                """
                UPDATE opening_reversal_contract_discovery_v1
                SET episode_id = 'quiet-state-episode'
                WHERE run_id = ?
                """,
                (metadata.run_id,),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="blocked_opening_reversal_option_outcome_is_immutable",
        ):
            connection.execute(
                """
                UPDATE opening_reversal_primary_option_outcome_v1
                SET prediction_receipt_hash_v1 = ?
                WHERE run_id = ?
                """,
                ("f" * 64, metadata.run_id),
            )


def test_v1_1_repository_rejects_quiet_state_episode_identity(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "quiet-alias.sqlite3")
    database.migrate()
    metadata = _metadata("quiet-alias", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    _seed_eligible_v1_1_episode(
        database,
        repository,
        metadata,
        quiet_state_alias=True,
    )

    with pytest.raises(ValueError, match="episode_not_eligible"):
        repository.record_opening_reversal_contract_discovery_v1(
            metadata,
            episode_id="fresh-aal",
            selection=_primary_selection(),
        )
    with database._connect() as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*)
            FROM opening_reversal_v1_1_eligible_episode
            WHERE run_id = ?
            """,
                (metadata.run_id,),
            ).fetchone()[0]
            == 0
        )


def test_v1_1_activation_requires_a_fresh_engineering_run(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "not-fresh.sqlite3")
    database.migrate()
    metadata = _metadata("not-fresh", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()
    repository.record_opening_reversal_activation_v1(metadata, base)
    # Rebuild through the public V1 builder so its immutable hash matches the
    # strict pre-entry receipt.
    source = OpeningReversalPredictionInputV1(
        experiment_version="1",
        activation_timestamp_utc=BASE_ACTIVATION,
        cohort_phase="prospective_development",
        transfer_status="cross_vendor_validation_not_configured",
        session=SESSION,
        stock="AAL",
        checkpoint=6,
        signal_timestamp_utc=ENTRY,
        entry_timestamp_utc=ENTRY,
        receipt_created_at_utc=ENTRY - timedelta(microseconds=1),
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
    )
    v1_receipt = build_prediction_receipt_v1(source)
    repository.record_opening_reversal_prediction_v1(
        metadata.model_copy(update={"recorded_at_utc": ENTRY - timedelta(microseconds=1)}),
        v1_receipt,
    )

    with pytest.raises(ValueError, match="fresh run"):
        repository.record_opening_reversal_activation_v1_1(
            metadata,
            activation,
        )


def test_checkpoint_engine_emits_non_scientific_v1_1_shadow_receipt_behind_causal_barrier(
    tmp_path: Path,
) -> None:
    class FakeFeatureBuilder:
        def build(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                scaled_features={"x": 1.0},
                feature_hash="e" * 64,
                scaling_artifact_hash="f" * 64,
            )

    class FakeRuntime:
        def missing_group_o_features(self, _: object) -> tuple[str, ...]:
            return ()

        def score(self, **_: object) -> FrozenM1CScore:
            return FrozenM1CScore(
                model_hash="b" * 64,
                probability=0.70,
                threshold=0.488333710794033,
                threshold_passed=True,
                feature_order=("x",),
                feature_values=(1.0,),
                transformed_values=(1.0,),
                feature_hash="c" * 64,
                missing_feature_count=0,
            )

    database = ProspectiveRepository(tmp_path / "engine-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("engine-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()
    repository.record_opening_reversal_activation_v1(metadata, base)
    repository.record_opening_reversal_activation_v1_1(metadata, activation)
    context = build_group_o_context(
        symbol="AAL",
        signal_session=SESSION,
        actual_option_observation_session=date(2026, 7, 29),
        front_expiry=date(2026, 7, 31),
        dte=1,
        atm_strike=100.0,
        previous_close_implied_movement_15m=0.01,
        features={"x": 1.0},
        missing_indicators={"x": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    repository.record_group_o_context(metadata, context)
    session_open = ENTRY - timedelta(minutes=30)
    stock_bars = tuple(
        LiveFeatureBar(
            symbol="AAL",
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=session_open + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=(session_open + timedelta(minutes=5 * (ordinal + 1))),
            open=100.0,
            high=101.0,
            low=99.0,
            close=99.5,
            volume=1_000.0,
            historical_relative_activity=1.0,
            finalised=True,
            source="fixture",
        )
        for ordinal in range(6)
    )
    market_bars = tuple(
        MarketShockBarV1(
            symbol="VTI",
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=session_open + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=(session_open + timedelta(minutes=5 * (ordinal + 1))),
            open=100.0,
            high=101.0,
            low=99.0,
            close=99.5,
            finalised=True,
        )
        for ordinal in range(6)
    )
    thresholds = OpeningTransitionThresholdsV1(
        market_opening_return_q10_v1=-0.00288963733897,
        market_opening_return_q90_v1=0.00225522676046,
        market_opening_range_q75_v1=0.00384818171835,
        market_overnight_gap_q10_v1=-0.00382056890751,
        market_overnight_gap_q90_v1=0.0063796856309,
        market_total_transition_q10_v1=-0.00536060944383,
        market_total_transition_q90_v1=0.00643755517767,
        market_opening_return_support_v1=247,
        market_opening_range_support_v1=247,
        market_overnight_gap_support_v1=247,
        market_total_transition_support_v1=247,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )
    engine = FrozenM1CRecorderEngine(
        m1c_runtime=cast(Any, FakeRuntime()),
        m1c_features=cast(Any, FakeFeatureBuilder()),
        direction_runtime=cast(Any, object()),
        direction_features=cast(Any, object()),
        repository=repository,
        opening_transition_thresholds_v1=thresholds,
        opening_transition_activation_status_v1="available",
        opening_reversal_activation_v1=base,
        opening_reversal_activation_v1_1=activation,
    )
    engine.set_opening_reversal_capacity_snapshot_provider_v1(lambda _metadata: "capacity-1")

    result = engine.process_checkpoint(
        RecorderCheckpointInput(
            metadata=metadata,
            symbol="AAL",
            session=SESSION,
            completed_m1c_bars=stock_bars,
            completed_direction_bars=(),
            group_o_context=context,
            market_data_type=MarketDataType.LIVE,
            capability_preflight_passed=True,
            m1c_parity_passed=True,
            direction_parity_passed=False,
            clock_drift_within_tolerance=True,
            underlying_quote_fresh=True,
            unresolved_bar_gap=False,
            raw_event_storage_writable=True,
            completed_market_shock_bars_v1=market_bars,
            market_previous_session_v1=date(2026, 7, 29),
            market_prior_regular_session_close_v1=100.0,
            opening_reversal_receipt_created_at_utc_v1_1=RECEIPT_CREATED,
            opening_reversal_first_buffered_event_received_at_utc_v1_1=(
                ENTRY + timedelta(milliseconds=1)
            ),
            opening_reversal_entry_data_admitted_before_receipt_v1_1=False,
            scientific_recording_authorized=False,
        )
    )

    receipt = result.opening_reversal_prediction_v1
    assert receipt is not None
    assert receipt.experiment_version == "1.1"
    assert receipt.eligibility_v1
    assert receipt.scientific_outcome_eligible_v1 is False
    assert receipt.scientific_exclusion_reason_v1 == "scientific_recording_not_authorized"
    assert receipt.receipt_created_at_utc == RECEIPT_CREATED
    assert receipt.timing_evidence_v1_1 is not None
    assert receipt.timing_evidence_v1_1.entry_or_post_entry_data_admitted_before_receipt is False


def test_v1_1_virtual_position_closes_only_after_both_primary_pair_outcomes(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "virtual-position-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("virtual-position-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    receipt = _seed_eligible_v1_1_episode(database, repository, metadata)
    selection = _primary_selection()
    repository.record_opening_reversal_contract_discovery_v1(
        metadata,
        episode_id="fresh-aal",
        selection=selection,
    )
    predicted = selection.call if receipt.prediction_v1 == "CALL" else selection.put
    opposite = selection.put if receipt.prediction_v1 == "CALL" else selection.call
    predicted_contract = OptionContract(
        underlying_con_id=1,
        con_id=predicted.con_id,
        expiry=predicted.expiry,
        dte=1,
        dte_bucket=DteBucket.ONE_DTE,
        strike=predicted.strike,
        right=predicted.right,
        multiplier=predicted.multiplier,
        exchange=predicted.exchange,
        trading_class=predicted.trading_class,
    )
    option_contract_id = repository.record_option_contract(
        metadata,
        episode_id="fresh-aal",
        contract=predicted_contract,
        selection_rank=1,
        resolution_status="recording",
        rejection_reason=None,
        recording_started_at_utc=ENTRY,
        recording_ends_at_utc=ENTRY + timedelta(minutes=30),
    )
    repository.update_option_quote_projection(
        option_contract_id=option_contract_id,
        event=OptionQuoteEvent(
            event_id="virtual-position-latest-quote",
            received_timestamp_utc=ENTRY + timedelta(seconds=2),
            received_monotonic_ns=1,
            provider_timestamp_utc=ENTRY + timedelta(seconds=2),
            source_sequence=1,
            session=SESSION,
            symbol="AAL",
            con_id=predicted.con_id,
            request_id=101,
            episode_id="fresh-aal",
            expiry=predicted.expiry,
            dte=1,
            dte_bucket=DteBucket.ONE_DTE,
            strike=predicted.strike,
            right=predicted.right,
            multiplier=predicted.multiplier,
            exchange=predicted.exchange,
            trading_class=predicted.trading_class,
            bid=1.05,
            bid_size=10.0,
            ask=1.15,
            ask_size=12.0,
            last=1.10,
            last_size=1.0,
            market_data_type=MarketDataType.LIVE,
        ),
        recording_status="recording",
        quote_quality_flags=(),
    )

    with database._connect() as connection:
        capturing = connection.execute(
            "SELECT * FROM opening_reversal_virtual_position_v1"
        ).fetchone()
    assert capturing is not None
    assert capturing["lifecycle_state"] == "CAPTURING"
    assert capturing["con_id"] == predicted.con_id
    assert capturing["role"] == "predicted_leg"
    assert capturing["planned_live_market_data_lines"] == 2
    assert capturing["pair_outcome_count"] == 0
    assert capturing["latest_observed_bid"] == pytest.approx(1.05)
    assert capturing["latest_observed_ask"] == pytest.approx(1.15)

    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=predicted,
            role="predicted_leg",
        ),
    )
    with database._connect() as connection:
        still_capturing = connection.execute(
            "SELECT * FROM opening_reversal_virtual_position_v1"
        ).fetchone()
    assert still_capturing["lifecycle_state"] == "CAPTURING"
    assert still_capturing["pair_outcome_count"] == 1
    assert still_capturing["status_reason"] == "awaiting_opposite_control_outcome"
    assert still_capturing["predicted_outcome_present"] == 1
    assert still_capturing["opposite_outcome_present"] == 0

    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=opposite,
            role="opposite_leg",
        ),
    )
    with database._connect() as connection:
        rows = connection.execute("SELECT * FROM opening_reversal_virtual_position_v1").fetchall()

    assert len(rows) == 1
    closed = rows[0]
    assert closed["lifecycle_state"] == "CLOSED"
    assert closed["prediction_receipt_hash_v1"] == receipt.receipt_hash_v1
    assert closed["experiment_id"] == "m1c-prospective-opening-reversal-v1"
    assert closed["experiment_version"] == "1.1"
    assert closed["right"] == predicted.right
    assert closed["dte"] == 1
    assert closed["quantity"] == 1
    assert closed["entry_ask"] == pytest.approx(1.1)
    assert closed["exit_bid"] == pytest.approx(1.2)
    assert datetime.fromisoformat(closed["exit_quote_timestamp_utc"]) > (
        ENTRY + timedelta(minutes=15)
    )
    assert closed["exit_convention"] == "first_valid_live_bid_at_or_after_frozen_15m_horizon"
    assert closed["gross_quote_pnl"] == pytest.approx(10.0)
    assert closed["execution_claimed"] == 0
    assert closed["paper_fill_claimed"] == 0

    projected = ProspectiveReadStore(
        database.database_path,
        run_id=metadata.run_id,
    ).opening_reversal_virtual_positions_v1()
    assert len(projected) == 1
    assert projected[0]["lifecycle_state"] == "CLOSED"
    assert projected[0]["right"] == predicted.right
    assert projected[0]["gross_quote_pnl"] == pytest.approx(10.0)


def test_v1_1_virtual_position_names_the_predicted_leg_when_control_arrives_first(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "control-arrives-first.sqlite3")
    database.migrate()
    metadata = _metadata("control-arrives-first", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    receipt = _seed_eligible_v1_1_episode(database, repository, metadata)
    selection = _primary_selection()
    repository.record_opening_reversal_contract_discovery_v1(
        metadata,
        episode_id="fresh-aal",
        selection=selection,
    )
    opposite = selection.put if receipt.prediction_v1 == "CALL" else selection.call
    repository.record_opening_reversal_primary_option_outcome_v1(
        metadata,
        _primary_outcome(
            receipt_hash=receipt.receipt_hash_v1,
            contract=opposite,
            role="opposite_leg",
        ),
    )

    with database._connect() as connection:
        capturing = connection.execute(
            "SELECT * FROM opening_reversal_virtual_position_v1"
        ).fetchone()

    assert capturing["lifecycle_state"] == "CAPTURING"
    assert capturing["status_reason"] == "awaiting_predicted_leg_outcome"
    assert capturing["predicted_outcome_present"] == 0
    assert capturing["opposite_outcome_present"] == 1
