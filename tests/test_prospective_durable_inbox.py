from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.durable_inbox import (
    CallbackClassification,
    CallbackInboxError,
    CallbackInboxOverflow,
    CallbackLeaseLost,
    DurableCallbackInbox,
)

NOW = datetime(2026, 7, 29, 10, 0, tzinfo=UTC)
RUN_ID = "run-hardening"


def durable_inbox(path: Path, **kwargs: Any) -> DurableCallbackInbox:
    return DurableCallbackInbox(
        path,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
        **kwargs,
    )


def migrated_database(tmp_path: Path) -> Path:
    path = tmp_path / "prospective.sqlite3"
    repository = ProspectiveRepository(path)
    repository.migrate()
    repository.create_run(
        EvidenceMetadata(
            run_id="run-hardening",
            prospective_start_utc=NOW - timedelta(days=1),
            app_version="test",
            git_commit="a" * 40,
            model_artifact_id="frozen-m1c",
            universe_id="anchor-frozen-20",
            cohort="anchor_frozen_20",
            source_timestamps=[NOW.isoformat()],
            recorded_at_utc=NOW,
        )
    )
    return path


def admit(
    inbox: DurableCallbackInbox,
    *,
    event_id: str,
    request_id: int = 7,
) -> int:
    result = inbox.admit(
        callback_kind="level1_quote_update",
        request_id=request_id,
        payload={
            "field": "bid",
            "value": None,
            "provider_timestamp_utc": NOW.isoformat(),
        },
        connection_generation=2,
        classification=CallbackClassification.ACCEPTED_ACTIVE,
        received_utc=NOW,
        received_monotonic_ns=123,
        inbox_event_id=event_id,
        subscription_owner="AAL:level1",
        symbol="AAL",
    )
    return result.event.source_sequence


def test_durable_insert_survives_restart_before_first_lease(tmp_path: Path) -> None:
    path = migrated_database(tmp_path)
    first = durable_inbox(path)
    source_sequence = admit(first, event_id="event-before-lease")

    restarted = durable_inbox(path)
    leased = restarted.lease(
        lease_owner="recorder-new",
        lease_generation=3,
        now=NOW + timedelta(seconds=1),
        lease_timeout=timedelta(seconds=30),
        limit=10,
    )

    assert len(leased) == 1
    assert leased[0].source_sequence == source_sequence
    assert leased[0].original_payload["value"] is None
    assert restarted.accounting().pending == 0
    assert restarted.accounting().leased == 1


def test_expired_lease_is_reclaimed_and_old_generation_cannot_ack(tmp_path: Path) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path)
    admit(inbox, event_id="event-expired")
    old = inbox.lease(
        lease_owner="old",
        lease_generation=2,
        now=NOW,
        lease_timeout=timedelta(seconds=5),
        limit=1,
    )
    new = inbox.lease(
        lease_owner="new",
        lease_generation=3,
        now=NOW + timedelta(seconds=6),
        lease_timeout=timedelta(seconds=5),
        limit=1,
    )
    inbox.commit_processing(
        new,
        run_id="run-hardening",
        recorder_generation=3,
        raw_partition_hashes=("abc",),
        committed_at=NOW + timedelta(seconds=7),
    )

    with pytest.raises(CallbackLeaseLost, match="CALLBACK_ACK_LEASE_CHANGED"):
        inbox.acknowledge(
            old,
            lease_owner="old",
            lease_generation=2,
            raw_partition_hashes=("abc",),
            acknowledged_at=NOW + timedelta(seconds=7),
        )

    assert (
        inbox.acknowledge(
            new,
            lease_owner="new",
            lease_generation=3,
            raw_partition_hashes=("abc",),
            acknowledged_at=NOW + timedelta(seconds=8),
        )
        == 1
    )
    accounting = inbox.accounting()
    assert accounting.acknowledged == 1
    assert accounting.highest_source_sequence == accounting.highest_acknowledged_sequence


def test_duplicate_delivery_and_retry_after_ack_do_not_duplicate_event(
    tmp_path: Path,
) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path)
    first_sequence = admit(inbox, event_id="stable-provider-delivery")
    duplicate_sequence = admit(inbox, event_id="stable-provider-delivery")
    assert first_sequence == duplicate_sequence
    assert inbox.accounting().admitted == 1

    events = inbox.lease(
        lease_owner="recorder",
        lease_generation=1,
        now=NOW,
        lease_timeout=timedelta(seconds=5),
        limit=5,
    )
    inbox.commit_processing(
        events,
        run_id="run-hardening",
        recorder_generation=1,
        raw_partition_hashes=("hash-1",),
        committed_at=NOW,
    )
    inbox.acknowledge(
        events,
        lease_owner="recorder",
        lease_generation=1,
        raw_partition_hashes=("hash-1",),
        acknowledged_at=NOW,
    )
    assert (
        inbox.compact_acknowledged(
            before=NOW + timedelta(seconds=1),
            retention_policy_enabled=True,
        )
        == 1
    )
    assert admit(inbox, event_id="stable-provider-delivery") == first_sequence
    assert (
        inbox.lease(
            lease_owner="recorder",
            lease_generation=1,
            now=NOW + timedelta(seconds=1),
            lease_timeout=timedelta(seconds=5),
            limit=5,
        )
        == ()
    )


def test_poison_event_is_quarantined_without_acknowledgement(tmp_path: Path) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path)
    admit(inbox, event_id="poison")
    leased = inbox.lease(
        lease_owner="recorder",
        lease_generation=1,
        now=NOW,
        lease_timeout=timedelta(seconds=30),
        limit=1,
    )
    inbox.quarantine(
        leased[0],
        failure_classification="CALLBACK_NORMALIZATION_FAILED",
        lease_owner="recorder",
        lease_generation=1,
        now=NOW,
    )

    accounting = inbox.accounting()
    assert accounting.quarantined == 1
    assert accounting.acknowledged == 0
    assert accounting.highest_acknowledged_sequence == 0


def test_inbox_overflow_is_explicit_and_does_not_drop_existing_row(
    tmp_path: Path,
) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path, max_unacknowledged=1)
    admit(inbox, event_id="first")

    with pytest.raises(CallbackInboxOverflow, match="CALLBACK_OVERFLOW"):
        admit(inbox, event_id="second")

    accounting = inbox.accounting()
    assert accounting.admitted == 1
    assert accounting.pending == 1


def test_quarantined_events_count_toward_the_bounded_unacknowledged_backlog(
    tmp_path: Path,
) -> None:
    inbox = durable_inbox(
        migrated_database(tmp_path),
        max_unacknowledged=1,
    )
    inbox.admit(
        inbox_event_id="poison",
        callback_kind="level1_quote_update",
        request_id=7,
        payload={"value": {"malformed": True}},
        connection_generation=1,
        classification=CallbackClassification.UNKNOWN,
        received_utc=NOW,
        received_monotonic_ns=1,
    )

    with pytest.raises(CallbackInboxOverflow, match="CALLBACK_OVERFLOW"):
        admit(inbox, event_id="would-exceed-bound")

    accounting = inbox.accounting()
    assert accounting.quarantined == 1
    assert accounting.highest_acknowledged_sequence == 0


def test_sqlite_busy_is_reported_to_callback_boundary_without_partial_insert(
    tmp_path: Path,
) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path, busy_timeout_ms=10)
    locker = sqlite3.connect(path)
    try:
        locker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            admit(inbox, event_id="locked")
    finally:
        locker.rollback()
        locker.close()

    assert inbox.accounting().admitted == 0


def test_expired_batch_membership_excludes_newly_arrived_callback(
    tmp_path: Path,
) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path)
    admit(inbox, event_id="batch-a")
    first = inbox.lease(
        lease_owner="old",
        lease_generation=1,
        now=NOW,
        lease_timeout=timedelta(seconds=5),
        limit=10,
    )
    assert tuple(event.inbox_event_id for event in first) == ("batch-a",)
    admit(inbox, event_id="new-b")

    retried = inbox.lease(
        lease_owner="new",
        lease_generation=2,
        now=NOW + timedelta(seconds=6),
        lease_timeout=timedelta(seconds=5),
        limit=10,
    )

    assert tuple(event.inbox_event_id for event in retried) == ("batch-a",)
    assert retried[0].lease_batch_id == first[0].lease_batch_id
    inbox.commit_processing(
        retried,
        run_id=RUN_ID,
        recorder_generation=2,
        raw_partition_hashes=("stable-a",),
        committed_at=NOW + timedelta(seconds=6),
    )
    inbox.acknowledge(
        retried,
        lease_owner="new",
        lease_generation=2,
        raw_partition_hashes=("stable-a",),
        acknowledged_at=NOW + timedelta(seconds=6),
    )
    following = inbox.lease(
        lease_owner="new",
        lease_generation=2,
        now=NOW + timedelta(seconds=7),
        lease_timeout=timedelta(seconds=5),
        limit=10,
    )
    assert tuple(event.inbox_event_id for event in following) == ("new-b",)
    assert following[0].lease_batch_id != retried[0].lease_batch_id


def test_pending_callback_from_abandoned_run_is_run_scoped(
    tmp_path: Path,
) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path)
    admit(inbox, event_id="old-run-evidence")
    repository = ProspectiveRepository(path)
    repository.create_run(
        EvidenceMetadata(
            run_id="run-new",
            prospective_start_utc=NOW - timedelta(days=1),
            app_version="test",
            git_commit="b" * 40,
            model_artifact_id="frozen-m1c",
            universe_id="anchor-frozen-20",
            cohort="anchor_frozen_20",
            source_timestamps=[NOW.isoformat()],
            recorded_at_utc=NOW,
        )
    )
    inbox.configure_recorder(
        run_id="run-new",
        recorder_generation=1,
        owner_id="new-recorder",
    )
    admit(inbox, event_id="new-run-evidence")

    leased = inbox.lease(
        lease_owner="new-recorder",
        lease_generation=1,
        now=NOW,
        lease_timeout=timedelta(seconds=5),
        limit=10,
    )

    assert tuple(event.inbox_event_id for event in leased) == ("new-run-evidence",)
    assert inbox.accounting().admitted == 1
    with inbox._connect() as connection:
        row = connection.execute(
            """
            SELECT status, admission_run_id
            FROM callback_inbox_v1
            WHERE inbox_event_id = 'old-run-evidence'
            """
        ).fetchone()
    assert tuple(row) == ("pending", RUN_ID)


def test_provider_envelope_and_canonical_row_transition_atomically(
    tmp_path: Path,
) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path)
    provider = inbox.admit(
        callback_kind="official_provider_tick_price",
        request_id=7,
        payload={"provider_arguments": [7, 1, 10.0]},
        connection_generation=2,
        classification=CallbackClassification.ACCEPTED_ACTIVE,
        received_utc=NOW,
        received_monotonic_ns=100,
        inbox_event_id="provider-envelope",
        provider_envelope=True,
    )
    assert provider.event.status.value == "provider_pending"

    canonical = inbox.admit(
        callback_kind="level1_quote_update",
        request_id=7,
        payload={"field": "bid", "value": 10.0},
        connection_generation=2,
        classification=CallbackClassification.ACCEPTED_ACTIVE,
        received_utc=NOW,
        received_monotonic_ns=101,
        inbox_event_id="canonical-event",
        provider_envelope_event_id="provider-envelope",
    )

    assert canonical.event.provider_envelope_event_id == "provider-envelope"
    with inbox._connect() as connection:
        rows = connection.execute(
            """
            SELECT inbox_event_id, status
            FROM callback_inbox_v1
            ORDER BY source_sequence
            """
        ).fetchall()
    assert tuple(tuple(row) for row in rows) == (
        ("provider-envelope", "diagnostic"),
        ("canonical-event", "pending"),
    )


def test_restart_quarantines_interrupted_provider_materialisation(
    tmp_path: Path,
) -> None:
    path = migrated_database(tmp_path)
    inbox = durable_inbox(path)
    admission = inbox.admit(
        callback_kind="official_provider_tick_price",
        request_id=7,
        payload={"provider_arguments": [7, 1, 10.0]},
        connection_generation=2,
        classification=CallbackClassification.ACCEPTED_ACTIVE,
        received_utc=NOW,
        received_monotonic_ns=100,
        inbox_event_id="interrupted-provider",
        provider_envelope=True,
    )
    assert admission.event.status.value == "provider_pending"
    inbox.configure_recorder(
        run_id=RUN_ID,
        recorder_generation=2,
        owner_id="replacement",
    )

    quarantined = inbox.quarantine_interrupted_provider_envelopes(
        current_recorder_generation=2,
        observed_at=NOW + timedelta(seconds=10),
    )

    assert tuple(event.inbox_event_id for event in quarantined) == ("interrupted-provider",)
    with inbox._connect() as connection:
        row = connection.execute(
            """
            SELECT status, failure_classification
            FROM callback_inbox_v1
            WHERE inbox_event_id = 'interrupted-provider'
            """
        ).fetchone()
    assert tuple(row) == (
        "quarantined",
        "CALLBACK_PROVIDER_MATERIALIZATION_INTERRUPTED",
    )


def test_empty_lease_has_no_precommitted_raw_materialisation(
    tmp_path: Path,
) -> None:
    inbox = durable_inbox(migrated_database(tmp_path))

    assert inbox.raw_materialization(()) is None


def test_raw_materialisation_fences_partition_and_event_identities(
    tmp_path: Path,
) -> None:
    inbox = durable_inbox(migrated_database(tmp_path))
    admit(inbox, event_id="callback-a")
    leased = inbox.lease(
        lease_owner="recorder",
        lease_generation=1,
        now=NOW,
        lease_timeout=timedelta(seconds=30),
        limit=10,
    )
    inbox.commit_raw_materialization(
        leased,
        run_id=RUN_ID,
        recorder_generation=1,
        raw_partition_hashes=("a" * 64,),
        raw_event_ids=("raw-a", "derived-a"),
        materialized_at=NOW,
    )

    materialization = inbox.raw_materialization(leased)
    assert materialization is not None
    assert materialization.partition_hashes == ("a" * 64,)
    assert materialization.raw_event_ids == ("derived-a", "raw-a")
    with pytest.raises(
        CallbackInboxError,
        match="CALLBACK_RAW_MATERIALIZATION_DIFFERS",
    ):
        inbox.commit_raw_materialization(
            leased,
            run_id=RUN_ID,
            recorder_generation=1,
            raw_partition_hashes=("a" * 64,),
            raw_event_ids=("raw-a", "derived-b"),
            materialized_at=NOW,
        )


def test_poison_resolution_requires_explicit_event_and_latch_evidence(
    tmp_path: Path,
) -> None:
    inbox = durable_inbox(migrated_database(tmp_path))
    admit(inbox, event_id="poison")
    leased = inbox.lease(
        lease_owner="recorder",
        lease_generation=1,
        now=NOW,
        lease_timeout=timedelta(seconds=30),
        limit=1,
    )
    inbox.quarantine(
        leased[0],
        failure_classification="CALLBACK_NORMALIZATION_FAILED",
        lease_owner="recorder",
        lease_generation=1,
        now=NOW,
    )
    inbox.latch_fatal(
        latch_kind="ingestion",
        stable_error_code="CALLBACK_NORMALIZATION_FAILED",
        occurred_at=NOW,
        error_class="ValueError",
        evidence_loss_possible=True,
        first_possibly_lost_source_sequence=leased[0].source_sequence,
    )

    with pytest.raises(
        CallbackInboxError,
        match="CALLBACK_FATAL_RESOLUTION_HAS_QUARANTINED_EVENTS",
    ):
        inbox.resolve_fatal_latch(
            latch_kind="ingestion",
            expected_stable_error_code="CALLBACK_NORMALIZATION_FAILED",
            resolution_evidence="independent fixture proves normalizer fix",
            resolved_at=NOW + timedelta(seconds=1),
        )
    inbox.release_quarantined_for_retry(
        inbox_event_id="poison",
        resolution_evidence="independent fixture proves normalizer fix",
        resolved_at=NOW + timedelta(seconds=2),
    )
    inbox.resolve_fatal_latch(
        latch_kind="ingestion",
        expected_stable_error_code="CALLBACK_NORMALIZATION_FAILED",
        resolution_evidence="operator audit incident-123",
        resolved_at=NOW + timedelta(seconds=3),
    )

    assert not inbox.has_active_fatal("ingestion")
    assert inbox.accounting().pending == 1
