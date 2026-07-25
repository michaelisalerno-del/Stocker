from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.config import (
    ProspectiveConfig,
    RuntimeSafetyError,
    validate_persistent_paths,
    validate_runtime_safety,
)
from stocker_prospective.context import (
    ContextValidationError,
    DailyContextUnsigned,
    create_signed_context,
    import_signed_context,
    load_imported_context,
    previous_xnys_session,
    verify_signed_context,
)
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    RecorderLeaseHeld,
    ScoreInput,
)
from stocker_prospective.signal import SignalEventizer


def _config(tmp_path: Path, **runtime_updates: object) -> ProspectiveConfig:
    runtime: dict[str, object] = {
        "mode": "record_only",
        "source": "replay",
        "prospective_start_utc": "2026-07-20T00:00:00Z",
        "instance_id": "test-server",
        "app_version": "0.1.0",
        "git_commit": "0d6de083",
    }
    runtime.update(runtime_updates)
    return ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": str(tmp_path / "shared" / "prospective.sqlite3"),
                "bundle_root": str(tmp_path / "shared" / "bundles"),
                "feature_parity_report": str(tmp_path / "feature-parity.json"),
            },
            "runtime": runtime,
            "risk": {"trading_enabled": False},
            "web": {"host": "127.0.0.1", "port": 8765},
            "ibkr": {},
            "context": {"mode": "signed_import", "hmac_secret_env": "CONTEXT_SECRET"},
        }
    )


class MarketDataOnly:
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...


class UnsafeOrderCapable:
    async def place_order(self, *_: object) -> None: ...


def test_runtime_configuration_fails_closed_and_defaults_to_loopback(tmp_path: Path) -> None:
    config = _config(tmp_path)

    validate_runtime_safety(config, MarketDataOnly())

    assert config.web.host == "127.0.0.1"
    assert config.runtime.mode == "record_only"
    with pytest.raises(RuntimeSafetyError, match="blocked_unsafe_runtime_configuration"):
        validate_runtime_safety(config, UnsafeOrderCapable())
    with pytest.raises(ValueError):
        _config(tmp_path, mode="live")
    unsafe = config.model_copy(
        update={"risk": config.risk.model_copy(update={"trading_enabled": True})}
    )
    with pytest.raises(RuntimeSafetyError, match="risk.trading_enabled"):
        validate_runtime_safety(unsafe, MarketDataOnly())
    with pytest.raises(ValueError, match="web host may not bind all interfaces"):
        ProspectiveConfig.model_validate(
            {
                **config.model_dump(),
                "web": {**config.web.model_dump(), "host": "0.0.0.0"},
            }
        )
    with pytest.raises(ValueError, match="IBKR host must be a literal loopback address"):
        ProspectiveConfig.model_validate(
            {
                **config.model_dump(),
                "ibkr": {**config.ibkr.model_dump(), "host": "10.0.0.5"},
            }
        )
    with pytest.raises(ValueError, match="heartbeat"):
        _config(
            tmp_path,
            heartbeat_seconds=10,
            recorder_lease_stale_seconds=20,
        )


def test_runtime_paths_cannot_depend_on_release_directory(tmp_path: Path) -> None:
    release = tmp_path / "releases" / "2026-07-24"
    release.mkdir(parents=True)
    config = _config(tmp_path)
    validate_persistent_paths(config, release)

    nested = config.model_copy(
        update={
            "paths": config.paths.model_copy(update={"database": release / "prospective.sqlite3"})
        }
    )
    with pytest.raises(RuntimeSafetyError, match="outside the application release"):
        validate_persistent_paths(nested, release)


def test_existing_prospective_run_identity_cannot_be_reused_with_different_commit(
    tmp_path: Path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    metadata = _metadata()
    repository.create_run(metadata)
    changed = metadata.model_copy(update={"git_commit": "f" * 40})

    with pytest.raises(ValueError, match="prospective run identity mismatch"):
        repository.create_run(changed)


def _unsigned_context(observation_date: date) -> DailyContextUnsigned:
    return DailyContextUnsigned(
        manifest_version="1",
        context_id=f"context-{observation_date.isoformat()}",
        session_date=observation_date,
        provider="eodhd_options_eod",
        source_record_ids=["record-1", "record-2"],
        created_at_utc=datetime(2026, 7, 20, 23, 0, tzinfo=UTC),
        schema_hash="a" * 64,
        feature_hash="b" * 64,
        completeness="complete",
        features_by_symbol={"AAL": {"front_options_implied_tension": 0.12}},
        key_id="daily-context-test",
    )


def test_signed_context_requires_exact_previous_xnys_session() -> None:
    current_session = date(2026, 7, 21)
    assert previous_xnys_session(current_session) == date(2026, 7, 20)
    package = create_signed_context(_unsigned_context(date(2026, 7, 20)), b"secret")

    verified = verify_signed_context(
        package,
        current_session=current_session,
        secret=b"secret",
        expected_schema_hash="a" * 64,
        expected_feature_hash="b" * 64,
    )

    assert verified.session_date == date(2026, 7, 20)
    assert verified.integrity_hash
    with pytest.raises(ContextValidationError, match="invalid_context_signature"):
        verify_signed_context(package, current_session=current_session, secret=b"wrong")
    with pytest.raises(
        ContextValidationError,
        match="anchor symbol context is incomplete",
    ):
        verify_signed_context(
            package,
            current_session=current_session,
            secret=b"secret",
            expected_symbols=("AAL", "AAOI"),
        )
    for invalid_date in (current_session, date(2026, 7, 22), date(2026, 7, 17)):
        invalid = create_signed_context(_unsigned_context(invalid_date), b"secret")
        with pytest.raises(
            ContextValidationError,
            match="blocked_missing_previous_session_options_context",
        ):
            verify_signed_context(
                invalid,
                current_session=current_session,
                secret=b"secret",
            )


def test_daily_context_import_is_exact_idempotent_and_not_stale_selected(
    tmp_path: Path,
) -> None:
    current_session = date(2026, 7, 21)
    source = tmp_path / "transfer" / "context.json"
    source.parent.mkdir()
    package = create_signed_context(_unsigned_context(date(2026, 7, 20)), b"secret")
    source.write_text(package.model_dump_json(), encoding="utf-8")
    root = tmp_path / "server-shared-context"

    first = import_signed_context(
        source,
        context_root=root,
        current_session=current_session,
        secret=b"secret",
        operator="test-operator",
    )
    source.unlink()
    second = load_imported_context(
        context_root=root,
        current_session=current_session,
        secret=b"secret",
    )

    assert first.context_id == second.context_id
    assert second.session_date == date(2026, 7, 20)
    assert (root / "sessions" / "2026-07-21.json").is_file()
    with pytest.raises(
        ContextValidationError,
        match="blocked_missing_previous_session_options_context",
    ):
        load_imported_context(
            context_root=root,
            current_session=date(2026, 7, 22),
            secret=b"secret",
        )


def _metadata(run_id: str = "prospective-test") -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=datetime(2026, 7, 20, tzinfo=UTC),
        app_version="0.1.0",
        git_commit="0d6de083",
        model_artifact_id="synthetic_replay_not_frozen_m1",
        universe_id="anchor-frozen-20-v1",
        cohort="anchor_frozen_20",
        source_timestamps=["2026-07-21T14:00:00Z"],
        recorded_at_utc=datetime(2026, 7, 21, 14, 0, 1, tzinfo=UTC),
    )


def _score(timestamp: datetime, probability: float) -> ScoreInput:
    return ScoreInput(
        metadata=_metadata(),
        symbol="AAL",
        bar_end_utc=timestamp,
        session_date=date(2026, 7, 21),
        feature_as_of_utc=timestamp,
        m0_probability=max(0.0, probability - 0.1),
        m1_probability=probability,
        frozen_threshold=0.5,
        feature_schema_hash="c" * 64,
        eligibility=True,
        rejection_reason=None,
        score_label="synthetic_replay_not_frozen_m1",
    )


def test_recorder_lease_is_single_owner_with_heartbeat_and_stale_recovery(
    tmp_path: Path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    now = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    lease = repository.acquire_recorder_lease(
        run_id="prospective-test",
        owner_id="owner-a",
        now=now,
        stale_after=timedelta(seconds=30),
    )
    with pytest.raises(RecorderLeaseHeld, match="blocked_recorder_lease_held"):
        repository.acquire_recorder_lease(
            run_id="prospective-test",
            owner_id="owner-b",
            now=now + timedelta(seconds=5),
            stale_after=timedelta(seconds=30),
        )
    heartbeat = repository.heartbeat_recorder_lease(
        run_id="prospective-test",
        owner_id="owner-a",
        now=now + timedelta(seconds=10),
    )
    recovered = repository.acquire_recorder_lease(
        run_id="prospective-test",
        owner_id="owner-b",
        now=now + timedelta(seconds=41),
        stale_after=timedelta(seconds=30),
    )

    assert lease.owner_id == "owner-a"
    assert heartbeat.heartbeat_at_utc == now + timedelta(seconds=10)
    assert recovered.owner_id == "owner-b"
    assert recovered.recovered_stale_owner is True


def test_recorder_anchor_keeps_wal_coordination_files_available(tmp_path: Path) -> None:
    database = tmp_path / "prospective.sqlite3"
    repository = ProspectiveRepository(database)
    repository.migrate()

    repository.open_anchor()
    try:
        with sqlite3.connect(database) as writer:
            assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
            writer.execute("SELECT count(*) FROM schema_migrations").fetchone()
        assert Path(f"{database}-wal").is_file()
        assert Path(f"{database}-shm").is_file()
    finally:
        repository.close_anchor()


def test_recorder_lease_can_only_be_released_by_its_exact_owner(tmp_path: Path) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    now = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    repository.acquire_recorder_lease(
        run_id="prospective-test",
        owner_id="process-a",
        now=now,
        stale_after=timedelta(seconds=30),
    )

    assert (
        repository.release_recorder_lease(
            run_id="prospective-test",
            owner_id="process-b",
        )
        is False
    )
    assert repository.release_recorder_lease(
        run_id="prospective-test",
        owner_id="process-a",
    )
    reacquired = repository.acquire_recorder_lease(
        run_id="prospective-test",
        owner_id="process-b",
        now=now + timedelta(seconds=1),
        stale_after=timedelta(seconds=30),
    )

    assert reacquired.owner_id == "process-b"


def test_signal_eventisation_is_crossing_based_and_restart_idempotent(tmp_path: Path) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    repository.create_run(_metadata())
    eventizer = SignalEventizer(repository)
    start = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)

    below = eventizer.record(_score(start, 0.40))
    first = eventizer.record(_score(start + timedelta(minutes=5), 0.52))
    continuing = eventizer.record(_score(start + timedelta(minutes=10), 0.55))
    duplicate = SignalEventizer(repository).record(_score(start + timedelta(minutes=10), 0.55))
    reset = eventizer.record(_score(start + timedelta(minutes=15), 0.45))
    second = eventizer.record(_score(start + timedelta(minutes=20), 0.60))

    assert below.status == "below_threshold"
    assert first.status == "crossing" and first.episode_id
    assert continuing.episode_id == first.episode_id
    assert duplicate.episode_id == first.episode_id
    assert reset.status == "below_threshold"
    assert second.status == "crossing" and second.episode_id != first.episode_id
    assert repository.count("signal_episode") == 2
    assert repository.count("signal_checkpoint") == 3
    assert repository.count("signal_eventization") == 5

    other = ProspectiveRepository(tmp_path / "startup.sqlite3")
    other.migrate()
    other.create_run(_metadata(run_id="startup-test"))
    startup_score = _score(start, 0.70)
    startup_score.metadata = _metadata(run_id="startup-test")
    startup = SignalEventizer(other).record(startup_score)
    assert startup.status == "startup_above_threshold"
    assert startup.episode_id is None
    assert other.count("signal_episode") == 0
    assert other.count("signal_eventization") == 1
