from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    RecorderLeaseHeld,
    SourceBarObservationInput,
)
from stocker_prospective.parallel import (
    EODHDParallelBarProvider,
    ParallelSourceCaptureService,
)
from stocker_prospective.recorder import RecorderDeploymentIdentity


def _metadata(recorded_at: datetime) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id="prospective-parallel-test",
        prospective_start_utc=datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
        app_version="0.1.0",
        git_commit="a" * 40,
        model_artifact_id="m1-feature-runtime-test-v1",
        universe_id="anchor-frozen-20-v1",
        cohort="anchor_frozen_20",
        source_timestamps=["2026-07-27T13:35:00+00:00"],
        recorded_at_utc=recorded_at,
    )


def test_parallel_source_bar_is_append_oriented_idempotent_and_never_scores(
    tmp_path: Path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    metadata = _metadata(datetime(2026, 7, 27, 22, tzinfo=UTC))
    repository.create_run(metadata)
    observed = SourceBarObservationInput(
        metadata=metadata,
        provider="eodhd",
        provider_record_id="AAPL.US:2026-07-27T13:30:00Z",
        symbol="AAPL",
        session_date=date(2026, 7, 27),
        bar_start_utc=datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
        bar_end_utc=datetime(2026, 7, 27, 13, 35, tzinfo=UTC),
        open=210.0,
        high=211.0,
        low=209.5,
        close=210.5,
        activity_value=None,
        activity_semantic_label="eodhd_historical_activity_proxy",
        source_timestamp_utc=datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
        receive_timestamp_utc=datetime(2026, 7, 27, 22, tzinfo=UTC),
        completeness="complete",
        eligibility=False,
        rejection_reason="parallel_validation_only",
    )

    first = repository.record_source_bar_observation(observed)
    second = repository.record_source_bar_observation(observed)

    assert first == second
    assert repository.count("source_bar_observation") == 1
    with repository._connect() as connection:
        row = connection.execute("SELECT * FROM source_bar_observation").fetchone()
    assert row is not None
    assert row["activity_value"] is None
    assert row["eligibility"] == 0
    assert row["rejection_reason"] == "parallel_validation_only"

    retried = observed.model_copy(
        update={
            "receive_timestamp_utc": observed.receive_timestamp_utc
            + timedelta(minutes=10)
        }
    )
    assert repository.record_source_bar_observation(retried) == first
    assert repository.count("source_bar_observation") == 1


class _FakeEODHDClient:
    def __init__(self, payload: list[dict[str, object]]) -> None:
        self.payload = payload
        self.requests: list[tuple[str, str, datetime, datetime]] = []

    def require_token(self) -> str:
        return "configured-not-exposed"

    def fetch_intraday_chunk(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, object]]:
        self.requests.append((symbol, interval, start, end))
        return self.payload


def test_eodhd_parallel_provider_uses_exact_rth_and_preserves_missing_activity() -> None:
    bar_start = datetime(2026, 7, 27, 13, 30, tzinfo=UTC)
    client = _FakeEODHDClient(
        [
            {
                "timestamp": int((bar_start - timedelta(minutes=5)).timestamp()),
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            },
            {
                "timestamp": int(bar_start.timestamp()),
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": None,
            },
        ]
    )
    provider = EODHDParallelBarProvider(client=client)

    bars = provider.fetch_session(
        symbol="AAL",
        session_date=date(2026, 7, 27),
        received_at_utc=datetime(2026, 7, 27, 22, tzinfo=UTC),
    )

    assert len(bars) == 1
    assert bars[0].bar_start_utc == bar_start
    assert bars[0].bar_end_utc == bar_start + timedelta(minutes=5)
    assert bars[0].activity_value is None
    assert bars[0].completeness == "partial"
    assert client.requests[0][0] == "AAL.US"
    assert client.requests[0][1] == "5m"


def _service_config(tmp_path: Path) -> ProspectiveConfig:
    return ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": tmp_path / "prospective.sqlite3",
                "bundle_root": tmp_path / "bundles",
                "feature_parity_report": tmp_path / "parity.json",
            },
            "runtime": {
                "mode": "record_only",
                "source": "ibkr",
                "prospective_start_utc": "2026-07-27T13:30:00Z",
                "instance_id": "parallel-test",
                "app_version": "0.1.0",
                "git_commit": "a" * 40,
                "run_id": "prospective-parallel-test",
            },
            "ibkr": {"port": 4003},
            "context": {
                "mode": "signed_import",
                "hmac_secret_env": "STOCKER_CONTEXT_TEST_SECRET",
            },
            "parallel_validation": {
                "enabled": True,
                "capture_delay_seconds": 7200,
                "requests_per_minute": 60,
            },
        }
    )


class _FakeParallelProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []

    def require_credentials(self) -> None:
        return None

    def fetch_session(
        self,
        *,
        symbol: str,
        session_date: date,
        received_at_utc: datetime,
    ) -> tuple[object, ...]:
        self.calls.append((symbol, session_date))
        return ()


def test_parallel_capture_waits_for_registered_delay_and_never_backfills(
    tmp_path: Path,
) -> None:
    config = _service_config(tmp_path)
    repository = ProspectiveRepository(config.paths.database)
    repository.migrate()
    provider = _FakeParallelProvider()
    symbols = (
        "AAL",
        "AAOI",
        "APLD",
        "ASTS",
        "CIFR",
        "HIMS",
        "IONQ",
        "IREN",
        "MARA",
        "MP",
        "MRNA",
        "MSTR",
        "NVTS",
        "QBTS",
        "RGTI",
        "RIOT",
        "RIVN",
        "SMCI",
        "SOFI",
        "WULF",
    )
    identity = RecorderDeploymentIdentity(
        model_artifact_id="m1-feature-runtime-test-v1",
        universe_id="anchor-frozen-20-v1",
        universe_hash="b" * 64,
        symbols=symbols,
        bundle_verified=True,
    )
    service = ParallelSourceCaptureService(
        config=config,
        repository=repository,
        identity=identity,
        provider=provider,
        sleep=lambda _seconds: None,
    )

    service.poll(now=datetime(2026, 7, 27, 21, 59, tzinfo=UTC))
    assert provider.calls == []

    service.poll(now=datetime(2026, 7, 27, 22, 0, tzinfo=UTC))
    service.poll(now=datetime(2026, 7, 27, 22, 5, tzinfo=UTC))

    assert len(provider.calls) == 20
    assert repository.source_capture_completed(
        run_id="prospective-parallel-test",
        provider="eodhd",
        session_date=date(2026, 7, 27),
    )
    with repository._connect() as connection:
        completion = connection.execute(
            "SELECT status, captured_symbol_count FROM source_capture_completion"
        ).fetchone()
    assert completion is not None
    assert completion["status"] == "partial"
    assert completion["captured_symbol_count"] == 0


def test_parallel_vendor_endpoint_is_fixed_to_approved_https_host(tmp_path: Path) -> None:
    payload = _service_config(tmp_path).model_dump(mode="python")
    payload["parallel_validation"]["base_url"] = "http://example.invalid/api"

    with pytest.raises(ValueError):
        ProspectiveConfig.model_validate(payload)


def test_pending_vendor_request_heartbeats_the_recorder_lease(tmp_path: Path) -> None:
    class SlowProvider(_FakeParallelProvider):
        def fetch_session(
            self,
            *,
            symbol: str,
            session_date: date,
            received_at_utc: datetime,
        ) -> tuple[object, ...]:
            time.sleep(1.05)
            return ()

    config = _service_config(tmp_path)
    config.runtime.heartbeat_seconds = 1
    repository = ProspectiveRepository(config.paths.database)
    identity = RecorderDeploymentIdentity(
        model_artifact_id="m1-feature-runtime-test-v1",
        universe_id="anchor-frozen-20-v1",
        universe_hash="b" * 64,
        symbols=(
            "AAL", "AAOI", "APLD", "ASTS", "CIFR", "HIMS", "IONQ", "IREN", "MARA", "MP",
            "MRNA", "MSTR", "NVTS", "QBTS", "RGTI", "RIOT", "RIVN", "SMCI", "SOFI", "WULF",
        ),
        bundle_verified=True,
    )
    heartbeats: list[bool] = []
    service = ParallelSourceCaptureService(
        config=config,
        repository=repository,
        identity=identity,
        provider=SlowProvider(),
        sleep=lambda _seconds: None,
        heartbeat=lambda: heartbeats.append(True),
    )

    result = service._fetch_with_lease_heartbeat(
        symbol="AAL",
        session_date=date(2026, 7, 27),
        received_at_utc=datetime(2026, 7, 27, 22, tzinfo=UTC),
    )

    assert result == ()
    assert heartbeats


def test_lease_loss_aborts_capture_without_later_evidence_writes(tmp_path: Path) -> None:
    class SlowProvider(_FakeParallelProvider):
        def fetch_session(
            self,
            *,
            symbol: str,
            session_date: date,
            received_at_utc: datetime,
        ) -> tuple[object, ...]:
            time.sleep(1.05)
            return ()

    config = _service_config(tmp_path)
    config.runtime.heartbeat_seconds = 1
    repository = ProspectiveRepository(config.paths.database)
    repository.migrate()
    identity = RecorderDeploymentIdentity(
        model_artifact_id="m1-feature-runtime-test-v1",
        universe_id="anchor-frozen-20-v1",
        universe_hash="b" * 64,
        symbols=(
            "AAL", "AAOI", "APLD", "ASTS", "CIFR", "HIMS", "IONQ", "IREN", "MARA", "MP",
            "MRNA", "MSTR", "NVTS", "QBTS", "RGTI", "RIOT", "RIVN", "SMCI", "SOFI", "WULF",
        ),
        bundle_verified=True,
    )
    heartbeat_count = 0

    def heartbeat() -> None:
        nonlocal heartbeat_count
        heartbeat_count += 1
        if heartbeat_count >= 2:
            raise RecorderLeaseHeld("blocked_recorder_lease_held")

    service = ParallelSourceCaptureService(
        config=config,
        repository=repository,
        identity=identity,
        provider=SlowProvider(),
        sleep=lambda _seconds: None,
        heartbeat=heartbeat,
    )

    with pytest.raises(RecorderLeaseHeld, match="blocked_recorder_lease_held"):
        service.poll(now=datetime(2026, 7, 27, 22, tzinfo=UTC))

    assert heartbeat_count == 2
    assert repository.count("source_bar_observation") == 0
    with repository._connect() as connection:
        completion_count = connection.execute(
            "SELECT COUNT(*) FROM source_capture_completion"
        ).fetchone()[0]
    assert completion_count == 0
