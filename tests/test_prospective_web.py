from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sqlite3
import statistics
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

import stocker_prospective.read_store as read_store_module
import stocker_prospective.web as web_module
from stocker_prospective.budget_reports import BudgetAwareDailyReportWriter
from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.contract import claims_boundary
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay
from stocker_prospective.web import create_web_app

ROOT = Path(__file__).parents[1]
WEB_UNIT = ROOT / "deploy/systemd/stocker-web.service"
WEB_BOUNDARY_SCRIPT = ROOT / "deploy/scripts/prepare-web-sqlite-boundary.py"
WEB_ENV_TEMPLATE = ROOT / "deploy/stocker-web.env.example"
RECORDER_ENV_TEMPLATE = ROOT / "deploy/stocker.env.example"


def load_web_boundary_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stocker_prepare_web_sqlite_boundary",
        WEB_BOUNDARY_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(
    tmp_path: Path,
    *,
    authenticated: bool = False,
    parallel_enabled: bool = False,
) -> ProspectiveConfig:
    return ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": str(tmp_path / "shared/data/prospective.sqlite3"),
                "bundle_root": str(tmp_path / "shared/bundles"),
                "prospective_report_root": str(tmp_path / "shared/daily-reports"),
                "aggregate_transfer_report": str(
                    tmp_path / "shared/twenty-session-transfer-report.json"
                ),
                "feature_parity_report": str(ROOT / "configs/prospective/feature-parity-m1.json"),
                "quiet_state_concentration_audit_root": str(
                    ROOT
                    / "research/options-feasibility"
                    / "20260727-m1c-quiet-state-concentration-audit-v0"
                    / "artifacts/primary"
                ),
            },
            "runtime": {
                "mode": "shadow",
                "source": "replay",
                "prospective_start_utc": "2026-07-24T13:00:00Z",
                "instance_id": "replay-server-01",
                "app_version": "0.1.0-test",
                "git_commit": "deadbeef",
                "run_id": "replay-run-001",
            },
            "risk": {"trading_enabled": False},
            "web": {
                "host": "127.0.0.1",
                "port": 8765,
                "production": True,
                "authentication_enabled": authenticated,
                "auth_token_env": "STOCKER_WEB_TEST_TOKEN" if authenticated else None,
                "auth_cookie_secure": True,
                "requests_per_minute": 1000,
                "allowed_hosts": ["127.0.0.1", "localhost", "testserver"],
            },
            "ibkr": {},
            "context": {
                "mode": "signed_import",
                "hmac_secret_env": "CONTEXT_SIGNING_SECRET",
            },
            "parallel_validation": {
                "enabled": parallel_enabled,
                "api_token_env": "EODHD_API_TOKEN",
                "credential_status_env": "STOCKER_EODHD_TOKEN_CONFIGURED",
            },
        }
    )


def seeded_app(tmp_path: Path, *, authenticated: bool = False) -> TestClient:
    cfg = config(tmp_path, authenticated=authenticated)
    run_deterministic_replay(
        ReplaySettings(
            database_path=cfg.paths.database,
            run_id="replay-run-001",
            prospective_start_utc=datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
            app_version="0.1.0-test",
            git_commit="deadbeef",
            universe_path=ROOT / "configs/prospective/anchor-frozen-20.json",
            owner_id="test-web-fixture",
            recorder_lease_stale_seconds=60,
        )
    )
    return TestClient(create_web_app(cfg))


def test_read_only_api_and_all_four_screens_smoke(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)

    response = client.get("/")
    assert response.status_code == 200
    assert "LIVE TRADING DISABLED" in response.text
    assert 'id="live-monitor"' in response.text
    assert 'id="signal-detail"' in response.text
    assert 'id="shadow-blotter"' in response.text
    assert 'id="safety-audit"' in response.text
    script = client.get("/assets/app.js")
    assert script.status_code == 200
    polling_script = client.get("/assets/polling.js")
    assert polling_script.status_code == 200
    assert 'fastEndpoints: Object.freeze(["/api/dashboard/summary"])' in polling_script.text
    assert "Official IBKR API" in script.text
    assert "Recorder readiness" in script.text
    assert "REPLAY ${clean(replay.state).toUpperCase()} // LIVE RECORDER" in script.text

    endpoints = (
        "/api/health",
        "/api/runtime",
        "/api/universe",
        "/api/signals",
        "/api/shadow",
        "/api/audit",
        "/api/config/public",
        "/api/market-data-budget",
        "/api/source-transfer",
        "/api/reports/daily",
    )
    for endpoint in endpoints:
        api_response = client.get(endpoint)
        assert api_response.status_code == 200, endpoint
        assert api_response.headers["content-type"].startswith("application/json")
        assert api_response.json()["claims_boundary"] == claims_boundary()
    health = client.get("/api/health").json()
    assert None not in health["blockers"]
    assert health["feature_parity"]["blocker"] == "blocked_feature_source_semantics_mismatch"
    assert health["no_order_path_verified"] is True
    assert health["market_data"]["current_budget"]["rejected_signals"] == 1
    assert health["ibkr_api"]["verified"] is False
    assert health["ibkr_api"]["automatic_installation"] is False
    assert client.get("/api/market-data-budget").json()["banner"] == ("RECORD ONLY — NO ORDERS")

    signal_id = client.get("/api/signals").json()["items"][0]["id"]
    structure_id = client.get("/api/shadow").json()["items"][0]["id"]
    signal_detail = client.get(f"/api/signals/{signal_id}")
    assert signal_detail.status_code == 200
    assert {item["computation_source"] for item in signal_detail.json()["option_computations"]} == {
        "ask",
        "bid",
        "model",
    }
    assert client.get(f"/api/shadow/{structure_id}").status_code == 200


def test_audit_events_skip_raw_partitions_the_web_identity_cannot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)
    protected_path = tmp_path / "recorder-only" / "protected.complete.parquet"
    with sqlite3.connect(cfg.paths.database) as connection:
        connection.execute(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "replay-run-001",
                "ibkr",
                "2026-07-24",
                "AAPL",
                "five_minute_bar_event",
                str(protected_path),
                1,
                "2026-07-24T13:30:00+00:00",
                "2026-07-24T13:35:00+00:00",
                "raw-event-v1",
                "protected-partition",
                1,
                0,
                "test",
                "test",
                "2026-07-24T13:35:01+00:00",
                "{}",
            ),
        )

    original_is_file = Path.is_file

    def inaccessible_partition(path: Path) -> bool:
        if path == protected_path:
            raise PermissionError(13, "Permission denied", str(path))
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", inaccessible_partition)

    response = client.get("/api/audit/events")

    assert response.status_code == 200
    assert any(
        item["audit_type"] == "raw_partition" and item["identity"] == "protected-partition"
        for item in response.json()["items"]
    )


def test_audit_events_are_sqlite_only_bounded_and_cursor_paginated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)
    with sqlite3.connect(cfg.paths.database) as connection:
        for index in range(5):
            connection.execute(
                """
                INSERT INTO raw_partition_manifest_v0(
                    run_id, data_source, session_date, symbol, event_type,
                    file_path, row_count, minimum_timestamp_utc,
                    maximum_timestamp_utc, schema_version, content_hash,
                    complete, gap_count, recorder_version, contract_version,
                    recorded_at_utc, claims_json
                ) VALUES (?, 'synthetic', '2026-07-30', 'AAPL', 'synthetic',
                          ?, 100, ?, ?, 'test', ?, 1, 0, 'test', 'test', ?, '{}')
                """,
                (
                    cfg.runtime.run_id,
                    str(tmp_path / f"must-not-open-{index}.parquet"),
                    f"2026-07-30T14:0{index}:00+00:00",
                    f"2026-07-30T14:0{index}:59+00:00",
                    f"{index + 1:064x}",
                    f"2026-07-30T14:0{index}:59+00:00",
                ),
            )

    monkeypatch.setattr(
        read_store_module,
        "read_parquet_tail",
        lambda *_args, **_kwargs: pytest.fail("audit listing opened Parquet"),
    )
    first = client.get("/api/audit/events?limit=2")
    second = client.get(
        "/api/audit/events",
        params={"limit": 2, "cursor": first.json()["next_cursor"]},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(first.json()["items"]) == 2
    assert len(second.json()["items"]) == 2
    assert first.json()["has_more"] is True
    assert set(item["audit_id"] for item in first.json()["items"]).isdisjoint(
        item["audit_id"] for item in second.json()["items"]
    )
    assert all("must-not-open" not in str(item) for item in first.json()["items"])


def test_audit_projection_records_immutable_subscription_transitions(
    tmp_path: Path,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)
    with sqlite3.connect(cfg.paths.database) as connection:
        envelope_id = int(
            connection.execute(
                "SELECT id FROM evidence_envelope WHERE run_id = ? ORDER BY id LIMIT 1",
                (cfg.runtime.run_id,),
            ).fetchone()[0]
        )
        connection.executemany(
            """
            INSERT INTO subscription_lifecycle_event_v0(
                envelope_id, run_id, occurred_at_utc, subscription_key,
                request_id, subscription_kind, subscription_class, symbol,
                con_id, status, owner_ids_json, owner_count, generation,
                reason, payload_json, claims_json
            ) VALUES (?, ?, ?, 'audit-subscription', 91001, 'level1', 1,
                      'AAPL', 265598, ?, '[]', 0, ?, ?, '{}', '{}')
            """,
            [
                (
                    envelope_id,
                    cfg.runtime.run_id,
                    "2026-07-30T14:00:00+00:00",
                    "started",
                    1,
                    None,
                ),
                (
                    envelope_id,
                    cfg.runtime.run_id,
                    "2026-07-30T14:01:00+00:00",
                    "cancelled",
                    2,
                    "test_cancel",
                ),
            ],
        )

    response = client.get("/api/audit/events?limit=200")

    assert response.status_code == 200
    transitions = [
        json.loads(item["details"])
        for item in response.json()["items"]
        if item["audit_type"] == "subscription_transition"
        and json.loads(item["details"])["subscription_key"] == "audit-subscription"
    ]
    assert [item["status"] for item in transitions] == ["cancelled", "started"]
    assert transitions[0]["reason"] == "test_cancel"


def test_raw_event_detail_requires_explicit_partition_and_is_bounded(
    tmp_path: Path,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)
    partition = tmp_path / "explicit-raw-detail.parquet"
    rows = [
        {
            "event_id": f"event-{index}",
            "received_timestamp_utc": f"2026-07-30T14:00:{index:02d}+00:00",
            "provider_timestamp_utc": f"2026-07-30T14:00:{index:02d}+00:00",
            "source_sequence": index,
            "symbol": "AAPL",
            "bid": 100.0 + index / 100,
            "ask": 100.1 + index / 100,
            "secret_vendor_payload": "must-not-be-projected",
        }
        for index in range(10)
    ]
    pq.write_table(pa.Table.from_pylist(rows), partition, row_group_size=5)
    content_hash = hashlib.sha256(partition.read_bytes()).hexdigest()
    with sqlite3.connect(cfg.paths.database) as connection:
        connection.execute(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (?, 'synthetic', '2026-07-30', 'AAPL',
                      'underlying_level1_quote_event', ?, 10,
                      '2026-07-30T14:00:00+00:00',
                      '2026-07-30T14:00:09+00:00', 'test', ?, 1, 0,
                      'test', 'test', '2026-07-30T14:00:10+00:00', '{}')
            """,
            (cfg.runtime.run_id, str(partition), content_hash),
        )

    response = client.get(f"/api/audit/raw-events/{content_hash}?limit=3")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 3
    assert body["items"][0]["event_id"] == "event-7"
    assert body["items"][-1]["event_id"] == "event-9"
    assert body["read_metrics"]["row_groups_read"] == 1
    assert "file_path" not in body["partition"]
    assert "secret_vendor_payload" not in str(body)


def test_health_does_not_apply_legacy_m1_parity_gate_to_frozen_m1c(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path).model_copy(
        update={
            "paths": config(tmp_path).paths.model_copy(
                update={
                    "frozen_m1c_artifact_root": (
                        ROOT
                        / "research"
                        / "directional-readiness"
                        / "20260726-stock-local-directional-archetypes-v0"
                        / "artifacts"
                        / "primary"
                    )
                }
            )
        }
    )
    ProspectiveRepository(cfg.paths.database).migrate()

    health = TestClient(create_web_app(cfg)).get("/api/health").json()

    assert health["feature_parity"]["blocker"] == "blocked_feature_source_semantics_mismatch"
    assert "blocked_feature_source_semantics_mismatch" not in health["blockers"]


def test_health_reports_live_recorder_waiting_for_prospective_start(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    prospective_start = datetime.now(UTC) + timedelta(hours=1)
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "mode": "record_only",
                    "prospective_start_utc": prospective_start,
                }
            )
        }
    )
    repository = ProspectiveRepository(cfg.paths.database)
    repository.migrate()
    repository.acquire_recorder_lease(
        run_id=cfg.runtime.run_id or "",
        owner_id="server-instance:recorder-process",
        now=datetime.now(UTC),
        stale_after=timedelta(seconds=cfg.runtime.recorder_lease_stale_seconds),
    )

    health = TestClient(create_web_app(cfg)).get("/api/health").json()

    assert health["recorder"]["lease"]["owner_id"] == "server-instance:recorder-process"
    assert health["recorder"]["operational_status"] == "waiting_for_prospective_start"
    recorder = TestClient(create_web_app(cfg)).get("/api/recorder/status").json()
    assert recorder["state"] == "waiting_for_prospective_start"
    assert (
        recorder["operational_state"]["reason"]
        == (health["recorder"]["operational_state"]["reason"])
    )
    with sqlite3.connect(cfg.paths.database) as connection:
        assert connection.execute("SELECT count(*) FROM prospective_run").fetchone() == (0,)


def test_health_and_recorder_status_share_terminal_operational_state(
    tmp_path: Path,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)
    with sqlite3.connect(cfg.paths.database) as connection:
        connection.execute(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (?, 'synthetic', '2026-07-24', 'AAPL', 'synthetic',
                      'synthetic.complete.parquet', 1,
                      '2026-07-24T13:30:00+00:00',
                      '2026-07-24T13:35:00+00:00', 'test',
                      'terminal-session-history', 1, 0, 'test', 'test',
                      '2026-07-24T13:35:01+00:00', '{}')
            """,
            (cfg.runtime.run_id,),
        )

    health = client.get("/api/health").json()
    recorder = client.get("/api/recorder/status").json()

    assert recorder["latest_checkpoint"] is None
    assert recorder["last_event_timestamp"] is not None
    assert recorder["state"] == "inactive"
    assert recorder["operational_state"]["reason"] == "runtime_session_stopped"
    assert (
        recorder["operational_state"]["state"] == (health["recorder"]["operational_state"]["state"])
    )
    assert (
        recorder["operational_state"]["reason"]
        == (health["recorder"]["operational_state"]["reason"])
    )
    assert health["recorder"]["operational_status"] == "inactive"


def test_runtime_projection_ignores_informational_ibkr_event_for_latest_health(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)
    repository = ProspectiveRepository(cfg.paths.database)
    metadata = EvidenceMetadata(
        run_id=cfg.runtime.run_id or "",
        prospective_start_utc=cfg.runtime.prospective_start_utc,
        app_version=cfg.runtime.app_version,
        git_commit=cfg.runtime.git_commit,
        model_artifact_id="synthetic_replay_not_frozen_m1",
        universe_id="anchor-frozen-20-v1",
        cohort="anchor_frozen_20",
        source_timestamps=["2026-07-24T14:00:00Z"],
        recorded_at_utc=datetime(2026, 7, 24, 14, 0, tzinfo=UTC),
    )
    repository.record_ibkr_connection_event(
        metadata,
        state="degraded",
        error_code=354,
        message="blocked_ibkr_market_data_subscription:missing_subscription",
        data_maintained=None,
        reconnect_attempt=None,
        details={"source": "official_ibkr_callback", "event_kind": "state_transition"},
    )
    repository.record_ibkr_connection_event(
        metadata,
        state="degraded",
        error_code=2104,
        message="Market data farm connection is OK",
        data_maintained=None,
        reconnect_attempt=None,
        details={
            "source": "official_ibkr_callback",
            "event_kind": "informational_notification",
        },
    )

    projected = ProspectiveReadStore(
        cfg.paths.database,
        run_id=cfg.runtime.run_id,
    ).runtime_projection()["ibkr_connection"]

    assert projected["error_code"] == 354
    assert projected["message"] == "blocked_ibkr_market_data_subscription:missing_subscription"


def test_no_order_account_threshold_or_upload_endpoint_exists(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)
    paths = client.get("/openapi.json").json()["paths"]

    mutation_paths = {
        path: set(methods) for path, methods in paths.items() if not set(methods) <= {"get"}
    }
    assert mutation_paths == {
        "/api/replay/start": {"post"},
        "/api/replay/stop": {"post"},
    }
    forbidden_segments = {
        "order",
        "orders",
        "account",
        "accounts",
        "position",
        "positions",
        "trade",
        "trades",
        "buy",
        "buys",
        "sell",
        "sells",
        "broker",
        "brokers",
        "execution",
        "executions",
        "upload",
        "uploads",
        "credential",
        "credentials",
    }
    assert not any(
        forbidden_segments.intersection(segment for segment in path.lower().split("/") if segment)
        for path in paths
    )
    assert client.post("/api/recorder/start").status_code == 404
    assert client.post("/api/orders").status_code == 404
    assert client.post("/api/replay/start", json={"mode": "accelerated"}).status_code == 200
    assert client.post("/api/replay/stop").status_code == 200


def test_application_shutdown_cancels_and_joins_active_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    run_deterministic_replay(
        ReplaySettings(
            database_path=cfg.paths.database,
            run_id="replay-run-001",
            prospective_start_utc=datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
            app_version="0.1.0-test",
            git_commit="deadbeef",
            universe_path=ROOT / "configs/prospective/anchor-frozen-20.json",
            owner_id="test-web-fixture",
            recorder_lease_stale_seconds=60,
        )
    )
    started = threading.Event()
    cancelled = threading.Event()

    def controlled_replay(**arguments: object) -> SimpleNamespace:
        stop_event = arguments["stop_event"]
        assert isinstance(stop_event, threading.Event)
        started.set()
        assert stop_event.wait(timeout=1.0)
        cancelled.set()
        return SimpleNamespace(
            records_replayed=0,
            raw_events_replayed=0,
            digest=hashlib.sha256(b"[]").hexdigest(),
            stage_counts={},
            maximum_floating_difference=0.0,
            ibkr_connections_attempted=0,
            broker_state_mutated=False,
        )

    monkeypatch.setattr(web_module, "replay_persisted_evidence", controlled_replay)

    with TestClient(create_web_app(cfg)) as client:
        response = client.post("/api/replay/start", json={"mode": "accelerated"})
        assert response.status_code == 200
        assert started.wait(timeout=1.0)

    assert cancelled.wait(timeout=1.0)


def test_production_error_is_logged_with_safe_request_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seeded = seeded_app(tmp_path)

    def fail_projection(_self: ProspectiveReadStore) -> list[dict[str, object]]:
        raise RuntimeError("synthetic operational failure")

    monkeypatch.setattr(ProspectiveReadStore, "universe_live_v0", fail_projection)
    caplog.set_level(logging.INFO, logger="stocker_prospective.web")
    client = TestClient(seeded.app, raise_server_exceptions=False)

    response = client.get("/api/dashboard/summary")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal_error"}
    request_id = response.headers["X-Request-ID"]
    assert request_id
    exception_records = [
        record
        for record in caplog.records
        if json.loads(record.getMessage()).get("event") == "unexpected_exception"
    ]
    assert len(exception_records) == 1
    record = exception_records[0]
    payload = json.loads(record.getMessage())
    assert payload["request_id"] == request_id
    assert payload["method"] == "GET"
    assert payload["route"] == "/api/dashboard/summary"
    assert payload["response_status"] == 500
    assert payload["run_id"] == "replay-run-001"
    assert payload["exception_class"] == "RuntimeError"
    assert "synthetic operational failure" not in record.getMessage()
    assert record.exc_info is not None


def test_request_log_includes_aggregate_sqlite_and_replay_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = seeded_app(tmp_path)
    caplog.set_level(logging.INFO, logger="stocker_prospective.web")

    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    request_records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if json.loads(record.getMessage()).get("event") == "request_completed"
    ]
    payload = next(
        item
        for item in request_records
        if item["request_id"] == request_id and item["route"] == "/api/dashboard/summary"
    )
    assert payload["response_status"] == 200
    assert payload["elapsed_ms"] >= 0.0
    assert payload["sqlite_operations"] > 0
    assert payload["sqlite_duration_ms"] >= 0.0
    assert payload["parquet_files_examined"] == 0
    assert payload["parquet_row_groups_examined"] == 0
    assert payload["parquet_input_rows"] == 0
    assert payload["parquet_output_rows"] == 0
    assert payload["replay_execution_id"] is None


def test_replay_control_request_log_uses_replay_execution_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = seeded_app(tmp_path)
    caplog.set_level(logging.INFO, logger="stocker_prospective.web")

    response = client.post("/api/replay/start", json={"mode": "accelerated"})

    assert response.status_code == 200
    execution_id = response.json()["execution_id"]
    request_id = response.headers["X-Request-ID"]
    request_records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if json.loads(record.getMessage()).get("event") == "request_completed"
    ]
    payload = next(item for item in request_records if item["request_id"] == request_id)
    assert payload["route"] == "/api/replay/start"
    assert payload["replay_execution_id"] == execution_id
    client.post("/api/replay/stop")


def test_dashboard_summary_latency_is_stable_with_enlarged_operational_history(
    tmp_path: Path,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)

    def median_summary_latency_ms() -> float:
        assert client.get("/api/dashboard/summary").status_code == 200
        timings: list[float] = []
        for _ in range(12):
            started = time.perf_counter()
            response = client.get("/api/dashboard/summary")
            timings.append((time.perf_counter() - started) * 1_000.0)
            assert response.status_code == 200
        return statistics.median(timings)

    baseline_ms = median_summary_latency_ms()
    base = datetime(2026, 7, 30, 13, 0, tzinfo=UTC)
    with sqlite3.connect(cfg.paths.database) as connection:
        connection.executemany(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (
                'replay-run-001', 'synthetic', '2026-07-30', 'AAL',
                'underlying_level1_quote_event', ?, 1, ?, ?, 'test', ?,
                1, 0, 'test', 'test', ?, '{}'
            )
            """,
            [
                (
                    f"/synthetic/not-opened/{index}.parquet",
                    (base + timedelta(seconds=index)).isoformat(),
                    (base + timedelta(seconds=index)).isoformat(),
                    f"summary-history-{index:05d}",
                    (base + timedelta(seconds=index)).isoformat(),
                )
                for index in range(5_000)
            ],
        )
        envelope_id = int(
            connection.execute(
                "SELECT id FROM evidence_envelope WHERE run_id = ? ORDER BY id LIMIT 1",
                (cfg.runtime.run_id,),
            ).fetchone()[0]
        )
        connection.executemany(
            """
            INSERT INTO subscription_lifecycle_v0(
                envelope_id, run_id, subscription_key, request_id,
                subscription_kind, symbol, con_id, priority, owner_episode,
                started_at_utc, cancelled_at_utc, cancellation_reason,
                ibkr_error_codes_json, capacity_denied, claims_json
            ) VALUES (?, ?, ?, ?, 'level1', 'AAPL', 265598, 1, NULL,
                      ?, ?, 'synthetic_history', '[]', 0, '{}')
            """,
            [
                (
                    envelope_id,
                    cfg.runtime.run_id,
                    f"summary-history-subscription-{index:05d}",
                    50_000 + index,
                    (base + timedelta(seconds=index)).isoformat(),
                    (base + timedelta(seconds=index + 1)).isoformat(),
                )
                for index in range(5_000)
            ],
        )
        connection.executemany(
            """
            INSERT INTO data_health_event(
                envelope_id, run_id, severity, blocker_code,
                component, message, details_json
            ) VALUES (?, ?, 'warning', ?, 'synthetic-history', ?, '{}')
            """,
            [
                (
                    envelope_id,
                    cfg.runtime.run_id,
                    (None if index == 4_999 else f"blocked_synthetic_history_{index % 3}"),
                    (
                        "previous_session_options_context_ready"
                        if index == 4_999
                        else "synthetic historical blocker"
                    ),
                )
                for index in range(5_000)
            ],
        )
        connection.executemany(
            """
            INSERT INTO m1c_checkpoint_v0(
                envelope_id, run_id, symbol, session_date, checkpoint,
                bar_start_utc, bar_end_utc, feature_as_of_utc, model_id,
                model_version, model_hash, feature_hash, session_context_hash,
                feature_values_json, probability, threshold, threshold_passed,
                eligible, feature_freshness, missing_feature_count,
                rejection_reasons_json, claims_json
            ) VALUES (?, ?, ?, '2026-07-30', 1, ?, ?, ?, 'M1C', 'perf',
                      'model-hash', ?, 'context-hash', '{}', 0.10, 0.20,
                      0, 0, 'fresh', 0, '[]', '{}')
            """,
            [
                (
                    envelope_id,
                    cfg.runtime.run_id,
                    f"PERF{index:04d}",
                    (base + timedelta(seconds=index)).isoformat(),
                    (base + timedelta(seconds=index + 300)).isoformat(),
                    (base + timedelta(seconds=index + 300)).isoformat(),
                    f"perf-feature-{index:04d}",
                )
                for index in range(1_000)
            ],
        )
        checkpoint_rows = connection.execute(
            """
            SELECT id, symbol, bar_end_utc
            FROM m1c_checkpoint_v0
            WHERE run_id = ? AND symbol LIKE 'PERF%'
            ORDER BY id
            """,
            (cfg.runtime.run_id,),
        ).fetchall()
        connection.executemany(
            """
            INSERT INTO m1c_episode_v0(
                episode_id, envelope_id, checkpoint_id, run_id, symbol,
                session_date, trigger_checkpoint, trigger_bar_end_utc,
                prospective_entry_timestamp_utc, m1c_probability,
                previous_m1c_probability, episode_number,
                minutes_since_previous_episode, scientific_recording_valid,
                rejection_reasons_json, phase, completion_status,
                completed_at_utc, claims_json
            ) VALUES (?, ?, ?, ?, ?, '2026-07-30', 1, ?, ?, 0.10, NULL,
                      1, NULL, 0, '[]', 'active', 'streaming', NULL, '{}')
            """,
            [
                (
                    f"perf-episode-{index:04d}",
                    envelope_id,
                    int(row[0]),
                    cfg.runtime.run_id,
                    str(row[1]),
                    str(row[2]),
                    (datetime.fromisoformat(str(row[2])) + timedelta(minutes=5)).isoformat(),
                )
                for index, row in enumerate(checkpoint_rows)
            ],
        )
        connection.executemany(
            """
            INSERT INTO ibkr_connection_event(
                envelope_id, run_id, state, error_code, message,
                data_maintained, reconnect_attempt, details_json
            ) VALUES (?, ?, 'connected', NULL, 'synthetic notification',
                      1, 0, '{"event_kind":"informational_notification"}')
            """,
            [(envelope_id, cfg.runtime.run_id) for _ in range(1_000)],
        )
    enlarged_ms = median_summary_latency_ms()

    print(
        "dashboard_summary_latency "
        f"baseline_median_ms={baseline_ms:.3f} "
        "manifest_history_rows=5000 subscription_history_rows=5000 "
        "data_health_history_rows=5000 "
        "checkpoint_history_rows=1000 episode_history_rows=1000 "
        "ibkr_event_history_rows=1000 "
        f"enlarged_median_ms={enlarged_ms:.3f}"
    )
    assert enlarged_ms <= baseline_ms * 2.5 + 5.0


def test_frozen_recorder_dashboard_and_read_only_api_surface_are_exposed(
    tmp_path: Path,
) -> None:
    client = seeded_app(tmp_path)
    page = client.get("/")
    assert page.status_code == 200
    assert "RESEARCH ONLY — RECORD ONLY — NO ORDERS" in page.text
    assert "A1 — PROSPECTIVE HYPOTHESIS, NOT VALIDATED" in page.text
    assert 'id="quiet-universe"' in page.text
    assert 'id="quiet-episode"' in page.text
    assert 'id="quiet-shadow"' in page.text
    assert 'id="concentration-audit"' in page.text
    assert "BLOCKED_INSUFFICIENT_LOW_TAIL_SUPPORT" in page.text
    assert "SHORT BID / LONG ASK OPEN" in page.text
    assert "retrospective oracle" not in page.text.lower()
    script = client.get("/assets/app.js").text
    assert "/api/quiet-state/universe" in script
    assert "renderConcentrationAudit" in script

    for path in (
        "/api/dashboard/summary",
        "/api/recorder/status",
        "/api/recorder/capabilities",
        "/api/recorder/session-reports",
        "/api/universe/live",
        "/api/episodes",
        "/api/shadow-outcomes",
        "/api/audit/events",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["claims_boundary"] == claims_boundary()

    for path in ("/api/replay/start", "/api/replay/stop"):
        response = client.post(
            path,
            json={"mode": "accelerated"} if path.endswith("start") else None,
        )
        assert response.status_code == 200
        assert response.json()["claims_boundary"] == claims_boundary()


def test_quiet_state_read_only_api_preserves_frozen_decision(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)

    for path in (
        "/api/quiet-state/status",
        "/api/quiet-state/universe",
        "/api/quiet-state/episodes",
        "/api/quiet-state/shadow-structures",
        "/api/quiet-state/concentration-audit",
        "/api/quiet-state/session-quality",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.json()["claims_boundary"] == claims_boundary()

    status = client.get("/api/quiet-state/status").json()
    assert status["banner"] == "RESEARCH ONLY — RECORD ONLY — NO ORDERS"
    assert status["thresholds"] == {
        "bottom_5": 0.115697407847643,
        "bottom_10": 0.135896965695626,
        "bottom_20": 0.167095528962669,
        "high_tail": 0.488333710794033,
    }
    assert status["order_path"] == "absent"

    audit = client.get("/api/quiet-state/concentration-audit").json()
    assert audit["available"] is True
    assert audit["original_gate_passed"] is False
    assert audit["original_decision"] == "blocked_insufficient_low_tail_support"
    assert audit["month_explanation"]["failed_stress_month"] == "2025-10"
    assert audit["month_explanation"]["exact_failed_share"] == pytest.approx(0.3709677419354839)


def test_daily_chatgpt_report_package_is_listed_and_downloadable(
    tmp_path: Path,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)
    session = datetime(2026, 7, 24, tzinfo=UTC).date()
    package = BudgetAwareDailyReportWriter(
        database_path=cfg.paths.database,
        run_id=cfg.runtime.run_id or "",
        report_root=cfg.paths.prospective_report_root or tmp_path / "reports",
    ).write(
        session=session,
        generated_at=datetime(2026, 7, 24, 22, 0, tzinfo=UTC),
        capacity_manifest={"claims_boundary": claims_boundary()},
        budget_snapshot={"budget_state": "budget_healthy"},
    )

    listing = client.get("/api/reports/daily")
    assert listing.status_code == 200
    assert listing.json()["items"][0]["session"] == session.isoformat()
    download = client.get(f"/api/reports/daily/{session.isoformat()}/{package.archive_path.name}")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert (
        client.get(f"/api/reports/daily/{session.isoformat()}/../prospective.sqlite3").status_code
        == 404
    )
    assert (
        client.get(
            f"/api/reports/daily/{session.isoformat()}/%2e%2e%2fprospective.sqlite3"
        ).status_code
        == 404
    )


def test_daily_report_listing_and_download_reject_symlink_escape(
    tmp_path: Path,
) -> None:
    client = seeded_app(tmp_path)
    cfg = config(tmp_path)
    report_root = cfg.paths.prospective_report_root
    assert report_root is not None
    session = "2026-07-25"
    session_root = report_root / session
    session_root.mkdir(parents=True)
    outside = tmp_path / "outside-sensitive.zip"
    outside.write_bytes(b"must-not-be-served")
    archive_name = "chatgpt-report-package-symlink.zip"
    archive = session_root / archive_name
    archive.symlink_to(outside)
    (session_root / "package-symlink.json").write_text(
        json.dumps({"session": session, "archive": archive_name}),
        encoding="utf-8",
    )

    listing = client.get("/api/reports/daily")
    download = client.get(f"/api/reports/daily/{session}/{archive_name}")

    assert download.status_code == 404
    assert all(item.get("archive") != archive_name for item in listing.json()["items"])


def test_web_sqlite_connections_cannot_write_domain_records(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)
    store = ProspectiveReadStore(cfg.paths.database, run_id=cfg.runtime.run_id)

    with (
        store._connect() as connection,
        pytest.raises(
            sqlite3.OperationalError,
            match="readonly",
        ),
    ):
        connection.execute("DELETE FROM prospective_run")


def test_web_store_anchor_keeps_wal_coordination_files_available(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)
    store = ProspectiveReadStore(cfg.paths.database, run_id=cfg.runtime.run_id)

    store.open_anchor()
    try:
        with sqlite3.connect(cfg.paths.database) as writer:
            assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
            writer.execute("SELECT count(*) FROM prospective_run").fetchone()
        assert Path(f"{cfg.paths.database}-shm").is_file()
        assert Path(f"{cfg.paths.database}-wal").is_file()
    finally:
        store.close_anchor()


def test_web_service_uses_a_distinct_os_identity_and_sqlite_boundary() -> None:
    unit = WEB_UNIT.read_text(encoding="utf-8")
    boundary = WEB_BOUNDARY_SCRIPT.read_text(encoding="utf-8")
    web_environment = WEB_ENV_TEMPLATE.read_text(encoding="utf-8")
    recorder_environment = RECORDER_ENV_TEMPLATE.read_text(encoding="utf-8")

    assert "ProtectSystem=strict" in unit
    assert "User=stocker-web" in unit
    assert "Group=stocker-readers" in unit
    assert "EnvironmentFile=/etc/stocker/stocker-web.env" in unit
    assert "ExecCondition=+/usr/local/libexec/stocker-prepare-web-sqlite-boundary" in unit
    assert "ExecStartPre=" not in unit
    assert "ReadOnlyPaths=/var/lib/stocker" in unit
    assert "ReadWritePaths=/var/lib/stocker/prospective" in unit
    assert "ReadWritePaths=/var/lib/stocker\n" not in unit
    assert "ReadWritePaths=/var/lib/stocker/bundles" not in unit
    assert boundary.startswith("#!/usr/bin/python3\n")
    assert 'DATABASE_DIRECTORY = "/var/lib/stocker/prospective"' in boundary
    assert 'DATABASE_NAME = "prospective.sqlite3"' in boundary
    assert "os.O_NOFOLLOW" in boundary
    assert "os.O_EXCL" in boundary
    assert "dir_fd=" in boundary
    assert "os.fchmod" in boundary
    assert "os.chmod(" not in boundary
    assert "STOCKER_CONTEXT_SIGNING_SECRET" not in web_environment
    assert "STOCKER_CONTEXT_SIGNING_SECRET" in recorder_environment
    assert "EnvironmentFile=/etc/stocker/stocker.env" not in unit


def test_web_boundary_refuses_auxiliary_symlinks(tmp_path: Path) -> None:
    boundary = load_web_boundary_module()
    target = tmp_path / "outside"
    target.touch()
    (tmp_path / "prospective.sqlite3-wal").symlink_to(target)
    directory_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)

    try:
        with pytest.raises(SystemExit) as blocked:
            boundary.open_or_create_auxiliary(
                directory_descriptor,
                "prospective.sqlite3-wal",
                mode=0o640,
                owner_uid=os.getuid(),
                group_gid=os.getgid(),
                allowed_group_gids=frozenset({os.getgid()}),
                label="wal",
            )
    finally:
        os.close(directory_descriptor)

    assert blocked.value.code == 78
    assert target.read_bytes() == b""


def test_web_boundary_maps_filesystem_errors_to_non_restart_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = load_web_boundary_module()

    def raise_os_error(*, migrate_existing: bool) -> None:
        assert migrate_existing is False
        raise OSError("synthetic boundary failure")

    monkeypatch.setattr(boundary, "main", raise_os_error)
    monkeypatch.setattr(boundary.sys, "argv", ["stocker-prepare-web-sqlite-boundary"])
    with pytest.raises(SystemExit) as blocked:
        boundary.run()

    assert blocked.value.code == 78


def test_web_boundary_migration_mode_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = load_web_boundary_module()
    requested: list[bool] = []

    def capture_mode(*, migrate_existing: bool) -> None:
        requested.append(migrate_existing)

    monkeypatch.setattr(boundary, "main", capture_mode)
    monkeypatch.setattr(
        boundary.sys,
        "argv",
        ["stocker-prepare-web-sqlite-boundary", "--migrate-existing"],
    )
    boundary.run()

    assert requested == [True]


def test_web_boundary_migrates_legacy_paths_without_path_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = load_web_boundary_module()
    persistent_root = tmp_path / "persistent"
    database_directory = persistent_root / "prospective"
    bundle_directory = persistent_root / "bundles"
    database_directory.mkdir(parents=True)
    bundle_directory.mkdir()
    installed_bundle = bundle_directory / "installed" / "test-bundle"
    installed_bundle.mkdir(parents=True)
    installed_artifact = installed_bundle / "artifact.joblib"
    installed_artifact.touch()
    active_pointer = bundle_directory / "active.json"
    active_pointer.touch()
    operator_actions = bundle_directory / "operator-actions.jsonl"
    operator_actions.touch()
    database = database_directory / "prospective.sqlite3"
    database.touch()
    fchown_calls: list[tuple[int, int, int]] = []

    monkeypatch.setattr(boundary, "PERSISTENT_ROOT", str(persistent_root))
    monkeypatch.setattr(boundary.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        boundary.pwd,
        "getpwnam",
        lambda name: SimpleNamespace(
            pw_uid=os.getuid() if name == "stocker" else os.getuid() + 1,
            pw_gid=os.getgid(),
        ),
    )
    monkeypatch.setattr(
        boundary.grp,
        "getgrnam",
        lambda _name: SimpleNamespace(gr_gid=os.getgid()),
    )
    monkeypatch.setattr(
        boundary.os,
        "fchown",
        lambda descriptor, uid, gid: fchown_calls.append((descriptor, uid, gid)),
    )

    boundary.main(migrate_existing=True)

    assert fchown_calls
    assert database.stat().st_mode & 0o777 == 0o640
    assert Path(f"{database}-wal").stat().st_mode & 0o777 == 0o640
    assert Path(f"{database}-shm").stat().st_mode & 0o777 == 0o660
    assert bundle_directory.stat().st_mode & 0o7777 == 0o2750
    assert (bundle_directory / "installed").stat().st_mode & 0o7777 == 0o2750
    assert installed_bundle.stat().st_mode & 0o777 == 0o550
    assert installed_artifact.stat().st_mode & 0o777 == 0o440
    assert active_pointer.stat().st_mode & 0o777 == 0o640
    assert operator_actions.stat().st_mode & 0o777 == 0o640


def test_web_boundary_refuses_symlinks_in_installed_bundle_tree(tmp_path: Path) -> None:
    boundary = load_web_boundary_module()
    installed = tmp_path / "installed"
    installed.mkdir()
    target = tmp_path / "outside"
    target.touch()
    (installed / "artifact.joblib").symlink_to(target)
    directory_descriptor = os.open(installed, os.O_RDONLY | os.O_DIRECTORY)

    try:
        with pytest.raises(SystemExit) as blocked:
            boundary.migrate_installed_bundle_tree(
                directory_descriptor,
                owner_uid=os.getuid(),
                allowed_group_gids=frozenset({os.getgid()}),
                reader_gid=os.getgid(),
            )
    finally:
        os.close(directory_descriptor)

    assert blocked.value.code == 78
    assert target.read_bytes() == b""


def test_dashboard_summary_is_compact_and_never_calls_heavy_read_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = seeded_app(tmp_path)

    def forbidden_heavy_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dashboard summary invoked a heavy read projection")

    for method_name in (
        "episode_quote_series_v0",
        "episode_depth_snapshot_v0",
        "raw_event_sample_v0",
        "audit_events_v0",
        "episodes_v0",
        "shadow_outcomes_v0",
        "quiet_state_episodes_v0",
        "quiet_state_shadow_structures_v0",
        "session_reports_v0",
    ):
        monkeypatch.setattr(ProspectiveReadStore, method_name, forbidden_heavy_read)

    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "health",
        "recorder",
        "latest_checkpoints",
        "current_universe",
        "capacity",
        "replay",
        "blockers",
        "claims_boundary",
    }
    assert body["recorder"]["state"] == body["health"]["recorder"]["operational_status"]
    assert body["current_universe"]["items"]
    assert body["claims_boundary"] == claims_boundary()
    assert not {"audit", "episodes", "reports", "parquet_series"}.intersection(body)


def test_two_tabs_fit_within_a_low_eight_request_per_minute_limit(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    cfg = cfg.model_copy(
        update={
            "web": cfg.web.model_copy(update={"requests_per_minute": 8}),
        }
    )
    ProspectiveRepository(cfg.paths.database).migrate()
    app = create_web_app(cfg)
    first_tab = TestClient(app)
    second_tab = TestClient(app)

    statuses = []
    for _ in range(4):
        statuses.append(first_tab.get("/api/dashboard/summary").status_code)
        statuses.append(second_tab.get("/api/dashboard/summary").status_code)

    assert statuses == [200] * 8
    assert first_tab.get("/api/dashboard/summary").status_code == 429


def test_trusted_host_middleware_rejects_unconfigured_host(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    ProspectiveRepository(cfg.paths.database).migrate()
    client = TestClient(create_web_app(cfg))

    rejected = client.get(
        "/api/dashboard/summary",
        headers={"Host": "attacker.invalid"},
    )
    assert rejected.status_code == 400
    assert client.get("/api/dashboard/summary", headers={"Host": "testserver"}).status_code == 200


def test_public_config_is_redacted_and_reports_no_order_path(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    client = seeded_app(tmp_path)
    body = client.get("/api/config/public").json()
    serialized = str(body)

    assert body["safety"] == {"trading_enabled": False, "order_path": "absent"}
    assert str(cfg.paths.database) not in serialized
    assert "CONTEXT_SIGNING_SECRET" not in serialized
    assert "EODHD_API_TOKEN" not in serialized
    assert "auth_token_env" not in serialized
    assert "password" not in serialized.lower()


def test_parallel_vendor_credential_blocker_is_boolean_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    monkeypatch.delenv("STOCKER_EODHD_TOKEN_CONFIGURED", raising=False)
    cfg = config(tmp_path, parallel_enabled=True)
    run_deterministic_replay(
        ReplaySettings(
            database_path=cfg.paths.database,
            run_id="replay-run-001",
            prospective_start_utc=datetime(2026, 7, 24, 13, 0, tzinfo=UTC),
            app_version="0.1.0-test",
            git_commit="deadbeef",
            universe_path=ROOT / "configs/prospective/anchor-frozen-20.json",
            owner_id="test-web-fixture",
            recorder_lease_stale_seconds=60,
        )
    )
    client = TestClient(create_web_app(cfg))

    health = client.get("/api/health").json()
    summary = client.get("/api/dashboard/summary").json()
    public = client.get("/api/config/public").json()

    assert "blocked_missing_eodhd_server_token" in health["blockers"]
    assert summary["health"]["status"] == health["status"] == "blocked"
    assert summary["health"]["blockers"] == health["blockers"]
    assert summary["blockers"] == health["blockers"]
    assert health["parallel_validation"]["credential_configured"] is False
    assert public["parallel_validation"]["credential_configured"] is False
    assert "EODHD_API_TOKEN" not in str(public)

    monkeypatch.setenv("EODHD_API_TOKEN", "must-not-enter-web-process")
    assert client.get("/api/health").json()["parallel_validation"]["credential_configured"] is False

    monkeypatch.setenv("STOCKER_EODHD_TOKEN_CONFIGURED", "1")
    projected = client.get("/api/health").json()
    assert projected["parallel_validation"]["credential_configured"] is True
    assert "must-not-enter-web-process" not in str(projected)


def test_dashboard_summary_and_health_share_all_cached_safety_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path).model_copy(
        update={
            "runtime": config(tmp_path).runtime.model_copy(update={"source": "ibkr"}),
        }
    )
    ProspectiveRepository(cfg.paths.database).migrate()
    monkeypatch.setattr(
        web_module,
        "_active_bundle_projection",
        lambda _config: {
            "bundle_id": None,
            "manifest_sha256": None,
            "verified": False,
            "blockers": ["blocked_test_bundle"],
            "feature_runtime": {
                "installed": False,
                "contract_version": None,
                "scoring_authorized_by_registry": False,
            },
        },
    )
    monkeypatch.setattr(
        web_module,
        "_parity_projection",
        lambda _config: {
            "scoring_allowed": False,
            "blocker": "blocked_test_parity",
            "counts": {},
            "report": None,
        },
    )
    monkeypatch.setattr(
        web_module,
        "official_ibkr_api_projection",
        lambda: {
            "verified": False,
            "automatic_installation": False,
            "blocker": "blocked_test_ibkr_api",
        },
    )
    client = TestClient(create_web_app(cfg))

    health = client.get("/api/health").json()
    summary = client.get("/api/dashboard/summary").json()

    assert health["status"] == summary["health"]["status"] == "blocked"
    assert health["blockers"] == summary["health"]["blockers"] == summary["blockers"]
    assert {
        "blocked_test_bundle",
        "blocked_test_parity",
        "blocked_test_ibkr_api",
    } <= set(summary["blockers"])


def test_health_projection_cache_refreshes_without_recreating_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    cfg = cfg.model_copy(
        update={
            "web": cfg.web.model_copy(update={"operational_projection_cache_seconds": 0.0}),
        }
    )
    ProspectiveRepository(cfg.paths.database).migrate()
    mutable_blockers: list[str] = []

    def bundle_projection(_config: ProspectiveConfig) -> dict[str, object]:
        return {
            "bundle_id": None,
            "manifest_sha256": None,
            "verified": not mutable_blockers,
            "blockers": list(mutable_blockers),
            "feature_runtime": {
                "installed": False,
                "contract_version": None,
                "scoring_authorized_by_registry": False,
            },
        }

    monkeypatch.setattr(web_module, "_active_bundle_projection", bundle_projection)
    client = TestClient(create_web_app(cfg))

    initial_health = client.get("/api/health").json()
    initial_summary = client.get("/api/dashboard/summary").json()
    mutable_blockers.append("blocked_bundle_changed_after_start")
    refreshed_health = client.get("/api/health").json()
    refreshed_summary = client.get("/api/dashboard/summary").json()

    assert "blocked_bundle_changed_after_start" not in initial_health["blockers"]
    assert initial_health["blockers"] == initial_summary["blockers"]
    assert refreshed_health["status"] == refreshed_summary["health"]["status"] == "blocked"
    assert refreshed_health["blockers"] == refreshed_summary["blockers"]
    assert "blocked_bundle_changed_after_start" in refreshed_summary["blockers"]


def test_optional_auth_protects_browser_and_api_with_secure_cookie_support(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    os.environ["STOCKER_WEB_TEST_TOKEN"] = "opaque-test-token"
    try:
        client = seeded_app(tmp_path, authenticated=True)
        assert client.get("/").status_code == 401
        assert client.get("/api/health").status_code == 401
        authorized = client.get(
            "/api/health",
            headers={"Authorization": "Bearer opaque-test-token"},
        )
        assert authorized.status_code == 200
        cookie_authorized = client.get(
            "/api/health",
            cookies={"__Host-stocker_session": "opaque-test-token"},
        )
        assert cookie_authorized.status_code == 200
    finally:
        os.environ.pop("STOCKER_WEB_TEST_TOKEN", None)


def test_production_errors_do_not_leak_stack_traces(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)
    response = client.get("/api/signals/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "not_found"}
    assert "Traceback" not in response.text


def test_web_fails_closed_when_trading_configuration_is_enabled(tmp_path: Path) -> None:
    unsafe = config(tmp_path).model_copy(
        update={"risk": config(tmp_path).risk.model_copy(update={"trading_enabled": True})}
    )

    with pytest.raises(RuntimeError, match="blocked_unsafe_runtime_configuration"):
        create_web_app(unsafe)
