from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from stocker_prospective.budget_reports import BudgetAwareDailyReportWriter
from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.contract import claims_boundary
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    SchemaVersionTooNew,
)
from stocker_prospective.operational_state import (
    RecorderOperationalRepository,
    RuntimeArtifactVerification,
)
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
    concentration_audit_root: Path | None = None,
) -> ProspectiveConfig:
    audit_root = concentration_audit_root or (tmp_path / "shared/synthetic-concentration-audit")
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
                "quiet_state_concentration_audit_root": str(audit_root),
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


def seeded_app(
    tmp_path: Path,
    *,
    authenticated: bool = False,
    app_config: ProspectiveConfig | None = None,
) -> TestClient:
    cfg = app_config or config(tmp_path, authenticated=authenticated)
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


def record_runtime_artifact_verification(
    cfg: ProspectiveConfig,
    *,
    observed_hash: str | None,
    verified: bool,
    expected_artifact_count: int = 1,
) -> None:
    repository = ProspectiveRepository(cfg.paths.database)
    now = datetime.now(UTC)
    lease = repository.acquire_recorder_lease(
        run_id=cfg.runtime.run_id or "",
        owner_id="test-web-fixture",
        now=now,
        stale_after=timedelta(seconds=60),
    )
    operational = RecorderOperationalRepository(cfg.paths.database)
    operational.start_generation(
        run_id=cfg.runtime.run_id or "",
        recorder_generation=lease.generation,
        owner_id=lease.owner_id,
        started_at=now,
        required_market_data_mode="LIVE",
        expected_artifact_count=expected_artifact_count,
    )
    operational.record_artifact_verification(
        RuntimeArtifactVerification(
            verification_id=f"artifact-verification-{verified}",
            run_id=cfg.runtime.run_id or "",
            recorder_generation=lease.generation,
            artifact_bundle_id="frozen-m1c-test-bundle",
            artifact_name="causal_movement_threshold.json",
            expected_hash="a" * 64,
            observed_hash=observed_hash,
            feature_contract_version="m1c-causal-movement-v1",
            activation_receipt_identity="activation-receipt-test",
            found=observed_hash is not None,
            loaded=observed_hash is not None,
            schema_validated=observed_hash is not None,
            hash_verified=verified,
            contract_compatible=observed_hash is not None,
            used_by_active_generation=verified,
            load_timestamp_utc=now,
            verification_result="verified" if verified else "blocked",
            blocker=None if verified else "RUNTIME_ARTIFACT_HASH_MISMATCH",
        )
    )


def test_read_only_web_fails_closed_on_newer_database_schema(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    repository = ProspectiveRepository(cfg.paths.database)
    repository.migrate()
    with repository._connect() as connection:
        connection.execute(
            """
            INSERT INTO schema_migrations(version, applied_at_utc)
            VALUES ('9999_future_schema.sql', ?)
            """,
            (datetime.now(UTC).isoformat(),),
        )

    with (
        pytest.raises(
            SchemaVersionTooNew,
            match="blocked_schema_newer_than_supported",
        ),
        TestClient(create_web_app(cfg)),
    ):
        pass


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


def test_audit_events_are_bounded_and_cursor_paginated(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)

    first = client.get("/api/audit/events", params={"limit": 2})

    assert first.status_code == 200
    first_page = first.json()
    assert len(first_page["items"]) <= 2
    assert first_page["limit"] == 2
    if first_page["has_more"]:
        assert first_page["next_cursor"]
        second = client.get(
            "/api/audit/events",
            params={"limit": 2, "cursor": first_page["next_cursor"]},
        )
        assert second.status_code == 200
        first_ids = {item["audit_id"] for item in first_page["items"]}
        second_ids = {item["audit_id"] for item in second.json()["items"]}
        assert first_ids.isdisjoint(second_ids)
    assert (
        client.get(
            "/api/audit/events",
            params={"limit": 2, "cursor": "malformed"},
        ).status_code
        == 422
    )


def test_configured_artifact_path_does_not_suppress_runtime_verification_blockers(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "synthetic-frozen-artifacts"
    artifact_root.mkdir()
    cfg = config(tmp_path).model_copy(
        update={
            "paths": config(tmp_path).paths.model_copy(
                update={"frozen_m1c_artifact_root": artifact_root}
            )
        }
    )
    ProspectiveRepository(cfg.paths.database).migrate()

    health = TestClient(create_web_app(cfg)).get("/api/health").json()

    assert health["feature_parity"]["blocker"] == "blocked_feature_source_semantics_mismatch"
    assert "blocked_missing_verified_frozen_bundle" in health["blockers"]
    assert health["runtime_artifact_verification"]["verified"] is False


def test_lease_without_active_recorder_generation_is_inactive(
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

    client = TestClient(create_web_app(cfg))
    health = client.get("/api/health").json()
    status = client.get("/api/recorder/status").json()
    summary = client.get("/api/dashboard/summary").json()

    assert health["recorder"]["lease"]["owner_id"] == "server-instance:recorder-process"
    assert health["recorder"]["operational_status"] == "INACTIVE"
    assert health["recorder"]["operational"]["reason_code"] == "NO_ACTIVE_RECORDER_GENERATION"
    assert status["state"] == health["recorder"]["operational_status"]
    assert summary["recorder"]["state"] == status["state"]
    with sqlite3.connect(cfg.paths.database) as connection:
        assert connection.execute("SELECT count(*) FROM prospective_run").fetchone() == (0,)


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
        "upload",
        "uploads",
        "credential",
        "credentials",
        "broker",
        "brokers",
        "execution",
        "executions",
    }
    assert not any(
        forbidden_segments.intersection(segment for segment in path.lower().split("/") if segment)
        for path in paths
    )
    assert client.post("/api/recorder/start").status_code == 404
    assert client.post("/api/orders").status_code == 404
    assert client.post("/api/replay/start", json={"mode": "accelerated"}).status_code == 200
    assert client.post("/api/replay/stop").status_code == 200


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
    assert 'id="virtual-ledgers"' in page.text
    assert 'id="opening-reversal-virtual-ledger"' in page.text
    assert 'id="quiet-state-capture-ledger"' in page.text
    assert 'id="quiet-state-virtual-ledger"' in page.text
    assert "VIRTUAL EVIDENCE — NOT BROKER POSITIONS" in page.text
    assert "BLOCKED_INSUFFICIENT_LOW_TAIL_SUPPORT" in page.text
    assert "SHORT BID / LONG ASK OPEN" in page.text
    assert "retrospective oracle" not in page.text.lower()
    script = client.get("/assets/app.js").text
    polling = client.get("/assets/polling.mjs").text
    assert 'api("/api/dashboard/summary"' in script
    assert 'fastEndpoints: Object.freeze(["/api/dashboard/summary"])' in polling
    assert "/api/dashboard-snapshot" not in script
    assert "/api/quiet-state/universe" in polling
    assert "DashboardPollCoordinator" in script
    assert "detailRequestPlan" in script
    assert "new AbortController()" in polling
    assert "if (this.active)" in polling
    assert "this.controller.abort()" in polling
    assert "episodeController.abort()" in script
    assert "quietEpisodeController.abort()" in script
    assert "signal: controller.signal" in script
    assert 'document.visibilityState === "visible"' in script
    assert "innerHTML" not in script
    assert "renderConcentrationAudit" in script
    assert "renderVirtualLedgers" in script
    assert '{ label: "Entry bid", value: "entry_bid" }' in script
    assert '{ label: "Exit ask", value: "exit_ask" }' in script

    for path in (
        "/api/recorder/status",
        "/api/recorder/capabilities",
        "/api/recorder/session-reports",
        "/api/universe/live",
        "/api/episodes",
        "/api/shadow-outcomes",
        "/api/virtual-ledgers",
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


def test_web_lifespan_stops_running_replay_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)
    started = threading.Event()
    stopped = threading.Event()

    def cooperative_replay(**kwargs: object) -> SimpleNamespace:
        stop_event = kwargs["stop_event"]
        assert isinstance(stop_event, threading.Event)
        started.set()
        assert stop_event.wait(2)
        stopped.set()
        return SimpleNamespace(
            records_replayed=0,
            raw_events_replayed=0,
            digest="shutdown",
            stage_counts={},
            maximum_floating_difference=0.0,
            ibkr_connections_attempted=0,
            broker_state_mutated=False,
        )

    monkeypatch.setattr(
        "stocker_prospective.web.replay_persisted_evidence",
        cooperative_replay,
    )
    with TestClient(create_web_app(cfg)) as client:
        response = client.post(
            "/api/replay/start",
            json={"mode": "accelerated"},
        )
        assert response.status_code == 200
        assert started.wait(1)

    assert stopped.wait(1)


def test_dashboard_snapshot_is_one_consistent_projection(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)

    response = client.get("/api/dashboard-snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert payload["partial"] is False
    assert payload["section_errors"] == {}
    assert {
        "health",
        "status",
        "capabilities",
        "universe",
        "episodes",
        "shadow",
        "virtual_ledgers",
        "audit",
        "session_reports",
        "quiet_status",
        "quiet_universe",
        "quiet_episodes",
        "quiet_shadow",
        "quiet_session_quality",
        "concentration_audit",
        "budget",
        "transfer",
        "report_packages",
    } == set(payload["sections"])
    assert payload["sections"]["health"]["recorder"]["run_id"] == "replay-run-001"
    assert payload["sections"]["status"]["run_id"] == "replay-run-001"
    ledgers = payload["sections"]["virtual_ledgers"]
    assert ledgers["opening_reversal"]["item_limit"] == 25
    assert ledgers["quiet_state"]["item_limit"] == 50
    assert ledgers["quiet_state"]["capture_item_limit"] == 25
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-correlation-id"]


def test_dashboard_summary_is_compact_consistent_and_never_reads_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = seeded_app(tmp_path)

    def forbidden_parquet_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fast summary attempted a Parquet read")

    monkeypatch.setattr(
        "stocker_prospective.read_store.read_parquet_window",
        forbidden_parquet_read,
    )
    monkeypatch.setattr(
        "stocker_prospective.read_store.read_parquet_tail",
        forbidden_parquet_read,
    )

    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    payload = response.json()
    assert {
        "health",
        "recorder",
        "latest_checkpoints",
        "current_universe",
        "capacity",
        "current_budget",
        "replay",
        "current_blockers",
    } <= payload.keys()
    assert payload["recorder"]["state"] == payload["health"]["recorder"]["operational_status"]
    assert payload["recorder"]["gap_details_included"] is False
    assert "audit" not in payload
    assert "report_packages" not in payload
    assert "shadow" not in payload
    assert response.headers["x-request-id"]


def test_virtual_ledgers_are_separate_read_only_projections(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)

    response = client.get("/api/virtual-ledgers")

    assert response.status_code == 200
    payload = response.json()
    assert payload["opening_reversal"]["ledger_scope"] == "opening_reversal_v1_1"
    assert payload["opening_reversal"]["items"] == []
    assert (
        payload["opening_reversal"]["entry_convention"] == "first_valid_live_ask_at_or_after_entry"
    )
    assert (
        payload["opening_reversal"]["exit_convention"]
        == "first_valid_live_bid_at_or_after_frozen_15m_horizon"
    )
    assert payload["quiet_state"]["ledger_scope"] == "quiet_state_short_premium"
    assert payload["quiet_state"]["capture_ledger_scope"] == "quiet_state_short_premium_capture"
    assert payload["quiet_state"]["capture_items"] == []
    assert payload["quiet_state"]["items"] == []
    assert (
        payload["quiet_state"]["fill_convention"]
        == "open_short_bid_long_ask_close_short_ask_long_bid"
    )
    assert payload["ledgers_combined_for_analysis"] is False
    assert payload["execution_claimed"] is False
    assert payload["broker_positions_claimed"] is False
    assert payload["claims_boundary"] == claims_boundary()

    bounded = client.get(
        "/api/virtual-ledgers",
        params={
            "opening_limit": 1,
            "quiet_limit": 1,
            "quiet_capture_limit": 1,
        },
    )
    assert bounded.status_code == 200
    assert bounded.json()["quiet_state"]["capture_item_limit"] == 1
    assert (
        client.get(
            "/api/virtual-ledgers",
            params={"quiet_capture_limit": 1001},
        ).status_code
        == 422
    )


def test_optional_snapshot_section_failure_preserves_other_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)

    def fail_optional_section(_self: ProspectiveReadStore) -> list[dict[str, object]]:
        raise RuntimeError("synthetic optional section failure")

    monkeypatch.setattr(
        ProspectiveReadStore,
        "quiet_state_session_quality_v0",
        fail_optional_section,
    )
    client = TestClient(create_web_app(cfg))

    with caplog.at_level("ERROR"):
        response = client.get(
            "/api/dashboard-snapshot",
            headers={"x-request-id": "snapshot-section-test"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["partial"] is True
    assert "health" in payload["sections"]
    assert "status" in payload["sections"]
    assert "quiet_session_quality" not in payload["sections"]
    assert payload["section_errors"]["quiet_session_quality"] == {
        "error_code": "DASHBOARD_SECTION_QUIET_SESSION_QUALITY_UNAVAILABLE",
    }
    logged = next(
        json.loads(record.message)
        for record in caplog.records
        if '"event":"dashboard_section_error"' in record.message
    )
    assert logged["request_id"] == "snapshot-section-test"
    assert logged["route"] == "/api/dashboard-snapshot"
    assert logged["section"] == "quiet_session_quality"
    assert logged["exception_class"] == "RuntimeError"
    assert "sqlite_duration_ms" in logged


@pytest.mark.parametrize("observed_hash", [None, "b" * 64])
def test_missing_or_wrong_hash_runtime_artifact_remains_blocked(
    tmp_path: Path,
    observed_hash: str | None,
) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)
    record_runtime_artifact_verification(
        cfg,
        observed_hash=observed_hash,
        verified=False,
    )

    payload = TestClient(create_web_app(cfg)).get("/api/recorder/status").json()

    verification = payload["runtime_artifact_verification"]
    assert verification["verified"] is False
    assert verification["blockers"] == ["RUNTIME_ARTIFACT_HASH_MISMATCH"]
    assert verification["items"][0]["expected_hash"] == "a" * 64
    assert verification["items"][0]["observed_hash"] == observed_hash
    assert verification["items"][0]["used_by_active_generation"] == 0


def test_persisted_runtime_artifact_evidence_is_displayed_as_verified(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)
    record_runtime_artifact_verification(
        cfg,
        observed_hash="a" * 64,
        verified=True,
    )

    payload = TestClient(create_web_app(cfg)).get("/api/recorder/status").json()

    verification = payload["runtime_artifact_verification"]
    assert verification["verified"] is True
    assert verification["blockers"] == []
    assert verification["items"][0]["verification_result"] == "verified"
    assert verification["items"][0]["used_by_active_generation"] == 1


def test_partial_runtime_artifact_set_cannot_claim_verified(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)
    record_runtime_artifact_verification(
        cfg,
        observed_hash="a" * 64,
        verified=True,
        expected_artifact_count=2,
    )

    verification = (
        TestClient(create_web_app(cfg))
        .get("/api/recorder/status")
        .json()["runtime_artifact_verification"]
    )

    assert verification["expected_artifact_count"] == 2
    assert len(verification["items"]) == 1
    assert verification["verified"] is False
    assert verification["blockers"] == ["RUNTIME_ARTIFACT_EVIDENCE_INCOMPLETE"]


def test_no_order_verdict_is_derived_from_named_checks(tmp_path: Path) -> None:
    health = seeded_app(tmp_path).get("/api/health").json()
    checks = health["no_order_checks"]

    named_checks = {
        "risk_trading_enabled_false",
        "web_has_no_broker_reference",
        "web_database_opened_read_only",
        "http_order_routes_absent",
        "adapter_order_methods_absent",
        "ibkr_read_only_configured",
        "ibkr_socket_loopback_only",
        "runtime_order_surface_absent",
        "broker_state_mutation_count_zero",
    }
    assert all(checks[name] is True for name in named_checks)
    assert checks["aggregate_no_order_verdict"] is True
    assert (
        checks["ibkr_read_only_evidence"]["external_ibkr_environment_verification"]
        == "not_externally_verifiable"
    )


def test_production_errors_have_correlation_ids_and_generic_browser_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = config(tmp_path)
    seeded_app(tmp_path)

    def fail_universe(_self: ProspectiveReadStore) -> dict[str, list[dict[str, object]]]:
        raise RuntimeError("synthetic secret-free server failure")

    monkeypatch.setattr(ProspectiveReadStore, "universe", fail_universe)
    client = TestClient(create_web_app(cfg), raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        response = client.get(
            "/api/universe",
            headers={"x-correlation-id": "test-correlation-123"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal_error"}
    assert response.headers["x-correlation-id"] == "test-correlation-123"
    assert response.headers["x-request-id"] == "test-correlation-123"
    unexpected = next(
        json.loads(record.message)
        for record in caplog.records
        if '"event":"unexpected_exception"' in record.message
    )
    assert unexpected["error_code"] == "WEB_INTERNAL_ERROR"
    assert unexpected["route"] == "/api/universe"
    assert unexpected["exception_class"] == "RuntimeError"
    assert unexpected["request_id"] == "test-correlation-123"
    assert "sqlite_duration_ms" in unexpected
    assert "parquet_files_examined" in unexpected


def test_quiet_state_read_only_api_preserves_frozen_decision(tmp_path: Path) -> None:
    audit_root = tmp_path / "synthetic-concentration-audit"
    audit_root.mkdir()
    artifacts = {
        "decision.json": {
            "decision": "blocked_insufficient_low_tail_support",
            "gate_passed": False,
        },
        "stress_month_concentration_explanation.json": {
            "failed_stress_month": "2025-10",
            "exact_failed_share": 0.3709677419354839,
        },
        "surprise_concentration_explanation.json": {"synthetic_fixture": True},
        "small_count_feasibility.json": {"synthetic_fixture": True},
    }
    for filename, payload in artifacts.items():
        (audit_root / filename).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    cfg = config(tmp_path, concentration_audit_root=audit_root)
    client = seeded_app(tmp_path, app_config=cfg)

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
    report_root = cfg.paths.prospective_report_root
    assert report_root is not None
    outside_archive = tmp_path / "outside-report.zip"
    outside_archive.write_bytes(b"not a report package")
    symlink_name = "chatgpt-report-package-symlink.zip"
    (report_root / session.isoformat() / symlink_name).symlink_to(outside_archive)
    assert client.get(f"/api/reports/daily/{session.isoformat()}/{symlink_name}").status_code == 404


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
    public = client.get("/api/config/public").json()

    assert "blocked_missing_eodhd_server_token" in health["blockers"]
    assert health["parallel_validation"]["credential_configured"] is False
    assert public["parallel_validation"]["credential_configured"] is False
    assert "EODHD_API_TOKEN" not in str(public)

    monkeypatch.setenv("EODHD_API_TOKEN", "must-not-enter-web-process")
    assert client.get("/api/health").json()["parallel_validation"]["credential_configured"] is False

    monkeypatch.setenv("STOCKER_EODHD_TOKEN_CONFIGURED", "1")
    projected = client.get("/api/health").json()
    assert projected["parallel_validation"]["credential_configured"] is True
    assert "must-not-enter-web-process" not in str(projected)


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


def test_trusted_host_middleware_rejects_unconfigured_hosts(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)

    rejected = client.get(
        "/api/config/public",
        headers={"host": "untrusted.example"},
    )
    allowed = client.get(
        "/api/config/public",
        headers={"host": "testserver"},
    )

    assert rejected.status_code == 400
    assert allowed.status_code == 200


def test_rate_limit_is_enforced_after_configured_request_budget(
    tmp_path: Path,
) -> None:
    seeded_app(tmp_path)
    baseline = config(tmp_path)
    cfg = baseline.model_copy(
        update={"web": baseline.web.model_copy(update={"requests_per_minute": 3})}
    )
    client = TestClient(create_web_app(cfg))

    responses = [client.get("/api/config/public") for _ in range(4)]

    assert [response.status_code for response in responses] == [200, 200, 200, 429]
    assert responses[-1].json()["detail"] == "rate_limit_exceeded"
    assert responses[-1].headers["x-request-id"]


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
