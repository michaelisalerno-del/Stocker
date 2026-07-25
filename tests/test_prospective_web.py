from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay
from stocker_prospective.web import create_web_app

ROOT = Path(__file__).parents[1]
WEB_UNIT = ROOT / "deploy/systemd/stocker-web.service"


def config(tmp_path: Path, *, authenticated: bool = False) -> ProspectiveConfig:
    return ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": str(tmp_path / "shared/data/prospective.sqlite3"),
                "bundle_root": str(tmp_path / "shared/bundles"),
                "feature_parity_report": str(ROOT / "configs/prospective/feature-parity-m1.json"),
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
    assert "Official IBKR API" in script.text

    endpoints = (
        "/api/health",
        "/api/runtime",
        "/api/universe",
        "/api/signals",
        "/api/shadow",
        "/api/audit",
        "/api/config/public",
    )
    for endpoint in endpoints:
        api_response = client.get(endpoint)
        assert api_response.status_code == 200, endpoint
        assert api_response.headers["content-type"].startswith("application/json")
    health = client.get("/api/health").json()
    assert None not in health["blockers"]
    assert health["no_order_path_verified"] is True
    assert health["market_data"]["current_budget"]["rejected_signals"] == 1
    assert health["ibkr_api"]["verified"] is False
    assert health["ibkr_api"]["automatic_installation"] is False

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


def test_no_order_account_threshold_or_upload_endpoint_exists(tmp_path: Path) -> None:
    client = seeded_app(tmp_path)
    paths = client.get("/openapi.json").json()["paths"]
    lowered = " ".join(paths).lower()

    assert all(set(methods) <= {"get"} for methods in paths.values())
    for forbidden in ("order", "account", "threshold", "upload", "credential", "trade"):
        assert forbidden not in lowered
    assert client.post("/api/recorder/start").status_code == 404
    assert client.post("/api/orders").status_code == 404


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


def test_web_service_writes_only_sqlite_wal_coordination_files() -> None:
    unit = WEB_UNIT.read_text(encoding="utf-8")

    assert "ProtectSystem=strict" in unit
    assert "ReadWritePaths=/var/lib/stocker/prospective" in unit
    assert "ReadWritePaths=/var/lib/stocker\n" not in unit
    assert "ReadWritePaths=/var/lib/stocker/bundles" not in unit


def test_public_config_is_redacted_and_reports_no_order_path(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    client = seeded_app(tmp_path)
    body = client.get("/api/config/public").json()
    serialized = str(body)

    assert body["safety"] == {"trading_enabled": False, "order_path": "absent"}
    assert str(cfg.paths.database) not in serialized
    assert "CONTEXT_SIGNING_SECRET" not in serialized
    assert "auth_token_env" not in serialized
    assert "password" not in serialized.lower()


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
