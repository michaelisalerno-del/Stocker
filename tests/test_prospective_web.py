from __future__ import annotations

import importlib.util
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from stocker_prospective.config import ProspectiveConfig
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
            "parallel_validation": {
                "enabled": parallel_enabled,
                "api_token_env": "EODHD_API_TOKEN",
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
    assert "Recorder readiness" in script.text

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
