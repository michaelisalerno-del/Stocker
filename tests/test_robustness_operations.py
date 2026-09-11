import base64
import importlib.util

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stocker_dashboard.security import DashboardSecurity


def app():
    result = FastAPI()
    result.add_middleware(DashboardSecurity)

    @result.api_route("/control", methods=["GET", "POST"])
    def control():
        return {"ok": True}

    return result


def test_local_dashboard_rejects_remote_and_cross_site(monkeypatch):
    monkeypatch.delenv("STOCKER_DASHBOARD_PASSWORD", raising=False)
    with TestClient(app(), base_url="http://127.0.0.1", client=("127.0.0.1", 123)) as client:
        assert client.post("/control").status_code == 200
        assert client.post("/control", headers={"Origin": "https://evil.test"}).status_code == 403
        assert client.get("/control", headers={"Host": "evil.test"}).status_code == 403
    with TestClient(app(), base_url="http://127.0.0.1", client=("192.0.2.1", 123)) as client:
        assert client.post("/control", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 403


def test_protected_backend_requires_credentials_on_every_route(monkeypatch):
    password = "isolated-test-password-123456789"
    monkeypatch.setenv("STOCKER_DASHBOARD_PASSWORD", password)
    monkeypatch.setenv("STOCKER_DASHBOARD_ORIGIN", "https://stocker.example")
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"stocker:{password}".encode()).decode()
    }
    with TestClient(app(), base_url="https://stocker.example") as client:
        assert client.get("/control").status_code == 401
        assert (
            client.post("/control", headers={"X-Authenticated-User": "stocker"}).status_code == 401
        )
        assert client.post("/control", headers=headers).status_code == 200
        assert (
            client.post("/control", headers={**headers, "Origin": "https://evil.test"}).status_code
            == 403
        )
        assert (
            client.post(
                "/control", headers={**headers, "Origin": "https://stocker.example"}
            ).status_code
            == 200
        )


def test_authenticated_proxy_requires_loopback_and_private_credential(monkeypatch):
    monkeypatch.delenv("STOCKER_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setenv("STOCKER_DASHBOARD_PROXY_TOKEN", "isolated-proxy-credential-123456789")
    monkeypatch.setenv("STOCKER_DASHBOARD_ORIGIN", "https://stocker.example")
    headers = {"X-Stocker-Proxy-Token": "isolated-proxy-credential-123456789"}
    with TestClient(app(), base_url="https://stocker.example", client=("127.0.0.1", 123)) as client:
        assert client.get("/control").status_code == 403
        assert (
            client.post("/control", headers={"X-Authenticated-User": "stocker"}).status_code == 403
        )
        assert client.post("/control", headers=headers).status_code == 200
        assert (
            client.post("/control", headers={**headers, "Origin": "https://evil.test"}).status_code
            == 403
        )
    with TestClient(app(), base_url="https://stocker.example", client=("192.0.2.1", 123)) as client:
        assert (
            client.post("/control", headers={**headers, "X-Forwarded-For": "127.0.0.1"}).status_code
            == 403
        )


def test_consistent_snapshot_restore_retains_unresolved_identity(tmp_path):
    import sqlite3
    from dataclasses import replace

    import yaml

    from stocker_core.config import load_runs_config
    from stocker_core.methods import ARTIFACTS
    from stocker_core.runs import Environment
    from stocker_execution.execution_ledger import ExecutionLedger
    from test_stage7_ledger import _plan
    from test_stage8_runtime import _run, _runs

    spec = importlib.util.spec_from_file_location("backup_state", "scripts/backup_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    db = tmp_path / "source.sqlite"
    ledger = ExecutionLedger(db)
    run = _run("restore-run")
    plan = replace(_plan(), run_id=run.run_id)
    ledger.reserve(plan, expected_account="DU123456")
    ledger.mark_submitting(plan.order_plan_id)
    runs, broker = tmp_path / "runs.yaml", tmp_path / "ibkr.yaml"
    runs.write_text(yaml.safe_dump(_runs(run).model_dump(mode="json")))
    broker.write_text("PAPER:\n  environment: PAPER\n  expected_account: DU123456\n")
    bundle = tmp_path / "bundle"
    # Keep a WAL writer open: copying only the main database file would miss
    # committed pages, while an uncommitted row must never enter the snapshot.
    with sqlite3.connect(db) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE restore_probe (value TEXT)")
        writer.execute("INSERT INTO restore_probe VALUES ('committed')")
        writer.commit()
        writer.execute("INSERT INTO restore_probe VALUES ('uncommitted')")
        module.backup(db, runs, broker, ARTIFACTS, bundle)
        writer.rollback()
    module.verify(bundle)
    restored = ExecutionLedger(bundle / "state.sqlite")
    with sqlite3.connect(bundle / "state.sqlite") as connection:
        assert connection.execute("SELECT value FROM restore_probe").fetchall() == [("committed",)]
    saved = load_runs_config(bundle / "configuration" / "runs.yaml")
    assert [item.run_id for item in saved.runs] == [plan.run_id]
    assert restored.has_signal(plan.signal_id)
    assert len(restored.active_records(Environment.PAPER, "DU123456")) == 1
    assert not restored.reserve(plan, expected_account="DU123456")
    assert (bundle / "configuration" / "runs.yaml").read_bytes() == runs.read_bytes()
    (bundle / "configuration" / "runs.yaml").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        module.verify(bundle)
