import asyncio
from dataclasses import replace

import yaml
from fastapi.testclient import TestClient

from stocker_dashboard.app import create_dashboard_app
from stocker_dashboard.controls import RunControlService
from stocker_execution.discovery import CandidateDiscovery, DiscoveryStore
from test_candidate_discovery import MARKET, NOW, Broker, CapBucket, make_run, row
from test_stage10_dashboard import _seed_authoritative_state, _write_control_files


def test_discovery_api_diagnostics_persistence_and_disabled_refresh(tmp_path):
    reads = _seed_authoritative_state(tmp_path)
    config, run = make_run()
    reads.config = config
    reads.clock = lambda: NOW
    status = reads.runtime_status()
    reads.runtime_status = lambda: replace(status, runs=())
    store = DiscoveryStore(reads.stage5_store.path)
    broker = Broker({CapBucket.MICRO: (row(),)})
    discovery = CandidateDiscovery(broker, store, lambda: NOW)
    first = asyncio.run(discovery.discover(run, MARKET, NOW.date(), NOW, NOW))
    runs_path, broker_path = _write_control_files(tmp_path)
    runs_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    controls = RunControlService(runs_path, broker_path)
    client = TestClient(create_dashboard_app(reads, controls))
    url = f"/api/runs/{run.run_id}"
    response = client.get(url)
    assert response.status_code == 200
    diagnostic = response.json()["discovery"]
    assert diagnostic["source"] == "DYNAMIC_IBKR"
    assert diagnostic["raw_candidates"] == diagnostic["unique_candidates"] == 1
    assert diagnostic["watch_pool_size"] == 1
    assert diagnostic["session_hard_qualified"] == 0
    assert diagnostic["last_successful_discovery"] == NOW.isoformat()
    assert client.get("/api/runs").json()[0]["candidate_count"] == 1
    assert client.post(f"{url}/discovery/refresh").status_code == 400
    assert client.post(f"{url}/disable").status_code == 200
    assert client.post(f"{url}/discovery/refresh").json()["apply_mode"] == "ON_ENABLE"
    assert store.generation(run.run_id) == 1
    assert client.get(url).json()["discovery"]["refresh_requested"]
    assert len(broker.scans) == 5
    assert client.post(f"{url}/enable", json={}).status_code == 200
    second = asyncio.run(discovery.discover(run, MARKET, NOW.date(), NOW, NOW))
    assert second["discovery_id"] != first["discovery_id"]
    audit = client.get(f"{url}/discovery?limit=1&offset=1").json()
    assert audit["runs"][0] == first
    assert audit["runs"][0]["candidates"][0]["stages"]["contract"] == "PASSED"
    assert client.get(f"{url}/discovery?limit=0").status_code == 422


def test_old_database_diagnostics_do_not_require_migration(tmp_path):
    reads = _seed_authoritative_state(tmp_path)
    config, run = make_run()
    reads.config = config
    diagnostic = reads.discovery_diagnostic(run.run_id, NOW.date())
    assert diagnostic["status"] == "NOT_STARTED"
    assert diagnostic["watch_pool_size"] == 0
