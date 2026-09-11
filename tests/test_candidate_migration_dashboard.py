import asyncio
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from legacy_discovery_support import add
from stocker_core.methods import SESSION_HARD
from stocker_dashboard.app import create_dashboard_app
from stocker_execution.candidate_pipeline import CandidatePipeline, CandidateStore
from test_candidate_pipeline import Provider, Source, drain, setup_run
from test_stage10_dashboard import _seed_authoritative_state


def migration():
    path = Path(__file__).parents[1] / "scripts" / "migrate_candidate_selection.py"
    spec = importlib.util.spec_from_file_location("candidate_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_retains_immutable_legacy_spec_and_risk_creates_disabled_paper_run():
    config, previous = add()
    payload = config.model_dump(mode="json")
    upgraded, mapping = migration().migrate(payload)
    old, new = upgraded.runs
    assert old.archived and not old.enabled
    assert old.method_spec == previous.method_spec
    assert old.method_spec_hash == previous.method_spec_hash
    assert old.universe_snapshot == previous.universe_snapshot
    assert new.risk == previous.risk
    assert not new.enabled and not new.archived
    assert new.strategy_version == SESSION_HARD.version
    assert new.uses_candidate_selection and not new.uses_activity_shortlist
    assert mapping == {old.run_id: new.run_id}
    assert payload == config.model_dump(mode="json")
    repeated, mapping = migration().migrate(upgraded.model_dump(mode="json"))
    assert repeated == upgraded and not mapping
    payload["runs"][0]["environment"] = "LIVE"
    with pytest.raises(ValueError, match="PAPER-only"):
        migration().migrate(payload)


@pytest.mark.parametrize("archived", [False, True])
def test_migration_refuses_to_change_an_existing_current_run(archived):
    config, _ = add()
    from stocker_core.markets import MarketId

    new_config, _, _ = setup_run(MarketId.US_NASDAQ)
    payload = config.model_dump(mode="json")
    current = new_config.model_dump(mode="json")["runs"][0]
    current.update(archived=archived, enabled=False)
    payload["runs"].append(current)
    # Match the markets while retaining both independent immutable snapshots.
    payload["runs"][0]["market_id"] = payload["runs"][1]["market_id"]
    with pytest.raises(ValueError):
        migration().migrate(payload)


def test_dashboard_candidate_summary_is_small_and_details_are_paginated(tmp_path):
    reads = _seed_authoritative_state(tmp_path)
    config, instance, session = setup_run()
    reads.config = config
    reads.clock = lambda: session.opens_at
    status = reads.runtime_status()
    reads.runtime_status = lambda: replace(status, runs=())
    store = CandidateStore(reads.stage5_store.path)
    pipeline = CandidatePipeline(store, Provider(), Source(), lambda: session.opens_at)
    asyncio.run(drain(pipeline, instance, session, [session.opens_at]))
    client = TestClient(create_dashboard_app(reads, None))
    url = f"/api/runs/{instance.config.run_id}"
    response = client.get("/api/runs")
    assert response.status_code == 200
    summary = response.json()[0]["candidate_selection"]
    assert summary["broad_eligible"] == 533
    assert "identities" not in summary and "rows" not in summary
    assert len(response.content) < 20000
    detail = client.get(url).json()["candidate_selection"]
    assert detail["session_hard_qualified"] == 0
    audit = client.get(f"{url}/candidate-selection?session={session.session}&limit=2&offset=1")
    assert audit.status_code == 200
    assert len(audit.json()["rows"]) == 2
    assert audit.json()["rows"][0]["identity"]["con_id"] == 2
    assert (
        client.get(f"{url}/candidate-selection?session={session.session}&limit=0").status_code
        == 422
    )


def test_original_v7_specification_hashes_are_unchanged_for_every_market():
    import json

    from stocker_core.markets import MarketId
    from stocker_core.methods import LEGACY_SESSION_HARD, content_hash

    path = Path(__file__).parent / "fixtures/session_hard_candidates/legacy_spec_hashes.json"
    for market, digest in json.loads(path.read_text()).items():
        assert content_hash(LEGACY_SESSION_HARD.specification(MarketId(market))) == digest


def test_original_v8_hashes_and_frozen_trading_spec_remain_unchanged():
    import json

    from stocker_core.markets import MarketId
    from stocker_core.methods import SESSION_HARD_CANDIDATES_V8, content_hash

    path = Path(__file__).parent / "fixtures/session_hard_candidates/v8_spec_hashes.json"
    for market, digest in json.loads(path.read_text()).items():
        previous = SESSION_HARD_CANDIDATES_V8.specification(MarketId(market))
        assert content_hash(previous) == digest
        current = SESSION_HARD.specification(MarketId(market))
        assert current["candidate_selection"] == previous["candidate_selection"]
        for field in set(previous) - {"method_version", "universe_search"}:
            assert current[field] == previous[field]
        assert content_hash(current) != digest
