import importlib.util
from pathlib import Path

import pytest

from legacy_discovery_support import add
from stocker_core.methods import content_hash, validate_run_method
from stocker_core.runs import RunConfig, RunRiskConfig


def migration():
    path = Path(__file__).parents[1] / "scripts" / "migrate_activity_filter.py"
    spec = importlib.util.spec_from_file_location("migrate_activity_filter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_us_migration_retains_old_members_but_new_discovery_has_no_listing_dependency():
    from stocker_core.discovery import UniverseSource
    from stocker_core.markets import MarketId

    config, _run = add(market=MarketId.US_ALL)
    payload = config.model_dump(mode="json")
    previous = payload["runs"][0]
    previous.update(
        run_id="old-fixed-us",
        strategy_version=migration().PREVIOUS_VERSION,
        candidate_screen_version=migration().PREVIOUS_VERSION,
        universe_source=None, discovery_profile=None,
        universe="US_ALL_METHOD_LISTINGS",
    )
    previous["method_spec"]["method_version"] = migration().PREVIOUS_VERSION
    previous["method_spec"]["universe_search"] = {
        "builder": "IBKR_ACTIVITY_LIQUIDITY_V2", "activity_profile": "ACTIVITY_LIQUIDITY_V2",
    }
    previous["method_spec_hash"] = content_hash(previous["method_spec"])
    previous["universe_snapshot"].update(
        universe_id="US_ALL_METHOD_LISTINGS",
        members=[{"symbol": "SEED", "exchange": "SMART", "currency": "USD"}],
    )
    payload["universes"] = [previous["universe_snapshot"]]
    upgraded, mapping = migration().migrate(payload)
    old, new = upgraded.runs
    assert old.archived and not old.enabled
    assert old.universe_snapshot.members[0].symbol == "SEED"
    assert new.universe_source is UniverseSource.DYNAMIC_IBKR
    assert not new.universe_snapshot.members
    assert mapping == {old.run_id: new.run_id}
    validate_run_method(new)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("catalogue_present", [True, False])
def test_upgrade_preserves_history_risk_environment_and_enabled_state(enabled, catalogue_present):
    config, _run = add()
    payload = config.model_dump(mode="json")
    if not catalogue_present:
        payload["universes"] = [u for u in payload["universes"] if "METHOD" in u["universe_id"]]
    old = payload["runs"][0]
    old.update(
        run_id="previous-run",
        strategy_version=migration().PREVIOUS_VERSION,
        candidate_screen_version=migration().PREVIOUS_VERSION,
        enabled=enabled,
        risk=RunRiskConfig(risk_per_trade=0.002, max_concurrent_positions=2).model_dump(),
    )
    old["method_spec"]["method_version"] = migration().PREVIOUS_VERSION
    old["method_spec_hash"] = content_hash(old["method_spec"])
    upgraded, mapping = migration().migrate(payload)
    previous, current = upgraded.runs
    assert previous.archived and not previous.enabled
    assert previous.method_spec == old["method_spec"]
    assert previous.method_spec_hash == old["method_spec_hash"]
    assert current.risk == previous.risk
    assert current.environment == previous.environment
    assert current.universe_snapshot == previous.universe_snapshot
    assert {u.universe_id for u in upgraded.universes} == {
        u["universe_id"] for u in payload["universes"]
    }
    assert current.enabled == enabled
    assert mapping == {previous.run_id: current.run_id}
    validate_run_method(current)
    with pytest.raises(ValueError, match="historical-only"):
        validate_run_method(previous)
    repeated, mapping = migration().migrate(upgraded.model_dump(mode="json"))
    assert repeated == upgraded and mapping == {}
    tampered = previous.model_dump(mode="json")
    tampered["method_spec"]["entry"]["trigger_M"] = 999
    with pytest.raises(ValueError, match="hash mismatch"):
        RunConfig.model_validate(tampered)
