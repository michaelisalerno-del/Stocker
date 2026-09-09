"""Current method construction; historical cap metadata is tested separately."""

import pytest

from stocker_core.config import RunsConfig
from stocker_core.markets import CapBucket, MarketId, get_market
from stocker_core.methods import SESSION_HARD, validate_run_method
from stocker_core.runs import Environment, RunConfig
from stocker_core.universes import UniverseDefinition
from stocker_dashboard.universe_runs import UniverseRunBuilder


def empty_config():
    return RunsConfig(
        universes=tuple(
            UniverseDefinition(
                universe_id=identity,
                name=identity,
                members=({"symbol": symbol, "exchange": "SMART", "currency": "USD"},),
            )
            for identity, symbol in (("US_ALL", "AAPL"), ("NASDAQ", "AAPL"), ("NYSE", "IBM"))
        )
    )


def add(config=None, market=MarketId.US_NASDAQ, environment=Environment.PAPER):
    return UniverseRunBuilder().add(
        config or empty_config(),
        market_id=market,
        strategy_id=SESSION_HARD.method_id,
        strategy_version=SESSION_HARD.version,
        environment=environment,
    )


def test_builder_options_expose_only_current_supported_markets_and_method():
    options = UniverseRunBuilder().options()
    assert {r["market_id"] for r in options["markets"]} == set(SESSION_HARD.supported_markets)
    assert "capitalisation" not in options
    assert [r["label"] for r in options["strategies"]] == ["Session HARD"]


@pytest.mark.parametrize("market", SESSION_HARD.supported_markets)
def test_method_owns_listing_universe_and_no_cap_filter(market):
    _, run = add(market=market)
    validate_run_method(run)
    assert run.market_id == market
    assert run.cap_bucket is CapBucket.ALL
    assert run.screen is None
    assert bool(run.universe_snapshot.members) == bool(get_market(market).listing_membership)
    assert run.uses_activity_shortlist
    assert run.method_spec["universe_search"]["cap_constraint"] is None
    assert run.candidate_screen_id == "METHOD_REQUIRED_DATA"


def test_other_markets_restore_paper_testing_without_claiming_validation():
    _, run = add(market=MarketId.UK_LSE)
    assert run.method_spec["universe_search"]["validation"] == "UNVALIDATED_CROSS_MARKET_PAPER_TEST"
    assert run.uses_activity_shortlist and run.screen is None
    assert run.session.calendar == "XLON"
    us = add(market=MarketId.US_ALL)[1]
    for component in ("qualification", "vetoes", "direction", "entry", "exits", "artifact_hashes"):
        assert run.method_spec[component] == us.method_spec[component]
    with pytest.raises(ValueError, match="PAPER-only"):
        add(environment=Environment.LIVE)


def test_disable_readd_retains_saved_run_identity_and_snapshot():
    config, run = add()
    disabled = UniverseRunBuilder().disable(config, run.run_id)
    config, resumed = add(disabled)
    assert resumed.run_id == run.run_id and resumed.enabled
    assert len(config.runs) == 1
    assert resumed.universe_snapshot == run.universe_snapshot


def test_explicit_custom_basket_cannot_replace_authoritative_market_listing():
    config = RunsConfig(
        universes=(
            UniverseDefinition(
                universe_id="CUSTOM",
                name="Research input",
                members=({"symbol": "AAPL", "exchange": "SMART", "currency": "USD"},),
            ),
        )
    )
    with pytest.raises(ValueError, match="authoritative listing"):
        add(config)


def test_cap_override_and_altered_spec_are_rejected():
    _, run = add()
    payload = run.model_dump(mode="python")
    payload["cap_bucket"] = CapBucket.MID
    with pytest.raises(ValueError, match="cap/screen override"):
        RunConfig.model_validate(payload)
    payload = run.model_dump(mode="python")
    payload["method_spec"]["exits"]["stop_M"] = 99
    with pytest.raises(ValueError, match="frozen package"):
        RunConfig.model_validate(payload)


def test_legacy_economic_setting_is_retained_but_cannot_override_current_method():
    config = empty_config().model_copy(update={"session_hard_hv_round_trip_cost_bps": 7.5})
    updated, run = add(config)
    assert updated.session_hard_hv_round_trip_cost_bps == 7.5
    assert run.method_spec["economics"]["round_trip_research_cost_bps"] == 10
