from __future__ import annotations

import pytest

from stocker_core.config import RunsConfig
from stocker_core.markets import CapBucket, MarketId
from stocker_core.runs import CandidateScreen, Environment
from stocker_core.strategies import SESSION_HARD_METHOD, installed_strategies
from stocker_core.universes import UniverseDefinition
from stocker_dashboard.universe_runs import UniverseRunBuilder


def empty_config() -> RunsConfig:
    return RunsConfig(
        universes=(
            UniverseDefinition(
                universe_id="CUSTOM_KEEP",
                name="Backend custom universe",
                members=(
                    {
                        "symbol": "AAPL",
                        "exchange": "SMART",
                        "primary_exchange": "NASDAQ",
                        "currency": "USD",
                    },
                ),
            ),
        ),
        runs=(),
    )


def test_builder_options_are_backend_owned_and_include_installed_method() -> None:
    builder = UniverseRunBuilder()
    options = builder.options(empty_config())

    assert {item["market_id"] for item in options["markets"]} == set(MarketId)
    assert {item["cap_bucket"] for item in options["capitalisation"]} == set(CapBucket)
    assert options["strategies"] == [
        {
            "strategy_id": SESSION_HARD_METHOD.strategy_id,
            "strategy_version": SESSION_HARD_METHOD.strategy_version,
            "label": "Session HARD",
        }
    ]
    assert installed_strategies() == (SESSION_HARD_METHOD,)


def test_any_installed_method_can_be_created_as_paper_on_any_supported_market() -> None:
    builder = UniverseRunBuilder()
    config, run = builder.add(
        empty_config(),
        market_id=MarketId.SOUTH_KOREA_KRX,
        cap_bucket=CapBucket.MID,
        strategy_id=SESSION_HARD_METHOD.strategy_id,
        strategy_version=SESSION_HARD_METHOD.strategy_version,
        environment=Environment.PAPER,
    )

    assert run.display_name == "KRX · HARD · MID"
    assert run.environment is Environment.PAPER
    assert run.market_id == MarketId.SOUTH_KOREA_KRX
    assert run.cap_bucket == CapBucket.MID
    assert run.cap_bucket_version == "CAP_BUCKETS_V1"
    assert run.strategy_id == SESSION_HARD_METHOD.strategy_id
    assert run.strategy_version == SESSION_HARD_METHOD.strategy_version
    assert run.candidate_screen_id == "ACTIVITY_SHORTLIST_V1"
    assert run.candidate_screen_version == "ACTIVITY_SHORTLIST_V1"
    assert run.screen is not None
    assert run.screen.method is CandidateScreen.ACTIVITY_SHORTLIST_V1
    universe = next(item for item in config.universes if item.universe_id == run.universe)
    assert universe.market_spec is not None
    assert universe.market_spec.market_id is MarketId.SOUTH_KOREA_KRX
    assert universe.market_spec.cap_bucket is CapBucket.MID


def test_live_requires_exact_paper_counterpart_and_does_not_modify_it() -> None:
    builder = UniverseRunBuilder()
    with pytest.raises(ValueError, match="matching PAPER run"):
        builder.add(
            empty_config(),
            market_id=MarketId.US_NASDAQ,
            cap_bucket=CapBucket.MID,
            strategy_id=SESSION_HARD_METHOD.strategy_id,
            strategy_version=SESSION_HARD_METHOD.strategy_version,
            environment=Environment.LIVE,
        )

    paper_config, paper = builder.add(
        empty_config(),
        market_id=MarketId.US_NASDAQ,
        cap_bucket=CapBucket.MID,
        strategy_id=SESSION_HARD_METHOD.strategy_id,
        strategy_version=SESSION_HARD_METHOD.strategy_version,
        environment=Environment.PAPER,
    )
    live_config, live = builder.add(
        paper_config,
        market_id=MarketId.US_NASDAQ,
        cap_bucket=CapBucket.MID,
        strategy_id=SESSION_HARD_METHOD.strategy_id,
        strategy_version=SESSION_HARD_METHOD.strategy_version,
        environment=Environment.LIVE,
    )

    assert live.run_id != paper.run_id
    assert next(item for item in live_config.runs if item.run_id == paper.run_id) == paper
    assert live.environment is Environment.LIVE


def test_disable_and_readd_reuses_exact_lineage() -> None:
    builder = UniverseRunBuilder()
    config, created = builder.add(
        empty_config(),
        market_id=MarketId.UK_LSE,
        cap_bucket=CapBucket.LARGE,
        strategy_id=SESSION_HARD_METHOD.strategy_id,
        strategy_version=SESSION_HARD_METHOD.strategy_version,
        environment=Environment.PAPER,
    )
    disabled = builder.disable(config, created.run_id)
    assert len(disabled.runs) == 1
    assert disabled.runs[0].enabled is False

    reenabled, same = builder.add(
        disabled,
        market_id=MarketId.UK_LSE,
        cap_bucket=CapBucket.LARGE,
        strategy_id=SESSION_HARD_METHOD.strategy_id,
        strategy_version=SESSION_HARD_METHOD.strategy_version,
        environment=Environment.PAPER,
    )
    assert same.run_id == created.run_id
    assert same.enabled is True
    assert len(reenabled.runs) == 1
