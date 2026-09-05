from __future__ import annotations

import pytest

from stocker_core.config import RunsConfig
from stocker_core.markets import CapBucket, MarketId
from stocker_core.runs import CandidateScreen, Environment, RunConfig, RunScreenConfig
from stocker_core.strategies import (
    SESSION_HARD_HV_METHOD,
    installed_strategies,
)
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
            UniverseDefinition(
                universe_id="NASDAQ",
                name="NASDAQ listing membership",
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
    options = builder.options()

    assert {item["market_id"] for item in options["markets"]} == set(MarketId)
    assert {item["cap_bucket"] for item in options["capitalisation"]} == set(CapBucket)
    assert options["strategies"] == [
        {
            "strategy_id": SESSION_HARD_HV_METHOD.strategy_id,
            "strategy_version": SESSION_HARD_HV_METHOD.strategy_version,
            "label": "Session HARD · HV",
            "environments": ["PAPER"],
        },
    ]
    assert installed_strategies() == (SESSION_HARD_HV_METHOD,)


def test_any_installed_method_can_be_created_as_paper_on_any_supported_market() -> None:
    builder = UniverseRunBuilder()
    config, run = builder.add(
        empty_config(),
        market_id=MarketId.SOUTH_KOREA_KRX,
        cap_bucket=CapBucket.MID,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        environment=Environment.PAPER,
    )

    assert run.display_name == "KRX · HARD-HV · MID"
    assert run.environment is Environment.PAPER
    assert run.market_id == MarketId.SOUTH_KOREA_KRX
    assert run.cap_bucket == CapBucket.MID
    assert run.cap_bucket_version == "CAP_BUCKETS_V1"
    assert run.strategy_id == SESSION_HARD_HV_METHOD.strategy_id
    assert run.strategy_version == SESSION_HARD_HV_METHOD.strategy_version
    assert run.candidate_screen_id == "ACTIVITY_SHORTLIST_V1"
    assert run.candidate_screen_version == "ACTIVITY_SHORTLIST_V1"
    assert run.screen is not None
    assert run.screen.method is CandidateScreen.ACTIVITY_SHORTLIST_V1
    universe = next(item for item in config.universes if item.universe_id == run.universe)
    assert universe.market_spec is not None
    assert universe.market_spec.market_id is MarketId.SOUTH_KOREA_KRX
    assert universe.market_spec.cap_bucket is CapBucket.MID


@pytest.mark.parametrize(
    ("market_id", "cap_bucket"),
    (
        (MarketId.UK_LSE, CapBucket.SMALL),
        (MarketId.AUSTRALIA_ASX, CapBucket.MID),
        (MarketId.US_ALL, CapBucket.SMALL),
    ),
)
def test_session_hard_hv_is_a_paper_strategy_for_every_supported_market(
    market_id: MarketId, cap_bucket: CapBucket
) -> None:
    _config, run = UniverseRunBuilder().add(
        empty_config(),
        market_id=market_id,
        cap_bucket=cap_bucket,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        environment=Environment.PAPER,
    )

    assert run.strategy == "SESSION_HARD_HV"
    assert run.strategy_id == "SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D"
    assert run.strategy_version == "SESSION_HARD_HV_V1"
    assert "hard-hv" in run.run_id
    assert run.environment is Environment.PAPER


def test_session_hard_hv_rejects_live_explicitly_even_with_matching_paper() -> None:
    builder = UniverseRunBuilder()
    paper_config, _paper = builder.add(
        empty_config(),
        market_id=MarketId.UK_LSE,
        cap_bucket=CapBucket.SMALL,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        environment=Environment.PAPER,
    )

    with pytest.raises(ValueError, match="SESSION_HARD_HV_V1 is PAPER-only"):
        builder.add(
            paper_config,
            market_id=MarketId.UK_LSE,
            cap_bucket=CapBucket.SMALL,
            strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
            strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
            environment=Environment.LIVE,
        )

    live_payload = paper_config.runs[0].model_dump(mode="python")
    live_payload["environment"] = Environment.LIVE
    with pytest.raises(ValueError, match="SESSION_HARD_HV_V1 is PAPER-only"):
        RunConfig.model_validate(live_payload)


def test_disable_and_readd_reuses_exact_lineage() -> None:
    builder = UniverseRunBuilder()
    config, created = builder.add(
        empty_config(),
        market_id=MarketId.UK_LSE,
        cap_bucket=CapBucket.LARGE,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        environment=Environment.PAPER,
    )
    disabled = builder.disable(config, created.run_id)
    assert len(disabled.runs) == 1
    assert disabled.runs[0].enabled is False

    reenabled, same = builder.add(
        disabled,
        market_id=MarketId.UK_LSE,
        cap_bucket=CapBucket.LARGE,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        environment=Environment.PAPER,
    )
    assert same.run_id == created.run_id
    assert same.enabled is True
    assert len(reenabled.runs) == 1


def test_exchange_specific_us_run_requires_authoritative_membership() -> None:
    builder = UniverseRunBuilder()
    config = RunsConfig(
        universes=(
            UniverseDefinition(
                universe_id="CUSTOM",
                name="Custom",
                members=(
                    {
                        "symbol": "VOD",
                        "exchange": "SMART",
                        "primary_exchange": "LSE",
                        "currency": "GBP",
                    },
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="authoritative listing membership"):
        builder.add(
            config,
            market_id=MarketId.US_NYSE,
            cap_bucket=CapBucket.MID,
            strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
            strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
            environment=Environment.PAPER,
        )


def test_activity_profile_and_generated_lineage_fail_closed_when_mislabeled() -> None:
    with pytest.raises(ValueError, match="max_results 50"):
        RunScreenConfig(
            method=CandidateScreen.ACTIVITY_SHORTLIST_V1,
            max_results=49,
            version="ACTIVITY_SHORTLIST_V1",
            scheduled_active_minutes=15,
        )

    config, created = UniverseRunBuilder().add(
        empty_config(),
        market_id=MarketId.US_NASDAQ,
        cap_bucket=CapBucket.MID,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        environment=Environment.PAPER,
    )
    del config
    payload = created.model_dump(mode="python")
    payload["candidate_screen_id"] = "MISLABELED"
    with pytest.raises(ValueError, match="matching immutable lineage"):
        RunConfig.model_validate(payload)


def test_changing_markets_or_disabling_run_preserves_hv_cost_input():
    builder = UniverseRunBuilder()
    config = empty_config().model_copy(update={"session_hard_hv_round_trip_cost_bps": 7.5})
    updated, run = builder.add(
        config,
        market_id=MarketId.UK_LSE,
        cap_bucket=CapBucket.SMALL,
        strategy_id=SESSION_HARD_HV_METHOD.strategy_id,
        strategy_version=SESSION_HARD_HV_METHOD.strategy_version,
        environment=Environment.PAPER,
    )
    assert updated.session_hard_hv_round_trip_cost_bps == 7.5
    assert builder.disable(updated, run.run_id).session_hard_hv_round_trip_cost_bps == 7.5
