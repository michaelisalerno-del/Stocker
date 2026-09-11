"""Construct original V7 run fixtures explicitly; never select them as new UI runs."""

from stocker_core.markets import MarketId
from stocker_core.methods import LEGACY_SESSION_HARD
from stocker_core.runs import Environment
from stocker_dashboard.universe_runs import UniverseRunBuilder as CurrentBuilder
from test_stage10_extension_builder import empty_config


class UniverseRunBuilder(CurrentBuilder):
    def add(self, *args, **kwargs):
        return super().add(*args, **kwargs, historical_reproduction=True)


def add(config=None, market=MarketId.US_NASDAQ, environment=Environment.PAPER):
    return UniverseRunBuilder().add(
        config or empty_config(),
        market_id=market,
        strategy_id=LEGACY_SESSION_HARD.method_id,
        strategy_version=LEGACY_SESSION_HARD.version,
        environment=environment,
    )
