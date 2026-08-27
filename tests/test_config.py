from pathlib import Path

from stocker_core.config import (
    ResearchConfig,
    load_research_config,
)


def test_load_research_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "research.yaml"
    config_path.write_text(
        """
data:
  data_dir: ./research-data
  timezone: Europe/London
  default_currency: GBP
costs:
  spread_bps: 1.5
  commission_bps: 0.5
  slippage_bps: 0.25
risk:
  max_position_size: 10000
  max_order_size: 2500
  max_daily_loss: 500
  max_orders_per_day: 12
  trading_enabled: false
research:
  starting_cash: 100000
  benchmark_symbol: SPY
""",
        encoding="utf-8",
    )

    config = load_research_config(config_path)

    assert isinstance(config, ResearchConfig)
    assert config.data.data_dir == Path("research-data")
    assert config.data.timezone == "Europe/London"
    assert config.costs.round_trip_bps() == 4.5
    assert config.risk.trading_enabled is False
    assert config.research.starting_cash == 100000
