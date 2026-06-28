from pathlib import Path

from stocker_core.config import (
    ResearchConfig,
    ServerConfig,
    load_research_config,
    load_server_config,
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


def test_load_server_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "server.yaml"
    config_path.write_text(
        """
data:
  data_dir: /srv/stocker/data
  timezone: UTC
  default_currency: USD
costs:
  spread_bps: 2.0
  commission_bps: 1.0
  slippage_bps: 0.5
risk:
  max_position_size: 5000
  max_order_size: 1000
  max_daily_loss: 250
  max_orders_per_day: 5
  trading_enabled: false
server:
  mode: paper
  host: 127.0.0.1
  port: 8000
  broker:
    provider: placeholder
    account_id_env: STOCKER_BROKER_ACCOUNT_ID
""",
        encoding="utf-8",
    )

    config = load_server_config(config_path)

    assert isinstance(config, ServerConfig)
    assert config.server.mode == "paper"
    assert config.server.broker.provider == "placeholder"
    assert config.risk.max_orders_per_day == 5
