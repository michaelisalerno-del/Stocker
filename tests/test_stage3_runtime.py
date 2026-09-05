import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import load_runs_config
from stocker_core.runs import Environment, RunConfig, RunManager, RunState, RunWindow
from stocker_core.universes import (
    NASDAQ_LISTED_URL,
    OTHER_LISTED_URL,
    InstrumentReference,
    UniverseCatalog,
    UniverseDefinition,
    load_us_universe_snapshot,
    refresh_us_universe_snapshot,
)


def test_multiple_substantially_different_universes_can_coexist() -> None:
    nasdaq = UniverseDefinition(
        universe_id="NASDAQ_TEST",
        name="NASDAQ test stocks",
        members=(
            InstrumentReference(
                symbol="AAPL",
                exchange="SMART",
                primary_exchange="NASDAQ",
                currency="USD",
            ),
        ),
    )
    ftse = UniverseDefinition(
        universe_id="FTSE_TEST",
        name="FTSE test stocks",
        members=(
            InstrumentReference(
                symbol="AZN",
                exchange="SMART",
                primary_exchange="LSE",
                currency="GBP",
            ),
        ),
    )

    catalog = UniverseCatalog((nasdaq, ftse))

    assert catalog.get_members("NASDAQ_TEST")[0].symbol == "AAPL"
    assert catalog.get_members("FTSE_TEST")[0].currency == "GBP"


def test_custom_universe_and_multiple_runs_load_without_runtime_code_changes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runs.yaml"
    config_path.write_text(
        """
universes:
  - universe_id: NASDAQ_TEST
    name: NASDAQ test stocks
    members:
      - symbol: AAPL
        exchange: SMART
        primary_exchange: NASDAQ
        currency: USD
  - universe_id: CUSTOM_RESEARCH
    name: User research list
    members:
      - symbol: aapl
        exchange: smart
        primary_exchange: nasdaq
        currency: usd
      - symbol: AAPL
        exchange: SMART
        primary_exchange: NASDAQ
        currency: USD
      - symbol: PLTR
        exchange: SMART
        primary_exchange: NASDAQ
        currency: USD
runs:
  - run_id: nasdaq_main
    universe: NASDAQ_TEST
    strategy: TEST_EXECUTION
    environment: PAPER
  - run_id: custom_experiment
    universe: CUSTOM_RESEARCH
    strategy: NEW_IDEA
    environment: PAPER
""",
        encoding="utf-8",
    )

    loaded = load_runs_config(config_path)

    assert [universe.universe_id for universe in loaded.universes] == [
        "NASDAQ_TEST",
        "CUSTOM_RESEARCH",
    ]
    assert [member.symbol for member in loaded.universes[1].members] == ["AAPL", "PLTR"]
    assert [run.run_id for run in loaded.runs] == ["nasdaq_main", "custom_experiment"]


def test_named_us_universes_load_real_members_from_one_timestamped_snapshot(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "us-listed.csv"
    snapshot_path.write_text(
        """# schema_version=1
# source=NASDAQ_TRADER_SYMBOL_DIRECTORY
# source_urls=https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt,https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt
# retrieved_at=2026-09-02T12:00:00+00:00
# nasdaq_source_updated_at=0902202611:55
# other_source_updated_at=0902202611:56
# universes=US_ALL,NASDAQ,NYSE
symbol,primary_exchange
AAPL,NASDAQ
MSFT,NASDAQ
IBM,NYSE
IBM,NYSE
PLTR,NYSE
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "runs.yaml"
    config_path.write_text(
        f"""
named_universe_snapshot: {snapshot_path.name}
universes:
  - universe_id: CUSTOM
    name: Custom stocks
    members:
      - {{symbol: aapl, exchange: smart, primary_exchange: nasdaq, currency: usd}}
      - {{symbol: AAPL, exchange: SMART, primary_exchange: NASDAQ, currency: USD}}
runs:
  - run_id: nasdaq
    universe: NASDAQ
    strategy: TEST_EXECUTION
    environment: PAPER
  - run_id: nyse
    universe: NYSE
    strategy: TEST_EXECUTION
    environment: PAPER
  - run_id: all
    universe: US_ALL
    strategy: TEST_EXECUTION
    environment: PAPER
  - run_id: custom
    universe: CUSTOM
    strategy: TEST_EXECUTION
    environment: PAPER
""",
        encoding="utf-8",
    )

    loaded = load_runs_config(config_path)
    catalog = UniverseCatalog(loaded.universes)

    assert [member.symbol for member in catalog.get_members("NASDAQ")] == ["AAPL", "MSFT"]
    assert [member.symbol for member in catalog.get_members("NYSE")] == ["IBM", "PLTR"]
    assert [member.symbol for member in catalog.get_members("US_ALL")] == [
        "AAPL",
        "IBM",
        "MSFT",
        "PLTR",
    ]
    assert catalog.get_members("CUSTOM") == (
        InstrumentReference(
            symbol="AAPL",
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
        ),
    )


def test_us_listing_refresh_filters_funds_and_test_issues_and_retains_provenance(
    tmp_path: Path,
) -> None:
    source_files = {
        NASDAQ_LISTED_URL: (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
            "Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc. - Common Stock|Q|N|N|40|N|N\n"
            "QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\n"
            "ZTEST|Test issue|S|Y|N|100|N|N\n"
            "File Creation Time: 0902202611:55|||||||\n"
        ),
        OTHER_LISTED_URL: (
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
            "Test Issue|NASDAQ Symbol\n"
            "IBM|International Business Machines Common Stock|N|IBM|N|40|N|IBM\n"
            "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"
            "ATEST|Test issue|A|ATEST|N|100|Y|ATEST\n"
            "File Creation Time: 0902202611:56|||||||\n"
        ),
    }
    output = tmp_path / "us-listed.csv"

    result = refresh_us_universe_snapshot(
        output,
        fetch_text=source_files.__getitem__,
        clock=lambda: datetime(2026, 9, 2, 12, 0, tzinfo=ZoneInfo("UTC")),
    )
    snapshot = load_us_universe_snapshot(output)

    assert result.us_all_count == 2
    assert result.nasdaq_count == 1
    assert result.nyse_count == 1
    assert [member.symbol for member in snapshot.members] == ["AAPL", "IBM"]
    assert snapshot.metadata.source == "NASDAQ_TRADER_SYMBOL_DIRECTORY"
    assert snapshot.metadata.retrieved_at.isoformat() == "2026-09-02T12:00:00+00:00"
    assert snapshot.metadata.nasdaq_source_updated_at == "0902202611:55"
    assert snapshot.metadata.other_source_updated_at == "0902202611:56"


def test_run_referencing_an_unknown_universe_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "runs.yaml"
    config_path.write_text(
        """
universes:
  - universe_id: NASDAQ_TEST
    name: NASDAQ test stocks
    members:
      - symbol: AAPL
        exchange: SMART
        currency: USD
runs:
  - run_id: unknown_universe_run
    universe: MISSING
    strategy: TEST_EXECUTION
    environment: PAPER
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown universe MISSING"):
        load_runs_config(config_path)


def test_duplicate_run_ids_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "runs.yaml"
    config_path.write_text(
        """
universes:
  - universe_id: NASDAQ_TEST
    name: NASDAQ test stocks
    members:
      - symbol: AAPL
        exchange: SMART
        currency: USD
runs:
  - run_id: duplicate
    universe: NASDAQ_TEST
    strategy: TEST_EXECUTION
    environment: PAPER
  - run_id: duplicate
    universe: NASDAQ_TEST
    strategy: NEW_IDEA
    environment: LIVE
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate run_id: duplicate"):
        load_runs_config(config_path)


def test_starting_two_runs_with_different_universes_keeps_both_active() -> None:
    nasdaq = UniverseDefinition(
        universe_id="NASDAQ_TEST",
        name="NASDAQ test stocks",
        members=(InstrumentReference(symbol="AAPL", exchange="SMART", currency="USD"),),
    )
    ftse = UniverseDefinition(
        universe_id="FTSE_TEST",
        name="FTSE test stocks",
        members=(InstrumentReference(symbol="AZN", exchange="SMART", currency="GBP"),),
    )
    manager = RunManager(
        UniverseCatalog((nasdaq, ftse)),
        (
            RunConfig(
                run_id="nasdaq_main",
                universe="NASDAQ_TEST",
                strategy="TEST_EXECUTION",
                environment=Environment.PAPER,
            ),
            RunConfig(
                run_id="ftse_morning",
                universe="FTSE_TEST",
                strategy="TEST_EXECUTION",
                environment=Environment.PAPER,
            ),
        ),
    )

    manager.start_run("nasdaq_main")
    manager.start_run("ftse_morning")

    assert manager.get_run("nasdaq_main").state is RunState.ACTIVE
    assert manager.get_run("ftse_morning").state is RunState.ACTIVE


def test_active_paper_and_live_runs_can_share_one_universe_and_remain_independent() -> None:
    nasdaq = UniverseDefinition(
        universe_id="NASDAQ_TEST",
        name="NASDAQ test stocks",
        members=(InstrumentReference(symbol="AMD", exchange="SMART", currency="USD"),),
    )
    manager = RunManager(
        UniverseCatalog((nasdaq,)),
        (
            RunConfig(
                run_id="nasdaq_live",
                universe="NASDAQ_TEST",
                strategy="TEST_EXECUTION",
                environment=Environment.LIVE,
            ),
            RunConfig(
                run_id="nasdaq_experiment",
                universe="NASDAQ_TEST",
                strategy="NEW_IDEA",
                environment=Environment.PAPER,
            ),
        ),
    )

    live = manager.start_run("nasdaq_live")
    experiment = manager.start_run("nasdaq_experiment")
    manager.stop_run("nasdaq_live")

    assert live.universe is experiment.universe
    assert live.config.environment is Environment.LIVE
    assert experiment.config.environment is Environment.PAPER
    assert manager.get_run("nasdaq_live").state is RunState.STOPPED
    assert manager.get_run("nasdaq_experiment").state is RunState.ACTIVE


def test_overlapping_universes_share_value_based_instrument_identity() -> None:
    nasdaq_amd = InstrumentReference(
        symbol="AMD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
    )
    research_amd = InstrumentReference(
        symbol="amd",
        exchange="smart",
        primary_exchange="nasdaq",
        currency="usd",
    )
    catalog = UniverseCatalog(
        (
            UniverseDefinition(
                universe_id="NASDAQ_TEST",
                name="NASDAQ test stocks",
                members=(nasdaq_amd,),
            ),
            UniverseDefinition(
                universe_id="US_RESEARCH",
                name="US research stocks",
                members=(research_amd,),
            ),
        )
    )

    left = catalog.get_members("NASDAQ_TEST")[0]
    right = catalog.get_members("US_RESEARCH")[0]

    assert left == right
    assert hash(left) == hash(right)


def test_run_can_carry_an_optional_session_window_without_scheduling_it() -> None:
    run = RunConfig(
        run_id="ftse_morning",
        universe="FTSE_TEST",
        strategy="TEST_EXECUTION",
        environment=Environment.PAPER,
        session=RunWindow(
            start=time(8, 0),
            end=time(11, 0),
            timezone="Europe/London",
        ),
    )

    assert run.session is not None
    assert run.session.start == time(8, 0)
    assert run.session.timezone == "Europe/London"


def test_run_manager_rejects_duplicate_ids_when_constructed_directly() -> None:
    nasdaq = UniverseDefinition(
        universe_id="NASDAQ_TEST",
        name="NASDAQ test stocks",
        members=(InstrumentReference(symbol="AAPL", exchange="SMART", currency="USD"),),
    )
    duplicate = RunConfig(
        run_id="duplicate",
        universe="NASDAQ_TEST",
        strategy="TEST_EXECUTION",
        environment=Environment.PAPER,
    )

    with pytest.raises(ValueError, match="Duplicate run_id: duplicate"):
        RunManager(UniverseCatalog((nasdaq,)), (duplicate, duplicate))


def test_status_can_activate_multiple_runs_without_loading_ibkr(tmp_path: Path) -> None:
    config_path = tmp_path / "runs.yaml"
    config_path.write_text(
        """
universes:
  - universe_id: NASDAQ_TEST
    name: NASDAQ test stocks
    members:
      - symbol: AAPL
        exchange: SMART
        currency: USD
  - universe_id: FTSE_TEST
    name: FTSE test stocks
    members:
      - symbol: AZN
        exchange: SMART
        primary_exchange: LSE
        currency: GBP
runs:
  - run_id: nasdaq_main
    universe: NASDAQ_TEST
    strategy: TEST_EXECUTION
    environment: PAPER
  - run_id: ftse_morning
    universe: FTSE_TEST
    strategy: TEST_EXECUTION
    environment: PAPER
""",
        encoding="utf-8",
    )
    for module_name in [name for name in sys.modules if name.startswith("stocker_execution")]:
        sys.modules.pop(module_name)

    result = CliRunner().invoke(
        app,
        [
            "runs-status",
            "--config",
            str(config_path),
            "--start",
            "nasdaq_main",
            "--start",
            "ftse_morning",
        ],
    )

    assert result.exit_code == 0
    assert "NASDAQ_TEST members=1" in result.stdout
    assert "FTSE_TEST members=1" in result.stdout
    assert "ACTIVE  nasdaq_main" in result.stdout
    assert "ACTIVE  ftse_morning" in result.stdout
    assert not any(name.startswith("stocker_execution") for name in sys.modules)


def test_run_lifecycle_exposes_no_broker_data_or_order_operations() -> None:
    forbidden_operations = {
        "connect",
        "resolve_stock",
        "historical_bars",
        "current_quote",
        "place_order",
        "submit_order",
        "cancel_order",
    }

    assert forbidden_operations.isdisjoint(dir(RunManager))
