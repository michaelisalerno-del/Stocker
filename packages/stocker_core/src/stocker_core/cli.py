"""Command-line interface for Stocker."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from stocker_core.config import (
    EODHDConfig,
    ResearchConfig,
    load_research_config,
    load_server_config,
)

console = Console()
app = typer.Typer(no_args_is_help=True, help="Stocker research and execution utilities.")
data_app = typer.Typer(no_args_is_help=True, help="Data utilities.")
research_app = typer.Typer(no_args_is_help=True, help="Research utilities.")
server_app = typer.Typer(no_args_is_help=True, help="Server utilities.")
universe_app = typer.Typer(no_args_is_help=True, help="Universe data-management utilities.")
app.add_typer(data_app, name="data")
app.add_typer(research_app, name="research")
app.add_typer(server_app, name="server")
app.add_typer(universe_app, name="universe")

DEFAULT_RESEARCH_CONFIG = Path("configs/research.example.yaml")


@app.command()
def check() -> None:
    """Run a lightweight environment check."""

    console.print("Stocker CLI is installed and importable.")


@data_app.command("validate")
def data_validate(
    dataset: Annotated[Path | None, typer.Argument(help="Optional Parquet dataset path.")] = None,
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1d",
    timezone: Annotated[str, typer.Option("--timezone")] = "UTC",
) -> None:
    """Run structured data validation checks."""

    if dataset is None:
        console.print("No dataset supplied. Data validators are importable.")
        return
    from stocker_data.storage import dataset_exists, read_parquet
    from stocker_data.validate import validate_ohlcv

    if not dataset_exists(dataset):
        raise typer.BadParameter(f"Dataset does not exist: {dataset}")
    issues = validate_ohlcv(read_parquet(dataset), timeframe=timeframe, timezone=timezone)
    if not issues:
        console.print("No validation issues.")
        return
    for issue in issues:
        console.print(issue.to_dict())


@data_app.command("import-csv")
def data_import_csv(
    file: Annotated[Path, typer.Option("--file", exists=True, file_okay=True, dir_okay=False)],
    symbol: Annotated[str, typer.Option("--symbol")],
    source: Annotated[str, typer.Option("--source")] = "manual",
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1d",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    timezone: Annotated[str, typer.Option("--timezone")] = "UTC",
    currency: Annotated[str, typer.Option("--currency")] = "USD",
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    column_map: Annotated[str | None, typer.Option("--column-map")] = None,
) -> None:
    """Import a local CSV into canonical Parquet storage."""

    from stocker_data.ingest import import_csv

    result = import_csv(
        file_path=file,
        data_dir=data_dir,
        symbol=symbol,
        source=source,
        timeframe=timeframe,
        instrument_type=instrument_type,
        timezone=timezone,
        currency=currency,
        column_mapping=column_map,
    )
    console.print(
        {
            "path": str(result.path),
            "rows": result.rows,
            "errors": result.error_count,
            "warnings": result.warning_count,
            "catalog": str(result.catalog_path),
        }
    )


@data_app.command("catalog")
def data_catalog(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
) -> None:
    """List locally available datasets."""

    from stocker_data.catalog import scan_catalog, write_catalog

    entries = scan_catalog(data_dir=data_dir)
    write_catalog(data_dir=data_dir)
    if not entries:
        console.print("No datasets found.")
        return
    for entry in entries:
        console.print(entry.to_dict())


@data_app.command("show")
def data_show(
    symbol: Annotated[str, typer.Option("--symbol")],
    timeframe: Annotated[str, typer.Option("--timeframe")],
    source: Annotated[str, typer.Option("--source")] = "manual",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
) -> None:
    """Show metadata for one local dataset."""

    from stocker_data.storage import DatasetKey, dataset_metadata

    metadata = dataset_metadata(
        DatasetKey(
            source=source,
            instrument_type=instrument_type,
            symbol=symbol.upper(),
            timeframe=timeframe,
        ),
        data_dir=data_dir,
    )
    console.print(metadata.to_dict())


@data_app.command("audit")
def data_audit(
    symbol: Annotated[str, typer.Option("--symbol")],
    timeframe: Annotated[str, typer.Option("--timeframe")],
    source: Annotated[str, typer.Option("--source")] = "manual",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
) -> None:
    """Create a dataset audit report."""

    from stocker_data.audit import create_audit_report

    report = create_audit_report(
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        instrument_type=instrument_type,
        market_calendar=market_calendar,
    )
    console.print(
        {
            "audit": str(report.markdown_path),
            "json": str(report.json_path),
            "passed": report.passed,
        }
    )


def _check_storage_mode(overwrite: bool, merge: bool) -> None:
    if overwrite and merge:
        raise typer.BadParameter("Use either --overwrite or --merge, not both.")


def _load_research_cli_config(config_path: Path) -> ResearchConfig:
    try:
        return load_research_config(config_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Research config not found: {config_path}") from exc


def _resolve_data_dir(config: ResearchConfig, data_dir: Path | None) -> Path:
    return data_dir if data_dir is not None else config.data.data_dir


def _resolve_currency(config: ResearchConfig, currency: str | None) -> str:
    return currency if currency is not None else config.data.default_currency


def _resolve_save_raw(eodhd_config: EODHDConfig, save_raw: bool | None) -> bool:
    return save_raw if save_raw is not None else eodhd_config.save_raw_by_default


def _require_eodhd_enabled(
    eodhd_config: EODHDConfig,
    *,
    dry_run: bool,
    enable_disabled_vendor: bool,
    config_path: Path,
) -> None:
    if dry_run or eodhd_config.enabled or enable_disabled_vendor:
        return
    raise typer.BadParameter(
        "EODHD is disabled in research config "
        f"{config_path}. Enable data_vendors.eodhd.enabled or pass --enable-disabled-vendor."
    )


def _run_eodhd_qa(
    *,
    data_dir: Path,
    symbol: str,
    timeframe: str,
    instrument_type: str,
    market_calendar: str | None,
    adjusted_price_policy: str,
    require_raw: bool,
) -> dict[str, object]:
    from stocker_data.vendors.eodhd_qa import create_eodhd_qa_report

    report = create_eodhd_qa_report(
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
        instrument_type=instrument_type,
        market_calendar=market_calendar,
        adjusted_price_policy=adjusted_price_policy,
        require_raw=require_raw,
    )
    return report.to_dict()


@data_app.command("qa-eodhd")
def data_qa_eodhd(
    symbol: Annotated[str, typer.Option("--symbol")],
    timeframe: Annotated[str, typer.Option("--timeframe")],
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
    adjusted_price_policy: Annotated[
        str, typer.Option("--adjusted-price-policy")
    ] = "adjusted_available",
    require_raw: Annotated[bool, typer.Option("--require-raw/--no-require-raw")] = False,
) -> None:
    """Create an EODHD-specific vendor QA report for a normalized dataset."""

    if source != "eodhd":
        raise typer.BadParameter("qa-eodhd only supports --source eodhd.")
    loaded = _load_research_cli_config(config)
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    console.print(
        {
            "source": "eodhd",
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "config_path": str(config),
            "data_dir": str(resolved_data_dir),
            "qa": _run_eodhd_qa(
                data_dir=resolved_data_dir,
                symbol=symbol,
                timeframe=timeframe,
                instrument_type=instrument_type,
                market_calendar=market_calendar,
                adjusted_price_policy=adjusted_price_policy,
                require_raw=require_raw,
            ),
        }
    )


@data_app.command("fetch-eodhd-eod")
def data_fetch_eodhd_eod(
    symbol: Annotated[str, typer.Option("--symbol")],
    from_date: Annotated[str, typer.Option("--from")],
    to_date: Annotated[str, typer.Option("--to")],
    period: Annotated[str, typer.Option("--period")] = "d",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
    currency: Annotated[str | None, typer.Option("--currency")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    save_raw: Annotated[bool | None, typer.Option("--save-raw/--no-save-raw")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    merge: Annotated[bool, typer.Option("--merge")] = False,
    audit: Annotated[bool, typer.Option("--audit")] = False,
    qa: Annotated[bool, typer.Option("--qa")] = False,
    enable_disabled_vendor: Annotated[
        bool,
        typer.Option(
            "--enable-disabled-vendor",
            help="Allow a live fetch even when data_vendors.eodhd.enabled is false.",
        ),
    ] = False,
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
    adjusted_price_policy: Annotated[
        str, typer.Option("--adjusted-price-policy")
    ] = "adjusted_available",
) -> None:
    """Fetch EODHD EOD data into normalized Stocker Parquet storage."""

    from stocker_data.vendors import eodhd

    _check_storage_mode(overwrite, merge)
    loaded = _load_research_cli_config(config)
    eodhd_config = loaded.data_vendors.eodhd
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    resolved_currency = _resolve_currency(loaded, currency)
    resolved_save_raw = _resolve_save_raw(eodhd_config, save_raw)
    _require_eodhd_enabled(
        eodhd_config,
        dry_run=dry_run,
        enable_disabled_vendor=enable_disabled_vendor,
        config_path=config,
    )
    if dry_run:
        plan = eodhd.plan_eod_fetch(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            period=period,
            instrument_type=instrument_type,
            data_dir=resolved_data_dir,
            save_raw=resolved_save_raw,
        )
        console.print(
            {
                "dry_run": True,
                "source": "eodhd",
                "symbol": symbol.upper(),
                "config_path": str(config),
                "vendor_enabled": eodhd_config.enabled,
                "data_dir": str(resolved_data_dir),
                "currency": resolved_currency,
                **plan.to_dict(),
            }
        )
        return

    result = eodhd.fetch_eod_to_storage(
        client=eodhd.EODHDClient(config=eodhd_config),
        data_dir=resolved_data_dir,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        period=period,
        instrument_type=instrument_type,
        currency=resolved_currency,
        save_raw=resolved_save_raw,
        overwrite=overwrite,
        merge=merge,
        audit=audit,
        market_calendar=market_calendar,
    )
    timeframe = eodhd.timeframe_for_eod_period(period)
    output: dict[str, object] = {
        "source": "eodhd",
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "config_path": str(config),
        "vendor_enabled": eodhd_config.enabled,
        "data_dir": str(resolved_data_dir),
        "currency": resolved_currency,
        **result.to_dict(),
    }
    if qa:
        output["qa"] = _run_eodhd_qa(
            data_dir=resolved_data_dir,
            symbol=symbol,
            timeframe=timeframe,
            instrument_type=instrument_type,
            market_calendar=market_calendar,
            adjusted_price_policy=adjusted_price_policy,
            require_raw=resolved_save_raw,
        )
    console.print(output)


@data_app.command("fetch-eodhd-intraday")
def data_fetch_eodhd_intraday(
    symbol: Annotated[str, typer.Option("--symbol")],
    interval: Annotated[str, typer.Option("--interval")],
    from_date: Annotated[str, typer.Option("--from")],
    to_date: Annotated[str, typer.Option("--to")],
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
    currency: Annotated[str | None, typer.Option("--currency")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    save_raw: Annotated[bool | None, typer.Option("--save-raw/--no-save-raw")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    merge: Annotated[bool, typer.Option("--merge")] = False,
    audit: Annotated[bool, typer.Option("--audit")] = False,
    qa: Annotated[bool, typer.Option("--qa")] = False,
    enable_disabled_vendor: Annotated[
        bool,
        typer.Option(
            "--enable-disabled-vendor",
            help="Allow a live fetch even when data_vendors.eodhd.enabled is false.",
        ),
    ] = False,
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
    adjusted_price_policy: Annotated[str, typer.Option("--adjusted-price-policy")] = "raw_close",
) -> None:
    """Fetch chunked EODHD intraday data into normalized Stocker Parquet storage."""

    from stocker_data.vendors import eodhd

    _check_storage_mode(overwrite, merge)
    loaded = _load_research_cli_config(config)
    eodhd_config = loaded.data_vendors.eodhd
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    resolved_currency = _resolve_currency(loaded, currency)
    resolved_save_raw = _resolve_save_raw(eodhd_config, save_raw)
    _require_eodhd_enabled(
        eodhd_config,
        dry_run=dry_run,
        enable_disabled_vendor=enable_disabled_vendor,
        config_path=config,
    )
    if dry_run:
        plan = eodhd.plan_intraday_fetch(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            instrument_type=instrument_type,
            data_dir=resolved_data_dir,
            save_raw=resolved_save_raw,
        )
        console.print(
            {
                "dry_run": True,
                "source": "eodhd",
                "symbol": symbol.upper(),
                "config_path": str(config),
                "vendor_enabled": eodhd_config.enabled,
                "data_dir": str(resolved_data_dir),
                "currency": resolved_currency,
                **plan.to_dict(),
            }
        )
        return

    result = eodhd.fetch_intraday_to_storage(
        client=eodhd.EODHDClient(config=eodhd_config),
        data_dir=resolved_data_dir,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
        instrument_type=instrument_type,
        currency=resolved_currency,
        save_raw=resolved_save_raw,
        overwrite=overwrite,
        merge=merge,
        audit=audit,
        market_calendar=market_calendar,
    )
    output = {
        "source": "eodhd",
        "symbol": symbol.upper(),
        "timeframe": interval,
        "config_path": str(config),
        "vendor_enabled": eodhd_config.enabled,
        "data_dir": str(resolved_data_dir),
        "currency": resolved_currency,
        **result.to_dict(),
    }
    if qa:
        output["qa"] = _run_eodhd_qa(
            data_dir=resolved_data_dir,
            symbol=symbol,
            timeframe=interval,
            instrument_type=instrument_type,
            market_calendar=market_calendar,
            adjusted_price_policy=adjusted_price_policy,
            require_raw=resolved_save_raw,
        )
    console.print(output)


def _require_vendor_for_live(
    eodhd_config: EODHDConfig,
    *,
    dry_run: bool,
    config_path: Path,
) -> None:
    if dry_run or eodhd_config.enabled:
        return
    raise typer.BadParameter(
        f"EODHD is disabled in research config {config_path}; enable it before live universe work."
    )


@universe_app.command("build-eodhd")
def universe_build_eodhd(
    universe_id: Annotated[str, typer.Option("--id")],
    name: Annotated[str, typer.Option("--name")],
    exchange: Annotated[str, typer.Option("--exchange")],
    output: Annotated[Path, typer.Option("--output")],
    description: Annotated[str, typer.Option("--description")] = "",
    min_price: Annotated[float | None, typer.Option("--min-price")] = None,
    min_market_cap: Annotated[float | None, typer.Option("--min-market-cap")] = None,
    min_avgvol_200d: Annotated[float | None, typer.Option("--min-avgvol-200d")] = None,
    sector: Annotated[list[str] | None, typer.Option("--sector")] = None,
    industry: Annotated[list[str] | None, typer.Option("--industry")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 100,
    max_pages: Annotated[int, typer.Option("--max-pages", min=1)] = 10,
    sort: Annotated[str, typer.Option("--sort")] = "market_capitalization.desc",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
) -> None:
    """Build a universe from EODHD screener results."""

    from stocker_data.universe import (
        UniverseDefinition,
        UniverseFilters,
        save_universe,
        write_universe_build_report,
    )
    from stocker_data.vendors import eodhd

    loaded = _load_research_cli_config(config)
    eodhd_config = loaded.data_vendors.eodhd
    _require_vendor_for_live(eodhd_config, dry_run=dry_run, config_path=config)
    filters = eodhd.build_screener_filters(
        exchange=exchange,
        min_price=min_price,
        min_market_cap=min_market_cap,
        min_avgvol_200d=min_avgvol_200d,
        sectors=sector,
        industries=industry,
    )
    planned_requests: list[dict[str, int]] = []
    remaining = limit
    offset = 0
    while remaining > 0 and len(planned_requests) < max_pages:
        page_limit = min(eodhd.SCREENER_MAX_LIMIT, remaining)
        planned_requests.append({"limit": page_limit, "offset": offset})
        remaining -= page_limit
        offset += page_limit
    if dry_run:
        console.print(
            {
                "dry_run": True,
                "source": "eodhd_screener",
                "config_path": str(config),
                "vendor_enabled": eodhd_config.enabled,
                "output": str(output),
                "filters": filters,
                "sort": sort,
                "planned_requests": planned_requests,
            }
        )
        return

    client = eodhd.EODHDClient(config=eodhd_config)
    symbols = eodhd.fetch_screener_all(
        client=client,
        filters=filters,
        signals=[],
        sort=sort,
        limit=limit,
        max_pages=max_pages,
    )
    deduped = {symbol.symbol: symbol for symbol in symbols}
    universe = UniverseDefinition(
        id=universe_id,
        name=name,
        description=description or f"{name} generated from EODHD screener",
        source="eodhd_screener",
        created_at=__import__("datetime")
        .datetime.now(tz=__import__("datetime").UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        filters=UniverseFilters(
            exchange=exchange,
            min_price=min_price,
            min_market_cap=min_market_cap,
            min_avgvol_200d=min_avgvol_200d,
            sectors=sector or [],
            industries=industry or [],
        ),
        symbols=list(deduped.values()),
    )
    save_universe(universe, output)
    report = write_universe_build_report(
        universe=universe,
        output_path=output,
        data_dir=loaded.data.data_dir,
        filters={"filters": filters, "sort": sort, "limit": limit, "max_pages": max_pages},
    )
    console.print(report.model_dump(mode="json"))


@universe_app.command("validate")
def universe_validate(
    universe: Annotated[Path, typer.Option("--universe", exists=True, file_okay=True)],
) -> None:
    """Validate a universe YAML/JSON file."""

    from stocker_data.universe import load_universe, validate_universe

    loaded = load_universe(universe)
    issues = validate_universe(loaded)
    payload = {
        "universe_id": loaded.id,
        "symbol_count": len(loaded.symbols),
        "issues": [issue.model_dump(mode="json") for issue in issues],
    }
    console.print(payload)
    if any(issue.severity == "error" for issue in issues):
        raise typer.Exit(1)


@universe_app.command("fetch")
def universe_fetch(
    universe: Annotated[Path, typer.Option("--universe", exists=True, file_okay=True)],
    from_date: Annotated[str, typer.Option("--from")],
    to_date: Annotated[str, typer.Option("--to")],
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1d",
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    merge: Annotated[bool, typer.Option("--merge")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    audit: Annotated[bool, typer.Option("--audit")] = False,
    qa: Annotated[bool, typer.Option("--qa")] = False,
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
    max_symbols: Annotated[int | None, typer.Option("--max-symbols")] = None,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    sleep_seconds_between_symbols: Annotated[
        float, typer.Option("--sleep-seconds-between-symbols", min=0.0)
    ] = 0.0,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    skip_existing: Annotated[bool, typer.Option("--skip-existing")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Batch fetch history for every symbol in a universe."""

    from stocker_data.universe import BatchFetchOptions, load_universe, run_universe_fetch

    _check_storage_mode(overwrite, merge)
    loaded_config = _load_research_cli_config(config)
    eodhd_config = loaded_config.data_vendors.eodhd
    _require_vendor_for_live(eodhd_config, dry_run=dry_run, config_path=config)
    resolved_data_dir = _resolve_data_dir(loaded_config, data_dir)
    loaded_universe = load_universe(universe)
    try:
        result = run_universe_fetch(
            universe=loaded_universe,
            data_dir=resolved_data_dir,
            options=BatchFetchOptions(
                from_date=from_date,
                to_date=to_date,
                timeframe=timeframe,
                source=source,
                currency=loaded_config.data.default_currency,
                merge=merge,
                overwrite=overwrite,
                audit=audit,
                qa=qa,
                market_calendar=market_calendar,
                max_symbols=max_symbols,
                fail_fast=fail_fast,
                sleep_seconds_between_symbols=sleep_seconds_between_symbols,
                resume=resume,
                skip_existing=skip_existing,
                dry_run=dry_run,
            ),
            eodhd_config=eodhd_config,
        )
    except RuntimeError as exc:
        console.print({"status": "failed", "error": str(exc)})
        raise typer.Exit(1) from exc
    console.print(result.model_dump(mode="json"))


@universe_app.command("qualify")
def universe_qualify(
    universe: Annotated[Path, typer.Option("--universe", exists=True, file_okay=True)],
    output: Annotated[Path, typer.Option("--output")],
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1d",
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    min_history_days: Annotated[int, typer.Option("--min-history-days", min=0)] = 750,
    min_row_count: Annotated[int, typer.Option("--min-row-count", min=0)] = 0,
    min_sessions: Annotated[int, typer.Option("--min-sessions", min=0)] = 0,
    min_last_close: Annotated[float, typer.Option("--min-last-close", min=0.0)] = 5.0,
    min_median_dollar_volume_60d: Annotated[
        float, typer.Option("--min-median-dollar-volume-60d", min=0.0)
    ] = 10_000_000.0,
    max_validation_errors: Annotated[int, typer.Option("--max-validation-errors", min=0)] = 0,
    max_missing_session_warnings: Annotated[
        int, typer.Option("--max-missing-session-warnings", min=0)
    ] = 5,
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
) -> None:
    """Filter a universe to symbols with sufficient local data quality/liquidity."""

    from stocker_data.universe import (
        UniverseQualificationRules,
        load_universe,
        qualify_universe,
    )

    result = qualify_universe(
        universe=load_universe(universe),
        data_dir=data_dir,
        timeframe=timeframe,
        source=source,
        rules=UniverseQualificationRules(
            min_history_days=min_history_days,
            min_row_count=min_row_count,
            min_sessions=min_sessions,
            min_last_close=min_last_close,
            min_median_dollar_volume_60d=min_median_dollar_volume_60d,
            max_validation_errors=max_validation_errors,
            max_missing_session_warnings=max_missing_session_warnings,
        ),
        output_path=output,
        market_calendar=market_calendar,
    )
    console.print(result.model_dump(mode="json"))


@universe_app.command("health")
def universe_health(
    universe: Annotated[Path, typer.Option("--universe", exists=True, file_okay=True)],
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1d",
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
) -> None:
    """Write a universe health report from local datasets and prior reports."""

    from stocker_data.universe import load_universe, universe_health_report

    report = universe_health_report(
        universe=load_universe(universe),
        data_dir=data_dir,
        timeframe=timeframe,
        source=source,
    )
    console.print(report.model_dump(mode="json"))


@universe_app.command("list")
def universe_list() -> None:
    """List available universe definition and research-ready files."""

    from stocker_data.universe import list_universe_files

    files = list_universe_files()
    if not files:
        console.print("No universe files found.")
        return
    for path in files:
        console.print(str(path))


@research_app.command("baseline")
def research_baseline(
    dataset: Annotated[
        Path | None, typer.Argument(help="Optional OHLC Parquet dataset path.")
    ] = None,
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1d",
    source: Annotated[str, typer.Option("--source")] = "manual",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    spread_bps: Annotated[float, typer.Option("--spread-bps")] = 0.0,
    commission_bps: Annotated[float, typer.Option("--commission-bps")] = 0.0,
    slippage_bps: Annotated[float, typer.Option("--slippage-bps")] = 0.0,
) -> None:
    """Run minimal baseline research."""

    if dataset is None and symbol is None:
        console.print("Supply a dataset path or --symbol/--timeframe for a stored dataset.")
        return
    if symbol is not None:
        from stocker_research.baselines import create_baseline_report

        report = create_baseline_report(
            data_dir=data_dir,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            instrument_type=instrument_type,
            spread_bps=spread_bps,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        console.print({"baseline": str(report.markdown_path), "json": str(report.json_path)})
        return

    from stocker_data.storage import read_parquet
    from stocker_research.baselines import ohlc_baseline_summary

    if dataset is None:
        raise typer.BadParameter("dataset cannot be None")
    summary = ohlc_baseline_summary(read_parquet(dataset))
    console.print(summary)


@research_app.command("run")
def research_run(
    hypothesis: Annotated[Path, typer.Option("--hypothesis", exists=True, file_okay=True)],
    symbol: Annotated[str, typer.Option("--symbol")],
    timeframe: Annotated[str, typer.Option("--timeframe")],
    source: Annotated[str, typer.Option("--source")] = "manual",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """Run a disciplined research experiment from a written hypothesis."""

    from stocker_research.experiments import run_research_experiment

    loaded = _load_research_cli_config(config)
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    result = run_research_experiment(
        hypothesis_path=hypothesis,
        data_dir=resolved_data_dir,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        instrument_type=instrument_type,
        market_calendar=market_calendar,
    )
    console.print(
        {
            "experiment_id": result.experiment_id,
            "classification": result.classification,
            "report": str(result.markdown_path),
            "json": str(result.json_path),
            "config_path": str(config),
            "data_dir": str(resolved_data_dir),
        }
    )


@research_app.command("run-universe")
def research_run_universe(
    hypothesis: Annotated[Path, typer.Option("--hypothesis", exists=True, file_okay=True)],
    qualified_universe: Annotated[
        Path, typer.Option("--qualified-universe", exists=True, file_okay=True)
    ],
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    timeframe: Annotated[str | None, typer.Option("--timeframe")] = None,
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = None,
    max_symbols: Annotated[int | None, typer.Option("--max-symbols")] = None,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    skip_existing: Annotated[bool, typer.Option("--skip-existing")] = False,
) -> None:
    """Run a written hypothesis across a research-ready universe export."""

    from stocker_research.experiments import run_universe_research

    loaded = _load_research_cli_config(config)
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    result = run_universe_research(
        hypothesis_path=hypothesis,
        qualified_universe_path=qualified_universe,
        data_dir=resolved_data_dir,
        source=source,
        timeframe=timeframe,
        instrument_type=instrument_type,
        max_symbols=max_symbols,
        fail_fast=fail_fast,
        resume=resume,
        skip_existing=skip_existing,
        market_calendar=market_calendar,
    )
    console.print(
        {
            "run_id": result.run_id,
            "classification_counts": result.classification_counts,
            "failed_count": result.failed_count,
            "report": str(result.markdown_path),
            "json": str(result.json_path),
            "config_path": str(config),
            "data_dir": str(resolved_data_dir),
        }
    )


@research_app.command("failure-anatomy")
def research_failure_anatomy(
    reports_dir: Annotated[Path, typer.Option("--reports-dir")] = Path("data/reports/research"),
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Build Stage 3.6 diagnostics from existing research reports."""

    from stocker_research.failure_anatomy import build_failure_anatomy_summary

    result = build_failure_anatomy_summary(report_root=reports_dir, output_dir=output_dir)
    files_created = [str(result.summary_json_path), str(result.summary_markdown_path)]
    if result.selected_cases_csv_path is not None:
        files_created.append(str(result.selected_cases_csv_path))
    console.print(
        {
            "output_name": "stage3_6_failure_anatomy",
            "files_created": files_created,
            "report_count_analyzed": result.report_count_analyzed,
            "malformed_report_count": result.malformed_report_count,
            "classification_counts": result.classification_counts,
            "top_diagnostic_findings": result.top_diagnostic_findings,
            "recommended_next_step": result.recommended_next_step,
        }
    )


@research_app.command("intraday-session-integrity")
def research_intraday_session_integrity(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = Path("data/reports/research/stage3_8_intraday_session_integrity"),
    stage3_7_summary: Annotated[
        Path, typer.Option("--stage3-7-summary", exists=True, file_okay=True)
    ] = Path("data/reports/research/stage3_7_intraday_5m_session_flat_smoke/summary.json"),
    symbol: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    timeframe: Annotated[str, typer.Option("--timeframe")] = "5m",
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    market_calendar: Annotated[str, typer.Option("--market-calendar")] = "XNYS",
) -> None:
    """Build Stage 3.8 intraday session-integrity diagnostics from local reports/data."""

    from stocker_research.intraday_session_integrity import (
        DEFAULT_SYMBOLS,
        build_intraday_session_integrity_summary,
    )

    result = build_intraday_session_integrity_summary(
        data_dir=data_dir,
        output_dir=output_dir,
        symbols=symbol or DEFAULT_SYMBOLS,
        timeframe=timeframe,
        source=source,
        instrument_type=instrument_type,
        market_calendar=market_calendar,
        stage3_7_summary_path=stage3_7_summary,
    )
    console.print(
        {
            "output_name": "stage3_8_intraday_session_integrity",
            "files_created": [
                str(result.summary_json_path),
                str(result.summary_markdown_path),
                str(result.incomplete_sessions_csv_path),
                str(result.session_bar_counts_csv_path),
                str(result.position_policy_actions_csv_path),
            ],
            "report_count_analyzed": result.report_count_analyzed,
            "incomplete_session_count_by_bucket": result.incomplete_session_count_by_bucket,
            "symbols_with_most_incomplete_sessions": (
                result.symbols_with_most_incomplete_sessions[:5]
            ),
            "position_policy_action_summary": result.position_policy_action_summary,
            "intraday_classification_anatomy": result.intraday_classification_anatomy,
            "stage_passed": result.stage_passed,
            "recommended_next_step": result.recommended_next_step,
        }
    )


@research_app.command("intraday-feature-audit")
def research_intraday_feature_audit(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    universe: Annotated[
        Path, typer.Option("--universe", exists=True, file_okay=True)
    ] = Path("data/universes/research_ready/us_liquid_25_5m_intraday.json"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = Path("data/reports/research/stage4_1_intraday_feature_audit"),
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    timeframe: Annotated[str, typer.Option("--timeframe")] = "5m",
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = "XNYS",
) -> None:
    """Build Stage 4.1 research-only intraday feature diagnostics from local data."""

    from stocker_research.intraday_features import build_intraday_feature_audit

    result = build_intraday_feature_audit(
        data_dir=data_dir,
        universe_path=universe,
        output_dir=output_dir,
        source=source,
        instrument_type=instrument_type,
        timeframe=timeframe,
        market_calendar=market_calendar,
    )
    console.print(
        {
            "output_name": "stage4_1_intraday_feature_audit",
            "files_created": [
                str(result.summary_json_path),
                str(result.summary_markdown_path),
                str(result.feature_availability_csv_path),
                str(result.session_feature_quality_csv_path),
                str(result.feature_null_rates_csv_path),
            ],
            "symbol_count": result.symbol_count,
            "feature_availability_summary": result.feature_availability_summary,
            "null_rate_summary": result.null_rate_summary,
            "session_warning_summary": result.session_warning_summary,
            "stage_passed": result.stage_passed,
        }
    )


@server_app.command("dry-run")
def server_dry_run(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/server.example.yaml"),
) -> None:
    """Load server config without connecting to a broker."""

    loaded = load_server_config(config)
    console.print(
        {
            "mode": loaded.server.mode,
            "broker": loaded.server.broker.provider,
            "trading_enabled": loaded.risk.trading_enabled,
        }
    )


if __name__ == "__main__":
    app()
