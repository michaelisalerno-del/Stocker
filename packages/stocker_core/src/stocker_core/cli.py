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


def _parse_symbol_inputs(
    *,
    symbol: list[str] | None = None,
    symbols: str | None = None,
) -> list[str]:
    requested: list[str] = []
    for item in symbol or []:
        requested.extend(part.strip() for part in item.split(",") if part.strip())
    if symbols:
        requested.extend(part.strip() for part in symbols.split(",") if part.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for requested_symbol in requested:
        normalized = requested_symbol.upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


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


@research_app.command("behavioral-state-similarity")
def research_behavioral_state_similarity(
    qualified_universe: Annotated[
        Path | None,
        typer.Option("--qualified-universe", exists=True, file_okay=True, dir_okay=False),
    ] = None,
    symbol: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    symbols: Annotated[str | None, typer.Option("--symbols")] = None,
    max_symbols: Annotated[int | None, typer.Option("--max-symbols", min=1)] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    timeframe: Annotated[str, typer.Option("--timeframe")] = "5m",
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = "XNYS",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/behavioral_state_similarity_v2"),
    event_mode: Annotated[str, typer.Option("--event-mode")] = "state_entry_non_overlapping",
    permutation_count: Annotated[int, typer.Option("--permutation-count", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_independent_events: Annotated[int, typer.Option("--min-independent-events", min=1)] = 30,
    run_template_overlay: Annotated[
        bool,
        typer.Option("--run-template-overlay/--no-run-template-overlay"),
    ] = False,
    template: Annotated[str, typer.Option("--template")] = "opening_range_breakout",
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
) -> None:
    """Run the research-only behavioral state/personality similarity lab."""

    from stocker_research.behavioral_state_similarity import (
        BehavioralStateConfig,
        _load_qualified_symbols,
        run_behavioral_state_similarity_lab,
    )

    loaded = _load_research_cli_config(config)
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    requested_symbols = _parse_symbol_inputs(symbol=symbol, symbols=symbols)
    if qualified_universe is not None:
        requested_symbols.extend(_load_qualified_symbols(qualified_universe))
    if not requested_symbols:
        raise typer.BadParameter("Supply --qualified-universe or one or more --symbol values.")

    deduped_symbols: list[str] = []
    seen_symbols: set[str] = set()
    for requested_symbol in requested_symbols:
        if requested_symbol in seen_symbols:
            continue
        seen_symbols.add(requested_symbol)
        deduped_symbols.append(requested_symbol)
    if max_symbols is not None:
        deduped_symbols = deduped_symbols[:max_symbols]

    result = run_behavioral_state_similarity_lab(
        data_dir=resolved_data_dir,
        symbols=deduped_symbols,
        source=source,
        instrument_type=instrument_type,
        timeframe=timeframe,
        market_calendar=market_calendar,
        output_dir=output_dir,
        config=BehavioralStateConfig(
            timeframe=timeframe,
            market_calendar=market_calendar,
            event_mode=event_mode,
            permutation_count=permutation_count,
            random_seed=random_seed,
            min_independent_events_per_state_horizon=min_independent_events,
            run_template_overlay=run_template_overlay,
            template=template,
        ),
    )
    console.print(
        {
            "output_name": "behavioral_state_similarity",
            "run_id": result.run_id,
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "event_csv_path": str(result.event_csv_path),
            "state_summary_csv_path": str(result.state_summary_csv_path),
            "match_summary_csv_path": str(result.match_summary_csv_path),
            "symbols_requested": result.symbols_requested,
            "symbols_completed": result.symbols_completed,
            "symbols_failed": result.symbols_failed,
            "state_counts": result.state_counts,
            "pipeline_passed": result.pipeline_passed,
            "label_similarity_supported": result.label_similarity_supported,
            "fingerprint_similarity_supported": result.fingerprint_similarity_supported,
            "state_similarity_supported": result.state_similarity_supported,
            "oos_similarity_supported": result.oos_similarity_supported,
            "template_overlay_supported": result.template_overlay_supported,
            "decision": result.decision,
            "decision_reasons": result.decision_reasons,
        }
    )


@research_app.command("state-event-detector")
def research_state_event_detector(
    symbol: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    symbols: Annotated[str | None, typer.Option("--symbols")] = None,
    max_symbols: Annotated[int | None, typer.Option("--max-symbols", min=1)] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    timeframe: Annotated[str, typer.Option("--timeframe")] = "5m",
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = "XNYS",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/state_event_detector_v0"),
    event_mode: Annotated[str, typer.Option("--event-mode")] = "state_entry_non_overlapping",
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_events_for_similarity: Annotated[
        int,
        typer.Option("--min-events-for-similarity", min=1),
    ] = 30,
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
) -> None:
    """Run the focused research-only state event detector v0."""

    from stocker_research.state_event_detector_v0 import (
        StateEventDetectorConfig,
        run_state_event_detector_lab,
    )

    loaded = _load_research_cli_config(config)
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    requested_symbols = _parse_symbol_inputs(symbol=symbol, symbols=symbols)
    if not requested_symbols:
        raise typer.BadParameter("Supply one or more symbols with --symbols or --symbol.")
    if max_symbols is not None:
        requested_symbols = requested_symbols[:max_symbols]

    result = run_state_event_detector_lab(
        data_dir=resolved_data_dir,
        symbols=requested_symbols,
        source=source,
        instrument_type=instrument_type,
        timeframe=timeframe,
        market_calendar=market_calendar,
        output_dir=output_dir,
        config=StateEventDetectorConfig(
            timeframe=timeframe,
            market_calendar=market_calendar,
            event_mode=event_mode,
            random_seed=random_seed,
            min_events_for_similarity=min_events_for_similarity,
        ),
    )
    console.print(
        {
            "output_name": "state_event_detector_v0",
            "run_id": result.run_id,
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "event_rows_csv_path": str(result.event_rows_csv_path),
            "manual_state_audit_csv_path": str(result.manual_state_audit_csv_path),
            "event_state_summary_csv_path": str(result.event_state_summary_csv_path),
            "same_event_cross_symbol_similarity_csv_path": str(
                result.same_event_cross_symbol_similarity_csv_path
            ),
            "random_baseline_csv_path": str(result.random_baseline_csv_path),
            "oos_event_response_csv_path": str(result.oos_event_response_csv_path),
            "concentration_warnings_csv_path": str(result.concentration_warnings_csv_path),
            "decision_json_path": str(result.decision_json_path),
            "symbols_requested": result.symbols_requested,
            "symbols_completed": result.symbols_completed,
            "symbols_failed": result.symbols_failed,
            "total_event_rows": result.total_event_rows,
            "manual_audit_status": "manual_reproduced"
            if result.manual_audit_passed
            else "manual_reproduction_failed",
            "decision": result.decision,
        }
    )


@research_app.command("frozen-template-technique")
def research_frozen_template_technique(
    symbol: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    symbols: Annotated[str | None, typer.Option("--symbols")] = None,
    from_date: Annotated[str, typer.Option("--from")] = "",
    to_date: Annotated[str, typer.Option("--to")] = "",
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/frozen_template_technique_v0"),
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Research config to load.")
    ] = DEFAULT_RESEARCH_CONFIG,
    timeframe: Annotated[str, typer.Option("--timeframe")] = "5m",
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    instrument_type: Annotated[str, typer.Option("--instrument-type")] = "stock",
    currency: Annotated[str | None, typer.Option("--currency")] = None,
    market_calendar: Annotated[str | None, typer.Option("--market-calendar")] = "XNYS",
    download_eodhd: Annotated[
        bool,
        typer.Option("--download-eodhd/--no-download-eodhd"),
    ] = False,
    merge: Annotated[bool, typer.Option("--merge/--no-merge")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    audit: Annotated[bool, typer.Option("--audit/--no-audit")] = True,
    qa: Annotated[bool, typer.Option("--qa/--no-qa")] = True,
    enable_disabled_vendor: Annotated[
        bool,
        typer.Option(
            "--enable-disabled-vendor",
            help="Allow EODHD download even when data_vendors.eodhd.enabled is false.",
        ),
    ] = False,
    event_mode: Annotated[str, typer.Option("--event-mode")] = "state_entry_non_overlapping",
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_events_for_similarity: Annotated[
        int,
        typer.Option("--min-events-for-similarity", min=1),
    ] = 1,
    template_universe_profile: Annotated[
        str,
        typer.Option("--template-universe-profile"),
    ] = "liquid_midcap",
) -> None:
    """Run the local packaged frozen-template research technique."""

    from stocker_research.frozen_template_technique_v0 import (
        FrozenTemplateTechniqueConfig,
        run_frozen_template_technique_v0,
    )

    requested_symbols = _parse_symbol_inputs(symbol=symbol, symbols=symbols)
    if not requested_symbols:
        raise typer.BadParameter("Supply one or more symbols with --symbols or --symbol.")
    if not from_date or not to_date:
        raise typer.BadParameter("Supply --from and --to dates for the technique run.")
    _check_storage_mode(overwrite, merge)

    loaded = _load_research_cli_config(config)
    eodhd_config = loaded.data_vendors.eodhd
    resolved_data_dir = _resolve_data_dir(loaded, data_dir)
    resolved_currency = _resolve_currency(loaded, currency)
    if download_eodhd:
        _require_eodhd_enabled(
            eodhd_config,
            dry_run=False,
            enable_disabled_vendor=enable_disabled_vendor,
            config_path=config,
        )

    result = run_frozen_template_technique_v0(
        data_dir=resolved_data_dir,
        symbols=requested_symbols,
        output_dir=output_dir,
        eodhd_config=eodhd_config,
        config=FrozenTemplateTechniqueConfig(
            from_date=from_date,
            to_date=to_date,
            timeframe=timeframe,
            source=source,
            instrument_type=instrument_type,
            currency=resolved_currency,
            market_calendar=market_calendar,
            download_eodhd=download_eodhd,
            merge=merge,
            overwrite=overwrite,
            audit=audit,
            qa=qa,
            event_mode=event_mode,
            random_seed=random_seed,
            min_events_for_similarity=min_events_for_similarity,
            template_universe_profile=template_universe_profile,
        ),
    )
    console.print(
        {
            "output_name": "frozen_template_technique_v0",
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "fetch_report_csv_path": str(result.fetch_report_csv_path),
            "bar_cleaner_report_csv_path": str(result.bar_cleaner_report_csv_path),
            "state_event_report_dir": str(result.state_event_report_dir),
            "event_rows_csv_path": str(result.event_rows_csv_path),
            "template_transfer_report_dir": str(result.template_transfer_report_dir),
            "template_transfer_summary_json_path": str(
                result.template_transfer_summary_json_path
            ),
            "decision_json_path": str(result.decision_json_path),
            "decision": result.decision,
        }
    )


@research_app.command("event-failure-cutter")
def research_event_failure_cutter(
    input_dir: Annotated[Path | None, typer.Option("--input-dir")] = None,
    input_base_dir: Annotated[
        Path,
        typer.Option("--input-base-dir"),
    ] = Path("data/reports/research/state_event_detector_v0"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/event_failure_cutter_v0"),
    horizons: Annotated[str, typer.Option("--horizons")] = "6,9,12,24",
    min_train_events: Annotated[int, typer.Option("--min-train-events", min=1)] = 30,
    min_test_events: Annotated[int, typer.Option("--min-test-events", min=1)] = 20,
    min_retained_events: Annotated[int, typer.Option("--min-retained-events", min=1)] = 10,
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 50,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    max_candidates_per_state_horizon: Annotated[
        int,
        typer.Option("--max-candidates-per-state-horizon", min=1),
    ] = 24,
    top_single_filters_for_pairs: Annotated[
        int,
        typer.Option("--top-single-filters-for-pairs", min=1),
    ] = 5,
) -> None:
    """Run the research-only event failure cutter v0."""

    from stocker_research.event_failure_cutter_v0 import (
        EventFailureCutterConfig,
        run_event_failure_cutter_lab,
    )

    parsed_horizons = tuple(
        int(part.strip()) for part in horizons.split(",") if part.strip()
    )
    if not parsed_horizons:
        raise typer.BadParameter("Supply at least one horizon with --horizons.")
    result = run_event_failure_cutter_lab(
        input_dir=input_dir,
        input_base_dir=input_base_dir,
        output_dir=output_dir,
        config=EventFailureCutterConfig(
            horizons=parsed_horizons,
            min_train_events=min_train_events,
            min_test_events=min_test_events,
            min_retained_events=min_retained_events,
            random_iterations=random_iterations,
            random_seed=random_seed,
            max_candidates_per_state_horizon=max_candidates_per_state_horizon,
            top_single_filters_for_pairs=top_single_filters_for_pairs,
        ),
    )
    console.print(
        {
            "output_name": "event_failure_cutter_v0",
            "run_id": result.run_id,
            "input_dir": str(result.input_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "filter_oos_results_csv_path": str(result.filter_oos_results_csv_path),
            "random_filter_baseline_csv_path": str(result.random_filter_baseline_csv_path),
            "blocker_quality_summary_csv_path": str(
                result.blocker_quality_summary_csv_path
            ),
            "decision": result.decision,
            "best_filter_count": result.best_filter_count,
        }
    )


@research_app.command("state-event-directional-interpretation")
def research_state_event_directional_interpretation(
    input_dir: Annotated[Path | None, typer.Option("--input-dir")] = None,
    input_base_dir: Annotated[
        Path,
        typer.Option("--input-base-dir"),
    ] = Path("data/reports/research/state_event_detector_v0"),
    horizons: Annotated[str, typer.Option("--horizons")] = "6,9,12,24",
    min_events: Annotated[int, typer.Option("--min-events", min=1)] = 30,
    min_symbols: Annotated[int, typer.Option("--min-symbols", min=1)] = 3,
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Update a state-event-detector report with role-aware directional scoring."""

    from stocker_research.state_directional_interpretation_v0 import (
        DirectionalInterpretationConfig,
        run_state_directional_interpretation_report,
    )

    parsed_horizons = tuple(
        int(part.strip()) for part in horizons.split(",") if part.strip()
    )
    if not parsed_horizons:
        raise typer.BadParameter("Supply at least one horizon with --horizons.")
    result = run_state_directional_interpretation_report(
        input_dir=input_dir,
        input_base_dir=input_base_dir,
        config=DirectionalInterpretationConfig(
            horizons=parsed_horizons,
            min_events=min_events,
            min_symbols=min_symbols,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "state_directional_interpretation_v0",
            "input_dir": str(result.input_dir),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "directional_state_summary_csv_path": str(
                result.directional_state_summary_csv_path
            ),
            "blocker_quality_summary_csv_path": str(result.blocker_quality_summary_csv_path),
            "short_candidate_summary_csv_path": str(result.short_candidate_summary_csv_path),
            "no_trade_quality_summary_csv_path": str(result.no_trade_quality_summary_csv_path),
            "oos_directional_state_response_csv_path": str(
                result.oos_directional_state_response_csv_path
            ),
            "decision": result.decision,
            "state_decision_count": result.state_decision_count,
        }
    )


@research_app.command("role-aware-event-cutter")
def research_role_aware_event_cutter(
    input_dir: Annotated[Path | None, typer.Option("--input-dir")] = None,
    input_base_dir: Annotated[
        Path,
        typer.Option("--input-base-dir"),
    ] = Path("data/reports/research/state_event_detector_v0"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/role_aware_event_cutter_v0"),
    horizons: Annotated[str, typer.Option("--horizons")] = "6,9,12,24",
    min_train_events: Annotated[int, typer.Option("--min-train-events", min=1)] = 30,
    min_test_events: Annotated[int, typer.Option("--min-test-events", min=1)] = 20,
    min_retained_events: Annotated[int, typer.Option("--min-retained-events", min=1)] = 10,
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 50,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    max_candidates_per_state_horizon: Annotated[
        int,
        typer.Option("--max-candidates-per-state-horizon", min=1),
    ] = 24,
) -> None:
    """Run the research-only role-aware event cutter v0."""

    from stocker_research.role_aware_event_cutter_v0 import (
        RoleAwareEventCutterConfig,
        run_role_aware_event_cutter_lab,
    )

    parsed_horizons = tuple(
        int(part.strip()) for part in horizons.split(",") if part.strip()
    )
    if not parsed_horizons:
        raise typer.BadParameter("Supply at least one horizon with --horizons.")
    result = run_role_aware_event_cutter_lab(
        input_dir=input_dir,
        input_base_dir=input_base_dir,
        output_dir=output_dir,
        config=RoleAwareEventCutterConfig(
            horizons=parsed_horizons,
            min_train_events=min_train_events,
            min_test_events=min_test_events,
            min_retained_events=min_retained_events,
            random_iterations=random_iterations,
            random_seed=random_seed,
            max_candidates_per_state_horizon=max_candidates_per_state_horizon,
        ),
    )
    console.print(
        {
            "output_name": "role_aware_event_cutter_v0",
            "run_id": result.run_id,
            "input_dir": str(result.input_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "filter_oos_results_csv_path": str(result.filter_oos_results_csv_path),
            "random_role_baselines_csv_path": str(result.random_role_baselines_csv_path),
            "selected_filters_csv_path": str(result.selected_filters_csv_path),
            "rejected_filters_csv_path": str(result.rejected_filters_csv_path),
            "decision": result.decision,
            "selected_filter_count": result.selected_filter_count,
        }
    )


@research_app.command("personality-discovery")
def research_personality_discovery(
    input_dir: Annotated[Path | None, typer.Option("--input-dir")] = None,
    input_base_dir: Annotated[
        Path,
        typer.Option("--input-base-dir"),
    ] = Path("data/reports/research/state_event_detector_v0"),
    spec_dir: Annotated[
        Path,
        typer.Option("--spec-dir"),
    ] = Path("configs/research/personalities/v0"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_discovery_v0"),
    horizons: Annotated[str, typer.Option("--horizons")] = "6,9,12,24",
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    max_filters_per_personality_horizon: Annotated[
        int,
        typer.Option("--max-filters-per-personality-horizon", min=1),
    ] = 80,
    max_pair_seed_filters: Annotated[
        int,
        typer.Option("--max-pair-seed-filters", min=1),
    ] = 12,
    default_min_train_events: Annotated[
        int,
        typer.Option("--default-min-train-events", min=1),
    ] = 30,
    default_min_test_events: Annotated[
        int,
        typer.Option("--default-min-test-events", min=1),
    ] = 12,
    default_min_retained_events: Annotated[
        int,
        typer.Option("--default-min-retained-events", min=1),
    ] = 8,
    default_min_symbols: Annotated[
        int,
        typer.Option("--default-min-symbols", min=1),
    ] = 3,
) -> None:
    """Run YAML-driven research-only personality discovery over event rows."""

    from stocker_research.personality_discovery_v0 import (
        PersonalityDiscoveryConfig,
        run_personality_discovery_lab,
    )

    parsed_horizons = tuple(
        int(part.strip()) for part in horizons.split(",") if part.strip()
    )
    if not parsed_horizons:
        raise typer.BadParameter("Supply at least one horizon with --horizons.")
    result = run_personality_discovery_lab(
        input_dir=input_dir,
        input_base_dir=input_base_dir,
        spec_dir=spec_dir,
        output_dir=output_dir,
        config=PersonalityDiscoveryConfig(
            horizons=parsed_horizons,
            random_iterations=random_iterations,
            random_seed=random_seed,
            max_filters_per_personality_horizon=max_filters_per_personality_horizon,
            max_pair_seed_filters=max_pair_seed_filters,
            default_min_train_events=default_min_train_events,
            default_min_test_events=default_min_test_events,
            default_min_retained_events=default_min_retained_events,
            default_min_symbols=default_min_symbols,
        ),
    )
    console.print(
        {
            "output_name": "personality_discovery_v0",
            "run_id": result.run_id,
            "input_dir": str(result.input_dir),
            "spec_dir": str(result.spec_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "candidate_rules_csv_path": str(result.candidate_rules_csv_path),
            "selected_rules_csv_path": str(result.selected_rules_csv_path),
            "passed_rules_csv_path": str(result.passed_rules_csv_path),
            "random_baseline_csv_path": str(result.random_baseline_csv_path),
            "decision": result.decision,
            "passed_rule_count": result.passed_rule_count,
        }
    )


@research_app.command("personality-rulebook-validation")
def research_personality_rulebook_validation(
    source_personality_dir: Annotated[
        Path,
        typer.Option("--source-personality-dir"),
    ],
    validation_event_dir: Annotated[
        Path,
        typer.Option("--validation-event-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_rulebook_validation_v0"),
    top_per_personality: Annotated[
        int,
        typer.Option("--top-per-personality", min=1),
    ] = 8,
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_validation_events: Annotated[
        int,
        typer.Option("--min-validation-events", min=1),
    ] = 12,
    min_validation_symbols: Annotated[
        int,
        typer.Option("--min-validation-symbols", min=1),
    ] = 5,
    min_validation_months: Annotated[
        int,
        typer.Option("--min-validation-months", min=1),
    ] = 3,
) -> None:
    """Validate a compact personality rulebook on holdout event symbols."""

    from stocker_research.personality_rulebook_validation_v0 import (
        RulebookValidationConfig,
        run_personality_rulebook_validation,
    )

    result = run_personality_rulebook_validation(
        source_personality_dir=source_personality_dir,
        validation_event_dir=validation_event_dir,
        output_dir=output_dir,
        config=RulebookValidationConfig(
            top_per_personality=top_per_personality,
            random_iterations=random_iterations,
            random_seed=random_seed,
            min_validation_events=min_validation_events,
            min_validation_symbols=min_validation_symbols,
            min_validation_months=min_validation_months,
        ),
    )
    console.print(
        {
            "output_name": "personality_rulebook_validation_v0",
            "run_id": result.run_id,
            "source_personality_dir": str(result.source_personality_dir),
            "validation_event_dir": str(result.validation_event_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "collapsed_rulebook_csv_path": str(result.collapsed_rulebook_csv_path),
            "validation_results_csv_path": str(result.validation_results_csv_path),
            "passed_rules_csv_path": str(result.passed_rules_csv_path),
            "decision": result.decision,
            "passed_rule_count": result.passed_rule_count,
        }
    )


@research_app.command("personality-rulebook")
def research_personality_rulebook(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    rulebook_path: Annotated[
        Path,
        typer.Option("--rulebook-path"),
    ] = Path("configs/research/personality_rulebook_v0/rules.yaml"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_rulebook_v0"),
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_events: Annotated[int, typer.Option("--min-events", min=1)] = 12,
    min_symbols: Annotated[int, typer.Option("--min-symbols", min=1)] = 5,
    min_months: Annotated[int, typer.Option("--min-months", min=1)] = 3,
) -> None:
    """Apply the fixed research-only personality rulebook to event rows."""

    from stocker_research.personality_rulebook_v0 import (
        PersonalityRulebookConfig,
        run_personality_rulebook_lab,
    )

    result = run_personality_rulebook_lab(
        input_event_dir=input_event_dir,
        rulebook_path=rulebook_path,
        output_dir=output_dir,
        config=PersonalityRulebookConfig(
            random_iterations=random_iterations,
            random_seed=random_seed,
            min_events=min_events,
            min_symbols=min_symbols,
            min_months=min_months,
        ),
    )
    console.print(
        {
            "output_name": "personality_rulebook_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "rulebook_path": str(result.rulebook_path),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "rule_summary_csv_path": str(result.rule_summary_csv_path),
            "personality_summary_csv_path": str(result.personality_summary_csv_path),
            "decision": result.decision,
            "passed_rule_count": result.passed_rule_count,
        }
    )


@research_app.command("personality-template")
def research_personality_template(
    input_event_dir: Annotated[
        Path | None,
        typer.Option("--input-event-dir"),
    ] = None,
    input_base_dir: Annotated[
        Path,
        typer.Option("--input-base-dir"),
    ] = Path("data/reports/research/state_event_detector_v0"),
    template_path: Annotated[
        Path,
        typer.Option("--template-path"),
    ] = Path("configs/research/personality_template_v0/templates.yaml"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_template_v0"),
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    max_candidates_per_template: Annotated[
        int,
        typer.Option("--max-candidates-per-template", min=1),
    ] = 96,
    max_selected_per_template: Annotated[
        int,
        typer.Option("--max-selected-per-template", min=1),
    ] = 12,
    stop_loss_bps: Annotated[
        str,
        typer.Option("--stop-loss-bps"),
    ] = "25,50,100",
) -> None:
    """Discover caveats for fixed research-only personality templates."""

    from stocker_research.personality_template_v0 import (
        PersonalityTemplateConfig,
        run_personality_template_lab,
    )

    parsed_stop_loss_bps = tuple(
        float(part.strip()) for part in stop_loss_bps.split(",") if part.strip()
    )
    if not parsed_stop_loss_bps:
        raise typer.BadParameter("Supply at least one stop distance with --stop-loss-bps.")
    result = run_personality_template_lab(
        input_event_dir=input_event_dir,
        input_base_dir=input_base_dir,
        template_path=template_path,
        output_dir=output_dir,
        config=PersonalityTemplateConfig(
            random_iterations=random_iterations,
            random_seed=random_seed,
            max_candidates_per_template=max_candidates_per_template,
            max_selected_per_template=max_selected_per_template,
            stop_loss_bps=parsed_stop_loss_bps,
        ),
    )
    console.print(
        {
            "output_name": "personality_template_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "template_path": str(result.template_path),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "selected_rules_csv_path": str(result.selected_rules_csv_path),
            "candidate_rules_csv_path": str(result.candidate_rules_csv_path),
            "decision": result.decision,
            "selected_rule_count": result.selected_rule_count,
        }
    )


@research_app.command("personality-stop-validation")
def research_personality_stop_validation(
    input_template_dir: Annotated[
        Path | None,
        typer.Option("--input-template-dir"),
    ] = None,
    input_base_dir: Annotated[
        Path,
        typer.Option("--input-base-dir"),
    ] = Path("data/reports/research/personality_template_v0"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_stop_validation_v0"),
    stop_loss_bps: Annotated[
        str,
        typer.Option("--stop-loss-bps"),
    ] = "25,50,75,100",
    target_r_multiples: Annotated[
        str,
        typer.Option("--target-r-multiples"),
    ] = "1,1.5,2",
    cost_bps: Annotated[
        str,
        typer.Option("--cost-bps"),
    ] = "0,5,10",
    train_fraction: Annotated[
        float,
        typer.Option("--train-fraction", min=0.1, max=0.9),
    ] = 0.60,
    max_candidate_book_rows: Annotated[
        int,
        typer.Option("--max-candidate-book-rows", min=1),
    ] = 12,
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Validate stop distances and R multiples for selected personality templates."""

    from stocker_research.personality_stop_validation_v0 import (
        PersonalityStopValidationConfig,
        run_personality_stop_validation_lab,
    )

    parsed_stop_loss_bps = tuple(
        float(part.strip()) for part in stop_loss_bps.split(",") if part.strip()
    )
    parsed_target_r = tuple(
        float(part.strip()) for part in target_r_multiples.split(",") if part.strip()
    )
    parsed_cost_bps = tuple(float(part.strip()) for part in cost_bps.split(",") if part.strip())
    if not parsed_stop_loss_bps:
        raise typer.BadParameter("Supply at least one stop distance with --stop-loss-bps.")
    if not parsed_target_r:
        raise typer.BadParameter("Supply at least one target multiple with --target-r-multiples.")
    if not parsed_cost_bps:
        raise typer.BadParameter("Supply at least one cost value with --cost-bps.")
    result = run_personality_stop_validation_lab(
        input_template_dir=input_template_dir,
        input_base_dir=input_base_dir,
        output_dir=output_dir,
        config=PersonalityStopValidationConfig(
            stop_loss_bps=parsed_stop_loss_bps,
            target_r_multiples=parsed_target_r,
            cost_bps=parsed_cost_bps,
            train_fraction=train_fraction,
            max_candidate_book_rows=max_candidate_book_rows,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "personality_stop_validation_v0",
            "run_id": result.run_id,
            "input_template_dir": str(result.input_template_dir),
            "input_event_dir": str(result.input_event_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "stop_model_results_csv_path": str(result.stop_model_results_csv_path),
            "selected_stop_models_csv_path": str(result.selected_stop_models_csv_path),
            "decision": result.decision,
            "selected_stop_model_count": result.selected_stop_model_count,
        }
    )


@research_app.command("personality-live-replay")
def research_personality_live_replay(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    candidate_book: Annotated[
        Path,
        typer.Option("--candidate-book"),
    ],
    template_path: Annotated[
        Path,
        typer.Option("--template-path"),
    ] = Path("configs/research/personality_template_v0/templates.yaml"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_live_replay_v0"),
    replay_start: Annotated[
        str,
        typer.Option("--replay-start"),
    ] = "2026-06-01",
    replay_end: Annotated[
        str,
        typer.Option("--replay-end"),
    ] = "2026-06-30",
    cost_bps: Annotated[
        float,
        typer.Option("--cost-bps"),
    ] = 10.0,
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Replay frozen personality candidates over a historical live-style window."""

    from stocker_research.personality_live_replay_v0 import (
        LiveReplayConfig,
        run_personality_live_replay_lab,
    )

    result = run_personality_live_replay_lab(
        input_event_dir=input_event_dir,
        candidate_book_path=candidate_book,
        template_path=template_path,
        output_dir=output_dir,
        replay_start=replay_start,
        replay_end=replay_end,
        config=LiveReplayConfig(
            cost_bps=cost_bps,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "personality_live_replay_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "candidate_book_path": str(result.candidate_book_path),
            "template_path": str(result.template_path),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "trades_csv_path": str(result.trades_csv_path),
            "decision": result.decision,
            "trade_count": result.trade_count,
        }
    )


@research_app.command("walk-forward-personality-filter-exit")
def research_walk_forward_personality_filter_exit(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    input_combined_regime_dir: Annotated[
        Path,
        typer.Option("--input-combined-regime-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/walk_forward_personality_filter_exit_v0"),
    replay_months: Annotated[
        str,
        typer.Option("--replay-months"),
    ] = "2026-01,2026-02,2026-03,2026-04,2026-05,2026-06",
    filter_features: Annotated[
        str,
        typer.Option("--filter-features"),
    ] = "",
    stop_models: Annotated[
        str,
        typer.Option("--stop-models"),
    ] = (
        "fixed_50bps,fixed_75bps,fixed_100bps,"
        "structure_session_extreme_10bps,structure_recent_extreme_10bps,"
        "structure_opening_range_extreme_10bps"
    ),
    target_r_multiples: Annotated[
        str,
        typer.Option("--target-r-multiples"),
    ] = "1,1.5,2",
    quantiles: Annotated[
        str,
        typer.Option("--quantiles"),
    ] = "0.20,0.35,0.50,0.65,0.80",
    cost_bps: Annotated[
        float,
        typer.Option("--cost-bps"),
    ] = 10.0,
    top_combos_per_personality: Annotated[
        int,
        typer.Option("--top-combos-per-personality", min=1),
    ] = 5,
    max_filter_candidates_per_combo: Annotated[
        int,
        typer.Option("--max-filter-candidates-per-combo", min=1),
    ] = 4,
    max_exit_candidates_per_month: Annotated[
        int,
        typer.Option("--max-exit-candidates-per-month", min=1),
    ] = 48,
    max_selected_per_month: Annotated[
        int,
        typer.Option("--max-selected-per-month", min=1),
    ] = 12,
    min_train_events: Annotated[
        int,
        typer.Option("--min-train-events", min=1),
    ] = 35,
    min_train_symbols: Annotated[
        int,
        typer.Option("--min-train-symbols", min=1),
    ] = 4,
    min_train_months: Annotated[
        int,
        typer.Option("--min-train-months", min=1),
    ] = 4,
    min_total_trades: Annotated[
        int,
        typer.Option("--min-total-trades", min=1),
    ] = 30,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Run prior-only monthly filter rediscovery with conservative exit replay."""

    from stocker_research.walk_forward_personality_filter_exit_v0 import (
        DEFAULT_FILTER_FEATURES,
        WalkForwardPersonalityFilterExitConfig,
        run_walk_forward_personality_filter_exit_lab,
    )

    parsed_months = tuple(part.strip() for part in replay_months.split(",") if part.strip())
    parsed_features = (
        tuple(part.strip() for part in filter_features.split(",") if part.strip())
        if filter_features.strip()
        else DEFAULT_FILTER_FEATURES
    )
    parsed_stop_models = tuple(part.strip() for part in stop_models.split(",") if part.strip())
    parsed_target_r = tuple(
        float(part.strip()) for part in target_r_multiples.split(",") if part.strip()
    )
    parsed_quantiles = tuple(float(part.strip()) for part in quantiles.split(",") if part.strip())
    if not parsed_months:
        raise typer.BadParameter("Supply at least one replay month with --replay-months.")
    if not parsed_features:
        raise typer.BadParameter("Supply at least one feature with --filter-features.")
    if not parsed_stop_models:
        raise typer.BadParameter("Supply at least one stop model with --stop-models.")
    if not parsed_target_r:
        raise typer.BadParameter("Supply at least one target multiple.")
    if not parsed_quantiles:
        raise typer.BadParameter("Supply at least one quantile.")

    result = run_walk_forward_personality_filter_exit_lab(
        input_event_dir=input_event_dir,
        input_combined_regime_dir=input_combined_regime_dir,
        output_dir=output_dir,
        config=WalkForwardPersonalityFilterExitConfig(
            replay_months=parsed_months,
            filter_features=parsed_features,
            stop_models=parsed_stop_models,
            target_r_multiples=parsed_target_r,
            quantiles=parsed_quantiles,
            cost_bps=cost_bps,
            top_combos_per_personality=top_combos_per_personality,
            max_filter_candidates_per_combo=max_filter_candidates_per_combo,
            max_exit_candidates_per_month=max_exit_candidates_per_month,
            max_selected_per_month=max_selected_per_month,
            min_train_events=min_train_events,
            min_train_symbols=min_train_symbols,
            min_train_months=min_train_months,
            min_total_trades=min_total_trades,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "walk_forward_personality_filter_exit_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "input_combined_regime_dir": str(result.input_combined_regime_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "selected_monthly_candidates_csv_path": str(
                result.selected_monthly_candidates_csv_path
            ),
            "trades_csv_path": str(result.trades_csv_path),
            "decision": result.decision,
            "trade_count": result.trade_count,
        }
    )


@research_app.command("walk-forward-selected-filter-exit")
def research_walk_forward_selected_filter_exit(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    input_filter_report_dir: Annotated[
        Path,
        typer.Option("--input-filter-report-dir"),
    ],
    input_blocker_report_dir: Annotated[
        Path | None,
        typer.Option("--input-blocker-report-dir"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/walk_forward_selected_filter_exit_v0"),
    replay_months: Annotated[
        str,
        typer.Option("--replay-months"),
    ] = "2026-01,2026-02,2026-03,2026-04,2026-05,2026-06",
    stop_models: Annotated[
        str,
        typer.Option("--stop-models"),
    ] = (
        "fixed_50bps,fixed_75bps,fixed_100bps,"
        "structure_session_extreme_10bps,structure_recent_extreme_10bps,"
        "structure_opening_range_extreme_10bps"
    ),
    target_r_multiples: Annotated[
        str,
        typer.Option("--target-r-multiples"),
    ] = "1,1.5,2",
    cost_bps: Annotated[
        float,
        typer.Option("--cost-bps"),
    ] = 10.0,
    max_exit_candidates_per_month: Annotated[
        int,
        typer.Option("--max-exit-candidates-per-month", min=1),
    ] = 48,
    max_selected_per_month: Annotated[
        int,
        typer.Option("--max-selected-per-month", min=1),
    ] = 18,
    max_blocker_rules: Annotated[
        int,
        typer.Option("--max-blocker-rules", min=0),
    ] = 12,
    min_train_events: Annotated[
        int,
        typer.Option("--min-train-events", min=1),
    ] = 35,
    min_train_symbols: Annotated[
        int,
        typer.Option("--min-train-symbols", min=1),
    ] = 4,
    min_train_months: Annotated[
        int,
        typer.Option("--min-train-months", min=1),
    ] = 4,
    min_total_trades: Annotated[
        int,
        typer.Option("--min-total-trades", min=1),
    ] = 30,
    max_single_month_share: Annotated[
        float,
        typer.Option("--max-single-month-share", min=0.0, max=1.0),
    ] = 0.50,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Replay frozen selected filters with prior-only monthly exit selection."""

    from stocker_research.walk_forward_personality_filter_exit_v0 import (
        WalkForwardSelectedFilterExitConfig,
        run_walk_forward_selected_filter_exit_lab,
    )

    parsed_months = tuple(part.strip() for part in replay_months.split(",") if part.strip())
    parsed_stop_models = tuple(part.strip() for part in stop_models.split(",") if part.strip())
    parsed_target_r = tuple(
        float(part.strip()) for part in target_r_multiples.split(",") if part.strip()
    )
    if not parsed_months:
        raise typer.BadParameter("Supply at least one replay month with --replay-months.")
    if not parsed_stop_models:
        raise typer.BadParameter("Supply at least one stop model with --stop-models.")
    if not parsed_target_r:
        raise typer.BadParameter("Supply at least one target multiple.")

    result = run_walk_forward_selected_filter_exit_lab(
        input_event_dir=input_event_dir,
        input_filter_report_dir=input_filter_report_dir,
        input_blocker_report_dir=input_blocker_report_dir,
        output_dir=output_dir,
        config=WalkForwardSelectedFilterExitConfig(
            replay_months=parsed_months,
            stop_models=parsed_stop_models,
            target_r_multiples=parsed_target_r,
            cost_bps=cost_bps,
            max_exit_candidates_per_month=max_exit_candidates_per_month,
            max_selected_per_month=max_selected_per_month,
            max_blocker_rules=max_blocker_rules,
            min_train_events=min_train_events,
            min_train_symbols=min_train_symbols,
            min_train_months=min_train_months,
            min_total_trades=min_total_trades,
            max_single_month_share=max_single_month_share,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "walk_forward_selected_filter_exit_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "input_filter_report_dir": str(result.input_filter_report_dir),
            "input_blocker_report_dir": str(result.input_blocker_report_dir)
            if result.input_blocker_report_dir is not None
            else None,
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "selected_monthly_candidates_csv_path": str(
                result.selected_monthly_candidates_csv_path
            ),
            "trades_csv_path": str(result.trades_csv_path),
            "blocked_signals_csv_path": str(result.blocked_signals_csv_path),
            "decision": result.decision,
            "trade_count": result.trade_count,
        }
    )


@research_app.command("bad-trade-sequence-caveat")
def research_bad_trade_sequence_caveat(
    input_selected_report_dir: Annotated[
        Path,
        typer.Option("--input-selected-report-dir"),
    ],
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/bad_trade_sequence_caveat_v0"),
    train_months: Annotated[
        str,
        typer.Option("--train-months"),
    ] = "2026-01,2026-02,2026-03,2026-04",
    test_months: Annotated[
        str,
        typer.Option("--test-months"),
    ] = "2026-05,2026-06",
    numeric_quantiles: Annotated[
        str,
        typer.Option("--numeric-quantiles"),
    ] = "0.20,0.25,0.33,0.50,0.67,0.75,0.80",
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 3000,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_candidate_count: Annotated[
        int,
        typer.Option("--min-candidate-count", min=1),
    ] = 1,
    max_single_symbol_share: Annotated[
        float,
        typer.Option("--max-single-symbol-share", min=0.0, max=1.0),
    ] = 0.50,
    max_single_session_share: Annotated[
        float,
        typer.Option("--max-single-session-share", min=0.0, max=1.0),
    ] = 0.20,
) -> None:
    """Run reusable research-only bad-trade sequence caveat diagnostics."""

    from stocker_research.bad_trade_sequence_caveat_v0 import (
        BadTradeSequenceCaveatConfig,
        run_bad_trade_sequence_caveat_lab,
    )

    parsed_train_months = tuple(part.strip() for part in train_months.split(",") if part.strip())
    parsed_test_months = tuple(part.strip() for part in test_months.split(",") if part.strip())
    parsed_quantiles = tuple(
        float(part.strip()) for part in numeric_quantiles.split(",") if part.strip()
    )
    if not parsed_train_months:
        raise typer.BadParameter("Supply at least one train month with --train-months.")
    if not parsed_test_months:
        raise typer.BadParameter("Supply at least one test month with --test-months.")
    if not parsed_quantiles:
        raise typer.BadParameter("Supply at least one numeric quantile.")

    result = run_bad_trade_sequence_caveat_lab(
        input_selected_report_dir=input_selected_report_dir,
        input_event_dir=input_event_dir,
        output_dir=output_dir,
        config=BadTradeSequenceCaveatConfig(
            train_months=parsed_train_months,
            test_months=parsed_test_months,
            numeric_quantiles=parsed_quantiles,
            random_iterations=random_iterations,
            random_seed=random_seed,
            min_candidate_count=min_candidate_count,
            max_single_symbol_share=max_single_symbol_share,
            max_single_session_share=max_single_session_share,
        ),
    )
    console.print(
        {
            "output_name": "bad_trade_sequence_caveat_v0",
            "run_id": result.run_id,
            "input_selected_report_dir": str(result.input_selected_report_dir),
            "input_event_dir": str(result.input_event_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "sequence_caveat_results_csv_path": str(result.sequence_caveat_results_csv_path),
            "current_personality_caveats_csv_path": str(
                result.current_personality_caveats_csv_path
            ),
            "prior_sequence_caveats_csv_path": str(result.prior_sequence_caveats_csv_path),
            "numeric_threshold_caveats_csv_path": str(result.numeric_threshold_caveats_csv_path),
            "strict_validation_results_csv_path": str(result.strict_validation_results_csv_path),
            "trade_caveat_flags_csv_path": str(result.trade_caveat_flags_csv_path),
            "decision": result.decision,
            "caveat_count": result.caveat_count,
        }
    )


@research_app.command("personality-expression-lab")
def research_personality_expression_lab(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    input_personality_discovery_dir: Annotated[
        Path,
        typer.Option("--input-personality-discovery-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_expression_lab_v0"),
    train_months: Annotated[
        str,
        typer.Option("--train-months"),
    ] = "2026-01,2026-02,2026-03,2026-04",
    test_months: Annotated[
        str,
        typer.Option("--test-months"),
    ] = "2026-05,2026-06",
    allowed_personalities: Annotated[
        str,
        typer.Option("--allowed-personalities"),
    ] = "active_liquidation,impulse_recoil,slow_repair",
    stop_models: Annotated[
        str,
        typer.Option("--stop-models"),
    ] = (
        "fixed_50bps,fixed_75bps,fixed_100bps,"
        "structure_session_extreme_10bps,structure_recent_extreme_10bps,"
        "structure_opening_range_extreme_10bps"
    ),
    target_r_multiples: Annotated[
        str,
        typer.Option("--target-r-multiples"),
    ] = "1,1.5,2",
    cost_bps: Annotated[
        float,
        typer.Option("--cost-bps"),
    ] = 10.0,
    max_rule_candidates_per_personality: Annotated[
        int,
        typer.Option("--max-rule-candidates-per-personality", min=1),
    ] = 80,
    max_expressions_per_personality: Annotated[
        int,
        typer.Option("--max-expressions-per-personality", min=1),
    ] = 1,
    min_train_trades: Annotated[
        int,
        typer.Option("--min-train-trades", min=1),
    ] = 20,
    min_train_months: Annotated[
        int,
        typer.Option("--min-train-months", min=1),
    ] = 3,
    min_train_total_net_r: Annotated[
        float,
        typer.Option("--min-train-total-net-r"),
    ] = 0.0,
    min_train_win_rate: Annotated[
        float,
        typer.Option("--min-train-win-rate", min=0.0, max=1.0),
    ] = 0.55,
    min_oos_trades: Annotated[
        int,
        typer.Option("--min-oos-trades", min=1),
    ] = 1,
) -> None:
    """Replay strict personality expressions selected on train months."""

    from stocker_research.personality_expression_lab_v0 import (
        PersonalityExpressionLabConfig,
        run_personality_expression_lab,
    )

    parsed_train_months = tuple(part.strip() for part in train_months.split(",") if part.strip())
    parsed_test_months = tuple(part.strip() for part in test_months.split(",") if part.strip())
    parsed_personalities = tuple(
        part.strip() for part in allowed_personalities.split(",") if part.strip()
    )
    parsed_stop_models = tuple(part.strip() for part in stop_models.split(",") if part.strip())
    parsed_target_r = tuple(
        float(part.strip()) for part in target_r_multiples.split(",") if part.strip()
    )
    if not parsed_train_months:
        raise typer.BadParameter("Supply at least one train month with --train-months.")
    if not parsed_test_months:
        raise typer.BadParameter("Supply at least one test month with --test-months.")
    if not parsed_personalities:
        raise typer.BadParameter("Supply at least one personality with --allowed-personalities.")
    if not parsed_stop_models:
        raise typer.BadParameter("Supply at least one stop model with --stop-models.")
    if not parsed_target_r:
        raise typer.BadParameter("Supply at least one target multiple.")

    result = run_personality_expression_lab(
        input_event_dir=input_event_dir,
        input_personality_discovery_dir=input_personality_discovery_dir,
        output_dir=output_dir,
        config=PersonalityExpressionLabConfig(
            train_months=parsed_train_months,
            test_months=parsed_test_months,
            allowed_personalities=parsed_personalities,
            stop_models=parsed_stop_models,
            target_r_multiples=parsed_target_r,
            cost_bps=cost_bps,
            max_rule_candidates_per_personality=max_rule_candidates_per_personality,
            max_expressions_per_personality=max_expressions_per_personality,
            min_train_trades=min_train_trades,
            min_train_months=min_train_months,
            min_train_total_net_r=min_train_total_net_r,
            min_train_win_rate=min_train_win_rate,
            min_oos_trades=min_oos_trades,
        ),
    )
    console.print(
        {
            "output_name": "personality_expression_lab_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "input_personality_discovery_dir": str(result.input_personality_discovery_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "expression_candidate_sweep_csv_path": str(
                result.expression_candidate_sweep_csv_path
            ),
            "selected_expressions_csv_path": str(result.selected_expressions_csv_path),
            "test_trades_csv_path": str(result.test_trades_csv_path),
            "decision": result.decision,
            "test_trade_count": result.test_trade_count,
        }
    )


@research_app.command("state-lifecycle-context")
def research_state_lifecycle_context(
    input_expression_report_dir: Annotated[
        Path,
        typer.Option("--input-expression-report-dir"),
    ],
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir"),
    ] = Path("data"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/state_lifecycle_context_lab_v0"),
    lookback_bars: Annotated[
        str,
        typer.Option("--lookback-bars"),
    ] = "6,12,24,36",
    source: Annotated[str, typer.Option("--source")] = "eodhd",
    instrument_type: Annotated[
        str,
        typer.Option("--instrument-type"),
    ] = "stock",
    timeframe: Annotated[str, typer.Option("--timeframe")] = "5m",
    market_calendar: Annotated[
        str,
        typer.Option("--market-calendar"),
    ] = "XNYS",
    min_train_count: Annotated[
        int,
        typer.Option("--min-train-count", min=1),
    ] = 8,
    min_oos_count: Annotated[
        int,
        typer.Option("--min-oos-count", min=1),
    ] = 5,
    min_train_mean_lift_r: Annotated[
        float,
        typer.Option("--min-train-mean-lift-r"),
    ] = 0.0,
    min_oos_mean_lift_r: Annotated[
        float,
        typer.Option("--min-oos-mean-lift-r"),
    ] = 0.0,
    max_selected_per_family: Annotated[
        int,
        typer.Option("--max-selected-per-family", min=1),
    ] = 40,
) -> None:
    """Test prior regime mix and sparse event clusters for personality trades."""

    from stocker_research.state_lifecycle_context_lab_v0 import (
        StateLifecycleContextConfig,
        run_state_lifecycle_context_lab,
    )

    parsed_lookbacks = tuple(
        int(part.strip()) for part in lookback_bars.split(",") if part.strip()
    )
    if not parsed_lookbacks:
        raise typer.BadParameter("Supply at least one lookback with --lookback-bars.")
    parsed_calendar = market_calendar.strip()

    result = run_state_lifecycle_context_lab(
        input_expression_report_dir=input_expression_report_dir,
        input_event_dir=input_event_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        config=StateLifecycleContextConfig(
            lookback_bars=parsed_lookbacks,
            source=source,
            instrument_type=instrument_type,
            timeframe=timeframe,
            market_calendar=parsed_calendar if parsed_calendar.lower() != "none" else None,
            min_train_count=min_train_count,
            min_oos_count=min_oos_count,
            min_train_mean_lift_r=min_train_mean_lift_r,
            min_oos_mean_lift_r=min_oos_mean_lift_r,
            max_selected_per_family=max_selected_per_family,
        ),
    )
    console.print(
        {
            "output_name": "state_lifecycle_context_lab_v0",
            "run_id": result.run_id,
            "input_expression_report_dir": str(result.input_expression_report_dir),
            "input_event_dir": str(result.input_event_dir),
            "data_dir": str(result.data_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "trade_context_features_csv_path": str(result.trade_context_features_csv_path),
            "base_summary_csv_path": str(result.base_summary_csv_path),
            "prior_regime_numeric_scan_csv_path": str(
                result.prior_regime_numeric_scan_csv_path
            ),
            "prior_regime_categorical_scan_csv_path": str(
                result.prior_regime_categorical_scan_csv_path
            ),
            "prior_event_cluster_scan_csv_path": str(result.prior_event_cluster_scan_csv_path),
            "selected_context_candidates_csv_path": str(
                result.selected_context_candidates_csv_path
            ),
            "decision": result.decision,
            "selected_candidate_count": result.selected_candidate_count,
        }
    )


@research_app.command("conditional-context-caveat")
def research_conditional_context_caveat(
    input_context_report_dir: Annotated[
        Path,
        typer.Option("--input-context-report-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/conditional_context_caveat_v0"),
    train_months: Annotated[
        str,
        typer.Option("--train-months"),
    ] = "2026-01,2026-02,2026-03,2026-04",
    test_months: Annotated[
        str,
        typer.Option("--test-months"),
    ] = "2026-05,2026-06",
    condition_features: Annotated[
        str,
        typer.Option("--condition-features"),
    ] = "",
    numeric_features: Annotated[
        str,
        typer.Option("--numeric-features"),
    ] = "",
    numeric_operators: Annotated[
        str,
        typer.Option("--numeric-operators"),
    ] = "<=,>=",
    numeric_quantiles: Annotated[
        str,
        typer.Option("--numeric-quantiles"),
    ] = "0.20,0.33,0.50,0.67,0.80",
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 3000,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_train_condition_count: Annotated[
        int,
        typer.Option("--min-train-condition-count", min=1),
    ] = 8,
    min_train_flagged_count: Annotated[
        int,
        typer.Option("--min-train-flagged-count", min=1),
    ] = 3,
    min_oos_flagged_count: Annotated[
        int,
        typer.Option("--min-oos-flagged-count", min=1),
    ] = 1,
    max_condition_values: Annotated[
        int,
        typer.Option("--max-condition-values", min=1),
    ] = 30,
    max_selected_rules: Annotated[
        int,
        typer.Option("--max-selected-rules", min=1),
    ] = 40,
    max_flag_rules: Annotated[
        int,
        typer.Option("--max-flag-rules", min=1),
    ] = 40,
    max_single_symbol_share: Annotated[
        float,
        typer.Option("--max-single-symbol-share", min=0.0, max=1.0),
    ] = 0.50,
    max_single_session_share: Annotated[
        float,
        typer.Option("--max-single-session-share", min=0.0, max=1.0),
    ] = 0.20,
) -> None:
    """Scan IF context == value THEN numeric veto caveats."""

    from stocker_research.conditional_context_caveat_v0 import (
        ConditionalContextCaveatConfig,
        run_conditional_context_caveat_lab,
    )

    parsed_train_months = tuple(part.strip() for part in train_months.split(",") if part.strip())
    parsed_test_months = tuple(part.strip() for part in test_months.split(",") if part.strip())
    parsed_conditions = tuple(
        part.strip() for part in condition_features.split(",") if part.strip()
    )
    parsed_numeric_features = tuple(
        part.strip() for part in numeric_features.split(",") if part.strip()
    )
    parsed_numeric_operators = tuple(
        part.strip() for part in numeric_operators.split(",") if part.strip()
    )
    parsed_quantiles = tuple(
        float(part.strip()) for part in numeric_quantiles.split(",") if part.strip()
    )
    if not parsed_train_months:
        raise typer.BadParameter("Supply at least one train month with --train-months.")
    if not parsed_test_months:
        raise typer.BadParameter("Supply at least one test month with --test-months.")
    if not parsed_numeric_operators:
        raise typer.BadParameter("Supply at least one numeric operator.")
    invalid_operators = sorted(set(parsed_numeric_operators) - {"<=", ">="})
    if invalid_operators:
        raise typer.BadParameter(f"Unsupported numeric operators: {invalid_operators}")
    if not parsed_quantiles:
        raise typer.BadParameter("Supply at least one numeric quantile.")

    result = run_conditional_context_caveat_lab(
        input_context_report_dir=input_context_report_dir,
        output_dir=output_dir,
        config=ConditionalContextCaveatConfig(
            train_months=parsed_train_months,
            test_months=parsed_test_months,
            condition_features=parsed_conditions,
            numeric_features=parsed_numeric_features,
            numeric_operators=parsed_numeric_operators,
            numeric_quantiles=parsed_quantiles,
            random_iterations=random_iterations,
            random_seed=random_seed,
            min_train_condition_count=min_train_condition_count,
            min_train_flagged_count=min_train_flagged_count,
            min_oos_flagged_count=min_oos_flagged_count,
            max_condition_values=max_condition_values,
            max_selected_rules=max_selected_rules,
            max_flag_rules=max_flag_rules,
            max_single_symbol_share=max_single_symbol_share,
            max_single_session_share=max_single_session_share,
        ),
    )
    console.print(
        {
            "output_name": "conditional_context_caveat_v0",
            "run_id": result.run_id,
            "input_context_report_dir": str(result.input_context_report_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "conditional_caveat_results_csv_path": str(
                result.conditional_caveat_results_csv_path
            ),
            "selected_conditional_caveats_csv_path": str(
                result.selected_conditional_caveats_csv_path
            ),
            "strict_validation_results_csv_path": str(result.strict_validation_results_csv_path),
            "trade_conditional_caveat_flags_csv_path": str(
                result.trade_conditional_caveat_flags_csv_path
            ),
            "decision": result.decision,
            "selected_caveat_count": result.selected_caveat_count,
        }
    )


@research_app.command("personality-context-admission")
def research_personality_context_admission(
    input_baseline_report_dir: Annotated[
        Path,
        typer.Option("--input-baseline-report-dir"),
    ],
    input_candidate_report_dir: Annotated[
        Path,
        typer.Option("--input-candidate-report-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_context_admission_v0"),
    train_months: Annotated[
        str,
        typer.Option("--train-months"),
    ] = "2026-01,2026-02,2026-03,2026-04",
    test_months: Annotated[
        str,
        typer.Option("--test-months"),
    ] = "2026-05,2026-06",
    target_personalities: Annotated[
        str,
        typer.Option("--target-personalities"),
    ] = "",
    context_features: Annotated[
        str,
        typer.Option("--context-features"),
    ] = "",
    trade_key_columns: Annotated[
        str,
        typer.Option("--trade-key-columns"),
    ] = (
        "symbol,timestamp,personality,stop_model,target_r,"
        "monthly_candidate_rank,selected_filter_rank"
    ),
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 3000,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    min_train_admitted_count: Annotated[
        int,
        typer.Option("--min-train-admitted-count", min=1),
    ] = 3,
    min_oos_admitted_count: Annotated[
        int,
        typer.Option("--min-oos-admitted-count", min=1),
    ] = 1,
    max_context_values: Annotated[
        int,
        typer.Option("--max-context-values", min=1),
    ] = 40,
    max_selected_rules_per_personality: Annotated[
        int,
        typer.Option("--max-selected-rules-per-personality", min=1),
    ] = 3,
    max_flag_rules: Annotated[
        int,
        typer.Option("--max-flag-rules", min=1),
    ] = 40,
    max_single_symbol_share: Annotated[
        float,
        typer.Option("--max-single-symbol-share", min=0.0, max=1.0),
    ] = 0.50,
    max_single_session_share: Annotated[
        float,
        typer.Option("--max-single-session-share", min=0.0, max=1.0),
    ] = 0.20,
) -> None:
    """Validate personality-specific context admissions over staged report diffs."""

    from stocker_research.personality_context_admission_v0 import (
        PersonalityContextAdmissionConfig,
        run_personality_context_admission_lab,
    )

    parsed_train_months = tuple(part.strip() for part in train_months.split(",") if part.strip())
    parsed_test_months = tuple(part.strip() for part in test_months.split(",") if part.strip())
    parsed_personalities = tuple(
        part.strip() for part in target_personalities.split(",") if part.strip()
    )
    parsed_context_features = tuple(
        part.strip() for part in context_features.split(",") if part.strip()
    )
    parsed_key_columns = tuple(
        part.strip() for part in trade_key_columns.split(",") if part.strip()
    )
    if not parsed_train_months:
        raise typer.BadParameter("Supply at least one train month with --train-months.")
    if not parsed_test_months:
        raise typer.BadParameter("Supply at least one test month with --test-months.")
    if not parsed_key_columns:
        raise typer.BadParameter("Supply at least one trade key column.")

    result = run_personality_context_admission_lab(
        input_baseline_report_dir=input_baseline_report_dir,
        input_candidate_report_dir=input_candidate_report_dir,
        output_dir=output_dir,
        config=PersonalityContextAdmissionConfig(
            train_months=parsed_train_months,
            test_months=parsed_test_months,
            target_personalities=parsed_personalities,
            context_features=parsed_context_features,
            trade_key_columns=parsed_key_columns,
            random_iterations=random_iterations,
            random_seed=random_seed,
            min_train_admitted_count=min_train_admitted_count,
            min_oos_admitted_count=min_oos_admitted_count,
            max_context_values=max_context_values,
            max_selected_rules_per_personality=max_selected_rules_per_personality,
            max_flag_rules=max_flag_rules,
            max_single_symbol_share=max_single_symbol_share,
            max_single_session_share=max_single_session_share,
        ),
    )
    console.print(
        {
            "output_name": "personality_context_admission_v0",
            "run_id": result.run_id,
            "input_baseline_report_dir": str(result.input_baseline_report_dir),
            "input_candidate_report_dir": str(result.input_candidate_report_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "admission_rule_results_csv_path": str(
                result.admission_rule_results_csv_path
            ),
            "selected_admissions_csv_path": str(result.selected_admissions_csv_path),
            "blocked_candidate_trades_csv_path": str(
                result.blocked_candidate_trades_csv_path
            ),
            "trade_admission_flags_csv_path": str(result.trade_admission_flags_csv_path),
            "decision": result.decision,
            "selected_admission_count": result.selected_admission_count,
        }
    )


def _parse_report_pair_specs(report_pair: list[str]) -> tuple[tuple[str, Path, Path], ...]:
    parsed: list[tuple[str, Path, Path]] = []
    for index, spec in enumerate(report_pair, start=1):
        label = f"pair_{index}"
        paths_part = spec
        if "=" in spec:
            label, paths_part = spec.split("=", 1)
            label = label.strip() or f"pair_{index}"
        parts = [part.strip() for part in paths_part.split(",", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise typer.BadParameter(
                "Report pairs must be `label=baseline_report_dir,candidate_report_dir`."
            )
        parsed.append((label, Path(parts[0]), Path(parts[1])))
    return tuple(parsed)


def _parse_event_rows_specs(event_rows: list[str]) -> tuple[tuple[str, Path], ...]:
    parsed: list[tuple[str, Path]] = []
    for index, spec in enumerate(event_rows, start=1):
        label = f"surface_{index}"
        path_part = spec
        if "=" in spec:
            label, path_part = spec.split("=", 1)
            label = label.strip() or f"surface_{index}"
        path_part = path_part.strip()
        if not path_part:
            raise typer.BadParameter(
                "Event-row inputs must be `label=event_rows.csv` or `event_rows.csv`."
            )
        parsed.append((label, Path(path_part)))
    return tuple(parsed)


@research_app.command("template-discovery-system")
def research_template_discovery_system(
    input_event_rows: Annotated[
        list[str] | None,
        typer.Option("--input-event-rows"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/template_discovery_system_v0"),
    mode: Annotated[str, typer.Option("--mode")] = "container-routing",
    universe_profile: Annotated[
        str,
        typer.Option("--universe-profile"),
    ] = "liquid_midcap",
    horizons: Annotated[str, typer.Option("--horizons")] = "6,9,12,24",
    behavior_loop_discovery_period: Annotated[
        str,
        typer.Option("--behavior-loop-discovery-period"),
    ] = "saved_year",
    min_behavior_loop_rows: Annotated[
        int,
        typer.Option("--min-behavior-loop-rows", min=1),
    ] = 500,
    min_behavior_loop_transition_rate: Annotated[
        float,
        typer.Option("--min-behavior-loop-transition-rate", min=0.0),
    ] = 0.20,
    min_behavior_loop_symbols: Annotated[
        int,
        typer.Option("--min-behavior-loop-symbols", min=1),
    ] = 20,
    min_behavior_loop_months: Annotated[
        int,
        typer.Option("--min-behavior-loop-months", min=1),
    ] = 10,
    min_behavior_loop_split_rows: Annotated[
        int,
        typer.Option("--min-behavior-loop-split-rows", min=1),
    ] = 100,
    min_loop_regime_rows: Annotated[
        int,
        typer.Option("--min-loop-regime-rows", min=1),
    ] = 30,
    min_loop_refinement_rows: Annotated[
        int,
        typer.Option("--min-loop-refinement-rows", min=1),
    ] = 80,
    max_loop_refinement_terms_per_loop: Annotated[
        int,
        typer.Option("--max-loop-refinement-terms-per-loop", min=1),
    ] = 3,
    min_atom_rows: Annotated[int, typer.Option("--min-atom-rows", min=1)] = 100,
    min_container_rows: Annotated[
        int,
        typer.Option("--min-container-rows", min=1),
    ] = 120,
    min_loop_inside_rows: Annotated[
        int,
        typer.Option("--min-loop-inside-rows", min=1),
    ] = 8,
    min_loop_outside_rows: Annotated[
        int,
        typer.Option("--min-loop-outside-rows", min=1),
    ] = 8,
    route_lift_bar: Annotated[
        float,
        typer.Option("--route-lift-bar", min=0.0),
    ] = 0.05,
    max_containers_to_route: Annotated[
        int,
        typer.Option("--max-containers-to-route", min=1),
    ] = 5,
    stop_models: Annotated[str, typer.Option("--stop-models")] = "fixed_50bps",
    target_r_multiples: Annotated[
        str,
        typer.Option("--target-r-multiples"),
    ] = "1.0",
    cost_bps_values: Annotated[
        str,
        typer.Option("--cost-bps-values"),
    ] = "0,5,10",
    frozen_combo_dir: Annotated[
        Path | None,
        typer.Option("--frozen-combo-dir"),
    ] = None,
    frozen_component_rows: Annotated[
        list[Path] | None,
        typer.Option("--frozen-component-rows"),
    ] = None,
    frozen_candidate_book: Annotated[
        Path | None,
        typer.Option("--frozen-candidate-book"),
    ] = None,
    component_candidate_dir: Annotated[
        Path | None,
        typer.Option("--component-candidate-dir"),
    ] = None,
    component_candidate_rows: Annotated[
        list[Path] | None,
        typer.Option("--component-candidate-rows"),
    ] = None,
    min_component_candidate_rows: Annotated[
        int,
        typer.Option("--min-component-candidate-rows", min=1),
    ] = 10,
    min_component_total_r: Annotated[
        float,
        typer.Option("--min-component-total-r"),
    ] = 0.0,
    max_component_negative_months: Annotated[
        int,
        typer.Option("--max-component-negative-months", min=0),
    ] = 8,
    max_component_single_symbol_share: Annotated[
        float,
        typer.Option("--max-component-single-symbol-share", min=0.0),
    ] = 0.35,
    max_component_candidates_per_family: Annotated[
        int,
        typer.Option("--max-component-candidates-per-family", min=1),
    ] = 20,
) -> None:
    """Run clean-slate research-only template discovery from event rows."""

    from stocker_research.template_discovery_system_v0 import (
        SUPPORTED_MODES,
        TemplateDiscoveryEventInput,
        TemplateDiscoverySystemConfig,
        run_template_discovery_system_lab,
    )

    parsed_inputs = _parse_event_rows_specs(input_event_rows or [])
    no_event_row_modes = {"frozen-combo-replay", "template-component-selection"}
    if not parsed_inputs and mode not in no_event_row_modes:
        raise typer.BadParameter("Supply at least one --input-event-rows.")
    if mode not in SUPPORTED_MODES:
        raise typer.BadParameter(
            f"Unsupported mode {mode!r}. Supported modes: {', '.join(sorted(SUPPORTED_MODES))}."
        )
    parsed_horizons = tuple(int(part.strip()) for part in horizons.split(",") if part.strip())
    if not parsed_horizons:
        raise typer.BadParameter("Supply at least one horizon.")
    parsed_stop_models = tuple(part.strip() for part in stop_models.split(",") if part.strip())
    parsed_target_r = tuple(
        float(part.strip()) for part in target_r_multiples.split(",") if part.strip()
    )
    parsed_cost_bps = tuple(
        float(part.strip()) for part in cost_bps_values.split(",") if part.strip()
    )

    result = run_template_discovery_system_lab(
        input_event_rows=tuple(
            TemplateDiscoveryEventInput(label, event_rows_path)
            for label, event_rows_path in parsed_inputs
        ),
        output_dir=output_dir,
        config=TemplateDiscoverySystemConfig(
            mode=mode,
            universe_profile=universe_profile,
            horizons=parsed_horizons,
            behavior_loop_discovery_period=behavior_loop_discovery_period,
            min_behavior_loop_rows=min_behavior_loop_rows,
            min_behavior_loop_transition_rate=min_behavior_loop_transition_rate,
            min_behavior_loop_symbols=min_behavior_loop_symbols,
            min_behavior_loop_months=min_behavior_loop_months,
            min_behavior_loop_split_rows=min_behavior_loop_split_rows,
            min_loop_regime_rows=min_loop_regime_rows,
            min_loop_refinement_rows=min_loop_refinement_rows,
            max_loop_refinement_terms_per_loop=max_loop_refinement_terms_per_loop,
            min_atom_rows=min_atom_rows,
            min_container_rows=min_container_rows,
            min_loop_inside_rows=min_loop_inside_rows,
            min_loop_outside_rows=min_loop_outside_rows,
            route_lift_bar=route_lift_bar,
            max_containers_to_route=max_containers_to_route,
            stop_models=parsed_stop_models,
            target_r_multiples=parsed_target_r,
            cost_bps_values=parsed_cost_bps,
            frozen_combo_dir=frozen_combo_dir,
            frozen_component_paths=tuple(frozen_component_rows or ()),
            frozen_candidate_book_path=frozen_candidate_book,
            component_candidate_dir=component_candidate_dir,
            component_candidate_paths=tuple(component_candidate_rows or ()),
            min_component_candidate_rows=min_component_candidate_rows,
            min_component_total_r=min_component_total_r,
            max_component_negative_months=max_component_negative_months,
            max_component_single_symbol_share=max_component_single_symbol_share,
            max_component_candidates_per_family=max_component_candidates_per_family,
        ),
    )
    console.print(
        {
            "output_name": "template_discovery_system_v0",
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "behavior_loop_scorecard_csv_path": str(
                result.behavior_loop_scorecard_csv_path
            ),
            "loop_regime_occupancy_csv_path": str(
                result.loop_regime_occupancy_csv_path
            ),
            "loop_mixed_regime_occupancy_csv_path": str(
                result.loop_mixed_regime_occupancy_csv_path
            ),
            "loop_transition_regime_occupancy_csv_path": str(
                result.loop_transition_regime_occupancy_csv_path
            ),
            "c0_parent_readout_csv_path": str(result.c0_parent_readout_csv_path),
            "b0_state_summary_csv_path": str(result.b0_state_summary_csv_path),
            "b0_route_detail_csv_path": str(result.b0_route_detail_csv_path),
            "loop_context_refinement_csv_path": str(
                result.loop_context_refinement_csv_path
            ),
            "loop_context_admissions_csv_path": str(
                result.loop_context_admissions_csv_path
            ),
            "loop_context_blockers_csv_path": str(
                result.loop_context_blockers_csv_path
            ),
            "atom_scorecard_csv_path": str(result.atom_scorecard_csv_path),
            "container_scorecard_csv_path": str(result.container_scorecard_csv_path),
            "loop_routing_detail_csv_path": str(result.loop_routing_detail_csv_path),
            "family_test_detail_csv_path": str(result.family_test_detail_csv_path),
            "concentration_warnings_csv_path": str(result.concentration_warnings_csv_path),
            "admission_candidates_csv_path": str(result.admission_candidates_csv_path),
            "blocker_candidates_csv_path": str(result.blocker_candidates_csv_path),
            "replay_results_csv_path": str(result.replay_results_csv_path),
            "family_r_replay_summary_csv_path": str(
                result.output_dir / "family_r_replay_summary.csv"
            ),
            "family_r_replay_scorecard_csv_path": str(
                result.output_dir / "family_r_replay_scorecard.csv"
            ),
            "family_r_replay_cost_sensitivity_csv_path": str(
                result.output_dir / "family_r_replay_cost_sensitivity.csv"
            ),
            "family_r_replay_selected_events_csv_path": str(
                result.output_dir / "family_r_replay_selected_events.csv"
            ),
            "component_candidate_scorecard_csv_path": str(
                result.output_dir / "component_candidate_scorecard.csv"
            ),
            "selected_candidate_book_csv_path": str(
                result.output_dir / "selected_candidate_book.csv"
            ),
            "selected_combo_exact_dedupe_trades_csv_path": str(
                result.output_dir / "selected_combo_exact_dedupe_trades.csv"
            ),
            "frozen_template_transfer_all_rows_csv_path": str(
                result.output_dir / "frozen_template_transfer_all_rows.csv"
            ),
            "frozen_template_transfer_exact_dedupe_trades_csv_path": str(
                result.output_dir
                / "frozen_template_transfer_exact_dedupe_trades.csv"
            ),
            "frozen_template_transfer_template_audit_csv_path": str(
                result.output_dir / "frozen_template_transfer_template_audit.csv"
            ),
            "frozen_template_transfer_summary_csv_path": str(
                result.output_dir / "frozen_template_transfer_summary.csv"
            ),
            "decision": result.decision,
        }
    )


@research_app.command("personality-context-workflow")
def research_personality_context_workflow(
    report_pair: Annotated[
        list[str] | None,
        typer.Option("--report-pair"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_context_workflow_v0"),
    target_personality: Annotated[
        str,
        typer.Option("--target-personality"),
    ] = "",
    saved_rules_path: Annotated[
        Path,
        typer.Option("--saved-rules-path"),
    ] = Path("configs/research/personality_context_admission_v0/candidate_rules.yaml"),
    categorical_features: Annotated[
        str,
        typer.Option("--categorical-features"),
    ] = "",
    numeric_features: Annotated[
        str,
        typer.Option("--numeric-features"),
    ] = "",
    quantiles: Annotated[
        str,
        typer.Option("--quantiles"),
    ] = "0.20,0.33,0.50,0.67,0.80",
    min_rule_trades: Annotated[
        int,
        typer.Option("--min-rule-trades", min=1),
    ] = 3,
    min_rule_windows: Annotated[
        int,
        typer.Option("--min-rule-windows", min=1),
    ] = 2,
    min_positive_windows: Annotated[
        int,
        typer.Option("--min-positive-windows", min=1),
    ] = 2,
    max_negative_windows: Annotated[
        int,
        typer.Option("--max-negative-windows", min=0),
    ] = 0,
    max_single_window_share: Annotated[
        float,
        typer.Option("--max-single-window-share", min=0.0, max=1.0),
    ] = 0.65,
    min_blocked_count: Annotated[
        int,
        typer.Option("--min-blocked-count", min=1),
    ] = 3,
    max_blocker_screens: Annotated[
        int,
        typer.Option("--max-blocker-screens", min=1),
    ] = 250,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 1000,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    save_yaml: Annotated[
        bool,
        typer.Option("--save-yaml/--no-save-yaml"),
    ] = False,
) -> None:
    """Run the research-only single-personality context workflow over report pairs."""

    from stocker_research.personality_context_rule_discovery_v0 import ReportPair
    from stocker_research.personality_context_workflow_v0 import (
        DEFAULT_WORKFLOW_CATEGORICAL_FEATURES,
        DEFAULT_WORKFLOW_NUMERIC_FEATURES,
        PersonalityContextWorkflowConfig,
        run_personality_context_workflow_lab,
    )

    parsed_pairs = _parse_report_pair_specs(report_pair or [])
    if not parsed_pairs:
        raise typer.BadParameter("Supply at least one --report-pair.")
    parsed_categorical_features = (
        tuple(part.strip() for part in categorical_features.split(",") if part.strip())
        if categorical_features.strip()
        else DEFAULT_WORKFLOW_CATEGORICAL_FEATURES
    )
    parsed_numeric_features = (
        tuple(part.strip() for part in numeric_features.split(",") if part.strip())
        if numeric_features.strip()
        else DEFAULT_WORKFLOW_NUMERIC_FEATURES
    )
    parsed_quantiles = tuple(float(part.strip()) for part in quantiles.split(",") if part.strip())
    if not parsed_categorical_features:
        raise typer.BadParameter("Supply at least one categorical feature.")
    if not parsed_numeric_features:
        raise typer.BadParameter("Supply at least one numeric feature.")
    if not parsed_quantiles:
        raise typer.BadParameter("Supply at least one quantile.")

    result = run_personality_context_workflow_lab(
        report_pairs=tuple(
            ReportPair(label, baseline, candidate)
            for label, baseline, candidate in parsed_pairs
        ),
        output_dir=output_dir,
        config=PersonalityContextWorkflowConfig(
            target_personality=target_personality.strip() or None,
            saved_rules_path=saved_rules_path,
            categorical_features=parsed_categorical_features,
            numeric_features=parsed_numeric_features,
            quantiles=parsed_quantiles,
            min_rule_trades=min_rule_trades,
            min_rule_windows=min_rule_windows,
            min_positive_windows=min_positive_windows,
            max_negative_windows=max_negative_windows,
            max_single_window_share=max_single_window_share,
            min_blocked_count=min_blocked_count,
            max_blocker_screens=max_blocker_screens,
            random_iterations=random_iterations,
            random_seed=random_seed,
            save_yaml_to_registry=save_yaml,
        ),
    )
    console.print(
        {
            "output_name": "personality_context_workflow_v0",
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "personality_ranking_csv_path": str(result.personality_ranking_csv_path),
            "selected_no_prior_trades_csv_path": str(
                result.selected_no_prior_trades_csv_path
            ),
            "selected_candidate_only_trades_csv_path": str(
                result.selected_candidate_only_trades_csv_path
            ),
            "categorical_commonality_csv_path": str(result.categorical_commonality_csv_path),
            "numeric_commonality_csv_path": str(result.numeric_commonality_csv_path),
            "no_prior_defensive_screens_csv_path": str(
                result.no_prior_defensive_screens_csv_path
            ),
            "yaml_draft_path": str(result.yaml_draft_path),
            "selected_personality": result.selected_personality,
            "decision": result.decision,
        }
    )


@research_app.command("personality-context-rule-discovery")
def research_personality_context_rule_discovery(
    report_pair: Annotated[
        list[str] | None,
        typer.Option("--report-pair"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/personality_context_rule_discovery_v0"),
    target_personalities: Annotated[
        str,
        typer.Option("--target-personalities"),
    ] = "",
    categorical_features: Annotated[
        str,
        typer.Option("--categorical-features"),
    ] = "",
    numeric_features: Annotated[
        str,
        typer.Option("--numeric-features"),
    ] = "",
    quantiles: Annotated[
        str,
        typer.Option("--quantiles"),
    ] = "0.20,0.33,0.50,0.67,0.80",
    min_rule_trades: Annotated[
        int,
        typer.Option("--min-rule-trades", min=1),
    ] = 3,
    min_rule_windows: Annotated[
        int,
        typer.Option("--min-rule-windows", min=1),
    ] = 2,
    min_positive_windows: Annotated[
        int,
        typer.Option("--min-positive-windows", min=1),
    ] = 2,
    max_negative_windows: Annotated[
        int,
        typer.Option("--max-negative-windows", min=0),
    ] = 0,
    max_single_window_share: Annotated[
        float,
        typer.Option("--max-single-window-share", min=0.0, max=1.0),
    ] = 0.65,
    max_atomic_rules_per_personality: Annotated[
        int,
        typer.Option("--max-atomic-rules-per-personality", min=1),
    ] = 800,
    max_union_base_rules_per_personality: Annotated[
        int,
        typer.Option("--max-union-base-rules-per-personality", min=1),
    ] = 24,
    max_union_rules_per_personality: Annotated[
        int,
        typer.Option("--max-union-rules-per-personality", min=1),
    ] = 250,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 3000,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Discover personality-specific context admission rules from report pairs."""

    from stocker_research.personality_context_rule_discovery_v0 import (
        DEFAULT_CATEGORICAL_FEATURES,
        DEFAULT_NUMERIC_FEATURES,
        PersonalityContextRuleDiscoveryConfig,
        ReportPair,
        run_personality_context_rule_discovery_lab,
    )

    parsed_pairs = _parse_report_pair_specs(report_pair or [])
    if not parsed_pairs:
        raise typer.BadParameter("Supply at least one --report-pair.")
    parsed_personalities = tuple(
        part.strip() for part in target_personalities.split(",") if part.strip()
    )
    parsed_categorical_features = (
        tuple(part.strip() for part in categorical_features.split(",") if part.strip())
        if categorical_features.strip()
        else DEFAULT_CATEGORICAL_FEATURES
    )
    parsed_numeric_features = (
        tuple(part.strip() for part in numeric_features.split(",") if part.strip())
        if numeric_features.strip()
        else DEFAULT_NUMERIC_FEATURES
    )
    parsed_quantiles = tuple(float(part.strip()) for part in quantiles.split(",") if part.strip())
    if not parsed_quantiles:
        raise typer.BadParameter("Supply at least one quantile.")

    result = run_personality_context_rule_discovery_lab(
        report_pairs=tuple(
            ReportPair(
                label=label,
                baseline_report_dir=baseline,
                candidate_report_dir=candidate,
            )
            for label, baseline, candidate in parsed_pairs
        ),
        output_dir=output_dir,
        config=PersonalityContextRuleDiscoveryConfig(
            target_personalities=parsed_personalities,
            categorical_features=parsed_categorical_features,
            numeric_features=parsed_numeric_features,
            quantiles=parsed_quantiles,
            min_rule_trades=min_rule_trades,
            min_rule_windows=min_rule_windows,
            min_positive_windows=min_positive_windows,
            max_negative_windows=max_negative_windows,
            max_single_window_share=max_single_window_share,
            max_atomic_rules_per_personality=max_atomic_rules_per_personality,
            max_union_base_rules_per_personality=(
                max_union_base_rules_per_personality
            ),
            max_union_rules_per_personality=max_union_rules_per_personality,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "personality_context_rule_discovery_v0",
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "rule_results_csv_path": str(result.rule_results_csv_path),
            "rule_window_results_csv_path": str(result.rule_window_results_csv_path),
            "selected_rules_csv_path": str(result.selected_rules_csv_path),
            "candidate_only_trades_csv_path": str(result.candidate_only_trades_csv_path),
            "decision": result.decision,
            "selected_rule_count": result.selected_rule_count,
        }
    )


@research_app.command("shadow-candidate-trigger-audit")
def research_shadow_candidate_trigger_audit(
    input_context_report_dir: Annotated[
        Path,
        typer.Option("--input-context-report-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/shadow_candidate_trigger_audit_v0"),
    shadow_window: Annotated[
        int,
        typer.Option("--shadow-window", min=1),
    ] = 20,
    min_prior_candidates: Annotated[
        int,
        typer.Option("--min-prior-candidates", min=0),
    ] = 8,
    weak_context_max_score: Annotated[
        int,
        typer.Option("--weak-context-max-score", min=0),
    ] = 3,
    weak_context_share_threshold: Annotated[
        float,
        typer.Option("--weak-context-share-threshold", min=0.0, max=1.0),
    ] = 0.75,
    shadow_net_r_threshold: Annotated[
        float,
        typer.Option("--shadow-net-r-threshold"),
    ] = -1.0,
    anti_stale_windows: Annotated[
        str,
        typer.Option("--anti-stale-windows"),
    ] = "6,12,24,36",
    anti_stale_feature_bases: Annotated[
        str,
        typer.Option("--anti-stale-feature-bases"),
    ] = "time_regime,time_x_vwap_regime",
    anti_stale_quantiles: Annotated[
        str,
        typer.Option("--anti-stale-quantiles"),
    ] = "0.10,0.20,0.33,0.50,0.67",
    min_train_count: Annotated[
        int,
        typer.Option("--min-train-count", min=1),
    ] = 30,
    min_rule_keep_count: Annotated[
        int,
        typer.Option("--min-rule-keep-count", min=1),
    ] = 8,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 1000,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Audit shadow-candidate weak-cluster deterioration triggers."""

    from stocker_research.shadow_candidate_trigger_audit_v0 import (
        ShadowCandidateTriggerConfig,
        run_shadow_candidate_trigger_audit,
    )

    parsed_windows = tuple(
        int(part.strip()) for part in anti_stale_windows.split(",") if part.strip()
    )
    parsed_feature_bases = tuple(
        part.strip() for part in anti_stale_feature_bases.split(",") if part.strip()
    )
    parsed_quantiles = tuple(
        float(part.strip()) for part in anti_stale_quantiles.split(",") if part.strip()
    )
    if not parsed_windows:
        raise typer.BadParameter("Supply at least one anti-stale window.")
    if not parsed_feature_bases:
        raise typer.BadParameter("Supply at least one anti-stale feature base.")
    if not parsed_quantiles:
        raise typer.BadParameter("Supply at least one anti-stale quantile.")

    result = run_shadow_candidate_trigger_audit(
        input_context_report_dir=input_context_report_dir,
        output_dir=output_dir,
        config=ShadowCandidateTriggerConfig(
            shadow_window=shadow_window,
            min_prior_candidates=min_prior_candidates,
            weak_context_max_score=weak_context_max_score,
            weak_context_share_threshold=weak_context_share_threshold,
            shadow_net_r_threshold=shadow_net_r_threshold,
            anti_stale_windows=parsed_windows,
            anti_stale_feature_bases=parsed_feature_bases,
            anti_stale_quantiles=parsed_quantiles,
            min_train_count=min_train_count,
            min_rule_keep_count=min_rule_keep_count,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "shadow_candidate_trigger_audit_v0",
            "run_id": result.run_id,
            "input_context_report_dir": str(result.input_context_report_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "shadow_candidate_features_csv_path": str(
                result.shadow_candidate_features_csv_path
            ),
            "monthly_policy_results_csv_path": str(result.monthly_policy_results_csv_path),
            "policy_summary_csv_path": str(result.policy_summary_csv_path),
            "trade_shadow_trigger_flags_csv_path": str(
                result.trade_shadow_trigger_flags_csv_path
            ),
            "decision": result.decision,
        }
    )


@research_app.command("pre-registered-edge-proof")
def research_pre_registered_edge_proof(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    input_staged_report_dir: Annotated[
        Path,
        typer.Option("--input-staged-report-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/pre_registered_edge_proof_v0"),
    registration_cutoff_month: Annotated[
        str,
        typer.Option("--registration-cutoff-month"),
    ] = "2026-06",
    evaluation_months: Annotated[
        str,
        typer.Option("--evaluation-months"),
    ] = "2026-07",
    source_month: Annotated[
        str,
        typer.Option("--source-month"),
    ] = "",
    personality: Annotated[
        str,
        typer.Option("--personality"),
    ] = "active_liquidation",
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", min=1),
    ] = 1,
    cost_bps: Annotated[
        float,
        typer.Option("--cost-bps"),
    ] = 10.0,
    min_replay_signals: Annotated[
        int,
        typer.Option("--min-replay-signals", min=1),
    ] = 1,
    min_forward_trades: Annotated[
        int,
        typer.Option("--min-forward-trades", min=1),
    ] = 15,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 1000,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Register and evaluate a frozen research tuple on future months."""

    from stocker_research.pre_registered_edge_proof_v0 import (
        PreRegisteredEdgeProofConfig,
        run_pre_registered_edge_proof,
    )

    parsed_months = tuple(part.strip() for part in evaluation_months.split(",") if part.strip())
    if not parsed_months:
        raise typer.BadParameter("Supply at least one month with --evaluation-months.")
    parsed_source_month = source_month.strip() or None

    result = run_pre_registered_edge_proof(
        input_event_dir=input_event_dir,
        input_staged_report_dir=input_staged_report_dir,
        output_dir=output_dir,
        config=PreRegisteredEdgeProofConfig(
            registration_cutoff_month=registration_cutoff_month,
            evaluation_months=parsed_months,
            source_month=parsed_source_month,
            personality=personality,
            max_candidates=max_candidates,
            cost_bps=cost_bps,
            min_replay_signals=min_replay_signals,
            min_forward_trades=min_forward_trades,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "pre_registered_edge_proof_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "input_staged_report_dir": str(result.input_staged_report_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "registration_json_path": str(result.registration_json_path),
            "frozen_candidates_csv_path": str(result.frozen_candidates_csv_path),
            "frozen_caveats_csv_path": str(result.frozen_caveats_csv_path),
            "evaluation_monthly_summary_csv_path": str(
                result.evaluation_monthly_summary_csv_path
            ),
            "evaluation_trades_csv_path": str(result.evaluation_trades_csv_path),
            "decision": result.decision,
            "trade_count": result.trade_count,
        }
    )


@research_app.command("walk-forward-staged-mixed-regime-caveat-exit")
def research_walk_forward_staged_mixed_regime_caveat_exit(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    input_personality_discovery_dir: Annotated[
        Path,
        typer.Option("--input-personality-discovery-dir"),
    ],
    input_caveat_report_dir: Annotated[
        Path | None,
        typer.Option("--input-caveat-report-dir"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/walk_forward_staged_mixed_regime_caveat_exit_v0"),
    warmup_months: Annotated[
        str,
        typer.Option("--warmup-months"),
    ] = "",
    replay_months: Annotated[
        str,
        typer.Option("--replay-months"),
    ] = "2026-01,2026-02,2026-03,2026-04,2026-05,2026-06",
    combined_regime_fields: Annotated[
        str,
        typer.Option("--combined-regime-fields"),
    ] = (
        "vwap_x_efficiency_regime,vwap_x_range_regime,"
        "compression_x_efficiency_regime,opening_mid_x_range_regime,"
        "time_x_vwap_regime,volume_x_vwap_regime"
    ),
    mixed_regime_value_contains: Annotated[
        str,
        typer.Option("--mixed-regime-value-contains"),
    ] = "",
    allowed_caveat_statuses: Annotated[
        str,
        typer.Option("--allowed-caveat-statuses"),
    ] = "strict_train_and_oos_supported",
    stop_models: Annotated[
        str,
        typer.Option("--stop-models"),
    ] = (
        "fixed_50bps,fixed_75bps,fixed_100bps,"
        "structure_session_extreme_10bps,structure_recent_extreme_10bps,"
        "structure_opening_range_extreme_10bps"
    ),
    target_r_multiples: Annotated[
        str,
        typer.Option("--target-r-multiples"),
    ] = "1,1.5,2",
    cost_bps: Annotated[
        float,
        typer.Option("--cost-bps"),
    ] = 10.0,
    max_filters_per_personality: Annotated[
        int,
        typer.Option("--max-filters-per-personality", min=1),
    ] = 4,
    max_exit_candidates_per_month: Annotated[
        int,
        typer.Option("--max-exit-candidates-per-month", min=1),
    ] = 48,
    max_selected_per_month: Annotated[
        int,
        typer.Option("--max-selected-per-month", min=1),
    ] = 18,
    max_caveat_rules: Annotated[
        int,
        typer.Option("--max-caveat-rules", min=0),
    ] = 12,
    max_staged_caveat_rules_per_month: Annotated[
        int,
        typer.Option("--max-staged-caveat-rules-per-month", min=0),
    ] = 2,
    min_train_events: Annotated[
        int,
        typer.Option("--min-train-events", min=1),
    ] = 35,
    min_train_symbols: Annotated[
        int,
        typer.Option("--min-train-symbols", min=1),
    ] = 4,
    min_train_months: Annotated[
        int,
        typer.Option("--min-train-months", min=1),
    ] = 4,
    min_symbol_train_events: Annotated[
        int,
        typer.Option("--min-symbol-train-events", min=1),
    ] = 3,
    min_symbol_train_total_net_r: Annotated[
        float,
        typer.Option("--min-symbol-train-total-net-r"),
    ] = 0.0,
    min_symbol_train_win_rate: Annotated[
        float,
        typer.Option("--min-symbol-train-win-rate", min=0.0, max=1.0),
    ] = 0.0,
    enable_personality_acceptance: Annotated[
        bool,
        typer.Option("--enable-personality-acceptance/--disable-personality-acceptance"),
    ] = True,
    min_personality_train_trades: Annotated[
        int,
        typer.Option("--min-personality-train-trades", min=1),
    ] = 3,
    min_personality_train_total_net_r: Annotated[
        float,
        typer.Option("--min-personality-train-total-net-r"),
    ] = 0.0,
    min_personality_train_win_rate: Annotated[
        float,
        typer.Option("--min-personality-train-win-rate", min=0.0, max=1.0),
    ] = 0.0,
    enable_prior_replay_personality_acceptance: Annotated[
        bool,
        typer.Option(
            "--enable-prior-replay-personality-acceptance/"
            "--disable-prior-replay-personality-acceptance"
        ),
    ] = False,
    min_prior_replay_personality_trades: Annotated[
        int,
        typer.Option("--min-prior-replay-personality-trades", min=1),
    ] = 15,
    min_prior_replay_personality_total_net_r: Annotated[
        float,
        typer.Option("--min-prior-replay-personality-total-net-r"),
    ] = -1.0,
    min_prior_replay_personality_win_rate: Annotated[
        float,
        typer.Option("--min-prior-replay-personality-win-rate", min=0.0, max=1.0),
    ] = 0.0,
    min_staged_caveat_train_trades: Annotated[
        int,
        typer.Option("--min-staged-caveat-train-trades", min=1),
    ] = 35,
    min_staged_caveat_flagged_trades: Annotated[
        int,
        typer.Option("--min-staged-caveat-flagged-trades", min=1),
    ] = 5,
    min_total_trades: Annotated[
        int,
        typer.Option("--min-total-trades", min=1),
    ] = 30,
    allow_sparse_quality_decision: Annotated[
        bool,
        typer.Option("--allow-sparse-quality-decision/--disallow-sparse-quality-decision"),
    ] = False,
    min_sparse_total_trades: Annotated[
        int,
        typer.Option("--min-sparse-total-trades", min=1),
    ] = 15,
    min_sparse_positive_months: Annotated[
        int,
        typer.Option("--min-sparse-positive-months", min=1),
    ] = 4,
    min_sparse_win_rate: Annotated[
        float,
        typer.Option("--min-sparse-win-rate", min=0.0, max=1.0),
    ] = 0.65,
    min_sparse_mean_net_r: Annotated[
        float,
        typer.Option("--min-sparse-mean-net-r"),
    ] = 0.20,
    max_sparse_single_month_share: Annotated[
        float,
        typer.Option("--max-sparse-single-month-share", min=0.0, max=1.0),
    ] = 0.75,
    max_single_month_share: Annotated[
        float,
        typer.Option("--max-single-month-share", min=0.0, max=1.0),
    ] = 0.50,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
    enable_staged_train_caveats: Annotated[
        bool,
        typer.Option("--enable-staged-train-caveats/--disable-staged-train-caveats"),
    ] = True,
) -> None:
    """Replay personality -> mixed regime -> filter -> caveat -> exit."""

    from stocker_research.walk_forward_staged_mixed_regime_caveat_exit_v0 import (
        StagedMixedRegimeCaveatExitConfig,
        run_staged_mixed_regime_caveat_exit_lab,
    )

    parsed_warmup_months = tuple(
        part.strip() for part in warmup_months.split(",") if part.strip()
    )
    parsed_months = tuple(part.strip() for part in replay_months.split(",") if part.strip())
    parsed_combined_fields = tuple(
        part.strip() for part in combined_regime_fields.split(",") if part.strip()
    )
    parsed_mixed_terms = tuple(
        part.strip() for part in mixed_regime_value_contains.split(",") if part.strip()
    )
    parsed_statuses = tuple(
        part.strip() for part in allowed_caveat_statuses.split(",") if part.strip()
    )
    parsed_stop_models = tuple(part.strip() for part in stop_models.split(",") if part.strip())
    parsed_target_r = tuple(
        float(part.strip()) for part in target_r_multiples.split(",") if part.strip()
    )
    if not parsed_months:
        raise typer.BadParameter("Supply at least one replay month with --replay-months.")
    if not parsed_combined_fields:
        raise typer.BadParameter("Supply at least one combined regime field.")
    if not parsed_statuses:
        raise typer.BadParameter("Supply at least one caveat status.")
    if not parsed_stop_models:
        raise typer.BadParameter("Supply at least one stop model with --stop-models.")
    if not parsed_target_r:
        raise typer.BadParameter("Supply at least one target multiple.")

    result = run_staged_mixed_regime_caveat_exit_lab(
        input_event_dir=input_event_dir,
        input_personality_discovery_dir=input_personality_discovery_dir,
        input_caveat_report_dir=input_caveat_report_dir,
        output_dir=output_dir,
        config=StagedMixedRegimeCaveatExitConfig(
            warmup_months=parsed_warmup_months,
            replay_months=parsed_months,
            combined_regime_fields=parsed_combined_fields,
            mixed_regime_value_contains=parsed_mixed_terms,
            allowed_caveat_statuses=parsed_statuses,
            stop_models=parsed_stop_models,
            target_r_multiples=parsed_target_r,
            cost_bps=cost_bps,
            max_filters_per_personality=max_filters_per_personality,
            max_exit_candidates_per_month=max_exit_candidates_per_month,
            max_selected_per_month=max_selected_per_month,
            max_caveat_rules=max_caveat_rules,
            max_staged_caveat_rules_per_month=max_staged_caveat_rules_per_month,
            min_train_events=min_train_events,
            min_train_symbols=min_train_symbols,
            min_train_months=min_train_months,
            min_symbol_train_events=min_symbol_train_events,
            min_symbol_train_total_net_r=min_symbol_train_total_net_r,
            min_symbol_train_win_rate=min_symbol_train_win_rate,
            enable_personality_acceptance=enable_personality_acceptance,
            min_personality_train_trades=min_personality_train_trades,
            min_personality_train_total_net_r=min_personality_train_total_net_r,
            min_personality_train_win_rate=min_personality_train_win_rate,
            enable_prior_replay_personality_acceptance=(
                enable_prior_replay_personality_acceptance
            ),
            min_prior_replay_personality_trades=min_prior_replay_personality_trades,
            min_prior_replay_personality_total_net_r=(
                min_prior_replay_personality_total_net_r
            ),
            min_prior_replay_personality_win_rate=min_prior_replay_personality_win_rate,
            min_staged_caveat_train_trades=min_staged_caveat_train_trades,
            min_staged_caveat_flagged_trades=min_staged_caveat_flagged_trades,
            min_total_trades=min_total_trades,
            allow_sparse_quality_decision=allow_sparse_quality_decision,
            min_sparse_total_trades=min_sparse_total_trades,
            min_sparse_positive_months=min_sparse_positive_months,
            min_sparse_win_rate=min_sparse_win_rate,
            min_sparse_mean_net_r=min_sparse_mean_net_r,
            max_sparse_single_month_share=max_sparse_single_month_share,
            max_single_month_share=max_single_month_share,
            random_iterations=random_iterations,
            random_seed=random_seed,
            enable_staged_train_caveats=enable_staged_train_caveats,
        ),
    )
    console.print(
        {
            "output_name": "walk_forward_staged_mixed_regime_caveat_exit_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "input_personality_discovery_dir": str(result.input_personality_discovery_dir),
            "input_caveat_report_dir": str(result.input_caveat_report_dir)
            if result.input_caveat_report_dir is not None
            else None,
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "mixed_regime_filter_book_csv_path": str(result.mixed_regime_filter_book_csv_path),
            "caveat_rule_book_csv_path": str(result.caveat_rule_book_csv_path),
            "personality_acceptance_csv_path": str(result.personality_acceptance_csv_path),
            "selected_monthly_candidates_csv_path": str(
                result.selected_monthly_candidates_csv_path
            ),
            "signals_csv_path": str(result.signals_csv_path),
            "caveated_signals_csv_path": str(result.caveated_signals_csv_path),
            "trades_csv_path": str(result.trades_csv_path),
            "decision": result.decision,
            "trade_count": result.trade_count,
        }
    )


@research_app.command("sidelined-personality-cross-regime")
def research_sidelined_personality_cross_regime(
    input_event_dir: Annotated[
        Path,
        typer.Option("--input-event-dir"),
    ],
    input_selected_filter_dir: Annotated[
        Path,
        typer.Option("--input-selected-filter-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/sidelined_personality_cross_regime_v0"),
    external_report_dir: Annotated[
        list[Path] | None,
        typer.Option("--external-report-dir"),
    ] = None,
    horizons: Annotated[
        str,
        typer.Option("--horizons"),
    ] = "6,9,12,24",
    regime_fields: Annotated[
        str,
        typer.Option("--regime-fields"),
    ] = "",
    filter_features: Annotated[
        str,
        typer.Option("--filter-features"),
    ] = "",
    quantiles: Annotated[
        str,
        typer.Option("--quantiles"),
    ] = "0.20,0.35,0.50,0.65,0.80",
    min_train_events: Annotated[
        int,
        typer.Option("--min-train-events", min=1),
    ] = 30,
    min_test_events: Annotated[
        int,
        typer.Option("--min-test-events", min=1),
    ] = 12,
    min_retained_events: Annotated[
        int,
        typer.Option("--min-retained-events", min=1),
    ] = 8,
    min_symbols: Annotated[
        int,
        typer.Option("--min-symbols", min=1),
    ] = 3,
    low_movement_threshold: Annotated[
        float,
        typer.Option("--low-movement-threshold", min=0.0),
    ] = 0.0015,
    random_iterations: Annotated[
        int,
        typer.Option("--random-iterations", min=1),
    ] = 50,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 1337,
) -> None:
    """Screen sidelined personalities across regimes without touching execution."""

    from stocker_research.sidelined_personality_cross_regime_v0 import (
        DEFAULT_FILTER_FEATURES,
        DEFAULT_REGIME_FIELDS,
        SidelinedPersonalityCrossRegimeConfig,
        run_sidelined_personality_cross_regime_lab,
    )

    parsed_horizons = tuple(int(part.strip()) for part in horizons.split(",") if part.strip())
    parsed_regimes = (
        tuple(part.strip() for part in regime_fields.split(",") if part.strip())
        if regime_fields.strip()
        else DEFAULT_REGIME_FIELDS
    )
    parsed_features = (
        tuple(part.strip() for part in filter_features.split(",") if part.strip())
        if filter_features.strip()
        else DEFAULT_FILTER_FEATURES
    )
    parsed_quantiles = tuple(float(part.strip()) for part in quantiles.split(",") if part.strip())
    if not parsed_horizons:
        raise typer.BadParameter("Supply at least one horizon with --horizons.")
    if not parsed_regimes:
        raise typer.BadParameter("Supply at least one regime field.")
    if not parsed_features:
        raise typer.BadParameter("Supply at least one filter feature.")
    if not parsed_quantiles:
        raise typer.BadParameter("Supply at least one quantile.")

    result = run_sidelined_personality_cross_regime_lab(
        input_event_dir=input_event_dir,
        input_selected_filter_dir=input_selected_filter_dir,
        output_dir=output_dir,
        external_report_dirs=tuple(external_report_dir or ()),
        config=SidelinedPersonalityCrossRegimeConfig(
            horizons=parsed_horizons,
            regime_fields=parsed_regimes,
            filter_features=parsed_features,
            quantiles=parsed_quantiles,
            min_train_events=min_train_events,
            min_test_events=min_test_events,
            min_retained_events=min_retained_events,
            min_symbols=min_symbols,
            low_movement_threshold=low_movement_threshold,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "sidelined_personality_cross_regime_v0",
            "run_id": result.run_id,
            "input_event_dir": str(result.input_event_dir),
            "input_selected_filter_dir": str(result.input_selected_filter_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "selected_sidelined_candidates_csv_path": str(
                result.selected_sidelined_candidates_csv_path
            ),
            "decision": result.decision,
            "selected_candidate_count": result.selected_candidate_count,
        }
    )


@research_app.command("sparse-exhaustion-extension")
def research_sparse_exhaustion_extension(
    input_candidate_report_dir: Annotated[
        Path,
        typer.Option("--input-candidate-report-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/sparse_exhaustion_extension_v0"),
) -> None:
    """Formalize existing sparse exhaustion-extension candidate rows."""

    from stocker_research.sparse_exhaustion_extension_v0 import (
        run_sparse_exhaustion_extension_lab,
    )

    result = run_sparse_exhaustion_extension_lab(
        input_candidate_report_dir=input_candidate_report_dir,
        output_dir=output_dir,
    )
    console.print(
        {
            "output_name": "sparse_exhaustion_extension_v0",
            "run_id": result.run_id,
            "input_candidate_report_dir": str(result.input_candidate_report_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "exhaustion_event_rows_csv_path": str(result.exhaustion_event_rows_csv_path),
            "decision": result.decision,
            "event_count": result.event_count,
        }
    )


@research_app.command("exhaustion-extension-exit-replay")
def research_exhaustion_extension_exit_replay(
    input_exhaustion_event_dir: Annotated[
        Path,
        typer.Option("--input-exhaustion-event-dir"),
    ],
    input_filter_report_dir: Annotated[
        Path,
        typer.Option("--input-filter-report-dir"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("data/reports/research/exhaustion_extension_exit_replay_v0"),
    replay_months: Annotated[
        str,
        typer.Option("--replay-months", help="Comma-separated YYYY-MM replay months."),
    ] = "2026-01,2026-02,2026-03,2026-04,2026-05,2026-06",
    stop_models: Annotated[
        str,
        typer.Option("--stop-models", help="Comma-separated stop model names."),
    ] = (
        "fixed_50bps,fixed_75bps,fixed_100bps,"
        "structure_session_extreme_10bps,structure_recent_extreme_10bps,"
        "structure_opening_range_extreme_10bps"
    ),
    target_r_multiples: Annotated[
        str,
        typer.Option("--target-r-multiples", help="Comma-separated target R multiples."),
    ] = "1.0,1.5,2.0",
    cost_bps: Annotated[float, typer.Option("--cost-bps", min=0.0)] = 10.0,
    min_train_events: Annotated[int, typer.Option("--min-train-events", min=1)] = 35,
    min_train_symbols: Annotated[int, typer.Option("--min-train-symbols", min=1)] = 4,
    min_train_months: Annotated[int, typer.Option("--min-train-months", min=1)] = 4,
    min_total_trades: Annotated[int, typer.Option("--min-total-trades", min=1)] = 30,
    max_single_symbol_share: Annotated[
        float,
        typer.Option("--max-single-symbol-share", min=0.0, max=1.0),
    ] = 0.50,
    max_single_session_share: Annotated[
        float,
        typer.Option("--max-single-session-share", min=0.0, max=1.0),
    ] = 0.20,
    max_single_month_share: Annotated[
        float,
        typer.Option("--max-single-month-share", min=0.0, max=1.0),
    ] = 0.50,
    random_iterations: Annotated[int, typer.Option("--random-iterations", min=1)] = 100,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 2701,
) -> None:
    """Replay selected exhaustion-extension filters with prior-only exit selection."""

    from stocker_research.exhaustion_extension_exit_replay_v0 import (
        ExhaustionExtensionExitReplayConfig,
        run_exhaustion_extension_exit_replay_lab,
    )

    parsed_months = tuple(part.strip() for part in replay_months.split(",") if part.strip())
    parsed_stop_models = tuple(part.strip() for part in stop_models.split(",") if part.strip())
    parsed_targets = tuple(float(part.strip()) for part in target_r_multiples.split(",") if part)
    if not parsed_months:
        raise typer.BadParameter("Supply at least one replay month.")
    if not parsed_stop_models:
        raise typer.BadParameter("Supply at least one stop model.")
    if not parsed_targets:
        raise typer.BadParameter("Supply at least one target R multiple.")

    result = run_exhaustion_extension_exit_replay_lab(
        input_exhaustion_event_dir=input_exhaustion_event_dir,
        input_filter_report_dir=input_filter_report_dir,
        output_dir=output_dir,
        config=ExhaustionExtensionExitReplayConfig(
            replay_months=parsed_months,
            stop_models=parsed_stop_models,
            target_r_multiples=parsed_targets,
            cost_bps=cost_bps,
            min_train_events=min_train_events,
            min_train_symbols=min_train_symbols,
            min_train_months=min_train_months,
            min_total_trades=min_total_trades,
            max_single_symbol_share=max_single_symbol_share,
            max_single_session_share=max_single_session_share,
            max_single_month_share=max_single_month_share,
            random_iterations=random_iterations,
            random_seed=random_seed,
        ),
    )
    console.print(
        {
            "output_name": "exhaustion_extension_exit_replay_v0",
            "run_id": result.run_id,
            "input_exhaustion_event_dir": str(result.input_exhaustion_event_dir),
            "input_filter_report_dir": str(result.input_filter_report_dir),
            "output_dir": str(result.output_dir),
            "summary_json_path": str(result.summary_json_path),
            "summary_markdown_path": str(result.summary_markdown_path),
            "decision_json_path": str(result.decision_json_path),
            "trades_csv_path": str(result.trades_csv_path),
            "decision": result.decision,
            "trade_count": result.trade_count,
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
