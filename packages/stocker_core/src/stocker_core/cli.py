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
