"""Command-line interface for Stocker."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from stocker_core.config import load_server_config

console = Console()
app = typer.Typer(no_args_is_help=True, help="Stocker research and execution utilities.")
data_app = typer.Typer(no_args_is_help=True, help="Data utilities.")
research_app = typer.Typer(no_args_is_help=True, help="Research utilities.")
server_app = typer.Typer(no_args_is_help=True, help="Server utilities.")
app.add_typer(data_app, name="data")
app.add_typer(research_app, name="research")
app.add_typer(server_app, name="server")


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
) -> None:
    """Create a dataset audit report."""

    from stocker_data.audit import create_audit_report

    report = create_audit_report(
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        instrument_type=instrument_type,
    )
    console.print(
        {
            "audit": str(report.markdown_path),
            "json": str(report.json_path),
            "passed": report.passed,
        }
    )


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
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
) -> None:
    """Run a disciplined research experiment from a written hypothesis."""

    from stocker_research.experiments import run_research_experiment

    result = run_research_experiment(
        hypothesis_path=hypothesis,
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        instrument_type=instrument_type,
    )
    console.print(
        {
            "experiment_id": result.experiment_id,
            "classification": result.classification,
            "report": str(result.markdown_path),
            "json": str(result.json_path),
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
