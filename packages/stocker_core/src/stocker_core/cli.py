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
) -> None:
    """Run placeholder data validation checks."""

    if dataset is None:
        console.print("No dataset supplied. Data validators are importable.")
        return
    from stocker_data.storage import dataset_exists

    if not dataset_exists(dataset):
        raise typer.BadParameter(f"Dataset does not exist: {dataset}")
    console.print(f"Dataset exists: {dataset}")


@research_app.command("baseline")
def research_baseline(
    dataset: Annotated[
        Path | None, typer.Argument(help="Optional OHLC Parquet dataset path.")
    ] = None,
) -> None:
    """Run the minimal OHLC baseline summary."""

    if dataset is None:
        console.print("No dataset supplied. Baseline command is ready for OHLC Parquet input.")
        return
    from stocker_data.storage import read_parquet
    from stocker_research.baselines import ohlc_baseline_summary

    summary = ohlc_baseline_summary(read_parquet(dataset))
    console.print(summary)


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
