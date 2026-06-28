"""Dry-run server execution entry point."""

from pathlib import Path

import typer

from stocker_core.config import load_server_config
from stocker_core.logging import configure_logging

app = typer.Typer(no_args_is_help=False)


@app.command()
def main(config: Path = Path("configs/server.example.yaml")) -> None:
    logger = configure_logging(json_logs=True)
    config_path = config
    if not config_path.exists():
        repo_root = Path(__file__).resolve().parents[3]
        config_path = repo_root / config
    loaded = load_server_config(config_path)
    logger.info(
        "executor_dry_run",
        mode=loaded.server.mode,
        broker=loaded.server.broker.provider,
        trading_enabled=loaded.risk.trading_enabled,
    )


if __name__ == "__main__":
    app()
