"""Run a lightweight local research lab check."""

from pathlib import Path

from stocker_core.config import load_research_config
from stocker_core.logging import configure_logging


def main() -> None:
    logger = configure_logging()
    repo_root = Path(__file__).resolve().parents[3]
    config = load_research_config(repo_root / "configs/research.example.yaml")
    logger.info(
        "research_lab_ready", data_dir=str(config.data.data_dir), timezone=config.data.timezone
    )


if __name__ == "__main__":
    main()
