"""Market-data ingestion placeholders.

No data vendors are connected in the bootstrap. Future ingest code should write raw,
immutable datasets before any cleaning or feature generation happens.
"""

from pathlib import Path


def ingest_not_configured_message(data_dir: Path) -> str:
    """Return a clear message for commands that try to ingest before a source exists."""

    return f"No market data ingestion source is configured. Raw data directory: {data_dir}"
