"""Placeholder for a future EODHD data adapter.

Expected future config fields:
- provider: eodhd
- api_key_env: environment variable name containing the API key
- base_url: EODHD API base URL
- timeout_seconds: HTTP timeout
- adjusted: whether to request adjusted data

Expected output schema:
- The adapter must return the canonical Stocker OHLCV schema from
  `stocker_data.schema`, with timezone-aware timestamps.

This module intentionally performs no network calls and contains no API keys.
Future commands must explicitly opt into vendor downloads.
"""

from stocker_data.schema import CANONICAL_COLUMNS, OPTIONAL_COLUMNS


def expected_output_columns() -> tuple[str, ...]:
    """Return the canonical columns expected from a future EODHD adapter."""

    return CANONICAL_COLUMNS + OPTIONAL_COLUMNS


def fetch_not_implemented() -> None:
    """Raise a clear error until EODHD ingestion is deliberately implemented."""

    raise NotImplementedError("EODHD ingestion is not implemented in Stage 2")
