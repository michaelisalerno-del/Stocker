"""Parquet storage helpers for local research datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stocker_data.schema import CANONICAL_COLUMNS

DataLayer = Literal["raw", "processed", "features", "reports"]


@dataclass(frozen=True)
class DatasetKey:
    """Storage identity for one market-data dataset."""

    source: str
    instrument_type: str
    symbol: str
    timeframe: str

    def parts(self) -> tuple[str, str, str, str]:
        """Return partition directory names."""

        return (
            f"source={self.source}",
            f"instrument_type={self.instrument_type}",
            f"symbol={self.symbol.upper()}",
            f"timeframe={self.timeframe}",
        )

    def to_dict(self) -> dict[str, str]:
        """Return JSON-serializable metadata."""

        return asdict(self)


@dataclass(frozen=True)
class DatasetMetadata:
    """Basic dataset metadata for catalog and report output."""

    key: DatasetKey
    path: Path
    row_count: int
    min_timestamp: str | None
    max_timestamp: str | None
    missing_fields: list[str]

    @property
    def source(self) -> str:
        """Return dataset source."""

        return self.key.source

    @property
    def instrument_type(self) -> str:
        """Return dataset instrument type."""

        return self.key.instrument_type

    @property
    def symbol(self) -> str:
        """Return dataset symbol."""

        return self.key.symbol

    @property
    def timeframe(self) -> str:
        """Return dataset timeframe."""

        return self.key.timeframe

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""

        return {
            **self.key.to_dict(),
            "path": str(self.path),
            "row_count": self.row_count,
            "min_timestamp": self.min_timestamp,
            "max_timestamp": self.max_timestamp,
            "missing_fields": self.missing_fields,
        }


def resolve_data_path(data_dir: str | Path, *parts: str | Path) -> Path:
    """Resolve a dataset path under the configured data directory."""

    base = Path(data_dir).expanduser()
    return base.joinpath(*parts)


def dataset_path(
    key: DatasetKey,
    *,
    data_dir: str | Path = "data",
    layer: DataLayer = "processed",
    filename: str = "data.parquet",
) -> Path:
    """Return the canonical partitioned Parquet path for a dataset."""

    return resolve_data_path(data_dir, layer, *key.parts(), filename)


def dataset_exists(path: str | Path) -> bool:
    """Return whether a dataset path exists on disk."""

    return Path(path).exists()


def write_parquet(frame: Any, path: str | Path) -> Path:
    """Write a pandas or polars DataFrame to Parquet, creating parent directories."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(frame, "write_parquet"):
        frame.write_parquet(output_path)
    elif hasattr(frame, "to_parquet"):
        frame.to_parquet(output_path, index=False)
    else:
        pd.DataFrame(frame).to_parquet(output_path, index=False)
    return output_path


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet dataset into a pandas DataFrame."""

    return pd.read_parquet(path)


def load_dataset(
    key: DatasetKey,
    *,
    data_dir: str | Path = "data",
    layer: DataLayer = "processed",
) -> pd.DataFrame:
    """Load a dataset by key from the canonical storage layout."""

    return read_parquet(dataset_path(key, data_dir=data_dir, layer=layer))


def _key_from_path(path: Path, data_dir: Path, layer: DataLayer) -> DatasetKey | None:
    try:
        relative = path.relative_to(data_dir / layer)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 5 or parts[-1] != "data.parquet":
        return None
    values: dict[str, str] = {}
    for part in parts[:-1]:
        if "=" not in part:
            return None
        name, value = part.split("=", 1)
        values[name] = value
    required = {"source", "instrument_type", "symbol", "timeframe"}
    if not required.issubset(values):
        return None
    return DatasetKey(
        source=values["source"],
        instrument_type=values["instrument_type"],
        symbol=values["symbol"],
        timeframe=values["timeframe"],
    )


def list_datasets(
    *,
    data_dir: str | Path = "data",
    layer: DataLayer = "processed",
) -> list[DatasetMetadata]:
    """List datasets available in the canonical storage layout."""

    root = Path(data_dir).expanduser() / layer
    if not root.exists():
        return []

    datasets: list[DatasetMetadata] = []
    for path in sorted(root.glob("source=*/instrument_type=*/symbol=*/timeframe=*/data.parquet")):
        key = _key_from_path(path, Path(data_dir).expanduser(), layer)
        if key is None:
            continue
        datasets.append(dataset_metadata(key, data_dir=data_dir, layer=layer))
    return datasets


def dataset_metadata(
    key: DatasetKey,
    *,
    data_dir: str | Path = "data",
    layer: DataLayer = "processed",
) -> DatasetMetadata:
    """Return row count, date range, and missing canonical fields for a dataset."""

    path = dataset_path(key, data_dir=data_dir, layer=layer)
    frame = read_parquet(path)
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    if "timestamp" in frame and not frame.empty:
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        min_value = timestamps.min()
        max_value = timestamps.max()
        min_timestamp = None if pd.isna(min_value) else str(min_value)
        max_timestamp = None if pd.isna(max_value) else str(max_value)
    return DatasetMetadata(
        key=key,
        path=path,
        row_count=int(len(frame)),
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        missing_fields=missing,
    )
