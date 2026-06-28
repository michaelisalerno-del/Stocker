"""Parquet storage helpers for local research datasets."""

from pathlib import Path
from typing import Any

import pandas as pd


def resolve_data_path(data_dir: str | Path, *parts: str | Path) -> Path:
    """Resolve a dataset path under the configured data directory."""

    base = Path(data_dir).expanduser()
    return base.joinpath(*parts)


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
