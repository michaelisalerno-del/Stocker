"""Simple local dataset catalog and DuckDB query helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stocker_data.storage import DataLayer, DatasetMetadata, list_datasets


def catalog_path(data_dir: str | Path = "data") -> Path:
    """Return the JSON catalog path."""

    return Path(data_dir).expanduser() / "catalog.json"


def scan_catalog(
    *,
    data_dir: str | Path = "data",
    layer: DataLayer = "processed",
) -> list[DatasetMetadata]:
    """Scan Parquet storage and return dataset metadata."""

    return list_datasets(data_dir=data_dir, layer=layer)


def write_catalog(
    *,
    data_dir: str | Path = "data",
    layer: DataLayer = "processed",
) -> Path:
    """Write a simple JSON catalog for local discovery."""

    entries = scan_catalog(data_dir=data_dir, layer=layer)
    path = catalog_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"datasets": [entry.to_dict() for entry in entries]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def query_dataset(path: str | Path, sql: str | None = None) -> pd.DataFrame:
    """Run a DuckDB query against a Parquet dataset."""

    import duckdb

    dataset = str(Path(path).expanduser())
    escaped_dataset = dataset.replace("'", "''")
    query = sql or "select * from dataset limit 10"
    if sql is not None and " from " not in f" {query.lower()} ":
        query = f"{query} from dataset"
    with duckdb.connect(database=":memory:") as connection:
        connection.execute(
            f"create view dataset as select * from read_parquet('{escaped_dataset}')"
        )
        return connection.execute(query).fetchdf()
