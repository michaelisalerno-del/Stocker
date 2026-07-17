"""Deterministic research artifact writers and safety checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_deterministic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def write_deterministic_csv(frame: pd.DataFrame, path: Path, *, sort_by: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    ordered.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_deterministic_parquet(frame: pd.DataFrame, path: Path, *, sort_by: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    table = pa.Table.from_pandas(ordered, preserve_index=False)
    table = table.replace_schema_metadata(None)
    pq.write_table(  # type: ignore[no-untyped-call]
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )


def assert_research_only_paths(paths: list[Path]) -> None:
    forbidden = ("stocker_execution", "broker", "orders", "positions", "deployment", "apps/")
    invalid = [
        str(path) for path in paths if any(token in str(path).lower() for token in forbidden)
    ]
    if invalid:
        raise ValueError(f"research-only boundary violation: {invalid}")
