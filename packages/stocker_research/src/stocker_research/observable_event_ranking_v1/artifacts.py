"""Canonical scientific artifact writing and exact-directory comparison."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stocker_research.observable_event_ranking_v1.contract import canonical_json_bytes


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactBinding:
    """Identity shared by every artifact through content or manifest binding."""

    git_sha: str
    branch: str
    contract_hash: str
    implementation_hash: str
    data_snapshot_hash: str
    universe_hash: str
    sector_map_hash: str
    run_id: str
    random_seeds: dict[str, int]
    dependency_versions: dict[str, str]
    safety: dict[str, bool | str]

    def to_dict(self) -> dict[str, Any]:
        """Return canonical-JSON-compatible fields."""

        return asdict(self)


@dataclass(frozen=True)
class ArtifactComparison:
    """File-by-file exact comparison result."""

    identical: bool
    compared_files: tuple[str, ...]
    excluded_files: tuple[str, ...]
    mismatches: tuple[str, ...]


class ArtifactWriter:
    """Write stable JSON, CSV, and Parquet artifacts."""

    def __init__(self, output_dir: Path, binding: ArtifactBinding) -> None:
        self.output_dir = output_dir
        self.binding = binding
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def json(self, name: str, payload: Any) -> Path:
        """Write canonical JSON with a final newline."""

        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(payload) + b"\n")
        return path

    def csv(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        columns: tuple[str, ...],
        sort_by: tuple[str, ...] = (),
    ) -> Path:
        """Write stable columns, row ordering, newlines, and float representation."""

        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"CSV columns missing: {missing}")
        ordered = frame.loc[:, list(columns)].copy()
        if sort_by and not ordered.empty:
            ordered = ordered.sort_values(list(sort_by), kind="mergesort", na_position="last")
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            for row in ordered.itertuples(index=False, name=None):
                writer.writerow(["" if pd.isna(value) else value for value in row])
        return path

    def parquet(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        columns: tuple[str, ...],
        sort_by: tuple[str, ...] = (),
    ) -> Path:
        """Write deterministic Parquet with the full scientific binding in schema metadata."""

        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Parquet columns missing: {missing}")
        ordered = frame.loc[:, list(columns)].copy()
        if sort_by and not ordered.empty:
            ordered = ordered.sort_values(list(sort_by), kind="mergesort", na_position="last")
        ordered = ordered.reset_index(drop=True)
        table = pa.Table.from_pandas(ordered, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"stocker_artifact_binding"] = canonical_json_bytes(self.binding.to_dict())
        table = table.replace_schema_metadata(metadata)
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(  # type: ignore[no-untyped-call]
            table,
            path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            version="2.6",
        )
        return path

    def manifest(self) -> Path:
        """Bind every non-self-referential file to one identity manifest."""

        entries = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.name == "artifact_manifest.json":
                continue
            entries.append(
                {
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        return self.json(
            "artifact_manifest.json",
            {
                "manifest_version": "observable_event_ranking_v1_artifacts",
                "binding": self.binding.to_dict(),
                "artifact_count": len(entries),
                "artifacts": entries,
            },
        )


def compare_artifact_directories(
    primary_dir: Path,
    exact_dir: Path,
    *,
    excluded: tuple[str, ...] = (),
) -> ArtifactComparison:
    """Compare all files by relative name and SHA-256."""

    excluded_set = set(excluded)

    def hashes(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.relative_to(root).as_posix() not in excluded_set
        }

    primary = hashes(primary_dir)
    exact = hashes(exact_dir)
    names = sorted(set(primary) | set(exact))
    mismatches = tuple(
        name
        for name in names
        if name not in primary or name not in exact or primary[name] != exact[name]
    )
    return ArtifactComparison(
        identical=not mismatches,
        compared_files=tuple(names),
        excluded_files=tuple(sorted(excluded_set)),
        mismatches=mismatches,
    )
