"""Deterministic artifact I/O for the right-censored regime repair V2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
    "part_b_interaction_scoring_enabled": False,
    "semantic_dictionary_promotion_enabled": False,
}

DETAILED_IDENTITY_DEFAULTS: dict[str, object] = {
    "symbol": "not_applicable",
    "session": "not_applicable",
    "segment_id": "not_applicable",
    "timestamp": "not_applicable",
    "state": -1,
    "age": -1,
    "source_artifact": "not_applicable",
    "source_hash": "not_applicable",
}


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    run_id: str
    git_sha: str
    contract_hash: str
    data_snapshot_hash: str
    panel_hash: str
    implementation_source_hash: str
    state_model_version: str
    state_model_hash: str
    model_lineage: str

    def as_dict(self) -> dict[str, object]:
        return dict(asdict(self))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


class ArtifactWriter:
    """Write all tables with one immutable research identity and safety surface."""

    def __init__(self, output_dir: Path, identity: ArtifactIdentity) -> None:
        self.output_dir = output_dir
        self.identity = identity
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def enriched(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for key, value in self.identity.as_dict().items():
            if key not in output:
                output[key] = value
        for key, value in DETAILED_IDENTITY_DEFAULTS.items():
            if key not in output:
                output[key] = value
        for key, value in SAFETY_FLAGS.items():
            if key not in output:
                output[key] = value
        return output

    def frame(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        output = self.enriched(frame)
        if path.suffix == ".csv":
            output.to_csv(
                path,
                index=False,
                float_format="%.12g",
                lineterminator="\n",
            )
        elif path.suffix == ".parquet":
            table = pa.Table.from_pandas(output, preserve_index=False)
            pq.write_table(
                table,
                path,
                compression="zstd",
                compression_level=9,
                use_dictionary=True,
                write_statistics=True,
                data_page_version="2.0",
                version="2.6",
            )
        else:
            raise ValueError(f"unsupported frame artifact: {path}")
        return path

    def json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = {
            **self.identity.as_dict(),
            **payload,
            **SAFETY_FLAGS,
        }
        path.write_bytes(
            json.dumps(
                merged,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            ).encode("utf-8")
            + b"\n"
        )
        return path

    def copy_pre_artifact(self, source: Path) -> Path:
        target = self.output_dir / source.name
        if target.resolve() != source.resolve():
            target.write_bytes(source.read_bytes())
        return target


def artifact_hashes(directory: Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    skipped = excluded or set()
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.relative_to(directory).as_posix() not in skipped
    }


def write_artifact_manifest(
    writer: ArtifactWriter,
    *,
    manifest_version: str,
    excluded: set[str],
) -> dict[str, Any]:
    artifacts = artifact_hashes(writer.output_dir, excluded=excluded | {"artifact_manifest.json"})
    manifest_hash = sha256_bytes(canonical_json_bytes(artifacts))
    payload = {
        "manifest_version": manifest_version,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "manifest_hash": manifest_hash,
        "excluded_self_referential_files": sorted(excluded | {"artifact_manifest.json"}),
    }
    writer.json("artifact_manifest.json", payload)
    return payload


def compare_artifact_directories(
    primary: Path,
    exact: Path,
    *,
    excluded: set[str],
) -> dict[str, Any]:
    left = artifact_hashes(primary, excluded=excluded)
    right = artifact_hashes(exact, excluded=excluded)
    all_names = sorted(set(left) | set(right))
    mismatched = [
        name for name in all_names if name in left and name in right and left[name] != right[name]
    ]
    missing_or_extra = [name for name in all_names if name not in left or name not in right]
    return {
        "byte_identical": not mismatched and not missing_or_extra,
        "compared_artifact_count": len(set(left) & set(right)),
        "mismatched_files": mismatched,
        "missing_or_extra_files": missing_or_extra,
        "primary_hash": sha256_bytes(canonical_json_bytes(left)),
        "exact_hash": sha256_bytes(canonical_json_bytes(right)),
        "excluded_files": sorted(excluded),
    }


__all__ = [
    "ArtifactIdentity",
    "ArtifactWriter",
    "DETAILED_IDENTITY_DEFAULTS",
    "SAFETY_FLAGS",
    "artifact_hashes",
    "canonical_json_bytes",
    "compare_artifact_directories",
    "sha256_bytes",
    "sha256_file",
    "write_artifact_manifest",
]
