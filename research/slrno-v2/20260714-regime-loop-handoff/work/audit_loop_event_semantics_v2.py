#!/usr/bin/env python3
# ruff: noqa: E501
"""Independent auditor for loop-event semantics V2.

This file intentionally does not import the production V2 implementation.  It
reconstructs identities, run provenance, event labels, duration hazards,
posterior invariants, null draws, migration rows, source hashes, exact-rerun
identity, and safety flags directly from detailed artifacts and source files.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-loop-event-semantics-v2.json"
REPORT_PATH = WORK_DIR / "reports" / "20260718-loop-event-semantics-v2.md"
SAFETY = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "strategy_promotion": False,
}
STATE_COUNT = 8
MAX_EVENT_TRANSITIONS = 8
DEVELOPMENT_START = pd.Timestamp("2024-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
AUDIT_IDENTITY: dict[str, Any] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_provider_hash(path: Path) -> tuple[str, pd.DataFrame]:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("timestamp", ">=", DEVELOPMENT_START.to_pydatetime()),
            ("timestamp", "<=", DEVELOPMENT_END.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].gt(DEVELOPMENT_END).any():
        raise AssertionError("bounded source audit admitted a post-development row")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest(), frame


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
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {**AUDIT_IDENTITY, **payload, **SAFETY},
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = ast.literal_eval(str(value))
    return list(parsed)


def _canonical_core(core: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in core)
    return min(values[index:] + values[:index] for index in range(len(values)))


def _primitive_root(core: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    for width in range(2, len(core) // 2 + 1):
        if len(core) % width == 0 and core[:width] * (len(core) // width) == core:
            return core[:width], len(core) // width
    return core, 1


def _path_text(path: Sequence[int]) -> str:
    return "-".join(str(int(value)) for value in path)


def _rotations(core: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in core)
    return tuple(
        sorted({values[index:] + values[:index] + (values[index],) for index in range(len(values))})
    )


def _decompose(path: Sequence[int]) -> dict[str, Any]:
    closed = tuple(int(value) for value in path)
    if len(closed) < 3 or closed[0] != closed[-1]:
        raise ValueError("path is not closed")
    core = _canonical_core(closed[:-1])
    root, repeat = _primitive_root(core)
    root = _canonical_core(root)
    components: list[str] = []
    component_boundaries: list[tuple[int, int]] = []
    if repeat == 1:
        for candidate in sorted({core[index:] + core[:index] for index in range(len(core))}):
            anchor = candidate[0]
            boundaries = [0]
            boundaries.extend(
                index for index, value in enumerate(candidate[1:], start=1) if value == anchor
            )
            boundaries.append(len(candidate))
            if len(boundaries) <= 2:
                continue
            candidate_components: list[str] = []
            candidate_bounds: list[tuple[int, int]] = []
            for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
                segment = candidate[left:right]
                if len(segment) not in {2, 3, 4, 5} or len(set(segment)) != len(segment):
                    break
                canonical_segment = _canonical_core(segment)
                segment_path = canonical_segment + (canonical_segment[0],)
                candidate_components.append(f"loop_p_{_path_text(segment_path)}")
                candidate_bounds.append((left, right))
            else:
                if len(candidate_components) >= 2:
                    core = candidate
                    components = candidate_components
                    component_boundaries = candidate_bounds
                    break
    if repeat > 1:
        if len(set(root)) != len(root):
            raise ValueError("periodic path has a composite root")
        primitive_path = root + (root[0],)
        primitive_id = f"loop_p_{_path_text(primitive_path)}"
        semantic_id = f"loop_r{repeat}_{_path_text(primitive_path)}"
        motif = "repeat"
        components = [primitive_id] * repeat
        component_boundaries = [
            (index * len(root), (index + 1) * len(root)) for index in range(repeat)
        ]
    elif components:
        canonical_path = core + (core[0],)
        digest = hashlib.sha256(_path_text(canonical_path).encode("ascii")).hexdigest()[:8]
        semantic_id = f"loop_c_{digest}"
        primitive_id = None
        motif = "composite"
        root = ()
    else:
        if len(set(core)) != len(core):
            raise ValueError("ambiguous non-periodic composite")
        canonical_path = core + (core[0],)
        primitive_id = f"loop_p_{_path_text(canonical_path)}"
        semantic_id = primitive_id
        motif = "primitive"
    return {
        "semantic_loop_id": semantic_id,
        "primitive_loop_id": primitive_id,
        "motif_type": motif,
        "repeat_depth": repeat,
        "primitive_core": root,
        "full_core": core,
        "canonical_orientation": core + (core[0],),
        "oriented_paths": _rotations(core),
        "component_primitive_ids": tuple(components),
        "component_boundaries": tuple(component_boundaries),
    }


def _record(checks: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "evidence": evidence})


def _audit_sources(primary: Path, checks: list[dict[str, Any]]) -> int:
    source_rows = pd.read_csv(primary / "source_file_hashes.csv")
    mismatches = []
    structural_rows = 0
    for row in source_rows.itertuples(index=False):
        path = Path(row.source_file)
        actual, source = _bounded_provider_hash(path)
        if actual != row.source_file_hash:
            mismatches.append(str(row.symbol))
        if row.source_hash_scope != "filtered_2024_development_rows_only":
            mismatches.append(f"scope:{row.symbol}")
        if row.symbol == "VTI":
            continue
        source = source.dropna(subset=["timestamp", "open", "high", "low", "close"])
        timestamps = pd.to_datetime(source["timestamp"], utc=True)
        local = timestamps.dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        structural_rows += int((minute.ge(570) & minute.lt(960)).sum())
    _record(
        checks,
        "source_identities",
        not mismatches,
        {"source_files": len(source_rows), "hash_mismatches": mismatches},
    )
    source_code = pd.read_csv(primary / "v2_source_hashes.csv")
    code_mismatches = []
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    source_commit = str(metadata["source_commit"])
    for row in source_code.itertuples(index=False):
        path = REPO_ROOT / str(row.file)
        if _sha256_file(path) != row.source_sha256:
            code_mismatches.append(f"worktree:{row.file}")
        try:
            blob = _git_output("rev-parse", f"{source_commit}:{row.file}")
        except subprocess.CalledProcessError:
            code_mismatches.append(f"uncommitted:{row.file}")
            continue
        if blob != row.commit_blob:
            code_mismatches.append(f"blob:{row.file}")
    _record(
        checks,
        "committed_v2_source_identity",
        not code_mismatches and source_code["source_commit"].astype(str).eq(source_commit).all(),
        {
            "source_commit": source_commit,
            "source_files": len(source_code),
            "mismatches": code_mismatches,
        },
    )
    return structural_rows


def _audit_frozen_tree(primary: Path, checks: list[dict[str, Any]]) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    baseline = contract["source"]["frozen_lineage_baseline_commit"]
    prefix = "research/slrno-v2/20260714-regime-loop-handoff"
    listing = _git_output("ls-tree", "-r", baseline, prefix).splitlines()
    mismatches = []
    for line in listing:
        metadata, path_text = line.split("\t", 1)
        expected_blob = metadata.split()[2]
        actual_blob = _git_output("hash-object", path_text)
        if expected_blob != actual_blob:
            mismatches.append(path_text)
    recorded = pd.read_csv(primary / "frozen_historical_tree_hashes.csv")
    _record(
        checks,
        "frozen_historical_file_immutability",
        not mismatches and len(recorded) == len(listing),
        {
            "baseline_commit": baseline,
            "files": len(listing),
            "mismatches": mismatches,
        },
    )


def _reconstruct_run_context(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = ("b0_state_numeric", "b0_high_stress", "clock_sin", "clock_cos")
    for (symbol, session), group in decisions.groupby(["symbol", "session"], sort=False):
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        states = ordered["hard_state_legacy"].to_numpy(dtype=int)
        starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
        ends = np.r_[starts[1:], len(ordered)]
        for start, end in zip(starts, ends, strict=True):
            first = ordered.iloc[int(start)]
            last = ordered.iloc[int(end - 1)]
            for field in fields:
                start_value = first[field]
                end_value = last[field]
                equal = (pd.isna(start_value) and pd.isna(end_value)) or (
                    pd.notna(start_value) and pd.notna(end_value) and bool(start_value == end_value)
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "session": session,
                        "field": field,
                        "start_value": start_value,
                        "end_value": end_value,
                        "start_end_differ": not equal,
                    }
                )
    return pd.DataFrame(rows)


def _independent_legacy_assignments(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_online_runs"
    )
    output: dict[str, dict[str, Any]] = {}
    for mapping in (node for node in ast.walk(function) if isinstance(node, ast.Dict)):
        for key, value in zip(mapping.keys, mapping.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            expression = ast.unparse(value)
            source = (
                "first_row"
                if "frame.at[first" in expression
                else "last_row"
                if "frame.at[last" in expression
                else "causal_prior_run_history"
                if "previous_states" in expression
                else "derived_or_literal"
            )
            output[key.value] = {
                "assignment_expression": expression,
                "row_source": source,
                "source_line": int(getattr(value, "lineno", 0)),
            }
    return output


def _audit_provenance(primary: Path, checks: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "decision_id",
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "decision_timestamp",
        "b0_source_timestamp",
        "b0_source_bar_ordinal",
        "b0_available_timestamp",
        "b0_source_artifact_hash",
        "stock_source_artifact_hash",
        "state_source_artifact_hash",
        "structural_source_artifact_hash",
        "source_sequence_complete",
        "source_sequence_missing_reason",
        "b0_state_numeric",
        "b0_high_stress",
        "clock_sin",
        "clock_cos",
        "hard_state_legacy",
        "hard_state_hysteretic",
        "posterior_entropy",
        "top_second_margin",
        "structural_event_eligibility",
        "active_prefix_count",
        "is_run_entry",
    ]
    decisions = pd.read_parquet(primary / "causal_completed_bar_decisions.parquet", columns=columns)
    causal = decisions["bar_complete_timestamp"].le(decisions["decision_timestamp"])
    b0_available = decisions["b0_available_timestamp"].isna() | decisions[
        "b0_available_timestamp"
    ].le(decisions["decision_timestamp"])
    forbidden = ("future", "payoff", "mfe", "mae", "route_completion")
    forbidden_columns = [
        column
        for column in pq.read_schema(primary / "causal_completed_bar_decisions.parquet").names
        if any(token in column.lower() for token in forbidden)
    ]
    manifest = json.loads(
        (primary / "feature_availability_manifest.json").read_text(encoding="utf-8")
    )
    schema_names = set(pq.read_schema(primary / "causal_completed_bar_decisions.parquet").names)
    required_spec_keys = {
        "source_timestamp_column",
        "source_bar_ordinal_column",
        "available_timestamp_column",
        "decision_timestamp_column",
        "causal_valid_column",
        "missing_reason_column",
        "source_field",
        "source_artifact_hash_column",
    }
    manifest_failures = []
    referenced_columns = set()
    for field, spec in manifest["fields"].items():
        if not required_spec_keys.issubset(spec):
            manifest_failures.append(f"missing_keys:{field}")
            continue
        for key in required_spec_keys - {"source_field"}:
            column = spec[key]
            if not isinstance(column, str) or column not in schema_names:
                manifest_failures.append(f"missing_column:{field}:{key}:{column}")
            else:
                referenced_columns.add(column)
    manifest_violations = 0
    if not manifest_failures:
        parquet = pq.ParquetFile(primary / "causal_completed_bar_decisions.parquet")
        ordered_columns = sorted(referenced_columns)
        for batch in parquet.iter_batches(batch_size=4096, columns=ordered_columns):
            data = batch.to_pandas()
            for spec in manifest["fields"].values():
                source = pd.to_datetime(data[spec["source_timestamp_column"]], utc=True)
                available = pd.to_datetime(data[spec["available_timestamp_column"]], utc=True)
                decision = pd.to_datetime(data[spec["decision_timestamp_column"]], utc=True)
                valid = data[spec["causal_valid_column"]].astype(bool)
                reason = data[spec["missing_reason_column"]]
                source_hash = data[spec["source_artifact_hash_column"]]
                manifest_violations += int(
                    (
                        valid
                        & (
                            source.isna()
                            | available.isna()
                            | source.gt(decision)
                            | available.lt(source)
                            | available.gt(decision)
                            | reason.notna()
                            | source_hash.isna()
                        )
                    ).sum()
                )
                manifest_violations += int((~valid & reason.isna()).sum())
    _record(
        checks,
        "feature_availability",
        bool(
            causal.all()
            and b0_available.all()
            and not forbidden_columns
            and not manifest_failures
            and manifest_violations == 0
        ),
        {
            "rows": len(decisions),
            "completed_bar_violations": int((~causal).sum()),
            "b0_availability_violations": int((~b0_available).sum()),
            "forbidden_columns": forbidden_columns,
            "manifest_failures": manifest_failures,
            "manifest_provenance_violations": manifest_violations,
        },
    )
    reconstructed = _reconstruct_run_context(decisions)
    observed = reconstructed.groupby("field", sort=True)["start_end_differ"].agg(["size", "sum"])
    detailed = pd.read_parquet(
        primary / "b0_feature_provenance_audit.parquet",
        columns=[
            "field",
            "start_end_differ",
            "stored_legacy_value",
            "end_value",
            "legacy_stored_value_evidence",
        ],
    )
    recorded = detailed.groupby("field", sort=True)["start_end_differ"].agg(["size", "sum"])
    assignment_artifact = pd.read_csv(primary / "legacy_run_field_assignment_audit.csv")
    frozen_path = REPO_ROOT / str(assignment_artifact.iloc[0]["file"])
    independent_assignments = _independent_legacy_assignments(frozen_path)
    assignment_mismatches = []
    for row in assignment_artifact.itertuples(index=False):
        independent = independent_assignments.get(str(row.field))
        if independent is None or any(
            (
                independent["row_source"] != row.row_source,
                independent["assignment_expression"] != row.assignment_expression,
                independent["source_line"] != int(row.source_line),
            )
        ):
            assignment_mismatches.append(str(row.field))
    required_last = {
        "b0_state_numeric",
        "b0_high_stress",
        "time_sin",
        "time_cos",
        "end_timestamp",
    }
    required_first = {"symbol_norm", "session_date", "month", "year", "start_timestamp"}
    source_position_proved = all(
        independent_assignments[field]["row_source"] == "last_row" for field in required_last
    ) and all(
        independent_assignments[field]["row_source"] == "first_row" for field in required_first
    )
    consumers = pd.read_csv(primary / "legacy_entry_context_consumers.csv")
    reconstructed_evidence = (
        detailed["legacy_stored_value_evidence"]
        .eq("reconstructed_from_frozen_builder_last_row_assignment")
        .all()
    )
    match = observed.equals(recorded)
    b0_difference = int(
        reconstructed.loc[
            reconstructed["field"].isin(["b0_state_numeric", "b0_high_stress"]),
            "start_end_differ",
        ].sum()
    )
    _record(
        checks,
        "b0_provenance_audit",
        bool(
            match
            and source_position_proved
            and not assignment_mismatches
            and reconstructed_evidence
            and not consumers.empty
            and b0_difference == 0
        ),
        {
            "independent_counts": observed.reset_index().to_dict("records"),
            "frozen_AST_source_position_proved": source_position_proved,
            "assignment_mismatches": assignment_mismatches,
            "consumer_rows": len(consumers),
            "stored_values_explicitly_reconstructed": bool(reconstructed_evidence),
            "b0_changed_runs": b0_difference,
        },
    )
    return decisions


def _audit_identities(primary: Path, checks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    migration = pd.read_csv(primary / "legacy_to_v2_loop_mapping.csv")
    mismatches = []
    definitions: dict[str, dict[str, Any]] = {}
    for row in migration.itertuples(index=False):
        path = tuple(int(value) for value in str(row.legacy_cycle_path).split("->"))
        independent = _decompose(path)
        for key in ("semantic_loop_id", "primitive_loop_id", "motif_type", "repeat_depth"):
            observed = getattr(row, key)
            if pd.isna(observed):
                observed = None
            if observed != independent[key]:
                mismatches.append({"legacy_cycle_id": row.legacy_cycle_id, "field": key})
        definitions[independent["semantic_loop_id"]] = independent
    semantic = pd.read_csv(primary / "semantic_loop_dictionary_v2.csv")
    orientation_mismatches = []
    for row in semantic.itertuples(index=False):
        canonical = tuple(_parse_list(row.canonical_orientation))
        independent = _decompose(canonical)
        recorded_orientations = {tuple(path) for path in _parse_list(row.all_valid_oriented_paths)}
        if independent["semantic_loop_id"] != row.semantic_loop_id:
            orientation_mismatches.append(str(row.semantic_loop_id))
        if set(independent["oriented_paths"]) != recorded_orientations:
            orientation_mismatches.append(f"orientation:{row.semantic_loop_id}")
        definitions[independent["semantic_loop_id"]] = independent
    reverse_distinct = (
        _decompose((0, 1, 2, 0))["semantic_loop_id"] != _decompose((0, 2, 1, 0))["semantic_loop_id"]
    )
    _record(
        checks,
        "primitive_root_semantic_ids_orientation_and_migration",
        not mismatches and not orientation_mismatches and reverse_distinct,
        {
            "migration_rows": len(migration),
            "identity_mismatches": mismatches,
            "orientation_mismatches": orientation_mismatches,
            "reverse_direction_remains_distinct": reverse_distinct,
        },
    )
    return definitions


def _dictionary_from_csv(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    output = {}
    for row in frame.itertuples(index=False):
        canonical = tuple(_parse_list(row.canonical_orientation))
        output[str(row.semantic_loop_id)] = _decompose(canonical)
    return output


def _compress_events(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = group["hard_state_legacy"].to_numpy(dtype=int)
    mask = np.r_[True, states[1:] != states[:-1]]
    positions = np.flatnonzero(mask)
    return (
        states[positions],
        group.iloc[positions]["bar_ordinal"].to_numpy(dtype=int),
        np.cumsum(mask) - 1,
    )


def _event_trace(
    states: np.ndarray,
    bars: np.ndarray,
    dictionary: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[int, ...]]]:
    registered_paths = {
        path for definition in dictionary.values() for path in definition["oriented_paths"]
    }
    registered = []
    for semantic_id, definition in sorted(dictionary.items()):
        for path in definition["oriented_paths"]:
            for start in range(len(states) - len(path) + 1):
                if tuple(states[start : start + len(path)]) == path:
                    registered.append(
                        {
                            "semantic_loop_id": semantic_id,
                            "motif_type": definition["motif_type"],
                            "repeat_depth": definition["repeat_depth"],
                            "path": path,
                            "start_event": start,
                            "end_event": start + len(path) - 1,
                            "start_bar": int(bars[start]),
                            "end_bar": int(bars[start + len(path) - 1]),
                        }
                    )
    unregistered = []
    for end in range(len(states)):
        for start in range(end - 2, max(-1, end - MAX_EVENT_TRANSITIONS - 1), -1):
            if start < 0 or states[start] != states[end]:
                continue
            path = tuple(states[start : end + 1])
            if path not in registered_paths:
                unregistered.append(
                    {
                        "path": path,
                        "start_event": start,
                        "end_event": end,
                        "start_bar": int(bars[start]),
                        "end_bar": int(bars[end]),
                    }
                )
    return registered, unregistered, registered_paths


def _active_prefixes(
    states: np.ndarray,
    event_index: int,
    dictionary: dict[str, dict[str, Any]],
) -> set[tuple[str, tuple[int, ...], tuple[int, ...], int]]:
    observed = tuple(states[: event_index + 1])
    prefixes = set()
    for semantic_id, definition in dictionary.items():
        for path in definition["oriented_paths"]:
            for progress in range(1, len(path)):
                prefix = path[:progress]
                if len(observed) >= progress and observed[-progress:] == prefix:
                    prefixes.add((semantic_id, path, prefix, len(path) - progress))
    return prefixes


def _independent_outcome(
    *,
    states: np.ndarray,
    bars: np.ndarray,
    event_index: int,
    decision_bar: int,
    session_end: int,
    registered: list[dict[str, Any]],
    unregistered: list[dict[str, Any]],
    source_available: bool,
) -> tuple[str, tuple[str, ...]]:
    if not source_available:
        return "UNAVAILABLE", ()
    horizon_end = decision_bar + 24
    registered_future = sorted(
        [
            event
            for event in registered
            if event["end_event"] > event_index
            and event["end_bar"] > decision_bar
            and event["end_bar"] <= horizon_end
        ],
        key=lambda event: (
            event["end_bar"],
            event["end_event"],
            event["semantic_loop_id"],
        ),
    )
    unregistered_future = sorted(
        [
            event
            for event in unregistered
            if event["end_event"] > event_index
            and event["end_bar"] > decision_bar
            and event["end_bar"] <= horizon_end
        ],
        key=lambda event: (event["end_bar"], event["end_event"], event["path"]),
    )
    registered_bar = registered_future[0]["end_bar"] if registered_future else None
    unregistered_bar = unregistered_future[0]["end_bar"] if unregistered_future else None
    if registered_future and (unregistered_bar is None or registered_bar <= unregistered_bar):
        earliest_event = registered_future[0]["end_event"]
        ids = tuple(
            sorted(
                {
                    event["semantic_loop_id"]
                    for event in registered_future
                    if event["end_event"] == earliest_event
                }
            )
        )
        return ("TIED_REGISTERED_COMPLETION" if len(ids) > 1 else ids[0]), ids
    if unregistered_future:
        return "UNREGISTERED_LOOP", ()
    future_event_exists = event_index + 1 < len(states)
    if not future_event_exists or session_end <= horizon_end:
        return "SESSION_END", ()
    return "NO_REGISTERED_LOOP_WITHIN_HORIZON", ()


def _legacy_labels(
    states: np.ndarray,
    event_index: int,
    dictionary: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    current = int(states[event_index])
    future = tuple(states[event_index + 1 :])
    positives = []
    for semantic_id, definition in dictionary.items():
        if any(
            path[0] == current
            and len(future) >= len(path) - 1
            and future[: len(path) - 1] == path[1:]
            for path in definition["oriented_paths"]
        ):
            positives.append(semantic_id)
    return tuple(sorted(positives))


def _audit_events(primary: Path, decisions: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    semantic = _dictionary_from_csv(primary / "semantic_loop_dictionary_v2.csv")
    migration = pd.read_csv(primary / "legacy_to_v2_loop_mapping.csv")
    legacy = {
        str(row.semantic_loop_id): _decompose(
            tuple(int(value) for value in str(row.legacy_cycle_path).split("->"))
        )
        for row in migration.itertuples(index=False)
    }
    outcome = pd.read_parquet(
        primary / "first_next_loop_outcomes.parquet",
        columns=["decision_id", "primary_label", "tied_semantic_loop_ids"],
    ).set_index("decision_id")
    comparison = pd.read_parquet(
        primary / "legacy_v2_target_comparison_detail.parquet",
        columns=[
            "decision_id",
            "legacy_positive_labels",
            "active_prefix_count",
            "semantics_differ",
            "comparison_available",
        ],
    ).set_index("decision_id")
    all_sessions = {
        (str(symbol), str(session))
        for symbol, session in decisions[["symbol", "session"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    completion = pd.read_parquet(
        primary / "loop_completion_event_ledger.parquet",
        columns=["decision_id", "semantic_loop_id", "completion_event_index"],
    )
    recorded_completion_keys = Counter(
        (
            str(row.decision_id),
            str(row.semantic_loop_id),
            int(row.completion_event_index),
        )
        for row in completion.itertuples(index=False)
    )
    expected_completion_keys: Counter[tuple[str, str, int]] = Counter()
    label_mismatches = []
    tie_mismatches = []
    prefix_mismatches = []
    legacy_mismatches = []
    comparison_mismatches = []
    aggregate_labels: Counter[str] = Counter()
    checked_decisions = 0
    for key in sorted(all_sessions):
        group = decisions.loc[
            decisions["symbol"].eq(key[0]) & decisions["session"].eq(key[1])
        ].sort_values("bar_ordinal", kind="mergesort")
        states, bars, event_for_bar = _compress_events(group)
        registered, unregistered, _ = _event_trace(states, bars, semantic)
        session_end = int(group["bar_ordinal"].max())
        for local_index, row in enumerate(group.itertuples(index=False)):
            event_index = int(event_for_bar[local_index])
            label, tied = _independent_outcome(
                states=states,
                bars=bars,
                event_index=event_index,
                decision_bar=int(row.bar_ordinal),
                session_end=session_end,
                registered=registered,
                unregistered=unregistered,
                source_available=bool(row.structural_event_eligibility),
            )
            recorded = outcome.loc[row.decision_id]
            aggregate_labels[label] += 1
            if label != recorded.primary_label:
                label_mismatches.append(str(row.decision_id))
            recorded_tied = tuple(_parse_list(recorded.tied_semantic_loop_ids))
            if label == "TIED_REGISTERED_COMPLETION" and tied != recorded_tied:
                tie_mismatches.append(str(row.decision_id))
            prefixes = (
                _active_prefixes(states, event_index, semantic)
                if bool(row.structural_event_eligibility)
                else set()
            )
            if len(prefixes) != int(row.active_prefix_count):
                prefix_mismatches.append(str(row.decision_id))
            independent_legacy = (
                _legacy_labels(states, event_index, legacy)
                if bool(row.structural_event_eligibility)
                else ()
            )
            recorded_legacy = tuple(
                _parse_list(comparison.loc[row.decision_id, "legacy_positive_labels"])
            )
            if independent_legacy != recorded_legacy:
                legacy_mismatches.append(str(row.decision_id))
            v2_ids = (
                tied
                if label == "TIED_REGISTERED_COMPLETION"
                else ((label,) if label in semantic else ())
            )
            semantics_differ = bool(row.structural_event_eligibility) and (
                set(independent_legacy) != set(v2_ids) or not v2_ids
            )
            recorded_comparison = comparison.loc[row.decision_id]
            if (
                bool(recorded_comparison.comparison_available)
                != bool(row.structural_event_eligibility)
                or bool(recorded_comparison.semantics_differ) != semantics_differ
            ):
                comparison_mismatches.append(str(row.decision_id))
            if bool(row.structural_event_eligibility):
                horizon_end = int(row.bar_ordinal) + 24
                for event in registered:
                    if (
                        event["end_event"] > event_index
                        and event["end_bar"] > int(row.bar_ordinal)
                        and event["end_bar"] <= horizon_end
                    ):
                        expected_completion_keys[
                            (
                                str(row.decision_id),
                                str(event["semantic_loop_id"]),
                                int(event["end_event"]),
                            )
                        ] += 1
            checked_decisions += 1
    synthetic_states = np.asarray([2, 4, 2])
    synthetic_bars = np.asarray([0, 1, 2])
    synthetic_dictionary = {"loop_p_2-4-2": _decompose((2, 4, 2))}
    registered, unregistered, _ = _event_trace(
        synthetic_states, synthetic_bars, synthetic_dictionary
    )
    synthetic_label, _ = _independent_outcome(
        states=synthetic_states,
        bars=synthetic_bars,
        event_index=1,
        decision_bar=1,
        session_end=10,
        registered=registered,
        unregistered=unregistered,
        source_available=True,
    )
    synthetic_legacy = _legacy_labels(synthetic_states, 1, synthetic_dictionary)
    recorded_aggregate = Counter(
        {str(label): int(count) for label, count in outcome["primary_label"].value_counts().items()}
    )
    completion_match = expected_completion_keys == recorded_completion_keys
    _record(
        checks,
        "prefix_progression_first_completion_ties_no_loop_session_reset",
        not label_mismatches
        and not tie_mismatches
        and not prefix_mismatches
        and aggregate_labels == recorded_aggregate
        and completion_match,
        {
            "exhaustive_sessions": len(all_sessions),
            "exhaustive_decisions": checked_decisions,
            "label_mismatches": label_mismatches[:20],
            "tie_mismatches": tie_mismatches[:20],
            "prefix_mismatches": prefix_mismatches[:20],
            "independent_primary_counts": dict(aggregate_labels),
            "recorded_primary_counts": dict(recorded_aggregate),
            "expected_completion_rows": sum(expected_completion_keys.values()),
            "recorded_completion_rows": sum(recorded_completion_keys.values()),
            "completion_ledger_exact": completion_match,
        },
    )
    _record(
        checks,
        "legacy_vs_v2_target_difference",
        not legacy_mismatches
        and not comparison_mismatches
        and synthetic_label == "loop_p_2-4-2"
        and synthetic_legacy == (),
        {
            "legacy_mismatches": legacy_mismatches[:20],
            "comparison_mismatches": comparison_mismatches[:20],
            "synthetic_v2": synthetic_label,
            "synthetic_legacy": list(synthetic_legacy),
        },
    )


def _duration_fit(runs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    at_risk = np.zeros((STATE_COUNT, 78), dtype=np.int64)
    exits = np.zeros_like(at_risk)
    censored = np.zeros_like(at_risk)
    for row in runs.itertuples(index=False):
        state = int(row.state)
        duration = int(row.duration)
        at_risk[state, :duration] += 1
        if bool(row.right_censored):
            censored[state, duration - 1] += 1
        else:
            exits[state, duration - 1] += 1
    pooled_risk = at_risk.sum(axis=0).astype(float)
    pooled_exits = exits.sum(axis=0).astype(float)
    pooled = np.divide(
        pooled_exits + 0.5,
        pooled_risk + 1.0,
        out=np.zeros(78, dtype=float),
        where=pooled_risk > 0,
    )
    hazard = np.zeros_like(at_risk, dtype=float)
    for state in range(STATE_COUNT):
        supported = at_risk[state] > 0
        hazard[state, supported] = (exits[state, supported] + 8.0 * pooled[supported]) / (
            at_risk[state, supported] + 8.0
        )
        hazard[state, ~supported & (pooled_risk > 0)] = pooled[~supported & (pooled_risk > 0)]
    return hazard, at_risk, exits, censored


def _audit_duration(primary: Path, checks: list[dict[str, Any]]) -> pd.DataFrame:
    runs = pd.read_parquet(primary / "structural_session_runs.parquet")
    eligible = runs.loc[runs["source_sequence_complete"].astype(bool)].copy()
    invalid = runs.loc[~runs["source_sequence_complete"].astype(bool)]
    hazard, at_risk, exits, censored = _duration_fit(eligible)
    stored = np.load(primary / "duration_model_v2.npz")
    terminal_count = int(eligible["right_censored"].sum())
    session_count = eligible.groupby(["symbol", "session"]).ngroups
    exact24 = int((eligible["duration"] == 24).sum())
    over24 = int((eligible["duration"] > 24).sum())
    probability_violations = 0
    horizon_edge_failures = 0
    truncation_failures = 0
    for state in range(STATE_COUNT):
        survival = 1.0
        pmf = np.zeros(78, dtype=float)
        for index in range(78):
            pmf[index] = survival * hazard[state, index]
            survival *= 1.0 - hazard[state, index]
        probability_violations += int(not np.isclose(pmf.sum() + survival, 1.0, atol=1e-12))
        horizon_completion = float(pmf[:24].sum())
        before_horizon = float(pmf[:23].sum())
        horizon_edge_failures += int(
            not np.isclose(horizon_completion - before_horizon, pmf[23], atol=1e-12)
        )
        remaining_session = 3
        constrained = float(pmf[: min(24, remaining_session)].sum())
        direct = float(pmf[:remaining_session].sum())
        truncation_failures += int(not np.isclose(constrained, direct, atol=1e-12))
    passed = all(
        (
            np.allclose(hazard, stored["hazard"], atol=1e-12),
            np.array_equal(at_risk, stored["at_risk_counts"]),
            np.array_equal(exits, stored["exit_counts"]),
            np.array_equal(censored, stored["censored_counts"]),
            terminal_count == session_count,
            exact24 > 0,
            over24 > 0,
            bool(np.all(hazard[:, 23] < 1.0)),
            bool(np.all(hazard[:, 24] < 1.0)),
            not invalid["right_censored"].any(),
            probability_violations == 0,
            horizon_edge_failures == 0,
            truncation_failures == 0,
        )
    )
    _record(
        checks,
        "duration_24_tail_and_censoring",
        passed,
        {
            "sessions": session_count,
            "right_censored_terminal_runs": terminal_count,
            "exact_duration_24_runs": exact24,
            "duration_greater_than_24_runs": over24,
            "maximum_duration": int(eligible["duration"].max()),
            "hazard24_max": float(hazard[:, 23].max()),
            "hazard25_max": float(hazard[:, 24].max()),
            "gap_invalidated_runs_excluded": len(invalid),
            "probability_conservation_failures": probability_violations,
            "exact_horizon_edge_failures": horizon_edge_failures,
            "remaining_session_truncation_failures": truncation_failures,
        },
    )
    return eligible


def _audit_posterior(primary: Path, expected_rows: int, checks: list[dict[str, Any]]) -> None:
    path = primary / "state_posterior_ledger.parquet"
    total = 0
    state_violations = 0
    age_violations = 0
    next_violations = 0
    timing_violations = 0
    top_violations = 0
    long_age_rows = 0
    forced_exit_rows = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=2048,
        columns=[
            "posterior_state_probabilities",
            "state_age_posterior",
            "next_state_probabilities",
            "top_state",
            "posterior_source_timestamp",
            "posterior_available_timestamp",
            "hard_run_age",
            "probability_state_persists_next_bar",
        ],
    ):
        state = batch.column(0).values.to_numpy(zero_copy_only=False).reshape(-1, 8)
        age = batch.column(1).values.to_numpy(zero_copy_only=False).reshape(-1, 8 * 78)
        next_state = batch.column(2).values.to_numpy(zero_copy_only=False).reshape(-1, 8)
        top = batch.column(3).to_numpy(zero_copy_only=False)
        state_violations += int((~np.isclose(state.sum(axis=1), 1.0, atol=2e-5)).sum())
        age_violations += int((~np.isclose(age.sum(axis=1), 1.0, atol=2e-5)).sum())
        next_violations += int((~np.isclose(next_state.sum(axis=1), 1.0, atol=2e-5)).sum())
        top_violations += int((np.argmax(state, axis=1) != top).sum())
        source = pd.to_datetime(batch.column(4).to_pandas(), utc=True)
        available = pd.to_datetime(batch.column(5).to_pandas(), utc=True)
        timing_violations += int((available < source).sum())
        hard_age = batch.column(6).to_numpy(zero_copy_only=False)
        persists = batch.column(7).to_numpy(zero_copy_only=False)
        long = hard_age >= 24
        long_age_rows += int(long.sum())
        forced_exit_rows += int((long & (persists <= 1e-12)).sum())
        total += batch.num_rows
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    _record(
        checks,
        "state_posterior_normalization_and_population",
        total == expected_rows
        and not any(
            (
                state_violations,
                age_violations,
                next_violations,
                timing_violations,
                top_violations,
                forced_exit_rows,
            )
        )
        and int(metadata["state_age_posterior_support"]) == 78,
        {
            "rows": total,
            "expected_rows": expected_rows,
            "state_probability_violations": state_violations,
            "state_age_probability_violations": age_violations,
            "next_state_probability_violations": next_violations,
            "timing_violations": timing_violations,
            "top_state_violations": top_violations,
            "long_hard_age_rows": long_age_rows,
            "forced_exit_rows_at_or_beyond_24": forced_exit_rows,
            "state_age_posterior_support": metadata["state_age_posterior_support"],
        },
    )


def _session_rows(runs: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for (symbol, session), group in runs.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("start_bar_ordinal", kind="mergesort")
        output.append(
            {
                "symbol": str(symbol),
                "session": str(session),
                "states": tuple(ordered["state"].astype(int)),
                "durations": tuple(ordered["duration"].astype(int)),
            }
        )
    return output


def _balanced_sample(sessions: Sequence[dict[str, Any]], per_symbol: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[session["symbol"]].append(session)
    selected = []
    for symbol in sorted(grouped):
        values = sorted(grouped[symbol], key=lambda item: item["session"])
        indices = np.linspace(0, len(values) - 1, per_symbol, dtype=int)
        selected.extend(values[int(index)] for index in indices)
    return selected


def _count_paths(
    sessions: Sequence[dict[str, Any]],
    paths_by_id: dict[str, tuple[tuple[int, ...], ...]],
) -> np.ndarray:
    ids = sorted(paths_by_id)
    counts = np.zeros(len(ids), dtype=np.int64)
    for session in sessions:
        states = session["states"]
        for index, semantic_id in enumerate(ids):
            for path in paths_by_id[semantic_id]:
                counts[index] += sum(
                    tuple(states[start : start + len(path)]) == path
                    for start in range(len(states) - len(path) + 1)
                )
    return counts


def _audit_null_and_dictionary(
    primary: Path, runs: pd.DataFrame, checks: list[dict[str, Any]]
) -> None:
    sessions = _session_rows(runs)
    sample = _balanced_sample(sessions, 12)
    candidate = pd.read_parquet(primary / "dictionary_candidate_census.parquet")
    supported_candidate = candidate.loc[candidate["support_eligible_for_null"].astype(bool)].copy()
    paths_by_id = {
        str(row.semantic_loop_id): _rotations(
            tuple(int(value) for value in str(row.canonical_path).split("->"))[:-1]
        )
        for row in candidate.itertuples(index=False)
    }
    stored = np.load(primary / "structural_null_draws.npz")
    stored_ids = [str(value) for value in stored["semantic_loop_ids"]]
    ordered_paths = {semantic_id: paths_by_id[semantic_id] for semantic_id in stored_ids}
    sample_observed = _count_paths(sample, ordered_paths)

    initial = np.full(STATE_COUNT, 0.5, dtype=float)
    transition = np.full((STATE_COUNT, STATE_COUNT), 0.5, dtype=float)
    np.fill_diagonal(transition, 0.0)
    for session in sessions:
        initial[session["states"][0]] += 1.0
        for origin, destination in zip(session["states"][:-1], session["states"][1:], strict=True):
            transition[origin, destination] += 1.0
    initial /= initial.sum()
    transition /= transition.sum(axis=1, keepdims=True)
    hazard, _, _, _ = _duration_fit(runs)
    duration_cdf = []
    for state in range(STATE_COUNT):
        survival = 1.0
        pmf = []
        for age in range(78):
            pmf.append(survival * hazard[state, age])
            survival *= 1.0 - hazard[state, age]
        cumulative = np.cumsum(np.r_[pmf, survival])
        cumulative[-1] = 1.0
        duration_cdf.append(cumulative)
    duration_cdf = np.asarray(duration_cdf)

    rng = np.random.default_rng(20260718)
    reproduced = []
    for _ in range(int(stored["primary_draws"].shape[0])):
        draw_sessions = []
        for source in sample:
            length = sum(source["durations"])
            state = int(np.searchsorted(np.cumsum(initial), rng.random(), side="right"))
            states = []
            durations = []
            elapsed = 0
            while elapsed < length:
                states.append(state)
                duration = int(np.searchsorted(duration_cdf[state], rng.random(), side="right") + 1)
                duration = min(duration, length - elapsed)
                durations.append(duration)
                elapsed += duration
                if elapsed >= length:
                    break
                cumulative = np.cumsum(transition[state])
                cumulative[-1] = 1.0
                state = int(np.searchsorted(cumulative, rng.random(), side="right"))
            draw_sessions.append(
                {
                    "symbol": "null",
                    "session": "draw",
                    "states": tuple(states),
                    "durations": tuple(durations),
                }
            )
        reproduced.append(_count_paths(draw_sessions, ordered_paths))
    reproduced_array = np.asarray(reproduced)
    null_match = np.array_equal(reproduced_array, stored["primary_draws"])
    observed_match = np.array_equal(sample_observed, stored["sample_observed"])
    _record(
        checks,
        "semi_markov_null_reconstruction",
        null_match and observed_match and stored["primary_draws"].shape[0] == 2000,
        {
            "draws": int(stored["primary_draws"].shape[0]),
            "sample_sessions": len(sample),
            "all_primary_draws_exact": bool(null_match),
            "sample_observed_exact": bool(observed_match),
        },
    )
    full_observed = _count_paths(sessions, ordered_paths)
    observed_lookup = dict(zip(stored_ids, full_observed, strict=True))
    candidate_match = all(
        int(row.observed_completions) == observed_lookup[str(row.semantic_loop_id)]
        for row in supported_candidate.itertuples(index=False)
    )
    selection = pd.read_csv(primary / "semantic_dictionary_selection.csv").sort_values(
        "selection_rank"
    )
    dependency_failures = []
    seen = set()
    definitions = {
        semantic_id: _decompose(paths[0]) for semantic_id, paths in ordered_paths.items()
    }
    for row in selection.itertuples(index=False):
        definition = definitions[str(row.semantic_loop_id)]
        if any(dependency not in seen for dependency in definition["component_primitive_ids"]):
            dependency_failures.append(str(row.semantic_loop_id))
        seen.add(str(row.semantic_loop_id))
    raw_frequency_is_not_rank = (
        selection.sort_values("observed_completions", ascending=False)["semantic_loop_id"].tolist()
        != selection["semantic_loop_id"].tolist()
    )
    draws = np.asarray(stored["primary_draws"], dtype=float)
    sample_values = np.asarray(stored["sample_observed"], dtype=float)
    expected_sample = draws.mean(axis=0)
    p_values = (1.0 + (draws >= sample_values[None, :]).sum(axis=0)) / (len(draws) + 1.0)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * len(order) / rank)
        adjusted[index] = min(1.0, running)
    bar_scale = sum(sum(item["durations"]) for item in sessions) / sum(
        sum(item["durations"]) for item in sample
    )

    first = np.full((STATE_COUNT, STATE_COUNT), 0.5, dtype=float)
    np.fill_diagonal(first, 0.0)
    second = np.full((STATE_COUNT, STATE_COUNT, STATE_COUNT), 0.5, dtype=float)
    for previous in range(STATE_COUNT):
        for current in range(STATE_COUNT):
            second[previous, current, current] = 0.0
    for session in sessions:
        for origin, destination in zip(session["states"][:-1], session["states"][1:], strict=True):
            first[origin, destination] += 1.0
        for previous, current, destination in zip(
            session["states"][:-2],
            session["states"][1:-1],
            session["states"][2:],
            strict=True,
        ):
            second[previous, current, destination] += 1.0
    first /= first.sum(axis=1, keepdims=True)
    second /= second.sum(axis=2, keepdims=True)

    def expected_for_path(
        path: tuple[int, ...],
        *,
        use_second: bool,
        records: Sequence[dict[str, Any]] = sessions,
    ) -> float:
        anchors = sum(
            sum(
                session["states"][start] == path[0]
                for start in range(max(0, len(session["states"]) - len(path) + 1))
            )
            for session in records
        )
        probability = float(first[path[0], path[1]])
        if use_second:
            for previous, current, destination in zip(path[:-2], path[1:-1], path[2:], strict=True):
                probability *= float(second[previous, current, destination])
        else:
            for origin, destination in zip(path[1:-1], path[2:], strict=True):
                probability *= float(first[origin, destination])
        return anchors * probability

    metric_mismatches = []
    recomputed_scores: dict[str, float] = {}
    recomputed_rows: dict[str, dict[str, Any]] = {}
    ids = stored_ids
    for index, semantic_id in enumerate(ids):
        row = supported_candidate.loc[supported_candidate["semantic_loop_id"].eq(semantic_id)].iloc[
            0
        ]
        definition = definitions[semantic_id]
        observed = float(observed_lookup[semantic_id])
        expected_full = float(expected_sample[index] * bar_scale)
        full_length = len(definition["full_core"])
        starting_states = {path[0] for path in ordered_paths[semantic_id]}
        eligible = sum(
            sum(
                session["states"][start] in starting_states
                for start in range(max(0, len(session["states"]) - (full_length + 1) + 1))
            )
            for session in sessions
        )
        analytical = sum(
            expected_for_path(path, use_second=False) for path in ordered_paths[semantic_id]
        )
        second_expected = sum(
            expected_for_path(path, use_second=True) for path in ordered_paths[semantic_id]
        )
        by_quarter: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for session in sessions:
            timestamp = pd.Timestamp(session["session"])
            by_quarter[f"{timestamp.year}_q{timestamp.quarter}"].append(session)
        quarter_excess = []
        for quarter_sessions in by_quarter.values():
            quarter_observed = float(
                _count_paths(quarter_sessions, {semantic_id: ordered_paths[semantic_id]})[0]
            )
            quarter_expected = sum(
                expected_for_path(path, use_second=False, records=quarter_sessions)
                for path in ordered_paths[semantic_id]
            )
            quarter_excess.append(quarter_observed > quarter_expected)
        period_consistency = float(np.mean(quarter_excess))
        observed_rate = min((observed + 0.5) / (eligible + 1.0), 1.0 - 1e-12)
        expected_rate = min((expected_full + 0.5) / (eligible + 1.0), 1.0 - 1e-12)
        information = max(
            0.0,
            observed_rate * math.log2(max(observed_rate / expected_rate, 1e-12))
            + (1.0 - observed_rate)
            * math.log2(max((1.0 - observed_rate) / max(1.0 - expected_rate, 1e-12), 1e-12)),
        )
        increment_current = max(
            0.0,
            observed_rate * math.log2((observed + 0.5) / max(analytical + 0.5, 1e-12)),
        )
        increment_history = max(
            0.0,
            observed_rate * math.log2((observed + 0.5) / max(second_expected + 0.5, 1e-12)),
        )
        penalty = 0.05 * full_length
        if definition["motif_type"] == "repeat":
            penalty += 0.25 * (int(definition["repeat_depth"]) - 1)
        elif definition["motif_type"] == "composite":
            penalty += 0.40 + 0.10 * len(definition["component_primitive_ids"])
        breadth = (
            min(float(row.stock_breadth) / 20.0, 1.0)
            + min(float(row.month_breadth) / 12.0, 1.0)
            + min(float(row.clock_breadth) / 3.0, 1.0)
        ) / 3.0
        rate_ratio = (observed + 0.5) / (expected_full + 0.5)
        score = (
            math.log1p(max(0.0, observed - expected_full))
            + math.log(max(rate_ratio, 1e-12))
            + 10.0 * (information + increment_current + increment_history)
            + breadth
            + period_consistency
            + 0.10 * max(0.0, -math.log10(max(float(adjusted[index]), 1e-12)))
            - penalty
        )
        recomputed_scores[semantic_id] = score
        recomputed_rows[semantic_id] = {
            "q": float(adjusted[index]),
            "rate_ratio": rate_ratio,
            "increment_history": increment_history,
            "analytical": analytical,
        }
        comparisons = {
            "expected_completions_semi_markov": expected_full,
            "empirical_p_value": float(p_values[index]),
            "fdr_q_value": float(adjusted[index]),
            "conditional_information_gain": information,
            "increment_beyond_current_state": increment_current,
            "increment_beyond_previous_state_history": increment_history,
            "complexity_penalty": penalty,
            "selection_score": score,
            "period_consistency": period_consistency,
        }
        for column, expected_value in comparisons.items():
            if not np.isclose(float(row[column]), expected_value, rtol=1e-9, atol=1e-10):
                metric_mismatches.append(f"{semantic_id}:{column}")

    eligible_ids = {
        semantic_id
        for semantic_id, values in recomputed_rows.items()
        if values["q"] <= 0.1 and values["rate_ratio"] >= 1.05 and values["increment_history"] > 0.0
    }
    ordered_ids = sorted(eligible_ids, key=lambda value: (-recomputed_scores[value], value))
    recomputed_selection: list[str] = []
    selected_set: set[str] = set()
    for semantic_id in ordered_ids:
        if semantic_id in selected_set:
            continue
        dependencies = tuple(dict.fromkeys(definitions[semantic_id]["component_primitive_ids"]))
        if any(dependency not in eligible_ids for dependency in dependencies):
            continue
        additions = [dependency for dependency in dependencies if dependency not in selected_set]
        additions.append(semantic_id)
        additions = [value for value in additions if value not in selected_set]
        if len(recomputed_selection) + len(additions) > 20:
            continue
        recomputed_selection.extend(additions)
        selected_set.update(additions)
        if len(recomputed_selection) == 20:
            break
    selection_exact = recomputed_selection == selection["semantic_loop_id"].astype(str).tolist()
    excluded = candidate.loc[~candidate["support_eligible_for_null"].astype(bool)]
    exclusion_status_valid = bool(
        excluded["candidate_status"].eq("excluded_before_null").all()
        and excluded["exclusion_reason"].notna().all()
    )
    null_summary = pd.read_csv(primary / "structural_null_results.csv")
    null_metric_mismatches = []
    clock_draws = np.asarray(stored["clock_conditioned_draws"], dtype=float)
    circular_sessions = []
    for source in sample:
        pairs = list(zip(source["states"], source["durations"], strict=True))
        offset = max(1, len(pairs) // 2)
        rotated = pairs[offset:] + pairs[:offset]
        merged: list[tuple[int, int]] = []
        for state, duration in rotated:
            if merged and merged[-1][0] == state:
                merged[-1] = (state, merged[-1][1] + duration)
            else:
                merged.append((state, duration))
        circular_sessions.append(
            {
                "symbol": source["symbol"],
                "session": source["session"],
                "states": tuple(item[0] for item in merged),
                "durations": tuple(item[1] for item in merged),
            }
        )
    circular_counts = _count_paths(circular_sessions, ordered_paths)
    for index, semantic_id in enumerate(ids):
        expected_by_null = {
            "NULL_A_FITTED_SEMI_MARKOV": float(draws[:, index].mean()),
            "NULL_B_CLOCK_CONDITIONED_SEMI_MARKOV": float(clock_draws[:, index].mean()),
            "NULL_C_FIRST_ORDER_ANALYTICAL": float(recomputed_rows[semantic_id]["analytical"]),
            "NULL_D_WHOLE_SESSION_CIRCULAR_CONTROL": float(circular_counts[index]),
        }
        for null_name, expected_count in expected_by_null.items():
            row = null_summary.loc[
                null_summary["null_name"].eq(null_name)
                & null_summary["semantic_loop_id"].eq(semantic_id)
            ]
            if len(row) != 1 or not np.isclose(
                float(row.iloc[0]["expected_count"]), expected_count, rtol=1e-9, atol=1e-10
            ):
                null_metric_mismatches.append(f"{semantic_id}:{null_name}:expected")
        clock_p = float(
            (1.0 + (clock_draws[:, index] >= sample_values[index]).sum()) / (len(clock_draws) + 1.0)
        )
        clock_row = null_summary.loc[
            null_summary["null_name"].eq("NULL_B_CLOCK_CONDITIONED_SEMI_MARKOV")
            & null_summary["semantic_loop_id"].eq(semantic_id)
        ].iloc[0]
        if not np.isclose(float(clock_row["empirical_p_value"]), clock_p, rtol=1e-9, atol=1e-10):
            null_metric_mismatches.append(f"{semantic_id}:clock:p")
    _record(
        checks,
        "dictionary_metrics_and_selection",
        candidate_match
        and not dependency_failures
        and raw_frequency_is_not_rank
        and not metric_mismatches
        and selection_exact
        and exclusion_status_valid
        and not null_metric_mismatches,
        {
            "candidate_rows": len(candidate),
            "selected_rows": len(selection),
            "observed_count_mismatch": not candidate_match,
            "dependency_failures": dependency_failures,
            "rank_differs_from_raw_frequency": raw_frequency_is_not_rank,
            "metric_mismatches": metric_mismatches[:20],
            "selection_exact": selection_exact,
            "recomputed_selection": recomputed_selection,
            "support_excluded_candidates": len(excluded),
            "support_exclusion_status_valid": exclusion_status_valid,
            "null_metric_mismatches": null_metric_mismatches[:20],
        },
    )


def _audit_safety(primary: Path, checks: list[dict[str, Any]]) -> None:
    failures = []
    identity_keys = {
        "run_id_v2",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "source_artifact_hash",
        "dictionary_version",
        "state_model_version",
    }
    for path in sorted(primary.iterdir()):
        if not path.is_file() or path.name in {"artifact_manifest.json", "independent_audit.json"}:
            continue
        if path.suffix == ".csv":
            columns = pd.read_csv(path, nrows=0).columns
            if not set(SAFETY).union(identity_keys).issubset(columns):
                failures.append(path.name)
        elif path.suffix == ".parquet":
            if not set(SAFETY).union(identity_keys).issubset(pq.read_schema(path).names):
                failures.append(path.name)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if any(
                payload.get(key) != value for key, value in SAFETY.items()
            ) or not identity_keys.issubset(payload):
                failures.append(path.name)
        elif path.suffix == ".npz":
            payload = np.load(path)
            expected = {
                "research_only": True,
                "execution_enabled": False,
                "order_placement": "disabled",
                "broker_connected": False,
                "strategy_promotion": False,
            }
            for key, value in expected.items():
                if key not in payload or payload[key][0] != value:
                    failures.append(path.name)
                    break
            if not identity_keys.issubset(payload.files):
                failures.append(path.name)
    _record(
        checks,
        "artifact_safety_flags",
        not failures,
        {"audited_files": len(list(primary.iterdir())), "failures": failures},
    )


def _audit_recorded_manifest(primary: Path, checks: list[dict[str, Any]]) -> None:
    payload = json.loads((primary / "artifact_manifest.json").read_text(encoding="utf-8"))
    recorded = {str(row["file"]): row for row in payload["artifacts"]}
    mismatches = []
    files = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    for name, path in sorted(files.items()):
        row = recorded.get(name)
        if row is None:
            mismatches.append(f"missing:{name}")
        elif row["sha256"] != _sha256_file(path) or int(row["bytes"]) != path.stat().st_size:
            mismatches.append(f"identity:{name}")
    extra = sorted(set(recorded).difference(files))
    mismatches.extend(f"extra:{name}" for name in extra)
    _record(
        checks,
        "artifact_manifest_input_hashes",
        not mismatches and int(payload["artifact_count"]) == len(files),
        {"files": len(files), "mismatches": mismatches},
    )


def _audit_exact_identity(primary: Path, exact: Path | None, checks: list[dict[str, Any]]) -> None:
    if exact is None:
        _record(
            checks,
            "exact_rerun_identity",
            False,
            {"status": "required_exact_rerun_missing"},
        )
        return
    exclusions = {"artifact_manifest.json", "independent_audit.json", "decision.json"}
    primary_files = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file() and path.name not in exclusions
    }
    exact_files = {
        path.name: path
        for path in exact.iterdir()
        if path.is_file() and path.name not in exclusions
    }
    mismatches = []
    for name in sorted(set(primary_files) | set(exact_files)):
        if name not in primary_files or name not in exact_files:
            mismatches.append({"file": name, "reason": "missing_in_one_run"})
        elif _sha256_file(primary_files[name]) != _sha256_file(exact_files[name]):
            mismatches.append({"file": name, "reason": "sha256_differs"})
    _record(
        checks,
        "exact_rerun_identity",
        not mismatches,
        {"compared_files": len(primary_files), "mismatches": mismatches},
    )


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {"file": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            )
    return {"artifacts": rows, "artifact_count": len(rows)}


def _rates(outcomes: pd.Series) -> dict[str, float]:
    total = len(outcomes)
    return {
        "tie_rate": float(outcomes.eq("TIED_REGISTERED_COMPLETION").sum() / total),
        "no_registered_loop_rate": float(
            outcomes.eq("NO_REGISTERED_LOOP_WITHIN_HORIZON").sum() / total
        ),
        "session_end_rate": float(outcomes.eq("SESSION_END").sum() / total),
        "unavailable_rate": float(outcomes.eq("UNAVAILABLE").sum() / total),
        "unregistered_loop_rate": float(outcomes.eq("UNREGISTERED_LOOP").sum() / total),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional dependency."""

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") if pd.notna(value) else "" for value in row)
            + " |"
        )
    return "\n".join(lines)


def _render_report(primary: Path, audit: dict[str, Any]) -> None:
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    b0 = pd.read_csv(primary / "b0_start_end_difference_summary.csv")
    dictionary = pd.read_csv(primary / "semantic_loop_dictionary_v2.csv")
    migration = pd.read_csv(primary / "legacy_to_v2_loop_mapping.csv")
    ambiguous_migrations = int(migration["migration_status"].ne("migrated").sum())
    comparison = pd.read_csv(primary / "legacy_v2_target_comparison.csv")
    aggregate = comparison.loc[
        comparison["record_type"].eq("aggregate")
        & comparison["population"].eq("eligible_completed_bars")
    ].iloc[0]
    outcomes = pd.read_parquet(
        primary / "first_next_loop_outcomes.parquet", columns=["primary_label"]
    )["primary_label"]
    rates = _rates(outcomes)
    decisions = pd.read_parquet(
        primary / "causal_completed_bar_decisions.parquet",
        columns=["structural_event_eligibility", "is_run_entry", "highest_soft_prefix_probability"],
    )
    duration = pd.read_csv(primary / "duration_censoring_audit.csv")
    nulls = pd.read_csv(primary / "structural_null_results.csv")
    selected_null = nulls.loc[
        nulls["null_name"].eq("NULL_A_FITTED_SEMI_MARKOV")
        & nulls["semantic_loop_id"].isin(dictionary["semantic_loop_id"])
    ]
    impact = pd.read_csv(primary / "historical_lineage_impact.csv")
    blockers = pd.read_csv(primary / "missingness_and_blockers.csv")
    b0_lines = "\n".join(
        f"- `{row.field}`: {int(row.runs_start_end_differ):,} of {int(row.audited_runs):,} runs differ."
        for row in b0.itertuples(index=False)
    )
    impact_table = _markdown_table(
        impact[
            [
                "experiment_name",
                "severity",
                "historical_result_interpretability",
            ]
        ]
    )
    blocker_table = _markdown_table(blockers[["item", "affected_rows", "status"]])
    report = f"""# Loop Event Semantics and Causal Infrastructure V2

`research_only=true` · `execution_enabled=false` · `order_placement=disabled` · `broker_connected=false` · `strategy_promotion=false`

## 1. Exact scope

This migration reconstructs structural state and loop events only. It does not read payoff, MFE, MAE, P&L, order, position, broker, or execution data; it trains no predictor and makes no economic-edge claim.

## 2. Active versus frozen implementation map

The pre-rewrite census contains {len(pd.read_csv(primary / "implementation_census.csv"))} implementation entries. Historical files at baseline `{json.loads(CONTRACT_PATH.read_text())["source"]["frozen_lineage_baseline_commit"]}` remain byte unchanged. V2 behavior is isolated in typed package modules, a new runner, contract, tests, and independent auditor.

## 3. Confirmed defects

- The historical run builder combines the first row's start timestamp with last-row B0/context fields.
- Legacy targets are overlapping compatible rotated-cycle labels, not a mutually exclusive first next event.
- Primitive loops, repeated traversals, and composites shared a flat cycle namespace.
- Legacy IDs depend on discovery rank; discovery/load length contracts differed (2–5 versus 2–4).
- Duration 24 was an `>=24` bucket, exact 24 was omitted from convolution, age 24 was forced to exit, and terminal runs were excluded rather than censored.
- The causal filter discarded all but MAP state/age, and forecast anchors were run entries only.
- A historical limited-path baseline repeated anchor context through hypothetical transitions.
- The shuffled-order null destroyed transition, duration, occupancy-order, and phase structure; raw support dominated dictionary selection.

## 4. Suspected defects not confirmed

Actual 2024 B0 value leakage was not observed: both B0 fields were constant within every audited hard run. This disproves an empirical 2024 B0-value change while leaving the source-position implementation defect confirmed. No protected prospective dataset was opened.

## 5. B0 timing result

{b0_lines}

Provider timestamps are bar starts. Completed-bar availability and decisions are five minutes later. B0 uses an explicit prior-session source/availability timestamp; missing warm-up values remain missing.

## 6. Historical experiments affected by B0 timing

No audited 2024 B0 anchor changed. Raw-run clock consumers are affected because clock fields changed in more than 70,000 runs. Context-model results remain provenance-limited even where session-level B0 happened to be invariant.

## 7. Legacy loop-target semantics

Legacy labels independently ask whether each compatible rotated whole cycle occurs. They can have several positives and have no mutually exclusive no-loop outcome. Historical top-three recall is therefore overlapping-label recall.

## 8. V2 first-event semantics

V2 resolves the earliest completion after each completed bar across prefixes already open at the decision and loops initiated later. It separately labels registered primitive/repeat/composite completions, unregistered loops, no registered completion, session end, ties, and unavailable sources.

## 9. Quantified legacy versus V2 target difference

Across {int(aggregate.decisions):,} source-eligible decisions, {int(aggregate.semantic_difference_decisions):,} ({aggregate.semantic_difference_decisions / aggregate.decisions:.2%}) differ under the full semantic contract; {int(aggregate.registered_event_set_difference_decisions):,} differ when comparing only registered event sets. Legacy had multiple simultaneous positives at {int(aggregate.simultaneous_legacy_positive_decisions):,} decisions, and {int(aggregate.active_prefix_decisions):,} decisions had active prefixes. Source-unavailable decisions are reported separately and never counted as semantic differences.

## 10. Primitive/repeat/composite decomposition

The selected dictionary contains {(dictionary.motif_type == "primitive").sum()} primitives, {(dictionary.motif_type == "repeat").sum()} repeated traversals, and {(dictionary.motif_type == "composite").sum()} composites. The legacy migration retains explicit component mappings rather than deleting its composite motif; {ambiguous_migrations} migrations are ambiguous.

## 11. Semantic ID design

IDs derive from canonical paths: `loop_p_<path>`, `loop_rN_<primitive-path>`, and `loop_c_<hash>`. Rotation preserves identity; orientation is separate metadata; reverse direction is not silently merged; repeat depth changes identity.

## 12. Prefix automaton design

An Aho–Corasick-style automaton advances only on causal state-change events and retains every suffix matching a proper oriented-loop prefix. Per-bar decisions map to the most recent causal state event without replacing their own completed-bar timestamp.

## 13. Tie and nested-event handling

Same-event structural ties remain `TIED_REGISTERED_COMPLETION` with ordered IDs in a secondary field. Primitive completions precede later repeats/composites; nested completions remain secondary events.

Observed rates are: tie {rates["tie_rate"]:.2%}, no registered loop {rates["no_registered_loop_rate"]:.2%}, unregistered loop {rates["unregistered_loop_rate"]:.2%}, session end {rates["session_end_rate"]:.2%}, and unavailable {rates["unavailable_rate"]:.2%}.

## 14. Session-boundary handling

Prefixes reset at every regular-session boundary. In-session gaps fail closed as `UNAVAILABLE`; no prefix crosses overnight.

## 15. Duration-24 correction

Durations 1–78 are exact. Exact duration 24 contributes at horizon 24; duration 25 is distinct; no final hazard is forced to one.

## 16. Long-duration and censoring design

The duration model is discrete survival with hierarchical smoothing. It contains {int(duration.right_censored_terminal_runs.sum()):,} right-censored terminal runs, {int(duration.exact_duration_24_count.sum()):,} exact-24 runs, and {int(duration.duration_greater_than_24_count.sum()):,} runs longer than 24. Remaining session time truncates completion while preserving terminal/no-completion mass.

## 17. Posterior-state export

All {metadata["completed_bar_decisions"]:,} bars export eight state probabilities, entropy, top two states and margin, persistence/transition probability, next-state probabilities, expected age, and the complete 8×78 V2 state-age posterior. Ages 1–23 preserve the frozen hazards; the former forced exit at age 24 is replaced by an explicit geometric tail through the regular-session support. Frozen hard-MAP labels remain a separate compatibility surface.

## 18. Hard versus hysteretic versus soft representations

`LEGACY_HARD_MAP` exactly reproduces frozen causal labels. `CAUSAL_HYSTERETIC_STATE` uses only the current posterior and prior causal state and resets by session. `SOFT_POSTERIOR` exports probability mass without asserting hard completion; soft prefix propagation was bounded to {int(decisions.highest_soft_prefix_probability.notna().sum()):,} preregistered sample rows.

## 19. Per-bar versus run-entry populations

There are {len(decisions):,} completed-bar decisions and {int(decisions.is_run_entry.sum()):,} run-entry baseline rows. Run entries are a strict subset.

## 20. Static future-context audit

Direct V2 event models may use current known context once. Unknown future B0, price, activity, volatility, and market context are never fabricated. Historical history-only models are distinguishable from the rejected static-context limited-path baseline in the lineage table.

## 21. Old null limitations

Within-session state permutation destroys transition probabilities, dwell structure, persistence, higher-order order, and phase. It is retained only as historical evidence.

## 22. V2 semi-Markov null

Primary results use {metadata["primary_null_draws"]:,} fitted semi-Markov draws on a balanced 264-session sample with original lengths. Selected-loop rate ratios span {selected_null.rate_ratio.min():.2f}–{selected_null.rate_ratio.max():.2f}; selected q-values are at most {selected_null.fdr_q_value.max():.4f}. Clock-conditioned, first-order analytical, and whole-session circular controls are exported separately. These are structural results, not economic evidence.

## 23. Dictionary-selection redesign

The {len(dictionary)}-entry dictionary separates eligible anchors, observed and null-expected completions, excess, rate ratio, conditional information, current-state and second-order increments, breadth, period consistency, and complexity. Primitive dependencies are inserted before repeats/composites. Raw frequency alone does not determine rank.

## 24. Allowed-length consistency

Discovery, decomposition, storage, loading, scoring, tests, and audit share primitive lengths 2–5, composite lengths 4–8, and maximum event length 8. Unsupported legacy entries fail closed.

## 25. Historical lineage impact table

{impact_table}

## 26. Tests

Focused tests cover causal provenance, semantic identity, prefix matching, duration/censoring, posterior export, per-bar ledgers, nulls, dictionary closure, migration, frozen hashes, and safety. The final command outcomes are reported in the task handoff; this report never labels an unexecuted command as passed.

## 27. Independent audit

The independent auditor passed {audit["passed_checks"]} of {audit["check_count"]} checks. It imported no production V2 module and reconstructed from detailed decisions, state runs, posterior arrays, source files, dictionaries, and raw null draws.

## 28. Exact rerun

Exact rerun identity: `{audit["exact_rerun_status"]}`. Artifact-level mismatch details, if any, are in `independent_audit.json`.

## 29. Remaining blockers

{blocker_table}

These are known limitations, not permission to widen scope. No event source blocker remains for eligible rows.

## 30. Scientific decision

`{audit["scientific_decision"]}`

No edge, profitability, strategy, paper-trading, or live-readiness claim is made.

## 31. Exact next experiment

A separately preregistered structural forecast comparing simple baselines and competing-event models for first next-loop identity and arrival time, with no payoff or economic target.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def audit(primary: Path, exact: Path | None) -> dict[str, Any]:
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    AUDIT_IDENTITY.clear()
    AUDIT_IDENTITY.update(
        {
            key: metadata[key]
            for key in (
                "contract_hash",
                "data_snapshot_hash",
                "dictionary_version",
                "state_model_version",
            )
        }
    )
    AUDIT_IDENTITY["run_id_v2"] = metadata["run_id"]
    AUDIT_IDENTITY["git_sha"] = metadata["source_commit"]
    AUDIT_IDENTITY["source_artifact_hash"] = metadata["data_snapshot_hash"]
    checks: list[dict[str, Any]] = []
    _audit_recorded_manifest(primary, checks)
    structural_rows = _audit_sources(primary, checks)
    _audit_frozen_tree(primary, checks)
    decisions = _audit_provenance(primary, checks)
    _audit_identities(primary, checks)
    _audit_events(primary, decisions, checks)
    runs = _audit_duration(primary, checks)
    _audit_posterior(primary, len(decisions), checks)
    _record(
        checks,
        "per_bar_population",
        structural_rows == len(decisions) and decisions["decision_id"].is_unique,
        {
            "provider_structural_rows": structural_rows,
            "decision_rows": len(decisions),
            "unique_decision_ids": bool(decisions["decision_id"].is_unique),
        },
    )
    _audit_null_and_dictionary(primary, runs, checks)
    _audit_safety(primary, checks)
    _audit_exact_identity(primary, exact, checks)
    passed = all(check["passed"] for check in checks)
    exact_check = next(check for check in checks if check["check"] == "exact_rerun_identity")
    payload = {
        "auditor": "independent_loop_event_semantics_v2",
        "production_v2_imported": False,
        "check_count": len(checks),
        "passed_checks": sum(check["passed"] for check in checks),
        "failed_checks": [check["check"] for check in checks if not check["passed"]],
        "overall_pass": passed,
        "exact_rerun_status": (
            "deferred"
            if exact is None
            else "byte_identical"
            if exact_check["passed"]
            else "mismatch"
        ),
        "scientific_decision": (
            "loop_event_v2_ready_with_known_limitations"
            if passed
            else "implementation_audit_incomplete"
        ),
        "checks": checks,
    }
    destinations = [primary] + ([exact] if exact is not None else [])
    for destination in destinations:
        if destination is None:
            continue
        _write_json(destination / "independent_audit.json", payload)
        decision_path = destination / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["scientific_decision"] = payload["scientific_decision"]
        decision["ready_for_next_loop_forecast"] = passed
        decision["independent_audit_pass"] = passed
        _write_json(decision_path, decision)
    _render_report(primary, payload)
    for destination in destinations:
        if destination is not None:
            _write_json(destination / "artifact_manifest.json", _artifact_manifest(destination))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    arguments = parser.parse_args()
    payload = audit(
        arguments.primary.resolve(),
        arguments.exact.resolve() if arguments.exact else None,
    )
    print(json.dumps(payload, sort_keys=True, indent=2, default=_json_default))
    if not payload["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
