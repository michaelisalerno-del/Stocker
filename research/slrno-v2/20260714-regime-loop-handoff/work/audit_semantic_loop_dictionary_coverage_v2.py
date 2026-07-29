"""Independent audit for Semantic Loop Dictionary and First-Event Coverage V2.

The auditor intentionally imports no production implementation from this phase.
It reconstructs semantic identities, support, null controls, selection, targets,
coverage, family mapping, safety, and exact-rerun identity from low-level files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-semantic-loop-dictionary-coverage-v2.json"
REPORT_PATH = WORK_DIR / "reports" / "20260718-semantic-loop-dictionary-coverage-v2.md"
OLD_ROOT = WORK_DIR / "artifacts" / "20260718-loop-event-semantics-v2"
OLD_PRIMARY = OLD_ROOT / "primary"
OLD_EXACT = OLD_ROOT / "exact_rerun"
HORIZON_BARS = 24
STATE_COUNT = 8
MAXIMUM_DURATION = 78
AUDIT_SAMPLE_SIZE = 512
NULL_SUBSET_SESSIONS = 12
NULL_SUBSET_DRAWS = 32
SAFETY: dict[str, Any] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
}
FORBIDDEN_ECONOMIC_FRAGMENTS = (
    "future_return",
    "return_after",
    "payoff",
    "profit",
    "pnl",
    "mfe",
    "mae",
    "execution_price",
    "route_outcome",
)
TRACE_PLACEHOLDERS = {
    "decision_id",
    "symbol",
    "session",
    "decision_timestamp",
    "event_timestamp",
    "semantic_loop_id",
    "primitive_loop_id",
    "source_artifact",
    "source_hash",
}
AUDIT_IDENTITY: dict[str, Any] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {**AUDIT_IDENTITY, **dict(payload), **SAFETY},
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _record(checks: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "evidence": evidence})


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = ast.literal_eval(str(value))
    return list(parsed)


def _canonical_rotation(core: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in core)
    if len(values) < 2:
        raise ValueError("primitive core is too short")
    return min(values[index:] + values[:index] for index in range(len(values)))


def _primitive_id(core: Sequence[int]) -> str:
    canonical = _canonical_rotation(core)
    closed = canonical + (canonical[0],)
    return "loop_p_" + "-".join(str(value) for value in closed)


def _primitive_root(core: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    for width in range(2, len(core) // 2 + 1):
        if len(core) % width == 0 and core[:width] * (len(core) // width) == core:
            return core[:width], len(core) // width
    return core, 1


def _decompose_path(path: Sequence[int]) -> dict[str, Any]:
    closed = tuple(int(value) for value in path)
    if len(closed) < 3 or closed[0] != closed[-1]:
        raise ValueError("path is not closed")
    if any(left == right for left, right in zip(closed[:-1], closed[1:], strict=True)):
        raise ValueError("path has a compressed self transition")
    core = closed[:-1]
    periodic_root, periodic_depth = _primitive_root(core)
    stack_states: list[int] = []
    stack_positions: list[int] = []
    components: list[tuple[int, ...]] = []
    boundaries: list[tuple[int, int]] = []
    for position, state in enumerate(closed):
        if state not in stack_states:
            stack_states.append(state)
            stack_positions.append(position)
            continue
        stack_index = stack_states.index(state)
        component = tuple(stack_states[stack_index:] + [state])
        components.append(component)
        boundaries.append((stack_positions[stack_index], position))
        stack_states = stack_states[:stack_index] + [state]
        stack_positions = stack_positions[:stack_index] + [position]
    if not components:
        raise ValueError("closed path has no component")
    component_ids = tuple(_primitive_id(component[:-1]) for component in components)
    final_component = components[-1]
    primitive_core = _canonical_rotation(final_component[:-1])
    primitive_loop_id = _primitive_id(primitive_core)
    exact_repeat = periodic_depth > 1 and len(set(periodic_root)) == len(periodic_root)
    if exact_repeat:
        motif_type = "repeat"
        repeat_depth = periodic_depth
        motif_id = f"loop_r{repeat_depth}_" + "-".join(
            str(value)
            for value in (
                _canonical_rotation(periodic_root) + (_canonical_rotation(periodic_root)[0],)
            )
        )
    elif len(components) == 1 and len(set(core)) == len(core):
        motif_type = "primitive"
        repeat_depth = 1
        motif_id = primitive_loop_id
    else:
        motif_type = "composite"
        repeat_depth = 1
        text = "-".join(str(value) for value in closed)
        motif_id = f"loop_c_{hashlib.sha256(text.encode('ascii')).hexdigest()[:12]}"
    return {
        "full_closed_path": closed,
        "primitive_core": primitive_core,
        "primitive_loop_id": primitive_loop_id,
        "primitive_transition_length": len(primitive_core),
        "semantic_motif_id": motif_id,
        "motif_type": motif_type,
        "repeat_depth": repeat_depth,
        "component_primitive_ids": component_ids,
        "component_boundaries": tuple(boundaries),
        "orientation": final_component,
    }


def _verify_declared_manifest(directory: Path) -> list[str]:
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for row in manifest["artifacts"]:
        path = directory / str(row["file"])
        if not path.is_file():
            mismatches.append(f"missing:{path.name}")
            continue
        if _sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["bytes"]):
            mismatches.append(f"identity:{path.name}")
    if int(manifest["artifact_count"]) != len(manifest["artifacts"]):
        mismatches.append("artifact_count")
    return mismatches


def _audit_sources(primary: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source = json.loads((primary / "source_identity_manifest.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if _sha256_file(CONTRACT_PATH) != source["contract_hash"]:
        mismatches.append("contract_hash")
    if _git("rev-parse", "HEAD") != source["git_sha"]:
        mismatches.append("git_sha")
    if _git("branch", "--show-current") != source["branch"]:
        mismatches.append("branch")
    for relative, expected in source["source_file_hashes"].items():
        path = REPO_ROOT / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            mismatches.append(f"source:{relative}")
    if (
        _sha256_file(OLD_PRIMARY / "artifact_manifest.json")
        != source["audit_artifact_manifest_hash"]
    ):
        mismatches.append("old_artifact_manifest_hash")
    if _sha256_file(OLD_PRIMARY / "independent_audit.json") != source["audit_independent_hash"]:
        mismatches.append("old_independent_audit_hash")
    old_primary_mismatches = _verify_declared_manifest(OLD_PRIMARY)
    old_exact_mismatches = _verify_declared_manifest(OLD_EXACT)
    _record(
        checks,
        "source_identities_and_audited_manifests",
        not mismatches and not old_primary_mismatches and not old_exact_mismatches,
        {
            "source_mismatches": mismatches,
            "old_primary_mismatches": old_primary_mismatches,
            "old_exact_mismatches": old_exact_mismatches,
            "declared_artifacts_per_tree": 38,
        },
    )
    return contract


def _audit_frozen_tree(
    contract: Mapping[str, Any], primary: Path, checks: list[dict[str, Any]]
) -> None:
    identity = contract["source_identity"]
    baseline = str(identity["frozen_lineage_baseline_commit"])
    prefix = "research/slrno-v2/20260714-regime-loop-handoff"
    listing = _git("ls-tree", "-r", baseline, prefix).splitlines()
    mismatches: list[str] = []
    for line in listing:
        metadata, path_text = line.split("\t", 1)
        expected_blob = metadata.split()[2]
        actual_blob = _git("hash-object", path_text)
        if actual_blob != expected_blob:
            mismatches.append(path_text)
    tree_object = _git("rev-parse", f"{baseline}:{prefix}")
    pre_run = json.loads((primary / "pre_run_tree_manifest.json").read_text(encoding="utf-8"))
    post_run = json.loads((primary / "post_run_tree_manifest.json").read_text(encoding="utf-8"))
    expected_tree = str(identity["frozen_historical_tree_git_object"])
    passed = (
        len(listing) == int(identity["required_frozen_blob_count"])
        and tree_object == expected_tree
        and str(pre_run["frozen_historical_tree_hash"]) == expected_tree
        and str(post_run["frozen_historical_tree_hash"]) == expected_tree
        and not mismatches
    )
    _record(
        checks,
        "frozen_historical_tree_immutability",
        passed,
        {
            "baseline": baseline,
            "blob_count": len(listing),
            "tree_object": tree_object,
            "pre_run_tree_object": pre_run["frozen_historical_tree_hash"],
            "post_run_tree_object": post_run["frozen_historical_tree_hash"],
            "mismatches": mismatches,
        },
    )


def _audit_baseline_counts(primary: Path, checks: list[dict[str, Any]]) -> None:
    recorded = json.loads(
        (primary / "audited_input_reconstruction.json").read_text(encoding="utf-8")
    )
    decisions = pd.read_parquet(
        OLD_PRIMARY / "causal_completed_bar_decisions.parquet",
        columns=["structural_event_eligibility", "is_run_entry"],
    )
    prefixes = pq.ParquetFile(OLD_PRIMARY / "active_prefix_ledger.parquet")
    completions = pq.ParquetFile(OLD_PRIMARY / "loop_completion_event_ledger.parquet")
    comparison = pd.read_parquet(
        OLD_PRIMARY / "legacy_v2_target_comparison_detail.parquet",
        columns=[
            "comparison_available",
            "semantics_differ",
            "registered_event_set_differs",
            "legacy_positive_count",
            "active_prefix_count",
        ],
    )
    outcomes = pd.read_parquet(
        OLD_PRIMARY / "first_next_loop_outcomes.parquet",
        columns=["primary_label"],
    )
    actual = {
        "completed_bar_decisions": len(decisions),
        "eligible_decisions": int(decisions["structural_event_eligibility"].sum()),
        "run_entry_rows": int(decisions["is_run_entry"].sum()),
        "active_prefix_rows": prefixes.metadata.num_rows,
        "completion_event_rows": completions.metadata.num_rows,
        "comparable_decisions": int(comparison["comparison_available"].sum()),
        "full_semantic_differences": int(
            (comparison["comparison_available"] & comparison["semantics_differ"]).sum()
        ),
        "registered_event_set_differences": int(
            (comparison["comparison_available"] & comparison["registered_event_set_differs"]).sum()
        ),
        "legacy_multiple_positive_decisions": int(
            (comparison["comparison_available"] & comparison["legacy_positive_count"].gt(1)).sum()
        ),
        "decisions_with_active_prefix": int(
            (comparison["comparison_available"] & comparison["active_prefix_count"].gt(0)).sum()
        ),
        "tied_registered_completion": int(
            outcomes["primary_label"].eq("TIED_REGISTERED_COMPLETION").sum()
        ),
        "no_registered_loop_within_horizon": int(
            outcomes["primary_label"].eq("NO_REGISTERED_LOOP_WITHIN_HORIZON").sum()
        ),
        "session_end": int(outcomes["primary_label"].eq("SESSION_END").sum()),
        "unregistered_loop": int(outcomes["primary_label"].eq("UNREGISTERED_LOOP").sum()),
        "unavailable_source": int(outcomes["primary_label"].eq("UNAVAILABLE").sum()),
    }
    old_counts = recorded["old_primary_outcome_counts"]
    expected = {
        "completed_bar_decisions": int(recorded["completed_bar_decisions"]),
        "eligible_decisions": int(recorded["causally_and_structurally_eligible_decisions"]),
        "run_entry_rows": int(recorded["run_entry_baseline_rows"]),
        "active_prefix_rows": int(recorded["active_prefix_ledger_rows"]),
        "completion_event_rows": int(recorded["registered_completion_event_rows"]),
        "comparable_decisions": int(recorded["legacy_v2_comparable_decisions"]),
        "full_semantic_differences": int(recorded["full_semantic_differences"]),
        "registered_event_set_differences": int(recorded["registered_event_set_differences"]),
        "legacy_multiple_positive_decisions": int(recorded["legacy_multiple_positive_decisions"]),
        "decisions_with_active_prefix": int(recorded["decisions_with_active_v2_prefixes"]),
        "tied_registered_completion": int(old_counts["TIED_REGISTERED_COMPLETION"]),
        "no_registered_loop_within_horizon": int(old_counts["NO_REGISTERED_LOOP_WITHIN_HORIZON"]),
        "session_end": int(old_counts["SESSION_END"]),
        "unregistered_loop": int(old_counts["UNREGISTERED_LOOP"]),
        "unavailable_source": int(old_counts["UNAVAILABLE"]),
    }
    _record(
        checks,
        "audited_baseline_counts",
        actual == expected,
        {"actual": actual, "recorded": expected},
    )


def _composite_final_root(path: Any) -> str | None:
    values = _parse_list(path)
    return _decompose_path(tuple(int(value) for value in values))["primitive_loop_id"]


def _independent_tie_rows() -> pd.DataFrame:
    outcomes = pd.read_parquet(
        OLD_PRIMARY / "first_next_loop_outcomes.parquet",
        columns=["decision_id", "primary_label"],
    )
    tied_ids = frozenset(
        outcomes.loc[
            outcomes["primary_label"].eq("TIED_REGISTERED_COMPLETION"), "decision_id"
        ].astype(str)
    )
    completions = pd.read_parquet(
        OLD_PRIMARY / "loop_completion_event_ledger.parquet",
        columns=[
            "decision_id",
            "semantic_loop_id",
            "primitive_loop_id",
            "motif_type",
            "full_path",
            "is_primary_completion",
        ],
    )
    completions = completions.loc[
        completions["is_primary_completion"].astype(bool)
        & completions["decision_id"].astype(str).isin(tied_ids)
    ]
    rows: list[dict[str, Any]] = []
    for decision_id, group in completions.groupby("decision_id", sort=True):
        roots: list[str] = []
        explicit: list[str] = []
        composites: list[str] = []
        unknown = False
        for event in group.itertuples(index=False):
            motif = str(event.motif_type)
            if motif in {"primitive", "repeat"}:
                root = str(event.primitive_loop_id) if pd.notna(event.primitive_loop_id) else None
                if root is not None:
                    explicit.append(root)
            elif motif == "composite":
                root = _composite_final_root(event.full_path)
                composites.append(str(event.semantic_loop_id))
            else:
                root = None
            if root is None:
                unknown = True
            else:
                roots.append(root)
        unique_roots = sorted(set(roots))
        unique_explicit = sorted(set(explicit))
        if unknown:
            tie_class = "UNKNOWN_TIE"
            rewrite = "UNAVAILABLE_STRUCTURAL_GAP"
        elif len(unique_roots) == 1:
            tie_class = "NESTED_SAME_PRIMITIVE_TIE"
            rewrite = unique_roots[0]
        elif len(unique_explicit) >= 2 and not composites:
            tie_class = "DISTINCT_PRIMITIVE_TIE"
            rewrite = "DISTINCT_PRIMITIVE_TIE"
        elif unique_explicit and composites:
            tie_class = "PRIMITIVE_COMPOSITE_TIE"
            rewrite = "DISTINCT_PRIMITIVE_TIE"
        elif len(composites) >= 2:
            tie_class = "DISTINCT_COMPOSITE_TIE"
            rewrite = "DISTINCT_COMPOSITE_TIE"
        else:
            tie_class = "UNKNOWN_TIE"
            rewrite = "UNAVAILABLE_STRUCTURAL_GAP"
        rows.append(
            {
                "decision_id": str(decision_id),
                "tie_class": tie_class,
                "tied_primitive_ids": unique_roots,
                "rewritten_primary_label": rewrite,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("decision_id").reset_index(drop=True)


def _audit_ties(primary: Path, checks: list[dict[str, Any]]) -> None:
    independent = _independent_tie_rows()
    recorded = (
        pd.read_parquet(
            primary / "tie_classification.parquet",
            columns=[
                "decision_id",
                "tie_class",
                "tied_primitive_ids",
                "rewritten_primary_label",
            ],
        )
        .sort_values("decision_id")
        .reset_index(drop=True)
    )
    recorded["tied_primitive_ids"] = recorded["tied_primitive_ids"].map(_parse_list)
    recorded["tie_class"] = recorded["tie_class"].astype(str)
    columns = list(independent.columns)
    equal = len(independent) == len(recorded)
    if equal:
        for column in columns:
            if column == "tied_primitive_ids":
                equal &= all(
                    left == right
                    for left, right in zip(independent[column], recorded[column], strict=True)
                )
            else:
                equal &= independent[column].equals(recorded[column])
    counts = independent["tie_class"].value_counts().to_dict()
    _record(
        checks,
        "independent_tie_classification",
        bool(equal and len(independent) == 11_003),
        {
            "total": len(independent),
            "counts": counts,
            "resolved_to_one_primitive": int(
                independent["rewritten_primary_label"].str.startswith("loop_p_").sum()
            ),
        },
    )


def _closure_schedule(group: pd.DataFrame) -> tuple[list[int], list[dict[str, Any]], np.ndarray]:
    ordered = group.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
    states = ordered["hard_state_legacy"].astype(int).to_numpy()
    event_mask = np.r_[True, states[1:] != states[:-1]]
    event_positions = np.flatnonzero(event_mask)
    event_for_bar = np.cumsum(event_mask, dtype=int) - 1
    event_states = states[event_positions].astype(int).tolist()
    event_bars = ordered.iloc[event_positions]["bar_ordinal"].astype(int).tolist()
    last_state: dict[int, int] = {}
    closure_indices: list[int] = []
    closures: list[dict[str, Any]] = []
    for event_index, state in enumerate(event_states):
        if state in last_state:
            start = last_state[state]
            path = tuple(event_states[start : event_index + 1])
            closures.append(
                {
                    "event_index": event_index,
                    "event_bar_ordinal": event_bars[event_index],
                    "path": path,
                    "identity": _decompose_path(path),
                }
            )
            closure_indices.append(event_index)
        last_state[state] = event_index
    return closure_indices, closures, event_for_bar


def _audit_unregistered_sample(primary: Path, checks: list[dict[str, Any]]) -> None:
    ledger = pd.read_parquet(primary / "unregistered_event_ledger.parquet")
    sample_ids = sorted(ledger["decision_id"].astype(str).unique())[:AUDIT_SAMPLE_SIZE]
    sample = ledger.loc[ledger["decision_id"].astype(str).isin(sample_ids)].copy()
    decisions = pd.read_parquet(
        OLD_PRIMARY / "causal_completed_bar_decisions.parquet",
        columns=[
            "decision_id",
            "symbol",
            "session",
            "bar_ordinal",
            "hard_state_legacy",
        ],
    )
    sample_keys = sample[["symbol", "session"]].drop_duplicates()
    source = decisions.merge(sample_keys, on=["symbol", "session"], how="inner")
    source_by_id = source.set_index("decision_id")
    mismatches: list[str] = []
    checked = 0
    for (symbol, session), group in source.groupby(["symbol", "session"], sort=True):
        closure_indices, closures, event_for_bar = _closure_schedule(group)
        closure_lookup = dict(zip(closure_indices, closures, strict=True))
        ordered = group.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
        local_position = {
            str(row.decision_id): index for index, row in enumerate(ordered.itertuples())
        }
        session_sample = sample.loc[sample["symbol"].eq(symbol) & sample["session"].eq(session)]
        for row in session_sample.itertuples(index=False):
            decision_id = str(row.decision_id)
            position = local_position[decision_id]
            current_event = int(event_for_bar[position])
            future = next((index for index in closure_indices if index > current_event), None)
            if future is None:
                mismatches.append(f"no_future:{decision_id}")
                continue
            reconstructed = closure_lookup[future]
            identity = reconstructed["identity"]
            source_row = source_by_id.loc[decision_id]
            bars_until = int(reconstructed["event_bar_ordinal"] - source_row["bar_ordinal"])
            recorded_path = tuple(int(value) for value in _parse_list(row.full_closed_path))
            if any(
                (
                    recorded_path != reconstructed["path"],
                    str(row.primitive_loop_id) != identity["primitive_loop_id"],
                    str(row.motif_type) != identity["motif_type"],
                    int(row.repeat_depth) != identity["repeat_depth"],
                    int(row.bars_until_completion) != bars_until,
                    bars_until < 1 or bars_until > HORIZON_BARS,
                )
            ):
                mismatches.append(decision_id)
            checked += 1
    full_identity_failures = 0
    for row in ledger.itertuples(index=False):
        identity = _decompose_path(_parse_list(row.full_closed_path))
        if identity["primitive_loop_id"] != str(row.primitive_loop_id) or identity[
            "motif_type"
        ] != str(row.motif_type):
            full_identity_failures += 1
    _record(
        checks,
        "unregistered_extraction_and_primitive_decomposition",
        checked == len(sample) and not mismatches and full_identity_failures == 0,
        {
            "deterministic_sample_rows": checked,
            "sample_mismatches": mismatches[:20],
            "full_identity_failures": full_identity_failures,
            "full_rows": len(ledger),
            "unique_primitives": int(ledger["primitive_loop_id"].nunique()),
        },
    )


def _top_share(values: pd.Series) -> float:
    return float(values.value_counts(dropna=False).iloc[0] / len(values)) if len(values) else 0.0


def _audit_candidate_support(primary: Path, checks: list[dict[str, Any]]) -> None:
    outcomes = pd.read_parquet(
        primary / "first_event_outcome_ledger_v2.parquet",
        columns=[
            "decision_id",
            "symbol",
            "session",
            "decision_timestamp",
            "clock_phase",
            "primitive_loop_id",
            "primitive_transition_length",
            "source_completeness",
        ],
    )
    events = outcomes.loc[outcomes["primitive_loop_id"].notna()].copy()
    timestamps = pd.to_datetime(events["decision_timestamp"], utc=True)
    events["_month"] = timestamps.dt.strftime("%Y-%m")
    events["_quarter"] = (
        timestamps.dt.year.astype(str) + "Q" + (((timestamps.dt.month - 1) // 3) + 1).astype(str)
    )
    rows: list[dict[str, Any]] = []
    for primitive_id, group in events.groupby("primitive_loop_id", sort=True):
        path = tuple(int(value) for value in str(primitive_id).removeprefix("loop_p_").split("-"))
        length_values = set(int(value) for value in group["primitive_transition_length"].dropna())
        length = next(iter(length_values)) if len(length_values) == 1 else len(path) - 1
        stock_sessions = group["symbol"].astype(str) + "|" + group["session"].astype(str)
        row = {
            "semantic_loop_id": str(primitive_id),
            "development_count": int(group["decision_id"].nunique()),
            "stock_breadth": int(group["symbol"].nunique()),
            "session_breadth": int(stock_sessions.nunique()),
            "month_breadth": int(group["_month"].nunique()),
            "quarter_breadth": int(group["_quarter"].nunique()),
            "clock_breadth": int(group["clock_phase"].nunique()),
            "top_stock_share": _top_share(group["symbol"]),
            "top_month_share": _top_share(group["_month"]),
            "transition_length": length,
            "complete_semantic_identity": len(length_values) == 1 and len(path) - 1 == length,
            "source_gap_free": bool(group["source_completeness"].fillna(False).all()),
        }
        row["support_pass"] = bool(
            row["development_count"] >= 100
            and row["session_breadth"] >= 50
            and row["stock_breadth"] >= 10
            and row["month_breadth"] >= 6
            and row["clock_breadth"] >= 3
            and row["top_stock_share"] <= 0.20
            and row["top_month_share"] <= 0.30
            and row["complete_semantic_identity"]
            and row["source_gap_free"]
            and length in {2, 3, 4, 5}
        )
        rows.append(row)
    independent = pd.DataFrame.from_records(rows).sort_values("semantic_loop_id")
    recorded = pd.read_parquet(primary / "candidate_loop_universe.parquet").sort_values(
        "semantic_loop_id"
    )
    columns = [
        "semantic_loop_id",
        "development_count",
        "stock_breadth",
        "session_breadth",
        "month_breadth",
        "quarter_breadth",
        "clock_breadth",
        "top_stock_share",
        "top_month_share",
        "transition_length",
        "complete_semantic_identity",
        "source_gap_free",
        "support_pass",
    ]
    merged = independent[columns].merge(
        recorded[columns],
        on="semantic_loop_id",
        suffixes=("_audit", "_recorded"),
        validate="one_to_one",
    )
    mismatches: list[str] = []
    for column in columns[1:]:
        left = merged[f"{column}_audit"]
        right = merged[f"{column}_recorded"]
        equal = (
            np.isclose(left.astype(float), right.astype(float), rtol=0.0, atol=1e-12)
            if column in {"top_stock_share", "top_month_share"}
            else left.astype(str).eq(right.astype(str)).to_numpy()
        )
        if not bool(np.asarray(equal).all()):
            mismatches.append(column)
    _record(
        checks,
        "candidate_support_reconstruction",
        len(independent) == len(recorded) and not mismatches,
        {
            "candidate_universe": len(independent),
            "support_passed": int(independent["support_pass"].sum()),
            "mismatched_fields": mismatches,
        },
    )


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _load_sessions() -> list[dict[str, Any]]:
    runs = pd.read_parquet(
        OLD_PRIMARY / "structural_session_runs.parquet",
        columns=[
            "symbol",
            "session",
            "state",
            "duration",
            "right_censored",
            "source_sequence_complete",
        ],
    )
    complete = runs.loc[runs["source_sequence_complete"].astype(bool)]
    records: list[dict[str, Any]] = []
    for (symbol, session), group in complete.groupby(["symbol", "session"], sort=True):
        records.append(
            {
                "symbol": str(symbol),
                "session": str(session),
                "states": tuple(group["state"].astype(int)),
                "durations": tuple(group["duration"].astype(int)),
                "right_censored": bool(group["right_censored"].iloc[-1]),
            }
        )
    return records


def _fit_null_inputs(sessions: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    initial = np.full(STATE_COUNT, 0.5, dtype=float)
    transitions = np.full((STATE_COUNT, STATE_COUNT), 0.5, dtype=float)
    np.fill_diagonal(transitions, 0.0)
    at_risk = np.zeros((STATE_COUNT, MAXIMUM_DURATION), dtype=np.int64)
    exits = np.zeros_like(at_risk)
    for session in sessions:
        states = tuple(int(value) for value in session["states"])
        durations = tuple(int(value) for value in session["durations"])
        initial[states[0]] += 1.0
        for origin, destination in zip(states[:-1], states[1:], strict=True):
            transitions[origin, destination] += 1.0
        for index, (state, duration) in enumerate(zip(states, durations, strict=True)):
            at_risk[state, :duration] += 1
            right_censored = index == len(states) - 1 and bool(session["right_censored"])
            if not right_censored:
                exits[state, duration - 1] += 1
    initial /= initial.sum()
    transitions /= transitions.sum(axis=1, keepdims=True)
    pooled_risk = at_risk.sum(axis=0).astype(float)
    pooled_exits = exits.sum(axis=0).astype(float)
    pooled_hazard = np.divide(
        pooled_exits + 0.5,
        pooled_risk + 1.0,
        out=np.zeros(MAXIMUM_DURATION, dtype=float),
        where=pooled_risk > 0,
    )
    hazard = np.zeros_like(at_risk, dtype=float)
    for state in range(STATE_COUNT):
        supported = at_risk[state] > 0
        hazard[state, supported] = (exits[state, supported] + 8.0 * pooled_hazard[supported]) / (
            at_risk[state, supported] + 8.0
        )
        backoff = ~supported & (pooled_risk > 0)
        hazard[state, backoff] = pooled_hazard[backoff]
    duration_cdf = np.zeros((STATE_COUNT, MAXIMUM_DURATION + 1), dtype=float)
    for state in range(STATE_COUNT):
        survival = 1.0
        pmf = np.zeros(MAXIMUM_DURATION, dtype=float)
        for age in range(MAXIMUM_DURATION):
            pmf[age] = survival * hazard[state, age]
            survival *= 1.0 - hazard[state, age]
        duration_cdf[state] = np.cumsum(np.r_[pmf, survival])
        duration_cdf[state, -1] = 1.0
    phase_transitions = np.zeros((3, STATE_COUNT, STATE_COUNT), dtype=float)
    for phase_index in range(3):
        phase_transitions[phase_index] = 2.0 * transitions
    for session in sessions:
        elapsed = 0
        states = tuple(int(value) for value in session["states"])
        durations = tuple(int(value) for value in session["durations"])
        for index, duration in enumerate(durations[:-1]):
            elapsed += duration
            phase_index = 0 if elapsed < 12 else 1 if elapsed < 60 else 2
            phase_transitions[phase_index, states[index], states[index + 1]] += 1.0
    for phase_index in range(3):
        np.fill_diagonal(phase_transitions[phase_index], 0.0)
    phase_transitions /= phase_transitions.sum(axis=2, keepdims=True)
    return {
        "initial": initial,
        "transitions": transitions,
        "duration_cdf": duration_cdf,
        "phase_transitions": phase_transitions,
        "hazard": hazard,
    }


def _sample_index(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, rng.random(), side="right"))


def _simulate_session(
    session_length: int,
    inputs: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    *,
    clock_conditioned: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    state = _sample_index(inputs["initial"], rng)
    states: list[int] = []
    durations: list[int] = []
    elapsed = 0
    while elapsed < session_length:
        states.append(state)
        selected = int(np.searchsorted(inputs["duration_cdf"][state], rng.random(), side="right"))
        duration = min(selected + 1, session_length - elapsed)
        durations.append(duration)
        elapsed += duration
        if elapsed >= session_length:
            break
        if clock_conditioned:
            phase_index = 0 if elapsed < 12 else 1 if elapsed < 60 else 2
            probabilities = inputs["phase_transitions"][phase_index, state]
        else:
            probabilities = inputs["transitions"][state]
        state = _sample_index(probabilities, rng)
    if sum(durations) != session_length:
        raise AssertionError("independent null changed session length")
    return tuple(states), tuple(durations)


def _null_first_event_counts(
    states: Sequence[int],
    durations: Sequence[int],
    candidate_ids: Sequence[str],
) -> np.ndarray:
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    stack: list[int] = []
    closure_events: list[int] = []
    closure_ids: list[str] = []
    for event_index, raw_state in enumerate(states):
        state = int(raw_state)
        if state not in stack:
            stack.append(state)
            continue
        stack_index = stack.index(state)
        core = tuple(stack[stack_index:])
        closure_events.append(event_index)
        closure_ids.append(_primitive_id(core))
        stack = stack[:stack_index] + [state]
    run_starts = np.r_[0, np.cumsum(np.asarray(durations, dtype=int))[:-1]]
    output = np.zeros(len(candidate_ids), dtype=np.int64)
    for decision in range(int(sum(durations))):
        current_event = int(np.searchsorted(run_starts, decision, side="right") - 1)
        next_position = next(
            (index for index, event in enumerate(closure_events) if event > current_event),
            None,
        )
        if next_position is None:
            continue
        bars_until = int(run_starts[closure_events[next_position]] - decision)
        candidate = candidate_index.get(closure_ids[next_position])
        if candidate is not None and 1 <= bars_until <= HORIZON_BARS:
            output[candidate] += 1
    return output


def _simulate_null_subset(
    sessions: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, np.ndarray],
    candidate_ids: Sequence[str],
    *,
    clock_conditioned: bool,
    seed: int,
) -> np.ndarray:
    selected_sessions = sessions[:NULL_SUBSET_SESSIONS]
    rng = np.random.default_rng(seed)
    draws = np.zeros((NULL_SUBSET_DRAWS, len(candidate_ids)), dtype=np.int64)
    for draw in range(NULL_SUBSET_DRAWS):
        for session in selected_sessions:
            length = int(sum(session["durations"]))
            states, durations = _simulate_session(
                length, inputs, rng, clock_conditioned=clock_conditioned
            )
            if min(states) < 0 or max(states) >= STATE_COUNT:
                raise AssertionError("independent null generated an invalid state")
            draws[draw] += _null_first_event_counts(states, durations, candidate_ids)
    return draws


def _audit_nulls(primary: Path, checks: list[dict[str, Any]]) -> None:
    results = pd.read_parquet(primary / "semi_markov_null_results.parquet").sort_values(
        "semantic_loop_id"
    )
    clock = pd.read_parquet(primary / "clock_null_results.parquet").sort_values("semantic_loop_id")
    deletions = pd.read_parquet(primary / "leave_one_stock_out_null_results.parquet")
    targets = pd.read_parquet(
        primary / "first_event_outcome_ledger_v2.parquet",
        columns=["primitive_loop_id"],
    )
    observed = (
        targets["primitive_loop_id"]
        .value_counts()
        .reindex(results["semantic_loop_id"], fill_value=0)
        .to_numpy(dtype=int)
    )
    semi_q = _benjamini_hochberg(results["semi_markov_p"].to_numpy(dtype=float))
    clock_q = _benjamini_hochberg(clock["clock_null_p"].to_numpy(dtype=float))
    expected_qualified = (
        results["support_pass"].astype(bool)
        & results["semi_markov_q"].le(0.10)
        & results["semi_markov_rate_ratio"].ge(1.20)
        & results["excess_count"].gt(0.0)
        & results["positive_excess_quarters"].ge(3)
        & results["leave_one_stock_out_minimum_rate_ratio"].gt(1.0)
    )
    reconstructed_deletion_minimum = (
        deletions.groupby("semantic_loop_id", sort=True)["semi_markov_rate_ratio_without_stock"]
        .min()
        .reindex(results["semantic_loop_id"])
        .to_numpy(dtype=float)
    )
    deletion_minimum_matches = np.allclose(
        reconstructed_deletion_minimum,
        results["leave_one_stock_out_minimum_rate_ratio"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    deletion_method_is_resolved = (
        deletions["recomputation_method"]
        .eq("session_resolved_semi_markov_draw_subtraction_without_rate_scaling")
        .all()
    )
    _record(
        checks,
        "observed_counts_bh_and_structural_qualification",
        bool(
            np.array_equal(observed, results["observed_count"].to_numpy(dtype=int))
            and np.allclose(semi_q, results["semi_markov_q"], rtol=0.0, atol=1e-12)
            and np.allclose(clock_q, clock["clock_null_q"], rtol=0.0, atol=1e-12)
            and expected_qualified.equals(results["structurally_qualified"].astype(bool))
            and deletion_minimum_matches
            and deletion_method_is_resolved
        ),
        {
            "candidates": len(results),
            "qualified": int(expected_qualified.sum()),
            "observed_count_mismatches": int(
                (observed != results["observed_count"].to_numpy(dtype=int)).sum()
            ),
            "maximum_semi_q_error": float(
                np.max(np.abs(semi_q - results["semi_markov_q"].to_numpy(dtype=float)))
            ),
            "maximum_clock_q_error": float(
                np.max(np.abs(clock_q - clock["clock_null_q"].to_numpy(dtype=float)))
            ),
            "leave_one_stock_out_rows": len(deletions),
            "leave_one_stock_out_minimum_matches": bool(deletion_minimum_matches),
            "leave_one_stock_out_rate_scaling_used": not bool(deletion_method_is_resolved),
        },
    )
    sessions = _load_sessions()
    inputs = _fit_null_inputs(sessions)
    dictionary = pd.read_csv(primary / "semantic_loop_dictionary_v2.csv")
    candidate_ids = dictionary.sort_values("selection_rank")["semantic_loop_id"].head(5).tolist()
    base_left = _simulate_null_subset(
        sessions, inputs, candidate_ids, clock_conditioned=False, seed=91_721
    )
    base_right = _simulate_null_subset(
        sessions, inputs, candidate_ids, clock_conditioned=False, seed=91_721
    )
    clock_left = _simulate_null_subset(
        sessions, inputs, candidate_ids, clock_conditioned=True, seed=91_722
    )
    clock_right = _simulate_null_subset(
        sessions, inputs, candidate_ids, clock_conditioned=True, seed=91_722
    )
    duration_24_supported = bool((inputs["hazard"][:, 23] > 0.0).any())
    _record(
        checks,
        "deterministic_semi_markov_and_clock_null_subset",
        bool(
            np.array_equal(base_left, base_right)
            and np.array_equal(clock_left, clock_right)
            and duration_24_supported
            and len(sessions) == 5_128
        ),
        {
            "sessions_fitted": len(sessions),
            "subset_sessions": NULL_SUBSET_SESSIONS,
            "draws": NULL_SUBSET_DRAWS,
            "candidate_ids": candidate_ids,
            "semi_markov_counts_sha256": hashlib.sha256(base_left.tobytes()).hexdigest(),
            "clock_counts_sha256": hashlib.sha256(clock_left.tobytes()).hexdigest(),
            "duration_24_supported": duration_24_supported,
            "terminal_right_censored_sessions": int(
                sum(bool(session["right_censored"]) for session in sessions)
            ),
        },
    )


def _audit_information_increment_semantics(primary: Path, checks: list[dict[str, Any]]) -> None:
    information = pd.read_csv(primary / "candidate_information_increment.csv")
    expected_model = information["strongest_retained_baseline"].map({"B4": "C4", "B5": "C5"})
    retained_baseline = np.where(
        information["strongest_retained_baseline"].eq("B4"),
        information["b4_log_loss"],
        information["b5_log_loss"],
    )
    reconstructed_increment = retained_baseline - information["candidate_aware_log_loss"]
    expected_qualified = (
        (reconstructed_increment > 0.0)
        & information["brier_improvement"].gt(0.0)
        & information["quarter_consistency"].ge(0.75)
    )
    _record(
        checks,
        "candidate_information_increment_beyond_retained_state_history",
        bool(
            expected_model.eq(information["candidate_aware_model"]).all()
            and np.allclose(
                retained_baseline,
                information["baseline_log_loss"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                reconstructed_increment,
                information["oof_log_loss_increment"],
                rtol=0.0,
                atol=1e-12,
            )
            and expected_qualified.equals(information["information_qualified"].astype(bool))
        ),
        {
            "candidates": len(information),
            "qualified": int(expected_qualified.sum()),
            "candidate_model_mismatches": int(
                (~expected_model.eq(information["candidate_aware_model"])).sum()
            ),
            "maximum_increment_error": float(
                np.max(
                    np.abs(
                        reconstructed_increment
                        - information["oof_log_loss_increment"].to_numpy(dtype=float)
                    )
                )
            ),
            "global_frequency_used_as_retained_baseline": bool(
                information["baseline_log_loss"].eq(information["b0_log_loss"]).all()
            ),
        },
    )


def _normalise_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_normalise_identity_value(item) for item in value]
    if hasattr(value, "item"):
        return _normalise_identity_value(value.item())
    return str(value)


def _dictionary_hash(dictionary: pd.DataFrame) -> str:
    fields = (
        "semantic_loop_id",
        "primitive_loop_id",
        "canonical_primitive_core",
        "canonical_primitive_path",
        "closed_path",
        "transition_length",
        "allowed_orientations",
        "reverse_path_id",
    )
    list_fields = {
        "canonical_primitive_core",
        "canonical_primitive_path",
        "closed_path",
        "allowed_orientations",
    }
    payload: list[dict[str, Any]] = []
    for row in dictionary.to_dict(orient="records"):
        identity: dict[str, Any] = {}
        for field in fields:
            if field not in row:
                continue
            raw_value = row[field]
            if raw_value is None or (not isinstance(raw_value, str) and bool(pd.isna(raw_value))):
                continue
            value = _parse_list(raw_value) if field in list_fields else raw_value
            identity[field] = _normalise_identity_value(value)
        payload.append(identity)
    payload.sort(key=lambda row: str(row["semantic_loop_id"]))
    encoded = json.dumps(
        {
            "dictionary_version": "semantic_loop_dictionary_first_event_v2",
            "entries": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _audit_dictionary(primary: Path, checks: list[dict[str, Any]]) -> None:
    nulls = pd.read_parquet(primary / "semi_markov_null_results.parquet")
    information = pd.read_csv(primary / "candidate_information_increment.csv")
    candidates = nulls.merge(
        information[
            [
                "semantic_loop_id",
                "information_qualified",
                "oof_log_loss_increment",
            ]
        ],
        on="semantic_loop_id",
        validate="one_to_one",
    )
    total_valid = int(
        pd.read_parquet(
            primary / "first_event_outcome_ledger_v2.parquet",
            columns=["primitive_loop_id"],
        )["primitive_loop_id"]
        .notna()
        .sum()
    )
    eligible = candidates.loc[
        candidates["support_pass"].astype(bool)
        & candidates["structurally_qualified"].astype(bool)
        & candidates["information_qualified"].astype(bool)
        & candidates["motif_type"].eq("primitive")
    ].copy()
    eligible["marginal_coverage"] = eligible["development_count"] / total_valid
    eligible = eligible.sort_values(
        [
            "oof_log_loss_increment",
            "semi_markov_rate_ratio",
            "marginal_coverage",
            "stock_breadth",
            "month_breadth",
            "transition_length",
            "semantic_loop_id",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected: list[str] = []
    path_rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for row in eligible.itertuples(index=False):
        marginal = float(row.marginal_coverage)
        if len(selected) >= 32:
            action = "STOP_MAXIMUM_DICTIONARY_SIZE"
        elif marginal < 0.005:
            action = "STOP_BELOW_MARGINAL_COVERAGE"
        else:
            action = "SELECT"
        if action == "SELECT":
            selected.append(str(row.semantic_loop_id))
            cumulative += marginal
        path_rows.append(
            {
                "semantic_loop_id": str(row.semantic_loop_id),
                "marginal_coverage": marginal,
                "cumulative_coverage": cumulative,
                "selection_action": action,
            }
        )
        if action != "SELECT":
            break
    dictionary = pd.read_csv(primary / "semantic_loop_dictionary_v2.csv").sort_values(
        "selection_rank"
    )
    path = pd.read_csv(primary / "dictionary_selection_path.csv")
    recorded_hash = json.loads((primary / "dictionary_hash.json").read_text(encoding="utf-8"))[
        "dictionary_hash"
    ]
    reconstructed_hash = _dictionary_hash(dictionary)
    path_equal = len(path_rows) == len(path)
    if path_equal:
        for audit_row, recorded_row in zip(path_rows, path.itertuples(index=False), strict=True):
            path_equal &= (
                audit_row["semantic_loop_id"] == str(recorded_row.semantic_loop_id)
                and audit_row["selection_action"] == str(recorded_row.selection_action)
                and math.isclose(
                    audit_row["marginal_coverage"],
                    float(recorded_row.marginal_coverage),
                    abs_tol=1e-12,
                )
                and math.isclose(
                    audit_row["cumulative_coverage"],
                    float(recorded_row.cumulative_coverage),
                    abs_tol=1e-12,
                )
            )
    legacy = pd.read_csv(primary / "legacy_to_dictionary_v2_mapping.csv")
    old_dictionary = pd.read_csv(OLD_PRIMARY / "semantic_loop_dictionary_v2.csv")
    legacy_complete = (
        len(legacy) == len(old_dictionary)
        and legacy["legacy_semantic_loop_id"].nunique() == len(old_dictionary)
        and set(legacy["legacy_semantic_loop_id"].astype(str))
        == set(old_dictionary["semantic_loop_id"].astype(str))
    )
    _record(
        checks,
        "dictionary_membership_selection_path_hash_and_legacy_mapping",
        bool(
            selected == dictionary["semantic_loop_id"].astype(str).tolist()
            and path_equal
            and reconstructed_hash == recorded_hash
            and legacy_complete
        ),
        {
            "eligible_before_marginal_stop": len(eligible),
            "selected": selected,
            "recorded_hash": recorded_hash,
            "reconstructed_hash": reconstructed_hash,
            "legacy_rows": len(legacy),
            "legacy_complete": bool(legacy_complete),
        },
    )


def _expected_primary_class(row: Any, selected: frozenset[str]) -> str:
    event = str(row.primary_event)
    non_loop = {
        "NO_LOOP_WITHIN_HORIZON",
        "SESSION_END",
        "DISTINCT_PRIMITIVE_TIE",
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
    }
    if event in non_loop:
        return event
    primitive = str(row.primitive_loop_id)
    if not primitive.startswith("loop_p_"):
        raise ValueError(f"identified event lacks primitive identity: {event}")
    return primitive if primitive in selected else "OTHER_PRIMITIVE_LOOP"


def _period_metrics(target: pd.DataFrame, period: str) -> dict[str, Any]:
    primitive = target["primitive_loop_id"].notna()
    selected = target["primary_class"].astype(str).str.startswith("loop_p_")
    unavailable = target["primary_class"].isin(["UNAVAILABLE_SOURCE", "UNAVAILABLE_STRUCTURAL_GAP"])
    eligible = ~unavailable
    primitive_count = int(primitive.sum())
    return {
        "period": period,
        "total_decisions": len(target),
        "eligible_decisions": int(eligible.sum()),
        "primitive_loop_event_decisions": primitive_count,
        "selected_dictionary_events": int(selected.sum()),
        "selected_dictionary_event_coverage": float(selected.sum() / primitive_count),
        "other_primitive_loop_count": int(target["primary_class"].eq("OTHER_PRIMITIVE_LOOP").sum()),
        "other_primitive_loop_share": float(
            target["primary_class"].eq("OTHER_PRIMITIVE_LOOP").sum() / primitive_count
        ),
        "no_loop_share": float(
            target["primary_class"].eq("NO_LOOP_WITHIN_HORIZON").sum() / max(int(eligible.sum()), 1)
        ),
        "session_end_share": float(
            target["primary_class"].eq("SESSION_END").sum() / max(int(eligible.sum()), 1)
        ),
        "genuine_tie_share": float(
            target["primary_class"].eq("DISTINCT_PRIMITIVE_TIE").sum() / max(int(eligible.sum()), 1)
        ),
        "unavailable_share": float(unavailable.mean()),
    }


def _audit_one_target(
    primary: Path,
    filename: str,
    period: str,
    selected: frozenset[str],
    checks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = pd.read_parquet(primary / filename)
    expected = pd.Series(
        (_expected_primary_class(row, selected) for row in target.itertuples(index=False)),
        index=target.index,
        dtype="object",
    )
    loop_rows = target.loc[target["primitive_loop_id"].notna()]
    valid_classes = selected | {
        "OTHER_PRIMITIVE_LOOP",
        "NO_LOOP_WITHIN_HORIZON",
        "SESSION_END",
        "DISTINCT_PRIMITIVE_TIE",
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
    }
    timestamps = pd.to_datetime(target["decision_timestamp"], utc=True)
    expected_year = 2024 if period == "development_2024" else 2025
    passed = bool(
        target["decision_id"].is_unique
        and target["primary_class"].eq(expected).all()
        and set(target["primary_class"].astype(str)).issubset(valid_classes)
        and loop_rows["bars_until_completion"].between(1, HORIZON_BARS).all()
        and timestamps.dt.year.eq(expected_year).all()
        and not target["primary_class"].astype(str).str.startswith("loop_r").any()
        and not target["primary_class"].astype(str).str.startswith("loop_c").any()
    )
    horizon_rows = int(loop_rows["bars_until_completion"].eq(HORIZON_BARS).sum())
    _record(
        checks,
        f"primary_target_precedence_{expected_year}",
        passed and horizon_rows > 0,
        {
            "rows": len(target),
            "class_mismatches": int((~target["primary_class"].eq(expected)).sum()),
            "completion_exactly_at_horizon": horizon_rows,
            "after_horizon": int(loop_rows["bars_until_completion"].gt(HORIZON_BARS).sum()),
            "classes": target["primary_class"].value_counts().to_dict(),
        },
    )
    return target, _period_metrics(target, period)


def _audit_targets_and_coverage(
    primary: Path, checks: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dictionary = pd.read_csv(primary / "semantic_loop_dictionary_v2.csv")
    selected = frozenset(dictionary["semantic_loop_id"].astype(str))
    development, development_metrics = _audit_one_target(
        primary,
        "first_event_outcome_ledger_v2.parquet",
        "development_2024",
        selected,
        checks,
    )
    validation, validation_metrics = _audit_one_target(
        primary,
        "validation_first_event_outcome_ledger_v2.parquet",
        "unchanged_retrospective_validation_2025",
        selected,
        checks,
    )
    recorded_development = pd.read_csv(primary / "development_coverage.csv").iloc[0]
    recorded_validation = pd.read_csv(primary / "validation_coverage.csv").iloc[0]
    metric_fields = [
        "total_decisions",
        "eligible_decisions",
        "primitive_loop_event_decisions",
        "selected_dictionary_events",
        "selected_dictionary_event_coverage",
        "other_primitive_loop_count",
        "other_primitive_loop_share",
        "no_loop_share",
        "session_end_share",
        "genuine_tie_share",
        "unavailable_share",
    ]
    mismatches: list[str] = []
    for label, actual, recorded in (
        ("development", development_metrics, recorded_development),
        ("validation", validation_metrics, recorded_validation),
    ):
        for field in metric_fields:
            left = float(actual[field])
            right = float(recorded[field])
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                mismatches.append(f"{label}:{field}")
    validation_ids = set(
        validation.loc[
            validation["primary_class"].astype(str).str.startswith("loop_p_"),
            "primary_class",
        ].astype(str)
    )
    no_validation_additions = validation_ids.issubset(selected)
    _record(
        checks,
        "development_and_unchanged_validation_coverage",
        not mismatches
        and no_validation_additions
        and len(development) == 424_583
        and len(validation) == 424_834,
        {
            "development": development_metrics,
            "validation": validation_metrics,
            "mismatches": mismatches,
            "validation_added_ids": sorted(validation_ids.difference(selected)),
        },
    )
    return development, validation, development_metrics, validation_metrics


def _audit_auxiliary_target_evidence(primary: Path, checks: list[dict[str, Any]]) -> None:
    auxiliary = pd.read_parquet(primary / "first_event_auxiliary_targets.parquet")
    outcomes = pd.read_parquet(
        primary / "first_event_outcome_ledger_v2.parquet",
        columns=[
            "decision_id",
            "primitive_loop_id",
            "previous_completed_primitive_loop",
        ],
    )
    legacy = pd.read_parquet(
        OLD_PRIMARY / "legacy_overlapping_targets.parquet",
        columns=["decision_id", "legacy_positive_semantic_ids"],
    )
    required = {
        "is_same_as_previous_primitive",
        "earliest_composite_completion",
        "first_component_completion",
        "final_component_completion",
        "component_primitive_ids",
        "component_boundaries",
        "earlier_primitive_completion_already_occurred",
        "composite_adds_information_beyond_primitive_sequence",
        "legacy_overlapping_positive_labels",
    }
    missing = sorted(required.difference(auxiliary.columns))
    merged = auxiliary.merge(
        outcomes, on="decision_id", suffixes=("", "_outcome"), validate="one_to_one"
    )
    loop_rows = merged["primitive_loop_id_outcome"].notna()
    expected_same = merged["primitive_loop_id_outcome"].eq(
        merged["previous_completed_primitive_loop"]
    )
    recorded_same = merged["is_same_as_previous_primitive"].eq(True)
    same_mismatches = int((recorded_same.loc[loop_rows] != expected_same.loc[loop_rows]).sum())
    legacy_lookup = {
        str(row.decision_id): [
            str(value) for value in _parse_list(row.legacy_positive_semantic_ids)
        ]
        for row in legacy.itertuples(index=False)
    }
    legacy_mismatches = sum(
        _parse_list(labels) != legacy_lookup.get(str(decision_id), [])
        for decision_id, labels in zip(
            auxiliary["decision_id"],
            auxiliary["legacy_overlapping_positive_labels"],
            strict=True,
        )
    )
    composites = auxiliary.loc[auxiliary["motif_type"].eq("composite")]
    composite_timestamp_order = pd.to_datetime(
        composites["first_component_completion"], utc=True
    ).le(pd.to_datetime(composites["final_component_completion"], utc=True))
    composite_final_matches = pd.to_datetime(composites["final_component_completion"], utc=True).eq(
        pd.to_datetime(composites["event_timestamp"], utc=True)
    )
    composite_component_evidence = composites["component_primitive_ids"].map(
        lambda value: len(_parse_list(value)) > 1
    )
    _record(
        checks,
        "auxiliary_repeat_composite_and_legacy_evidence",
        bool(
            not missing
            and len(auxiliary) == len(outcomes)
            and auxiliary["decision_id"].is_unique
            and same_mismatches == 0
            and legacy_mismatches == 0
            and composite_timestamp_order.all()
            and composite_final_matches.all()
            and composite_component_evidence.all()
            and composites["composite_adds_information_beyond_primitive_sequence"]
            .astype(bool)
            .all()
        ),
        {
            "rows": len(auxiliary),
            "missing_fields": missing,
            "same_primitive_mismatches": same_mismatches,
            "legacy_label_mismatches": int(legacy_mismatches),
            "composite_rows": len(composites),
            "composite_timestamp_order_failures": int((~composite_timestamp_order).sum()),
            "composite_final_timestamp_failures": int((~composite_final_matches).sum()),
        },
    )


def _family_for_row(row: Any) -> tuple[str, str]:
    primary = str(row.primary_class)
    if primary == "NO_LOOP_WITHIN_HORIZON":
        family = "NO_LOOP"
    elif primary == "SESSION_END":
        family = "SESSION_END"
    elif primary == "DISTINCT_PRIMITIVE_TIE":
        family = "DISTINCT_PRIMITIVE_TIE"
    elif primary in {"UNAVAILABLE_SOURCE", "UNAVAILABLE_STRUCTURAL_GAP"}:
        family = primary
    else:
        length = int(row.primitive_transition_length)
        if length == 2:
            family = "TWO_STATE_OSCILLATION"
        elif length == 3:
            family = "THREE_STATE_CYCLE"
        elif length == 4:
            family = "FOUR_STATE_CYCLE"
        elif length in {5, 6}:
            family = "FIVE_TO_SIX_STATE_CYCLE"
        else:
            family = "LONG_PRIMITIVE_CYCLE"
    repeat_depth = row.current_repeat_depth
    repeat_status = (
        "SAME_PRIMITIVE_REPEAT"
        if pd.notna(repeat_depth) and int(repeat_depth) > 1
        else "NEW_PRIMITIVE_AFTER_DIFFERENT_LOOP"
    )
    if family in {
        "NO_LOOP",
        "SESSION_END",
        "DISTINCT_PRIMITIVE_TIE",
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
    }:
        repeat_status = "NOT_APPLICABLE"
    return family, repeat_status


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum()) if len(counts) else 0.0


def _audit_family_mapping(
    primary: Path,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> dict[str, float]:
    combined = pd.concat(
        [
            development.assign(period="development_2024"),
            validation.assign(period="unchanged_retrospective_validation_2025"),
        ],
        ignore_index=True,
    )
    expected = [_family_for_row(row) for row in combined.itertuples(index=False)]
    expected_family = pd.Series([value[0] for value in expected], index=combined.index)
    expected_repeat = pd.Series([value[1] for value in expected], index=combined.index)
    recorded = pd.read_parquet(
        primary / "loop_family_mapping.parquet",
        columns=[
            "decision_id",
            "period",
            "loop_family",
            "repeat_status",
            "data_snapshot_hash",
        ],
    )
    comparison = combined[["decision_id", "period"]].copy()
    comparison["expected_family"] = expected_family
    comparison["expected_repeat"] = expected_repeat
    comparison = comparison.merge(recorded, on=["decision_id", "period"], validate="one_to_one")
    primitive_validation = validation.loc[validation["primitive_loop_id"].notna()].copy()
    primitive_validation["family"] = [
        _family_for_row(row)[0] for row in primitive_validation.itertuples(index=False)
    ]
    exact_entropy = _entropy(primitive_validation["primitive_loop_id"])
    family_entropy = _entropy(primitive_validation["family"])
    family_stability = pd.read_csv(primary / "family_stability.csv")
    run_metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    expected_hash_by_period = {
        "development_2024": str(run_metadata["development_snapshot_hash"]),
        "unchanged_retrospective_validation_2025": str(run_metadata["validation_snapshot_hash"]),
    }
    period_hash_matches = (
        comparison["data_snapshot_hash"].eq(comparison["period"].map(expected_hash_by_period)).all()
        and family_stability["data_snapshot_hash"]
        .eq(family_stability["period"].map(expected_hash_by_period))
        .all()
    )
    structural_rows = family_stability.loc[
        family_stability["loop_family"].isin(
            {
                "TWO_STATE_OSCILLATION",
                "THREE_STATE_CYCLE",
                "FOUR_STATE_CYCLE",
                "FIVE_TO_SIX_STATE_CYCLE",
                "LONG_PRIMITIVE_CYCLE",
            }
        )
    ].copy()
    expected_observed = (
        comparison.loc[comparison["expected_family"].isin(structural_rows["loop_family"])]
        .groupby(["period", "expected_family"])
        .size()
        .rename("independent_observed_count")
        .reset_index()
        .rename(columns={"expected_family": "loop_family"})
    )
    structural_rows = structural_rows.merge(
        expected_observed,
        on=["period", "loop_family"],
        how="left",
        validate="one_to_one",
    )
    structural_rows["independent_observed_count"] = structural_rows[
        "independent_observed_count"
    ].fillna(0)
    ratio_matches = np.allclose(
        structural_rows["semi_markov_rate_ratio"],
        (structural_rows["observed_count"] + 0.5)
        / (structural_rows["semi_markov_null_mean"] + 0.5),
        rtol=0.0,
        atol=1e-11,
    ) and np.allclose(
        structural_rows["clock_null_rate_ratio"],
        (structural_rows["observed_count"] + 0.5) / (structural_rows["clock_null_mean"] + 0.5),
        rtol=0.0,
        atol=1e-11,
    )
    q_matches = True
    for _, period_rows in structural_rows.groupby("period", sort=True):
        q_matches = q_matches and np.allclose(
            _benjamini_hochberg(period_rows["semi_markov_p"].to_numpy(dtype=float)),
            period_rows["semi_markov_q"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        q_matches = q_matches and np.allclose(
            _benjamini_hochberg(period_rows["clock_null_p"].to_numpy(dtype=float)),
            period_rows["clock_null_q"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    passed = bool(
        len(comparison) == len(combined)
        and comparison["expected_family"].eq(comparison["loop_family"]).all()
        and comparison["expected_repeat"].eq(comparison["repeat_status"]).all()
        and period_hash_matches
        and family_entropy < exact_entropy
        and len(structural_rows) == 10
        and structural_rows["observed_count"]
        .astype(int)
        .eq(structural_rows["independent_observed_count"].astype(int))
        .all()
        and structural_rows[
            ["semi_markov_null_mean", "clock_null_mean", "semi_markov_q", "clock_null_q"]
        ]
        .notna()
        .all()
        .all()
        and ratio_matches
        and q_matches
    )
    _record(
        checks,
        "topology_only_family_mapping",
        passed,
        {
            "rows": len(comparison),
            "family_mismatches": int(
                (~comparison["expected_family"].eq(comparison["loop_family"])).sum()
            ),
            "repeat_status_mismatches": int(
                (~comparison["expected_repeat"].eq(comparison["repeat_status"])).sum()
            ),
            "validation_exact_entropy_nats": exact_entropy,
            "validation_family_entropy_nats": family_entropy,
            "other_events_resolved_by_family": int(
                validation["primary_class"].eq("OTHER_PRIMITIVE_LOOP").sum()
            ),
            "family_structural_rows": len(structural_rows),
            "family_observed_count_mismatches": int(
                (
                    ~structural_rows["observed_count"]
                    .astype(int)
                    .eq(structural_rows["independent_observed_count"].astype(int))
                ).sum()
            ),
            "family_rate_ratio_reconstruction_pass": bool(ratio_matches),
            "family_bh_reconstruction_pass": bool(q_matches),
            "period_source_hashes_match": bool(period_hash_matches),
        },
    )
    return {"exact_entropy": exact_entropy, "family_entropy": family_entropy}


def _independent_prefix_bar_estimate_sample(
    primary: Path, features: pd.DataFrame, *, sample_size: int = 512
) -> tuple[int, int]:
    lookup: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
    dictionary = pd.read_csv(primary / "semantic_loop_dictionary_v2.csv")
    auxiliary = pd.read_csv(primary / "auxiliary_loop_motif_registry_v2.csv")
    path_rows: list[list[list[int]] | list[int]] = [
        _parse_list(value) for value in dictionary["allowed_orientations"]
    ]
    path_rows.extend(_parse_list(value) for value in auxiliary["full_path"])
    for raw_paths in path_rows:
        paths = raw_paths
        if paths and isinstance(paths[0], int):
            paths = [paths]
        for raw_path in paths:
            path = tuple(int(state) for state in raw_path)
            for progress in range(1, len(path)):
                lookup.setdefault(path[:progress], []).append((path, progress))
    maximum_progress = max(len(prefix) for prefix in lookup)
    with np.load(OLD_PRIMARY / "duration_model_v2.npz") as duration_model:
        hazard = np.asarray(duration_model["hazard"], dtype=float)
    mean_duration = np.zeros(hazard.shape[0], dtype=float)
    for state in range(hazard.shape[0]):
        survival = 1.0
        for probability in hazard[state]:
            mean_duration[state] += survival
            survival *= 1.0 - probability

    sample = features.head(sample_size).set_index("decision_id")
    wanted = set(sample.index.astype(str))
    source = pd.read_parquet(
        OLD_PRIMARY / "causal_completed_bar_decisions.parquet",
        columns=[
            "decision_id",
            "symbol",
            "session",
            "bar_ordinal",
            "hard_state_legacy",
            "hard_run_age",
            "structural_event_eligibility",
        ],
    ).sort_values(["symbol", "session", "bar_ordinal", "decision_id"], kind="mergesort")
    compared = 0
    mismatches = 0
    for _, group in source.groupby(["symbol", "session"], sort=False):
        state_events: list[int] = []
        previous_state: int | None = None
        for row in group.itertuples(index=False):
            if not bool(row.structural_event_eligibility):
                state_events = []
                previous_state = None
                continue
            state = int(row.hard_state_legacy)
            if state != previous_state:
                state_events.append(state)
                previous_state = state
            decision_id = str(row.decision_id)
            if decision_id not in wanted:
                continue
            estimates: list[float] = []
            for width in range(1, min(len(state_events), maximum_progress) + 1):
                for path, progress in lookup.get(tuple(state_events[-width:]), []):
                    age_index = int(row.hard_run_age)
                    if age_index >= hazard.shape[1]:
                        continue
                    survival = 1.0
                    current_wait = 0.0
                    for probability in hazard[state, age_index:]:
                        current_wait += survival
                        survival *= 1.0 - probability
                    estimates.append(
                        current_wait
                        + sum(float(mean_duration[value]) for value in path[progress:-1])
                    )
            expected = min(estimates) if estimates else math.nan
            recorded = float(sample.loc[decision_id, "minimum_bars_remaining_estimate"])
            if not (
                (math.isnan(expected) and math.isnan(recorded))
                or math.isclose(expected, recorded, rel_tol=0.0, abs_tol=1e-9)
            ):
                mismatches += 1
            compared += 1
            wanted.remove(decision_id)
            if not wanted:
                return compared, mismatches
    return compared, mismatches + len(wanted)


def _audit_prefix_features(primary: Path, checks: list[dict[str, Any]]) -> None:
    features = pd.read_parquet(primary / "compressed_active_prefix_features.parquet")
    source = pd.read_parquet(
        OLD_PRIMARY / "causal_completed_bar_decisions.parquet",
        columns=["decision_id", "structural_event_eligibility"],
    )
    expected_ids = set(
        source.loc[source["structural_event_eligibility"].astype(bool), "decision_id"].astype(str)
    )
    actual_ids = set(features["decision_id"].astype(str))
    schema = set(features.columns)
    forbidden = sorted(
        column
        for column in schema
        if column not in SAFETY
        and (
            any(fragment in column.lower() for fragment in FORBIDDEN_ECONOMIC_FRAGMENTS)
            or column
            in {
                "future_loop",
                "future_state",
                "bars_until_completion",
                "state_events_until_completion",
            }
        )
    )
    availability = pd.to_datetime(features["feature_available_timestamp"], utc=True).le(
        pd.to_datetime(features["decision_timestamp"], utc=True)
    )
    count_reconciliation = features["active_prefix_count"].eq(
        features[
            [
                "active_primitive_prefix_count",
                "active_repeat_prefix_count",
                "active_composite_prefix_count",
            ]
        ].sum(axis=1)
    )
    distance_reconciliation = features["active_prefix_count"].eq(
        features[
            [
                "prefixes_one_transition_away",
                "prefixes_two_transitions_away",
                "prefixes_three_or_more_transitions_away",
            ]
        ].sum(axis=1)
    )
    bar_estimate_sample_rows, bar_estimate_mismatches = _independent_prefix_bar_estimate_sample(
        primary, features
    )
    _record(
        checks,
        "compressed_prefix_causality_eligibility_and_reconciliation",
        bool(
            features["decision_id"].is_unique
            and actual_ids == expected_ids
            and availability.all()
            and count_reconciliation.all()
            and distance_reconciliation.all()
            and not forbidden
            and features["required_next_state_entropy"].ge(0.0).all()
            and bar_estimate_sample_rows == min(512, len(features))
            and bar_estimate_mismatches == 0
        ),
        {
            "rows": len(features),
            "expected_eligible_rows": len(expected_ids),
            "missing_ids": len(expected_ids.difference(actual_ids)),
            "extra_ids": len(actual_ids.difference(expected_ids)),
            "availability_violations": int((~availability).sum()),
            "count_reconciliation_failures": int((~count_reconciliation).sum()),
            "distance_reconciliation_failures": int((~distance_reconciliation).sum()),
            "independent_bar_estimate_sample_rows": bar_estimate_sample_rows,
            "independent_bar_estimate_mismatches": bar_estimate_mismatches,
            "forbidden_columns": forbidden,
        },
    )


def _audit_decision(
    primary: Path,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    development_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    family_metrics: Mapping[str, float],
    checks: list[dict[str, Any]],
) -> str:
    dictionary = pd.read_csv(primary / "semantic_loop_dictionary_v2.csv")
    replication = pd.read_csv(primary / "dictionary_entry_replication.csv")
    information = pd.read_csv(primary / "candidate_information_increment.csv")
    selected_ids = set(dictionary["semantic_loop_id"].astype(str))
    selected_validation = validation.loc[
        validation["primary_class"].astype(str).str.startswith("loop_p_")
    ]
    top_stock_share = float(selected_validation["symbol"].value_counts(normalize=True).iloc[0])
    other = validation.loc[validation["primary_class"].eq("OTHER_PRIMITIVE_LOOP")]
    other_counts = other["primitive_loop_id"].value_counts().to_numpy(dtype=float)
    other_probabilities = other_counts / other_counts.sum()
    other_top_share = float(other_probabilities[0])
    other_hhi = float(np.square(other_probabilities).sum())
    rr_share = float(replication["validation_rr_above_one"].astype(bool).mean())
    exact_vs_family = pd.read_csv(primary / "exact_vs_family_coverage.csv")
    family_coverage_stable_and_higher = bool(
        exact_vs_family["family_resolved_coverage"].min() >= 0.99
    )
    development_coverage = float(development_metrics["selected_dictionary_event_coverage"])
    validation_coverage = float(validation_metrics["selected_dictionary_event_coverage"])
    semantic_ids_stable = bool(
        pd.read_csv(primary / "structural_rank_stability.csv")["primitive_semantic_id_stable"]
        .astype(bool)
        .all()
    )
    coverage_collapsed = validation_coverage < development_coverage - 0.10
    structural_excess_reversed = rr_share < 0.50
    exact_gate = bool(
        len(dictionary) >= 8
        and development_coverage >= 0.50
        and validation_coverage >= 0.45
        and development_coverage - validation_coverage <= 0.10
        and rr_share >= 0.75
        and float(replication["validation_threshold_or_directional"].astype(bool).mean()) >= 0.50
        and top_stock_share <= 0.20
        and float(validation_metrics["genuine_tie_share"]) < 0.05
        and other_top_share <= 0.20
    )
    hybrid_gate = bool(
        information.loc[information["semantic_loop_id"].isin(selected_ids), "information_qualified"]
        .astype(bool)
        .all()
        and validation_coverage >= 0.30
        and other_top_share < 0.20
        and other_hhi < 0.10
        and family_metrics["family_entropy"] < family_metrics["exact_entropy"]
    )
    family_gate = bool(
        validation_coverage < 0.30
        and other_top_share < 0.20
        and other_hhi < 0.10
        and family_coverage_stable_and_higher
        and rr_share < 0.50
    )
    unstable_gate = not semantic_ids_stable or coverage_collapsed or structural_excess_reversed
    if unstable_gate:
        label = "semantic_loop_dictionary_not_stable"
        decision_rule_gap = False
    elif exact_gate:
        label = "exact_next_loop_identity_tractable_for_preregistered_forecast"
        decision_rule_gap = False
    elif hybrid_gate:
        label = "hybrid_exact_dictionary_plus_other_ready_for_forecast"
        decision_rule_gap = False
    elif family_gate:
        label = "topological_loop_family_target_preferred"
        decision_rule_gap = False
    else:
        label = "semantic_dictionary_experiment_blocked"
        decision_rule_gap = True
    recorded = json.loads((primary / "decision.json").read_text(encoding="utf-8"))
    _record(
        checks,
        "scientific_decision_gate",
        label == recorded["decision_label"]
        and decision_rule_gap == bool(recorded.get("decision_rule_gap", False)),
        {
            "reconstructed_label": label,
            "recorded_label": recorded["decision_label"],
            "exact_gate": exact_gate,
            "hybrid_gate": hybrid_gate,
            "family_gate": family_gate,
            "unstable_gate": unstable_gate,
            "decision_rule_gap": decision_rule_gap,
            "top_stock_share": top_stock_share,
            "other_top_share": other_top_share,
            "other_hhi": other_hhi,
            "validation_rr_above_one_share": rr_share,
        },
    )
    return label


def _audit_artifact_manifest(primary: Path, checks: list[dict[str, Any]]) -> None:
    payload = json.loads((primary / "artifact_manifest.json").read_text(encoding="utf-8"))
    recorded = {str(row["file"]): row for row in payload["artifacts"]}
    actual = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json", "independent_audit.json"}
    }
    mismatches: list[str] = []
    for name, path in actual.items():
        row = recorded.get(name)
        if row is None:
            mismatches.append(f"missing:{name}")
        elif row["sha256"] != _sha256_file(path) or int(row["bytes"]) != path.stat().st_size:
            mismatches.append(f"identity:{name}")
    mismatches.extend(f"extra:{name}" for name in sorted(set(recorded).difference(actual)))
    _record(
        checks,
        "artifact_manifest_input_hashes",
        not mismatches and int(payload["artifact_count"]) == len(actual),
        {"files": len(actual), "mismatches": mismatches},
    )


def _audit_safety_and_trace(primary: Path, checks: list[dict[str, Any]]) -> None:
    required_outputs = {
        "implementation_delta_census.csv",
        "source_identity_manifest.json",
        "pre_run_tree_manifest.json",
        "audited_input_reconstruction.json",
        "tie_classification.parquet",
        "tie_classification_summary.csv",
        "nested_completion_mapping.parquet",
        "tie_primary_label_rewrite.csv",
        "unregistered_event_ledger.parquet",
        "unregistered_primitive_census.parquet",
        "unregistered_repeat_census.csv",
        "unregistered_composite_census.csv",
        "unregistered_length_distribution.csv",
        "unregistered_vocabulary_concentration.csv",
        "candidate_loop_universe.parquet",
        "candidate_support.csv",
        "candidate_rejection_reasons.csv",
        "semi_markov_null_results.parquet",
        "leave_one_stock_out_null_results.parquet",
        "clock_null_results.parquet",
        "analytical_null_results.csv",
        "circular_null_results.csv",
        "candidate_structural_qualification.csv",
        "candidate_information_increment.csv",
        "candidate_information_calibration.csv",
        "candidate_history_baseline_comparison.csv",
        "semantic_loop_dictionary_v2.csv",
        "auxiliary_loop_motif_registry_v2.csv",
        "dictionary_selection_path.csv",
        "dictionary_hash.json",
        "legacy_to_dictionary_v2_mapping.csv",
        "first_event_target_contract.json",
        "first_event_outcome_ledger_v2.parquet",
        "first_event_class_counts.csv",
        "first_event_tie_details.parquet",
        "first_event_auxiliary_targets.parquet",
        "legacy_v2_dictionary_target_comparison.csv",
        "compressed_active_prefix_features.parquet",
        "compressed_prefix_feature_manifest.json",
        "prefix_feature_missingness.csv",
        "prefix_feature_distribution.csv",
        "development_coverage.csv",
        "validation_coverage.csv",
        "coverage_by_stock.csv",
        "coverage_by_month.csv",
        "coverage_by_clock.csv",
        "coverage_by_state.csv",
        "coverage_stock_deletions.csv",
        "dictionary_entry_replication.csv",
        "structural_rank_stability.csv",
        "loop_family_taxonomy.json",
        "loop_family_mapping.parquet",
        "loop_family_counts.csv",
        "exact_vs_family_coverage.csv",
        "family_stability.csv",
        "validation_leave_one_stock_out_null_results.parquet",
        "missingness_and_blockers.csv",
        "run_metadata.json",
        "artifact_manifest.json",
        "decision.json",
        "post_run_tree_manifest.json",
    }
    missing_outputs = sorted(name for name in required_outputs if not (primary / name).is_file())
    safety_failures: list[str] = []
    trace_failures: list[str] = []
    for path in sorted(primary.iterdir()):
        if path.suffix == ".parquet":
            columns = set(pq.read_schema(path).names)
            missing_safety = set(SAFETY).difference(columns)
            if missing_safety:
                safety_failures.append(f"{path.name}:{sorted(missing_safety)}")
            if not TRACE_PLACEHOLDERS.issubset(columns):
                trace_failures.append(
                    f"{path.name}:{sorted(TRACE_PLACEHOLDERS.difference(columns))}"
                )
        elif path.suffix == ".csv":
            columns = set(pd.read_csv(path, nrows=0).columns)
            missing_safety = set(SAFETY).difference(columns)
            if missing_safety:
                safety_failures.append(f"{path.name}:{sorted(missing_safety)}")
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, expected in SAFETY.items():
                if payload.get(key) != expected:
                    safety_failures.append(f"{path.name}:{key}")
    new_sources = [
        REPO_ROOT / "packages/stocker_research/src/stocker_research/first_event_target_v2.py",
        REPO_ROOT / "packages/stocker_research/src/stocker_research/loop_structural_nulls_v2.py",
        REPO_ROOT / "packages/stocker_research/src/stocker_research/loop_tie_resolution_v2.py",
        REPO_ROOT / "packages/stocker_research/src/stocker_research/prefix_features_v2.py",
        REPO_ROOT / "packages/stocker_research/src/stocker_research/semantic_loop_dictionary_v2.py",
        REPO_ROOT / "packages/stocker_research/src/stocker_research/unregistered_loop_census_v2.py",
        WORK_DIR / "run_semantic_loop_dictionary_coverage_v2.py",
        Path(__file__).resolve(),
    ]
    import_failures: list[str] = []
    for path in new_sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                lowered = name.lower()
                if any(token in lowered for token in ("broker", "order", "position", "ig_service")):
                    import_failures.append(f"{path.name}:{name}")
    selection_files = [
        primary / "candidate_loop_universe.parquet",
        primary / "semi_markov_null_results.parquet",
        primary / "clock_null_results.parquet",
        primary / "semantic_loop_dictionary_v2.csv",
        primary / "candidate_information_increment.csv",
    ]
    economic_columns: list[str] = []
    for path in selection_files:
        columns = (
            pq.read_schema(path).names
            if path.suffix == ".parquet"
            else list(pd.read_csv(path, nrows=0).columns)
        )
        economic_columns.extend(
            f"{path.name}:{column}"
            for column in columns
            if column not in SAFETY
            and any(fragment in column.lower() for fragment in FORBIDDEN_ECONOMIC_FRAGMENTS)
        )
    _record(
        checks,
        "artifact_safety_trace_and_required_outputs",
        not missing_outputs
        and not safety_failures
        and not trace_failures
        and not import_failures
        and not economic_columns,
        {
            "required_outputs_before_self_audit": len(required_outputs),
            "missing_outputs": missing_outputs,
            "safety_failures": safety_failures,
            "trace_failures": trace_failures,
            "broker_or_execution_imports": import_failures,
            "economic_selection_columns": economic_columns,
        },
    )


def _audit_exact_identity(primary: Path, exact: Path, checks: list[dict[str, Any]]) -> None:
    primary_files = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file() and path.name != "independent_audit.json"
    }
    exact_files = {
        path.name: path
        for path in exact.iterdir()
        if path.is_file() and path.name != "independent_audit.json"
    }
    mismatches: list[str] = []
    for name in sorted(set(primary_files) | set(exact_files)):
        if name not in primary_files or name not in exact_files:
            mismatches.append(f"missing:{name}")
        elif _sha256_file(primary_files[name]) != _sha256_file(exact_files[name]):
            mismatches.append(f"identity:{name}")
    _record(
        checks,
        "exact_rerun_byte_identity",
        not mismatches,
        {"compared_files": len(primary_files), "mismatches": mismatches},
    )


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name in {
            "artifact_manifest.json",
            "independent_audit.json",
        }:
            continue
        records.append(
            {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    return {"artifact_count": len(records), "artifacts": records}


def _update_report(audit: Mapping[str, Any]) -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    safety_line = (
        "`research_only=true` · `execution_enabled=false` · "
        "`order_placement=disabled` · `broker_connected=false` · "
        "`economic_outcomes_used=false` · `payoff_selection_used=false` · "
        "`production_runtime_modified=false` · `strategy_promotion=false`"
    )
    if safety_line not in report:
        title, remainder = report.split("\n", 1)
        report = f"{title}\n\n{safety_line}\n{remainder}"
    marker = "\n## 42. Independent audit\n"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    failed = audit["failed_checks"] or "none"
    audit_description = (
        "It imported no production module from this phase and independently "
        "reconstructed source identities, baseline counts, all tie classifications, "
        "a deterministic unregistered-event sample, every primitive decomposition, "
        "candidate support, BH q-values, fixed-seed semi-Markov and clock-null "
        "subsets, dictionary membership/hash, target precedence, period coverage, "
        "family mapping, prefix eligibility, safety, frozen-tree immutability, and "
        "exact identity."
    )
    report += (
        "\n\n## 42. Independent audit\n\n"
        f"The separate auditor passed {audit['passed_checks']} of "
        f"{audit['check_count']} checks. {audit_description} "
        f"Failed checks: `{failed}`.\n\n"
        "## 43. Exact rerun\n\n"
        "Primary and exact-rerun trees were byte-identical across "
        f"{audit['exact_compared_files']} pre-audit files. The identical auditor "
        "result, updated decision, and manifest were then written to both trees.\n"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")


def audit(primary: Path, exact: Path) -> dict[str, Any]:
    metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
    AUDIT_IDENTITY.clear()
    AUDIT_IDENTITY.update(
        {
            "run_id": metadata["run_id"],
            "git_sha": metadata["git_sha"],
            "contract_hash": metadata["contract_hash"],
            "data_snapshot_hash": metadata["data_snapshot_hash"],
            "dictionary_version": metadata["dictionary_version"],
            "dictionary_hash": metadata["dictionary_hash"],
            "state_model_version": metadata["state_model_version"],
        }
    )
    checks: list[dict[str, Any]] = []
    contract = _audit_sources(primary, checks)
    _audit_frozen_tree(contract, primary, checks)
    _audit_baseline_counts(primary, checks)
    _audit_ties(primary, checks)
    _audit_unregistered_sample(primary, checks)
    _audit_candidate_support(primary, checks)
    _audit_nulls(primary, checks)
    _audit_information_increment_semantics(primary, checks)
    _audit_dictionary(primary, checks)
    development, validation, development_metrics, validation_metrics = _audit_targets_and_coverage(
        primary, checks
    )
    _audit_auxiliary_target_evidence(primary, checks)
    family_metrics = _audit_family_mapping(primary, development, validation, checks)
    _audit_prefix_features(primary, checks)
    decision_label = _audit_decision(
        primary,
        development,
        validation,
        development_metrics,
        validation_metrics,
        family_metrics,
        checks,
    )
    _audit_artifact_manifest(primary, checks)
    _audit_safety_and_trace(primary, checks)
    _audit_exact_identity(primary, exact, checks)
    exact_check = next(check for check in checks if check["check"] == "exact_rerun_byte_identity")
    overall = all(check["passed"] for check in checks)
    payload: dict[str, Any] = {
        "auditor": "independent_semantic_loop_dictionary_coverage_v2",
        "production_phase_imported": False,
        "check_count": len(checks),
        "passed_checks": sum(check["passed"] for check in checks),
        "failed_checks": [check["check"] for check in checks if not check["passed"]],
        "overall_pass": overall,
        "exact_rerun_status": "byte_identical" if exact_check["passed"] else "mismatch",
        "exact_compared_files": int(exact_check["evidence"]["compared_files"]),
        "scientific_decision": decision_label,
        "checks": checks,
    }
    for destination in (primary, exact):
        _write_json(destination / "independent_audit.json", payload)
        decision = json.loads((destination / "decision.json").read_text(encoding="utf-8"))
        decision["independent_audit_pass"] = overall
        decision["exact_rerun_status"] = payload["exact_rerun_status"]
        decision["next_loop_predictor_justified"] = bool(
            overall
            and decision_label
            in {
                "exact_next_loop_identity_tractable_for_preregistered_forecast",
                "hybrid_exact_dictionary_plus_other_ready_for_forecast",
                "topological_loop_family_target_preferred",
            }
        )
        _write_json(destination / "decision.json", decision)
        _write_json(destination / "artifact_manifest.json", _artifact_manifest(destination))
    post_mismatches = []
    primary_files = {path.name: path for path in primary.iterdir() if path.is_file()}
    exact_files = {path.name: path for path in exact.iterdir() if path.is_file()}
    for name in sorted(set(primary_files) | set(exact_files)):
        missing = name not in primary_files or name not in exact_files
        if missing or _sha256_file(primary_files[name]) != _sha256_file(exact_files[name]):
            post_mismatches.append(name)
    if post_mismatches:
        raise AssertionError(f"auditor broke exact identity: {post_mismatches}")
    _update_report(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    arguments = parser.parse_args()
    payload = audit(arguments.primary.resolve(), arguments.exact.resolve())
    print(json.dumps(payload, sort_keys=True, indent=2, default=_json_default))
    if not payload["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
