"""Fail-closed core for the frozen per-loop movement-quality shadow.

The current contract has no eligible cycle.  The validation code remains
complete so a non-empty input cannot be smuggled into the dormant ledger and
so the required structural, conditional-quality, and joint probabilities are
kept as distinct fields.  Nothing in this module reads or evaluates outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
QUALIFIED_GRADES = frozenset(
    {"good_movement_quality", "high_movement_quality"}
)
FORBIDDEN_BATCH_FRAGMENTS = (
    "actual",
    "observed",
    "outcome",
    "direction",
    "signed_return",
    "long_short",
    "pnl",
    "position",
    "broker",
    "order",
    "spread",
    "slippage",
    "cost",
)
BASE_COLUMNS = (
    "prediction_id",
    "contract_id",
    "issued_at_utc",
    "start_timestamp",
    "session_date",
    "symbol_norm",
    "state",
    "cycle_id",
    "movement_quality_grade",
    "structural_probability",
)


class DormantNoEligibleCycles(RuntimeError):
    """Raised before any candidate prediction batch is read."""


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_snapshot(root: Path, relative_to: Path) -> dict[str, Any]:
    """Return a stable file-content snapshot, excluding mutable directory metadata."""

    root = root.resolve()
    relative_to = relative_to.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.resolve().relative_to(relative_to)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": files,
        "snapshot_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def safety_payload() -> dict[str, Any]:
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "economic_edge_claim": False,
        "deployment_enabled": False,
    }


def validate_contract(contract: dict[str, Any]) -> list[str]:
    checks = {
        "contract_id": contract.get("contract_id")
        == "frozen_loop_quality_shadow_v1",
        "research_only": contract.get("research_only") is True,
        "live_disabled": contract.get("live_ordering_enabled") is False,
        "orders_disabled": contract.get("order_placement") == "disabled",
        "not_trading_performance": contract.get(
            "movement_quality_not_trading_performance"
        )
        is True,
        "no_current_issuance": contract["issuance"].get(
            "current_issuance_permitted"
        )
        is False,
        "no_outcome_evaluator": contract["issuance"].get(
            "outcome_evaluator_in_this_runtime_permitted"
        )
        is False,
        "dormant": contract["eligibility_freeze"].get("activation_state")
        == "dormant_no_eligible_cycles",
        "zero_eligible": contract["eligibility_freeze"].get(
            "eligible_cycle_ids"
        )
        == [],
        "no_unqualified_surface": contract["eligibility_freeze"].get(
            "unqualified_cycles_may_surface"
        )
        is False,
        "separate_axes": contract["prediction_semantics"].get(
            "structural_and_conditional_probabilities_must_remain_separate"
        )
        is True,
        "chain_rule": contract["prediction_semantics"].get(
            "chain_rule_must_hold_rowwise"
        )
        is True,
        "not_summed": contract["prediction_semantics"].get(
            "probabilities_may_not_be_summed_across_overlapping_cycles"
        )
        is True,
        "outcomes_closed": contract["integrity"].get("outcomes_opened") is False,
        "final_scoring_complete": contract.get("final_certification", {}).get(
            "sealed_scoring_complete"
        )
        is True,
        "final_zero_qualified": contract.get("final_certification", {}).get(
            "qualified_good_or_high_cycles"
        )
        == 0,
        "independent_audit_complete": contract.get("final_certification", {}).get(
            "independent_post_score_audit_status"
        )
        == "complete_passed",
        "independent_audit_48_of_48": contract.get("final_certification", {}).get(
            "independent_post_score_audit_check_count"
        )
        == 48
        and contract.get("final_certification", {}).get(
            "independent_post_score_audit_all_passed"
        )
        is True,
        "both_snapshot_domains_declared": contract["integrity"].get(
            "snapshot_hashes_are_not_expected_to_match"
        )
        is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError(f"quality shadow contract checks failed: {failed}")
    return sorted(checks)


def validate_provisional_tiers(frame: pd.DataFrame) -> list[str]:
    required = {
        "period",
        "cycle_id",
        "h6_grade",
        "h12_grade",
        "h24_grade",
        "global_grade",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"provisional tier columns missing: {missing}")
    if len(frame) != 20 or frame["cycle_id"].nunique() != 20:
        raise AssertionError("expected exactly twenty unique frozen cycles")
    if set(frame["period"].astype(str)) != {"2024_oof"}:
        raise AssertionError("provisional tiers are not the 2024 OOF cohort")
    cycle_ids = sorted(frame["cycle_id"].astype(str))
    if cycle_ids != [f"cycle_{index:02d}" for index in range(1, 21)]:
        raise AssertionError("frozen cycle identity drift")
    if not frame["global_grade"].astype(str).eq("unqualified").all():
        raise AssertionError(
            "a globally qualified provisional cycle requires a separately frozen "
            "non-dormant contract"
        )
    return cycle_ids


def validate_final_tiers(frame: pd.DataFrame) -> list[str]:
    required = {
        "cycle_id",
        "provisional_2024_oof_grade",
        "development_2025_grade",
        "backward_2023_grade",
        "final_grade",
        "prospective_validated",
        "economic_edge_claim",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"final tier columns missing: {missing}")
    if len(frame) != 20 or frame["cycle_id"].nunique() != 20:
        raise AssertionError("expected exactly twenty unique final cycle tiers")
    cycle_ids = sorted(frame["cycle_id"].astype(str))
    if cycle_ids != [f"cycle_{index:02d}" for index in range(1, 21)]:
        raise AssertionError("final frozen cycle identity drift")
    grade_columns = (
        "provisional_2024_oof_grade",
        "development_2025_grade",
        "backward_2023_grade",
        "final_grade",
    )
    if not all(frame[column].astype(str).eq("unqualified").all() for column in grade_columns):
        raise AssertionError("a non-unqualified final grade requires a new contract")
    prospective = frame["prospective_validated"].astype(str).str.lower()
    economic = frame["economic_edge_claim"].astype(str).str.lower()
    if not prospective.eq("false").all() or not economic.eq("false").all():
        raise AssertionError("final tier interpretation drift")
    return cycle_ids


def probability_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            suffix = f"{target}__h{horizon}"
            columns.extend(
                (
                    f"conditional_q75__{suffix}",
                    f"conditional_q90__{suffix}",
                    f"joint_j75__{suffix}",
                    f"joint_j90__{suffix}",
                )
            )
    return tuple(columns)


def required_prediction_columns() -> tuple[str, ...]:
    return (*BASE_COLUMNS, *probability_columns())


def validate_prediction_batch(
    frame: pd.DataFrame, contract: dict[str, Any]
) -> pd.DataFrame:
    """Validate only prediction-time fields; no outcome field is accepted."""

    validate_contract(contract)
    missing = sorted(set(required_prediction_columns()).difference(frame.columns))
    if missing:
        raise AssertionError(f"prediction columns missing: {missing}")
    forbidden = sorted(
        column
        for column in frame.columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_BATCH_FRAGMENTS)
    )
    if forbidden:
        raise AssertionError(f"forbidden prediction columns: {forbidden}")
    if frame.empty:
        raise AssertionError("empty batches are not appended")
    if frame["prediction_id"].astype(str).duplicated().any():
        raise AssertionError("prediction IDs must be unique")
    if set(frame["contract_id"].astype(str)) != {contract["contract_id"]}:
        raise AssertionError("prediction contract ID drift")
    eligible = set(contract["eligibility_freeze"]["eligible_cycle_ids"])
    if not eligible:
        raise DormantNoEligibleCycles(
            "quality shadow is dormant: zero cycles passed the frozen global "
            "good/high movement-quality rule"
        )
    cycle_ids = set(frame["cycle_id"].astype(str))
    if not cycle_ids.issubset(eligible):
        raise AssertionError("an unqualified cycle attempted to surface")
    grades = set(frame["movement_quality_grade"].astype(str))
    if not grades.issubset(QUALIFIED_GRADES):
        raise AssertionError("only good/high movement-quality grades may surface")

    structural = pd.to_numeric(
        frame["structural_probability"], errors="raise"
    ).to_numpy(dtype=float)
    probabilities = frame.loc[:, probability_columns()].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    all_values = np.column_stack((structural, probabilities))
    if not np.isfinite(all_values).all():
        raise AssertionError("non-finite probability")
    if all_values.min() < -1e-12 or all_values.max() > 1.0 + 1e-12:
        raise AssertionError("probability outside [0, 1]")

    for target in TARGETS:
        for horizon in HORIZONS:
            suffix = f"{target}__h{horizon}"
            q75 = frame[f"conditional_q75__{suffix}"].to_numpy(dtype=float)
            q90 = frame[f"conditional_q90__{suffix}"].to_numpy(dtype=float)
            j75 = frame[f"joint_j75__{suffix}"].to_numpy(dtype=float)
            j90 = frame[f"joint_j90__{suffix}"].to_numpy(dtype=float)
            if (q90 > q75 + 1e-12).any() or (j90 > j75 + 1e-12).any():
                raise AssertionError("ordered movement-quality probability drift")
            if not np.allclose(j75, structural * q75, atol=1e-12, rtol=1e-12):
                raise AssertionError("p75 chain-rule identity failed")
            if not np.allclose(j90, structural * q90, atol=1e-12, rtol=1e-12):
                raise AssertionError("p90 chain-rule identity failed")
    return frame.loc[:, required_prediction_columns()].copy()


def validate_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        stored = record.pop("record_sha256")
        expected = sha256_bytes(canonical_json_bytes(record))
        if stored != expected:
            raise AssertionError(f"ledger hash mismatch at line {line_number}")
        if record.get("previous_record_sha256") != previous:
            raise AssertionError(f"ledger chain mismatch at line {line_number}")
        if int(record.get("sequence", -1)) != len(records) + 1:
            raise AssertionError("ledger sequence drift")
        record["record_sha256"] = stored
        records.append(record)
        previous = stored
    return records
