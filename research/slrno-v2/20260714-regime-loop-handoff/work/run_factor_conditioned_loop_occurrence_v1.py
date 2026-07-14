"""Phase-separated factor-conditioned loop-occurrence experiment.

This file owns orchestration only.  Causal feature/model mechanics live in
``factor_conditioned_loop_occurrence_core.py`` and statistical evaluation
lives in ``factor_conditioned_loop_occurrence_eval.py``.  Fit/validation may
resolve only 2024 inputs.  Later paths are unreachable until an independently
written, hash-bound authorization marker passes and the scoring namespace is
proved pristine.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-factor-conditioned-loop-occurrence-v1.json"
CONTRACT_SHA256 = "ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d"
CORE_SOURCE = HERE / "factor_conditioned_loop_occurrence_core.py"
EVALUATION_SOURCE = HERE / "factor_conditioned_loop_occurrence_eval.py"
AUDITOR_SOURCE = HERE / "audit_factor_conditioned_loop_occurrence_v1.py"
OUT = Path("/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711")

HEADS = ("qpattern", "qlimited4", "qfull9")
LAMBDA_GRID = (0.0001, 0.0003, 0.001, 0.003)
OUTER_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
FULL_SELECTION_MONTHS = tuple(f"2024-{month:02d}" for month in range(4, 13))
INNER_SCHEDULE = {
    outer: tuple(f"2024-{month:02d}" for month in range(4, int(outer[-2:])))
    for outer in OUTER_MONTHS
}
LAMBDA_TIE_TOLERANCE = 1e-12

FIT_COMPLETE_NAME = "fit_complete.json"
FIT_MANIFEST_NAME = "complete_fit_artifact_manifest.json"
AUDIT_RESULT_NAME = "pre_score_audit.json"
AUTHORIZATION_NAME = "pre_score_authorization.json"
MODEL_BUNDLE_NAME = "model_bundle.npz"
REQUIRED_SCORE_INPUT_ARTIFACTS = frozenset(
    {
        MODEL_BUNDLE_NAME,
        "fit_source_manifest_2024.json",
        "full_fit_audit_2024.json",
        "full_lambda_selection_2024.csv",
        "full_replay_inputs_2024.npz",
        "gates_2024.json",
        "grid_replay_inputs_2024.npz",
        "lambda_grid_scores_2024.csv",
        "oof_predictions_2024.parquet",
        "outer_replay_inputs_2024.npz",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if is_dataclass(value):
        return safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def _matches_exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual is expected
    return actual == expected


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_production_modules() -> tuple[ModuleType, ModuleType]:
    return (
        _load_module(CORE_SOURCE, "factor_loop_occurrence_core"),
        _load_module(EVALUATION_SOURCE, "factor_loop_occurrence_eval"),
    )


def load_contract() -> dict[str, Any]:
    actual = sha256(CONTRACT)
    if actual != CONTRACT_SHA256:
        raise AssertionError(
            f"factor-conditioned loop contract changed: {actual} != {CONTRACT_SHA256}"
        )
    contract = json.loads(CONTRACT.read_text())
    if contract.get("contract_id") != "factor_conditioned_loop_occurrence_v1":
        raise AssertionError("wrong contract id")
    if contract.get("research_only") is not True:
        raise AssertionError("research-only label changed")
    if contract.get("live_ordering_enabled") is not False:
        raise AssertionError("live-ordering safety label changed")
    if contract.get("order_placement") != "disabled":
        raise AssertionError("order-placement safety label changed")
    if contract.get("deployment_enabled") is not False:
        raise AssertionError("deployment safety label changed")
    frozen_grid = tuple(float(value) for value in contract["models"]["ridge_lambda_grid"])
    if frozen_grid != LAMBDA_GRID:
        raise AssertionError("lambda grid changed")
    frozen_outer = tuple(contract["periods"]["outer_oof_months"])
    if frozen_outer != OUTER_MONTHS:
        raise AssertionError("outer schedule changed")
    literal = {
        key: tuple(value)
        for key, value in contract["causal_nested_selection"]["literal_outer_schedule"].items()
    }
    if literal != INNER_SCHEDULE:
        raise AssertionError("inner schedule changed")
    if tuple(
        contract["causal_nested_selection"]["full_fit_lambda_validation_months"]
    ) != FULL_SELECTION_MONTHS:
        raise AssertionError("full-fit selection months changed")
    if tuple(contract["causal_nested_selection"]["lambda_selected_separately_per_head"]) != HEADS:
        raise AssertionError("head-specific lambda selection changed")
    if contract["periods"]["forbidden"] != [2026, "prospective_shadow"]:
        raise AssertionError("forbidden-period boundary changed")
    if Path(contract["planned_artifact_root"]) != OUT:
        raise AssertionError("planned artifact root changed")
    placeholder = contract["feature_construction"]["provider_scan_and_factor_table"]
    placeholder_audit = placeholder["fit_2024_placeholder_audit"]
    if (
        int(placeholder_audit["discarded_rows"]) != 5539
        or int(placeholder_audit["all_four_OHLC_null_rows"]) != 5539
        or int(placeholder_audit["partial_null_rows"]) != 0
        or int(placeholder_audit["run_entry_natural_key_intersection"]) != 0
        or placeholder_audit["canonical_discarded_key_sha256"]
        != "0925d79df089152ff694326c6b3b28d89e03e7ac6885dd4493112fa845bfc997"
        or "only_after_score_authorization" not in placeholder["later_placeholder_rule"]
    ):
        raise AssertionError("frozen placeholder-cleanup contract changed")
    return contract


def select_lambda(
    month_scores: Mapping[float, Mapping[str, float]],
    months: Sequence[str],
) -> tuple[float, dict[float, float]]:
    """Select one head's lambda by equal mean validation-month loss."""

    required_months = tuple(str(month) for month in months)
    if not required_months:
        raise AssertionError("lambda selection needs validation months")
    objectives: dict[float, float] = {}
    for value in LAMBDA_GRID:
        by_month = month_scores.get(value)
        if by_month is None:
            raise AssertionError(f"missing lambda grid row: {value}")
        if set(required_months).difference(by_month):
            raise AssertionError(f"missing validation month for lambda {value}")
        losses = np.asarray([by_month[month] for month in required_months], dtype=float)
        if not np.isfinite(losses).all():
            raise AssertionError("non-finite lambda selection loss")
        objectives[value] = float(losses.mean())
    minimum = min(objectives.values())
    tied = [
        value
        for value in LAMBDA_GRID
        if objectives[value] <= minimum + LAMBDA_TIE_TOLERANCE
    ]
    return max(tied), objectives


def select_head_lambdas(
    grid_scores: pd.DataFrame,
    months: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    required = {"head", "lambda", "validation_month", "log_loss"}
    missing = required.difference(grid_scores.columns)
    if missing:
        raise AssertionError(f"lambda-grid result lacks columns: {sorted(missing)}")
    selected: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for head in HEADS:
        head_rows = grid_scores.loc[grid_scores["head"].eq(head)]
        month_scores: dict[float, dict[str, float]] = {}
        for value in LAMBDA_GRID:
            selected_rows = head_rows.loc[np.isclose(head_rows["lambda"], value)]
            if selected_rows.duplicated("validation_month").any():
                raise AssertionError("duplicate head/lambda/month selection result")
            month_scores[value] = {
                str(row.validation_month): float(row.log_loss)
                for row in selected_rows.itertuples(index=False)
            }
        choice, objectives = select_lambda(month_scores, months)
        selected[head] = choice
        for value in LAMBDA_GRID:
            rows.append(
                {
                    "head": head,
                    "lambda": value,
                    "selection_months": json.dumps(list(months), separators=(",", ":")),
                    "equal_month_mean_log_loss": objectives[value],
                    "selected": value == choice,
                }
            )
    return selected, pd.DataFrame(rows)


def _prepared_value(prepared: Any, name: str, default: Any = None) -> Any:
    if isinstance(prepared, Mapping):
        return prepared.get(name, default)
    return getattr(prepared, name, default)


def _require_frame(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return value


def _normalize_grid_result(value: Any, validation_month: str) -> pd.DataFrame:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping) and "scores" in value:
        value = value["scores"]
    frame = _require_frame(value, "score_lambda_grid result").copy()
    if "lambda" not in frame and "ridge_lambda" in frame:
        frame = frame.rename(columns={"ridge_lambda": "lambda"})
    if "validation_month" not in frame:
        frame["validation_month"] = validation_month
    if not frame["validation_month"].astype(str).eq(validation_month).all():
        raise AssertionError("grid result contains the wrong validation month")
    expected = {
        (head, value)
        for head in HEADS
        for value in LAMBDA_GRID
    }
    observed = {
        (str(row.head), float(row.lambda_))
        for row in frame.rename(columns={"lambda": "lambda_"}).itertuples(index=False)
    }
    if observed != expected or len(frame) != len(expected):
        raise AssertionError("grid result is not the exact head x lambda product")
    return frame


def _assert_exact_prediction_population(
    observed_rows: pd.DataFrame,
    expected_rows: pd.DataFrame,
    label: str,
) -> None:
    observed = _require_frame(observed_rows, f"{label} predictions")
    expected = _require_frame(expected_rows, f"{label} expected population")
    population_columns = ["anchor_id", "cycle_id", "target"]
    for name, frame in (("observed", observed), ("expected", expected)):
        missing = set(population_columns).difference(frame.columns)
        if missing:
            raise AssertionError(
                f"{label} {name} population lacks columns: {sorted(missing)}"
            )
        if frame.duplicated(["anchor_id", "cycle_id"]).any():
            raise AssertionError(f"duplicate anchor-cycle in {label} {name} population")
    comparison = expected.loc[:, population_columns].merge(
        observed.loc[:, population_columns],
        on=["anchor_id", "cycle_id"],
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
        validate="one_to_one",
    )
    if len(comparison) != len(expected) or not comparison["_merge"].eq("both").all():
        raise AssertionError(
            f"{label} does not reproduce the exact expected anchor-cycle population"
        )
    expected_target = pd.to_numeric(
        comparison["target_expected"], errors="raise"
    ).to_numpy(int)
    observed_target = pd.to_numeric(
        comparison["target_observed"], errors="raise"
    ).to_numpy(int)
    if not np.array_equal(expected_target, observed_target):
        raise AssertionError(f"{label} target changed from the frozen label population")


def _validate_replay_inputs(value: Any, label: str) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise AssertionError(f"{label} replay inputs must be a nonempty mapping")
    required = {"target", "offset", "anchor_id", "cycle_index"}
    for head in HEADS:
        required.update(
            {
                f"{head}__data",
                f"{head}__indices",
                f"{head}__indptr",
                f"{head}__shape",
            }
        )
    missing = required.difference(value)
    if missing:
        raise AssertionError(f"{label} replay inputs lack arrays: {sorted(missing)}")
    arrays = {str(key): np.asarray(item) for key, item in value.items()}
    rows = len(arrays["target"])
    if rows <= 0 or any(
        len(arrays[key]) != rows for key in ("offset", "anchor_id", "cycle_index")
    ):
        raise AssertionError(f"{label} replay row arrays are empty or misaligned")
    for head in HEADS:
        shape = arrays[f"{head}__shape"].reshape(-1)
        indptr = arrays[f"{head}__indptr"].reshape(-1)
        data = arrays[f"{head}__data"].reshape(-1)
        indices = arrays[f"{head}__indices"].reshape(-1)
        if (
            len(shape) != 2
            or int(shape[0]) != rows
            or int(shape[1]) <= 0
            or len(indptr) != rows + 1
            or len(data) != len(indices)
        ):
            raise AssertionError(f"{label} {head} CSR replay arrays are inconsistent")
    return arrays


def _normalize_fold_result(
    value: Any,
    validation_month: str,
    expected_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Any, Mapping[str, np.ndarray]]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        predictions = value.get("predictions")
        audit = value.get("audit", {})
        replay_inputs = value.get("replay_inputs", {})
    else:
        predictions = getattr(value, "predictions", None)
        audit = getattr(value, "audit", {})
        replay_inputs = getattr(value, "replay_inputs", {})
    frame = _require_frame(predictions, "fit_predict_fold predictions").copy()
    if "validation_month" not in frame:
        frame["validation_month"] = validation_month
    if not frame["validation_month"].astype(str).eq(validation_month).all():
        raise AssertionError("fold prediction month mismatch")
    required = {"anchor_id", "cycle_id", "target", "qhistory", "qold_limited_path", *HEADS}
    missing = required.difference(frame.columns)
    if missing:
        raise AssertionError(f"fold predictions lack columns: {sorted(missing)}")
    if frame.duplicated(["anchor_id", "cycle_id"]).any():
        raise AssertionError("duplicate anchor-cycle OOF prediction")
    if expected_rows is not None:
        _assert_exact_prediction_population(frame, expected_rows, "OOF fold")
    if not isinstance(audit, Mapping) or not audit:
        raise AssertionError("fold fit audit must be a nonempty mapping")
    replay_inputs = _validate_replay_inputs(
        replay_inputs, f"{validation_month} outer fold"
    )
    return frame, audit, replay_inputs


def _evaluation_payload(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("evaluation result must be a mapping or dataclass")
    payload = dict(value)
    if "primary_pass" not in payload or not isinstance(payload["primary_pass"], (bool, np.bool_)):
        raise AssertionError("evaluation result lacks a boolean primary_pass")
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise TypeError("evaluation artifacts must be a mapping")
    mandatory = {
        "support",
        "overall",
        "ranking",
        "calibration",
        "comparisons",
        "bootstrap",
        "slices",
        "falsification",
        "gates",
    }
    missing = mandatory.difference(artifacts)
    if missing:
        raise AssertionError(
            f"evaluation result lacks mandatory artifacts: {sorted(missing)}"
        )
    gates = artifacts["gates"]
    if not isinstance(gates, Mapping) or not _matches_exact(
        gates.get("primary_pass"), bool(payload["primary_pass"])
    ):
        raise AssertionError("evaluation primary_pass disagrees with gates artifact")
    payload["artifacts"] = dict(artifacts)
    return payload


def evaluation_artifact_names(
    artifacts: Mapping[str, Any], period: str
) -> dict[str, Any]:
    """Convert evaluator semantic keys into deterministic on-disk names."""

    tag = "2024" if period == "2024_oof" else str(period)
    output: dict[str, Any] = {}
    for key, value in artifacts.items():
        if Path(str(key)).suffix in {".json", ".csv", ".parquet", ".npz"}:
            name = str(key).replace("{period}", tag)
        else:
            extension = ".csv" if isinstance(value, pd.DataFrame) else ".json"
            name = f"{key}_{tag}{extension}"
        if name in output:
            raise AssertionError(f"duplicate evaluation artifact name: {name}")
        output[name] = value
    return output


def _write_value(path: Path, value: Any) -> None:
    if path.suffix == ".json":
        write_json(path, value)
    elif path.suffix == ".csv":
        _require_frame(value, path.name).to_csv(path, index=False)
    elif path.suffix == ".parquet":
        _require_frame(value, path.name).to_parquet(path, index=False)
    elif path.suffix == ".npz":
        if not isinstance(value, Mapping):
            raise TypeError(f"{path.name} NPZ payload must be a mapping")
        np.savez_compressed(path, **value)
    else:
        raise AssertionError(f"unsupported artifact extension: {path.name}")


def _validated_artifact_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AssertionError("artifact name must be a nonempty string")
    relative = Path(value)
    if relative.is_absolute() or relative.name != value or value in {".", ".."}:
        raise AssertionError(f"invalid artifact name: {value}")
    return value


def write_artifacts(root: Path, artifacts: Mapping[str, Any]) -> list[str]:
    reserved = {
        FIT_COMPLETE_NAME,
        FIT_MANIFEST_NAME,
        AUDIT_RESULT_NAME,
        AUTHORIZATION_NAME,
        "scoring_complete.json",
    }
    written: list[str] = []
    for raw_name in sorted(artifacts):
        name = _validated_artifact_name(raw_name)
        if name in reserved:
            raise AssertionError(f"invalid or reserved artifact name: {name}")
        path = root / name
        if path.exists():
            raise FileExistsError(path)
        _write_value(path, artifacts[name])
        written.append(name)
    return written


def build_artifact_manifest(
    root: Path,
    names: Iterable[str],
    *,
    later_period_paths_resolved: bool = False,
) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(set(names)):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name] = {"sha256": sha256(path), "size": path.stat().st_size}
    return {
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "artifacts": entries,
        "later_period_paths_resolved": bool(later_period_paths_resolved),
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def verify_artifact_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    required_names: Iterable[str] = (),
    forbid_later_period_artifacts: bool = False,
) -> None:
    if manifest.get("contract_sha256") != CONTRACT_SHA256:
        raise AssertionError("fit manifest contract mismatch")
    if manifest.get("runner_sha256") != sha256(Path(__file__)):
        raise AssertionError("runner changed after fit")
    entries = manifest.get("artifacts")
    if not isinstance(entries, Mapping) or not entries:
        raise AssertionError("empty fit artifact manifest")
    names = {_validated_artifact_name(name) for name in entries}
    missing_required = set(required_names).difference(names)
    if missing_required:
        raise AssertionError(
            f"fit manifest lacks required score inputs: {sorted(missing_required)}"
        )
    if forbid_later_period_artifacts:
        forbidden = sorted(
            name
            for name in names
            if (
                name in SCORING_OUTPUT_NAMES
                or "_2025" in Path(name).stem
                or "_2023" in Path(name).stem
                or name.startswith("scoring_")
            )
        )
        if forbidden:
            raise AssertionError(
                f"fit manifest contains a later-period artifact: {forbidden}"
            )
    for name, metadata in entries.items():
        name = _validated_artifact_name(name)
        if not isinstance(metadata, Mapping):
            raise AssertionError(f"invalid manifest metadata: {name}")
        path = root / name
        if path.is_symlink():
            raise AssertionError(f"fit artifact may not be a symlink: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != metadata.get("sha256"):
            raise AssertionError(f"fit artifact changed: {name}")
        if path.stat().st_size != int(metadata.get("size", -1)):
            raise AssertionError(f"fit artifact size changed: {name}")


def _assert_pristine_fit_root(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "fit-only requires a pristine artifact root; archive the prior bundle"
        )


SCORING_OUTPUT_NAMES = (
    "predictions_2025.parquet",
    "predictions_2023.parquet",
    "metrics_2025.csv",
    "metrics_2023.csv",
    "calibration_2025.csv",
    "calibration_2023.csv",
    "ranking_2025.csv",
    "ranking_2023.csv",
    "slices_2025.csv",
    "slices_2023.csv",
    "falsification_2025.json",
    "falsification_2023.json",
    "gates_2025.json",
    "gates_2023.json",
    "irregular_cohort_2025.json",
    "pinned_path_replay_2025.json",
    "pinned_path_replay_2023.json",
    "transfer_decision.json",
    "scoring_source_manifest.json",
    "scoring_complete.json",
    "summary.json",
    "post_score_audit.json",
)


def assert_pristine_scoring_namespace(
    root: Path, fit_manifest: Mapping[str, Any] | None = None
) -> None:
    if root.is_symlink():
        raise AssertionError("scoring artifact root may not be a symlink")
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_entries = (
        set(fit_manifest.get("artifacts", {}))
        if isinstance(fit_manifest, Mapping)
        else set()
    )
    allowed = {
        *manifest_entries,
        FIT_COMPLETE_NAME,
        FIT_MANIFEST_NAME,
        AUDIT_RESULT_NAME,
        AUTHORIZATION_NAME,
    }
    unexpected = sorted(
        path.name
        for path in root.iterdir()
        if path.name not in allowed or path.is_dir() or path.is_symlink()
    )
    if unexpected:
        raise FileExistsError(
            f"score-only requires a pristine scoring namespace: {unexpected}"
        )


def validate_pre_score_authorization(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate every lock without importing core or resolving later paths."""

    fit_path = root / FIT_COMPLETE_NAME
    manifest_path = root / FIT_MANIFEST_NAME
    audit_path = root / AUDIT_RESULT_NAME
    authorization_path = root / AUTHORIZATION_NAME
    if root.is_symlink():
        raise AssertionError("scoring artifact root may not be a symlink")
    for path in (fit_path, manifest_path, audit_path, authorization_path):
        if path.is_symlink():
            raise AssertionError(f"pre-score marker may not be a symlink: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
    fit = json.loads(fit_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    authorization = json.loads(authorization_path.read_text())

    required = set(
        load_contract()["independent_audit_contract"]["authorization_marker_required_fields"]
    )
    missing = required.difference(authorization)
    if missing:
        raise AssertionError(f"authorization marker lacks fields: {sorted(missing)}")
    exact = {
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "fit_complete_sha256": sha256(fit_path),
        "complete_fit_artifact_manifest_sha256": sha256(manifest_path),
        "auditor_source_sha256": sha256(AUDITOR_SOURCE),
        "auditor_result_sha256": sha256(audit_path),
        "auditor_all_passed": True,
        "development_2024_primary_pass": True,
        "scoring_authorized": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "shadow_tree_read": False,
        "shadow_tree_written": False,
    }
    mismatch = {
        key: {"expected": value, "actual": authorization.get(key)}
        for key, value in exact.items()
        if not _matches_exact(authorization.get(key), value)
    }
    if mismatch:
        raise AssertionError(f"invalid pre-score authorization: {mismatch}")
    if audit.get("all_passed") is not True or audit.get("scoring_authorized") is not True:
        raise AssertionError("independent audit result does not authorize scoring")
    expected_fit = {
        "status": "fit_frozen_pending_independent_pre_score_audit",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "development_2024_primary_pass": True,
        "scoring_authorized": False,
        "later_period_paths_resolved": False,
        "later_period_rows_read": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    invalid_fit = {
        key: {"expected": value, "actual": fit.get(key)}
        for key, value in expected_fit.items()
        if not _matches_exact(fit.get(key), value)
    }
    if invalid_fit:
        raise AssertionError(f"invalid fit-complete phase or safety claim: {invalid_fit}")
    if fit.get("complete_fit_artifact_manifest_sha256") != sha256(manifest_path):
        raise AssertionError("fit-complete manifest mismatch")
    source_hashes = fit.get("production_source_hashes", {})
    expected_sources = {
        CORE_SOURCE.name: sha256(CORE_SOURCE),
        EVALUATION_SOURCE.name: sha256(EVALUATION_SOURCE),
    }
    if source_hashes != expected_sources:
        raise AssertionError("production core/evaluation changed after fit")
    expected_manifest = {
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "later_period_paths_resolved": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    invalid_manifest = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_manifest.items()
        if not _matches_exact(manifest.get(key), value)
    }
    if invalid_manifest:
        raise AssertionError(f"invalid fit manifest phase or safety claim: {invalid_manifest}")
    # Only follow artifact names after the authorization marker has bound the
    # exact manifest bytes.  Names are restricted to this root and fit-only
    # period labels before any artifact content is opened.
    verify_artifact_manifest(
        root,
        manifest,
        required_names=REQUIRED_SCORE_INPUT_ARTIFACTS,
        forbid_later_period_artifacts=True,
    )
    return fit, audit, authorization, manifest


def _source_hashes() -> dict[str, str]:
    return {
        CORE_SOURCE.name: sha256(CORE_SOURCE),
        EVALUATION_SOURCE.name: sha256(EVALUATION_SOURCE),
    }


def run_self_test_only(
    core_module: ModuleType | None = None,
    evaluation_module: ModuleType | None = None,
) -> dict[str, Any]:
    load_contract()
    if core_module is None or evaluation_module is None:
        core_module, evaluation_module = load_production_modules()
    core_result = core_module.self_tests()
    evaluation_result = evaluation_module.self_tests()
    if evaluation_result is None:
        evaluation_result = {"status": "passed_without_data"}
    # Exercise the exact independent selector without touching source data.
    synthetic = {
        value: {month: 1.0 + value for month in FULL_SELECTION_MONTHS}
        for value in LAMBDA_GRID
    }
    selected, _ = select_lambda(synthetic, FULL_SELECTION_MONTHS)
    if selected != min(LAMBDA_GRID):
        raise AssertionError("synthetic lambda self-test failed")
    return {
        "status": "self_tests_passed",
        "contract_sha256": CONTRACT_SHA256,
        "core": core_result,
        "evaluation": evaluation_result,
        "later_period_paths_resolved": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def _prepare_2024(core_module: ModuleType) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    core_module.validate_contract(CONTRACT)
    prepared = core_module.prepare_2024()
    cycles = _require_frame(core_module.load_cycles(), "cycles")
    anchors = _require_frame(_prepared_value(prepared, "anchors"), "prepared anchors")
    expanded = _prepared_value(prepared, "expanded")
    if expanded is None:
        expanded = core_module.expand_compatible_labels(
            anchors,
            cycles,
            expected_rows=759212,
            expected_positives=46630,
        )
    expanded = _require_frame(expanded, "expanded 2024 labels")
    if (
        len(anchors) != 110949
        or len(expanded) != 759212
        or int(expanded["target"].sum()) != 46630
        or len(cycles) != 20
    ):
        raise AssertionError("2024 population changed")
    if "symbol_norm" in anchors and anchors["symbol_norm"].nunique() != 22:
        raise AssertionError("2024 stock population changed")
    return prepared, cycles, expanded


def _validated_population_result(value: Any, period: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("population validation must be a mapping")
    if not _matches_exact(value.get("pass"), True):
        raise AssertionError(f"{period} population validation failed closed")
    return value


def run_validate_only(
    core_module: ModuleType | None = None,
    evaluation_module: ModuleType | None = None,
) -> dict[str, Any]:
    contract = load_contract()
    if core_module is None or evaluation_module is None:
        core_module, evaluation_module = load_production_modules()
    prepared, cycles, expanded = _prepare_2024(core_module)
    validation = _validated_population_result(
        evaluation_module.validate_population(
            anchors=_prepared_value(prepared, "anchors"),
            expanded=expanded,
            cycles=cycles,
            period="2024",
            contract=contract,
        ),
        "2024",
    )
    return {
        "status": "validated_without_fit",
        "contract_sha256": CONTRACT_SHA256,
        "anchors": len(_prepared_value(prepared, "anchors")),
        "compatible_rows": len(expanded),
        "positive_rows": int(expanded["target"].sum()),
        "cycles": len(cycles),
        "population_validation": validation,
        "outer_schedule": {key: list(value) for key, value in INNER_SCHEDULE.items()},
        "full_selection_months": list(FULL_SELECTION_MONTHS),
        "separate_lambda_heads": list(HEADS),
        "later_period_paths_resolved": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def _grid_scores_2024(
    core_module: ModuleType,
    prepared: Any,
    expanded: pd.DataFrame,
    cycles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    parts: list[pd.DataFrame] = []
    replay_inputs: dict[str, np.ndarray] = {}
    for validation_month in FULL_SELECTION_MONTHS:
        result = core_module.score_lambda_grid(
            prepared=prepared,
            expanded=expanded,
            cycles=cycles,
            validation_month=validation_month,
            lambdas=LAMBDA_GRID,
            heads=HEADS,
        )
        parts.append(_normalize_grid_result(result, validation_month))
        if isinstance(result, Mapping):
            month_replay = _validate_replay_inputs(
                result.get("replay_inputs", {}),
                f"{validation_month} lambda grid",
            )
            prefix = validation_month.replace("-", "_")
            for key, value in month_replay.items():
                replay_inputs[f"{prefix}__{key}"] = np.asarray(value)
    frame = pd.concat(parts, ignore_index=True)
    expected_rows = len(FULL_SELECTION_MONTHS) * len(HEADS) * len(LAMBDA_GRID)
    if len(frame) != expected_rows:
        raise AssertionError("incomplete 2024 lambda grid")
    return frame, replay_inputs


def _outer_oof_2024(
    core_module: ModuleType,
    prepared: Any,
    expanded: pd.DataFrame,
    cycles: pd.DataFrame,
    grid_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any], dict[str, np.ndarray]]:
    predictions: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    audits: list[Any] = []
    replay_inputs: dict[str, np.ndarray] = {}
    anchors = _require_frame(_prepared_value(prepared, "anchors"), "prepared anchors")
    for outer_month in OUTER_MONTHS:
        chosen, rows = select_head_lambdas(grid_scores, INNER_SCHEDULE[outer_month])
        rows.insert(0, "outer_month", outer_month)
        selections.append(rows)
        result = core_module.fit_predict_fold(
            prepared=prepared,
            validation_month=outer_month,
            selected_lambdas=chosen,
        )
        expected_anchor_ids = set(
            anchors.loc[anchors["month"].astype(str).eq(outer_month), "anchor_id"]
        )
        expected_rows = expanded.loc[
            expanded["anchor_id"].isin(expected_anchor_ids),
            ["anchor_id", "cycle_id", "target"],
        ]
        frame, audit, fold_replay = _normalize_fold_result(
            result, outer_month, expected_rows
        )
        predictions.append(frame)
        audits.append({"outer_month": outer_month, "selected_lambdas": chosen, "audit": audit})
        prefix = outer_month.replace("-", "_")
        for key, value in fold_replay.items():
            replay_inputs[f"{prefix}__{key}"] = np.asarray(value)
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["anchor_id", "cycle_id"], kind="stable"
    ).reset_index(drop=True)
    if oof.duplicated(["anchor_id", "cycle_id"]).any():
        raise AssertionError("duplicate row across OOF folds")
    return oof, pd.concat(selections, ignore_index=True), audits, replay_inputs


def _fit_source_manifest(prepared: Any) -> dict[str, Any]:
    manifest = _prepared_value(prepared, "source_manifest")
    if manifest is None:
        manifest = {
            "prepared_period_audit": _prepared_value(prepared, "audit", {}),
            "canonical_factor_table_sha256": _prepared_value(
                prepared, "factor_table_hash"
            ),
        }
    if is_dataclass(manifest):
        manifest = asdict(manifest)
    if not isinstance(manifest, Mapping):
        raise TypeError("PreparedPeriod.source_manifest must be a mapping")
    payload = dict(manifest)
    expected_phase = {
        "period": "2024",
        "year": 2024,
        "run_rows": 110949,
        "compatible_rows": 759212,
        "positive_rows": 46630,
        "later_period_paths_resolved": False,
        "later_period_rows_read": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    invalid_phase = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected_phase.items()
        if not _matches_exact(payload.get(key), value)
    }
    if invalid_phase:
        raise AssertionError(f"prepared 2024 source phase claim changed: {invalid_phase}")
    prepared_audit = _prepared_value(prepared, "audit", {})
    if not isinstance(prepared_audit, Mapping):
        raise TypeError("PreparedPeriod.audit must be a mapping")
    placeholder_observed = prepared_audit.get("factor_table", {}).get(
        "placeholder_cleanup", {}
    )
    if not isinstance(placeholder_observed, Mapping):
        raise TypeError("2024 placeholder cleanup audit must be a mapping")
    placeholder_expected = load_contract()["feature_construction"][
        "provider_scan_and_factor_table"
    ]["fit_2024_placeholder_audit"]
    placeholder_fields = (
        "discarded_rows",
        "all_four_OHLC_null_rows",
        "partial_null_rows",
        "run_entry_natural_key_intersection",
        "canonical_discarded_key_sha256",
        "per_symbol",
    )
    placeholder_mismatch = {
        key: {
            "expected": placeholder_expected.get(key),
            "actual": placeholder_observed.get(key),
        }
        for key in placeholder_fields
        if placeholder_observed.get(key) != placeholder_expected.get(key)
    }
    if placeholder_mismatch:
        raise AssertionError(
            f"prepared 2024 placeholder cleanup changed: {placeholder_mismatch}"
        )
    payload["placeholder_cleanup"] = dict(placeholder_observed)
    payload.update(
        {
            "contract_sha256": CONTRACT_SHA256,
            "production_source_hashes": _source_hashes(),
            "fit_phase_year": 2024,
        }
    )
    return payload


def run_fit_only(
    root: Path = OUT,
    core_module: ModuleType | None = None,
    evaluation_module: ModuleType | None = None,
) -> dict[str, Any]:
    contract = load_contract()
    _assert_pristine_fit_root(root)
    if core_module is None or evaluation_module is None:
        core_module, evaluation_module = load_production_modules()
    root.mkdir(parents=True, exist_ok=True)
    prepared, cycles, expanded = _prepare_2024(core_module)
    population_validation = _validated_population_result(
        evaluation_module.validate_population(
            _prepared_value(prepared, "anchors"),
            expanded,
            cycles,
            "2024",
            contract,
        ),
        "2024",
    )
    grid_scores, grid_replay_inputs = _grid_scores_2024(
        core_module, prepared, expanded, cycles
    )
    oof, outer_selection, outer_audits, outer_replay_inputs = _outer_oof_2024(
        core_module, prepared, expanded, cycles, grid_scores
    )
    evaluation = _evaluation_payload(
        evaluation_module.evaluate_period(
            predictions=oof,
            period="2024",
            contract=contract,
        )
    )
    fit_source_manifest = _fit_source_manifest(prepared)

    base_artifacts: dict[str, Any] = {
        "lambda_grid_scores_2024.csv": grid_scores,
        "grid_replay_inputs_2024.npz": grid_replay_inputs,
        "outer_lambda_selection_2024.csv": outer_selection,
        "outer_fit_audit_2024.json": outer_audits,
        "outer_replay_inputs_2024.npz": outer_replay_inputs,
        "oof_predictions_2024.parquet": oof,
        "fit_source_manifest_2024.json": fit_source_manifest,
        "population_validation_2024.json": population_validation,
        "fold_schedule.json": {
            "outer": {key: list(value) for key, value in INNER_SCHEDULE.items()},
            "full_selection_months": list(FULL_SELECTION_MONTHS),
            "heads_select_independently": list(HEADS),
            "training_rule": "month strictly before validation month",
        },
        "hyperparameter_grid.json": {
            "lambdas": list(LAMBDA_GRID),
            "tie_tolerance": LAMBDA_TIE_TOLERANCE,
            "tie_break": "largest_lambda",
            "selection_objective": "equal_mean_of_validation_month_log_loss",
        },
        "provisional_decision.json": {
            "development_2024_primary_pass": bool(evaluation["primary_pass"]),
            "label": (
                "factor_conditioned_loop_occurrence_development_candidate"
                if evaluation["primary_pass"]
                else "factor_conditioned_loop_occurrence_rejected_2024_and_do_not_score_later_periods"
            ),
            "later_scoring_authorized": False,
            "prospective_validated": False,
            "movement_quality_grade_changed": False,
        },
    }
    base_artifacts.update(
        evaluation_artifact_names(evaluation["artifacts"], "2024")
    )
    written = write_artifacts(root, base_artifacts)

    full_selected: dict[str, float] | None = None
    if evaluation["primary_pass"]:
        full_selected, full_selection_rows = select_head_lambdas(
            grid_scores, FULL_SELECTION_MONTHS
        )
        full_selection_rows.to_csv(root / "full_lambda_selection_2024.csv", index=False)
        written.append("full_lambda_selection_2024.csv")
        bundle_result = core_module.fit_full_bundle(
            prepared=prepared,
            expanded=expanded,
            cycles=cycles,
            selected_lambdas=full_selected,
        )
        bundle = (
            bundle_result.get("bundle")
            if isinstance(bundle_result, Mapping)
            else getattr(bundle_result, "bundle", bundle_result)
        )
        full_audit = (
            bundle_result.get("audit", {})
            if isinstance(bundle_result, Mapping)
            else getattr(bundle_result, "audit", {})
        )
        replay_inputs = (
            bundle_result.get("replay_inputs")
            if isinstance(bundle_result, Mapping)
            else getattr(bundle_result, "replay_inputs", None)
        )
        if not isinstance(full_audit, Mapping) or not full_audit:
            raise AssertionError("full fit must expose a nonempty audit")
        replay_inputs = _validate_replay_inputs(replay_inputs, "full 2024 fit")
        core_module.save_bundle(root / MODEL_BUNDLE_NAME, bundle)
        write_json(root / "full_fit_audit_2024.json", full_audit)
        written.extend([MODEL_BUNDLE_NAME, "full_fit_audit_2024.json"])
        np.savez_compressed(root / "full_replay_inputs_2024.npz", **replay_inputs)
        written.append("full_replay_inputs_2024.npz")

    manifest = build_artifact_manifest(root, written)
    write_json(root / FIT_MANIFEST_NAME, manifest)
    fit_complete = {
        "status": (
            "fit_frozen_pending_independent_pre_score_audit"
            if evaluation["primary_pass"]
            else "stopped_2024_primary_gates_failed"
        ),
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "production_source_hashes": _source_hashes(),
        "complete_fit_artifact_manifest_sha256": sha256(root / FIT_MANIFEST_NAME),
        "development_2024_primary_pass": bool(evaluation["primary_pass"]),
        "selected_full_lambdas": full_selected,
        "scoring_authorized": False,
        "later_period_paths_resolved": fit_source_manifest[
            "later_period_paths_resolved"
        ],
        "later_period_rows_read": fit_source_manifest["later_period_rows_read"],
        "shadow_tree_read": fit_source_manifest["shadow_tree_read"],
        "shadow_tree_written": fit_source_manifest["shadow_tree_written"],
        "research_only": fit_source_manifest["research_only"],
        "live_ordering_enabled": fit_source_manifest["live_ordering_enabled"],
        "order_placement": fit_source_manifest["order_placement"],
    }
    write_json(root / FIT_COMPLETE_NAME, fit_complete)
    return fit_complete


def _later_prepared(
    core_module: ModuleType,
    period: str,
    contract: Mapping[str, Any],
    paths: Mapping[str, Any],
    guard_token: Any,
) -> Any:
    year = int(period)
    run_key = f"runs_{period}"
    run_spec = contract["frozen_sources"][run_key]
    run_path = paths[f"runs_{period}"]
    provider_root = paths[f"provider_root_{period}"]
    return core_module.prepare_period_anchors(
        run_path=Path(run_path),
        provider_root=Path(provider_root),
        year=year,
        period=period,
        expected_rows=int(run_spec["rows"]),
        expected_stocks=int(run_spec["stocks"]),
        expected_run_sha256=str(run_spec["sha256"]),
        expected_compatible_rows=int(
            contract["population_and_target"]["compatible_anchor_cycle_rows_expected"][period]
        ),
        expected_positives=int(
            contract["population_and_target"]["positive_rows_expected"][period]
        ),
        expected_provider_hashes=core_module.provider_hashes_for_scoring_year(
            year, guard_token
        ),
        authorization=guard_token,
    )


def _validated_later_source_manifest(prepared: Any, period: str) -> dict[str, Any]:
    source_manifest = _prepared_value(prepared, "source_manifest", {})
    if is_dataclass(source_manifest):
        source_manifest = asdict(source_manifest)
    if not isinstance(source_manifest, Mapping):
        raise TypeError(f"{period} source manifest must be a mapping")
    payload = dict(source_manifest)
    expected = {
        "period": period,
        "year": int(period),
        "later_period_paths_resolved": True,
        "later_period_rows_read": True,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if not _matches_exact(payload.get(key), value)
    }
    if mismatch:
        raise AssertionError(f"invalid {period} source phase or safety claim: {mismatch}")
    prepared_audit = _prepared_value(prepared, "audit", {})
    if not isinstance(prepared_audit, Mapping):
        raise TypeError(f"{period} prepared-period audit must be a mapping")
    cleanup = prepared_audit.get("factor_table", {}).get("placeholder_cleanup", {})
    if not isinstance(cleanup, Mapping):
        raise TypeError(f"{period} placeholder-cleanup audit must be a mapping")
    discarded = cleanup.get("discarded_rows")
    all_null = cleanup.get("all_four_OHLC_null_rows")
    partial = cleanup.get("partial_null_rows")
    intersection = cleanup.get("run_entry_natural_key_intersection")
    key_hash = cleanup.get("canonical_discarded_key_sha256")
    per_symbol = cleanup.get("per_symbol")
    if (
        type(discarded) is not int
        or discarded < 0
        or all_null != discarded
        or partial != 0
        or intersection != 0
        or not isinstance(key_hash, str)
        or len(key_hash) != 64
        or not isinstance(per_symbol, Mapping)
    ):
        raise AssertionError(f"invalid {period} placeholder-cleanup audit")
    payload["prepared_period_audit"] = dict(prepared_audit)
    return payload


def _validated_pinned_replay(predictions: pd.DataFrame, period: str) -> dict[str, Any]:
    replay = predictions.attrs.get("pinned_path_replay")
    if not isinstance(replay, Mapping):
        raise AssertionError(f"{period} predictions lack pinned path replay evidence")
    expected = {
        "period": period,
        "rows": len(predictions),
        "tolerance": 1e-12,
        "pinned_probabilities_substituted_before_candidate_scoring": True,
    }
    mismatch = {
        key: {"expected": value, "actual": replay.get(key)}
        for key, value in expected.items()
        if not _matches_exact(replay.get(key), value)
    }
    if mismatch:
        raise AssertionError(f"invalid {period} pinned path replay claim: {mismatch}")
    for key in (
        "qhistory_max_abs_error",
        "qold_limited_path_max_abs_error",
    ):
        error = replay.get(key)
        if not isinstance(error, (int, float, np.number)) or not math.isfinite(
            float(error)
        ) or float(error) > 1e-12:
            raise AssertionError(f"{period} pinned path replay failed: {key}={error}")
    return dict(replay)


def run_score_only(
    root: Path = OUT,
    core_module: ModuleType | None = None,
    evaluation_module: ModuleType | None = None,
) -> dict[str, Any]:
    # These checks intentionally precede module import and every later path lookup.
    fit, audit, authorization, fit_manifest = validate_pre_score_authorization(root)
    assert_pristine_scoring_namespace(root, fit_manifest)
    contract = load_contract()
    if core_module is None or evaluation_module is None:
        core_module, evaluation_module = load_production_modules()
    guard_token = core_module.authorize_later_phase(
        authorization=authorization,
        fit_complete=fit,
        fit_manifest=fit_manifest,
    )
    later_paths = core_module.phase_source_paths("score", guard_token=guard_token)
    cycles = _require_frame(core_module.load_cycles(), "cycles")
    bundle = core_module.load_bundle(root / MODEL_BUNDLE_NAME)
    period_results: dict[str, Any] = {}
    period_evaluations: dict[str, dict[str, Any]] = {}
    later_source_manifests: dict[str, Any] = {}
    scoring_artifacts: list[str] = []
    all_passed = True
    for period in ("2025", "2023"):
        prepared = _later_prepared(core_module, period, contract, later_paths, guard_token)
        later_source_manifests[period] = _validated_later_source_manifest(
            prepared, period
        )
        anchors = _require_frame(_prepared_value(prepared, "anchors"), f"{period} anchors")
        expanded = _prepared_value(prepared, "expanded")
        if expanded is None:
            expanded = core_module.expand_compatible_labels(
                anchors,
                cycles,
                expected_rows=contract["population_and_target"]["compatible_anchor_cycle_rows_expected"][period],
                expected_positives=contract["population_and_target"]["positive_rows_expected"][period],
            )
        expanded = _require_frame(expanded, f"{period} expanded labels")
        population_validation = _validated_population_result(
            evaluation_module.validate_population(
                anchors,
                expanded,
                cycles,
                period,
                contract,
            ),
            period,
        )
        later_source_manifests[period]["population_validation"] = population_validation
        predictions = core_module.predict_bundle(
            bundle=bundle,
            anchors=prepared,
            expanded=expanded,
            cycles=cycles,
            period=period,
            phase_guard=guard_token,
        )
        predictions = _require_frame(predictions, f"{period} predictions")
        _assert_exact_prediction_population(
            predictions, expanded, f"{period} scored panel"
        )
        pinned_replay = _validated_pinned_replay(predictions, period)
        pinned_name = f"pinned_path_replay_{period}.json"
        write_json(root / pinned_name, pinned_replay)
        scoring_artifacts.append(pinned_name)
        result = _evaluation_payload(
            evaluation_module.evaluate_period(
                predictions=predictions,
                period=period,
                contract=contract,
            )
        )
        name = f"predictions_{period}.parquet"
        predictions.to_parquet(root / name, index=False)
        scoring_artifacts.append(name)
        renamed = evaluation_artifact_names(result["artifacts"], period)
        scoring_artifacts.extend(write_artifacts(root, renamed))

        if period == "2025":
            irregular = evaluation_module.evaluate_irregular_deletion(
                predictions,
                original_evaluation=result,
                contract=contract,
            )
            irregular_pass = next(
                (
                    bool(irregular[key])
                    for key in ("pass", "secondary_pass", "gate_pass")
                    if key in irregular
                ),
                None,
            )
            if irregular_pass is None:
                raise AssertionError("irregular-deletion result lacks a pass flag")
            result["irregular_deletion"] = irregular
            result["primary_pass"] = bool(result["primary_pass"] and irregular_pass)
            write_json(root / "irregular_cohort_2025.json", irregular)
            scoring_artifacts.append("irregular_cohort_2025.json")

        period_evaluations[period] = result
        period_results[period] = {
            "primary_pass": bool(result["primary_pass"]),
            "summary": result.get("summary", {}),
        }
        all_passed = all_passed and bool(result["primary_pass"])

    transfer = dict(
        evaluation_module.derive_decision(
            {"primary_pass": bool(fit["development_2024_primary_pass"])},
            period_evaluations.get("2025"),
            period_evaluations.get("2023"),
        )
    )
    transfer.update({
        "periods": period_results,
        "all_later_primary_pass": all_passed,
        "later_periods_promoted_candidate": False,
        "prospective_validated": False,
        "movement_quality_grade_changed": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    })
    write_json(root / "transfer_decision.json", transfer)
    scoring_artifacts.append("transfer_decision.json")
    scoring_manifest = build_artifact_manifest(
        root, scoring_artifacts, later_period_paths_resolved=True
    )
    scoring_manifest["period_source_manifests"] = later_source_manifests
    write_json(root / "scoring_source_manifest.json", scoring_manifest)
    scoring_complete = {
        "status": "scoring_complete_pending_independent_post_score_audit",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "fit_complete_sha256": sha256(root / FIT_COMPLETE_NAME),
        "pre_score_audit_sha256": sha256(root / AUDIT_RESULT_NAME),
        "pre_score_authorization_sha256": sha256(root / AUTHORIZATION_NAME),
        "scoring_artifact_manifest_sha256": sha256(root / "scoring_source_manifest.json"),
        "transfer": transfer,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(root / "scoring_complete.json", scoring_complete)
    return scoring_complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--self-test-only", action="store_true")
    phases.add_argument("--validate-only", action="store_true")
    phases.add_argument("--fit-only", action="store_true")
    phases.add_argument("--score-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.self_test_only:
        result = run_self_test_only()
    elif args.validate_only:
        result = run_validate_only()
    elif args.fit_only:
        result = run_fit_only()
    elif args.score_only:
        result = run_score_only()
    else:  # pragma: no cover - argparse's required group makes this unreachable.
        raise AssertionError("unreachable phase")
    print(json.dumps(safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
