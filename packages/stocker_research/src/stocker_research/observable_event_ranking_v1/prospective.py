"""Hash-bound append-only prospective prediction and settlement primitives."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stocker_research.observable_event_ranking_v1.contract import (
    PRIMARY_FEATURES,
    REQUIRED_SAFETY_FLAGS,
    canonical_json_bytes,
)


class PredictionLedgerError(RuntimeError):
    """Invalid or non-append-only prospective prediction."""


class SettlementLedgerError(RuntimeError):
    """Invalid, early, or duplicate prospective settlement."""


_SAFE_RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_BUNDLE_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PREDICTION_REQUIRED = (
    "prediction_id",
    "event_id",
    "slate_id",
    "decision_timestamp",
    "prediction_timestamp",
    "planned_entry_reference_time",
    "planned_exit_reference_time",
    "outcome_available_at",
    "symbol",
    "source_provider",
    "source_dataset_id",
    "source_hash",
    "score",
    "frozen_baseline_score",
    "bundle_hash",
    "safety",
)
_SETTLEMENT_REQUIRED = ("prediction_id", "future_return_60m", "settlement_status")


@dataclass(frozen=True)
class ValidatedPrediction:
    """Validated prospective identity and causal timing."""

    prediction_id: str
    prediction_timestamp: pd.Timestamp
    outcome_available_at: pd.Timestamp
    bundle_hash: str


def _validate_prediction(
    prediction: dict[str, Any], *, expected_bundle_hash: str | None = None
) -> ValidatedPrediction:
    missing = sorted(set(_PREDICTION_REQUIRED).difference(prediction))
    if missing:
        raise PredictionLedgerError(f"prediction missing required fields: {missing}")
    for field in (
        "prediction_id",
        "event_id",
        "slate_id",
        "symbol",
        "source_provider",
        "source_dataset_id",
        "source_hash",
    ):
        if not isinstance(prediction[field], str) or not prediction[field]:
            raise PredictionLedgerError(f"prediction {field} must be a non-empty string")
    prediction_id = str(prediction["prediction_id"])
    try:
        _record_path(Path("."), prediction_id)
    except ValueError as exc:
        raise PredictionLedgerError(str(exc)) from exc
    decision_at = pd.Timestamp(prediction["decision_timestamp"])
    predicted_at = pd.Timestamp(prediction["prediction_timestamp"])
    entry_at = pd.Timestamp(prediction["planned_entry_reference_time"])
    exit_at = pd.Timestamp(prediction["planned_exit_reference_time"])
    available_at = pd.Timestamp(prediction["outcome_available_at"])
    times = (decision_at, predicted_at, entry_at, exit_at, available_at)
    if any(timestamp.tzinfo is None for timestamp in times):
        raise PredictionLedgerError("prediction timestamps must be timezone-aware")
    if not decision_at <= predicted_at < entry_at < exit_at <= available_at:
        raise PredictionLedgerError(
            "prediction timing must satisfy decision <= prediction < entry < exit <= availability"
        )
    for field in ("score", "frozen_baseline_score"):
        value = prediction[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PredictionLedgerError(f"prediction {field} must be numeric")
        if not math.isfinite(float(value)):
            raise PredictionLedgerError(f"prediction {field} must be finite")
    bundle_hash = str(prediction["bundle_hash"])
    if _BUNDLE_HASH.fullmatch(bundle_hash) is None:
        raise PredictionLedgerError("prediction bundle_hash must be lowercase SHA-256")
    if expected_bundle_hash is not None and bundle_hash != expected_bundle_hash:
        raise PredictionLedgerError("prediction does not match the frozen prospective bundle")
    if prediction["safety"] != REQUIRED_SAFETY_FLAGS:
        raise PredictionLedgerError("prediction safety flags do not match the frozen contract")
    return ValidatedPrediction(
        prediction_id=prediction_id,
        prediction_timestamp=predicted_at,
        outcome_available_at=available_at,
        bundle_hash=bundle_hash,
    )


def _record_path(root: Path, record_id: str) -> Path:
    if _SAFE_RECORD_ID.fullmatch(record_id) is None:
        raise ValueError("record id must contain only bounded ASCII letters, digits, _ or -")
    return root / f"{record_id}.json"


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def prospective_ledger_schemas() -> dict[str, dict[str, Any]]:
    """Return frozen prediction and separate settlement schemas."""

    return {
        "prediction": {
            "schema_version": "observable_event_ranking_v1_prediction",
            "append_only": True,
            "outcome_fields_permitted_at_prediction": False,
            "required_fields": [
                "prediction_id",
                "event_id",
                "slate_id",
                "decision_timestamp",
                "prediction_timestamp",
                "planned_entry_reference_time",
                "planned_exit_reference_time",
                "outcome_available_at",
                "symbol",
                "source_provider",
                "source_dataset_id",
                "source_hash",
                "score",
                "frozen_baseline_score",
                "bundle_hash",
                "safety",
            ],
        },
        "settlement": {
            "schema_version": "observable_event_ranking_v1_settlement",
            "append_only": True,
            "separate_from_prediction": True,
            "refuse_before_outcome_available": True,
            "required_fields": [
                "prediction_id",
                "settlement_time",
                "outcome_available_at",
                "future_return_60m",
                "settlement_status",
            ],
        },
    }


def score_frozen_prediction(
    prediction: dict[str, Any],
    *,
    model_parameters: dict[str, Any],
    baseline_parameters: dict[str, Any],
    bundle_hash: str,
) -> dict[str, Any]:
    """Apply only the serialized M1 coefficients and frozen selected baseline."""

    provided_bundle = prediction.get("bundle_hash")
    if provided_bundle is not None and provided_bundle != bundle_hash:
        raise PredictionLedgerError("input prediction names a different prospective bundle")
    feature_values = prediction.get("features")
    if not isinstance(feature_values, dict):
        raise PredictionLedgerError("prospective scoring requires a features object")
    if set(feature_values) != set(PRIMARY_FEATURES):
        missing = sorted(set(PRIMARY_FEATURES).difference(feature_values))
        unexpected = sorted(set(feature_values).difference(PRIMARY_FEATURES))
        raise PredictionLedgerError(
            f"prospective features differ from frozen surface; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if tuple(model_parameters.get("feature_names", ())) != PRIMARY_FEATURES:
        raise PredictionLedgerError("serialized model feature order differs from frozen contract")
    raw: list[float] = []
    for feature in PRIMARY_FEATURES:
        value = feature_values[feature]
        if value is None:
            raw.append(np.nan)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PredictionLedgerError(f"feature {feature} must be numeric or null")
        elif not math.isfinite(float(value)):
            raise PredictionLedgerError(f"feature {feature} must be finite or null")
        else:
            raw.append(float(value))
    preprocessor = model_parameters.get("preprocessor", {})
    try:
        medians = np.asarray(preprocessor["medians"], dtype="float64")
        lower = np.asarray(preprocessor["lower_clip"], dtype="float64")
        upper = np.asarray(preprocessor["upper_clip"], dtype="float64")
        means = np.asarray(preprocessor["means"], dtype="float64")
        scales = np.asarray(preprocessor["scales"], dtype="float64")
        coefficients = np.asarray(model_parameters["coefficients"], dtype="float64")
        intercept = float(model_parameters["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionLedgerError("serialized model parameters are incomplete") from exc
    arrays = (medians, lower, upper, means, scales, coefficients)
    if any(values.shape != (len(PRIMARY_FEATURES),) for values in arrays):
        raise PredictionLedgerError("serialized model parameter dimensions are invalid")
    values = np.asarray(raw, dtype="float64")
    values = np.where(np.isfinite(values), values, medians)
    standardized = (np.clip(values, lower, upper) - means) / scales
    score = intercept + float(standardized @ coefficients)

    baseline_id = str(baseline_parameters.get("baseline_id", ""))
    if baseline_parameters.get("kind") == "direct_observable_feature":
        source_feature = str(baseline_parameters.get("source_feature", ""))
        baseline_value = feature_values.get(source_feature)
        if (
            baseline_value is None
            or isinstance(baseline_value, bool)
            or not isinstance(baseline_value, (int, float))
            or not math.isfinite(float(baseline_value))
        ):
            raise PredictionLedgerError("frozen direct baseline feature is unavailable")
        baseline_score = float(baseline_value)
    elif baseline_id in {
        "B8_TRAINING_ONLY_STOCK_CLOCK_EVENT_FREQUENCY",
        "B9_TRAINING_ONLY_STOCK_CLOCK_MEAN_TARGET_PRIOR",
    }:
        decision_value = prediction.get("decision_timestamp")
        if not isinstance(decision_value, str):
            raise PredictionLedgerError("baseline scoring requires a decision timestamp")
        decision_at = pd.Timestamp(decision_value)
        if decision_at.tzinfo is None:
            raise PredictionLedgerError("baseline scoring decision timestamp must be aware")
        clock = decision_at.tz_convert(ZoneInfo("America/New_York")).strftime("%H:%M")
        symbol = str(prediction.get("symbol", ""))
        cells = {
            (str(row["symbol"]), str(row["decision_clock"])): float(row["score"])
            for row in baseline_parameters.get("cell_scores", [])
        }
        baseline_score = cells.get(
            (symbol, clock),
            float(baseline_parameters["global_prior"]),
        )
    else:
        raise PredictionLedgerError("frozen baseline parameters are unsupported")
    if not math.isfinite(score) or not math.isfinite(baseline_score):
        raise PredictionLedgerError("frozen scoring produced a non-finite value")
    return {
        **prediction,
        "score": score,
        "frozen_baseline_score": baseline_score,
        "bundle_hash": bundle_hash,
        "safety": REQUIRED_SAFETY_FLAGS,
    }


def _outcome_key(key: str) -> bool:
    normalized = key.lower()
    if normalized == "outcome_available_at":
        return False
    return any(
        token in normalized
        for token in ("future_return", "target", "outcome", "pnl", "cost", "settlement")
    )


def append_prediction(
    root: Path,
    prediction: dict[str, Any],
    *,
    expected_bundle_hash: str | None = None,
) -> Path:
    """Append one outcome-free prediction as an immutable identity-named record."""

    validated = _validate_prediction(
        prediction,
        expected_bundle_hash=expected_bundle_hash,
    )
    prediction_id = validated.prediction_id
    try:
        path = _record_path(root, prediction_id)
    except ValueError as exc:
        raise PredictionLedgerError(str(exc)) from exc
    forbidden = sorted(key for key in prediction if _outcome_key(key))
    if forbidden:
        raise PredictionLedgerError(f"prediction contains outcome fields: {forbidden}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        _exclusive_write(path, canonical_json_bytes(prediction) + b"\n")
    except FileExistsError as exc:
        raise PredictionLedgerError("prediction record already exists") from exc
    except OSError as exc:
        raise PredictionLedgerError(f"prediction append failed: {type(exc).__name__}") from exc
    return path


def append_settlement(
    root: Path,
    *,
    prediction: dict[str, Any],
    settlement: dict[str, Any],
    settlement_time: str,
) -> Path:
    """Append settlement only after the frozen outcome availability timestamp."""

    try:
        validated = _validate_prediction(prediction)
    except PredictionLedgerError as exc:
        raise SettlementLedgerError(f"invalid source prediction: {exc}") from exc
    prediction_id = validated.prediction_id
    try:
        path = _record_path(root, prediction_id)
    except ValueError as exc:
        raise SettlementLedgerError(str(exc)) from exc
    available_at = validated.outcome_available_at
    settled_at = pd.Timestamp(settlement_time)
    if available_at.tzinfo is None or settled_at.tzinfo is None:
        raise SettlementLedgerError("availability and settlement timestamps must be timezone-aware")
    if settled_at < available_at:
        raise SettlementLedgerError("outcome is not yet causally available")
    missing_settlement = sorted(set(_SETTLEMENT_REQUIRED).difference(settlement))
    if missing_settlement:
        raise SettlementLedgerError(f"settlement missing required fields: {missing_settlement}")
    if settlement["prediction_id"] != prediction_id:
        raise SettlementLedgerError("settlement prediction_id does not match prediction")
    if not isinstance(settlement["settlement_status"], str) or not settlement["settlement_status"]:
        raise SettlementLedgerError("settlement_status must be a non-empty string")
    future_return = settlement["future_return_60m"]
    if isinstance(future_return, bool) or not isinstance(future_return, (int, float)):
        raise SettlementLedgerError("future_return_60m must be numeric")
    if not math.isfinite(float(future_return)):
        raise SettlementLedgerError("future_return_60m must be finite")
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "prediction_id": prediction_id,
        "settlement_time": settled_at.isoformat(),
        "outcome_available_at": available_at.isoformat(),
        "settlement": settlement,
    }
    try:
        _exclusive_write(path, canonical_json_bytes(payload) + b"\n")
    except FileExistsError as exc:
        raise SettlementLedgerError("settlement record already exists") from exc
    except OSError as exc:
        raise SettlementLedgerError(f"settlement append failed: {type(exc).__name__}") from exc
    return path
