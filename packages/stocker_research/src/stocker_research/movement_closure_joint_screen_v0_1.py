"""Primitives for Movement x Closure-History Joint Increment V0.1.

The module is deliberately limited to causal joining, probability transforms,
fixed logistic stackers, paired uncertainty, support gates, and the preregistered
decision. It has no price-direction, payoff, broker, order, or runtime surface.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score

from stocker_research.movement_regime_path_screen_v0 import (
    FrozenLinearModel,
    circular_shift_session_blocks,
)
from stocker_research.movement_regime_path_screen_v0 import (
    fit_fixed_logistic as _fit_fixed_logistic,
)

EPSILON = 1e-6
MAX_JOINED_ROWS = 10_000

SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "feasibility_screen": True,
    "representation_specific": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}

DECISIONS = (
    "mutually_informative_movement_closure_process",
    "movement_adds_to_closure_only",
    "closure_history_adds_to_movement_only",
    "joint_interaction_only",
    "separate_predictable_processes_no_increment",
    "blocked_missing_frozen_joint_inputs",
    "blocked_join_semantics_failure",
    "blocked_chronology_or_leakage_failure",
    "blocked_insufficient_joint_increment_support",
    "blocked_model_convergence_failure",
    "blocked_quick_screen_resource_limit",
    "blocked_reproducibility_or_audit_failure",
)

FORBIDDEN_FIELD_FRAGMENTS = (
    "future_signed",
    "signed_return",
    "long_probability",
    "short_probability",
    "strategy_return",
    "pnl",
    "profit",
    "payoff",
    "mfe",
    "mae",
    "entry_price",
    "stop_price",
    "target_price",
    "broker",
    "account",
    "position",
    "order_id",
    "exact_five_state",
    "exact_loop",
)

FloatArray = npt.NDArray[np.float64]

_MOVEMENT_REQUIRED = {
    "movement_row_id",
    "representation_id",
    "source_lineage_id",
    "stock",
    "session",
    "decision_ordinal",
    "fixed_clock_timestamp",
    "movement_horizon_terminal_timestamp",
    "origin_segment_id",
    "current_state_b",
    "scheduled_bars_remaining",
    "p_move",
    "predicted_absolute_movement_bps",
    "large_move",
    "movement_available",
    "source_gap",
}

_CLOSURE_REQUIRED = {
    "pair_forecast_id",
    "representation_id",
    "source_lineage_id",
    "stock",
    "session",
    "pair_forecast_timestamp",
    "closure_resolution_timestamp",
    "segment_id",
    "current_state_b",
    "pair_orientation",
    "p_close_m2",
    "p_close_m5",
    "immediate_pair_closure",
    "closure_available",
    "source_gap",
}


@dataclass(frozen=True, slots=True)
class JoinResult:
    """Exact joined rows and mutually exclusive movement-row accounting."""

    frame: pd.DataFrame
    accounting: dict[str, int]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _joined_row_id(representation: str, stock: str, session: str, pair_id: str) -> str:
    payload = "|".join((representation, stock, session, pair_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _timestamp(value: object) -> pd.Timestamp:
    result = pd.Timestamp(cast(Any, value))
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def exact_active_pair_join(
    movement: pd.DataFrame,
    closure: pd.DataFrame,
    *,
    max_rows: int = MAX_JOINED_ROWS,
) -> JoinResult:
    """Join each fixed clock to the one same-lineage unresolved active pair.

    Every movement row receives exactly one terminal classification. A pair may
    survive two clocks, but only its earliest eligible clock is retained.
    """

    _require_columns(movement, _MOVEMENT_REQUIRED, "movement surface")
    _require_columns(closure, _CLOSURE_REQUIRED, "closure surface")
    if max_rows > MAX_JOINED_ROWS or max_rows <= 0:
        raise ValueError("quick-screen row cap must be in [1, 10000]")

    movement_rows = movement.copy()
    closure_rows = closure.copy()
    for column in ("fixed_clock_timestamp", "movement_horizon_terminal_timestamp"):
        movement_rows[column] = pd.to_datetime(movement_rows[column], utc=True, errors="raise")
    for column in ("pair_forecast_timestamp", "closure_resolution_timestamp"):
        closure_rows[column] = pd.to_datetime(closure_rows[column], utc=True, errors="coerce")
    movement_rows = movement_rows.sort_values(
        ["session", "decision_ordinal", "stock", "movement_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    closure_rows = closure_rows.sort_values(
        ["session", "stock", "pair_forecast_timestamp", "pair_forecast_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    accounting = {
        "movement_rows_inspected": int(len(movement_rows)),
        "closure_forecasts_inspected": int(len(closure_rows)),
        "exact_joined_rows": 0,
        "excluded_resolved_before_clock": 0,
        "excluded_no_active_pair": 0,
        "excluded_representation_mismatch": 0,
        "excluded_source_gap": 0,
        "excluded_duplicate_later_clock": 0,
        "excluded_closure_unavailable": 0,
        "excluded_movement_target_unavailable": 0,
    }
    closure_groups = {
        (str(stock), str(session)): group.copy()
        for (stock, session), group in closure_rows.groupby(
            ["stock", "session"], sort=True, observed=True
        )
    }
    joined: list[dict[str, Any]] = []
    movement_records = cast(list[dict[str, Any]], movement_rows.to_dict(orient="records"))
    for movement_dict in movement_records:
        if (
            not bool(movement_dict["movement_available"])
            or pd.isna(movement_dict["p_move"])
            or pd.isna(movement_dict["predicted_absolute_movement_bps"])
            or pd.isna(movement_dict["large_move"])
        ):
            accounting["excluded_movement_target_unavailable"] += 1
            continue
        key = (str(movement_dict["stock"]), str(movement_dict["session"]))
        session_pairs = closure_groups.get(key)
        if session_pairs is None or session_pairs.empty:
            accounting["excluded_no_active_pair"] += 1
            continue
        clock = _timestamp(movement_dict["fixed_clock_timestamp"])
        created = session_pairs.loc[session_pairs["pair_forecast_timestamp"].le(clock)]
        if created.empty:
            accounting["excluded_no_active_pair"] += 1
            continue
        active_mask = created["closure_resolution_timestamp"].isna() | created[
            "closure_resolution_timestamp"
        ].gt(clock)
        active = created.loc[active_mask]
        if active.empty:
            accounting["excluded_resolved_before_clock"] += 1
            continue
        same_identity = active.loc[
            active["representation_id"].astype(str).eq(str(movement_dict["representation_id"]))
            & active["source_lineage_id"].astype(str).eq(str(movement_dict["source_lineage_id"]))
        ]
        if same_identity.empty:
            accounting["excluded_representation_mismatch"] += 1
            continue
        if bool(movement_dict["source_gap"]):
            accounting["excluded_source_gap"] += 1
            continue
        gap_free = same_identity.loc[~same_identity["source_gap"].astype(bool)]
        if gap_free.empty:
            accounting["excluded_source_gap"] += 1
            continue
        exact = gap_free.loc[
            gap_free["segment_id"].astype(str).eq(str(movement_dict["origin_segment_id"]))
            & pd.to_numeric(gap_free["current_state_b"], errors="raise").eq(
                int(movement_dict["current_state_b"])
            )
        ]
        if exact.empty:
            accounting["excluded_no_active_pair"] += 1
            continue
        available = exact.loc[
            exact["closure_available"].astype(bool)
            & exact["closure_resolution_timestamp"].notna()
            & exact["immediate_pair_closure"].notna()
        ]
        if available.empty:
            accounting["excluded_closure_unavailable"] += 1
            continue
        if len(available) != 1:
            raise ValueError("blocked_join_semantics_failure: overlapping active pairs")
        pair = cast(dict[str, Any], available.iloc[0].to_dict())
        resolution = _timestamp(pair["closure_resolution_timestamp"])
        if resolution <= clock:
            raise ValueError("blocked_join_semantics_failure: closure resolved before clock")
        forecast = _timestamp(pair["pair_forecast_timestamp"])
        age = (clock - forecast) / pd.Timedelta(minutes=5)
        rounded_age = int(round(float(age)))
        if age < 0 or not math.isclose(float(age), rounded_age, abs_tol=1e-9):
            raise ValueError("blocked_join_semantics_failure: non-integral pair age")
        joined.append(
            {
                **movement_dict,
                "joined_row_id": _joined_row_id(
                    str(pair["representation_id"]),
                    str(pair["stock"]),
                    str(pair["session"]),
                    str(pair["pair_forecast_id"]),
                ),
                "pair_forecast_id": str(pair["pair_forecast_id"]),
                "pair_forecast_timestamp": forecast,
                "closure_resolution_timestamp": resolution,
                "pair_age_bars": rounded_age,
                "pair_orientation": str(pair["pair_orientation"]),
                "p_close_m2": float(pair["p_close_m2"]),
                "p_close_m5": float(pair["p_close_m5"]),
                "immediate_pair_closure": int(pair["immediate_pair_closure"]),
                "closure_available": True,
                "closure_m2_oof": bool(pair.get("closure_m2_oof", False)),
                "closure_m5_oof": bool(pair.get("closure_m5_oof", False)),
                "closure_trained_through": pair.get("closure_trained_through", pd.NaT),
                "closure_frozen_before_outcome": bool(
                    pair.get("closure_frozen_before_outcome", False)
                ),
                "closure_training_rows": pair.get("closure_training_rows", np.nan),
                "closure_chronology_evidence_id": pair.get("closure_chronology_evidence_id", ""),
            }
        )

    joined_frame = pd.DataFrame(joined)
    if not joined_frame.empty:
        joined_frame = joined_frame.sort_values(
            [
                "representation_id",
                "stock",
                "session",
                "pair_forecast_id",
                "fixed_clock_timestamp",
                "decision_ordinal",
            ],
            kind="mergesort",
        )
        duplicate = joined_frame.duplicated(
            ["representation_id", "stock", "session", "pair_forecast_id"], keep="first"
        )
        accounting["excluded_duplicate_later_clock"] = int(duplicate.sum())
        joined_frame = joined_frame.loc[~duplicate].copy()
        joined_frame = joined_frame.sort_values(
            ["session", "decision_ordinal", "stock", "pair_forecast_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    if len(joined_frame) > max_rows:
        raise RuntimeError("blocked_quick_screen_resource_limit")
    accounting["exact_joined_rows"] = int(len(joined_frame))
    excluded = sum(value for key, value in accounting.items() if key.startswith("excluded_"))
    if accounting["exact_joined_rows"] + excluded != accounting["movement_rows_inspected"]:
        raise AssertionError("join accounting does not reconcile")
    return JoinResult(joined_frame, accounting)


def logit_probability(values: Sequence[float] | npt.NDArray[np.float64]) -> FloatArray:
    """Return epsilon-clipped logits without mutating the probability surface."""

    probability = np.asarray(values, dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("logit input must be finite probabilities")
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def add_joint_probability_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the three logits, history increment, and Boolean joint target."""

    required = {
        "p_move",
        "p_close_m2",
        "p_close_m5",
        "predicted_absolute_movement_bps",
        "large_move",
        "immediate_pair_closure",
    }
    _require_columns(frame, required, "joined probability panel")
    output = frame.copy()
    output["logit_p_move"] = logit_probability(output["p_move"].to_numpy(float))
    output["logit_p_close_m2"] = logit_probability(output["p_close_m2"].to_numpy(float))
    output["logit_p_close_m5"] = logit_probability(output["p_close_m5"].to_numpy(float))
    output["closure_history_increment"] = output["logit_p_close_m5"] - output["logit_p_close_m2"]
    output["joint_large_move_and_closure"] = (
        output["large_move"].astype(bool) & output["immediate_pair_closure"].astype(bool)
    ).astype(np.int8)
    return output


def with_equal_slate_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Give each retained stock/session/clock slate total weight one."""

    _require_columns(frame, {"session", "decision_ordinal"}, "slate panel")
    output = frame.copy()
    output["slate_id"] = (
        output["session"].astype(str)
        + "|"
        + pd.to_numeric(output["decision_ordinal"], errors="raise").astype(int).astype(str)
    )
    sizes = output.groupby("slate_id", sort=True)["slate_id"].transform("size")
    output["joined_slate_size"] = sizes.astype(int)
    output["row_weight"] = 1.0 / sizes.to_numpy(dtype=float)
    totals = output.groupby("slate_id", sort=True)["row_weight"].sum().to_numpy(float)
    if not np.allclose(totals, 1.0, atol=1e-12):
        raise AssertionError("slate weights do not total one")
    return output


def assert_protected_date_boundary(
    timestamps: Sequence[object] | pd.Series,
    *,
    cutoff: str = "2025-08-23T00:00:00Z",
) -> None:
    """Reject any materialised timestamp at or beyond the joint boundary."""

    values = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    if values.ge(pd.Timestamp(cutoff)).any():
        raise ValueError("protected timestamp materialised")


def assert_compact_panel_has_no_forbidden_fields(frame: pd.DataFrame) -> None:
    """Reject direction, payoff, execution, and exact-sequence fields."""

    invalid = sorted(
        column
        for column in frame.columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_FIELD_FRAGMENTS)
    )
    if invalid:
        raise ValueError(f"forbidden joint-screen field(s): {invalid}")


def assert_upstream_chronology(frame: pd.DataFrame) -> None:
    """Verify 2024 OOF and 2025 pre-outcome frozen provenance."""

    required = {
        "year",
        "session",
        "movement_oof",
        "closure_m2_oof",
        "closure_m5_oof",
        "movement_trained_through",
        "movement_size_trained_through",
        "closure_trained_through",
        "movement_frozen_before_outcome",
        "closure_frozen_before_outcome",
        "movement_chronology_evidence_id",
        "closure_chronology_evidence_id",
    }
    _require_columns(frame, required, "chronology panel")
    years = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if not set(years.unique()).issubset({2024, 2025}):
        raise AssertionError("joint stacker chronology permits only 2024 and 2025")
    sessions = pd.to_datetime(frame["session"], utc=True, errors="raise")
    development = years.eq(2024)
    if development.any():
        for column in ("movement_oof", "closure_m2_oof", "closure_m5_oof"):
            if not frame.loc[development, column].astype(bool).all():
                raise AssertionError(f"in-sample upstream probability detected: {column}")
        for column in (
            "movement_trained_through",
            "movement_size_trained_through",
            "closure_trained_through",
        ):
            trained = pd.to_datetime(frame.loc[development, column], utc=True, errors="raise")
            if trained.ge(sessions.loc[development]).any():
                raise AssertionError(f"in-sample upstream probability detected: {column}")
    assessment = years.eq(2025)
    if assessment.any():
        for column in ("movement_frozen_before_outcome", "closure_frozen_before_outcome"):
            if not frame.loc[assessment, column].astype(bool).all():
                raise AssertionError(f"2025 probability was not frozen: {column}")
        for column in (
            "movement_trained_through",
            "movement_size_trained_through",
            "closure_trained_through",
        ):
            trained = pd.to_datetime(frame.loc[assessment, column], utc=True, errors="raise")
            if trained.ge(sessions.loc[assessment]).any():
                raise AssertionError(f"2025 upstream fit crossed assessment: {column}")
    for column in (
        "movement_chronology_evidence_id",
        "closure_chronology_evidence_id",
    ):
        if frame[column].isna().any() or not frame[column].astype(str).str.len().gt(0).all():
            raise AssertionError(f"missing hash-bound chronology evidence: {column}")


def split_development_assessment(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 2024 fitting rows and untouched 2025 assessment rows."""

    years = pd.to_numeric(frame["year"], errors="raise").astype(int)
    development = frame.loc[years.eq(2024)].copy().reset_index(drop=True)
    assessment = frame.loc[years.eq(2025)].copy().reset_index(drop=True)
    if development.empty or assessment.empty:
        raise ValueError("joint stackers require both 2024 and 2025 rows")
    return development, assessment


def fit_fixed_logistic(
    frame: pd.DataFrame,
    target: Sequence[int] | pd.Series,
    *,
    features: Sequence[str],
    slate_column: str,
    model_id: str,
    random_state: int = 20260720,
) -> FrozenLinearModel:
    """Fit the frozen C=1 L2 liblinear model through the shared primitive."""

    return _fit_fixed_logistic(
        frame,
        target,
        features=features,
        slate_column=slate_column,
        model_id=model_id,
        random_state=random_state,
    )


def _weights(frame: pd.DataFrame) -> FloatArray:
    if "row_weight" in frame:
        result = frame["row_weight"].to_numpy(dtype=float)
    elif "slate_id" in frame:
        sizes = frame.groupby("slate_id", sort=True)["slate_id"].transform("size")
        result = 1.0 / sizes.to_numpy(dtype=float)
    else:
        result = np.ones(len(frame), dtype=float)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("metric weights must be finite and positive")
    return np.asarray(result, dtype=np.float64)


def _weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def paired_loss_improvements(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    candidate: str,
) -> dict[str, float]:
    """Return baseline-minus-candidate losses so positive means improvement."""

    labels = frame[target].to_numpy(dtype=float)
    base = np.clip(frame[baseline].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON)
    cand = np.clip(frame[candidate].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON)
    weights = _weights(frame)
    brier = _weighted_mean((base - labels) ** 2 - (cand - labels) ** 2, weights)
    base_log = -(labels * np.log(base) + (1.0 - labels) * np.log1p(-base))
    candidate_log = -(labels * np.log(cand) + (1.0 - labels) * np.log1p(-cand))
    log_loss = _weighted_mean(base_log - candidate_log, weights)
    return {"brier_improvement": brier, "log_loss_improvement": log_loss}


def _calibration_coefficients(
    labels: FloatArray, probability: FloatArray, weights: FloatArray
) -> tuple[float, float]:
    predictor = logit_probability(probability)
    design = np.column_stack((np.ones(len(predictor)), predictor))
    beta = np.asarray([0.0, 1.0], dtype=np.float64)
    for _ in range(100):
        linear = np.clip(design @ beta, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (weights * (labels - fitted))
        variance = weights * fitted * (1.0 - fitted)
        information = design.T @ (design * variance[:, None])
        information += np.eye(2) * 1e-10
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def probability_metrics(
    frame: pd.DataFrame,
    *,
    target: str,
    probability: str,
    model_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute weighted discrimination, loss, calibration, and reliability."""

    labels = frame[target].to_numpy(dtype=float)
    prediction = frame[probability].to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError(f"{model_id} contains invalid probabilities")
    clipped = np.clip(prediction, EPSILON, 1.0 - EPSILON)
    weights = _weights(frame)
    brier = _weighted_mean((prediction - labels) ** 2, weights)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    log_loss = _weighted_mean(losses, weights)
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, prediction, sample_weight=weights))
        calibration_intercept, calibration_slope = _calibration_coefficients(
            np.asarray(labels, dtype=np.float64),
            np.asarray(prediction, dtype=np.float64),
            weights,
        )
    else:
        auc = float("nan")
        calibration_intercept, calibration_slope = float("nan"), float("nan")
    bin_index = np.minimum((prediction * 10.0).astype(int), 9)
    bin_rows: list[dict[str, Any]] = []
    ece = 0.0
    total_weight = float(weights.sum())
    for index in range(10):
        mask = bin_index == index
        bin_weight = float(weights[mask].sum())
        if mask.any():
            mean_prediction = _weighted_mean(prediction[mask], weights[mask])
            outcome_rate = _weighted_mean(labels[mask], weights[mask])
            ece += (bin_weight / total_weight) * abs(mean_prediction - outcome_rate)
        else:
            mean_prediction = float("nan")
            outcome_rate = float("nan")
        bin_rows.append(
            {
                "model": model_id,
                "target": target,
                "bin": index + 1,
                "lower": index / 10.0,
                "upper": (index + 1) / 10.0,
                "rows": int(mask.sum()),
                "weight": bin_weight,
                "mean_prediction": mean_prediction,
                "outcome_rate": outcome_rate,
            }
        )
    metrics = {
        "model": model_id,
        "target": target,
        "brier_score": brier,
        "log_loss": log_loss,
        "auc": auc,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "expected_calibration_error": float(ece),
        "outcome_rate": _weighted_mean(labels, weights),
        "mean_prediction": _weighted_mean(prediction, weights),
        "row_count": int(len(frame)),
        "session_count": int(frame["session"].nunique()),
        "stock_count": int(frame["stock"].nunique()),
    }
    return metrics, pd.DataFrame(bin_rows)


def session_block_bootstrap_improvements(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    candidate: str,
    draws: int = 500,
    seed: int = 20260720,
) -> pd.DataFrame:
    """Paired whole-session bootstrap retaining every row and target."""

    if draws <= 0 or draws > 500:
        raise ValueError("bootstrap draws must be in [1, 500]")
    sessions = np.asarray(sorted(frame["session"].astype(str).unique()), dtype=object)
    if len(sessions) == 0:
        raise ValueError("bootstrap requires sessions")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for draw in range(draws):
        sampled = rng.choice(sessions, size=len(sessions), replace=True).astype(str)
        parts: list[pd.DataFrame] = []
        for occurrence, session in enumerate(sampled):
            part = frame.loc[frame["session"].astype(str).eq(session)].copy()
            original_slate = (
                part["slate_id"].astype(str)
                if "slate_id" in part
                else part["session"].astype(str) + "|" + part["decision_ordinal"].astype(str)
            )
            part["slate_id"] = f"{occurrence}|" + original_slate
            parts.append(part)
        sampled_frame = pd.concat(parts, ignore_index=True)
        slate_sizes = sampled_frame.groupby("slate_id", sort=True)["slate_id"].transform("size")
        sampled_frame["joined_slate_size"] = slate_sizes.astype(int)
        sampled_frame["row_weight"] = 1.0 / slate_sizes.to_numpy(dtype=float)
        improvement = paired_loss_improvements(
            sampled_frame,
            target=target,
            baseline=baseline,
            candidate=candidate,
        )
        rows.extend(
            [
                {
                    "draw": draw,
                    "metric": "brier",
                    "improvement": improvement["brier_improvement"],
                    "sampled_session_count": len(sampled),
                },
                {
                    "draw": draw,
                    "metric": "log_loss",
                    "improvement": improvement["log_loss_improvement"],
                    "sampled_session_count": len(sampled),
                },
            ]
        )
    return pd.DataFrame(rows)


def bootstrap_intervals(draws: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return fixed 90% and 95% percentile intervals by metric."""

    output: dict[str, dict[str, float]] = {}
    for metric, group in draws.groupby("metric", sort=True):
        values = group["improvement"].to_numpy(dtype=float)
        output[str(metric)] = {
            "lower_90": float(np.quantile(values, 0.05)),
            "upper_90": float(np.quantile(values, 0.95)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return output


def whole_session_shift(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    draw: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Shift every same-ordinal session block without any identity assignment.

    Exact stock-membership groups are required so a complete stock cross-section
    can move as one block. Singleton groups make the preregistered null
    unidentified and therefore fail closed instead of remaining unshifted.
    """

    feasibility = whole_session_shift_feasibility(frame)
    if int(feasibility["unshiftable_blocks"].sum()) > 0:
        raise RuntimeError(
            "blocked_join_semantics_failure: whole-session null has "
            "unshiftable stock-membership blocks"
        )

    shifted, manifest = circular_shift_session_blocks(
        frame,
        value_columns=value_columns,
        draw=draw,
        seed=seed,
        session_column="session",
        ordinal_column="decision_ordinal",
        symbol_column="stock",
    )
    if any(row["source_session"] == row["destination_session"] for row in manifest):
        raise RuntimeError(
            "blocked_join_semantics_failure: whole-session null retained identity blocks"
        )
    return shifted, manifest


def whole_session_shift_feasibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Report whether exact-membership session blocks can all be shifted."""

    _require_columns(frame, {"session", "decision_ordinal", "stock"}, "null panel")
    groups: dict[tuple[int, tuple[str, ...]], list[str]] = {}
    for (ordinal, session), slate in frame.groupby(
        ["decision_ordinal", "session"], sort=True, observed=True
    ):
        membership = tuple(sorted(slate["stock"].astype(str)))
        groups.setdefault((int(str(ordinal)), membership), []).append(str(session))
    rows: list[dict[str, Any]] = []
    for (ordinal, membership), raw_sessions in sorted(groups.items()):
        sessions = sorted(set(raw_sessions))
        shiftable = len(sessions) if len(sessions) > 1 else 0
        rows.append(
            {
                "decision_ordinal": ordinal,
                "membership_hash": hashlib.sha256("|".join(membership).encode("utf-8")).hexdigest()[
                    :16
                ],
                "membership_size": len(membership),
                "session_blocks": len(sessions),
                "shiftable_blocks": shiftable,
                "unshiftable_blocks": len(sessions) - shiftable,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["decision_ordinal", "membership_hash"], kind="mergesort")
        .reset_index(drop=True)
    )


def evaluate_support(frame: pd.DataFrame) -> dict[str, Any]:
    """Apply assessment support and concentration gates without relaxing them."""

    required = {
        "session",
        "stock",
        "large_move",
        "immediate_pair_closure",
        "joint_large_move_and_closure",
        "pair_orientation",
    }
    _require_columns(frame, required, "assessment support panel")
    month = pd.to_datetime(frame["session"], errors="raise").dt.strftime("%Y-%m")
    stock_fraction = frame["stock"].astype(str).value_counts(normalize=True)
    month_fraction = month.value_counts(normalize=True)
    orientation_fraction = frame["pair_orientation"].astype(str).value_counts(normalize=True)
    result: dict[str, Any] = {
        "joined_rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["stock"].nunique()),
        "immediate_closures": int(frame["immediate_pair_closure"].sum()),
        "large_moves": int(frame["large_move"].sum()),
        "joint_positive_events": int(frame["joint_large_move_and_closure"].sum()),
        "maximum_stock_fraction": float(stock_fraction.max()) if len(frame) else float("nan"),
        "maximum_month_fraction": float(month_fraction.max()) if len(frame) else float("nan"),
        "maximum_pair_orientation_fraction": (
            float(orientation_fraction.max()) if len(frame) else float("nan")
        ),
    }
    concentration_passed = bool(
        result["maximum_stock_fraction"] <= 0.125
        and result["maximum_month_fraction"] <= 0.25
        and result["maximum_pair_orientation_fraction"] <= 0.20
    )
    primary_passed = bool(
        result["joined_rows"] >= 1_500
        and result["sessions"] >= 75
        and result["stocks"] >= 15
        and result["immediate_closures"] >= 250
        and result["large_moves"] >= 250
        and concentration_passed
    )
    result["concentration_passed"] = concentration_passed
    result["primary_support_passed"] = primary_passed
    result["joint_support_status"] = (
        "sufficient"
        if result["joint_positive_events"] >= 100
        else "secondary_insufficient_joint_support"
    )
    result["blocker"] = None if primary_passed else "blocked_insufficient_joint_increment_support"
    return result


def primary_arm_passes(
    *,
    brier_improvement: float,
    log_loss_improvement: float,
    brier_lower_90: float,
    log_loss_lower_90: float,
    positive_months: int,
    represented_months: int,
    null_percentile: float,
    concentration_passed: bool,
) -> bool:
    """Apply the exact Arm A/B primary gate."""

    stable_months = positive_months >= 4 and positive_months >= math.ceil(0.60 * represented_months)
    return bool(
        brier_improvement > 0.0
        and log_loss_improvement > 0.0
        and brier_lower_90 >= 0.0
        and log_loss_lower_90 >= 0.0
        and stable_months
        and null_percentile >= 0.90
        and concentration_passed
    )


def joint_arm_passes(
    *,
    brier_improvement: float,
    log_loss_improvement: float,
    brier_lower_90: float,
    log_loss_lower_90: float,
    positive_months: int,
    represented_months: int,
    joint_support_status: str,
) -> bool:
    """Apply the secondary Arm C gate."""

    return bool(
        brier_improvement > 0.0
        and log_loss_improvement > 0.0
        and brier_lower_90 >= 0.0
        and log_loss_lower_90 >= 0.0
        and positive_months > represented_months / 2.0
        and joint_support_status == "sufficient"
    )


def classify_joint_decision(
    *,
    arm_a_pass: bool,
    arm_b_pass: bool,
    arm_c_pass: bool,
    blocker: str | None = None,
) -> str:
    """Return exactly one preregistered primary decision."""

    if blocker is not None:
        if blocker not in DECISIONS or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown blocker decision: {blocker}")
        return blocker
    if arm_a_pass and arm_b_pass:
        return "mutually_informative_movement_closure_process"
    if arm_a_pass:
        return "movement_adds_to_closure_only"
    if arm_b_pass:
        return "closure_history_adds_to_movement_only"
    if arm_c_pass:
        return "joint_interaction_only"
    return "separate_predictable_processes_no_increment"


__all__ = [
    "DECISIONS",
    "EPSILON",
    "MAX_JOINED_ROWS",
    "SAFETY_FLAGS",
    "FrozenLinearModel",
    "JoinResult",
    "add_joint_probability_features",
    "assert_compact_panel_has_no_forbidden_fields",
    "assert_protected_date_boundary",
    "assert_upstream_chronology",
    "bootstrap_intervals",
    "classify_joint_decision",
    "evaluate_support",
    "exact_active_pair_join",
    "fit_fixed_logistic",
    "joint_arm_passes",
    "logit_probability",
    "paired_loss_improvements",
    "primary_arm_passes",
    "probability_metrics",
    "session_block_bootstrap_improvements",
    "split_development_assessment",
    "whole_session_shift",
    "whole_session_shift_feasibility",
    "with_equal_slate_weights",
]
