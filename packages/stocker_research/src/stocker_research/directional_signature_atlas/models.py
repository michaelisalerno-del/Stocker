"""Frozen baselines, atlas voting, and predictive/economic metrics."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stocker_research.directional_signature_atlas.analysis import signature_from_dict
from stocker_research.directional_signature_atlas.signatures import apply_signature

CLASSES = ("LONG", "SHORT", "NEUTRAL")
PRIOR_PRICE_NUMERIC = (
    "barrier_bps",
    "current_range_scale",
    "current_body_scale",
    "current_close_location",
    "current_upper_wick_fraction",
    "current_lower_wick_fraction",
    "return_1_scale",
    "return_3_scale",
    "return_6_scale",
    "return_12_scale",
    "mean_abs_return_6_scale",
    "compression_3_to_12",
    "session_return_scale",
    "session_mean_distance_scale",
    "opening_range_position",
    "opening_range_width_scale",
)


def _laplace_probabilities(group: pd.DataFrame) -> dict[str, float]:
    counts = group["target"].value_counts()
    denominator = len(group) + len(CLASSES)
    return {label: float((counts.get(label, 0) + 1.0) / denominator) for label in CLASSES}


def _conditional_probabilities(
    discovery: pd.DataFrame,
    full: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    base = _laplace_probabilities(discovery)
    mapping: dict[tuple[Any, ...], dict[str, float]] = {}
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, group in discovery.groupby(grouper, dropna=False, sort=True):
        normalized = key if isinstance(key, tuple) else (key,)
        mapping[normalized] = _laplace_probabilities(group)
    rows = []
    for values in full[columns].itertuples(index=False, name=None):
        probabilities = mapping.get(tuple(values), base)
        rows.append([probabilities[label] for label in CLASSES])
    return np.asarray(rows, dtype=float)


def _hard_probabilities(states: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(states), len(CLASSES)), dtype=float)
    for index, label in enumerate(CLASSES):
        probabilities[:, index] = states == label
    return probabilities


def _fit_logistic(
    discovery: pd.DataFrame,
    full: pd.DataFrame,
    *,
    numeric: list[str],
    categorical: list[str],
) -> np.ndarray:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
    model = Pipeline(
        [
            ("features", transformer),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    random_state=20260717,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(discovery[[*numeric, *categorical]], discovery["target"])
    raw = model.predict_proba(full[[*numeric, *categorical]])
    fitted = model.named_steps["model"]
    classes = [str(value) for value in fitted.classes_]
    aligned = np.zeros((len(full), len(CLASSES)), dtype=float)
    for index, label in enumerate(CLASSES):
        aligned[:, index] = raw[:, classes.index(label)]
    return aligned


def _prior_price_context_prequential(
    full: pd.DataFrame,
    first_touch: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the predecessor M1 equation and first-touch training target."""

    target = first_touch[["opportunity_id", "first_touch_target"]].copy()
    target["actual_class"] = target["first_touch_target"].map(
        {
            "UPPER_FIRST": "long",
            "LOWER_FIRST": "short",
            "NEITHER": "neutral",
            "SAME_BAR_DUAL_TOUCH": "neutral",
        }
    )
    joined = full.merge(target, on="opportunity_id", how="left", validate="one_to_one")
    usable = joined.loc[joined["score_status"].eq("scored") & joined["actual_class"].notna()].copy()
    probabilities = np.full((len(full), 3), 1.0 / 3.0, dtype=float)
    states = np.full(len(full), "NEUTRAL", dtype=object)
    eligible = np.zeros(len(full), dtype=bool)
    full_positions = pd.Series(np.arange(len(full)), index=full["opportunity_id"].astype(str))
    score_dates = sorted(
        usable.loc[usable["period"].isin([2025, 2026]), "session"].astype(str).unique()
    )
    for score_date in score_dates:
        prior_dates = sorted(
            usable.loc[usable["session"].astype(str).lt(score_date), "session"].astype(str).unique()
        )
        training_dates = prior_dates[-120:]
        if len(training_dates) < 60:
            continue
        train = usable.loc[usable["session"].astype(str).isin(training_dates)].copy()
        score = usable.loc[usable["session"].astype(str).eq(score_date)].copy()
        if len(train) < 1500 or score.empty or train["actual_class"].nunique() < 3:
            continue
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        model = Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            ("numeric", numeric_pipeline, list(PRIOR_PRICE_NUMERIC)),
                            ("categorical", categorical_pipeline, ["decision_clock"]),
                        ]
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        max_iter=1000,
                        random_state=20260714,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        columns = [*PRIOR_PRICE_NUMERIC, "decision_clock"]
        model.fit(train[columns], train["actual_class"])
        raw = model.predict_proba(score[columns])
        classes = [str(value) for value in model.named_steps["model"].classes_]
        aligned = np.column_stack(
            [raw[:, classes.index(label)] for label in ("long", "short", "neutral")]
        )
        for offset, row in enumerate(score.itertuples(index=False)):
            position = int(full_positions.loc[str(row.opportunity_id)])
            probabilities[position] = aligned[offset]
            barrier = float(cast(Any, row.barrier_bps))
            frozen_cost = float(cast(Any, row.round_trip_cost_bps))
            long_ev = (aligned[offset, 0] - aligned[offset, 1]) * barrier - frozen_cost
            short_ev = (aligned[offset, 1] - aligned[offset, 0]) * barrier - frozen_cost
            if long_ev > 0.0 and long_ev > short_ev:
                states[position] = "LONG"
            elif short_ev > 0.0 and short_ev > long_ev:
                states[position] = "SHORT"
            eligible[position] = True
    return probabilities, states, eligible


def baseline_predictions(
    discovery: pd.DataFrame,
    full: pd.DataFrame,
    *,
    first_touch: pd.DataFrame,
) -> pd.DataFrame:
    """Fit every required baseline on discovery and score the identical population."""

    outputs: list[pd.DataFrame] = []

    def add(
        model_id: str,
        probabilities: np.ndarray,
        *,
        state: np.ndarray | None = None,
        metric_eligible: np.ndarray | None = None,
    ) -> None:
        if state is None:
            state = np.asarray(CLASSES, dtype=object)[np.argmax(probabilities, axis=1)]
        if metric_eligible is None:
            metric_eligible = np.ones(len(full), dtype=bool)
        outputs.append(
            pd.DataFrame(
                {
                    "opportunity_id": full["opportunity_id"].to_numpy(),
                    "period": full["period"].to_numpy(),
                    "chronology_stage": full.get(
                        "chronology_stage", full["period"].astype(str)
                    ).to_numpy(),
                    "session": full["session"].to_numpy(),
                    "symbol": full["symbol"].to_numpy(),
                    "decision_clock": full["decision_clock"].to_numpy(),
                    "model_id": model_id,
                    "predicted_state": state,
                    "p_long": probabilities[:, 0],
                    "p_short": probabilities[:, 1],
                    "p_neutral": probabilities[:, 2],
                    "metric_eligible": metric_eligible,
                }
            )
        )

    neutral = np.zeros((len(full), 3), dtype=float)
    neutral[:, 2] = 1.0
    add("always_neutral", neutral)
    add("clock_only_base_rate", _conditional_probabilities(discovery, full, ["decision_clock"]))

    prior_probabilities, prior_state, prior_eligible = _prior_price_context_prequential(
        full, first_touch
    )
    add(
        "prior_static_price_context_multinomial",
        prior_probabilities,
        state=prior_state,
        metric_eligible=prior_eligible,
    )
    return_1 = pd.to_numeric(full["return_1_scale"], errors="coerce")
    momentum = np.where(return_1 > 0.0, "LONG", np.where(return_1 < 0.0, "SHORT", "NEUTRAL"))
    reversal = np.where(return_1 > 0.0, "SHORT", np.where(return_1 < 0.0, "LONG", "NEUTRAL"))
    add("one_bar_momentum", _hard_probabilities(momentum))
    add("one_bar_reversal", _hard_probabilities(reversal))
    opening = pd.to_numeric(full["opening_range_position"], errors="coerce")
    opening_state = np.where(opening > 0.5, "LONG", np.where(opening < 0.5, "SHORT", "NEUTRAL"))
    add("opening_range_position_sign", _hard_probabilities(opening_state))
    add("current_state_alone", _conditional_probabilities(discovery, full, ["current_state"]))
    add(
        "current_state_plus_history",
        _conditional_probabilities(discovery, full, ["current_state", "state_motif_3"]),
    )
    permitted_momentum = np.where(
        full["movement_permission"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        momentum,
        "NEUTRAL",
    )
    add("movement_permission_plus_momentum", _hard_probabilities(permitted_momentum))
    compact_categorical = [
        "decision_clock",
        "current_state",
        "state_motif_3",
        "return_1_bin",
        "return_6_bin",
        "opening_range_position_bin",
        "compression_bin",
        "rolling_high_low_location_bin",
        "return_6_cross_sectional_rank_bin",
        "universe_breadth_bin",
        "parent_loop_family",
        "movement_permission",
    ]
    add(
        "shallow_logistic_compact_features",
        _fit_logistic(discovery, full, numeric=[], categorical=compact_categorical),
    )
    return pd.concat(outputs, ignore_index=True)


def apply_atlas_controller(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
    *,
    base_probabilities: dict[str, float],
) -> pd.DataFrame:
    """Apply frozen one-rule/one-vote logic without holdout weighting."""

    long_votes = np.zeros(len(frame), dtype=int)
    short_votes = np.zeros(len(frame), dtype=int)
    long_value_sums = np.zeros(len(frame), dtype=float)
    short_value_sums = np.zeros(len(frame), dtype=float)
    firing: list[list[str]] = [[] for _ in range(len(frame))]
    probability_sums = np.zeros((len(frame), 3), dtype=float)
    probability_counts = np.zeros(len(frame), dtype=int)
    required_features = {
        str(condition["feature"])
        for entry in library
        for condition in entry["signature"]["conditions"]
    }
    missing_required = np.zeros(len(frame), dtype=bool)
    for feature in sorted(required_features):
        if feature not in frame:
            missing_required[:] = True
        else:
            missing_required |= frame[feature].isna().to_numpy()
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        mask = apply_signature(frame, signature).to_numpy(bool)
        if signature.direction == "LONG":
            long_votes += mask.astype(int)
            long_value_sums[mask] += float(entry.get("conservative_value_bps", 0.0))
        else:
            short_votes += mask.astype(int)
            short_value_sums[mask] += float(entry.get("conservative_value_bps", 0.0))
        conditional = entry["frozen_class_probabilities"]
        vector = np.asarray([conditional[label] for label in CLASSES], dtype=float)
        probability_sums[mask] += vector
        probability_counts[mask] += 1
        for position in np.flatnonzero(mask):
            firing[position].append(signature.signature_id)
    movement_available = frame["movement_permission"].notna().to_numpy()
    movement = frame["movement_permission"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    conflict = (long_votes > 0) & (short_votes > 0)
    no_vote = (long_votes == 0) & (short_votes == 0)
    long_value_positive = (long_votes > 0) & (long_value_sums / np.maximum(long_votes, 1) > 0.0)
    short_value_positive = (short_votes > 0) & (short_value_sums / np.maximum(short_votes, 1) > 0.0)
    state = np.full(len(frame), "NEUTRAL", dtype=object)
    state[
        movement
        & ~missing_required
        & ~conflict
        & (long_votes > 0)
        & (short_votes == 0)
        & long_value_positive
    ] = "LONG"
    state[
        movement
        & ~missing_required
        & ~conflict
        & (short_votes > 0)
        & (long_votes == 0)
        & short_value_positive
    ] = "SHORT"
    reason = np.full(len(frame), "no_directional_vote", dtype=object)
    reason[~movement] = "movement_permission_failed"
    reason[~movement_available] = "required_causal_feature_unavailable"
    reason[movement & conflict] = "conflicting_votes"
    reason[movement & no_vote] = "no_directional_vote"
    reason[movement & ~conflict & ~no_vote & ~long_value_positive & ~short_value_positive] = (
        "non_positive_conservative_value"
    )
    reason[state != "NEUTRAL"] = "supported_directional_vote"
    reason[missing_required] = "required_causal_feature_unavailable"
    base = np.asarray([base_probabilities[label] for label in CLASSES], dtype=float)
    probabilities = np.tile(base, (len(frame), 1))
    has_probability = probability_counts > 0
    probabilities[has_probability] = (
        probability_sums[has_probability] / probability_counts[has_probability, None]
    )
    output_columns = ["opportunity_id", "period", "session", "symbol", "decision_clock"]
    if "chronology_stage" in frame:
        output_columns.append("chronology_stage")
    output = frame[output_columns].copy()
    output["model_id"] = "directional_signature_atlas_v1"
    output["predicted_state"] = state
    output["long_vote_count"] = long_votes
    output["short_vote_count"] = short_votes
    output["conflict"] = conflict
    output["reason_code"] = reason
    output["firing_signature_ids_json"] = [json_dumps(sorted(values)) for values in firing]
    output["p_long"] = probabilities[:, 0]
    output["p_short"] = probabilities[:, 1]
    output["p_neutral"] = probabilities[:, 2]
    return output


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probability)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(float(probability[mask].mean() - target[mask].mean()))
    return value if total else math.nan


def _calibration_slope(probability: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if len(np.unique(target)) < 2 or float(np.std(logit)) == 0.0:
        return math.nan
    model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    model.fit(logit, target)
    return float(model.coef_[0, 0])


def prediction_metrics(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate predictive and economic metrics for each model and period."""

    joined = predictions.merge(
        outcomes[
            [
                "opportunity_id",
                "target",
                "long_net_bps",
                "short_net_bps",
                "round_trip_cost_bps",
            ]
        ],
        on="opportunity_id",
        how="inner",
        validate="many_to_one",
    )
    joined = joined.loc[joined["target"].ne("UNAVAILABLE")].copy()
    if "metric_eligible" in joined:
        metric_eligible = (
            joined["metric_eligible"].astype("boolean").fillna(True).to_numpy(dtype=bool)
        )
        joined = joined.loc[metric_eligible].copy()
    predictive_rows: list[dict[str, Any]] = []
    economic_rows: list[dict[str, Any]] = []
    stage_column = "chronology_stage" if "chronology_stage" in joined else "period"
    for (model_id, stage), group in joined.groupby(["model_id", stage_column], sort=True):
        period = int(group["period"].iloc[0])
        order_columns = [
            column
            for column in ("decision_timestamp", "session", "decision_clock", "symbol")
            if column in group
        ]
        group = group.sort_values(order_columns, kind="mergesort")
        truth = group["target"].astype(str)
        long_target = truth.eq("LONG").to_numpy(float)
        short_target = truth.eq("SHORT").to_numpy(float)
        neutral_target = truth.eq("NEUTRAL").to_numpy(float)
        p_long = group["p_long"].to_numpy(float)
        p_short = group["p_short"].to_numpy(float)
        p_neutral = group["p_neutral"].to_numpy(float)
        probabilities = group[["p_long", "p_short", "p_neutral"]].to_numpy(float)
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predicted = group["predicted_state"].astype(str)
        directional = predicted.isin(["LONG", "SHORT"])
        correct_direction = predicted.eq(truth) & directional
        false_direction = (predicted.eq("LONG") & truth.eq("SHORT")) | (
            predicted.eq("SHORT") & truth.eq("LONG")
        )
        predicted_long = predicted.eq("LONG")
        predicted_short = predicted.eq("SHORT")
        base_long_rate = float(truth.eq("LONG").mean())
        base_short_rate = float(truth.eq("SHORT").mean())
        long_precision = (
            float(truth.loc[predicted_long].eq("LONG").mean()) if predicted_long.any() else math.nan
        )
        short_precision = (
            float(truth.loc[predicted_short].eq("SHORT").mean())
            if predicted_short.any()
            else math.nan
        )
        directional_base_precision = (
            float(
                (predicted_long.sum() * base_long_rate + predicted_short.sum() * base_short_rate)
                / directional.sum()
            )
            if directional.any()
            else math.nan
        )
        directional_precision = (
            float(correct_direction.sum() / directional.sum()) if directional.any() else math.nan
        )
        predictive_rows.append(
            {
                "model_id": str(model_id),
                "period": period,
                "chronology_stage": str(stage),
                "rows": len(group),
                "brier_long": float(np.mean(np.square(p_long - long_target))),
                "brier_short": float(np.mean(np.square(p_short - short_target))),
                "brier_neutral": float(np.mean(np.square(p_neutral - neutral_target))),
                "macro_brier": float(
                    np.mean(
                        [
                            np.mean(np.square(p_long - long_target)),
                            np.mean(np.square(p_short - short_target)),
                            np.mean(np.square(p_neutral - neutral_target)),
                        ]
                    )
                ),
                "log_loss": float(
                    -np.mean(
                        np.log(
                            probabilities[
                                np.arange(len(group)),
                                [CLASSES.index(label) for label in truth],
                            ]
                        )
                    )
                ),
                "ece_long": _ece(p_long, long_target),
                "ece_short": _ece(p_short, short_target),
                "ece_neutral": _ece(p_neutral, neutral_target),
                "calibration_slope_long": _calibration_slope(p_long, long_target),
                "calibration_slope_short": _calibration_slope(p_short, short_target),
                "base_long_rate": base_long_rate,
                "base_short_rate": base_short_rate,
                "long_precision": long_precision,
                "short_precision": short_precision,
                "long_precision_lift": long_precision - base_long_rate
                if math.isfinite(long_precision)
                else math.nan,
                "short_precision_lift": short_precision - base_short_rate
                if math.isfinite(short_precision)
                else math.nan,
                "directional_base_precision": directional_base_precision,
                "directional_precision": directional_precision,
                "directional_precision_lift": directional_precision - directional_base_precision
                if math.isfinite(directional_precision)
                and math.isfinite(directional_base_precision)
                else math.nan,
                "directional_recall": float(
                    correct_direction.sum() / truth.isin(["LONG", "SHORT"]).sum()
                )
                if truth.isin(["LONG", "SHORT"]).any()
                else math.nan,
                "false_direction_rate": float(false_direction.mean()),
                "auc_long": float(roc_auc_score(long_target, p_long))
                if len(np.unique(long_target)) == 2
                else math.nan,
                "auc_short": float(roc_auc_score(short_target, p_short))
                if len(np.unique(short_target)) == 2
                else math.nan,
            }
        )
        payoff = np.where(
            predicted.eq("LONG"),
            group["long_net_bps"],
            np.where(predicted.eq("SHORT"), group["short_net_bps"], 0.0),
        ).astype(float)
        payoff_frame = group.assign(_directional_payoff=payoff)
        if "decision_timestamp" in payoff_frame:
            batched_payoff = payoff_frame.groupby("decision_timestamp", sort=True)[
                "_directional_payoff"
            ].sum()
        else:
            batched_payoff = payoff_frame.groupby(["session", "decision_clock"], sort=True)[
                "_directional_payoff"
            ].sum()
        cumulative = np.cumsum(batched_payoff.to_numpy(float))
        peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
        losses = abs(float(payoff[payoff < 0.0].sum()))
        profit_factor = (
            float(payoff[payoff > 0.0].sum()) / losses
            if losses > 0.0
            else (math.inf if (payoff > 0.0).any() else math.nan)
        )
        economic_rows.append(
            {
                "model_id": str(model_id),
                "period": period,
                "chronology_stage": str(stage),
                "opportunities": len(group),
                "directional_outputs": int(directional.sum()),
                "directional_coverage": float(directional.mean()),
                "neutral_rate": float(predicted.eq("NEUTRAL").mean()),
                "long_count": int(predicted.eq("LONG").sum()),
                "short_count": int(predicted.eq("SHORT").sum()),
                "conflict_count": int(group["conflict"].sum()) if "conflict" in group else 0,
                "mean_net_bps_per_directional_output": float(payoff[directional].mean())
                if directional.any()
                else math.nan,
                "net_bps_per_full_opportunity": float(payoff.mean()),
                "total_net_bps": float(payoff.sum()),
                "total_cost_bps": float(group.loc[directional, "round_trip_cost_bps"].sum()),
                "hit_rate": float((payoff[directional] > 0.0).mean())
                if directional.any()
                else math.nan,
                "profit_factor": profit_factor,
                "maximum_drawdown_bps": float(np.max(peak - cumulative))
                if len(cumulative)
                else math.nan,
            }
        )
    return pd.DataFrame(predictive_rows), pd.DataFrame(economic_rows)
