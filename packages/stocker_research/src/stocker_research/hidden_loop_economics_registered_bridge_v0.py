"""Pure helpers for the frozen hidden-loop economics and bridge quick screen.

The public surface is retrospective research infrastructure only. It contains no
broker, order, position, sizing, deployment, or runtime integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

FROZEN_FAMILIES: Final[tuple[str, ...]] = (
    "unregistered_primitive_like__5-6-5",
    "unregistered_primitive_like__2-3-2",
    "unregistered_primitive_like__2-5-2",
    "unregistered_primitive_like__4-7-4",
)
OTHER_FAMILY: Final[str] = "OTHER_UNREGISTERED_FAMILY"
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2025-08-23")


@dataclass(frozen=True)
class FittedWeightedLogistic:
    """A serialisable standardisation-plus-logistic research model."""

    features: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict positive-class probabilities in deterministic row order."""

        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        return np.asarray(
            self.estimator.predict_proba(self.scaler.transform(matrix))[:, 1],
            dtype=float,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return all quantities needed for independent probability reconstruction."""

        return {
            "feature_names": list(self.features),
            "scaler_mean": self.scaler.mean_.astype(float).tolist(),
            "scaler_scale": self.scaler.scale_.astype(float).tolist(),
            "coefficient": self.estimator.coef_[0].astype(float).tolist(),
            "intercept": float(self.estimator.intercept_[0]),
            "n_iter": int(self.estimator.n_iter_[0]),
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
        }


def fit_weighted_logistic(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    weight_column: str = "row_weight",
) -> FittedWeightedLogistic:
    """Fit the frozen deterministic weighted L2 logistic specification."""

    required = {target, weight_column, *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"model-fit columns missing: {missing}")
    labels = frame[target].to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("model fitting requires both target classes")
    matrix = frame.loc[:, list(features)].to_numpy(dtype=float)
    weights = frame[weight_column].to_numpy(dtype=float)
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(weights).all()
        or bool((weights <= 0.0).any())
    ):
        raise ValueError("model fitting requires finite features and positive weights")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    transformed = scaler.fit_transform(matrix)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        max_iter=300,
        class_weight=None,
        n_jobs=1,
        random_state=20260721,
    )
    estimator.fit(transformed, labels, sample_weight=weights)
    if bool(np.any(estimator.n_iter_ >= 300)):
        raise ValueError("weighted logistic model did not converge")
    return FittedWeightedLogistic(features, scaler, estimator)


def reconstruct_serialised_probability(
    frame: pd.DataFrame, specification: dict[str, Any]
) -> np.ndarray:
    """Reconstruct probabilities directly from serialised scaler coefficients."""

    features = [str(value) for value in specification["feature_names"]]
    matrix = frame.loc[:, features].to_numpy(dtype=float)
    mean = np.asarray(specification["scaler_mean"], dtype=float)
    scale = np.asarray(specification["scaler_scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    intercept = float(specification["intercept"])
    if not (
        matrix.shape[1] == len(mean) == len(scale) == len(coefficient)
        and np.isfinite(matrix).all()
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and np.isfinite(coefficient).all()
        and np.isfinite(intercept)
        and bool((scale > 0.0).all())
    ):
        raise ValueError("serialised model specification is invalid")
    logits = ((matrix - mean) / scale) @ coefficient + intercept
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0))), dtype=float)


def _weighted_logistic_calibration(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Estimate calibration intercept/slope with a bounded two-parameter IRLS."""

    if set(labels.astype(int)) != {0, 1}:
        return float("nan"), float("nan")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(len(labels), dtype=float), logits])
    beta = np.asarray([0.0, 1.0], dtype=float)
    for _ in range(50):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35.0, 35.0)))
        gradient = design.T @ (weights * (labels - fitted))
        curvature_weight = weights * fitted * (1.0 - fitted)
        information = design.T @ (curvature_weight[:, None] * design)
        information += np.eye(2, dtype=float) * 1e-12
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += step
        if float(np.max(np.abs(step))) <= 1e-10:
            break
    return float(beta[0]), float(beta[1])


def binary_model_metrics(
    labels: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Calculate the fixed bridge-model metric surface with slate weights."""

    target = np.asarray(labels, dtype=int)
    prediction = np.asarray(probabilities, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    if not (
        len(target) == len(prediction) == len(sample_weight)
        and len(target) > 0
        and np.isfinite(prediction).all()
        and np.isfinite(sample_weight).all()
        and bool((sample_weight > 0.0).all())
        and bool(((prediction >= 0.0) & (prediction <= 1.0)).all())
    ):
        raise ValueError("binary metrics require aligned finite inputs")
    total_weight = float(np.sum(sample_weight))
    brier = float(np.sum(sample_weight * (prediction - target) ** 2) / total_weight)
    realised_probability = np.where(target == 1, prediction, 1.0 - prediction)
    base_rate = float(np.sum(sample_weight * target) / total_weight)
    if set(target) == {0, 1}:
        auc = float(roc_auc_score(target, prediction, sample_weight=sample_weight))
        average_precision = float(
            average_precision_score(target, prediction, sample_weight=sample_weight)
        )
    else:
        auc = float("nan")
        average_precision = float("nan")
    calibration_intercept, calibration_slope = _weighted_logistic_calibration(
        target, prediction, sample_weight
    )
    bin_id = np.minimum((prediction * 10.0).astype(int), 9)
    calibration_error = 0.0
    for value in range(10):
        mask = bin_id == value
        if not bool(mask.any()):
            continue
        bin_weight = sample_weight[mask]
        bin_total = float(bin_weight.sum())
        predicted = float(np.sum(bin_weight * prediction[mask]) / bin_total)
        observed = float(np.sum(bin_weight * target[mask]) / bin_total)
        calibration_error += bin_total / total_weight * abs(predicted - observed)
    return {
        "log_loss": float(log_loss(target, prediction, sample_weight=sample_weight, labels=[0, 1])),
        "brier_score": brier,
        "auc": auc,
        "average_precision": average_precision,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "expected_calibration_error": float(calibration_error),
        "base_rate": base_rate,
        "rows": int(len(target)),
        "mean_probability_realised_class": float(
            np.sum(sample_weight * realised_probability) / total_weight
        ),
    }


def reject_protected_dates(frame: pd.DataFrame, *, column: str = "session") -> None:
    """Fail closed if a materialised analytical row reaches the protected boundary."""

    if column not in frame:
        raise ValueError(f"{column} column is required")
    timestamps = pd.to_datetime(frame[column], utc=True, errors="raise")
    boundary = PROTECTED_START.tz_localize("UTC")
    if bool(timestamps.ge(boundary).any()):
        raise ValueError("protected date 2025-08-23 or later materialised")


def deduplicate_hidden_events(events: pd.DataFrame) -> pd.DataFrame:
    """Select the latest source decision strictly before each frozen event identity."""

    identity = ["symbol", "session", "event_timestamp_utc", "family_id"]
    required = {*identity, "decision_ordinal", "decision_timestamp_utc"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"hidden-event columns missing: {missing}")
    frame = events.copy()
    frame["event_timestamp_utc"] = pd.to_datetime(
        frame["event_timestamp_utc"], utc=True, errors="raise"
    )
    frame["decision_timestamp_utc"] = pd.to_datetime(
        frame["decision_timestamp_utc"], utc=True, errors="raise"
    )
    eligibility_cutoff = frame["event_timestamp_utc"]
    if "event_available_timestamp_utc" in frame:
        frame["event_available_timestamp_utc"] = pd.to_datetime(
            frame["event_available_timestamp_utc"], utc=True, errors="raise"
        )
        eligibility_cutoff = frame["event_available_timestamp_utc"]
    eligible = frame.loc[frame["decision_timestamp_utc"].lt(eligibility_cutoff)].copy()
    if eligible.empty and not frame.empty:
        raise ValueError("no hidden event has a source decision strictly before completion")
    eligible = eligible.sort_values(
        [*identity, "decision_timestamp_utc", "decision_ordinal"], kind="mergesort"
    )
    result = eligible.drop_duplicates(identity, keep="last")
    if result.duplicated(identity).any():
        raise AssertionError("hidden-event identities are not unique after deduplication")
    return result.sort_values(
        ["session", "event_timestamp_utc", "symbol", "family_id"], kind="mergesort"
    ).reset_index(drop=True)


def score_event_horizons(
    bars: pd.DataFrame,
    *,
    completion_timestamp: pd.Timestamp,
    direction: int,
    horizons: tuple[int, ...] = (6, 12),
) -> pd.DataFrame:
    """Score fixed completed-bar horizons from the first exact post-completion open."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"market-bar columns missing: {missing}")
    if not horizons or any(int(value) <= 0 for value in horizons):
        raise ValueError("economic horizons must be positive")
    frame = bars.loc[:, sorted(required)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("market-bar timestamps are not unique")
    numeric = frame.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or bool((numeric <= 0.0).any()):
        raise ValueError("market bars contain non-positive or non-finite prices")
    completion = pd.Timestamp(completion_timestamp)
    completion = (
        completion.tz_localize("UTC") if completion.tzinfo is None else completion.tz_convert("UTC")
    )
    completion_locations = np.flatnonzero(frame["timestamp"].eq(completion).to_numpy())
    if len(completion_locations) != 1:
        raise ValueError("completion bar is unavailable or ambiguous")
    entry_index = int(completion_locations[0]) + 1
    if entry_index >= len(frame):
        raise ValueError("next-bar entry is unavailable")
    entry_timestamp = pd.Timestamp(frame.iloc[entry_index]["timestamp"])
    if entry_timestamp != completion + pd.Timedelta(minutes=5):
        raise ValueError("source gap before next-bar entry")
    entry_price = float(frame.iloc[entry_index]["open"])
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for horizon in horizons:
        exit_index = entry_index + int(horizon) - 1
        if exit_index >= len(frame):
            continue
        expected_exit_start = entry_timestamp + pd.Timedelta(minutes=5 * (int(horizon) - 1))
        exit_start = pd.Timestamp(frame.iloc[exit_index]["timestamp"])
        if exit_start != expected_exit_start:
            continue
        exit_price = float(frame.iloc[exit_index]["close"])
        raw_return = 10_000.0 * (exit_price / entry_price - 1.0)
        rows.append(
            {
                "horizon_bars": int(horizon),
                "entry_timestamp_utc": entry_timestamp,
                "entry_price": entry_price,
                "exit_timestamp_utc": exit_start + pd.Timedelta(minutes=5),
                "exit_bar_start_timestamp_utc": exit_start,
                "exit_price": exit_price,
                "raw_return_bps": raw_return,
                "signed_return_bps": float(direction) * raw_return,
            }
        )
    return pd.DataFrame(rows)


def opening_pressure_direction(signed_pressure: float) -> int | None:
    """Return the direction fixed at the source checkpoint, excluding zero/missing values."""

    value = float(signed_pressure)
    if not np.isfinite(value) or value == 0.0:
        return None
    return 1 if value > 0.0 else -1


def opposite_opening_pressure_direction(direction: int) -> int:
    """Return the fixed opposite-pressure control direction."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    return -int(direction)


def completion_momentum_direction(
    stock_return_bps: float, other_stock_returns_bps: list[float]
) -> int | None:
    """Sign the completion-known stock return after leave-one-stock-out demeaning."""

    values = np.asarray(other_stock_returns_bps, dtype=float)
    stock = float(stock_return_bps)
    if not np.isfinite(stock) or values.size == 0 or not np.isfinite(values).all():
        return None
    relative = stock - float(np.mean(values))
    if relative == 0.0:
        return None
    return 1 if relative > 0.0 else -1


def cohort_relative_signed_return_bps(
    *,
    stock_raw_return_bps: float,
    other_stock_raw_returns_bps: list[float],
    direction: int,
) -> float:
    """Return the signed stock move net of the leave-one-stock-out cohort mean."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    others = np.asarray(other_stock_raw_returns_bps, dtype=float)
    stock = float(stock_raw_return_bps)
    if others.size == 0 or not np.isfinite(others).all() or not np.isfinite(stock):
        raise ValueError("cohort-relative returns require finite stock and cohort values")
    return float(direction) * (stock - float(np.mean(others)))


def net_after_friction_bps(signed_return_bps: float, *, friction_bps: float) -> float:
    """Subtract fixed synthetic round-trip friction from a historical signed return."""

    value = float(signed_return_bps)
    friction = float(friction_bps)
    if not np.isfinite(value) or not np.isfinite(friction) or friction < 0.0:
        raise ValueError("return and non-negative friction must be finite")
    return value - friction


def eligible_matched_controls(
    candidates: pd.DataFrame,
    hidden_events: pd.DataFrame,
    *,
    focal_symbol: str,
    decision_timestamp: pd.Timestamp,
    completion_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Return causal same-slate controls with their own fixed pressure direction."""

    required = {"symbol", "signed_pressure", "entry_price", "exit_price"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"matched-control columns missing: {missing}")
    controls = candidates.loc[candidates["symbol"].astype(str).ne(str(focal_symbol))].copy()
    controls["direction"] = controls["signed_pressure"].map(opening_pressure_direction)
    entry = pd.to_numeric(controls["entry_price"], errors="coerce")
    exit_price = pd.to_numeric(controls["exit_price"], errors="coerce")
    controls = controls.loc[
        controls["direction"].notna()
        & entry.gt(0.0)
        & exit_price.gt(0.0)
        & np.isfinite(entry)
        & np.isfinite(exit_price)
    ].copy()
    if not hidden_events.empty:
        if {"symbol", "event_timestamp_utc"}.difference(hidden_events.columns):
            raise ValueError("hidden-event eligibility columns are missing")
        timestamp_column = (
            "event_available_timestamp_utc"
            if "event_available_timestamp_utc" in hidden_events
            else "event_timestamp_utc"
        )
        events = hidden_events.loc[:, ["symbol", timestamp_column]].copy()
        events = events.rename(columns={timestamp_column: "causal_completion_timestamp_utc"})
        events["causal_completion_timestamp_utc"] = pd.to_datetime(
            events["causal_completion_timestamp_utc"], utc=True, errors="raise"
        )
        decision = pd.Timestamp(decision_timestamp)
        decision = (
            decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
        )
        completion = pd.Timestamp(completion_timestamp)
        completion = (
            completion.tz_localize("UTC")
            if completion.tzinfo is None
            else completion.tz_convert("UTC")
        )
        if decision >= completion:
            raise ValueError("control decision must precede focal completion")
        prior = set(
            events.loc[
                events["causal_completion_timestamp_utc"].le(completion),
                "symbol",
            ].astype(str)
        )
        controls = controls.loc[~controls["symbol"].astype(str).isin(prior)].copy()
    controls["direction"] = controls["direction"].astype(int)
    return cast(
        pd.DataFrame,
        controls.sort_values("symbol", kind="mergesort").reset_index(drop=True),
    )


def registered_completion_targets(
    origin_bar_ordinal: int, completions: pd.DataFrame
) -> dict[str, bool | int | str | None]:
    """Resolve fixed six/twelve-bar registered-completion targets strictly after origin."""

    required = {"completion_bar_ordinal", "semantic_loop_id", "motif_type"}
    missing = sorted(required.difference(completions.columns))
    if missing:
        raise ValueError(f"registered-completion columns missing: {missing}")
    frame = completions.copy()
    frame["completion_bar_ordinal"] = pd.to_numeric(
        frame["completion_bar_ordinal"], errors="raise"
    ).astype(int)
    frame = frame.loc[frame["completion_bar_ordinal"].gt(int(origin_bar_ordinal))].copy()
    frame["bars_after_origin"] = frame["completion_bar_ordinal"] - int(origin_bar_ordinal)
    frame = frame.sort_values(
        ["bars_after_origin", "semantic_loop_id"], kind="mergesort"
    ).reset_index(drop=True)
    within_six = frame.loc[frame["bars_after_origin"].le(6)]
    within_twelve = frame.loc[frame["bars_after_origin"].le(12)]
    first = frame.iloc[0] if not frame.empty else None
    return {
        "registered_within_6_bars": bool(not within_six.empty),
        "registered_within_12_bars": bool(not within_twelve.empty),
        "bars_to_first_registered_completion": (
            int(first["bars_after_origin"]) if first is not None else None
        ),
        "first_registered_semantic_loop_id": (
            str(first["semantic_loop_id"]) if first is not None else None
        ),
        "first_registered_motif_type": str(first["motif_type"]) if first is not None else None,
    }


def registered_loop_bridge_target(decision_bar_ordinal: int, completions: pd.DataFrame) -> int:
    """Return whether any registered semantic loop completes in the next twelve bars."""

    targets = registered_completion_targets(decision_bar_ordinal, completions)
    return int(bool(targets["registered_within_12_bars"]))


def session_block_bootstrap_indices(
    frame: pd.DataFrame, *, draws: int, seed: int, session_column: str = "session"
) -> list[np.ndarray]:
    """Sample whole sessions with replacement and return positional row indices."""

    if draws <= 0 or session_column not in frame:
        raise ValueError("positive draws and a session column are required")
    sessions = np.asarray(sorted(frame[session_column].astype(str).unique()), dtype=object)
    if sessions.size == 0:
        raise ValueError("cannot bootstrap an empty session population")
    values = frame[session_column].astype(str).to_numpy()
    positions = {session: np.flatnonzero(values == session) for session in sessions}
    generator = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for _ in range(draws):
        sampled = generator.choice(sessions, size=len(sessions), replace=True)
        result.append(np.concatenate([positions[str(session)] for session in sampled]))
    return result


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=float)
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or bool(((values < 0) | (values > 1)).any())
    ):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    count = len(ranked)
    for index in range(count - 1, -1, -1):
        running = min(running, float(ranked[index]) * count / (index + 1))
        adjusted[index] = min(1.0, running)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return [float(value) for value in restored]


def permute_feature_within_slates(
    frame: pd.DataFrame,
    *,
    feature: str,
    seed: int,
    slate_column: str = "slate_id",
) -> pd.DataFrame:
    """Permute one bridge feature among stocks without crossing a session/checkpoint slate."""

    if feature not in frame or slate_column not in frame:
        raise ValueError("feature and slate columns are required")
    result = frame.copy()
    generator = np.random.default_rng(seed)
    permuted = result[feature].to_numpy(copy=True)
    for _, positions in result.groupby(slate_column, sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        permuted[index] = generator.permutation(permuted[index])
    result[feature] = permuted
    return result


def expanding_logistic_crossfit(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    folds: int = 4,
    warmup_fraction: float = 0.2,
    session_column: str = "session",
    weight_column: str = "row_weight",
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate causal expanding-window probabilities for chronological session blocks."""

    required = {session_column, weight_column, target, *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"crossfit columns missing: {missing}")
    if folds != 4 or not 0.0 < warmup_fraction < 1.0:
        raise ValueError("the frozen crossfit requires four folds and a valid warmup fraction")
    sessions = np.asarray(sorted(frame[session_column].astype(str).unique()), dtype=object)
    warmup_count = max(2, int(np.ceil(len(sessions) * warmup_fraction)))
    remaining = sessions[warmup_count:]
    blocks = [block for block in np.array_split(remaining, folds) if len(block)]
    if len(blocks) != folds:
        raise ValueError("insufficient chronological sessions for four prediction folds")
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    manifest_rows: list[dict[str, float | int | str]] = []
    session_values = frame[session_column].astype(str)
    for fold_index, block in enumerate(blocks, start=1):
        prediction_start = str(block[0])
        prediction_end = str(block[-1])
        train_mask = session_values.lt(prediction_start) & frame[target].notna()
        predict_mask = session_values.isin([str(value) for value in block])
        train = frame.loc[train_mask]
        predict = frame.loc[predict_mask]
        labels = train[target].to_numpy(dtype=int)
        if set(labels) != {0, 1} or predict.empty:
            raise ValueError("each expanding fold requires both training classes and predictions")
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        train_matrix = scaler.fit_transform(train.loc[:, list(features)].to_numpy(dtype=float))
        estimator = LogisticRegression(
            penalty="l2",
            C=0.25,
            solver="liblinear",
            max_iter=300,
            class_weight=None,
            n_jobs=1,
            random_state=20260721,
        )
        estimator.fit(
            train_matrix,
            labels,
            sample_weight=train[weight_column].to_numpy(dtype=float),
        )
        if bool(np.any(estimator.n_iter_ >= 300)):
            raise ValueError("expanding crossfit model did not converge")
        values = estimator.predict_proba(
            scaler.transform(predict.loc[:, list(features)].to_numpy(dtype=float))
        )[:, 1]
        predictions.loc[predict.index] = values
        manifest_rows.append(
            {
                "fold": fold_index,
                "train_session_start": str(train[session_column].min()),
                "train_session_end": str(train[session_column].max()),
                "prediction_session_start": prediction_start,
                "prediction_session_end": prediction_end,
                "train_rows": len(train),
                "prediction_rows": len(predict),
                "positive_train_rows": int(labels.sum()),
            }
        )
    return predictions, pd.DataFrame(manifest_rows)


def bridge_feature_sets(t0_features: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Freeze B0 to predecessor T0 and B1 to its single hidden-probability increment."""

    if not t0_features or "p_unregistered_within_6_bars" in t0_features:
        raise ValueError("T0 features must be non-empty and exclude the bridge increment")
    b0 = tuple(str(value) for value in t0_features)
    return b0, (*b0, "p_unregistered_within_6_bars")


def stock_clock_session_permutation(
    events: pd.DataFrame, eligible_sessions: pd.DataFrame, *, seed: int
) -> pd.DataFrame:
    """Permute hidden-event sessions within stock and frozen 30-minute clock bins."""

    event_required = {
        "symbol",
        "session",
        "clock_bin",
        "hidden_family_class",
        "completion_bar_ordinal",
    }
    eligible_required = {"symbol", "session"}
    if event_required.difference(events.columns) or eligible_required.difference(
        eligible_sessions.columns
    ):
        raise ValueError("stock-clock null inputs are incomplete")
    result = events.copy()
    generator = np.random.default_rng(seed)
    group_columns = ["symbol", "clock_bin"]
    if "period" in events and "period" in eligible_sessions:
        group_columns.append("period")
    permuted_sessions = result["session"].astype(object).to_numpy(copy=True)
    for key, positions in result.groupby(group_columns, sort=True).indices.items():
        values = key if isinstance(key, tuple) else (key,)
        pool = eligible_sessions.copy()
        for column, value in zip(group_columns, values, strict=True):
            if column == "clock_bin":
                continue
            pool = pool.loc[pool[column].astype(str).eq(str(value))]
        sessions = np.asarray(sorted(pool["session"].astype(str).unique()), dtype=object)
        index = np.asarray(positions, dtype=int)
        if len(sessions) < len(index):
            raise ValueError("stock-clock null has fewer eligible sessions than events")
        sampled = generator.permutation(sessions)[: len(index)]
        permuted_sessions[index] = sampled
    result["session"] = permuted_sessions
    return result


def choose_primary_decision(
    *,
    economic_status: str,
    registered_lead_status: str,
    predictive_bridge_status: str,
) -> str:
    """Map three independent part statuses to one preregistered primary decision."""

    allowed = {"supported", "descriptive_only", "not_supported", "insufficient_support"}
    statuses = (economic_status, registered_lead_status, predictive_bridge_status)
    if any(value not in allowed for value in statuses):
        raise ValueError("unknown independent status")
    if economic_status == "supported" and predictive_bridge_status == "supported":
        return "hidden_loops_economic_and_registered_bridge_supported"
    if economic_status == "supported":
        return "hidden_loop_economic_consequence_only"
    if predictive_bridge_status == "supported":
        return "hidden_loop_registered_bridge_only"
    if registered_lead_status == "supported":
        return "hidden_loop_structural_lead_only"
    if all(value == "insufficient_support" for value in statuses):
        return "blocked_support_failure"
    if "descriptive_only" in statuses:
        return "descriptive_hidden_loop_effects_only"
    return "no_hidden_loop_economic_or_bridge_increment"
