"""Fixed mechanics for M1C Asymmetric Downside Residual V1.

This module is retrospective-research only. It has no broker, account, order,
option-P&L, or execution interface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

M1C_HIGH_MOVEMENT_THRESHOLD: Final[float] = 0.488333710794033
PROTECTED_START: Final[str] = "2026-01-01"
PRIMARY_HORIZON_MINUTES: Final[int] = 15

OutcomeState = Literal[
    "UP_MOVE",
    "DOWN_MOVE",
    "NO_MOVE",
    "UP_FIRST",
    "DOWN_FIRST",
    "NO_BREACH",
    "AMBIGUOUS_BOTH_WITHIN_BAR",
]

DOWNSIDE_FEATURES: Final[tuple[str, ...]] = (
    "D1_signed_return_5m",
    "D2_signed_return_15m",
    "D3_close_location_15m",
    "D4_distance_from_session_vwap_iv",
)

OOF_FOLDS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("2024-01-01", "2024-03-31", "2024-04-01", "2024-06-30"),
    ("2024-01-01", "2024-06-30", "2024-07-01", "2024-09-30"),
    ("2024-01-01", "2024-09-30", "2024-10-01", "2024-12-31"),
)
MODEL_RANDOM_SEED: Final[int] = 2026072801


@dataclass(frozen=True)
class StandardisationParameters:
    """Development-only feature means and population scales."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the frozen parameters without imputation."""

        missing = sorted(set(self.feature_names).difference(frame.columns))
        if missing:
            raise ValueError(f"standardisation features missing: {missing}")
        values = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("standardisation requires complete finite features")
        return cast(np.ndarray, (values - self.means) / self.scales)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "means": dict(zip(self.feature_names, self.means.tolist(), strict=True)),
            "scales": dict(zip(self.feature_names, self.scales.tolist(), strict=True)),
            "fit_scope": "supplied_training_rows_only",
            "variance_definition": "population_ddof_0",
        }


@dataclass(frozen=True)
class FrozenDownsideModel:
    """One fixed L2 logistic model and its development-only scaler."""

    feature_names: tuple[str, ...]
    standardisation: StandardisationParameters
    coefficients: np.ndarray
    intercept: float
    iterations: int

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        scaled = self.standardisation.transform(frame)
        logits = scaled @ self.coefficients + self.intercept
        return cast(np.ndarray, 1.0 / (1.0 + np.exp(-logits)))

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": "sklearn.linear_model.LogisticRegression",
            "penalty": "l2",
            "l1_ratio": 0.0,
            "C": 1.0,
            "solver": "lbfgs",
            "fit_intercept": True,
            "class_weight": None,
            "max_iter": 1000,
            "random_state": MODEL_RANDOM_SEED,
            "feature_names": list(self.feature_names),
            "coefficients_standardised": dict(
                zip(self.feature_names, self.coefficients.tolist(), strict=True)
            ),
            "intercept": self.intercept,
            "iterations": self.iterations,
        }


def _fit_standardisation(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> StandardisationParameters:
    names = tuple(feature_names)
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"training features missing: {missing}")
    values = frame.loc[:, names].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("training requires complete finite features")
    means = np.mean(values, axis=0)
    raw_scales = np.std(values, axis=0, ddof=0)
    scales = np.where(raw_scales > 0.0, raw_scales, 1.0)
    return StandardisationParameters(names, means, scales)


def fit_downside_model(
    development: pd.DataFrame,
    *,
    target_column: str,
    feature_names: Sequence[str] = DOWNSIDE_FEATURES,
) -> FrozenDownsideModel:
    """Fit the only permitted model on unambiguous 2024 material movers."""

    if "session" not in development:
        raise ValueError("training requires session")
    assert_unprotected_sessions(development["session"])
    sessions = pd.to_datetime(development["session"], utc=True, errors="raise")
    if not bool(sessions.dt.year.eq(2024).all()):
        raise ValueError("downside model fitting is restricted to 2024 development rows")
    if target_column not in development:
        raise ValueError(f"training target missing: {target_column}")
    target = pd.to_numeric(development[target_column], errors="raise").to_numpy(int)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("training target must contain both binary classes")
    standardisation = _fit_standardisation(development, feature_names)
    scaled = standardisation.transform(development)
    estimator = LogisticRegression(
        l1_ratio=0.0,
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        max_iter=1000,
        random_state=MODEL_RANDOM_SEED,
    )
    estimator.fit(scaled, target)
    return FrozenDownsideModel(
        feature_names=tuple(feature_names),
        standardisation=standardisation,
        coefficients=estimator.coef_[0].astype(float, copy=True),
        intercept=float(estimator.intercept_[0]),
        iterations=int(estimator.n_iter_[0]),
    )


def expanding_time_ordered_oof(
    development: pd.DataFrame,
    *,
    target_column: str,
    feature_names: Sequence[str] = DOWNSIDE_FEATURES,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Generate the preregistered strictly forward 2024 OOF predictions."""

    if "session" not in development:
        raise ValueError("OOF generation requires session")
    assert_unprotected_sessions(development["session"])
    sessions = pd.to_datetime(development["session"], utc=True, errors="raise")
    if not bool(sessions.dt.year.eq(2024).all()):
        raise ValueError("OOF generation is restricted to 2024 development rows")
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold_number, boundaries in enumerate(OOF_FOLDS, start=1):
        train_start, train_end, predict_start, predict_end = (
            pd.Timestamp(boundary, tz="UTC") for boundary in boundaries
        )
        train_mask = sessions.between(train_start, train_end, inclusive="both")
        predict_mask = sessions.between(predict_start, predict_end, inclusive="both")
        train = development.loc[train_mask].copy()
        predict = development.loc[predict_mask].copy()
        if train.empty or predict.empty:
            raise ValueError(f"fold {fold_number} has empty train or prediction rows")
        model = fit_downside_model(
            train,
            target_column=target_column,
            feature_names=feature_names,
        )
        fold_predictions = predict.copy()
        fold_predictions["q_down_oof"] = model.predict_proba(predict)
        fold_predictions["fold"] = fold_number
        predictions.append(fold_predictions)
        latest_train_session = sessions.loc[train_mask].max()
        earliest_predict_session = sessions.loc[predict_mask].min()
        if latest_train_session >= earliest_predict_session:
            raise AssertionError("OOF chronology violated")
        audits.append(
            {
                "fold": fold_number,
                "train_start": train_start,
                "train_end": latest_train_session,
                "predict_start": earliest_predict_session,
                "predict_end": sessions.loc[predict_mask].max(),
                "train_rows": int(len(train)),
                "predict_rows": int(len(predict)),
                "standardisation": model.standardisation.as_dict(),
                "model": model.as_dict(),
            }
        )
    result = pd.concat(predictions, ignore_index=True)
    result = result.sort_values("session", kind="mergesort").reset_index(drop=True)
    return result, audits


def freeze_action_thresholds(q_down_oof: pd.Series) -> dict[str, float]:
    """Freeze 20th/80th cuts from OOF scores without consulting outcomes."""

    values = pd.to_numeric(q_down_oof, errors="raise").to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("action thresholds require complete finite OOF scores")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("OOF downside probabilities must be within [0, 1]")
    low, high = np.quantile(values, [0.2, 0.8], method="linear")
    if not float(low) < float(high):
        raise ValueError("OOF score distribution does not support distinct action thresholds")
    return {
        "low": float(low),
        "high": float(high),
        "low_quantile": 0.2,
        "high_quantile": 0.8,
    }


def apply_asymmetric_policy(
    q_down: pd.Series,
    *,
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    """Map complete scores to CALL/PUT; missing and middle scores abstain."""

    low = float(low_threshold)
    high = float(high_threshold)
    if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or high > 1.0 or low >= high:
        raise ValueError("action thresholds must satisfy 0 <= low < high <= 1")
    scores = pd.to_numeric(q_down, errors="coerce")
    actions = pd.Series("ABSTAIN", index=q_down.index, dtype="string")
    finite = scores.notna() & np.isfinite(scores)
    actions.loc[finite & scores.le(low)] = "CALL"
    actions.loc[finite & scores.ge(high)] = "PUT"
    return actions


def joint_probabilities(
    p_move: pd.Series,
    q_down: pd.Series,
    *,
    exact_target_compatibility: bool,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Construct joint probabilities only after an affirmative target audit."""

    if not exact_target_compatibility:
        raise ValueError("target mismatch prevents exact joint probabilities")
    move = pd.to_numeric(p_move, errors="raise").to_numpy(dtype=float)
    down = pd.to_numeric(q_down, errors="raise").to_numpy(dtype=float)
    if move.shape != down.shape or not np.isfinite(move).all() or not np.isfinite(down).all():
        raise ValueError("joint probabilities require aligned finite inputs")
    if bool(((move < 0.0) | (move > 1.0) | (down < 0.0) | (down > 1.0)).any()):
        raise ValueError("joint probability inputs must be within [0, 1]")
    output = pd.DataFrame(
        {
            "p_down_joint_v1": move * down,
            "p_up_joint_v1": move * (1.0 - down),
            "p_no_move_joint_v1": 1.0 - move,
        },
        index=p_move.index,
    )
    values = output.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or bool(((values < -tolerance) | (values > 1.0 + tolerance)).any())
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=tolerance)
    ):
        raise AssertionError("joint probability coherence check failed")
    return output.clip(0.0, 1.0)


def assert_unprotected_sessions(sessions: pd.Series) -> None:
    """Reject protected historical rows before target or feature materialisation."""

    parsed = pd.to_datetime(sessions, utc=True, errors="raise")
    boundary = pd.Timestamp(PROTECTED_START, tz="UTC")
    if bool(parsed.ge(boundary).any()):
        raise ValueError("protected 2026 historical sessions must not be materialised")


def partition_endpoint_return(
    signed_return: float,
    *,
    implied_movement: float,
) -> OutcomeState:
    """Apply the preregistered inclusive endpoint-direction partition."""

    observed = float(signed_return)
    threshold = float(implied_movement)
    if not math.isfinite(observed) or not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("endpoint partition requires a finite return and positive threshold")
    if observed >= threshold:
        return "UP_MOVE"
    if observed <= -threshold:
        return "DOWN_MOVE"
    return "NO_MOVE"


def partition_first_breach_ohlc(
    bars: pd.DataFrame,
    *,
    entry_price: float,
    implied_log_movement: float,
) -> OutcomeState:
    """Audit first breach from OHLC without assuming ordering inside one bar."""

    if not {"high", "low"}.issubset(bars.columns):
        raise ValueError("first-breach partition requires high and low")
    entry = float(entry_price)
    threshold = float(implied_log_movement)
    if not math.isfinite(entry) or entry <= 0.0 or not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("first-breach partition requires positive entry and threshold")
    highs = pd.to_numeric(bars["high"], errors="raise").to_numpy(float)
    lows = pd.to_numeric(bars["low"], errors="raise").to_numpy(float)
    if (
        not len(highs)
        or not np.isfinite(highs).all()
        or not np.isfinite(lows).all()
        or bool((highs <= 0.0).any())
        or bool((lows <= 0.0).any())
        or bool((highs < lows).any())
    ):
        raise ValueError("first-breach OHLC rows must be finite, positive, and ordered")
    upper = entry * math.exp(threshold)
    lower = entry * math.exp(-threshold)
    for high, low in zip(highs, lows, strict=True):
        up = bool(high >= upper)
        down = bool(low <= lower)
        if up and down:
            return "AMBIGUOUS_BOTH_WITHIN_BAR"
        if up:
            return "UP_FIRST"
        if down:
            return "DOWN_FIRST"
    return "NO_BREACH"


def build_downside_features(
    checkpoints: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Build exactly D1-D4 from one stock's completed same-session prefix."""

    checkpoint_required = {
        "stock",
        "session",
        "checkpoint",
        "feature_available_timestamp_utc",
        "implied_movement_15m_price",
    }
    bar_required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_complete_timestamp",
        "high",
        "low",
        "close",
        "volume",
    }
    missing_checkpoint = sorted(checkpoint_required.difference(checkpoints.columns))
    missing_bars = sorted(bar_required.difference(completed_bars.columns))
    if missing_checkpoint or missing_bars:
        raise ValueError(
            "downside feature inputs missing: "
            f"checkpoints={missing_checkpoint}, bars={missing_bars}"
        )
    if checkpoints.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("checkpoint identities must be unique")
    if completed_bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ValueError("completed-bar identities must be unique")
    assert_unprotected_sessions(checkpoints["session"])
    assert_unprotected_sessions(completed_bars["session"])

    bars = completed_bars.copy()
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"],
        utc=True,
        errors="raise",
    )
    bars = bars.sort_values(
        ["stock", "session", "bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = {
        (str(stock), str(session)): group
        for (stock, session), group in bars.groupby(["stock", "session"], sort=False)
    }
    records: list[dict[str, object]] = []
    for raw_checkpoint in checkpoints.itertuples(index=False):
        checkpoint_row = cast(Any, raw_checkpoint)
        stock = str(checkpoint_row.stock)
        session = str(checkpoint_row.session)
        checkpoint = int(checkpoint_row.checkpoint)
        group = grouped.get((stock, session))
        if group is None:
            prefix = pd.DataFrame(columns=bars.columns)
        else:
            prefix = group.loc[group["bar_ordinal"].astype(int).lt(checkpoint)].sort_values(
                "bar_ordinal",
                kind="mergesort",
            )
        expected_ordinals = np.arange(checkpoint, dtype=int)
        observed_ordinals = (
            prefix["bar_ordinal"].to_numpy(int) if len(prefix) else np.asarray([], dtype=int)
        )
        complete_prefix = bool(
            checkpoint >= 4 and np.array_equal(observed_ordinals, expected_ordinals)
        )
        feature_timestamp = pd.to_datetime(
            checkpoint_row.feature_available_timestamp_utc,
            utc=True,
            errors="raise",
        )
        timestamp_matches = bool(
            complete_prefix
            and pd.Timestamp(prefix.iloc[-1]["bar_complete_timestamp"]) == feature_timestamp
        )
        if not complete_prefix or not timestamp_matches:
            records.append(
                {
                    **{name: math.nan for name in DOWNSIDE_FEATURES},
                    "maximum_predictor_bar_ordinal": (
                        int(observed_ordinals.max()) if len(observed_ordinals) else None
                    ),
                    "maximum_predictor_timestamp": (
                        pd.Timestamp(prefix["bar_complete_timestamp"].max())
                        if len(prefix)
                        else None
                    ),
                    "D4_causal_volume_available": False,
                    "downside_features_complete": False,
                    "downside_feature_missing_reason": (
                        "incomplete_same_session_prefix"
                        if not complete_prefix
                        else "feature_timestamp_not_latest_completed_bar"
                    ),
                }
            )
            continue

        highs = pd.to_numeric(prefix["high"], errors="coerce").to_numpy(float)
        lows = pd.to_numeric(prefix["low"], errors="coerce").to_numpy(float)
        closes = pd.to_numeric(prefix["close"], errors="coerce").to_numpy(float)
        volumes = pd.to_numeric(prefix["volume"], errors="coerce").to_numpy(float)
        price_valid = bool(
            np.isfinite(highs).all()
            and np.isfinite(lows).all()
            and np.isfinite(closes).all()
            and (highs > 0.0).all()
            and (lows > 0.0).all()
            and (closes > 0.0).all()
            and (highs >= lows).all()
        )
        if not price_valid:
            records.append(
                {
                    **{name: math.nan for name in DOWNSIDE_FEATURES},
                    "maximum_predictor_bar_ordinal": checkpoint - 1,
                    "maximum_predictor_timestamp": feature_timestamp,
                    "D4_causal_volume_available": False,
                    "downside_features_complete": False,
                    "downside_feature_missing_reason": "invalid_same_session_prices",
                }
            )
            continue

        d1 = math.log(closes[-1] / closes[-2])
        d2 = math.log(closes[-1] / closes[-4])
        trailing_high = float(np.max(highs[-3:]))
        trailing_low = float(np.min(lows[-3:]))
        trailing_range = trailing_high - trailing_low
        d3 = (
            2.0 * ((float(closes[-1]) - trailing_low) / trailing_range) - 1.0
            if trailing_range > 0.0
            else math.nan
        )
        usable_volume = np.where(np.isfinite(volumes) & (volumes > 0.0), volumes, 0.0)
        volume_available = bool(float(np.sum(usable_volume)) > 0.0)
        if volume_available:
            typical_price = (highs + lows + closes) / 3.0
            vwap = float(np.sum(typical_price * usable_volume) / np.sum(usable_volume))
        else:
            vwap = math.nan
        implied_price_movement = float(checkpoint_row.implied_movement_15m_price)
        denominator_available = bool(
            math.isfinite(implied_price_movement) and implied_price_movement > 0.0
        )
        d4 = (
            (float(closes[-1]) - vwap) / implied_price_movement
            if volume_available and denominator_available
            else math.nan
        )
        values = np.asarray([d1, d2, d3, d4], dtype=float)
        complete = bool(np.isfinite(values).all())
        missing_reason: str | None = None
        if not math.isfinite(d3):
            missing_reason = "zero_trailing_15m_range"
        elif not volume_available:
            missing_reason = "causal_stock_volume_unavailable"
        elif not denominator_available:
            missing_reason = "previous_close_implied_price_movement_unavailable"
        records.append(
            {
                "D1_signed_return_5m": d1,
                "D2_signed_return_15m": d2,
                "D3_close_location_15m": d3,
                "D4_distance_from_session_vwap_iv": d4,
                "maximum_predictor_bar_ordinal": checkpoint - 1,
                "maximum_predictor_timestamp": feature_timestamp,
                "D4_causal_volume_available": volume_available,
                "downside_features_complete": complete,
                "downside_feature_missing_reason": missing_reason,
            }
        )
    return pd.concat(
        [checkpoints.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )


__all__ = [
    "DOWNSIDE_FEATURES",
    "FrozenDownsideModel",
    "M1C_HIGH_MOVEMENT_THRESHOLD",
    "MODEL_RANDOM_SEED",
    "OOF_FOLDS",
    "PRIMARY_HORIZON_MINUTES",
    "PROTECTED_START",
    "OutcomeState",
    "StandardisationParameters",
    "apply_asymmetric_policy",
    "assert_unprotected_sessions",
    "build_downside_features",
    "expanding_time_ordered_oof",
    "fit_downside_model",
    "freeze_action_thresholds",
    "joint_probabilities",
    "partition_endpoint_return",
    "partition_first_breach_ohlc",
]
