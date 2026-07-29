"""Pure helpers for Frozen Causal M1C Low-Movement Screen V0.

The module contains retrospective underlying-stock research calculations only.
It has no broker, order, execution, portfolio, or deployment integration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

TAIL_QUANTILES: Final[dict[str, float]] = {
    "bottom_5_percent": 0.05,
    "bottom_10_percent": 0.10,
    "bottom_20_percent": 0.20,
}
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390
HORIZONS_MINUTES: Final[tuple[int, ...]] = (5, 10, 15, 30, 60)
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2026-01-01", tz="UTC")
EPSILON: Final[float] = 1e-12
CHECKPOINT_GROUP_ORDER: Final[dict[str, int]] = {"early": 0, "middle": 1, "late": 2}
OVERALL_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "m1c_low_movement_veto_supported_and_short_premium_recording_prioritised",
        "m1c_low_movement_veto_supported_short_premium_readiness_unproven",
        "m1c_bottom_tail_below_iv_descriptive_only",
        "m1c_low_movement_veto_not_supported",
        "blocked_m1c_reconstruction_failure",
        "blocked_insufficient_low_tail_support",
        "blocked_insufficient_fresh_quiet_episode_support",
        "blocked_previous_close_iv_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_reproducibility_or_audit_failure",
    }
)


def weighted_quantile(
    values: Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray,
    quantile: float,
) -> float:
    """Return the deterministic midpoint-CDF weighted quantile."""

    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if (
        data.ndim != 1
        or mass.ndim != 1
        or len(data) == 0
        or len(data) != len(mass)
        or not np.isfinite(data).all()
        or not np.isfinite(mass).all()
        or bool((mass <= 0.0).any())
        or not 0.0 <= quantile <= 1.0
    ):
        raise ValueError("weighted quantile requires finite values and positive aligned weights")
    order = np.argsort(data, kind="mergesort")
    sorted_values = data[order]
    sorted_weights = mass[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return float(
        np.interp(
            quantile,
            positions,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    )


def freeze_weighted_boundaries(
    values: Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray,
    *,
    quantiles: Sequence[float],
) -> tuple[float, ...]:
    """Fit an ordered set of deterministic weighted boundaries."""

    requested = tuple(float(value) for value in quantiles)
    if not requested or any(
        current <= 0.0 or current >= 1.0 or (index > 0 and current <= requested[index - 1])
        for index, current in enumerate(requested)
    ):
        raise ValueError("boundary quantiles must be strictly increasing inside zero and one")
    return tuple(weighted_quantile(values, weights, quantile) for quantile in requested)


def assign_frozen_bins(
    values: Sequence[float] | FloatArray,
    boundaries: Sequence[float],
) -> IntArray:
    """Apply fixed inclusive upper boundaries without assessment-period refitting."""

    data = np.asarray(values, dtype=np.float64)
    cuts = np.asarray(tuple(boundaries), dtype=np.float64)
    if (
        data.ndim != 1
        or cuts.ndim != 1
        or not np.isfinite(data).all()
        or not np.isfinite(cuts).all()
        or bool((np.diff(cuts) < 0.0).any())
    ):
        raise ValueError("frozen bin inputs must be finite and boundaries ordered")
    return np.asarray(np.searchsorted(cuts, data, side="left") + 1, dtype=np.int64)


def tail_memberships(
    probabilities: pd.Series,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    """Return inclusive memberships for each frozen numeric low-tail threshold."""

    values = pd.to_numeric(probabilities, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("tail probabilities must be finite")
    output: dict[str, npt.NDArray[np.bool_]] = {}
    for name, threshold in thresholds.items():
        boundary = float(threshold)
        if not np.isfinite(boundary):
            raise ValueError("tail thresholds must be finite")
        output[str(name)] = values <= boundary
    return pd.DataFrame(output, index=probabilities.index)


def validate_causal_features(
    feature_names: Sequence[str],
    *,
    forbidden: Sequence[str],
) -> tuple[str, ...]:
    """Reject every explicitly contaminated or peer-normalised feature."""

    features = tuple(str(value) for value in feature_names)
    if not features or len(features) != len(set(features)):
        raise ValueError("causal feature order must be non-empty and unique")
    contaminated = sorted(set(features).intersection(str(value) for value in forbidden))
    if contaminated:
        raise ValueError(f"contaminated features are excluded: {contaminated}")
    return features


def reconstruct_frozen_probabilities(
    frame: pd.DataFrame,
    specification: Mapping[str, object],
) -> FloatArray:
    """Manually apply a frozen serialized M0/M1C preprocessing and coefficient surface."""

    features = tuple(
        str(value) for value in cast(Sequence[object], specification.get("numeric_features", ()))
    )
    missing = sorted(set(features).difference(frame.columns))
    if not features or missing:
        raise ValueError(f"frozen probability features are invalid or missing: {missing}")
    medians = np.asarray(specification.get("numeric_medians"), dtype=np.float64)
    means = np.asarray(specification.get("numeric_means"), dtype=np.float64)
    scales = np.asarray(specification.get("numeric_scales"), dtype=np.float64)
    if (
        medians.shape != (len(features),)
        or means.shape != (len(features),)
        or scales.shape != (len(features),)
        or not np.isfinite(medians).all()
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or bool((scales <= 0.0).any())
    ):
        raise ValueError("frozen numeric preprocessing is invalid")
    raw = frame.loc[:, list(features)].to_numpy(float)
    values = np.where(np.isfinite(raw), raw, medians)
    parts: list[FloatArray] = [np.asarray((values - means) / scales, dtype=np.float64)]
    category_levels = cast(Mapping[str, Sequence[object]], specification.get("category_levels", {}))
    controls = tuple(
        str(value) for value in cast(Sequence[object], specification.get("category_controls", ()))
    )
    for control in controls:
        if control == "stock":
            column = "stock" if "stock" in frame else "symbol"
            if column not in frame:
                raise ValueError("frozen stock control is unavailable")
            observed = frame[column].astype(str).to_numpy()
        elif control == "checkpoint":
            if "checkpoint" not in frame:
                raise ValueError("frozen checkpoint control is unavailable")
            observed = frame["checkpoint"].astype(int).astype(str).to_numpy()
        elif control == "month_of_year":
            if "session" not in frame:
                raise ValueError("frozen month control is unavailable")
            observed = pd.to_datetime(frame["session"], errors="raise").dt.strftime("%m").to_numpy()
        elif control in frame:
            observed = frame[control].astype(str).to_numpy()
        else:
            raise ValueError(f"unsupported frozen category control: {control}")
        levels = tuple(str(value) for value in category_levels[control])
        for level in levels[1:]:
            parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
    design = np.concatenate(parts, axis=1)
    design_columns = tuple(
        str(value) for value in cast(Sequence[object], specification.get("design_columns", ()))
    )
    coefficients = np.asarray(specification.get("coefficients"), dtype=np.float64)
    if design.shape[1] != len(design_columns) or coefficients.shape != (design.shape[1],):
        raise ValueError("frozen model design order or coefficient width is invalid")
    intercept = float(cast(Any, specification.get("intercept")))
    if not math.isfinite(intercept):
        raise ValueError("frozen model intercept must be finite")
    linear = design @ coefficients + intercept
    kind = str(specification.get("kind"))
    if kind == "ridge":
        return np.asarray(linear, dtype=np.float64)
    if kind != "logistic":
        raise ValueError(f"unsupported frozen model kind: {kind}")
    return np.asarray(
        1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))),
        dtype=np.float64,
    )


def assert_unprotected_sessions(
    sessions: Sequence[object] | pd.Series,
    *,
    protected_start: pd.Timestamp = PROTECTED_START,
) -> None:
    """Reject protected sessions before constructing or inspecting outcomes."""

    parsed = pd.to_datetime(pd.Series(sessions), errors="raise", utc=True)
    boundary = pd.Timestamp(protected_start)
    boundary = (
        boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    )
    if bool(parsed.ge(boundary).any()):
        raise ValueError("protected sessions must not be read or materialised")


def iv_sigma(atm_iv: float, horizon_minutes: int) -> float:
    """Scale previous-close annualised ATM IV to one intraday horizon."""

    volatility = float(atm_iv)
    horizon = int(horizon_minutes)
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("ATM IV must be finite and positive")
    if horizon not in HORIZONS_MINUTES:
        raise ValueError(f"unsupported outcome horizon: {horizon}")
    return volatility * math.sqrt(horizon / ANNUAL_TRADING_MINUTES)


def iv_expected_absolute(atm_iv: float, horizon_minutes: int) -> float:
    """Return the Gaussian expected absolute move implied by previous-close IV."""

    return iv_sigma(atm_iv, horizon_minutes) * math.sqrt(2.0 / math.pi)


def _finite_positive(value: object, *, name: str) -> float:
    number = float(cast(Any, value))
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def calculate_checkpoint_outcomes(
    checkpoints: pd.DataFrame,
    five_minute_bars: pd.DataFrame,
    *,
    horizons: Sequence[int] = HORIZONS_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct terminal-movement and path-excursion outcomes from completed bars."""

    checkpoint_required = {
        "row_id",
        "stock",
        "session",
        "checkpoint",
        "feature_available_timestamp_utc",
        "atm_iv",
    }
    bar_required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
        "vti__bar_log_return",
    }
    missing_checkpoints = sorted(checkpoint_required.difference(checkpoints.columns))
    missing_bars = sorted(bar_required.difference(five_minute_bars.columns))
    if missing_checkpoints:
        raise ValueError(f"checkpoint outcomes missing columns: {missing_checkpoints}")
    if missing_bars:
        raise ValueError(f"five-minute bars missing columns: {missing_bars}")
    requested = tuple(int(value) for value in horizons)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(value not in HORIZONS_MINUTES for value in requested)
    ):
        raise ValueError("outcome horizons must be unique members of the frozen horizon set")
    assert_unprotected_sessions(checkpoints["session"])
    assert_unprotected_sessions(five_minute_bars["session"])
    if checkpoints.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("checkpoint identities must be unique")
    if five_minute_bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ValueError("bar identities must be unique")

    bars = five_minute_bars.copy()
    bars["bar_start_timestamp"] = pd.to_datetime(
        bars["bar_start_timestamp"], utc=True, errors="raise"
    )
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"], utc=True, errors="raise"
    )
    grouped = {
        (str(stock), str(session)): group.sort_values("bar_ordinal", kind="mergesort").set_index(
            "bar_ordinal", drop=False
        )
        for (stock, session), group in bars.groupby(["stock", "session"], sort=False)
    }

    movement_records: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    ordered = checkpoints.sort_values(["stock", "session", "checkpoint"], kind="mergesort")
    for checkpoint_row in ordered.itertuples(index=False):
        stock = str(checkpoint_row.stock)
        session = str(checkpoint_row.session)
        checkpoint = int(cast(Any, checkpoint_row.checkpoint))
        session_bars = grouped.get((stock, session))
        if session_bars is None or checkpoint not in session_bars.index:
            raise ValueError(f"entry bar is unavailable for {stock}|{session}|{checkpoint}")
        entry_bar = session_bars.loc[checkpoint]
        if isinstance(entry_bar, pd.DataFrame):
            raise ValueError("entry bar identity is not unique")
        entry_price = _finite_positive(entry_bar["open"], name="entry price")
        entry_timestamp = pd.Timestamp(entry_bar["bar_start_timestamp"])
        feature_timestamp = pd.to_datetime(
            cast(Any, checkpoint_row.feature_available_timestamp_utc),
            utc=True,
            errors="raise",
        )
        if entry_timestamp != feature_timestamp:
            raise ValueError(
                "entry must begin exactly when the checkpoint feature becomes available"
            )
        atm_iv = _finite_positive(checkpoint_row.atm_iv, name="previous-close ATM IV")
        identity = {
            "row_id": str(checkpoint_row.row_id),
            "stock": stock,
            "session": session,
            "checkpoint": checkpoint,
            "entry_timestamp": entry_timestamp,
            "entry_price": entry_price,
        }
        movement: dict[str, object] = dict(identity)
        path: dict[str, object] = dict(identity)
        available_horizons: list[str] = []
        for horizon in requested:
            bars_needed = horizon // 5
            ordinals = list(range(checkpoint, checkpoint + bars_needed))
            available = all(ordinal in session_bars.index for ordinal in ordinals)
            movement[f"available_{horizon}m"] = available
            path[f"available_{horizon}m"] = available
            if not available:
                for name in (
                    "terminal_close",
                    "signed_return",
                    "absolute_return",
                    "iv_sigma",
                    "iv_expected_absolute",
                    "terminal_iv_residual",
                    "terminal_iv_ratio",
                    "terminal_return_sign",
                    "movement_exceeds_iv",
                    "movement_remains_below_iv",
                ):
                    movement[f"{name}_{horizon}m"] = math.nan
                for name in (
                    "maximum_up_excursion",
                    "maximum_down_excursion",
                    "maximum_absolute_excursion",
                    "realised_path_range",
                    "time_to_maximum_up_excursion",
                    "time_to_maximum_down_excursion",
                    "time_to_maximum_absolute_excursion",
                    "excursion_sigma_ratio",
                    "path_range_sigma_ratio",
                    "crossed_above_and_below_entry",
                    "large_excursion_mean_reverted",
                    "market_maximum_absolute_movement",
                    "market_terminal_return",
                ):
                    path[f"{name}_{horizon}m"] = math.nan
                for label in ("1sigma", "1_5sigma", "2sigma"):
                    path[f"time_to_{label}_breach_{horizon}m"] = math.nan
                    path[f"breach_direction_{label}_{horizon}m"] = "none"
                    path[f"breach_mean_reverted_{label}_{horizon}m"] = False
                    path[f"breach_distance_{label}_{horizon}m"] = 0.0
                continue
            future = session_bars.loc[ordinals]
            terminal_close = _finite_positive(future.iloc[-1]["close"], name="terminal close")
            highs = pd.to_numeric(future["high"], errors="raise").to_numpy(float)
            lows = pd.to_numeric(future["low"], errors="raise").to_numpy(float)
            if (
                not np.isfinite(highs).all()
                or not np.isfinite(lows).all()
                or bool((highs <= 0.0).any())
                or bool((lows <= 0.0).any())
            ):
                raise ValueError("future high/low path must be finite and positive")
            signed_return = math.log(terminal_close / entry_price)
            absolute_return = abs(signed_return)
            sigma = iv_sigma(atm_iv, horizon)
            expected_absolute = iv_expected_absolute(atm_iv, horizon)
            upward = np.log(highs / entry_price)
            downward = np.log(lows / entry_price)
            up_index = int(np.argmax(upward))
            down_index = int(np.argmin(downward))
            maximum_up = float(upward[up_index])
            maximum_down = float(downward[down_index])
            if maximum_up >= abs(maximum_down):
                maximum_absolute = maximum_up
                absolute_index = up_index
            else:
                maximum_absolute = abs(maximum_down)
                absolute_index = down_index
            market_returns = pd.to_numeric(future["vti__bar_log_return"], errors="coerce").to_numpy(
                float
            )
            market_cumulative = (
                np.cumsum(market_returns)
                if np.isfinite(market_returns).all()
                else np.full(len(market_returns), np.nan)
            )
            excursion_ratio = maximum_absolute / (sigma + EPSILON)
            movement.update(
                {
                    f"terminal_close_{horizon}m": terminal_close,
                    f"signed_return_{horizon}m": signed_return,
                    f"absolute_return_{horizon}m": absolute_return,
                    f"iv_sigma_{horizon}m": sigma,
                    f"iv_expected_absolute_{horizon}m": expected_absolute,
                    f"terminal_iv_residual_{horizon}m": (absolute_return - expected_absolute),
                    f"terminal_iv_ratio_{horizon}m": (
                        absolute_return / (expected_absolute + EPSILON)
                    ),
                    f"terminal_return_sign_{horizon}m": int(np.sign(signed_return)),
                    f"movement_exceeds_iv_{horizon}m": bool(absolute_return > expected_absolute),
                    f"movement_remains_below_iv_{horizon}m": bool(
                        absolute_return <= expected_absolute
                    ),
                }
            )
            path.update(
                {
                    f"maximum_up_excursion_{horizon}m": maximum_up,
                    f"maximum_down_excursion_{horizon}m": maximum_down,
                    f"maximum_absolute_excursion_{horizon}m": maximum_absolute,
                    f"realised_path_range_{horizon}m": float(
                        math.log(float(np.max(highs)) / float(np.min(lows)))
                    ),
                    f"time_to_maximum_up_excursion_{horizon}m": (up_index + 1) * 5,
                    f"time_to_maximum_down_excursion_{horizon}m": (down_index + 1) * 5,
                    f"time_to_maximum_absolute_excursion_{horizon}m": (absolute_index + 1) * 5,
                    f"excursion_sigma_ratio_{horizon}m": excursion_ratio,
                    f"path_range_sigma_ratio_{horizon}m": (
                        math.log(float(np.max(highs)) / float(np.min(lows)))
                        / (2.0 * sigma + EPSILON)
                    ),
                    f"crossed_above_and_below_entry_{horizon}m": bool(
                        (highs > entry_price).any() and (lows < entry_price).any()
                    ),
                    f"large_excursion_mean_reverted_{horizon}m": bool(
                        excursion_ratio >= 1.5 and absolute_return <= 0.5 * maximum_absolute
                    ),
                    f"market_maximum_absolute_movement_{horizon}m": (
                        float(np.max(np.abs(market_cumulative)))
                        if np.isfinite(market_cumulative).all()
                        else math.nan
                    ),
                    f"market_terminal_return_{horizon}m": (
                        float(market_cumulative[-1])
                        if np.isfinite(market_cumulative).all()
                        else math.nan
                    ),
                }
            )
            for multiplier, label in (
                (1.0, "1sigma"),
                (1.5, "1_5sigma"),
                (2.0, "2sigma"),
            ):
                boundary = multiplier * sigma
                upward_breaches = upward > boundary
                downward_breaches = np.abs(downward) > boundary
                either_breach = upward_breaches | downward_breaches
                breach_indices = np.flatnonzero(either_breach)
                time_to_breach: float
                if len(breach_indices):
                    first_breach = int(breach_indices[0])
                    up_at_first = bool(upward_breaches[first_breach])
                    down_at_first = bool(downward_breaches[first_breach])
                    direction = (
                        "both"
                        if up_at_first and down_at_first
                        else ("up" if up_at_first else "down")
                    )
                    time_to_breach = float((first_breach + 1) * 5)
                else:
                    direction = "none"
                    time_to_breach = math.nan
                path[f"time_to_{label}_breach_{horizon}m"] = time_to_breach
                path[f"breach_direction_{label}_{horizon}m"] = direction
                path[f"breach_mean_reverted_{label}_{horizon}m"] = bool(
                    len(breach_indices) and absolute_return <= boundary
                )
                path[f"breach_distance_{label}_{horizon}m"] = max(
                    maximum_up - boundary,
                    abs(maximum_down) - boundary,
                    0.0,
                )
            available_horizons.append(str(horizon))
        movement["available_horizons"] = ",".join(available_horizons)
        path["available_horizons"] = ",".join(available_horizons)
        movement_records.append(movement)
        path_records.append(path)
    return pd.DataFrame(movement_records), pd.DataFrame(path_records)


def construct_fresh_quiet_episodes(
    checkpoint_rows: pd.DataFrame,
    *,
    threshold: float,
    probability_column: str = "m1c_probability",
    minimum_spacing_minutes: int = 30,
) -> pd.DataFrame:
    """Select fresh inclusive low-tail crossings with fixed thirty-minute spacing."""

    required = {
        "row_id",
        "stock",
        "session",
        "checkpoint",
        probability_column,
        "entry_timestamp",
        "available_horizons",
    }
    missing = sorted(required.difference(checkpoint_rows.columns))
    if missing:
        raise ValueError(f"quiet episode inputs missing columns: {missing}")
    if not math.isfinite(float(threshold)):
        raise ValueError("quiet-tail threshold must be finite")
    if minimum_spacing_minutes != 30:
        raise ValueError("fresh quiet episodes require thirty-minute spacing")
    assert_unprotected_sessions(checkpoint_rows["session"])
    ordered = checkpoint_rows.copy()
    ordered["entry_timestamp"] = pd.to_datetime(
        ordered["entry_timestamp"], utc=True, errors="raise"
    )
    ordered = ordered.sort_values(["stock", "session", "checkpoint"], kind="mergesort").reset_index(
        drop=True
    )
    if ordered.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("quiet episode checkpoint identities must be unique")
    probabilities = pd.to_numeric(ordered[probability_column], errors="raise").to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise ValueError("quiet episode probabilities must be finite")
    previous = ordered.groupby(["stock", "session"], sort=False)[probability_column].shift()
    ordered["_previous_probability"] = previous
    ordered["_in_low_tail"] = probabilities <= float(threshold)
    ordered["_fresh_crossing"] = ordered["_in_low_tail"] & (
        previous.isna() | previous.gt(float(threshold))
    )

    selected: list[int] = []
    episode_numbers: dict[int, int] = {}
    elapsed_minutes: dict[int, float] = {}
    for _, crossings in ordered.loc[ordered["_fresh_crossing"]].groupby(
        ["stock", "session"], sort=True
    ):
        previous_start: pd.Timestamp | None = None
        episode_number = 0
        for index, row in crossings.iterrows():
            current_start = pd.Timestamp(row["entry_timestamp"])
            elapsed = (
                math.nan
                if previous_start is None
                else (current_start - previous_start).total_seconds() / 60.0
            )
            if math.isfinite(elapsed) and elapsed < minimum_spacing_minutes:
                continue
            integer_index = int(cast(Any, index))
            selected.append(integer_index)
            episode_number += 1
            episode_numbers[integer_index] = episode_number
            elapsed_minutes[integer_index] = elapsed
            previous_start = current_start

    output = ordered.loc[selected].copy()
    output["episode_number"] = [episode_numbers[index] for index in selected]
    output["minutes_since_previous_quiet_episode"] = [elapsed_minutes[index] for index in selected]
    prefix = "m1c" if probability_column.lower().startswith("m1c") else "m0"
    output[f"previous_{prefix}_probability"] = output["_previous_probability"]
    output[f"current_{prefix}_probability"] = output[probability_column]
    output["quiet_tail_threshold"] = float(threshold)
    return output.drop(
        columns=["_previous_probability", "_in_low_tail", "_fresh_crossing"]
    ).reset_index(drop=True)


def tail_overlap(
    m1c_row_ids: Sequence[str],
    m0_row_ids: Sequence[str],
) -> dict[str, int | float]:
    """Summarise exact M1C/M0 low-tail row overlap."""

    m1c = {str(value) for value in m1c_row_ids}
    m0 = {str(value) for value in m0_row_ids}
    intersection = m1c.intersection(m0)
    union = m1c.union(m0)
    return {
        "intersection_rows": len(intersection),
        "union_rows": len(union),
        "jaccard_overlap": float(len(intersection) / len(union)) if union else math.nan,
        "m1c_only_rows": len(m1c.difference(m0)),
        "m0_only_rows": len(m0.difference(m1c)),
    }


def _month_ordinal(value: object) -> int:
    return int(cast(Any, pd.Period(str(value), freq="M").ordinal))


def _matched_cell(row: pd.Series, match_columns: Sequence[str]) -> tuple[object, ...]:
    return tuple(row[column] for column in match_columns)


def _fallback_distance(
    wanted: tuple[object, ...],
    candidate: tuple[object, ...],
) -> tuple[int, int, int, int]:
    wanted_period, wanted_stock, wanted_month, wanted_checkpoint, wanted_iv = wanted
    candidate_period, candidate_stock, candidate_month, candidate_checkpoint, candidate_iv = (
        candidate
    )
    if str(wanted_period) != str(candidate_period):
        return (10**9, 10**9, 10**9, 10**9)
    return (
        int(str(wanted_stock) != str(candidate_stock)),
        abs(_month_ordinal(wanted_month) - _month_ordinal(candidate_month)),
        abs(
            CHECKPOINT_GROUP_ORDER[str(wanted_checkpoint)]
            - CHECKPOINT_GROUP_ORDER[str(candidate_checkpoint)]
        ),
        abs(int(cast(Any, wanted_iv)) - int(cast(Any, candidate_iv))),
    )


def matched_random_selection(
    population: pd.DataFrame,
    real_tail: pd.DataFrame,
    *,
    seed: int,
    match_columns: Sequence[str] = (
        "period",
        "stock",
        "month",
        "checkpoint_group",
        "atm_iv_quartile",
    ),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select one unique non-tail matched row per real-tail row."""

    required = {"row_id", *match_columns}
    missing_population = sorted(required.difference(population.columns))
    missing_tail = sorted(required.difference(real_tail.columns))
    if missing_population:
        raise ValueError(f"matched population missing columns: {missing_population}")
    if missing_tail:
        raise ValueError(f"matched real tail missing columns: {missing_tail}")
    if population["row_id"].astype(str).duplicated().any():
        raise ValueError("matched population row identities must be unique")
    tail_ids = set(real_tail["row_id"].astype(str))
    if not tail_ids.issubset(set(population["row_id"].astype(str))):
        raise ValueError("real-tail rows must belong to the matched population")

    rng = np.random.default_rng(int(seed))
    candidates = population.loc[~population["row_id"].astype(str).isin(tail_ids)].copy()
    buckets: dict[tuple[object, ...], list[int]] = {}
    for index, row in candidates.iterrows():
        buckets.setdefault(_matched_cell(row, match_columns), []).append(int(cast(Any, index)))
    for values in buckets.values():
        rng.shuffle(values)

    selected_indices: list[int] = []
    matched_to: list[str] = []
    match_types: list[str] = []
    maximum_fallback_distance = 0
    exact_matches = 0
    fallback_matches = 0
    ordered_tail = real_tail.sort_values("row_id", kind="mergesort")
    for _, tail_row in ordered_tail.iterrows():
        wanted = _matched_cell(tail_row, match_columns)
        bucket = buckets.get(wanted, [])
        if bucket:
            selected_index = bucket.pop()
            exact_matches += 1
            match_type = "exact"
        else:
            available_keys = [key for key, values in buckets.items() if values]
            if not available_keys:
                raise ValueError("matched random selection exhausted unique non-tail candidates")
            ranked = sorted(
                (
                    _fallback_distance(wanted, key),
                    tuple(str(value) for value in key),
                    key,
                )
                for key in available_keys
            )
            distance, _, selected_key = ranked[0]
            if distance[0] >= 10**9:
                raise ValueError("matched random selection cannot cross period boundaries")
            selected_index = buckets[selected_key].pop()
            fallback_matches += 1
            maximum_fallback_distance = max(maximum_fallback_distance, sum(distance))
            match_type = "nearest_cell_fallback"
        selected_indices.append(selected_index)
        matched_to.append(str(tail_row["row_id"]))
        match_types.append(match_type)
    selected = population.loc[selected_indices].copy().reset_index(drop=True)
    selected["_matched_to_row_id"] = matched_to
    selected["_match_type"] = match_types
    return selected, {
        "selected_rows": len(selected),
        "exact_match_rows": exact_matches,
        "nearest_cell_fallback_rows": fallback_matches,
        "maximum_fallback_distance": maximum_fallback_distance,
    }


def permute_probabilities_within_sessions(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    seed: int,
) -> pd.Series:
    """Permute probabilities only within session × checkpoint-group slates."""

    required = {"session", "checkpoint_group", probability_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"probability permutation missing columns: {missing}")
    values = pd.to_numeric(frame[probability_column], errors="raise")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("probability permutation requires finite values")
    output = values.copy()
    rng = np.random.default_rng(int(seed))
    for labels in frame.groupby(["session", "checkpoint_group"], sort=True).groups.values():
        group_labels = list(labels)
        output.loc[group_labels] = rng.permutation(values.loc[group_labels].to_numpy(float))
    return output


def whole_session_bootstrap_plan(
    sessions: pd.Series,
    *,
    draws: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Freeze whole-session bootstrap identities without touching row contents."""

    unique = tuple(sorted(pd.Series(sessions).astype(str).unique()))
    if not unique or int(draws) <= 0:
        raise ValueError("whole-session bootstrap requires sessions and positive draws")
    rng = np.random.default_rng(int(seed))
    return tuple(
        tuple(str(value) for value in rng.choice(unique, size=len(unique), replace=True))
        for _ in range(int(draws))
    )


def evaluate_support_gate(
    frame: pd.DataFrame,
    *,
    period: str,
    population: str,
) -> dict[str, object]:
    """Apply the frozen checkpoint or fresh-episode support/concentration gate."""

    if period not in {"assessment", "stress"}:
        raise ValueError("support period must be assessment or stress")
    if population not in {"checkpoint", "fresh_episode"}:
        raise ValueError("support population must be checkpoint or fresh_episode")
    required = {"stock", "session"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"support frame missing columns: {missing}")
    months = frame["session"].astype(str).str[:7]
    expected_months = (
        {f"2025-{month:02d}" for month in range(1, 9)}
        if period == "assessment"
        else {f"2025-{month:02d}" for month in range(9, 13)}
    )
    rows = int(len(frame))
    stocks = int(frame["stock"].astype(str).nunique())
    sessions = int(frame["session"].astype(str).nunique())
    represented_months = set(months)
    maximum_stock_share = (
        float(frame.groupby("stock", sort=True).size().max() / rows) if rows else math.inf
    )
    maximum_month_share = float(months.value_counts().max() / rows) if rows else math.inf
    maximum_session_share = (
        float(frame.groupby("session", sort=True).size().max() / rows) if rows else math.inf
    )
    if population == "checkpoint" and period == "assessment":
        minimum_rows, minimum_sessions, minimum_stocks = 500, 60, 15
        stock_limit, month_limit, session_limit = 0.15, 0.25, 0.05
    elif population == "checkpoint":
        minimum_rows, minimum_sessions, minimum_stocks = 300, 45, 15
        stock_limit, month_limit, session_limit = 0.15, 0.35, 0.07
    else:
        minimum_rows, minimum_sessions, minimum_stocks = 100, 40, 12
        stock_limit, month_limit, session_limit = 0.20, 0.35, 0.08
    checks = {
        "minimum_rows": rows >= minimum_rows,
        "minimum_sessions": sessions >= minimum_sessions,
        "minimum_stocks": stocks >= minimum_stocks,
        "every_period_month_represented": expected_months.issubset(represented_months),
        "maximum_stock_share": maximum_stock_share <= stock_limit,
        "maximum_month_share": maximum_month_share <= month_limit,
        "maximum_session_share": maximum_session_share <= session_limit,
    }
    return {
        "period": period,
        "population": population,
        "rows": rows,
        "sessions": sessions,
        "stocks": stocks,
        "months": len(represented_months.intersection(expected_months)),
        "maximum_stock_share": maximum_stock_share,
        "maximum_month_share": maximum_month_share,
        "maximum_session_share": maximum_session_share,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _evidence_mapping(evidence: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = evidence.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"gate evidence missing mapping: {key}")
    return cast(Mapping[str, object], value)


def _number(evidence: Mapping[str, object], key: str) -> float:
    value = float(cast(Any, evidence.get(key)))
    if not math.isfinite(value):
        raise ValueError(f"gate evidence must be finite: {key}")
    return value


def _flag(evidence: Mapping[str, object], key: str) -> bool:
    value = evidence.get(key)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"gate evidence must be boolean: {key}")
    return bool(value)


def evaluate_low_movement_veto_gate(evidence: Mapping[str, object]) -> dict[str, object]:
    """Apply every binding long-premium veto requirement without relaxation."""

    checks: dict[str, bool] = {}
    for period in ("assessment", "stress"):
        values = _evidence_mapping(evidence, period)
        prefix = f"{period}_"
        checks[f"{prefix}remains_below_at_least_85_percent"] = (
            _number(values, "remains_below_iv_rate") >= 0.85
        )
        checks[f"{prefix}npv_lift_at_least_8_points"] = _number(values, "npv_lift") >= 0.08
        checks[f"{prefix}mean_residual_negative"] = _number(values, "mean_iv_residual") < 0.0
        checks[f"{prefix}median_residual_negative"] = _number(values, "median_iv_residual") < 0.0
        checks[f"{prefix}bootstrap_80_npv_lift_lower_above_zero"] = (
            _number(values, "bootstrap_80_npv_lift_lower") > 0.0
        )
        checks[f"{prefix}bootstrap_80_mean_residual_upper_below_zero"] = (
            _number(values, "bootstrap_80_mean_residual_upper") < 0.0
        )
        checks[f"{prefix}m1c_beats_m0_remains_below"] = _flag(values, "m1c_beats_m0_remains_below")
        checks[f"{prefix}m1c_beats_m0_mean_residual"] = _flag(values, "m1c_beats_m0_mean_residual")
        checks[f"{prefix}m1c_beats_m0_1_5_sigma_breach"] = _flag(
            values, "m1c_beats_m0_1_5_sigma_breach"
        )
        checks[f"{prefix}matched_npv_lift_wins"] = _number(values, "matched_npv_lift_wins") >= 18
        checks[f"{prefix}matched_mean_residual_wins"] = (
            _number(values, "matched_mean_residual_wins") >= 18
        )
        checks[f"{prefix}permutation_npv_lift_wins"] = (
            _number(values, "permutation_npv_lift_wins") >= 9
        )
        checks[f"{prefix}permutation_mean_residual_wins"] = (
            _number(values, "permutation_mean_residual_wins") >= 9
        )
        checks[f"{prefix}monthly_negative_residual_support"] = _number(
            values, "negative_residual_months"
        ) >= _number(values, "required_negative_residual_months")
        checks[f"{prefix}support_and_concentration"] = _flag(values, "support_passed")
        checks[f"{prefix}not_dependent_on_one_stock"] = _flag(values, "not_dependent_on_one_stock")
    checks["score_decile_direction_correct"] = _flag(evidence, "score_decile_direction_correct")
    checks["protected_boundary_passed"] = _flag(evidence, "protected_boundary_passed")
    checks["chronology_audit_passed"] = _flag(evidence, "chronology_audit_passed")
    return {"checks": checks, "passed": all(checks.values())}


def evaluate_short_premium_readiness_gate(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Apply the range-containment-only prospective recorder gate."""

    checks: dict[str, bool] = {
        "binding_low_movement_veto_passed": _flag(evidence, "veto_gate_passed"),
        "surprise_movers_not_concentrated": _flag(evidence, "surprise_movers_not_concentrated"),
        "thirty_minute_containment_favourable": _flag(
            evidence, "thirty_minute_containment_favourable"
        ),
    }
    for period in ("assessment", "stress"):
        values = _evidence_mapping(evidence, period)
        prefix = f"{period}_"
        checks[f"{prefix}fresh_1_5_sigma_lower_than_full"] = _flag(
            values, "fresh_1_5_sigma_lower_than_full"
        )
        checks[f"{prefix}fresh_1_5_sigma_lower_than_m0"] = _flag(
            values, "fresh_1_5_sigma_lower_than_m0"
        )
        checks[f"{prefix}fresh_2_sigma_lower_than_full"] = _flag(
            values, "fresh_2_sigma_lower_than_full"
        )
        checks[f"{prefix}bootstrap_80_1_5_sigma_difference_upper_below_zero"] = (
            _number(values, "bootstrap_80_1_5_sigma_difference_upper") < 0.0
        )
        checks[f"{prefix}two_sigma_containment_at_least_80_percent"] = (
            _number(values, "two_sigma_containment_rate") >= 0.80
        )
        checks[f"{prefix}support_passed"] = _flag(values, "support_passed")
    return {"checks": checks, "passed": all(checks.values())}


def choose_overall_decision(
    *,
    blocker: str | None,
    veto_supported: bool,
    readiness_supported: bool,
    descriptive_signal: bool,
) -> str:
    """Choose exactly one frozen overall decision."""

    if blocker is not None:
        if blocker not in OVERALL_DECISIONS or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown experiment blocker: {blocker}")
        return blocker
    if readiness_supported and not veto_supported:
        raise ValueError("short-premium readiness cannot pass without the veto gate")
    if veto_supported and readiness_supported:
        return "m1c_low_movement_veto_supported_and_short_premium_recording_prioritised"
    if veto_supported:
        return "m1c_low_movement_veto_supported_short_premium_readiness_unproven"
    if descriptive_signal:
        return "m1c_bottom_tail_below_iv_descriptive_only"
    return "m1c_low_movement_veto_not_supported"
