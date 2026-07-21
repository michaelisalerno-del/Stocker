"""Pure research primitives for One-Minute Activity-Price Lead Screen V0."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd

TimestampConvention = Literal["bar_start", "bar_end"]
OnsetLabel = Literal["UP_ONSET", "DOWN_ONSET", "NO_ONSET"]
DecisionCategory = Literal[
    "one_minute_activity_leads_onset_and_direction",
    "one_minute_activity_leads_onset_only",
    "one_minute_activity_adds_direction_only",
    "activity_price_response_interaction_only",
    "one_minute_price_sequence_only",
    "no_one_minute_activity_increment",
]
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
TIMESTAMP_CONVENTIONS: tuple[TimestampConvention, TimestampConvention] = (
    "bar_start",
    "bar_end",
)
FORBIDDEN_FEATURE_TOKENS = (
    "regime",
    "state",
    "loop",
    "closure",
    "excursion",
    "transition",
    "posterior",
    "structural_score",
    "profit_history",
    "future_volume",
    "future_range",
    "future_return",
    "symbol_identity",
    "month_identity",
    "news",
    "bid_ask",
    "order_book",
    "broker",
)


class ActivityPersistence(TypedDict):
    """Fixed elevated-activity persistence summaries."""

    above_one_count: int
    above_same_clock_p90_count: int
    longest_elevated_run: int


class ActivityPeakLead(TypedDict):
    """Fixed activity-versus-price peak timing summaries."""

    activity_peak_index: int
    price_peak_index: int
    price_peak_index_minus_activity_peak_index: int


class Efficiency(TypedDict):
    """Signed and absolute directional efficiency."""

    signed_efficiency: float
    absolute_efficiency: float


class ContinuationInteractions(TypedDict):
    """Fixed continuation-pressure interactions."""

    activity_continuation_3: float
    activity_continuation_5: float


class AbsorptionInteractions(TypedDict):
    """Fixed absorption-proxy interactions."""

    activity_absorption_3: float
    activity_absorption_wick: float


class ProgressPerActivity(TypedDict):
    """Signed and absolute progress divided by total activity."""

    signed_progress_per_activity_3: float
    absolute_progress_per_activity_3: float


@dataclass(frozen=True, slots=True)
class OutcomeWindow:
    """Fixed delayed entry, onset path, and terminal prices."""

    entry_minute_ordinal: int
    entry_open: float
    onset_minute_ordinals: tuple[int, int, int, int, int]
    fifteen_minute_terminal_ordinal: int
    fifteen_minute_terminal_close: float
    thirty_minute_terminal_ordinal: int
    thirty_minute_terminal_close: float


@dataclass(frozen=True)
class IncrementEvidence:
    """Fixed evidence fields for one ladder increment."""

    brier_improvement: float
    log_loss_improvement: float
    bootstrap_90_lower_brier: float
    bootstrap_90_lower_log_loss: float
    auc_not_reduced: bool
    positive_months: int
    neither_checkpoint_materially_adverse: bool
    exceeds_null_90th_percentile: bool
    concentration_passes: bool

    def passes(self, *, requires_null: bool, requires_concentration: bool) -> bool:
        """Apply the frozen predictive gate without economic-reference overrides."""

        return bool(
            self.brier_improvement > 0.0
            and self.log_loss_improvement > 0.0
            and self.bootstrap_90_lower_brier >= 0.0
            and self.bootstrap_90_lower_log_loss >= 0.0
            and self.auc_not_reduced
            and self.positive_months >= 5
            and self.neither_checkpoint_materially_adverse
            and (self.exceeds_null_90th_percentile or not requires_null)
            and (self.concentration_passes or not requires_concentration)
        )


def _aggregate_five_minutes(
    one_minute: pd.DataFrame, *, convention: TimestampConvention
) -> pd.DataFrame:
    timestamps = pd.to_datetime(one_minute["timestamp"], utc=True, errors="raise")
    starts = timestamps if convention == "bar_start" else timestamps - pd.Timedelta(minutes=1)
    working = one_minute.loc[:, list(OHLCV_COLUMNS)].copy()
    working["timestamp"] = starts.dt.floor("5min")
    return (
        working.groupby("timestamp", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )


def _aggregations_match(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    reference = expected.loc[:, ["timestamp", *OHLCV_COLUMNS]].copy()
    reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True, errors="raise")
    reference = reference.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    actual = actual.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if len(actual) != len(reference) or not actual["timestamp"].eq(reference["timestamp"]).all():
        return False
    left = actual.loc[:, OHLCV_COLUMNS].to_numpy(dtype=float)
    right = reference.loc[:, OHLCV_COLUMNS].to_numpy(dtype=float)
    return bool(np.isfinite(left).all() and np.allclose(left, right, rtol=0.0, atol=1e-9))


def prove_timestamp_convention(
    one_minute: pd.DataFrame, five_minute: pd.DataFrame
) -> TimestampConvention:
    """Prove a unique one-minute timestamp convention by exact 5-minute aggregation."""

    required = {"timestamp", *OHLCV_COLUMNS}
    for name, frame in (("one-minute", one_minute), ("five-minute", five_minute)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} aggregation columns missing: {missing}")
    matches = {
        convention: _aggregations_match(
            _aggregate_five_minutes(one_minute, convention=convention), five_minute
        )
        for convention in TIMESTAMP_CONVENTIONS
    }
    proved = [convention for convention, matched in matches.items() if matched]
    if len(proved) != 1:
        raise ValueError("one-minute timestamp convention is not uniquely proved")
    return proved[0]


def causal_ten_minute_window(
    bars: pd.DataFrame,
    decision_timestamp: pd.Timestamp,
    *,
    convention: TimestampConvention,
) -> pd.DataFrame:
    """Return the ten consecutive bars fully complete at a fixed decision timestamp."""

    if convention not in {"bar_start", "bar_end"}:
        raise ValueError("timestamp convention must be proved as bar_start or bar_end")
    if "timestamp" not in bars:
        raise ValueError("timestamp column missing")
    output = bars.copy()
    labels = pd.to_datetime(output["timestamp"], utc=True, errors="raise")
    starts = labels if convention == "bar_start" else labels - pd.Timedelta(minutes=1)
    completes = starts + pd.Timedelta(minutes=1)
    decision = pd.Timestamp(decision_timestamp)
    decision = (
        decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    )
    output["bar_start_timestamp"] = starts
    output["bar_complete_timestamp"] = completes
    output = output.loc[starts.lt(decision) & completes.le(decision)].copy()
    output = (
        output.sort_values("bar_start_timestamp", kind="mergesort").tail(10).reset_index(drop=True)
    )
    if len(output) != 10:
        raise ValueError("ten fully completed one-minute bars are unavailable")
    start_values = pd.to_datetime(output["bar_start_timestamp"], utc=True)
    if not start_values.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("causal one-minute window is not consecutive")
    local = start_values.dt.tz_convert("America/New_York")
    ordinals = local.dt.hour * 60 + local.dt.minute - 570
    if not ordinals.between(0, 389).all():
        raise ValueError("causal one-minute window is outside the regular session")
    output["minute_of_session_ordinal"] = ordinals.astype("int16")
    output["relative_minute"] = np.arange(-10, 0, dtype=np.int8)
    return output


def _finite_values(values: Sequence[float], *, expected: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or (expected is not None and len(array) != expected):
        raise ValueError("expected a fixed one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("sequence values must be finite")
    return array


def historical_activity_normalisation(
    frame: pd.DataFrame,
    *,
    development_end_exclusive: str = "2025-01-01",
    minimum_prior_observations: int = 20,
) -> pd.DataFrame:
    """Calculate causal same-stock/same-minute medians and freeze them after development."""

    required = {"symbol", "session", "minute_of_session_ordinal", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"activity normalisation columns missing: {missing}")
    if minimum_prior_observations != 20:
        raise ValueError("minimum prior observations is frozen at 20")
    output = frame.copy().reset_index(drop=True)
    output["_original_order"] = np.arange(len(output))
    output["_session_date"] = pd.to_datetime(output["session"], errors="raise")
    output["volume"] = pd.to_numeric(output["volume"], errors="raise")
    if (
        not np.isfinite(output["volume"].to_numpy(dtype=float)).all()
        or output["volume"].lt(0).any()
    ):
        raise ValueError("activity volume must be finite and non-negative")
    output = output.sort_values(
        ["symbol", "minute_of_session_ordinal", "_session_date", "_original_order"],
        kind="mergesort",
    )
    output["historical_median_volume"] = np.nan
    cutoff = pd.Timestamp(development_end_exclusive)
    for _, positions in output.groupby(
        ["symbol", "minute_of_session_ordinal"], sort=False
    ).groups.items():
        group = output.loc[positions].sort_values("_session_date", kind="mergesort")
        development = group.loc[group["_session_date"].lt(cutoff)]
        development_values = development["volume"].to_numpy(dtype=float)
        for offset, row_index in enumerate(development.index):
            if offset >= minimum_prior_observations:
                output.loc[row_index, "historical_median_volume"] = float(
                    np.median(development_values[:offset])
                )
        if len(development_values) >= minimum_prior_observations:
            frozen_median = float(np.median(development_values))
            assessment_index = group.index[group["_session_date"].ge(cutoff)]
            output.loc[assessment_index, "historical_median_volume"] = frozen_median
    baseline = output["historical_median_volume"].to_numpy(dtype=float)
    volume = output["volume"].to_numpy(dtype=float)
    relative = np.divide(
        volume,
        baseline,
        out=np.full(len(output), np.nan, dtype=float),
        where=np.isfinite(baseline) & (baseline > 0.0),
    )
    output["relative_activity"] = relative
    output["log_relative_activity"] = np.log1p(relative)
    return (
        output.sort_values("_original_order", kind="mergesort")
        .drop(columns=["_original_order", "_session_date"])
        .reset_index(drop=True)
    )


def activity_acceleration(log_relative_activity: Sequence[float]) -> float:
    """Return latest-two minus preceding-three mean log activity."""

    values = _finite_values(log_relative_activity, expected=5)
    return float(values[-2:].mean() - values[:3].mean())


def activity_slope(log_relative_activity: Sequence[float]) -> float:
    """Return the deterministic least-squares slope across the latest five minutes."""

    values = _finite_values(log_relative_activity, expected=5)
    x = np.arange(5, dtype=float)
    centered_x = x - x.mean()
    return float(np.dot(centered_x, values - values.mean()) / np.dot(centered_x, centered_x))


def activity_persistence(
    relative_activity: Sequence[float], *, same_clock_p90: Sequence[float]
) -> ActivityPersistence:
    """Count fixed elevated thresholds and the longest consecutive above-one run."""

    values = _finite_values(relative_activity, expected=5)
    thresholds = _finite_values(same_clock_p90, expected=5)
    elevated = values > 1.0
    longest = 0
    current = 0
    for flag in elevated:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return {
        "above_one_count": int(elevated.sum()),
        "above_same_clock_p90_count": int((values > thresholds).sum()),
        "longest_elevated_run": longest,
    }


def activity_peak_lead(
    relative_activity: Sequence[float], one_minute_returns: Sequence[float]
) -> ActivityPeakLead:
    """Return fixed peak indices and clipped price-minus-activity lead timing."""

    activity = _finite_values(relative_activity)
    returns = _finite_values(one_minute_returns)
    if len(activity) != 10 or len(returns) != 10:
        raise ValueError("activity peak lead requires the fixed ten-minute window")
    relative_indices = np.arange(-10, 0, dtype=int)
    activity_index = int(relative_indices[int(np.argmax(activity))])
    price_index = int(relative_indices[int(np.argmax(np.abs(returns)))])
    lead = int(np.clip(price_index - activity_index, -9, 9))
    return {
        "activity_peak_index": activity_index,
        "price_peak_index": price_index,
        "price_peak_index_minus_activity_peak_index": lead,
    }


def bar_sign_weighted_activity_proxy(
    one_minute_returns: Sequence[float], relative_activity: Sequence[float]
) -> float:
    """Return bar-sign-weighted activity; this is not signed trade flow."""

    returns = _finite_values(one_minute_returns)
    activity = _finite_values(relative_activity)
    if len(returns) != len(activity) or len(returns) not in {3, 5}:
        raise ValueError("signed activity proxy requires aligned three- or five-minute values")
    return float(np.dot(np.sign(returns), activity))


def directional_efficiency(one_minute_returns: Sequence[float]) -> Efficiency:
    """Return signed progress divided by absolute progress with a safe zero case."""

    returns = _finite_values(one_minute_returns)
    if len(returns) not in {3, 5}:
        raise ValueError("directional efficiency requires three or five returns")
    absolute_progress = float(np.abs(returns).sum())
    signed = 0.0 if absolute_progress == 0.0 else float(returns.sum() / absolute_progress)
    return {"signed_efficiency": signed, "absolute_efficiency": abs(signed)}


def activity_continuation_interactions(
    *,
    mean_relative_activity_3: float,
    signed_efficiency_3: float,
    mean_relative_activity_5: float,
    signed_efficiency_5: float,
) -> ContinuationInteractions:
    """Return exactly the fixed three- and five-minute continuation interactions."""

    values = _finite_values(
        [
            mean_relative_activity_3,
            signed_efficiency_3,
            mean_relative_activity_5,
            signed_efficiency_5,
        ]
    )
    return {
        "activity_continuation_3": float(values[0] * values[1]),
        "activity_continuation_5": float(values[2] * values[3]),
    }


def activity_absorption_interactions(
    *,
    mean_relative_activity_3: float,
    absolute_efficiency_3: float,
    absolute_wick_imbalance_3: float,
) -> AbsorptionInteractions:
    """Return exactly the fixed inefficiency and wick absorption interactions."""

    values = _finite_values(
        [mean_relative_activity_3, absolute_efficiency_3, absolute_wick_imbalance_3]
    )
    return {
        "activity_absorption_3": float(values[0] * (1.0 - values[1])),
        "activity_absorption_wick": float(values[0] * abs(values[2])),
    }


def progress_per_activity(
    cohort_relative_return_3: float,
    relative_activity_3: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> ProgressPerActivity:
    """Return fixed signed and absolute progress per total relative activity."""

    progress = _finite_values([cohort_relative_return_3])[0]
    activity = _finite_values(relative_activity_3, expected=3)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    denominator = max(float(activity.sum()), epsilon)
    return {
        "signed_progress_per_activity_3": float(progress / denominator),
        "absolute_progress_per_activity_3": float(abs(progress) / denominator),
    }


def activity_lead_price_response(
    *,
    early_activity: float,
    cumulative_return_last_2: float,
    cumulative_return_minutes_minus_5_through_minus_3: float,
) -> float:
    """Return the fixed early-activity times late-price-acceleration interaction."""

    values = _finite_values(
        [
            early_activity,
            cumulative_return_last_2,
            cumulative_return_minutes_minus_5_through_minus_3,
        ]
    )
    late_acceleration = abs(values[1]) - abs(values[2])
    return float(values[0] * late_acceleration)


def activity_range_response(
    *, activity_acceleration_value: float, range_acceleration: float
) -> float:
    """Return the fixed activity-acceleration times range-acceleration interaction."""

    values = _finite_values([activity_acceleration_value, range_acceleration])
    return float(values[0] * values[1])


def development_onset_barriers(paths: pd.DataFrame) -> dict[int, float]:
    """Return checkpoint-specific 2024 75th-percentile maximum absolute paths."""

    required = {
        "decision_id",
        "year",
        "decision_ordinal",
        "relative_minute",
        "cumulative_residual_return_bps",
    }
    missing = sorted(required.difference(paths.columns))
    if missing:
        raise ValueError(f"onset barrier columns missing: {missing}")
    development = paths.loc[paths["year"].eq(2024)].copy()
    if development.empty:
        raise ValueError("2024 development paths are unavailable")
    if not development["relative_minute"].astype(int).between(2, 6).all():
        raise ValueError("onset paths must use only minutes +2 through +6")
    values = pd.to_numeric(development["cumulative_residual_return_bps"], errors="raise").astype(
        float
    )
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("onset paths must be finite")
    development["absolute_path_bps"] = values.abs()
    maxima = (
        development.groupby(["decision_ordinal", "decision_id"], sort=True)["absolute_path_bps"]
        .max()
        .reset_index()
    )
    barriers = {
        int(str(checkpoint)): float(
            group["absolute_path_bps"].quantile(0.75, interpolation="linear")
        )
        for checkpoint, group in maxima.groupby("decision_ordinal", sort=True)
    }
    if set(barriers) != {6, 12}:
        raise ValueError("onset barriers require exactly checkpoints 6 and 12")
    return barriers


def classify_onset(
    cumulative_residual_return_bps: Sequence[float], *, barrier_bps: float
) -> OnsetLabel:
    """Classify the first completed-close barrier crossing over minutes +2 through +6."""

    path = _finite_values(cumulative_residual_return_bps)
    if len(path) < 1 or len(path) > 5 or not np.isfinite(barrier_bps) or barrier_bps <= 0.0:
        raise ValueError("onset classification requires a positive barrier and up to five closes")
    for value in path:
        if value >= barrier_bps:
            return "UP_ONSET"
        if value <= -barrier_bps:
            return "DOWN_ONSET"
    return "NO_ONSET"


def extract_outcome_window(bars: pd.DataFrame, decision_timestamp: pd.Timestamp) -> OutcomeWindow:
    """Return the fixed +2 entry, +2:+6 onset, +16, and +31 terminal window."""

    required = {"minute_of_session_ordinal", "bar_start_timestamp", "open", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"outcome window columns missing: {missing}")
    ordered = bars.sort_values("minute_of_session_ordinal", kind="mergesort").copy()
    ordinals = pd.to_numeric(ordered["minute_of_session_ordinal"], errors="raise").astype(int)
    if not ordinals.is_unique:
        raise ValueError("outcome window contains duplicate minute ordinals")
    starts = pd.to_datetime(ordered["bar_start_timestamp"], utc=True, errors="raise")
    decision = pd.Timestamp(decision_timestamp)
    decision = (
        decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    )
    local = decision.tz_convert("America/New_York")
    decision_start_ordinal = local.hour * 60 + local.minute - 570
    if decision_start_ordinal not in {30, 60}:
        raise ValueError("decision timestamp must be the fixed 10:00 or 10:30 checkpoint")
    decision_completed_ordinal = decision_start_ordinal - 1
    entry = decision_completed_ordinal + 2
    onset = tuple(range(entry, entry + 5))
    fifteen_terminal = decision_completed_ordinal + 16
    thirty_terminal = decision_completed_ordinal + 31
    by_ordinal = ordered.assign(_start=starts).set_index("minute_of_session_ordinal")
    required_ordinals = [entry, *onset, fifteen_terminal, thirty_terminal]
    if not set(required_ordinals).issubset(set(ordinals)):
        raise ValueError("fixed delayed outcome minutes are unavailable")
    entry_start = pd.Timestamp(cast(Any, by_ordinal.loc[entry, "_start"]))
    if entry_start != decision + pd.Timedelta(minutes=1):
        raise ValueError("entry is not the open of minute +2")
    return OutcomeWindow(
        entry_minute_ordinal=entry,
        entry_open=float(cast(Any, by_ordinal.loc[entry, "open"])),
        onset_minute_ordinals=(onset[0], onset[1], onset[2], onset[3], onset[4]),
        fifteen_minute_terminal_ordinal=fifteen_terminal,
        fifteen_minute_terminal_close=float(cast(Any, by_ordinal.loc[fifteen_terminal, "close"])),
        thirty_minute_terminal_ordinal=thirty_terminal,
        thirty_minute_terminal_close=float(cast(Any, by_ordinal.loc[thirty_terminal, "close"])),
    )


def cohort_relative_cumulative_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract the leave-one-stock-out cohort median at every path minute."""

    required = {"slate_id", "symbol", "relative_minute", "cumulative_return_bps"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cohort return columns missing: {missing}")
    output = frame.copy().reset_index(drop=True)
    output["cohort_median_cumulative_return_bps"] = np.nan
    for _, positions in output.groupby(["slate_id", "relative_minute"], sort=True).groups.items():
        group = output.loc[positions]
        if group["symbol"].nunique() < 2:
            raise ValueError("leave-one-stock-out cohort requires at least two stocks")
        for row_index, row in group.iterrows():
            peers = group.loc[group["symbol"].ne(row["symbol"]), "cumulative_return_bps"]
            output.loc[row_index, "cohort_median_cumulative_return_bps"] = float(
                np.median(peers.to_numpy(dtype=float))
            )
    output["cumulative_residual_return_bps"] = (
        pd.to_numeric(output["cumulative_return_bps"], errors="raise")
        - output["cohort_median_cumulative_return_bps"]
    )
    return output


def slate_row_weights(frame: pd.DataFrame, *, slate_column: str = "parent_slate_id") -> pd.Series:
    """Give every admitted slate total weight one."""

    if slate_column not in frame:
        raise ValueError(f"slate column missing: {slate_column}")
    counts = frame.groupby(slate_column, sort=False)[slate_column].transform("size").astype(float)
    if counts.le(0.0).any():
        raise ValueError("admitted slate count must be positive")
    return (1.0 / counts).rename("row_weight")


def manual_logistic_probability(
    features: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    means: Sequence[float],
    scales: Sequence[float],
    coefficients: Sequence[float],
    intercept: float,
) -> np.ndarray:
    """Reconstruct probabilities from frozen standardisation and logistic coefficients."""

    names = list(feature_names)
    missing = sorted(set(names).difference(features.columns))
    if missing:
        raise ValueError(f"manual logistic features missing: {missing}")
    mean_values = _finite_values(means)
    scale_values = _finite_values(scales)
    coefficient_values = _finite_values(coefficients)
    if not (len(names) == len(mean_values) == len(scale_values) == len(coefficient_values)):
        raise ValueError("manual logistic dimensions differ")
    if (scale_values <= 0.0).any() or not np.isfinite(intercept):
        raise ValueError("manual logistic scale or intercept is invalid")
    values = features.loc[:, names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("manual logistic features must be finite")
    standardized = (values - mean_values) / scale_values
    linear = float(intercept) + standardized @ coefficient_values
    clipped = np.clip(linear, -700.0, 700.0)
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-clipped)))


def session_block_bootstrap_draws(
    sessions: Sequence[str], *, draws: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Sample whole session identifiers with replacement using a fixed seed."""

    if draws < 1 or draws > 200:
        raise ValueError("session-block bootstrap draws must be between 1 and 200")
    unique = np.asarray(sorted({str(session) for session in sessions}), dtype=object)
    if len(unique) == 0:
        raise ValueError("session-block bootstrap requires sessions")
    generator = np.random.default_rng(seed)
    return tuple(
        tuple(str(value) for value in generator.choice(unique, size=len(unique), replace=True))
        for _ in range(draws)
    )


def permute_activity_bundle_within_slates(
    frame: pd.DataFrame,
    *,
    bundle_columns: Sequence[str],
    seed: int,
    slate_column: str = "parent_slate_id",
) -> pd.DataFrame:
    """Permute the complete activity bundle together within each fixed parent slate."""

    columns = list(bundle_columns)
    if not columns:
        raise ValueError("activity bundle must contain at least one field")
    missing = sorted({slate_column, *columns}.difference(frame.columns))
    if missing:
        raise ValueError(f"activity null columns missing: {missing}")
    output = frame.copy().reset_index(drop=True)
    generator = np.random.default_rng(seed)
    for _, positions in output.groupby(slate_column, sort=True).groups.items():
        indices = list(positions)
        if len(indices) < 2:
            continue
        source = output.loc[indices, columns].to_numpy(copy=True)
        permutation = generator.permutation(len(indices))
        output.loc[indices, columns] = source[permutation]
    return output


def assert_unprotected_timestamps(
    timestamps: Sequence[pd.Timestamp],
    *,
    protected_start: pd.Timestamp = pd.Timestamp("2025-08-23T00:00:00Z"),
) -> None:
    """Reject protected timestamps before any feature or outcome materialisation."""

    values = pd.to_datetime(list(timestamps), utc=True, errors="raise")
    boundary = pd.Timestamp(protected_start)
    boundary = (
        boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    )
    if (values >= boundary).any():
        raise ValueError("protected market timestamp")


def forbidden_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Return predictor names matching the frozen forbidden-family policy."""

    return sorted(
        name
        for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )


def assert_allowed_feature_names(feature_names: Sequence[str]) -> None:
    """Reject every forbidden predictor family by explicit name token."""

    violations = forbidden_feature_names(feature_names)
    if violations:
        raise ValueError(f"forbidden predictor fields: {violations}")


def decide_activity_screen(
    *,
    price_onset: bool = False,
    price_direction: bool = False,
    raw_activity_onset: bool = False,
    raw_activity_direction: bool = False,
    interaction_onset: bool = False,
    interaction_direction: bool = False,
) -> DecisionCategory:
    """Return the exact non-blocked category using fixed ladder precedence."""

    if raw_activity_onset and raw_activity_direction:
        return "one_minute_activity_leads_onset_and_direction"
    if raw_activity_onset:
        return "one_minute_activity_leads_onset_only"
    if raw_activity_direction:
        return "one_minute_activity_adds_direction_only"
    if interaction_onset or interaction_direction:
        return "activity_price_response_interaction_only"
    if price_onset or price_direction:
        return "one_minute_price_sequence_only"
    return "no_one_minute_activity_increment"
