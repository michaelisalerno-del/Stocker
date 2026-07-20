"""Causal fixed-model regime-pair closure history diagnostic V1.

This module contains structural research utilities only.  It has no price
target, payoff, broker, order, position, or production-runtime surface.
Numeric state identities are meaningful only inside the bound fitted model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, cast

import numpy as np
import pandas as pd

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
ECONOMIC_OUTCOMES_USED = False
PAYOFF_SELECTION_USED = False
PRODUCTION_RUNTIME_MODIFIED = False
STRATEGY_PROMOTION = False
LIVE_ORDERING_ENABLED = False
PROTECTED_2026_OPENED = False
PROMOTABLE = False

STATE_SENTINEL = 8
PRIMARY_BASELINE = "M2_IMMEDIATE_PAIR"
PRIMARY_HISTORY_MODEL = "M5_LAST_FIVE_STATES"

MODEL_CONTEXT_LEVELS: dict[str, tuple[tuple[str, ...], ...]] = {
    "M0_GLOBAL": (),
    "M1_CURRENT_STATE": (("current_state",),),
    "M2_IMMEDIATE_PAIR": (
        ("current_state",),
        ("previous_state_1", "current_state"),
    ),
    "M3_LAST_THREE_STATES": (
        ("current_state",),
        ("previous_state_1", "current_state"),
        ("previous_state_2", "previous_state_1", "current_state"),
    ),
    "M4_LAST_FOUR_STATES": (
        ("current_state",),
        ("previous_state_1", "current_state"),
        ("previous_state_2", "previous_state_1", "current_state"),
        (
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
        ),
    ),
    "M5_LAST_FIVE_STATES": (
        ("current_state",),
        ("previous_state_1", "current_state"),
        ("previous_state_2", "previous_state_1", "current_state"),
        (
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
        ),
    ),
    "M6_PAIR_PLUS_COMPLETED_DURATIONS": (
        ("current_state",),
        ("previous_state_1", "current_state"),
        ("previous_state_1", "current_state", "previous_duration_1_bucket"),
        (
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
            "previous_duration_2_bucket",
        ),
    ),
    "M7_PAIR_PLUS_PREVIOUS_CLOSURE_RECENCY": (
        ("current_state",),
        ("previous_state_1", "current_state"),
        ("previous_state_1", "current_state", "previous_loop_pair"),
        (
            "previous_state_1",
            "current_state",
            "previous_loop_pair",
            "previous_loop_recency_bucket",
        ),
    ),
    "M8_FULL_CAUSAL_CONTEXT": (
        ("current_state",),
        ("previous_state_1", "current_state"),
        ("previous_state_2", "previous_state_1", "current_state"),
        (
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
            "previous_duration_2_bucket",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
            "previous_duration_2_bucket",
            "previous_loop_pair",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
            "previous_duration_2_bucket",
            "previous_loop_pair",
            "previous_loop_recency_bucket",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
            "previous_duration_2_bucket",
            "previous_loop_pair",
            "previous_loop_recency_bucket",
            "posterior_margin_bucket",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
            "previous_duration_2_bucket",
            "previous_loop_pair",
            "previous_loop_recency_bucket",
            "posterior_margin_bucket",
            "departure_probability_bucket",
        ),
        (
            "previous_state_4",
            "previous_state_3",
            "previous_state_2",
            "previous_state_1",
            "current_state",
            "previous_duration_1_bucket",
            "previous_duration_2_bucket",
            "previous_loop_pair",
            "previous_loop_recency_bucket",
            "posterior_margin_bucket",
            "departure_probability_bucket",
            "clock_phase",
        ),
    ),
}

_REQUIRED_PANEL_COLUMNS = {
    "symbol",
    "session",
    "segment_id",
    "segment_end_reason",
    "bar_ordinal",
    "bar_start_timestamp",
    "bar_complete_timestamp",
    "clock_phase",
    "source_artifact",
    "source_hash",
    "data_snapshot_hash",
}


def safety_flags() -> dict[str, object]:
    """Return the immutable research-only safety boundary."""

    return {
        "research_only": RESEARCH_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "order_placement": ORDER_PLACEMENT,
        "broker_connected": BROKER_CONNECTED,
        "economic_outcomes_used": ECONOMIC_OUTCOMES_USED,
        "payoff_selection_used": PAYOFF_SELECTION_USED,
        "production_runtime_modified": PRODUCTION_RUNTIME_MODIFIED,
        "strategy_promotion": STRATEGY_PROMOTION,
        "live_ordering_enabled": LIVE_ORDERING_ENABLED,
        "protected_2026_opened": PROTECTED_2026_OPENED,
        "promotable": PROMOTABLE,
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    """Hash canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple | set | frozenset):
        return list(value)
    if pd.isna(value):
        return None
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _stable_id(*parts: object) -> str:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _duration_bucket(value: int) -> int:
    if value <= 0:
        return -1
    if value == 1:
        return 0
    if value == 2:
        return 1
    if value <= 4:
        return 2
    if value <= 8:
        return 3
    if value <= 16:
        return 4
    if value <= 32:
        return 5
    return 6


def _probability_bucket(value: float, cuts: tuple[float, ...]) -> int:
    return int(np.searchsorted(np.asarray(cuts, dtype=float), float(value), side="right"))


def _loop_pair(left: int, right: int) -> str:
    lower, upper = sorted((int(left), int(right)))
    return f"loop_p_{lower}-{upper}-{lower}"


def _top_two_margin(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("state probabilities require at least two states")
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


def build_pair_closure_population(
    panel: pd.DataFrame,
    *,
    semantic_states: np.ndarray,
    state_probabilities: np.ndarray,
    posterior_entropy: np.ndarray,
    departure_probability: np.ndarray,
    representation: str,
) -> pd.DataFrame:
    """Build one causal decision at every observed state-run start.

    Population membership is determined at the first completed bar assigned to
    the current run.  Terminal decisions remain present with an unavailable
    structural target, so future availability cannot alter the decision set.
    """

    missing = sorted(_REQUIRED_PANEL_COLUMNS.difference(panel.columns))
    if missing:
        raise ValueError(f"panel lacks pair-closure columns: {missing}")
    if not panel.index.equals(pd.RangeIndex(len(panel))):
        raise ValueError("panel must use a zero-based contiguous index")
    states = np.asarray(semantic_states, dtype=int)
    probabilities = np.asarray(state_probabilities, dtype=float)
    entropy = np.asarray(posterior_entropy, dtype=float)
    departure = np.asarray(departure_probability, dtype=float)
    if states.shape != (len(panel),):
        raise ValueError("state vector differs from panel length")
    if probabilities.shape != (len(panel), 8):
        raise ValueError("state-probability matrix must be rows by eight states")
    if entropy.shape != (len(panel),) or departure.shape != (len(panel),):
        raise ValueError("posterior diagnostics differ from panel length")
    if states.min(initial=0) < 0 or states.max(initial=0) >= 8:
        raise ValueError("semantic state lies outside [0, 7]")
    if not np.isfinite(probabilities).all() or not np.isfinite(entropy).all():
        raise ValueError("posterior diagnostics contain non-finite values")

    rows: list[dict[str, object]] = []
    for segment_id, group in panel.groupby("segment_id", sort=False):
        positions = group.index.to_numpy(dtype=int)
        local_states = states[positions]
        if len(positions) == 0:
            continue
        run_starts = np.r_[0, np.flatnonzero(local_states[1:] != local_states[:-1]) + 1]
        run_ends = np.r_[run_starts[1:] - 1, len(local_states) - 1]
        run_states = local_states[run_starts]
        run_durations = run_ends - run_starts + 1
        last_completed_closure: int | None = None
        terminal_reason = str(group.iloc[-1]["segment_end_reason"])
        for run_index in range(1, len(run_starts)):
            completed_run_index = run_index - 1
            if completed_run_index >= 2 and int(run_states[completed_run_index]) == int(
                run_states[completed_run_index - 2]
            ):
                last_completed_closure = completed_run_index

            start_position = int(positions[int(run_starts[run_index])])
            source = panel.iloc[start_position]
            current_state = int(run_states[run_index])
            history_states = [
                int(run_states[run_index - offset]) if run_index - offset >= 0 else STATE_SENTINEL
                for offset in range(1, 5)
            ]
            history_durations = [
                int(run_durations[run_index - offset]) if run_index - offset >= 0 else 0
                for offset in range(1, 5)
            ]
            previous_state = history_states[0]
            if previous_state == current_state:
                raise AssertionError("compressed runs contain a self transition")

            if last_completed_closure is None:
                previous_loop_pair = "NO_PREVIOUS_CLOSURE"
                previous_loop_recency = -1
            else:
                closure_left = int(run_states[last_completed_closure - 2])
                closure_right = int(run_states[last_completed_closure - 1])
                previous_loop_pair = _loop_pair(closure_left, closure_right)
                previous_loop_recency = run_index - last_completed_closure

            has_next_state = run_index + 1 < len(run_starts)
            next_state = int(run_states[run_index + 1]) if has_next_state else STATE_SENTINEL
            if has_next_state:
                target_available = True
                censor_reason = "AVAILABLE"
                target = int(next_state == previous_state)
                target_label = "IMMEDIATE_PAIR_CLOSURE" if target else "NEXT_STATE_DIFFERS"
                event_position = int(positions[int(run_starts[run_index + 1])])
                target_available_timestamp: object = panel.iloc[event_position][
                    "bar_complete_timestamp"
                ]
            else:
                target_available = False
                target = -1
                target_label = "TARGET_UNAVAILABLE"
                censor_reason = (
                    "RIGHT_CENSORED_SESSION_END"
                    if terminal_reason == "scheduled_session_end"
                    else "UNAVAILABLE_STRUCTURAL_GAP"
                )
                target_available_timestamp = pd.NaT

            margin = _top_two_margin(probabilities[start_position])
            pair_id = _loop_pair(previous_state, current_state)
            orientation = f"{previous_state}->{current_state}->{previous_state}"
            decision_id = _stable_id(
                "pair_closure_v1",
                representation,
                source["symbol"],
                source["session"],
                segment_id,
                source["bar_ordinal"],
            )
            rows.append(
                {
                    "decision_id": decision_id,
                    "representation": representation,
                    "symbol": str(source["symbol"]),
                    "session": str(source["session"]),
                    "segment_id": str(segment_id),
                    "bar_ordinal": int(source["bar_ordinal"]),
                    "bar_start_timestamp": source["bar_start_timestamp"],
                    "decision_timestamp": source["bar_complete_timestamp"],
                    "target_available_timestamp": target_available_timestamp,
                    "current_state": current_state,
                    "previous_state_1": history_states[0],
                    "previous_state_2": history_states[1],
                    "previous_state_3": history_states[2],
                    "previous_state_4": history_states[3],
                    "previous_duration_1": history_durations[0],
                    "previous_duration_2": history_durations[1],
                    "previous_duration_3": history_durations[2],
                    "previous_duration_4": history_durations[3],
                    "previous_duration_1_bucket": _duration_bucket(history_durations[0]),
                    "previous_duration_2_bucket": _duration_bucket(history_durations[1]),
                    "previous_loop_pair": previous_loop_pair,
                    "previous_loop_recency_runs": previous_loop_recency,
                    "previous_loop_recency_bucket": _duration_bucket(previous_loop_recency),
                    "loop_pair_id": pair_id,
                    "loop_orientation": orientation,
                    "posterior_entropy": float(entropy[start_position]),
                    "posterior_top_two_margin": margin,
                    "posterior_margin_bucket": _probability_bucket(margin, (0.10, 0.25, 0.50)),
                    "departure_probability": float(departure[start_position]),
                    "departure_probability_bucket": _probability_bucket(
                        departure[start_position], (0.05, 0.15, 0.35)
                    ),
                    "clock_phase": str(source["clock_phase"]),
                    "target_available": target_available,
                    "target_pair_closure": target,
                    "target_label": target_label,
                    "next_state": next_state,
                    "censor_reason": censor_reason,
                    "source_provider": "EODHD",
                    "source_artifact": "/".join(
                        PurePath(str(source["source_artifact"])).parts[-3:]
                    ),
                    "source_hash": str(source["source_hash"]),
                    "data_snapshot_hash": str(source["data_snapshot_hash"]),
                    "volume_meaning": "provider_reported_historical_volume_activity",
                    **safety_flags(),
                }
            )
    output = pd.DataFrame.from_records(rows)
    if output.empty:
        raise ValueError("pair-closure population is empty")
    output["decision_timestamp"] = pd.to_datetime(output["decision_timestamp"], utc=True)
    output["target_available_timestamp"] = pd.to_datetime(
        output["target_available_timestamp"], utc=True
    )
    output = output.sort_values(
        ["representation", "symbol", "session", "decision_timestamp", "decision_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if output["decision_id"].duplicated().any():
        raise AssertionError("pair-closure decision IDs are not unique")
    available = output["target_available"].astype(bool)
    if (
        output.loc[available, "target_available_timestamp"]
        <= output.loc[available, "decision_timestamp"]
    ).any():
        raise AssertionError("structural target is not strictly after its decision")
    expected_target = output.loc[available, "next_state"].eq(
        output.loc[available, "previous_state_1"]
    )
    if not np.array_equal(
        expected_target.to_numpy(dtype=int),
        output.loc[available, "target_pair_closure"].to_numpy(dtype=int),
    ):
        raise AssertionError("pair-closure target differs from A-to-B-to-A semantics")
    return output


def _normalize_key(value: object, width: int) -> tuple[object, ...]:
    if width == 1 and not isinstance(value, tuple):
        return (value,)
    if isinstance(value, tuple):
        return value
    raise TypeError("multi-column context key is not a tuple")


@dataclass(frozen=True, slots=True)
class HierarchicalBinaryFrequencyModel:
    """Deterministic beta-binomial context model with parent shrinkage."""

    levels: tuple[tuple[str, ...], ...]
    global_probability: float
    tables: tuple[dict[tuple[object, ...], float], ...]
    tau: float
    alpha: float
    beta: float

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        levels: Sequence[Sequence[str]],
        target: str = "target_pair_closure",
        tau: float = 64.0,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> HierarchicalBinaryFrequencyModel:
        if frame.empty:
            raise ValueError("frequency model requires training rows")
        if tau <= 0.0 or alpha <= 0.0 or beta <= 0.0:
            raise ValueError("frequency-model priors must be positive")
        normalized = tuple(tuple(str(value) for value in level) for level in levels)
        required = {target, *(value for level in normalized for value in level)}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"frequency-model frame lacks columns: {missing}")
        previous: tuple[str, ...] = ()
        for level in normalized:
            if not level or len(set(level)) != len(level):
                raise ValueError("context levels must contain unique fields")
            if not set(previous).issubset(level):
                raise ValueError("context levels must be nested")
            previous = level
        truth = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=int)
        if not np.isin(truth, (0, 1)).all():
            raise ValueError("binary target must contain only zero and one")
        global_probability = float((truth.sum() + alpha) / (len(truth) + alpha + beta))
        tables: list[dict[tuple[object, ...], float]] = []
        parent_keys: tuple[str, ...] = ()
        parent_table: Mapping[tuple[object, ...], float] = {}
        for keys in normalized:
            grouped = (
                frame.assign(_target=truth)
                .groupby(list(keys), sort=True, dropna=False)["_target"]
                .agg(["sum", "count"])
            )
            table: dict[tuple[object, ...], float] = {}
            parent_positions = tuple(keys.index(value) for value in parent_keys)
            for raw_key, row in grouped.iterrows():
                key = _normalize_key(raw_key, len(keys))
                if parent_keys:
                    parent_key = tuple(key[index] for index in parent_positions)
                    parent_probability = parent_table.get(parent_key, global_probability)
                else:
                    parent_probability = global_probability
                table[key] = float(
                    (float(row["sum"]) + tau * parent_probability) / (float(row["count"]) + tau)
                )
            tables.append(table)
            parent_keys = keys
            parent_table = table
        return cls(
            levels=normalized,
            global_probability=global_probability,
            tables=tuple(tables),
            tau=float(tau),
            alpha=float(alpha),
            beta=float(beta),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict with exact parent fallback for unseen contexts."""

        missing = sorted(
            {value for level in self.levels for value in level}.difference(frame.columns)
        )
        if missing:
            raise ValueError(f"prediction frame lacks columns: {missing}")
        probabilities = np.full(len(frame), self.global_probability, dtype=float)
        for keys, table in zip(self.levels, self.tables, strict=True):
            for row_index, raw_key in enumerate(
                frame.loc[:, list(keys)].itertuples(index=False, name=None)
            ):
                key = tuple(raw_key)
                if key in table:
                    probabilities[row_index] = table[key]
        return np.asarray(np.clip(probabilities, 1e-9, 1.0 - 1e-9), dtype=float)


def expanding_month_predictions(
    population: pd.DataFrame,
    *,
    model_names: Sequence[str] = tuple(MODEL_CONTEXT_LEVELS),
    minimum_train_months: int = 3,
) -> pd.DataFrame:
    """Create expanding calendar-month OOF predictions within one year."""

    frame = population.loc[population["target_available"].astype(bool)].copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    years = sorted(frame["decision_timestamp"].dt.year.unique())
    if len(years) != 1:
        raise ValueError("expanding folds require exactly one calendar year")
    frame["score_month"] = frame["decision_timestamp"].dt.strftime("%Y-%m")
    months = sorted(frame["score_month"].unique())
    if len(months) <= minimum_train_months:
        raise ValueError("insufficient months for expanding folds")
    names = tuple(str(value) for value in model_names)
    unknown = sorted(set(names).difference(MODEL_CONTEXT_LEVELS))
    if unknown:
        raise ValueError(f"unknown model names: {unknown}")
    records: list[pd.DataFrame] = []
    for score_month in months[minimum_train_months:]:
        train = frame.loc[frame["score_month"].lt(score_month)]
        score = frame.loc[frame["score_month"].eq(score_month)].copy()
        if train.empty or score.empty:
            continue
        for name in names:
            model = HierarchicalBinaryFrequencyModel.fit(
                train,
                levels=MODEL_CONTEXT_LEVELS[name],
            )
            part = score[
                [
                    "decision_id",
                    "representation",
                    "symbol",
                    "session",
                    "decision_timestamp",
                    "target_pair_closure",
                    "score_month",
                ]
            ].copy()
            part["evaluation_period"] = "DEVELOPMENT_2024_OOF"
            part["model"] = name
            part["probability"] = model.predict(score)
            part["training_rows"] = len(train)
            records.append(part)
    if not records:
        raise ValueError("expanding folds produced no predictions")
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["representation", "decision_id", "model"], kind="mergesort")
        .reset_index(drop=True)
    )


def frozen_assessment_predictions(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    model_names: Sequence[str] = tuple(MODEL_CONTEXT_LEVELS),
) -> pd.DataFrame:
    """Fit on all development rows and score unchanged assessment rows."""

    train = development.loc[development["target_available"].astype(bool)].copy()
    score = assessment.loc[assessment["target_available"].astype(bool)].copy()
    if train.empty or score.empty:
        raise ValueError("assessment prediction requires train and score rows")
    if pd.to_datetime(train["decision_timestamp"], utc=True).dt.year.max() >= 2025:
        raise ValueError("assessment fitting admitted a post-2024 row")
    if set(pd.to_datetime(score["decision_timestamp"], utc=True).dt.year.unique()) != {2025}:
        raise ValueError("assessment scoring must contain only 2025 rows")
    records: list[pd.DataFrame] = []
    for name in tuple(str(value) for value in model_names):
        model = HierarchicalBinaryFrequencyModel.fit(
            train,
            levels=MODEL_CONTEXT_LEVELS[name],
        )
        part = score[
            [
                "decision_id",
                "representation",
                "symbol",
                "session",
                "decision_timestamp",
                "target_pair_closure",
            ]
        ].copy()
        part["score_month"] = pd.to_datetime(part["decision_timestamp"], utc=True).dt.strftime(
            "%Y-%m"
        )
        part["evaluation_period"] = "ASSESSMENT_2025"
        part["model"] = name
        part["probability"] = model.predict(score)
        part["training_rows"] = len(train)
        records.append(part)
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["representation", "decision_id", "model"], kind="mergesort")
        .reset_index(drop=True)
    )


def binary_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    """Return deterministic binary logarithmic loss."""

    y = np.asarray(truth, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    if y.shape != p.shape or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("log-loss inputs are incompatible")
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))


def brier_score(truth: np.ndarray, probability: np.ndarray) -> float:
    """Return mean squared probability error."""

    y = np.asarray(truth, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape:
        raise ValueError("Brier inputs are incompatible")
    return float(np.mean(np.square(y - p)))


def roc_auc(truth: np.ndarray, probability: np.ndarray) -> float:
    """Compute tie-aware binary ROC AUC from average ranks."""

    y = np.asarray(truth, dtype=int)
    p = np.asarray(probability, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def calibration_error(truth: np.ndarray, probability: np.ndarray, *, bins: int = 10) -> float:
    """Return equal-width expected calibration error."""

    y = np.asarray(truth, dtype=float)
    p = np.asarray(probability, dtype=float)
    indices = np.minimum((np.clip(p, 0.0, 1.0) * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def top_decile_lift(
    truth: np.ndarray, probability: np.ndarray, decision_ids: Sequence[str]
) -> float:
    """Return closure-rate lift in the deterministic top probability decile."""

    y = np.asarray(truth, dtype=float)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0:
        return math.nan
    count = max(1, int(math.ceil(0.10 * len(y))))
    order = np.lexsort((np.asarray(decision_ids, dtype=str), -p))
    baseline = float(y.mean())
    return float(y[order[:count]].mean() - baseline)


def prediction_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize each representation/evaluation/model on common rows."""

    rows: list[dict[str, object]] = []
    grouping = ["representation", "evaluation_period", "model"]
    for keys, group in predictions.groupby(grouping, sort=True):
        truth = group["target_pair_closure"].to_numpy(dtype=int)
        probability = group["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "representation": keys[0],
                "evaluation_period": keys[1],
                "model": keys[2],
                "rows": len(group),
                "closure_rate": float(truth.mean()),
                "log_loss": binary_log_loss(truth, probability),
                "brier_score": brier_score(truth, probability),
                "roc_auc": roc_auc(truth, probability),
                "calibration_error": calibration_error(truth, probability),
                "top_decile_lift": top_decile_lift(
                    truth,
                    probability,
                    group["decision_id"].astype(str).tolist(),
                ),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(grouping).reset_index(drop=True)


def _paired_prediction_frame(
    predictions: pd.DataFrame,
    *,
    baseline: str = PRIMARY_BASELINE,
    candidate: str = PRIMARY_HISTORY_MODEL,
) -> pd.DataFrame:
    subset = predictions.loc[predictions["model"].isin((baseline, candidate))].copy()
    key_columns = [
        "decision_id",
        "representation",
        "evaluation_period",
        "symbol",
        "session",
        "decision_timestamp",
        "target_pair_closure",
    ]
    wide = subset.pivot(index=key_columns, columns="model", values="probability").reset_index()
    if baseline not in wide or candidate not in wide:
        raise ValueError("paired prediction models are missing")
    if wide[[baseline, candidate]].isna().any().any():
        raise ValueError("paired prediction rows are incomplete")
    truth = wide["target_pair_closure"].to_numpy(dtype=float)
    for model in (baseline, candidate):
        probability = np.clip(wide[model].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        wide[f"{model}_row_log_loss"] = -(
            truth * np.log(probability) + (1.0 - truth) * np.log1p(-probability)
        )
        wide[f"{model}_row_brier"] = np.square(truth - probability)
    wide["log_loss_improvement"] = (
        wide[f"{baseline}_row_log_loss"] - wide[f"{candidate}_row_log_loss"]
    )
    wide["brier_improvement"] = wide[f"{baseline}_row_brier"] - wide[f"{candidate}_row_brier"]
    return wide


def paired_session_bootstrap(
    predictions: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 20_260_720,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap complete session dates with baseline/candidate rows paired."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    records: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for (representation, evaluation_period), source in predictions.groupby(
        ["representation", "evaluation_period"], sort=True
    ):
        paired = _paired_prediction_frame(source)
        session = paired.groupby("session", sort=True)[
            ["log_loss_improvement", "brier_improvement"]
        ].agg(["sum", "count"])
        log_sum = session[("log_loss_improvement", "sum")].to_numpy(dtype=float)
        log_count = session[("log_loss_improvement", "count")].to_numpy(dtype=float)
        brier_sum = session[("brier_improvement", "sum")].to_numpy(dtype=float)
        brier_count = session[("brier_improvement", "count")].to_numpy(dtype=float)
        rng = np.random.default_rng(
            seed + int(hashlib.sha256(str(representation).encode()).hexdigest()[:8], 16) % 10_000
        )
        sampled = rng.integers(0, len(session), size=(draws, len(session)))
        log_values = log_sum[sampled].sum(axis=1) / log_count[sampled].sum(axis=1)
        brier_values = brier_sum[sampled].sum(axis=1) / brier_count[sampled].sum(axis=1)
        draw_frame = pd.DataFrame(
            {
                "representation": representation,
                "evaluation_period": evaluation_period,
                "draw": np.arange(draws, dtype=int),
                "log_loss_improvement": log_values,
                "brier_improvement": brier_values,
            }
        )
        records.append(draw_frame)
        summaries.append(
            {
                "representation": representation,
                "evaluation_period": evaluation_period,
                "rows": len(paired),
                "sessions": paired["session"].nunique(),
                "log_loss_improvement": float(paired["log_loss_improvement"].mean()),
                "log_loss_ci_low": float(np.quantile(log_values, 0.025)),
                "log_loss_ci_high": float(np.quantile(log_values, 0.975)),
                "brier_improvement": float(paired["brier_improvement"].mean()),
                "brier_ci_low": float(np.quantile(brier_values, 0.025)),
                "brier_ci_high": float(np.quantile(brier_values, 0.975)),
                "draws": draws,
                "seed": seed,
            }
        )
    return (
        pd.concat(records, ignore_index=True).sort_values(
            ["representation", "evaluation_period", "draw"]
        ),
        pd.DataFrame.from_records(summaries).sort_values(["representation", "evaluation_period"]),
    )


def quarter_and_stock_deletion_metrics(
    assessment_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired calendar-quarter and leave-one-stock-out comparisons."""

    quarter_rows: list[dict[str, object]] = []
    deletion_rows: list[dict[str, object]] = []
    for representation, source in assessment_predictions.groupby("representation", sort=True):
        paired = _paired_prediction_frame(source)
        timestamps = pd.to_datetime(paired["decision_timestamp"], utc=True)
        paired["quarter"] = (
            timestamps.dt.year.astype(str)
            + "Q"
            + (((timestamps.dt.month - 1) // 3) + 1).astype(str)
        )
        for quarter, group in paired.groupby("quarter", sort=True):
            quarter_rows.append(
                {
                    "representation": representation,
                    "quarter": quarter,
                    "rows": len(group),
                    "log_loss_improvement": float(group["log_loss_improvement"].mean()),
                    "brier_improvement": float(group["brier_improvement"].mean()),
                    "positive_log_loss_direction": bool(group["log_loss_improvement"].mean() > 0.0),
                }
            )
        for symbol in sorted(paired["symbol"].astype(str).unique()):
            group = paired.loc[paired["symbol"].ne(symbol)]
            deletion_rows.append(
                {
                    "representation": representation,
                    "removed_symbol": symbol,
                    "rows": len(group),
                    "log_loss_improvement": float(group["log_loss_improvement"].mean()),
                    "brier_improvement": float(group["brier_improvement"].mean()),
                    "positive_log_loss_direction": bool(group["log_loss_improvement"].mean() > 0.0),
                }
            )
    return (
        pd.DataFrame.from_records(quarter_rows).sort_values(["representation", "quarter"]),
        pd.DataFrame.from_records(deletion_rows).sort_values(["representation", "removed_symbol"]),
    )


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted q-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be one-dimensional values in [0, 1]")
    count = len(values)
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * count / np.arange(1, count + 1, dtype=float)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty(count, dtype=float)
    output[order] = np.minimum(ranked, 1.0)
    return output


def pair_orientation_metrics(
    population: pd.DataFrame,
    *,
    period: str,
    draws: int = 2_000,
    seed: int = 20_260_720,
    minimum_rows: int = 200,
    minimum_stocks: int = 10,
    minimum_sessions: int = 30,
) -> pd.DataFrame:
    """Estimate A→B→A closure lifts against other predecessors of B."""

    frame = population.loc[population["target_available"].astype(bool)].copy()
    records: list[dict[str, object]] = []
    for representation, source in frame.groupby("representation", sort=True):
        sessions = sorted(source["session"].astype(str).unique())
        session_position = {value: index for index, value in enumerate(sessions)}
        for (previous_state, current_state), pair in source.groupby(
            ["previous_state_1", "current_state"], sort=True
        ):
            previous_state_int = int(cast(Any, previous_state))
            current_state_int = int(cast(Any, current_state))
            baseline = source.loc[
                source["current_state"].eq(current_state_int)
                & source["previous_state_1"].ne(previous_state_int)
            ]
            support = len(pair)
            stocks = pair["symbol"].nunique()
            pair_sessions = pair["session"].nunique()
            supported = bool(
                support >= minimum_rows
                and stocks >= minimum_stocks
                and pair_sessions >= minimum_sessions
                and not baseline.empty
            )
            closure_rate = float(pair["target_pair_closure"].mean())
            baseline_rate = (
                float(baseline["target_pair_closure"].mean()) if len(baseline) else math.nan
            )
            lift = closure_rate - baseline_rate if len(baseline) else math.nan
            ci_low = math.nan
            ci_high = math.nan
            p_value = math.nan
            if supported:
                pair_aggregate = pair.groupby("session")["target_pair_closure"].agg(
                    ["sum", "count"]
                )
                baseline_aggregate = baseline.groupby("session")["target_pair_closure"].agg(
                    ["sum", "count"]
                )
                pair_sum = np.zeros(len(sessions), dtype=float)
                pair_count = np.zeros(len(sessions), dtype=float)
                base_sum = np.zeros(len(sessions), dtype=float)
                base_count = np.zeros(len(sessions), dtype=float)
                for session, row in pair_aggregate.iterrows():
                    row_position = session_position[str(session)]
                    pair_sum[row_position] = float(row["sum"])
                    pair_count[row_position] = float(row["count"])
                for session, row in baseline_aggregate.iterrows():
                    row_position = session_position[str(session)]
                    base_sum[row_position] = float(row["sum"])
                    base_count[row_position] = float(row["count"])
                pair_seed = int(
                    hashlib.sha256(
                        f"{representation}:{previous_state_int}:{current_state_int}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                rng = np.random.default_rng(seed + pair_seed % 100_000)
                sampled = rng.integers(0, len(sessions), size=(draws, len(sessions)))
                valid = (pair_count[sampled].sum(axis=1) > 0) & (
                    base_count[sampled].sum(axis=1) > 0
                )
                values = (
                    pair_sum[sampled].sum(axis=1) / pair_count[sampled].sum(axis=1)
                    - base_sum[sampled].sum(axis=1) / base_count[sampled].sum(axis=1)
                )[valid]
                ci_low = float(np.quantile(values, 0.025))
                ci_high = float(np.quantile(values, 0.975))
                lower_tail = (np.sum(values <= 0.0) + 1.0) / (len(values) + 1.0)
                upper_tail = (np.sum(values >= 0.0) + 1.0) / (len(values) + 1.0)
                p_value = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))
            records.append(
                {
                    "representation": representation,
                    "period": period,
                    "previous_state": previous_state_int,
                    "current_state": current_state_int,
                    "loop_pair_id": _loop_pair(previous_state_int, current_state_int),
                    "loop_orientation": (
                        f"{previous_state_int}->{current_state_int}->{previous_state_int}"
                    ),
                    "rows": support,
                    "stocks": stocks,
                    "sessions": pair_sessions,
                    "closure_rate": closure_rate,
                    "same_current_state_other_predecessor_rate": baseline_rate,
                    "closure_lift": lift,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "p_value": p_value,
                    "supported": supported,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                }
            )
    output = pd.DataFrame.from_records(records)
    output["q_value"] = math.nan
    for _, indices in output.groupby(["representation", "period"], sort=True).groups.items():
        group_indices = list(indices)
        supported_index = [value for value in group_indices if bool(output.at[value, "supported"])]
        if supported_index:
            output.loc[supported_index, "q_value"] = benjamini_hochberg(
                output.loc[supported_index, "p_value"].to_numpy(dtype=float)
            )
    return output.sort_values(
        ["representation", "period", "previous_state", "current_state"]
    ).reset_index(drop=True)


def pair_replication_table(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Join development-selected pair effects to unchanged assessment effects."""

    development = pair_metrics.loc[pair_metrics["period"].eq("DEVELOPMENT_2024")].copy()
    assessment = pair_metrics.loc[pair_metrics["period"].eq("ASSESSMENT_2025")].copy()
    keys = ["representation", "previous_state", "current_state", "loop_pair_id", "loop_orientation"]
    columns = keys + [
        "rows",
        "stocks",
        "sessions",
        "closure_rate",
        "closure_lift",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "q_value",
        "supported",
    ]
    merged = development[columns].merge(
        assessment[columns],
        on=keys,
        suffixes=("_development", "_assessment"),
        validate="one_to_one",
    )
    merged["development_selected"] = merged["supported_development"].astype(bool) & merged[
        "q_value_development"
    ].lt(0.05)
    merged["same_direction"] = np.sign(merged["closure_lift_development"]) == np.sign(
        merged["closure_lift_assessment"]
    )
    merged["assessment_significant_same_direction"] = (
        merged["development_selected"].astype(bool)
        & merged["supported_assessment"].astype(bool)
        & merged["same_direction"].astype(bool)
        & merged["q_value_assessment"].lt(0.05)
    )
    return merged.sort_values(keys).reset_index(drop=True)


def primary_decision_inputs(
    bootstrap_summary: pd.DataFrame,
    quarter_metrics: pd.DataFrame,
    stock_deletions: pd.DataFrame,
    *,
    primary_representation: str,
    sensitivity_representation: str,
) -> dict[str, object]:
    """Evaluate the frozen non-promotable statistical conditions."""

    assessment = bootstrap_summary.loc[bootstrap_summary["evaluation_period"].eq("ASSESSMENT_2025")]
    primary = assessment.loc[assessment["representation"].eq(primary_representation)]
    sensitivity = assessment.loc[assessment["representation"].eq(sensitivity_representation)]
    if len(primary) != 1 or len(sensitivity) != 1:
        raise ValueError("primary decision requires both assessment representations")
    primary_row = primary.iloc[0]
    sensitivity_row = sensitivity.iloc[0]
    quarters = quarter_metrics.loc[quarter_metrics["representation"].eq(primary_representation)]
    deletions = stock_deletions.loc[stock_deletions["representation"].eq(primary_representation)]
    conditions = {
        "primary_log_loss_improvement_positive": bool(primary_row["log_loss_improvement"] > 0.0),
        "primary_brier_improvement_positive": bool(primary_row["brier_improvement"] > 0.0),
        "session_bootstrap_log_loss_lower_bound_positive": bool(
            primary_row["log_loss_ci_low"] > 0.0
        ),
        "at_least_three_of_four_quarters_positive": bool(
            len(quarters) == 4 and quarters["positive_log_loss_direction"].astype(bool).sum() >= 3
        ),
        "leave_one_stock_out_positive_fraction_at_least_0_90": bool(
            len(deletions) > 0
            and deletions["positive_log_loss_direction"].astype(bool).mean() >= 0.90
        ),
        "hard_state_sensitivity_same_direction": bool(
            sensitivity_row["log_loss_improvement"] > 0.0
        ),
    }
    statistical_pass = all(conditions.values())
    return {
        "preliminary_statistical_decision": (
            "fixed_model_history_increment_observed_nonpromotable"
            if statistical_pass
            else "fixed_model_no_history_increment"
        ),
        "statistical_conditions_pass": statistical_pass,
        "conditions": conditions,
        "primary_representation": primary_representation,
        "sensitivity_representation": sensitivity_representation,
        "primary_log_loss_improvement": float(primary_row["log_loss_improvement"]),
        "primary_log_loss_ci_low": float(primary_row["log_loss_ci_low"]),
        "primary_log_loss_ci_high": float(primary_row["log_loss_ci_high"]),
        "primary_brier_improvement": float(primary_row["brier_improvement"]),
        "primary_brier_ci_low": float(primary_row["brier_ci_low"]),
        "primary_brier_ci_high": float(primary_row["brier_ci_high"]),
        "positive_quarter_fraction": float(
            quarters["positive_log_loss_direction"].astype(bool).mean()
        ),
        "positive_leave_one_stock_out_fraction": float(
            deletions["positive_log_loss_direction"].astype(bool).mean()
        ),
        "sensitivity_log_loss_improvement": float(sensitivity_row["log_loss_improvement"]),
        "part_b_contract_reopened": False,
        "numeric_state_semantic_validity_claimed": False,
        "promotion_authorized": False,
        "economic_testing_authorized": False,
        **safety_flags(),
    }


__all__ = [
    "MODEL_CONTEXT_LEVELS",
    "PRIMARY_BASELINE",
    "PRIMARY_HISTORY_MODEL",
    "HierarchicalBinaryFrequencyModel",
    "benjamini_hochberg",
    "binary_log_loss",
    "brier_score",
    "build_pair_closure_population",
    "calibration_error",
    "canonical_json_bytes",
    "canonical_json_hash",
    "expanding_month_predictions",
    "frozen_assessment_predictions",
    "pair_orientation_metrics",
    "pair_replication_table",
    "paired_session_bootstrap",
    "prediction_metric_table",
    "primary_decision_inputs",
    "quarter_and_stock_deletion_metrics",
    "roc_auc",
    "safety_flags",
    "top_decile_lift",
]
