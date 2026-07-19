"""Hard, hysteretic, and posterior-support sensitivity for loop events."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stocker_research.causal_state_export_v2 import HysteresisConfig, hysteretic_states
from stocker_research.loop_dictionary_v2 import LoopDictionary
from stocker_research.loop_events_v2 import PrimaryOutcomeLabel
from stocker_research.loop_ledger_v2 import (
    LoopEventLedgerBundle,
    build_loop_event_ledgers,
    session_source_is_complete,
)
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine


@dataclass(frozen=True, slots=True)
class HierarchicalStateIdentity:
    numeric: np.ndarray
    tokens: tuple[str, ...]


def hierarchical_state_ids(
    market_states: np.ndarray,
    stock_states: np.ndarray,
    *,
    stock_state_count: int,
) -> HierarchicalStateIdentity:
    """Build deterministic market × stock identities without numeric ambiguity."""

    market = np.asarray(market_states, dtype=int)
    stock = np.asarray(stock_states, dtype=int)
    if market.shape != stock.shape or market.ndim != 1:
        raise ValueError("market and stock state arrays must be one-dimensional and aligned")
    invalid_stock = (stock < 0) | (stock >= stock_state_count)
    if stock_state_count <= 1 or np.any(market < 0) or np.any(invalid_stock):
        raise ValueError("hierarchical state values are outside their declared support")
    numeric = market * stock_state_count + stock
    tokens = tuple(
        f"market_{int(market_state)}::stock_{int(stock_state)}"
        for market_state, stock_state in zip(market, stock, strict=True)
    )
    return HierarchicalStateIdentity(numeric=numeric.astype(np.int32), tokens=tokens)


def classify_soft_support(
    hard_event: bool,
    completion_probability: float,
    *,
    low_support_below: float = 0.25,
    robust_support_at_or_above: float = 0.50,
) -> str:
    """Classify support attached to a hard event; soft mass never creates one."""

    if not 0.0 <= completion_probability <= 1.0 or not math.isfinite(completion_probability):
        raise ValueError("completion_probability must be a finite probability")
    if not hard_event:
        return "NO_HARD_EVENT"
    if completion_probability < low_support_below:
        return "LOW_SOFT_SUPPORT_HARD_EVENT"
    if completion_probability >= robust_support_at_or_above:
        return "SOFT_SUPPORTED_ROBUST_EVENT"
    return "INTERMEDIATE_SOFT_SUPPORT_HARD_EVENT"


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(
        -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)), axis=1),
        dtype=float,
    )


def hysteretic_states_by_session(
    state_probabilities: np.ndarray,
    *,
    session_groups: Sequence[np.ndarray],
    config: HysteresisConfig,
) -> np.ndarray:
    """Apply causal hysteresis independently inside every declared session."""

    probabilities = np.asarray(state_probabilities, dtype=float)
    if probabilities.ndim != 2 or len(probabilities) == 0:
        raise ValueError("state probabilities must be a non-empty matrix")
    output = np.full(len(probabilities), -1, dtype=np.int16)
    assigned = np.zeros(len(probabilities), dtype=bool)
    for raw_group in session_groups:
        group = np.asarray(raw_group, dtype=int)
        if len(group) == 0:
            continue
        if (
            np.any(np.diff(group) <= 0)
            or group.min() < 0
            or group.max() >= len(probabilities)
            or assigned[group].any()
        ):
            raise ValueError("session groups overlap or contain invalid positions")
        assigned[group] = True
        output[group] = hysteretic_states(probabilities[group], config=config)
    if not assigned.all():
        raise ValueError("session groups do not cover every posterior row")
    return output


def cleaning_run_changes(
    raw_states: np.ndarray,
    cleaned_states: np.ndarray,
    *,
    session_groups: Sequence[np.ndarray],
) -> pd.DataFrame:
    """Identify which original hard-state runs contain a cleanup relabel."""

    raw = np.asarray(raw_states, dtype=int)
    cleaned = np.asarray(cleaned_states, dtype=int)
    if raw.ndim != 1 or cleaned.shape != raw.shape:
        raise ValueError("raw and cleaned states must be aligned vectors")
    rows: list[dict[str, object]] = []
    assigned = np.zeros(len(raw), dtype=bool)
    for group_ordinal, raw_group in enumerate(session_groups):
        group = np.asarray(raw_group, dtype=int)
        if len(group) == 0:
            continue
        if (
            np.any(np.diff(group) <= 0)
            or group.min() < 0
            or group.max() >= len(raw)
            or assigned[group].any()
        ):
            raise ValueError("session groups overlap or contain invalid positions")
        assigned[group] = True
        local = raw[group]
        starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
        ends = np.r_[starts[1:], len(local)]
        for run_ordinal, (start, end) in enumerate(zip(starts, ends, strict=True)):
            positions = group[int(start) : int(end)]
            changed_bars = int(np.sum(raw[positions] != cleaned[positions]))
            rows.append(
                {
                    "group_ordinal": group_ordinal,
                    "run_ordinal": run_ordinal,
                    "start_position": int(positions[0]),
                    "end_position_exclusive": int(positions[-1]) + 1,
                    "raw_state": int(raw[positions[0]]),
                    "raw_duration": len(positions),
                    "changed_bars": changed_bars,
                    "changed": changed_bars > 0,
                }
            )
    if not assigned.all():
        raise ValueError("session groups do not cover every cleanup row")
    return pd.DataFrame(rows)


def transition_confidence(
    state_probabilities: np.ndarray,
    *,
    hard_states: np.ndarray,
    hysteretic_states: np.ndarray,
    session_groups: Sequence[np.ndarray],
) -> pd.DataFrame:
    """Describe every causal hard-state transition and bounded reversal."""

    probabilities = np.asarray(state_probabilities, dtype=float)
    hard = np.asarray(hard_states, dtype=int)
    hysteretic = np.asarray(hysteretic_states, dtype=int)
    if (
        probabilities.ndim != 2
        or hard.shape != (len(probabilities),)
        or hysteretic.shape != hard.shape
    ):
        raise ValueError("probability and hard-state arrays are not aligned")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("state probabilities do not normalize")
    entropy = _entropy(probabilities)
    entropy_threshold = float(np.quantile(entropy, 0.75, method="linear"))
    rows: list[dict[str, object]] = []
    assigned = np.zeros(len(hard), dtype=bool)
    for raw_group in session_groups:
        group = np.asarray(raw_group, dtype=int)
        if len(group) == 0:
            continue
        if np.any(np.diff(group) <= 0) or assigned[group].any():
            raise ValueError("session groups overlap or are not increasing")
        assigned[group] = True
        run_age = 1
        ages = np.ones(len(group), dtype=int)
        for local_index in range(1, len(group)):
            current_position = int(group[local_index])
            previous_position = int(group[local_index - 1])
            run_age = run_age + 1 if hard[current_position] == hard[previous_position] else 1
            ages[local_index] = run_age
        for local_index in range(1, len(group)):
            position = int(group[local_index])
            previous_position = int(group[local_index - 1])
            previous_state = int(hard[previous_position])
            new_state = int(hard[position])
            if new_state == previous_state:
                continue
            order = np.argsort(-probabilities[position], kind="stable")
            margin = float(probabilities[position, order[0]] - probabilities[position, order[1]])
            future = hard[group[local_index + 1 : local_index + 3]]
            one_bar_reversal = bool(len(future) >= 1 and int(future[0]) == previous_state)
            two_bar_reversal = bool(np.any(future == previous_state))
            rows.append(
                {
                    "position": position,
                    "previous_hard_state": previous_state,
                    "new_hard_state": new_state,
                    "previous_top_state_probability": float(
                        np.max(probabilities[previous_position])
                    ),
                    "new_top_state_probability": float(np.max(probabilities[position])),
                    "posterior_entropy": float(entropy[position]),
                    "top_two_margin": margin,
                    "probability_mass_on_old_state": float(probabilities[position, previous_state]),
                    "probability_mass_on_new_state": float(probabilities[position, new_state]),
                    "one_bar_reversal": one_bar_reversal,
                    "two_bar_reversal": two_bar_reversal,
                    "hysteretic_state_agreement": bool(hysteretic[position] == new_state),
                    "departing_state_duration": int(ages[local_index - 1]),
                    "new_state_age": int(ages[local_index]),
                    "state_duration": int(ages[local_index - 1]),
                    "margin_lt_0_02": margin < 0.02,
                    "margin_lt_0_05": margin < 0.05,
                    "entropy_top_quartile": bool(entropy[position] >= entropy_threshold),
                    "new_state_probability_lt_0_50": bool(
                        probabilities[position, new_state] < 0.50
                    ),
                    "soft_supports_transition": bool(
                        probabilities[position, new_state]
                        >= probabilities[position, previous_state]
                    ),
                }
            )
    if not assigned.all():
        raise ValueError("session groups do not cover every probability row")
    return pd.DataFrame(rows)


def build_loop_ledgers_by_representation(
    decisions: pd.DataFrame,
    *,
    dictionary: LoopDictionary,
    horizon_bars: int,
    allowed_states: frozenset[int],
) -> dict[str, LoopEventLedgerBundle]:
    """Reconstruct the same loop dictionary under both causal hard surfaces."""

    required = {"hard_state_legacy", "hard_state_hysteretic"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"decision frame lacks required state representations: {missing}")
    return {
        "legacy_hard_map": build_loop_event_ledgers(
            decisions,
            dictionary=dictionary,
            horizon_bars=horizon_bars,
            allowed_states=allowed_states,
            state_column="hard_state_legacy",
        ),
        "causal_hysteretic": build_loop_event_ledgers(
            decisions,
            dictionary=dictionary,
            horizon_bars=horizon_bars,
            allowed_states=allowed_states,
            state_column="hard_state_hysteretic",
        ),
    }


def reconstruct_first_event_outcomes(
    decisions: pd.DataFrame,
    states: np.ndarray,
    *,
    dictionary: LoopDictionary,
    horizon_bars: int,
    allowed_states: frozenset[int],
) -> pd.DataFrame:
    """Reconstruct only mutually exclusive outcomes for a sensitivity state path.

    This is semantically equivalent to the outcome surface from
    :func:`build_loop_event_ledgers`, but deliberately omits prefix, legacy-label,
    and completion ledgers that are not needed for K/seed attribution.
    """

    required = {
        "decision_id",
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "decision_timestamp",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"decision surface lacks required fields: {missing}")
    labels = np.asarray(states, dtype=int)
    if labels.shape != (len(decisions),):
        raise ValueError("sensitivity state path does not align with decisions")
    if horizon_bars <= 0 or not allowed_states:
        raise ValueError("horizon and allowed state support must be non-empty")
    if decisions["decision_id"].duplicated().any():
        raise ValueError("decision IDs are not unique")

    columns = sorted(required)
    frame = decisions[columns].copy()
    frame["_sensitivity_state"] = labels
    frame = frame.sort_values(
        ["symbol", "session", "bar_ordinal", "decision_id"], kind="mergesort"
    ).reset_index(drop=True)
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=allowed_states)
    rows: list[dict[str, object]] = []
    for (symbol, session), local in frame.groupby(["symbol", "session"], sort=False):
        local = local.reset_index(drop=True)
        local_states = local["_sensitivity_state"].to_numpy(dtype=int)
        complete = session_source_is_complete(local) and bool(
            np.isin(local_states, list(allowed_states)).all()
        )
        if not complete:
            rows.extend(
                {
                    "decision_id": str(decision_id),
                    "primary_label": str(PrimaryOutcomeLabel.UNAVAILABLE),
                    "bars_until_completion": None,
                }
                for decision_id in local["decision_id"]
            )
            continue

        event_mask = np.r_[True, local_states[1:] != local_states[:-1]]
        event_positions = np.flatnonzero(event_mask)
        event_for_bar = np.cumsum(event_mask, dtype=int) - 1
        event_starts = pd.to_datetime(local.iloc[event_positions]["bar_start_timestamp"], utc=True)
        event_available = pd.to_datetime(
            local.iloc[event_positions]["bar_complete_timestamp"], utc=True
        )
        trace = engine.scan_state_events(
            tuple(int(value) for value in local_states[event_positions]),
            bar_ordinals=tuple(
                int(value)
                for value in local.iloc[event_positions]["bar_ordinal"].to_numpy(dtype=int)
            ),
            event_timestamps=tuple(value.to_pydatetime() for value in event_starts),
            available_timestamps=tuple(value.to_pydatetime() for value in event_available),
        )
        session_end = int(local["bar_ordinal"].max())
        decision_ids = local["decision_id"].astype(str).to_numpy()
        decision_ordinals = local["bar_ordinal"].to_numpy(dtype=int)
        decision_timestamps = pd.to_datetime(local["decision_timestamp"], utc=True)
        for local_index in range(len(local)):
            decision_timestamp = decision_timestamps.iloc[local_index].to_pydatetime()
            outcome = engine.outcome_for_decision(
                trace,
                decision_id=str(decision_ids[local_index]),
                decision_event_index=int(event_for_bar[local_index]),
                decision_bar_ordinal=int(decision_ordinals[local_index]),
                decision_timestamp=decision_timestamp,
                decision_available_timestamp=decision_timestamp,
                horizon_bars=horizon_bars,
                session_end_bar_ordinal=session_end,
                source_available=True,
                symbol=str(symbol),
                session=str(session),
            )
            rows.append(
                {
                    "decision_id": outcome.decision_id,
                    "primary_label": str(outcome.primary_label),
                    "bars_until_completion": outcome.bars_until_completion,
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class EventAgreementMetrics:
    decisions: int
    exact_fraction: float
    same_primitive_bounded_shift_fraction: float
    primitive_mismatch_fraction: float


def _missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def _integer_scalar(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(float(value))
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"value is not integer-like: {type(value).__name__}")


def _same_number(left: object, right: object) -> bool:
    if _missing_scalar(left) and _missing_scalar(right):
        return True
    if _missing_scalar(left) or _missing_scalar(right):
        return False
    return _integer_scalar(left) == _integer_scalar(right)


def compare_representation_events(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    allowed_shift_bars: int,
) -> tuple[pd.DataFrame, EventAgreementMetrics]:
    """Classify exact, bounded-shift, and identity-mismatch first events."""

    required = ("decision_id", "primary_label", "bars_until_completion")
    required_set = set(required)
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = sorted(required_set.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} event frame lacks required columns: {missing}")
        if frame["decision_id"].duplicated().any():
            raise ValueError(f"{name} event frame contains duplicate decisions")
    if allowed_shift_bars < 0:
        raise ValueError("allowed_shift_bars must be nonnegative")
    merged = reference[list(required)].merge(
        candidate[list(required)],
        on="decision_id",
        how="outer",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    classes: list[str] = []
    for row in merged.itertuples(index=False):
        reference_label = str(row.primary_label_reference)
        candidate_label = str(row.primary_label_candidate)
        same_label = reference_label == candidate_label
        if same_label and _same_number(
            row.bars_until_completion_reference, row.bars_until_completion_candidate
        ):
            classes.append("EXACT_EVENT_AGREEMENT")
            continue
        bounded_shift = False
        if same_label and reference_label.startswith("loop_p_"):
            left = row.bars_until_completion_reference
            right = row.bars_until_completion_candidate
            if not _missing_scalar(left) and not _missing_scalar(right):
                bounded_shift = (
                    abs(_integer_scalar(left) - _integer_scalar(right)) <= allowed_shift_bars
                )
        classes.append(
            "SAME_PRIMITIVE_SHIFTED_TIMESTAMP" if bounded_shift else "PRIMITIVE_MISMATCH"
        )
    merged["agreement_class"] = classes
    count = len(merged)
    exact = int(merged["agreement_class"].eq("EXACT_EVENT_AGREEMENT").sum())
    shifted = int(merged["agreement_class"].eq("SAME_PRIMITIVE_SHIFTED_TIMESTAMP").sum())
    mismatch = count - exact - shifted
    denominator = max(count, 1)
    return merged, EventAgreementMetrics(
        decisions=count,
        exact_fraction=exact / denominator,
        same_primitive_bounded_shift_fraction=(exact + shifted) / denominator,
        primitive_mismatch_fraction=mismatch / denominator,
    )


__all__ = [
    "EventAgreementMetrics",
    "HierarchicalStateIdentity",
    "build_loop_ledgers_by_representation",
    "classify_soft_support",
    "cleaning_run_changes",
    "compare_representation_events",
    "hysteretic_states_by_session",
    "hierarchical_state_ids",
    "reconstruct_first_event_outcomes",
    "transition_confidence",
]
