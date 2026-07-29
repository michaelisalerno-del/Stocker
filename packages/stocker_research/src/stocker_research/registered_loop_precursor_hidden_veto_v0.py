"""Pure structural helpers for the registered-loop precursor/veto quick screen.

This module is retrospective research infrastructure only.  It has no return,
direction, order, broker, execution, deployment, or production-runtime surface.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final, cast

import numpy as np
import pandas as pd

PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2025-08-23T00:00:00Z")
OPENING_KEYS: Final[tuple[str, ...]] = ("symbol", "session", "decision_ordinal")
FROZEN_HIDDEN_FAMILIES: Final[tuple[str, ...]] = (
    "unregistered_primitive_like__5-6-5",
    "unregistered_primitive_like__2-3-2",
    "unregistered_primitive_like__2-5-2",
    "unregistered_primitive_like__4-7-4",
)
OTHER_HIDDEN_FAMILY: Final[str] = "OTHER_UNREGISTERED_FAMILY"


def exact_precursor_identity_eligible(kind: str, identity: str) -> bool:
    """Return whether a completed precursor is an exact preregistered identity."""

    if kind == "registered":
        return bool(identity)
    if kind == "hidden":
        return identity in FROZEN_HIDDEN_FAMILIES
    return False


def opening_panel_differences(
    archived: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    shared_fields: Sequence[str],
    probability_fields: Sequence[str],
    tolerance: float = 1e-12,
) -> dict[str, float | int | bool]:
    """Compare an independently reconstructed opening panel to its frozen source."""

    required = {*OPENING_KEYS, *shared_fields, *probability_fields}
    for name, frame in (("archived", archived), ("reconstructed", reconstructed)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} opening panel lacks fields: {missing}")
        if frame.duplicated(list(OPENING_KEYS)).any():
            raise ValueError(f"{name} opening panel keys are not unique")
    comparison = archived.loc[:, list(required)].merge(
        reconstructed.loc[:, list(required)],
        on=list(OPENING_KEYS),
        how="outer",
        suffixes=("_archived", "_reconstructed"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("opening panel keys do not reconstruct exactly")

    def maximum_difference(fields: Sequence[str]) -> float:
        maximum = 0.0
        for field in fields:
            left = pd.to_numeric(comparison[f"{field}_archived"], errors="raise").to_numpy(
                dtype=float
            )
            right = pd.to_numeric(comparison[f"{field}_reconstructed"], errors="raise").to_numpy(
                dtype=float
            )
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(f"opening field {field} is non-finite")
            maximum = max(maximum, float(np.max(np.abs(left - right), initial=0.0)))
        return maximum

    shared_difference = maximum_difference(shared_fields)
    probability_difference = maximum_difference(probability_fields)
    return {
        "rows": int(len(comparison)),
        "maximum_shared_field_difference": shared_difference,
        "maximum_probability_difference": probability_difference,
        "tolerance": float(tolerance),
        "passed": shared_difference <= tolerance and probability_difference <= tolerance,
    }


def _structural_target(
    origin_bar_ordinal: int,
    events: pd.DataFrame,
    *,
    horizon_bars: int,
) -> int:
    if "completion_bar_ordinal" not in events:
        raise ValueError("completion_bar_ordinal is required")
    ordinal = pd.to_numeric(events["completion_bar_ordinal"], errors="raise").to_numpy(dtype=int)
    after = ordinal > int(origin_bar_ordinal)
    within = ordinal <= int(origin_bar_ordinal) + int(horizon_bars)
    return int(bool(np.any(after & within)))


def registered_completion_target(origin_bar_ordinal: int, events: pd.DataFrame) -> int:
    """Return the frozen registered-completion-within-twelve-bars target."""

    return _structural_target(origin_bar_ordinal, events, horizon_bars=12)


def hidden_event_target(origin_bar_ordinal: int, events: pd.DataFrame) -> int:
    """Return the frozen unregistered-event-within-six-bars target."""

    return _structural_target(origin_bar_ordinal, events, horizon_bars=6)


def deduplicate_registered_completions(
    completions: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    """Deduplicate semantic completions and attach the latest eligible checkpoint."""

    identity = ["symbol", "session", "completion_timestamp_utc", "semantic_loop_id"]
    completion_required = {
        *identity,
        "completion_bar_ordinal",
        "motif_type",
        "orientation_id",
    }
    decision_required = {
        *OPENING_KEYS,
        "repo_bar_start_ordinal",
        "feature_available_timestamp_utc",
    }
    missing_completion = sorted(completion_required.difference(completions.columns))
    missing_decision = sorted(decision_required.difference(decisions.columns))
    if missing_completion or missing_decision:
        raise ValueError(
            f"registered deduplication fields missing: completions={missing_completion}, "
            f"decisions={missing_decision}"
        )
    events = completions.copy()
    events["completion_timestamp_utc"] = pd.to_datetime(
        events["completion_timestamp_utc"], utc=True, errors="raise"
    )
    checkpoints = decisions.copy()
    checkpoints["feature_available_timestamp_utc"] = pd.to_datetime(
        checkpoints["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    rows: list[dict[str, Any]] = []
    for key, group in events.groupby(identity, sort=True, dropna=False):
        first = group.sort_values("orientation_id", kind="mergesort").iloc[0]
        symbol, session, completion_timestamp, semantic_loop_id = key
        origin = checkpoints.loc[
            checkpoints["symbol"].astype(str).eq(str(symbol))
            & checkpoints["session"].astype(str).eq(str(session))
        ].copy()
        completion_ordinal = int(first["completion_bar_ordinal"])
        eligibility_cutoff = pd.Timestamp(cast(Any, completion_timestamp))
        if "completion_available_timestamp_utc" in group:
            eligibility_cutoff = pd.Timestamp(
                pd.to_datetime(
                    group["completion_available_timestamp_utc"], utc=True, errors="raise"
                ).min()
            )
        origin = origin.loc[
            origin["repo_bar_start_ordinal"].astype(int).lt(completion_ordinal)
            & origin["repo_bar_start_ordinal"].astype(int).add(12).ge(completion_ordinal)
            & origin["feature_available_timestamp_utc"].lt(eligibility_cutoff)
        ]
        if origin.empty:
            continue
        latest = origin.sort_values(
            ["feature_available_timestamp_utc", "decision_ordinal"], kind="mergesort"
        ).iloc[-1]
        row: dict[str, Any] = {str(column): value for column, value in first.to_dict().items()}
        row.update(
            {
                "symbol": str(symbol),
                "session": str(session),
                "completion_timestamp_utc": pd.Timestamp(cast(Any, completion_timestamp)),
                "semantic_loop_id": str(semantic_loop_id),
                "orientation_ids_json": json.dumps(
                    sorted(group["orientation_id"].astype(str).unique().tolist())
                ),
                "orientation_count": int(group["orientation_id"].astype(str).nunique()),
                "decision_ordinal": int(latest["decision_ordinal"]),
                "decision_timestamp_utc": latest["feature_available_timestamp_utc"],
                "decision_repo_bar_start_ordinal": int(latest["repo_bar_start_ordinal"]),
                "year_month": str(session)[:7],
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if result.duplicated(identity).any():
        raise AssertionError("registered completion identities remain duplicated")
    return result.sort_values(
        ["session", "completion_timestamp_utc", "symbol", "semantic_loop_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def nearest_precursor_label(
    *,
    completed_loop: bool,
    matching_active_prefix: bool,
    other_active_prefix: bool,
    regime_transition: bool,
) -> str:
    """Apply the frozen mutually exclusive nearest-precursor priority."""

    if completed_loop:
        return "NEAREST_COMPLETED_LOOP_EVENT"
    if matching_active_prefix:
        return "ACTIVE_MATCHING_PREFIX"
    if other_active_prefix:
        return "OTHER_ACTIVE_PREFIX"
    if regime_transition:
        return "REGIME_TRANSITION"
    return "NO_IDENTIFIED_STRUCTURAL_PRECURSOR"


def _window(frame: pd.DataFrame, ordinal_column: str, start: int, end: int) -> pd.DataFrame:
    if ordinal_column not in frame:
        raise ValueError(f"{ordinal_column} is required")
    ordinal = pd.to_numeric(frame[ordinal_column], errors="raise").astype(int)
    return frame.loc[ordinal.ge(start) & ordinal.le(end)].copy()


def precursor_window_features(
    completion: pd.Series,
    registered_completions: pd.DataFrame,
    hidden_completions: pd.DataFrame,
    active_prefixes: pd.DataFrame,
    state_bars: pd.DataFrame,
    *,
    lookback_bars: int,
) -> dict[str, Any]:
    """Census all preregistered precursors in one strict pre-completion window."""

    if int(lookback_bars) not in (3, 6, 12):
        raise ValueError("precursor lookback must be exactly 3, 6, or 12 bars")
    required_completion = {"completion_bar_ordinal", "semantic_loop_id", "motif_type"}
    missing_completion = sorted(required_completion.difference(completion.index))
    if missing_completion:
        raise ValueError(f"completion fields missing: {missing_completion}")
    completion_bar = int(completion["completion_bar_ordinal"])
    window_start = completion_bar - int(lookback_bars)
    window_end = completion_bar - 1
    target_identity = str(completion["semantic_loop_id"])
    target_family = str(completion["motif_type"])

    registered = _window(
        registered_completions,
        "completion_bar_ordinal",
        window_start,
        window_end,
    )
    hidden = _window(
        hidden_completions,
        "completion_bar_ordinal",
        window_start,
        window_end,
    )
    prefixes = _window(active_prefixes, "bar_ordinal", window_start, window_end)
    states = _window(state_bars, "bar_ordinal", window_start, window_end).sort_values(
        "bar_ordinal", kind="mergesort"
    )
    transition_states = _window(
        state_bars,
        "bar_ordinal",
        window_start - 1,
        window_end,
    ).sort_values("bar_ordinal", kind="mergesort")

    registered_required = {"semantic_loop_id", "motif_type"}
    hidden_required = {"hidden_family_class"}
    prefix_required = {"semantic_loop_id", "orientation_id", "progress_states"}
    state_required = {
        "causal_hard_state",
        "transition_probability",
        "posterior_entropy",
        "top_state_probability",
        "expected_state_age",
    }
    for name, frame, required in (
        ("registered", registered, registered_required),
        ("hidden", hidden, hidden_required),
        ("prefix", prefixes, prefix_required),
        ("state", states, state_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} precursor fields missing: {missing}")
    if states["bar_ordinal"].duplicated().any():
        raise ValueError("state precursor bars are duplicated")
    if transition_states["bar_ordinal"].duplicated().any():
        raise ValueError("transition state precursor bars are duplicated")

    registered_identity = registered["semantic_loop_id"].astype(str)
    registered_family = registered["motif_type"].astype(str)
    same_identity = bool(registered_identity.eq(target_identity).any())
    same_family_different = bool(
        (registered_family.eq(target_family) & registered_identity.ne(target_identity)).any()
    )
    different_family = bool(registered_family.ne(target_family).any())

    hidden_family = hidden["hidden_family_class"].astype(str)
    hidden_flags = {
        "hidden_5_6_5": bool(hidden_family.eq(FROZEN_HIDDEN_FAMILIES[0]).any()),
        "hidden_2_3_2": bool(hidden_family.eq(FROZEN_HIDDEN_FAMILIES[1]).any()),
        "hidden_2_5_2": bool(hidden_family.eq(FROZEN_HIDDEN_FAMILIES[2]).any()),
        "hidden_4_7_4": bool(hidden_family.eq(FROZEN_HIDDEN_FAMILIES[3]).any()),
        "hidden_other_unregistered_family": bool(hidden_family.eq(OTHER_HIDDEN_FAMILY).any()),
    }
    prefix_identity = prefixes["semantic_loop_id"].astype(str)
    matching_prefix = bool(prefix_identity.eq(target_identity).any())
    other_prefix = bool(prefix_identity.ne(target_identity).any())
    immediate_prefix = bool(
        pd.to_numeric(prefixes["bar_ordinal"], errors="raise").astype(int).eq(window_end).any()
    )
    prefix_candidates = prefixes.drop_duplicates(
        ["bar_ordinal", "semantic_loop_id", "orientation_id", "progress_states"]
    )
    maximum_depth = (
        int(pd.to_numeric(prefixes["progress_states"], errors="raise").max())
        if not prefixes.empty
        else 0
    )

    hard_states = pd.to_numeric(states["causal_hard_state"], errors="raise").to_numpy(dtype=int)
    transition_ordinals = pd.to_numeric(transition_states["bar_ordinal"], errors="raise").to_numpy(
        dtype=int
    )
    transition_hard_states = pd.to_numeric(
        transition_states["causal_hard_state"], errors="raise"
    ).to_numpy(dtype=int)
    transition_count = int(
        sum(
            right_ordinal == left_ordinal + 1
            and right_ordinal >= window_start
            and left_state != right_state
            for left_ordinal, right_ordinal, left_state, right_state in zip(
                transition_ordinals[:-1],
                transition_ordinals[1:],
                transition_hard_states[:-1],
                transition_hard_states[1:],
                strict=True,
            )
        )
    )

    def endpoint_change(column: str) -> float:
        if states.empty:
            return float("nan")
        values = pd.to_numeric(states[column], errors="raise").to_numpy(dtype=float)
        return float(values[-1] - values[0])

    any_completed = bool(not registered.empty or not hidden.empty)
    no_identified = not (any_completed or not prefixes.empty or transition_count > 0)
    nearest_label = nearest_precursor_label(
        completed_loop=any_completed,
        matching_active_prefix=matching_prefix,
        other_active_prefix=other_prefix,
        regime_transition=transition_count > 0,
    )
    completed_rows: list[dict[str, Any]] = []
    for row in registered.itertuples(index=False):
        completed_rows.append(
            {
                "bar_ordinal": int(cast(Any, row.completion_bar_ordinal)),
                "kind": "registered",
                "identity": str(row.semantic_loop_id),
                "motif_type": str(row.motif_type),
            }
        )
    for row in hidden.itertuples(index=False):
        completed_rows.append(
            {
                "bar_ordinal": int(cast(Any, row.completion_bar_ordinal)),
                "kind": "hidden",
                "identity": str(row.hidden_family_class),
            }
        )
    completed_rows.sort(key=lambda value: (int(value["bar_ordinal"]), str(value["identity"])))
    nearest_completed = completed_rows[-1] if completed_rows else None

    result: dict[str, Any] = {
        "lookback_bars": int(lookback_bars),
        "window_start_bar_ordinal": window_start,
        "window_end_bar_ordinal": window_end,
        "history_bars_available": int(states["bar_ordinal"].nunique()),
        "complete_prior_history": int(states["bar_ordinal"].nunique()) == int(lookback_bars),
        "same_registered_identity": same_identity,
        "same_registered_broad_family_different_identity": same_family_different,
        "different_registered_broad_family": different_family,
        "any_prior_registered_completion": bool(not registered.empty),
        "any_hidden_unregistered_completion": bool(not hidden.empty),
        "active_prefix_immediately_before_completion": immediate_prefix,
        "active_prefix_any": bool(not prefixes.empty),
        "prefix_candidate_count": int(len(prefix_candidates)),
        "maximum_prefix_depth": maximum_depth,
        "matching_prefix_any": matching_prefix,
        "other_prefix_any": other_prefix,
        "any_regime_transition": transition_count > 0,
        "regime_transition_count": transition_count,
        "posterior_transition_probability": (
            float(states.iloc[-1]["transition_probability"]) if not states.empty else float("nan")
        ),
        "posterior_entropy_change": endpoint_change("posterior_entropy"),
        "top_state_probability_change": endpoint_change("top_state_probability"),
        "expected_state_age_change": endpoint_change("expected_state_age"),
        "no_identified_structural_precursor": no_identified,
        "nearest_precursor_label": nearest_label,
        "nearest_completed_kind": nearest_completed["kind"] if nearest_completed else None,
        "nearest_completed_identity": nearest_completed["identity"] if nearest_completed else None,
        "nearest_completed_bars_before": (
            completion_bar - int(nearest_completed["bar_ordinal"]) if nearest_completed else None
        ),
        "registered_precursors_json": json.dumps(completed_rows, sort_keys=True),
        "prefix_precursors_json": json.dumps(
            prefixes.loc[
                :,
                ["bar_ordinal", "semantic_loop_id", "orientation_id", "progress_states"],
            ]
            .sort_values(["bar_ordinal", "semantic_loop_id", "orientation_id"], kind="mergesort")
            .to_dict(orient="records"),
            sort_keys=True,
        ),
        "state_ordinals_json": json.dumps(states["bar_ordinal"].astype(int).tolist()),
        "hard_state_path_json": json.dumps(hard_states.tolist()),
        "transition_state_path_json": json.dumps(
            transition_states.loc[:, ["bar_ordinal", "causal_hard_state"]]
            .astype({"bar_ordinal": int, "causal_hard_state": int})
            .to_dict(orient="records"),
            sort_keys=True,
        ),
        "state_metrics_json": json.dumps(
            states.loc[
                :,
                [
                    "bar_ordinal",
                    "transition_probability",
                    "posterior_entropy",
                    "top_state_probability",
                    "expected_state_age",
                ],
            ].to_dict(orient="records"),
            sort_keys=True,
        ),
    }
    result.update(hidden_flags)
    return result


def sample_matched_pseudo_completions(
    observed: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Sample one stock/month/clock-matched non-completion timestamp per event."""

    strata = {"symbol", "year_month", "clock_bin"}
    observed_required = {
        "event_id",
        "session",
        "completion_bar_ordinal",
        "completion_timestamp_utc",
        *strata,
    }
    eligible_required = {
        "session",
        "completion_bar_ordinal",
        "completion_timestamp_utc",
        "full_prior_history",
        "registered_completion_at_timestamp",
        *strata,
    }
    missing_observed = sorted(observed_required.difference(observed.columns))
    missing_eligible = sorted(eligible_required.difference(eligible.columns))
    if missing_observed or missing_eligible:
        raise ValueError(
            f"matched pseudo-completion fields missing: observed={missing_observed}, "
            f"eligible={missing_eligible}"
        )
    pool = eligible.loc[
        eligible["full_prior_history"].astype(bool)
        & ~eligible["registered_completion_at_timestamp"].astype(bool)
    ].copy()
    pool["completion_timestamp_utc"] = pd.to_datetime(
        pool["completion_timestamp_utc"], utc=True, errors="raise"
    )
    reject_protected_dates(pool, column="completion_timestamp_utc")
    pool = pool.sort_values(
        ["symbol", "year_month", "clock_bin", "session", "completion_timestamp_utc"],
        kind="mergesort",
    ).reset_index(drop=True)
    generator = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for source in observed.sort_values("event_id", kind="mergesort").itertuples(index=False):
        candidates = pool.loc[
            pool["symbol"].astype(str).eq(str(source.symbol))
            & pool["year_month"].astype(str).eq(str(source.year_month))
            & pool["clock_bin"].astype(str).eq(str(source.clock_bin))
        ]
        if candidates.empty:
            raise ValueError(
                "no full-history non-completion pseudo timestamp for "
                f"{source.symbol}/{source.year_month}/{source.clock_bin}"
            )
        sampled = candidates.iloc[int(generator.integers(0, len(candidates)))]
        row: dict[str, Any] = dict(cast(Any, source)._asdict())
        row.update(
            {
                "source_event_id": str(source.event_id),
                "observed_session": str(source.session),
                "observed_completion_bar_ordinal": int(cast(Any, source.completion_bar_ordinal)),
                "observed_completion_timestamp_utc": pd.Timestamp(
                    cast(Any, source.completion_timestamp_utc)
                ),
                "session": str(sampled["session"]),
                "year_month": str(sampled["year_month"]),
                "clock_bin": str(sampled["clock_bin"]),
                "completion_bar_ordinal": int(sampled["completion_bar_ordinal"]),
                "completion_timestamp_utc": pd.Timestamp(sampled["completion_timestamp_utc"]),
                "pseudo_completion": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or bool(((values < 0.0) | (values > 1.0)).any())
    ):
        raise ValueError("p-values must be a finite one-dimensional vector in [0, 1]")
    if values.size == 0:
        return []
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return cast(list[float], output.astype(float).tolist())


def reject_protected_dates(frame: pd.DataFrame, *, column: str = "session") -> None:
    """Fail closed if any materialised analytical row reaches the protected boundary."""

    if column not in frame:
        raise ValueError(f"{column} is required for protected-date checking")
    timestamps = pd.to_datetime(frame[column], utc=True, errors="raise")
    if bool(timestamps.ge(PROTECTED_START).any()):
        raise ValueError("protected date 2025-08-23 or later materialised")


def candidate_threshold(probabilities: pd.Series | np.ndarray) -> float:
    """Freeze the unweighted development 80th percentile of valid OOF B0 scores."""

    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("candidate threshold requires valid probabilities")
    return float(np.quantile(values, 0.80))


def freeze_hidden_risk_thresholds(
    probabilities: pd.Series | np.ndarray,
) -> dict[str, float | list[float]]:
    """Freeze development hidden-risk quartile and quintile boundaries."""

    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("hidden-risk thresholds require valid probabilities")
    quartiles = np.quantile(values, [0.25, 0.75]).astype(float)
    quintiles = np.quantile(values, [0.20, 0.40, 0.60, 0.80]).astype(float)
    return {
        "low_maximum": float(quartiles[0]),
        "high_minimum": float(quartiles[1]),
        "quintile_boundaries": quintiles.tolist(),
    }


def assign_hidden_risk(
    probabilities: pd.Series | np.ndarray,
    thresholds: dict[str, float | list[float]],
) -> pd.DataFrame:
    """Apply frozen hidden-risk groups and quintiles without assessment tuning."""

    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("hidden-risk assignment requires valid probabilities")
    low_value = thresholds["low_maximum"]
    high_value = thresholds["high_minimum"]
    boundary_values = thresholds["quintile_boundaries"]
    if isinstance(low_value, list) or isinstance(high_value, list):
        raise ValueError("hidden-risk quartile thresholds must be scalar")
    if not isinstance(boundary_values, list):
        raise ValueError("hidden-risk quintile thresholds must be a list")
    low = float(low_value)
    high = float(high_value)
    boundaries = np.asarray(boundary_values, dtype=float)
    if boundaries.shape != (4,) or not np.all(boundaries[:-1] <= boundaries[1:]):
        raise ValueError("four ordered quintile boundaries are required")
    groups = np.where(values <= low, "low", np.where(values >= high, "high", "middle"))
    quintiles = np.searchsorted(boundaries, values, side="right") + 1
    return pd.DataFrame(
        {
            "hidden_risk_group": groups.astype(object),
            "hidden_risk_quintile": quintiles.astype(int),
        }
    )


def _clipped_logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if not np.isfinite(probabilities).all() or bool(
        ((probabilities < 0.0) | (probabilities > 1.0)).any()
    ):
        raise ValueError("logit input must contain valid probabilities")
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return cast(np.ndarray, np.log(clipped / (1.0 - clipped)))


def veto_feature_frame(
    frame: pd.DataFrame,
    *,
    include_hidden_risk: bool,
    b0_column: str = "B0_probability",
    u1_column: str = "U1_probability",
) -> pd.DataFrame:
    """Build the frozen V0/V1 candidate-only feature surface."""

    required = {b0_column, "decision_ordinal"}
    if include_hidden_risk:
        required.add(u1_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"veto feature fields missing: {missing}")
    result = pd.DataFrame(index=frame.index)
    result["logit_B0_probability"] = _clipped_logit(frame[b0_column])
    result["checkpoint_12"] = frame["decision_ordinal"].astype(int).eq(12).astype(float)
    if include_hidden_risk:
        result["logit_U1_probability"] = _clipped_logit(frame[u1_column])
    return result.reset_index(drop=True)


def session_block_bootstrap_indices(
    frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> list[np.ndarray]:
    """Return whole-session fixed-seed bootstrap positional indices."""

    if "session" not in frame or int(draws) <= 0:
        raise ValueError("session column and positive draw count are required")
    sessions = sorted(frame["session"].astype(str).unique().tolist())
    if not sessions:
        raise ValueError("bootstrap population is empty")
    positions = {
        session: np.flatnonzero(frame["session"].astype(str).eq(session).to_numpy())
        for session in sessions
    }
    generator = np.random.default_rng(int(seed))
    output: list[np.ndarray] = []
    for _ in range(int(draws)):
        sampled = generator.choice(
            np.asarray(sessions, dtype=object), size=len(sessions), replace=True
        )
        output.append(np.concatenate([positions[str(session)] for session in sampled]).astype(int))
    return output


def permute_hidden_probability_within_slates(
    frame: pd.DataFrame,
    *,
    seed: int,
    feature: str = "U1_probability",
    slate_column: str = "slate_id",
) -> pd.DataFrame:
    """Permute only U1 within each frozen session/checkpoint slate."""

    missing = sorted({feature, slate_column}.difference(frame.columns))
    if missing:
        raise ValueError(f"hidden-probability permutation fields missing: {missing}")
    result = frame.copy()
    generator = np.random.default_rng(int(seed))
    permuted_values = result[feature].to_numpy(copy=True)
    for _, positions in result.groupby(slate_column, sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        values = permuted_values[index].copy()
        permuted_values[index] = generator.permutation(values)
    result[feature] = permuted_values
    return result


def choose_primary_decision(
    *,
    precursor_status: str,
    predictive_veto_status: str,
    realised_diversion_status: str,
) -> str:
    """Combine independently evaluated precursor, predictive, and mechanism statuses."""

    allowed = {"supported", "descriptive_only", "not_supported", "insufficient_support"}
    statuses = {precursor_status, predictive_veto_status, realised_diversion_status}
    if not statuses.issubset(allowed):
        raise ValueError(f"unknown experiment status: {sorted(statuses.difference(allowed))}")
    precursor_supported = precursor_status == "supported"
    veto_supported = predictive_veto_status == "supported"
    if precursor_supported and veto_supported:
        return "hidden_diversion_veto_and_registered_precursor_supported"
    if veto_supported:
        return "hidden_diversion_veto_supported_only"
    if precursor_supported:
        return "registered_precursor_structure_supported_only"
    if "descriptive_only" in statuses or realised_diversion_status == "supported":
        return "descriptive_precursor_or_veto_structure_only"
    if precursor_status == "insufficient_support":
        return "blocked_precursor_support_failure"
    if predictive_veto_status == "insufficient_support":
        return "blocked_candidate_veto_support_failure"
    return "no_hidden_veto_or_precursor_enrichment"


__all__ = [
    "assign_hidden_risk",
    "benjamini_hochberg",
    "candidate_threshold",
    "choose_primary_decision",
    "deduplicate_registered_completions",
    "freeze_hidden_risk_thresholds",
    "hidden_event_target",
    "nearest_precursor_label",
    "opening_panel_differences",
    "permute_hidden_probability_within_slates",
    "precursor_window_features",
    "registered_completion_target",
    "reject_protected_dates",
    "sample_matched_pseudo_completions",
    "session_block_bootstrap_indices",
    "veto_feature_frame",
]
