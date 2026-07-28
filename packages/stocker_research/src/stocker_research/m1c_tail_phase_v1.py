"""Deterministic retrospective mechanics for M1C Tail Phase V1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from stocker_prospective.contract import M1C_FROZEN_THRESHOLD
from stocker_prospective.direction import FrozenDirectionRuntime
from stocker_prospective.direction_features import (
    DirectionFeatureBar,
    FrozenDirectionFeatureBuilder,
)
from stocker_prospective.frozen_m1c import FreshEpisodeTracker, FrozenM1CRuntime
from stocker_prospective.tail_phase_v1 import (
    M1C_TAIL_PHASE_V1_VERSION,
    MovementConsumedBarV1,
    TailPhaseTrackerV1,
    assert_tail_phase_unprotected_sessions,
    assign_movement_consumed_bucket_v1,
    calculate_movement_consumed_v1,
)
from stocker_research.m1c_low_movement_v0 import (
    calculate_checkpoint_outcomes,
    iv_expected_absolute,
)
from stocker_research.stock_local_directional_archetypes_v0 import (
    construct_fresh_episodes,
)

DEVELOPMENT_START_V1: Final[date] = date(2024, 1, 1)
DEVELOPMENT_END_V1: Final[date] = date(2024, 12, 31)
M1C_MODEL_VERSION_V1: Final[str] = "frozen-m1c-v0"


@dataclass(frozen=True)
class FrozenConsumedMedianV1:
    value: float
    complete_observations: int
    start: date = DEVELOPMENT_START_V1
    end: date = DEVELOPMENT_END_V1
    predictor_values_only: bool = True


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _session_date(value: object) -> date:
    return pd.Timestamp(cast(Any, value)).date()


def _bar_groups(
    completed_bars: pd.DataFrame,
) -> dict[tuple[str, str], tuple[MovementConsumedBarV1, ...]]:
    required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "high",
        "low",
    }
    _require_columns(completed_bars, required, label="completed bars")
    assert_tail_phase_unprotected_sessions(completed_bars["session"])
    output: dict[tuple[str, str], tuple[MovementConsumedBarV1, ...]] = {}
    ordered = completed_bars.sort_values(
        ["stock", "session", "bar_ordinal"],
        kind="mergesort",
    )
    for (stock, session), group in ordered.groupby(["stock", "session"], sort=False):
        session_rows: list[MovementConsumedBarV1] = []
        for raw_row in group.itertuples(index=False):
            row = cast(Any, raw_row)
            session_rows.append(
                MovementConsumedBarV1(
                    symbol=str(stock),
                    session=_session_date(session),
                    bar_ordinal=int(row.bar_ordinal),
                    bar_start_timestamp=pd.Timestamp(row.bar_start_timestamp).to_pydatetime(),
                    bar_complete_timestamp=pd.Timestamp(row.bar_complete_timestamp).to_pydatetime(),
                    high=float(row.high),
                    low=float(row.low),
                    finalised=bool(getattr(row, "finalised", True)),
                )
            )
        rows = tuple(session_rows)
        output[(str(stock), str(session))] = rows
    return output


def _direction_bar_groups(
    completed_bars: pd.DataFrame,
    *,
    required_stock_sessions: set[tuple[str, str]],
) -> dict[tuple[str, str], tuple[DirectionFeatureBar, ...]]:
    required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "historical_relative_activity",
        "bar_log_return",
        "vti__bar_log_return",
    }
    _require_columns(completed_bars, required, label="frozen A1 bars")
    assert_tail_phase_unprotected_sessions(completed_bars["session"])
    output: dict[tuple[str, str], tuple[DirectionFeatureBar, ...]] = {}
    keys = completed_bars["stock"].astype(str) + "\x1f" + completed_bars["session"].astype(str)
    encoded_required = {f"{stock}\x1f{session}" for stock, session in required_stock_sessions}
    ordered = completed_bars.loc[keys.isin(encoded_required)].sort_values(
        ["stock", "session", "bar_ordinal"],
        kind="mergesort",
    )
    for (stock, session), group in ordered.groupby(["stock", "session"], sort=False):
        rows: list[DirectionFeatureBar] = []
        for raw_row in group.itertuples(index=False):
            row = cast(Any, raw_row)
            rows.append(
                DirectionFeatureBar(
                    symbol=str(stock),
                    session=_session_date(session),
                    bar_ordinal=int(row.bar_ordinal),
                    bar_start_timestamp=pd.Timestamp(row.bar_start_timestamp).to_pydatetime(),
                    bar_complete_timestamp=pd.Timestamp(row.bar_complete_timestamp).to_pydatetime(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    historical_relative_activity=float(row.historical_relative_activity),
                    stock_log_return=float(row.bar_log_return),
                    market_log_return=float(row.vti__bar_log_return),
                    finalised=bool(getattr(row, "finalised", True)),
                )
            )
        output[(str(stock), str(session))] = tuple(rows)
    return output


def _alignment_state(stock_return: float, context_return: float) -> str:
    if (
        not math.isfinite(stock_return)
        or not math.isfinite(context_return)
        or stock_return == 0.0
        or context_return == 0.0
    ):
        return "FLAT_OR_UNKNOWN"
    return "ALIGNED" if np.sign(stock_return) == np.sign(context_return) else "OPPOSED"


def score_frozen_m1c_checkpoint_rows_v1(
    checkpoint_rows: pd.DataFrame,
    *,
    runtime: FrozenM1CRuntime,
) -> pd.DataFrame:
    """Score a checkpoint panel with the committed no-fit M1C runtime."""

    required = {
        "stock",
        "session",
        "checkpoint",
        *runtime.required_group_o_features,
        *runtime.causal_group_i_features,
    }
    _require_columns(checkpoint_rows, required, label="frozen M1C checkpoints")
    assert_tail_phase_unprotected_sessions(checkpoint_rows["session"])
    records: list[dict[str, object]] = []
    for raw_row in checkpoint_rows.itertuples(index=False):
        row = cast(Any, raw_row)
        values = row._asdict()
        score = runtime.score(
            symbol=str(row.stock),
            checkpoint=int(row.checkpoint),
            group_o_context={name: values[name] for name in runtime.required_group_o_features},
            causal_group_i={name: values[name] for name in runtime.causal_group_i_features},
        )
        records.append(
            {
                "M1C_probability": score.probability,
                "m1c_high_tail_v1": score.threshold_passed,
                "m1c_high_tail_threshold_v1": score.threshold,
                "m1c_model_hash_v1": score.model_hash,
                "m1c_feature_hash_v1": score.feature_hash,
                "m1c_missing_feature_count_v1": score.missing_feature_count,
            }
        )
    return pd.concat(
        [checkpoint_rows.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )


def _incomplete_a1_record(
    *,
    direction_runtime: FrozenDirectionRuntime,
    reason: str,
    checkpoint: int,
    bars: tuple[DirectionFeatureBar, ...] = (),
) -> dict[str, object]:
    by_ordinal = {bar.bar_ordinal: bar for bar in bars}
    expected = (checkpoint - 3, checkpoint - 2)
    pre_entry = tuple(by_ordinal[ordinal] for ordinal in expected if ordinal in by_ordinal)
    pre_entry_complete = bool(
        len(pre_entry) == 2
        and tuple(item.bar_ordinal for item in pre_entry) == expected
        and pre_entry[1].bar_start_timestamp == pre_entry[0].bar_complete_timestamp
    )
    stock_return = (
        float(sum(item.stock_log_return for item in pre_entry))
        if pre_entry_complete and all(math.isfinite(item.stock_log_return) for item in pre_entry)
        else math.nan
    )
    market_return = (
        float(sum(item.market_log_return for item in pre_entry))
        if pre_entry_complete and all(math.isfinite(item.market_log_return) for item in pre_entry)
        else math.nan
    )
    return {
        "A1_complete_v1": False,
        "A1_missing_reason_v1": reason,
        "A1_probability_up_v1": math.nan,
        "A1_confidence_v1": math.nan,
        "A1_action_v1": None,
        "A1_boundary_v1": math.nan,
        "A1_feature_hash_v1": None,
        "A1_model_hash_v1": direction_runtime.model_hash,
        "A1_preprocessing_hash_v1": direction_runtime.preprocessing_hash,
        "A1_maximum_feature_timestamp_v1": None,
        "pre_entry_stock_signed_return_10m_v1": stock_return,
        "pre_entry_broad_market_signed_return_10m_v1": market_return,
        "stock_market_alignment_v1": _alignment_state(
            stock_return,
            market_return,
        ),
        "pre_entry_sector_signed_return_v1": math.nan,
        "stock_sector_alignment_v1": "UNKNOWN",
        "sector_context_status_v1": "out_of_scope_not_available",
        "market_volatility_state_status_v1": "out_of_scope_not_available",
    }


def attach_frozen_a1_and_regime_v1(
    checkpoint_rows: pd.DataFrame,
    completed_bars: pd.DataFrame,
    *,
    feature_builder: FrozenDirectionFeatureBuilder,
    direction_runtime: FrozenDirectionRuntime,
) -> pd.DataFrame:
    """Attach unchanged A1 and the existing fixed pre-entry market horizon."""

    _require_columns(
        checkpoint_rows,
        {"stock", "session", "checkpoint"},
        label="frozen A1 checkpoints",
    )
    assert_tail_phase_unprotected_sessions(checkpoint_rows["session"])
    required_stock_sessions = {
        (str(cast(Any, row).stock), str(cast(Any, row).session))
        for row in checkpoint_rows.itertuples(index=False)
    }
    grouped_bars = _direction_bar_groups(
        completed_bars,
        required_stock_sessions=required_stock_sessions,
    )
    records: list[dict[str, object]] = []
    for raw_row in checkpoint_rows.itertuples(index=False):
        row = cast(Any, raw_row)
        stock = str(row.stock)
        checkpoint = int(row.checkpoint)
        available = grouped_bars.get((stock, str(row.session)), ())
        by_ordinal = {bar.bar_ordinal: bar for bar in available}
        required_ordinals = tuple(range(checkpoint))
        bars = tuple(by_ordinal[ordinal] for ordinal in required_ordinals if ordinal in by_ordinal)
        if len(bars) != checkpoint:
            records.append(
                _incomplete_a1_record(
                    direction_runtime=direction_runtime,
                    reason="incomplete_contiguous_checkpoint_bars",
                    checkpoint=checkpoint,
                    bars=bars,
                )
            )
            continue
        try:
            result = feature_builder.build(
                symbol=stock,
                checkpoint=checkpoint,
                completed_bars=bars,
            )
            classification = direction_runtime.classify_one(
                model_id="A1",
                raw_features=result.raw_features,
                symbol=stock,
                checkpoint=checkpoint,
                checkpoint_category=result.checkpoint_category,
                day_of_week=result.day_of_week,
            )
        except ValueError as error:
            records.append(
                _incomplete_a1_record(
                    direction_runtime=direction_runtime,
                    reason=f"frozen_a1_input_incomplete:{error}",
                    checkpoint=checkpoint,
                    bars=bars,
                )
            )
            continue
        pre_entry = bars[-3:-1]
        stock_return = float(sum(item.stock_log_return for item in pre_entry))
        market_return = float(sum(item.market_log_return for item in pre_entry))
        records.append(
            {
                "A1_complete_v1": True,
                "A1_missing_reason_v1": None,
                "A1_probability_up_v1": classification.probability_up,
                "A1_confidence_v1": classification.confidence,
                "A1_action_v1": classification.action,
                "A1_boundary_v1": classification.boundary,
                "A1_feature_hash_v1": result.feature_hash,
                "A1_model_hash_v1": classification.model_hash,
                "A1_preprocessing_hash_v1": classification.preprocessing_hash,
                "A1_maximum_feature_timestamp_v1": (result.maximum_direction_feature_timestamp),
                "pre_entry_stock_signed_return_10m_v1": stock_return,
                "pre_entry_broad_market_signed_return_10m_v1": market_return,
                "stock_market_alignment_v1": _alignment_state(
                    stock_return,
                    market_return,
                ),
                "pre_entry_sector_signed_return_v1": math.nan,
                "stock_sector_alignment_v1": "UNKNOWN",
                "sector_context_status_v1": "out_of_scope_not_available",
                "market_volatility_state_status_v1": "out_of_scope_not_available",
            }
        )
    return pd.concat(
        [
            checkpoint_rows.reset_index(drop=True),
            pd.DataFrame(records),
        ],
        axis=1,
    )


def build_tail_phase_checkpoint_rows_v1(
    checkpoint_rows: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Attach stock-local V1 predictor fields to every observed frozen checkpoint."""

    required = {
        "stock",
        "session",
        "checkpoint",
        "feature_available_timestamp_utc",
        "M1C_probability",
        "atm_iv",
    }
    _require_columns(checkpoint_rows, required, label="checkpoint rows")
    assert_tail_phase_unprotected_sessions(checkpoint_rows["session"])
    if checkpoint_rows.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("checkpoint identities must be unique")
    grouped_bars = _bar_groups(completed_bars)
    tracker = TailPhaseTrackerV1()
    output = checkpoint_rows.copy()
    output["_input_order_v1"] = np.arange(len(output), dtype=np.int64)
    output = output.sort_values(
        ["stock", "session", "checkpoint"],
        kind="mergesort",
    ).reset_index(drop=True)
    records: list[dict[str, object]] = []
    for raw_row in output.itertuples(index=False):
        row = cast(Any, raw_row)
        stock = str(row.stock)
        session_text = str(row.session)
        session = _session_date(row.session)
        checkpoint = int(row.checkpoint)
        probability_raw = row.M1C_probability
        probability = (
            None if probability_raw is None or pd.isna(probability_raw) else float(probability_raw)
        )
        timestamp = pd.Timestamp(row.feature_available_timestamp_utc)
        if timestamp.tzinfo is None:
            raise ValueError("checkpoint feature timestamp must be timezone-aware")
        session_bars = grouped_bars.get((stock, session_text), ())
        bar_by_ordinal = {bar.bar_ordinal: bar for bar in session_bars}
        trigger = bar_by_ordinal.get(checkpoint - 1)
        valid = bool(getattr(row, "checkpoint_valid", True))
        invalid_reasons: list[str] = []
        supplied_missing_reason = getattr(row, "checkpoint_missing_reason", None)
        if supplied_missing_reason is not None and not pd.isna(supplied_missing_reason):
            valid = False
            invalid_reasons.append(str(supplied_missing_reason))
        if probability is None:
            valid = False
            invalid_reasons.append("m1c_probability_missing")
        if trigger is None:
            valid = False
            invalid_reasons.append("trigger_bar_missing")
        elif trigger.bar_complete_timestamp != timestamp.to_pydatetime():
            valid = False
            invalid_reasons.append("feature_timestamp_not_trigger_close")
        if probability is not None and (
            not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("M1C probabilities must be finite and lie in [0, 1]")
        phase = tracker.evaluate(
            symbol=stock,
            session=session,
            checkpoint=checkpoint,
            causal_timestamp=timestamp.to_pydatetime(),
            probability=probability,
            valid=valid,
            invalid_reason=(None if not invalid_reasons else ";".join(invalid_reasons)),
        )
        atm_iv_raw = row.atm_iv
        atm_iv = math.nan if atm_iv_raw is None or pd.isna(atm_iv_raw) else float(atm_iv_raw)
        denominator = (
            iv_expected_absolute(atm_iv, 15) if math.isfinite(atm_iv) and atm_iv > 0.0 else None
        )
        consumed = calculate_movement_consumed_v1(
            symbol=stock,
            session=session,
            checkpoint=checkpoint,
            completed_bars=session_bars,
            previous_close_implied_movement_15m=denominator,
        )
        entry = bar_by_ordinal.get(checkpoint)
        entry_timestamp = timestamp.to_pydatetime() if entry is None else entry.bar_start_timestamp
        records.append(
            {
                **phase.model_dump(mode="python"),
                **consumed.model_dump(mode="python"),
                "movement_consumed_bucket_v1": "UNKNOWN_INCOMPLETE",
                "m1c_high_tail_threshold_v1": M1C_FROZEN_THRESHOLD,
                "m1c_model_version_v1": M1C_MODEL_VERSION_V1,
                "tail_phase_schema_version_v1": M1C_TAIL_PHASE_V1_VERSION,
                "signal_timestamp": timestamp,
                "prospective_entry_timestamp": entry_timestamp,
                "prospective_entry_timestamp_complete_v1": entry is not None,
            }
        )
    record_frame = pd.DataFrame(records, index=output.index)
    if "m1c_high_tail_v1" in output:
        observed = output["m1c_high_tail_v1"].notna()
        expected = output.loc[observed, "M1C_probability"].ge(M1C_FROZEN_THRESHOLD)
        if not output.loc[observed, "m1c_high_tail_v1"].astype(bool).equals(expected):
            raise ValueError("precomputed M1C tail membership differs from exact threshold")
        output = output.drop(columns="m1c_high_tail_v1")
    if "m1c_high_tail_threshold_v1" in output:
        thresholds = pd.to_numeric(
            output["m1c_high_tail_threshold_v1"],
            errors="raise",
        )
        if not bool(thresholds.eq(M1C_FROZEN_THRESHOLD).all()):
            raise ValueError("precomputed M1C tail threshold differs")
        output = output.drop(columns="m1c_high_tail_threshold_v1")
    attached = pd.concat([output, record_frame], axis=1)
    return (
        attached.sort_values("_input_order_v1", kind="mergesort")
        .drop(columns="_input_order_v1")
        .reset_index(drop=True)
    )


def freeze_movement_consumed_median_v1(
    checkpoint_rows: pd.DataFrame,
) -> FrozenConsumedMedianV1:
    """Freeze the unweighted median of complete 2024 predictor values only."""

    _require_columns(
        checkpoint_rows,
        {"session", "movement_consumed_v1", "movement_consumed_complete_v1"},
        label="movement-consumed calibration",
    )
    assert_tail_phase_unprotected_sessions(checkpoint_rows["session"])
    sessions = pd.to_datetime(checkpoint_rows["session"], errors="raise", utc=True)
    development = checkpoint_rows.loc[
        sessions.between(
            pd.Timestamp(DEVELOPMENT_START_V1, tz="UTC"),
            pd.Timestamp(DEVELOPMENT_END_V1, tz="UTC"),
        )
        & checkpoint_rows["movement_consumed_complete_v1"].astype(bool)
    ].copy()
    values = pd.to_numeric(
        development["movement_consumed_v1"],
        errors="coerce",
    ).to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or bool((values < 0.0).any()):
        raise ValueError("complete 2024 movement-consumed predictors are unavailable")
    return FrozenConsumedMedianV1(
        value=float(np.median(values)),
        complete_observations=int(len(values)),
    )


def apply_frozen_consumed_bucket_v1(
    checkpoint_rows: pd.DataFrame,
    *,
    frozen_median: float,
) -> pd.DataFrame:
    """Apply the same frozen predictor split to every chronology partition."""

    _require_columns(
        checkpoint_rows,
        {"movement_consumed_v1"},
        label="movement-consumed bucket",
    )
    output = checkpoint_rows.copy()
    output["movement_consumed_bucket_v1"] = [
        assign_movement_consumed_bucket_v1(
            None if pd.isna(value) else float(value),
            frozen_median=frozen_median,
        )
        for value in output["movement_consumed_v1"]
    ]
    output["movement_consumed_frozen_median_v1"] = float(frozen_median)
    return output


def construct_fresh_tail_episodes_v1(checkpoint_rows: pd.DataFrame) -> pd.DataFrame:
    """Reuse the existing crossing/30-minute-spacing definition and add its live ID."""

    required = {
        "stock",
        "session",
        "checkpoint",
        "signal_timestamp",
        "prospective_entry_timestamp",
        "M1C_probability",
        "partition",
    }
    _require_columns(checkpoint_rows, required, label="fresh episode rows")
    assert_tail_phase_unprotected_sessions(checkpoint_rows["session"])
    source = checkpoint_rows.copy()
    source["movement_probability"] = source["M1C_probability"]
    episodes = construct_fresh_episodes(
        source,
        threshold=M1C_FROZEN_THRESHOLD,
        probability_column="movement_probability",
    )

    tracker = FreshEpisodeTracker()
    identifiers: dict[tuple[str, str, int], str] = {}
    for raw_row in source.sort_values(
        ["stock", "session", "checkpoint"],
        kind="mergesort",
    ).itertuples(index=False):
        row = cast(Any, raw_row)
        decision = tracker.evaluate(
            symbol=str(row.stock),
            session=_session_date(row.session),
            checkpoint=int(row.checkpoint),
            trigger_bar_end=pd.Timestamp(row.signal_timestamp).to_pydatetime(),
            probability=float(row.M1C_probability),
            eligible=True,
        )
        if decision.fresh_episode:
            assert decision.episode_id is not None
            identifiers[(str(row.stock), str(row.session), int(row.checkpoint))] = (
                decision.episode_id
            )
    episode_keys = {
        (
            str(cast(Any, row).stock),
            str(cast(Any, row).session),
            int(cast(Any, row).checkpoint),
        )
        for row in episodes.itertuples(index=False)
    }
    if episode_keys != set(identifiers):
        raise AssertionError("prospective and retrospective fresh-episode definitions diverged")
    episodes["episode_id"] = [
        identifiers[
            (
                str(cast(Any, row).stock),
                str(cast(Any, row).session),
                int(cast(Any, row).checkpoint),
            )
        ]
        for row in episodes.itertuples(index=False)
    ]
    episodes["existing_fresh_episode_identifier"] = episodes["episode_id"]
    return episodes.reset_index(drop=True)


def attach_canonical_tail_outcomes_v1(
    checkpoint_rows: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Attach existing 10/15-minute outcomes and one bounded remaining-range ratio."""

    assert_tail_phase_unprotected_sessions(checkpoint_rows["session"])
    assert_tail_phase_unprotected_sessions(completed_bars["session"])
    source = checkpoint_rows.copy()
    if "row_id" not in source:
        source["row_id"] = [
            f"{cast(Any, row).stock}|{cast(Any, row).session}|{int(cast(Any, row).checkpoint)}"
            for row in source.itertuples(index=False)
        ]
    movement, path = calculate_checkpoint_outcomes(
        source,
        completed_bars,
        horizons=(10, 15),
    )
    movement_values = movement.drop(
        columns=[
            "stock",
            "session",
            "checkpoint",
            "entry_timestamp",
            "entry_price",
        ],
    )
    path_values = path.drop(
        columns=[
            "stock",
            "session",
            "checkpoint",
            "entry_timestamp",
            "entry_price",
        ],
    )
    output = source.merge(movement_values, on="row_id", how="left", validate="one_to_one")
    output = output.merge(
        path_values,
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_path"),
    )
    output["future_10m_absolute_return_v1"] = output["absolute_return_10m"]
    output["future_10m_signed_return_v1"] = output["signed_return_10m"]
    output["future_15m_absolute_movement_v1"] = output["absolute_return_15m"]
    output["future_15m_iv_residual_v1"] = output["terminal_iv_residual_15m"]
    output["future_15m_exceed_iv_v1"] = output["movement_exceeds_iv_15m"]
    ratios: list[float] = []
    complete: list[bool] = []
    reasons: list[str | None] = []
    for raw_row in output.itertuples(index=False):
        row = cast(Any, raw_row)
        pre_raw = row.movement_consumed_numerator_v1
        post_raw = row.realised_path_range_10m
        pre = float(pre_raw) if pre_raw is not None and not pd.isna(pre_raw) else math.nan
        post = float(post_raw) if post_raw is not None and not pd.isna(post_raw) else math.nan
        available = bool(row.available_10m)
        denominator = pre + post
        valid = bool(
            available
            and math.isfinite(pre)
            and pre >= 0.0
            and math.isfinite(post)
            and post >= 0.0
            and denominator > 0.0
        )
        ratios.append(post / denominator if valid else math.nan)
        complete.append(valid)
        reasons.append(None if valid else "pre_or_post_range_incomplete_or_nonpositive")
    output["post_share_of_local_range_v1"] = ratios
    output["post_share_of_local_range_complete_v1"] = complete
    output["post_share_of_local_range_missing_reason_v1"] = reasons
    return output


__all__ = [
    "FrozenConsumedMedianV1",
    "apply_frozen_consumed_bucket_v1",
    "attach_canonical_tail_outcomes_v1",
    "attach_frozen_a1_and_regime_v1",
    "build_tail_phase_checkpoint_rows_v1",
    "construct_fresh_tail_episodes_v1",
    "freeze_movement_consumed_median_v1",
    "score_frozen_m1c_checkpoint_rows_v1",
]
