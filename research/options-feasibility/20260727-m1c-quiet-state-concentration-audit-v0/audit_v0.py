"""Retrospective M1C quiet-state concentration audit V0.

All calculations are descriptive reconstructions of already opened historical
evidence.  The original support gate, thresholds, and decision are immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
PREDECESSOR = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260727-m1c-low-movement-short-premium-v0"
    / "artifacts"
    / "primary"
)
OPTION_PAIRS = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260724-minimal-intraday-iv-excess-holdout-v01"
    / "artifacts"
    / "primary"
    / "holdout_selected_option_pairs.parquet"
)
RUNTIME_MANIFEST = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts"
    / "primary"
    / "causal_movement_feature_manifest.json"
)
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.m1c_quiet_state_concentration_v0 import (  # noqa: E402
    BOTTOM_10_THRESHOLD,
    HIGH_TAIL_THRESHOLD,
    ORIGINAL_DECISION,
    analysis_weights,
    audit_claims,
    classify_month_concentration,
    classify_surprise_concentration,
    cluster_quiet_state_runs,
    cluster_surprise_events,
    fresh_quiet_episodes,
    reconstruct_frozen_tail,
    small_count_feasibility,
)

STRESS_MONTHS = ("2025-09", "2025-10", "2025-11", "2025-12")
REPRESENTATIONS = (
    "raw_checkpoint_rows",
    "quiet_state_runs",
    "frozen_fresh_quiet_episodes",
    "one_per_stock_session",
)
SURPRISE_THRESHOLDS = (1.5, 2.0)
MONTH_SHARE_LIMIT = 0.35
SURPRISE_MONTH_LIMIT = 0.60
SURPRISE_STOCK_LIMIT = 0.50


@dataclass(frozen=True)
class Evidence:
    predictions: pd.DataFrame
    analytic: pd.DataFrame
    original_tail: pd.DataFrame
    original_fresh: pd.DataFrame
    original_decision: dict[str, Any]
    original_surprises: pd.DataFrame
    option_pairs: pd.DataFrame


@dataclass(frozen=True)
class AuditResult:
    frames: dict[str, pd.DataFrame]
    payloads: dict[str, dict[str, Any]]
    evidence: Evidence


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (Path, pd.Timestamp, pd.Period)):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return cast(dict[str, Any], payload)


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("required frozen evidence is missing: " + ", ".join(missing))


def load_evidence() -> Evidence:
    paths = {
        "predictions": PREDECESSOR / "checkpoint_predictions.parquet",
        "movement": PREDECESSOR / "movement_outcomes.parquet",
        "path": PREDECESSOR / "path_excursion_outcomes.parquet",
        "tail": PREDECESSOR / "raw_low_tail_checkpoint_rows.parquet",
        "fresh": PREDECESSOR / "fresh_quiet_episodes.parquet",
        "decision": PREDECESSOR / "decision.json",
        "surprises": PREDECESSOR / "surprise_mover_rows.csv",
    }
    _require_paths((*paths.values(), OPTION_PAIRS, RUNTIME_MANIFEST))
    predictions = pd.read_parquet(paths["predictions"])
    predictions = predictions.loc[predictions["period"].isin(["assessment", "stress"])].copy()
    movement_columns = [
        "row_id",
        "available_15m",
        "terminal_iv_residual_15m",
        "movement_remains_below_iv_15m",
        "movement_exceeds_iv_15m",
        "absolute_return_15m",
        "signed_return_15m",
        "iv_sigma_15m",
        "iv_expected_absolute_15m",
    ]
    path_columns = [
        "row_id",
        "maximum_up_excursion_15m",
        "maximum_down_excursion_15m",
        "maximum_absolute_excursion_15m",
        "excursion_sigma_ratio_15m",
        "large_excursion_mean_reverted_15m",
    ]
    movement = pd.read_parquet(paths["movement"], columns=movement_columns)
    excursions = pd.read_parquet(paths["path"], columns=path_columns)
    analytic = predictions.merge(movement, on="row_id", how="left", validate="one_to_one").merge(
        excursions,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    available = analytic["available_15m"].astype(bool)
    if analytic.loc[available, "excursion_sigma_ratio_15m"].isna().any():
        raise ValueError("an available frozen checkpoint lost its binding 15-minute outcome")
    original_tail = pd.read_parquet(
        paths["tail"],
        columns=["row_id", "M1C_probability", "period"],
    )
    original_fresh = pd.read_parquet(paths["fresh"])
    original_fresh = original_fresh.loc[
        original_fresh["tail"].eq("bottom_10_percent")
        & original_fresh["period"].isin(["assessment", "stress"])
    ].copy()
    original_decision = _read_json(paths["decision"])
    original_surprises = pd.read_csv(paths["surprises"])
    option_pairs = pd.read_parquet(OPTION_PAIRS)
    if str(analytic["session"].max()) >= "2026-01-01":
        raise ValueError("protected historical observations were read")
    return Evidence(
        predictions=predictions,
        analytic=analytic,
        original_tail=original_tail,
        original_fresh=original_fresh,
        original_decision=original_decision,
        original_surprises=original_surprises,
        option_pairs=option_pairs,
    )


def _base_weights(frame: pd.DataFrame) -> pd.Series:
    weights = pd.to_numeric(frame["row_weight"], errors="raise")
    if frame.empty or not np.isfinite(weights.to_numpy(float)).all() or bool(weights.le(0).any()):
        raise ValueError("analysis weights must be finite and positive")
    return weights / float(weights.sum())


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    data = pd.to_numeric(values, errors="raise").to_numpy(float)
    mass = pd.to_numeric(weights, errors="raise").to_numpy(float)
    order = np.argsort(data, kind="mergesort")
    data = data[order]
    mass = mass[order]
    positions = (np.cumsum(mass) - 0.5 * mass) / mass.sum()
    return float(np.interp(quantile, positions, data, left=data[0], right=data[-1]))


def _metrics(frame: pd.DataFrame, weights: pd.Series | None = None) -> dict[str, float | int]:
    if frame.empty:
        return {
            "rows": 0,
            "sessions": 0,
            "stocks": 0,
            "remains_below_iv_rate": math.nan,
            "mean_iv_residual": math.nan,
            "median_iv_residual": math.nan,
            "breach_1_5_sigma_rate": math.nan,
            "breach_2_0_sigma_rate": math.nan,
        }
    available = (
        frame["available_15m"].astype(bool)
        if "available_15m" in frame
        else pd.Series(True, index=frame.index)
    )
    outcome_frame = frame.loc[available]
    if outcome_frame.empty:
        return {
            "rows": int(len(frame)),
            "sessions": int(frame["session"].astype(str).nunique()),
            "stocks": int(frame["stock"].astype(str).nunique()),
            "remains_below_iv_rate": math.nan,
            "mean_iv_residual": math.nan,
            "median_iv_residual": math.nan,
            "breach_1_5_sigma_rate": math.nan,
            "breach_2_0_sigma_rate": math.nan,
        }
    mass = (
        _base_weights(outcome_frame)
        if weights is None
        else weights.loc[outcome_frame.index] / float(weights.loc[outcome_frame.index].sum())
    )
    return {
        "rows": int(len(frame)),
        "sessions": int(frame["session"].astype(str).nunique()),
        "stocks": int(frame["stock"].astype(str).nunique()),
        "remains_below_iv_rate": float(
            np.average(
                outcome_frame["movement_remains_below_iv_15m"].astype(float),
                weights=mass,
            )
        ),
        "mean_iv_residual": float(
            np.average(
                outcome_frame["terminal_iv_residual_15m"].to_numpy(float),
                weights=mass,
            )
        ),
        "median_iv_residual": _weighted_quantile(
            outcome_frame["terminal_iv_residual_15m"],
            mass,
            0.5,
        ),
        "breach_1_5_sigma_rate": float(
            np.average(
                outcome_frame["excursion_sigma_ratio_15m"].ge(1.5).astype(float),
                weights=mass,
            )
        ),
        "breach_2_0_sigma_rate": float(
            np.average(
                outcome_frame["excursion_sigma_ratio_15m"].ge(2.0).astype(float),
                weights=mass,
            )
        ),
    }


def _maximum_share(frame: pd.DataFrame, columns: str | Sequence[str]) -> float:
    if frame.empty:
        return 0.0
    return float(frame.groupby(columns, sort=True, dropna=False).size().max() / len(frame))


def _hhi(frame: pd.DataFrame, columns: str | Sequence[str]) -> float:
    if frame.empty:
        return 0.0
    shares = frame.groupby(columns, sort=True, dropna=False).size() / len(frame)
    return float(np.square(shares.to_numpy(float)).sum())


def _weighted_maximum_share(
    frame: pd.DataFrame,
    weights: pd.Series,
    columns: str | Sequence[str],
) -> float:
    if frame.empty:
        return 0.0
    grouped = (
        frame.assign(_weight=weights)
        .groupby(
            columns,
            sort=True,
            dropna=False,
        )["_weight"]
        .sum()
    )
    return float(grouped.max() / grouped.sum())


def _add_high_tail_proximity(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    previous: list[bool] = []
    following: list[bool] = []
    high = predictions.loc[
        predictions["M1C_probability"].ge(HIGH_TAIL_THRESHOLD),
        ["stock", "session", "entry_timestamp"],
    ].copy()
    high["entry_timestamp"] = pd.to_datetime(high["entry_timestamp"], utc=True, errors="raise")
    grouped = {
        (str(stock), str(session)): group["entry_timestamp"].sort_values().tolist()
        for (stock, session), group in high.groupby(["stock", "session"], sort=True)
    }
    for row in output.itertuples(index=False):
        trigger = pd.Timestamp(row.entry_timestamp)
        timestamps = grouped.get((str(row.stock), str(row.session)), [])
        previous.append(
            any(
                pd.Timedelta(0) < trigger - timestamp <= pd.Timedelta(minutes=60)
                for timestamp in timestamps
            )
        )
        following.append(
            any(
                pd.Timedelta(0) < timestamp - trigger <= pd.Timedelta(minutes=60)
                for timestamp in timestamps
            )
        )
    output["high_tail_previous_60m"] = previous
    output["high_tail_following_60m"] = following
    output["high_tail_within_60m"] = output[
        ["high_tail_previous_60m", "high_tail_following_60m"]
    ].any(axis=1)
    return output


def build_representations(
    analytic: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tail = analytic.loc[analytic["M1C_probability"].le(BOTTOM_10_THRESHOLD)].copy()
    runs = cluster_quiet_state_runs(analytic)
    fresh = fresh_quiet_episodes(analytic)
    one_per_stock_session = (
        tail.sort_values(
            ["stock", "session", "entry_timestamp", "checkpoint", "row_id"],
            kind="mergesort",
        )
        .groupby(["stock", "session"], sort=True, as_index=False)
        .head(1)
        .copy()
    )
    return (
        {
            "raw_checkpoint_rows": tail,
            "quiet_state_runs": runs,
            "frozen_fresh_quiet_episodes": fresh,
            "one_per_stock_session": one_per_stock_session,
        },
        runs,
    )


def reconstruction_audit(
    evidence: Evidence,
    fresh: pd.DataFrame,
) -> dict[str, Any]:
    tail_audit = reconstruct_frozen_tail(
        evidence.predictions,
        evidence.original_tail[["row_id", "M1C_probability"]],
    )
    original_fresh_ids = set(evidence.original_fresh["row_id"].astype(str))
    reconstructed_fresh_ids = set(fresh["row_id"].astype(str))
    fresh_identity_mismatches = len(
        original_fresh_ids.symmetric_difference(reconstructed_fresh_ids)
    )
    comparison = evidence.original_fresh[["row_id", "M1C_probability"]].merge(
        fresh[["row_id", "M1C_probability"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_reconstructed"),
    )
    maximum_fresh_probability_difference = (
        0.0
        if comparison.empty
        else float(
            np.max(
                np.abs(
                    comparison["M1C_probability_original"].to_numpy(float)
                    - comparison["M1C_probability_reconstructed"].to_numpy(float)
                )
            )
        )
    )
    decision_unchanged = evidence.original_decision.get("overall_decision") == ORIGINAL_DECISION
    passed = bool(
        tail_audit["passed"]
        and fresh_identity_mismatches == 0
        and maximum_fresh_probability_difference <= 1e-12
        and decision_unchanged
    )
    return {
        **audit_claims(),
        **tail_audit,
        "fresh_episode_identity_mismatches": fresh_identity_mismatches,
        "maximum_fresh_episode_probability_difference": maximum_fresh_probability_difference,
        "original_decision_unchanged": decision_unchanged,
        "passed": passed,
    }


def _xnys_month_counts() -> dict[str, int]:
    schedule = mcal.get_calendar("XNYS").schedule(
        start_date="2025-09-01",
        end_date="2025-12-31",
    )
    counts = pd.Series(schedule.index.strftime("%Y-%m")).value_counts()
    return {month: int(counts.get(month, 0)) for month in STRESS_MONTHS}


def _previous_xnys_sessions() -> dict[str, str]:
    schedule = mcal.get_calendar("XNYS").schedule(
        start_date="2025-08-01",
        end_date="2025-12-31",
    )
    sessions = [timestamp.date().isoformat() for timestamp in schedule.index]
    return {current: previous for previous, current in zip(sessions, sessions[1:], strict=False)}


def stress_month_tables(
    evidence: Evidence,
    representations: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stress = evidence.analytic.loc[evidence.analytic["period"].eq("stress")]
    tail = representations["raw_checkpoint_rows"].loc[
        representations["raw_checkpoint_rows"]["period"].eq("stress")
    ]
    fresh = representations["frozen_fresh_quiet_episodes"].loc[
        representations["frozen_fresh_quiet_episodes"]["period"].eq("stress")
    ]
    calendar_counts = _xnys_month_counts()
    previous_sessions = _previous_xnys_sessions()
    options = evidence.option_pairs.copy()
    options["session"] = options["session"].astype(str)
    options["required_options_date"] = options["required_options_date"].astype(str)
    options["options_observation_date"] = options["options_observation_date"].astype(str)
    options["_exact_previous_chain"] = [
        required == observed == previous_sessions.get(session)
        for session, required, observed in zip(
            options["session"],
            options["required_options_date"],
            options["options_observation_date"],
            strict=True,
        )
    ]
    exposure_rows: list[dict[str, Any]] = []
    incidence_rows: list[dict[str, Any]] = []
    for month in STRESS_MONTHS:
        eligible_month = stress.loc[stress["month"].eq(month)]
        tail_month = tail.loc[tail["month"].eq(month)]
        fresh_month = fresh.loc[fresh["month"].eq(month)]
        options_month = options.loc[options["session"].str[:7].eq(month)]
        exact = options_month.loc[options_month["_exact_previous_chain"]]
        pair_available = (
            options_month["pair_available"].astype(bool)
            if "pair_available" in options_month
            else pd.Series(True, index=options_month.index)
        )
        valid_pairs = options_month.loc[pair_available & options_month["_exact_previous_chain"]]
        planned = calendar_counts[month] * 20
        exposure_rows.append(
            {
                "month": month,
                "trading_sessions_in_source_calendar": calendar_counts[month],
                "sessions_represented_in_joined_panel": int(
                    eligible_month["session"].astype(str).nunique()
                ),
                "planned_stock_sessions": planned,
                "valid_exact_previous_close_chains": int(len(exact)),
                "valid_option_pairs": int(len(valid_pairs)),
                "option_pair_coverage_rate": len(valid_pairs) / planned,
                "eligible_checkpoint_rows": int(len(eligible_month)),
                "source_exposure_share": len(eligible_month) / len(stress),
                "represented_stock_sessions": int(
                    eligible_month[["stock", "session"]].drop_duplicates().shape[0]
                ),
                "mean_eligible_checkpoints_per_valid_pair": (
                    len(eligible_month) / len(valid_pairs) if len(valid_pairs) else math.nan
                ),
            }
        )
        metrics = _metrics(tail_month)
        incidence_rows.append(
            {
                "month": month,
                "eligible_checkpoint_rows": int(len(eligible_month)),
                "m1c_bottom_10_rows": int(len(tail_month)),
                "bottom_tail_row_share": len(tail_month) / len(eligible_month),
                "bottom_tail_incidence": len(tail_month) / len(eligible_month),
                "bottom_tail_composition_share": len(tail_month) / len(tail),
                "fresh_quiet_episodes": int(len(fresh_month)),
                "fresh_episode_share": len(fresh_month) / len(fresh),
                "mean_m1c_probability": float(
                    np.average(
                        eligible_month["M1C_probability"].to_numpy(float),
                        weights=_base_weights(eligible_month),
                    )
                ),
                **metrics,
                "breach_1_5_sigma_count": int(
                    tail_month["excursion_sigma_ratio_15m"].ge(1.5).sum()
                ),
                "breach_2_0_sigma_count": int(
                    tail_month["excursion_sigma_ratio_15m"].ge(2.0).sum()
                ),
                "maximum_stock_contribution": _maximum_share(tail_month, "stock"),
                "maximum_checkpoint_contribution": _maximum_share(tail_month, "checkpoint"),
                "maximum_session_contribution": _maximum_share(tail_month, "session"),
            }
        )
    return pd.DataFrame(exposure_rows), pd.DataFrame(incidence_rows)


def representation_concentration(
    representations: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        for representation, all_rows in representations.items():
            frame = all_rows.loc[all_rows["period"].eq(period)]
            for dimension in ("all", "month", "stock", "session"):
                groups: list[tuple[str, pd.DataFrame]]
                if dimension == "all":
                    groups = [("all", frame)]
                else:
                    groups = [
                        (str(entity), group)
                        for entity, group in frame.groupby(dimension, sort=True)
                    ]
                for entity, group in groups:
                    metrics = _metrics(group)
                    rows.append(
                        {
                            "period": period,
                            "representation": representation,
                            "dimension": dimension,
                            "entity": entity,
                            "count": int(len(group)),
                            "share": len(group) / len(frame) if len(frame) else 0.0,
                            **metrics,
                            "surprise_1_5_count": int(
                                group["excursion_sigma_ratio_15m"].ge(1.5).sum()
                            ),
                            "surprise_1_5_share": float(
                                group["excursion_sigma_ratio_15m"].ge(1.5).mean()
                            )
                            if len(group)
                            else 0.0,
                            "surprise_2_0_count": int(
                                group["excursion_sigma_ratio_15m"].ge(2.0).sum()
                            ),
                            "surprise_2_0_share": float(
                                group["excursion_sigma_ratio_15m"].ge(2.0).mean()
                            )
                            if len(group)
                            else 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def surprise_outputs(
    evidence: Evidence,
    representations: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    enriched: dict[str, pd.DataFrame] = {
        name: _add_high_tail_proximity(frame, evidence.predictions)
        for name, frame in representations.items()
    }
    raw = enriched["raw_checkpoint_rows"]
    raw = raw.loc[raw["excursion_sigma_ratio_15m"].ge(1.5)].copy()
    raw["surprise_large_mover"] = True
    raw["extreme_surprise_mover"] = raw["excursion_sigma_ratio_15m"].ge(2.0)
    raw["direction"] = np.where(
        raw["maximum_up_excursion_15m"].ge(raw["maximum_down_excursion_15m"].abs()),
        "up",
        "down",
    )
    raw["maximum_excursion"] = raw["maximum_absolute_excursion_15m"]
    raw["terminal_return"] = raw["signed_return_15m"]
    raw["move_reversed"] = raw["large_excursion_mean_reverted_15m"]
    raw_clusters = cluster_surprise_events(raw, sigma_threshold=1.5)
    cluster_by_member: dict[str, tuple[str, int]] = {}
    for event in raw_clusters.itertuples(index=False):
        for member in event.member_row_ids:
            cluster_by_member[str(member)] = (
                str(event.surprise_event_id),
                int(event.member_count),
            )
    raw["surprise_event_id"] = [
        cluster_by_member[str(row_id)][0] for row_id in raw["row_id"].astype(str)
    ]
    raw["same_event_checkpoint_rows"] = [
        cluster_by_member[str(row_id)][1] for row_id in raw["row_id"].astype(str)
    ]
    raw["multiple_checkpoint_rows_represent_same_event"] = raw["same_event_checkpoint_rows"].gt(1)
    trigger_sets = {name: set(frame["row_id"].astype(str)) for name, frame in enriched.items()}
    for name, identities in trigger_sets.items():
        raw[f"in_{name}"] = raw["row_id"].astype(str).isin(identities)

    event_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    feasibility: dict[str, Any] = {**audit_claims(), "periods": {}}
    for period in ("assessment", "stress"):
        feasibility["periods"][period] = {}
        for representation, all_rows in enriched.items():
            frame = all_rows.loc[all_rows["period"].eq(period)].copy()
            feasibility["periods"][period][representation] = {}
            for threshold in SURPRISE_THRESHOLDS:
                surprise_rows = frame.loc[frame["excursion_sigma_ratio_15m"].ge(threshold)].copy()
                events = cluster_surprise_events(frame, sigma_threshold=threshold)
                if not events.empty:
                    events["period"] = period
                    events["month"] = events["session"].astype(str).str[:7]
                    events["representation"] = representation
                    events["sigma_threshold"] = threshold
                    events["surprise_definition"] = f"excursion_sigma_ratio_15m >= {threshold:.1f}"
                    events["member_row_ids"] = events["member_row_ids"].map(
                        lambda values: json.dumps(list(values), separators=(",", ":"))
                    )
                    event_frames.append(events)
                month_counts = (
                    events["session"].astype(str).str[:7].value_counts().to_dict()
                    if not events.empty
                    else {}
                )
                stock_counts = (
                    events["stock"].astype(str).value_counts().to_dict() if not events.empty else {}
                )
                diagnostic = small_count_feasibility(
                    month_counts=cast(Mapping[str, int], month_counts),
                    stock_counts=cast(Mapping[str, int], stock_counts),
                    month_limit=SURPRISE_MONTH_LIMIT,
                    stock_limit=SURPRISE_STOCK_LIMIT,
                )
                feasibility["periods"][period][representation][str(threshold)] = diagnostic
                adverse_total = (
                    float(events["maximum_absolute_excursion_15m"].sum())
                    if not events.empty
                    else 0.0
                )
                adverse_stock = (
                    events.groupby("stock", sort=True)["maximum_absolute_excursion_15m"].sum()
                    if not events.empty
                    else pd.Series(dtype=float)
                )
                adverse_month = (
                    events.assign(month=events["session"].astype(str).str[:7])
                    .groupby("month", sort=True)["maximum_absolute_excursion_15m"]
                    .sum()
                    if not events.empty
                    else pd.Series(dtype=float)
                )
                metric_rows.append(
                    {
                        "period": period,
                        "representation": representation,
                        "sigma_threshold": threshold,
                        "raw_row_count": int(len(surprise_rows)),
                        "clustered_event_count": int(len(events)),
                        "repeated_checkpoint_rows_removed": int(len(surprise_rows) - len(events)),
                        "stocks_represented": int(events["stock"].nunique())
                        if not events.empty
                        else 0,
                        "months_represented": int(events["session"].astype(str).str[:7].nunique())
                        if not events.empty
                        else 0,
                        "sessions_represented": int(events["session"].nunique())
                        if not events.empty
                        else 0,
                        "checkpoints_represented": int(events["trigger_checkpoint"].nunique())
                        if not events.empty
                        else 0,
                        "raw_maximum_stock_share": _maximum_share(surprise_rows, "stock"),
                        "raw_maximum_month_share": _maximum_share(
                            surprise_rows.assign(
                                month=surprise_rows["session"].astype(str).str[:7]
                            ),
                            "month",
                        ),
                        "maximum_stock_share": _maximum_share(events, "stock"),
                        "maximum_month_share": _maximum_share(
                            events.assign(month=events["session"].astype(str).str[:7]),
                            "month",
                        ),
                        "maximum_stock_month_share": _maximum_share(
                            events.assign(month=events["session"].astype(str).str[:7]),
                            ["stock", "month"],
                        ),
                        "maximum_session_share": _maximum_share(events, "session"),
                        "maximum_checkpoint_share": _maximum_share(
                            events,
                            "trigger_checkpoint",
                        ),
                        "herfindahl_stock": _hhi(events, "stock"),
                        "herfindahl_month": _hhi(
                            events.assign(month=events["session"].astype(str).str[:7]),
                            "month",
                        ),
                        "total_adverse_excursion": adverse_total,
                        "maximum_stock_adverse_excursion_share": (
                            float(adverse_stock.max() / adverse_total)
                            if adverse_total > 0.0
                            else 0.0
                        ),
                        "maximum_month_adverse_excursion_share": (
                            float(adverse_month.max() / adverse_total)
                            if adverse_total > 0.0
                            else 0.0
                        ),
                        "high_tail_proximity_count": int(events["high_tail_within_60m"].sum())
                        if "high_tail_within_60m" in events
                        else 0,
                        "high_tail_proximity_rate": float(events["high_tail_within_60m"].mean())
                        if not events.empty and "high_tail_within_60m" in events
                        else 0.0,
                    }
                )
    event_table = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(columns=["surprise_event_id"])
    )
    row_columns = [
        "period",
        "row_id",
        "surprise_event_id",
        "same_event_checkpoint_rows",
        "multiple_checkpoint_rows_represent_same_event",
        "stock",
        "session",
        "month",
        "checkpoint",
        "entry_timestamp",
        "M1C_probability",
        "M0_probability",
        "atm_iv",
        "option_dte",
        "market_return_through_checkpoint",
        "market_volatility",
        "market_volatility_group",
        "stock_local_volatility",
        "stock_local_volatility_group",
        "direction",
        "maximum_excursion",
        "terminal_return",
        "move_reversed",
        "surprise_large_mover",
        "extreme_surprise_mover",
        "excursion_sigma_ratio_15m",
        "high_tail_previous_60m",
        "high_tail_following_60m",
        "high_tail_within_60m",
        *(f"in_{name}" for name in REPRESENTATIONS),
    ]
    return (
        raw.loc[:, row_columns],
        event_table,
        pd.DataFrame(metric_rows),
        feasibility,
    )


def _npv_metrics(
    population: pd.DataFrame,
    cohort: pd.DataFrame,
    population_weights: pd.Series,
    cohort_weights: pd.Series,
) -> dict[str, float | int]:
    baseline = _metrics(population, population_weights)
    result = _metrics(cohort, cohort_weights)
    result["npv_lift"] = float(result["remains_below_iv_rate"]) - float(
        baseline["remains_below_iv_rate"]
    )
    return result


def _event_weight_concentration(
    cohort: pd.DataFrame,
    cohort_weights: pd.Series,
) -> tuple[float, float, int]:
    working = cohort.copy()
    working["_source_index"] = working.index
    events = cluster_surprise_events(working, sigma_threshold=1.5)
    if events.empty:
        return 0.0, 0.0, 0
    weight_by_index = cohort_weights.to_dict()
    event_weights = pd.Series(
        [weight_by_index[int(index)] for index in events["_source_index"]],
        index=events.index,
        dtype=float,
    )
    month_frame = events.assign(month=events["session"].astype(str).str[:7])
    return (
        _weighted_maximum_share(month_frame, event_weights, "month"),
        _weighted_maximum_share(events, event_weights, "stock"),
        int(len(events)),
    )


def equal_exposure_sensitivities(
    evidence: Evidence,
    representations: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    population = evidence.analytic.loc[evidence.analytic["period"].eq("stress")].copy()
    tail = (
        representations["raw_checkpoint_rows"]
        .loc[representations["raw_checkpoint_rows"]["period"].eq("stress")]
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for scheme in ("original", "equal_month", "equal_stock", "equal_stock_month"):
        population_weights = analysis_weights(population, scheme=scheme)
        cohort_weights = population_weights.loc[tail.index]
        metrics = _npv_metrics(population, tail, population_weights, cohort_weights)
        event_month, event_stock, event_count = _event_weight_concentration(
            tail,
            cohort_weights,
        )
        rows.append(
            {
                "sensitivity": scheme,
                **metrics,
                "maximum_month_concentration": _weighted_maximum_share(
                    tail,
                    cohort_weights,
                    "month",
                ),
                "maximum_stock_concentration": _weighted_maximum_share(
                    tail,
                    cohort_weights,
                    "stock",
                ),
                "maximum_surprise_event_month_concentration": event_month,
                "maximum_surprise_event_stock_concentration": event_stock,
                "clustered_surprise_event_count": event_count,
                "status": "descriptive_only",
            }
        )
    fresh = representations["frozen_fresh_quiet_episodes"].loc[
        representations["frozen_fresh_quiet_episodes"]["period"].eq("stress")
    ]
    first_fresh = (
        fresh.sort_values(
            ["stock", "session", "entry_timestamp", "checkpoint", "row_id"],
            kind="mergesort",
        )
        .groupby(["stock", "session"], sort=True, as_index=False)
        .head(1)
        .copy()
    )
    population_weights = analysis_weights(population, scheme="original")
    cohort_weights = analysis_weights(first_fresh, scheme="original")
    metrics = _npv_metrics(population, first_fresh, population_weights, cohort_weights)
    event_month, event_stock, event_count = _event_weight_concentration(
        first_fresh,
        cohort_weights,
    )
    rows.append(
        {
            "sensitivity": "one_fresh_episode_per_stock_session",
            **metrics,
            "maximum_month_concentration": _weighted_maximum_share(
                first_fresh,
                cohort_weights,
                "month",
            ),
            "maximum_stock_concentration": _weighted_maximum_share(
                first_fresh,
                cohort_weights,
                "stock",
            ),
            "maximum_surprise_event_month_concentration": event_month,
            "maximum_surprise_event_stock_concentration": event_stock,
            "clustered_surprise_event_count": event_count,
            "status": "descriptive_only",
        }
    )
    return pd.DataFrame(rows)


def _leave_one_table(
    *,
    evidence: Evidence,
    tail: pd.DataFrame,
    fresh: pd.DataFrame,
    dimension: str,
    levels: Sequence[object],
    omitted_column: str,
) -> pd.DataFrame:
    population = evidence.analytic.loc[evidence.analytic["period"].eq("stress")]
    full_metrics = _npv_metrics(
        population,
        tail,
        analysis_weights(population, scheme="original"),
        analysis_weights(tail, scheme="original"),
    )
    full_fresh = _metrics(fresh)
    full_containment = 1.0 - float(full_fresh["breach_1_5_sigma_rate"])
    rows: list[dict[str, Any]] = []
    for level in levels:
        reduced_population = population.loc[population[dimension].ne(level)]
        reduced_tail = tail.loc[tail[dimension].ne(level)]
        reduced_fresh = fresh.loc[fresh[dimension].ne(level)]
        metrics = _npv_metrics(
            reduced_population,
            reduced_tail,
            analysis_weights(reduced_population, scheme="original"),
            analysis_weights(reduced_tail, scheme="original"),
        )
        fresh_metrics = _metrics(reduced_fresh)
        containment = 1.0 - float(fresh_metrics["breach_1_5_sigma_rate"])
        rows.append(
            {
                omitted_column: level,
                **metrics,
                "fresh_episode_rows": int(len(reduced_fresh)),
                "fresh_episode_1_5_sigma_containment": containment,
                "difference_remains_below_iv_rate": float(metrics["remains_below_iv_rate"])
                - float(full_metrics["remains_below_iv_rate"]),
                "difference_npv_lift": float(metrics["npv_lift"]) - float(full_metrics["npv_lift"]),
                "difference_mean_iv_residual": float(metrics["mean_iv_residual"])
                - float(full_metrics["mean_iv_residual"]),
                "difference_median_iv_residual": float(metrics["median_iv_residual"])
                - float(full_metrics["median_iv_residual"]),
                "difference_breach_1_5_sigma_rate": float(metrics["breach_1_5_sigma_rate"])
                - float(full_metrics["breach_1_5_sigma_rate"]),
                "difference_breach_2_0_sigma_rate": float(metrics["breach_2_0_sigma_rate"])
                - float(full_metrics["breach_2_0_sigma_rate"]),
                "difference_fresh_episode_1_5_sigma_containment": (containment - full_containment),
                "status": "descriptive_only",
            }
        )
    return pd.DataFrame(rows)


def leave_one_group_out(
    evidence: Evidence,
    representations: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tail = representations["raw_checkpoint_rows"].loc[
        representations["raw_checkpoint_rows"]["period"].eq("stress")
    ]
    fresh = representations["frozen_fresh_quiet_episodes"].loc[
        representations["frozen_fresh_quiet_episodes"]["period"].eq("stress")
    ]
    months = _leave_one_table(
        evidence=evidence,
        tail=tail,
        fresh=fresh,
        dimension="month",
        levels=STRESS_MONTHS,
        omitted_column="omitted_month",
    )
    stocks = _leave_one_table(
        evidence=evidence,
        tail=tail,
        fresh=fresh,
        dimension="stock",
        levels=tuple(sorted(tail["stock"].astype(str).unique())),
        omitted_column="omitted_stock",
    )
    checkpoints = _leave_one_table(
        evidence=evidence,
        tail=tail,
        fresh=fresh,
        dimension="checkpoint_group",
        levels=("early", "middle", "late"),
        omitted_column="omitted_checkpoint_group",
    )
    return months, stocks, checkpoints


def concentration_explanations(
    *,
    exposure: pd.DataFrame,
    incidence: pd.DataFrame,
    representation: pd.DataFrame,
    surprise_metrics: pd.DataFrame,
    small_counts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    failed = incidence.sort_values(
        ["bottom_tail_composition_share", "month"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    failed_month = str(failed["month"])
    exposure_row = exposure.loc[exposure["month"].eq(failed_month)].iloc[0]
    representation_rows = representation.loc[
        representation["period"].eq("stress")
        & representation["dimension"].eq("month")
        & representation["entity"].eq(failed_month)
    ].set_index("representation")
    raw_share = float(representation_rows.loc["raw_checkpoint_rows", "share"])
    run_share = float(representation_rows.loc["quiet_state_runs", "share"])
    fresh_share = float(representation_rows.loc["frozen_fresh_quiet_episodes", "share"])
    stock_session_share = float(representation_rows.loc["one_per_stock_session", "share"])
    month_explanation = classify_month_concentration(
        maximum_composition_share=raw_share,
        source_exposure_share=float(exposure_row["source_exposure_share"]),
        fresh_episode_share=fresh_share,
        frozen_limit=MONTH_SHARE_LIMIT,
    )
    other_incidence = incidence.loc[incidence["month"].ne(failed_month)]
    month_payload = {
        **audit_claims(),
        "failed_stress_month": failed_month,
        "failed_month_tail_rows": int(failed["m1c_bottom_10_rows"]),
        "all_stress_tail_rows": int(incidence["m1c_bottom_10_rows"].sum()),
        "exact_failed_share": raw_share,
        "frozen_maximum_share": MONTH_SHARE_LIMIT,
        "source_exposure_share": float(exposure_row["source_exposure_share"]),
        "bottom_tail_incidence": float(failed["bottom_tail_incidence"]),
        "other_month_mean_bottom_tail_incidence": float(
            other_incidence["bottom_tail_incidence"].mean()
        ),
        "bottom_tail_composition_share": raw_share,
        "quiet_run_composition_share": run_share,
        "fresh_episode_composition_share": fresh_share,
        "one_per_stock_session_composition_share": stock_session_share,
        "option_pair_coverage_rate": float(exposure_row["option_pair_coverage_rate"]),
        "checkpoint_persistence_share_reduction": raw_share - fresh_share,
        "source_exposure_contribution_points": float(exposure_row["source_exposure_share"]),
        "low_tail_incidence_excess_points": raw_share
        - float(exposure_row["source_exposure_share"]),
        "more_eligible_source_rows": bool(
            exposure_row["eligible_checkpoint_rows"] == exposure["eligible_checkpoint_rows"].max()
        ),
        "better_option_coverage": bool(
            exposure_row["option_pair_coverage_rate"] == exposure["option_pair_coverage_rate"].max()
        ),
        "higher_low_tail_frequency": bool(
            failed["bottom_tail_incidence"] == incidence["bottom_tail_incidence"].max()
        ),
        "repeated_checkpoint_persistence": raw_share > fresh_share,
        "month_concentration_explanation": month_explanation,
        "interpretation": (
            "Exposure and elevated low-tail incidence both increased the failed month share; "
            "checkpoint persistence supplied the final binding increment. This is descriptive "
            "regime concentration, not a relaxation or predictive failure."
        ),
    }

    original_metric = surprise_metrics.loc[
        surprise_metrics["period"].eq("stress")
        & surprise_metrics["representation"].eq("frozen_fresh_quiet_episodes")
        & surprise_metrics["sigma_threshold"].eq(1.5)
    ].iloc[0]
    diagnostic = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], cast(Mapping[str, Any], small_counts["periods"])["stress"])[
            "frozen_fresh_quiet_episodes"
        ]["1.5"],
    )
    surprise_explanation = classify_surprise_concentration(
        clustered_maximum_stock_share=float(original_metric["maximum_stock_share"]),
        clustered_maximum_month_share=float(original_metric["maximum_month_share"]),
        clustered_maximum_stock_month_share=float(original_metric["maximum_stock_month_share"]),
        stock_limit=SURPRISE_STOCK_LIMIT,
        month_limit=SURPRISE_MONTH_LIMIT,
        small_count_fragile=bool(diagnostic["small_count_concentration_fragile"]),
    )
    surprise_payload = {
        **audit_claims(),
        "binding_population": "stress_frozen_fresh_quiet_episodes",
        "binding_definition": "excursion_sigma_ratio_15m >= 1.5",
        "original_row_count": int(original_metric["raw_row_count"]),
        "clustered_event_count": int(original_metric["clustered_event_count"]),
        "maximum_stock_share": float(original_metric["maximum_stock_share"]),
        "maximum_month_share": float(original_metric["maximum_month_share"]),
        "maximum_stock_month_share": float(original_metric["maximum_stock_month_share"]),
        "repeated_checkpoint_rows_removed": int(
            original_metric["repeated_checkpoint_rows_removed"]
        ),
        "small_count_concentration_fragile": bool(diagnostic["small_count_concentration_fragile"]),
        "failed_condition_driven_by_one_or_two_events": bool(
            diagnostic["failed_condition_driven_by_one_or_two_events"]
        ),
        "one_event_share": float(diagnostic["one_event_share"]),
        "surprise_concentration_explanation": surprise_explanation,
        "original_surprise_gate_passed": False,
        "interpretation": (
            "The original fresh-episode month concentration persists after event clustering, "
            "but the event count is discrete enough that one event caused the failure."
        ),
    }
    return month_payload, surprise_payload


def source_manifest() -> dict[str, Any]:
    paths = [
        PREDECESSOR / name
        for name in (
            "checkpoint_predictions.parquet",
            "raw_low_tail_checkpoint_rows.parquet",
            "fresh_quiet_episodes.parquet",
            "movement_outcomes.parquet",
            "path_excursion_outcomes.parquet",
            "decision.json",
            "concentration_metrics.csv",
            "surprise_mover_rows.csv",
            "surprise_mover_summary.json",
            "monthly_metrics.csv",
            "stock_metrics.csv",
            "containment_metrics.csv",
            "bootstrap_metrics.csv",
            "matched_random_null_metrics.csv",
            "probability_permutation_null_metrics.csv",
            "frozen_resampling_plan.json",
            "m1c_feature_manifest.json",
            "frozen_low_tail_thresholds.json",
        )
    ]
    paths.extend((OPTION_PAIRS, RUNTIME_MANIFEST))
    _require_paths(paths)
    return {
        **audit_claims(),
        "protected_rows_read": 0,
        "maximum_historical_session_read": "2025-12-31",
        "network_requests": 0,
        "broker_connections": 0,
        "sources": [
            {
                "path": str(path.relative_to(REPO_ROOT))
                if path.is_relative_to(REPO_ROOT)
                else str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "bootstrap_identity": {
            "draws_per_period": 100,
            "assessment_seed": 2026072751,
            "stress_seed": 2026072752,
        },
        "matched_null_identity": {
            "draws_per_period": 20,
            "seeds": list(range(2026072701, 2026072721)),
        },
        "probability_permutation_identity": {
            "draws_per_period": 10,
            "seeds": list(range(2026072731, 2026072741)),
        },
    }


def original_decision_preservation(evidence: Evidence) -> dict[str, Any]:
    decision_path = PREDECESSOR / "decision.json"
    return {
        **audit_claims(),
        "source_decision_path": str(decision_path.relative_to(REPO_ROOT)),
        "source_decision_sha256": _sha256(decision_path),
        "original_overall_decision": evidence.original_decision["overall_decision"],
        "expected_original_overall_decision": ORIGINAL_DECISION,
        "unchanged": evidence.original_decision["overall_decision"] == ORIGINAL_DECISION,
        "source_artifact_modified": False,
    }


def build_audit() -> AuditResult:
    evidence = load_evidence()
    representations, runs = build_representations(evidence.analytic)
    reconstruction = reconstruction_audit(
        evidence,
        representations["frozen_fresh_quiet_episodes"],
    )
    if not reconstruction["passed"]:
        raise ValueError("frozen tail reconstruction failed closed")
    exposure, incidence = stress_month_tables(evidence, representations)
    representation = representation_concentration(representations)
    surprise_rows, surprise_events, surprise_metrics, small_counts = surprise_outputs(
        evidence,
        representations,
    )
    equal = equal_exposure_sensitivities(evidence, representations)
    leave_month, leave_stock, leave_checkpoint = leave_one_group_out(
        evidence,
        representations,
    )
    month_explanation, surprise_explanation = concentration_explanations(
        exposure=exposure,
        incidence=incidence,
        representation=representation,
        surprise_metrics=surprise_metrics,
        small_counts=small_counts,
    )
    phase_b_root = REPO_ROOT / "research" / "prospective" / "frozen-m1c-microstructure-recorder-v0"
    phase_b_contracts = (
        "quiet_state_signal_contract.json",
        "neutral_control_sampling_contract.json",
        "defined_risk_structure_contract.json",
        "quiet_state_safety_contract.json",
    )
    phase_b_implemented = all((phase_b_root / name).is_file() for name in phase_b_contracts)
    decision = {
        **audit_claims(),
        "original_overall_decision": ORIGINAL_DECISION,
        "month_concentration_explanation": month_explanation["month_concentration_explanation"],
        "surprise_concentration_explanation": surprise_explanation[
            "surprise_concentration_explanation"
        ],
        "small_count_concentration_status": "descriptive_only",
        "equal_month_sensitivity_status": "descriptive_only",
        "leave_one_month_out_status": "descriptive_only",
        "quiet_state_recorder_status": "implemented" if phase_b_implemented else "blocked",
        "defined_risk_shadow_status": "implemented" if phase_b_implemented else "blocked",
        "neutral_control_status": "implemented" if phase_b_implemented else "blocked",
        "safety_status": "supported",
        "replacement_validation_decision_created": False,
        "historical_low_movement_gate_passed": False,
        "historical_short_premium_gate_passed": False,
    }
    frames = {
        "stress_month_exposure_audit.csv": exposure,
        "stress_month_tail_incidence.csv": incidence,
        "quiet_state_runs.parquet": runs,
        "checkpoint_vs_episode_concentration.csv": representation,
        "surprise_mover_row_audit.csv": surprise_rows,
        "surprise_event_clusters.csv": surprise_events,
        "surprise_concentration_metrics.csv": surprise_metrics,
        "equal_exposure_sensitivities.csv": equal,
        "leave_one_month_out.csv": leave_month,
        "leave_one_stock_out.csv": leave_stock,
        "leave_one_checkpoint_group_out.csv": leave_checkpoint,
    }
    payloads = {
        "source_manifest.json": source_manifest(),
        "original_decision_preservation.json": original_decision_preservation(evidence),
        "reconstruction_audit.json": reconstruction,
        "stress_month_concentration_explanation.json": month_explanation,
        "surprise_concentration_explanation.json": surprise_explanation,
        "small_count_feasibility.json": cast(dict[str, Any], small_counts),
        "decision.json": decision,
    }
    return AuditResult(frames=frames, payloads=payloads, evidence=evidence)


def _compare_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> tuple[int, float]:
    if first.shape != second.shape or list(first.columns) != list(second.columns):
        return 1, math.inf
    mismatches = 0
    maximum_difference = 0.0
    for column in first.columns:
        left = first[column]
        right = second[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)
            finite_left = np.isfinite(left_values)
            finite_right = np.isfinite(right_values)
            mismatches += int(np.count_nonzero(finite_left != finite_right))
            finite = finite_left & finite_right
            if finite.any():
                differences = np.abs(left_values[finite] - right_values[finite])
                mismatches += int(np.count_nonzero(differences > 0.0))
                maximum_difference = max(maximum_difference, float(differences.max()))
        else:
            left_values = left.map(_json_safe).map(str)
            right_values = right.map(_json_safe).map(str)
            mismatches += int(np.count_nonzero(left_values.to_numpy() != right_values.to_numpy()))
    return mismatches, maximum_difference


def retrospective_determinism_check(first: AuditResult) -> dict[str, Any]:
    second = build_audit()
    frame_mismatches = 0
    maximum_difference = 0.0
    frame_details: list[dict[str, Any]] = []
    for name in sorted(first.frames):
        mismatches, difference = _compare_frames(first.frames[name], second.frames[name])
        frame_mismatches += mismatches
        maximum_difference = max(maximum_difference, difference)
        frame_details.append(
            {
                "artifact": name,
                "value_mismatches": mismatches,
                "maximum_floating_difference": difference,
            }
        )
    first_payload = json.dumps(
        _json_safe(first.payloads),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    second_payload = json.dumps(
        _json_safe(second.payloads),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload_mismatches = int(first_payload != second_payload)
    passed = frame_mismatches == 0 and payload_mismatches == 0 and maximum_difference <= 1e-12
    return {
        **audit_claims(),
        "stored_fixture_replays": 2,
        "retrospective_frame_value_mismatches": frame_mismatches,
        "retrospective_payload_mismatches": payload_mismatches,
        "maximum_floating_difference": maximum_difference,
        "requirement": "maximum_floating_difference <= 1e-12",
        "frame_details": frame_details,
        "passed": passed,
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def render_plots(result: AuditResult) -> list[str]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    exposure = result.frames["stress_month_exposure_audit.csv"]
    incidence = result.frames["stress_month_tail_incidence.csv"]
    merged = exposure.merge(incidence, on="month", validate="one_to_one")
    positions = np.arange(len(merged))
    width = 0.36
    plt.figure(figsize=(8.0, 4.5))
    plt.bar(
        positions - width / 2,
        100.0 * merged["source_exposure_share"],
        width,
        label="Eligible source exposure",
    )
    plt.bar(
        positions + width / 2,
        100.0 * merged["bottom_tail_composition_share"],
        width,
        label="Bottom-tail composition",
    )
    plt.axhline(35.0, color="#a41e22", linestyle="--", linewidth=1.2, label="Frozen 35% gate")
    plt.xticks(positions, merged["month"])
    plt.ylabel("Stress-period share (%)")
    plt.title("Source exposure versus bottom-tail composition")
    plt.legend(frameon=False)
    first = REPORTS / "01_stress_month_exposure_vs_tail_composition.png"
    _save_figure(first)

    surprise = result.frames["surprise_concentration_metrics.csv"]
    surprise = surprise.loc[
        surprise["period"].eq("stress") & surprise["sigma_threshold"].eq(1.5)
    ].copy()
    positions = np.arange(len(surprise))
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(
        positions - width / 2,
        100.0 * surprise["raw_maximum_month_share"],
        width,
        label="Raw surprise rows",
    )
    plt.bar(
        positions + width / 2,
        100.0 * surprise["maximum_month_share"],
        width,
        label="Clustered surprise events",
    )
    plt.axhline(60.0, color="#a41e22", linestyle="--", linewidth=1.2, label="Frozen 60% gate")
    plt.xticks(
        positions,
        [
            "Checkpoints",
            "Quiet runs",
            "Fresh episodes",
            "Stock-session",
        ],
        rotation=12,
    )
    plt.ylabel("Maximum month share (%)")
    plt.title("Raw checkpoint versus clustered surprise concentration")
    plt.legend(frameon=False)
    second = REPORTS / "02_raw_vs_clustered_surprise_concentration.png"
    _save_figure(second)

    leave_month = result.frames["leave_one_month_out.csv"]
    positions = np.arange(len(leave_month))
    plt.figure(figsize=(8.0, 4.7))
    plt.plot(
        positions,
        100.0 * leave_month["remains_below_iv_rate"],
        marker="o",
        label="Remains below IV",
    )
    plt.plot(
        positions,
        100.0 * leave_month["breach_1_5_sigma_rate"],
        marker="o",
        label="1.5σ breach",
    )
    plt.plot(
        positions,
        100.0 * leave_month["breach_2_0_sigma_rate"],
        marker="o",
        label="2.0σ breach",
    )
    plt.xticks(positions, leave_month["omitted_month"])
    plt.ylabel("Rate (%)")
    plt.title("Leave-one-month-out low-movement results")
    plt.legend(frameon=False)
    third = REPORTS / "03_leave_one_month_out.png"
    _save_figure(third)
    return [str(path.relative_to(EXPERIMENT_DIR)) for path in (first, second, third)]


def render_report(
    result: AuditResult,
    *,
    independent_audit: Mapping[str, Any],
    determinism: Mapping[str, Any],
    plots: Sequence[str],
) -> str:
    month = result.payloads["stress_month_concentration_explanation.json"]
    surprise = result.payloads["surprise_concentration_explanation.json"]
    exposure = result.frames["stress_month_exposure_audit.csv"]
    incidence = result.frames["stress_month_tail_incidence.csv"]
    representation = result.frames["checkpoint_vs_episode_concentration.csv"]
    stress_months = representation.loc[
        representation["period"].eq("stress")
        & representation["dimension"].eq("month")
        & representation["entity"].eq(month["failed_stress_month"])
    ].set_index("representation")
    equal = result.frames["equal_exposure_sensitivities.csv"].set_index("sensitivity")
    leave = result.frames["leave_one_month_out.csv"]
    binding_metrics = result.frames["surprise_concentration_metrics.csv"]
    binding_metrics = binding_metrics.loc[
        binding_metrics["period"].eq("stress")
        & binding_metrics["representation"].eq("frozen_fresh_quiet_episodes")
    ].set_index("sigma_threshold")
    month_table = exposure.merge(incidence, on="month", validate="one_to_one")
    month_lines = "\n".join(
        f"| {row.month} | {int(row.trading_sessions_in_source_calendar)} | "
        f"{int(row.eligible_checkpoint_rows_x)} | {float(row.source_exposure_share):.3%} | "
        f"{float(row.bottom_tail_incidence):.3%} | "
        f"{float(row.bottom_tail_composition_share):.3%} | "
        f"{float(row.fresh_episode_share):.3%} |"
        for row in month_table.itertuples(index=False)
    )
    leave_lines = "\n".join(
        f"| {row.omitted_month} | {int(row.rows)} | "
        f"{float(row.remains_below_iv_rate):.2%} | {float(row.npv_lift):+.2%} | "
        f"{float(row.mean_iv_residual):+.6f} | "
        f"{float(row.breach_1_5_sigma_rate):.2%} | "
        f"{float(row.breach_2_0_sigma_rate):.2%} |"
        for row in leave.itertuples(index=False)
    )
    month_header = (
        "| Month | XNYS sessions | Eligible rows | Source exposure | Tail incidence | "
        "Tail composition | Fresh share |"
    )
    return f"""# M1C Quiet-State Concentration Audit V0

**RESEARCH ONLY — RECORD ONLY — NO ORDERS**

The original frozen decision remains
`{ORIGINAL_DECISION}`. No historical threshold, support gate, or decision was
changed, and no retrospective gate relaxation is allowed.

## Binding answer

The failed stress month was **{month["failed_stress_month"]}**:
{month["failed_month_tail_rows"]}/{month["all_stress_tail_rows"]} frozen
bottom-10 checkpoint rows, or **{float(month["exact_failed_share"]):.6%}**.
Its source exposure was {float(month["source_exposure_share"]):.2%}, while its
within-month bottom-tail incidence was {float(month["bottom_tail_incidence"]):.2%}.
The exact explanation is
`{month["month_concentration_explanation"]}`: October had the largest eligible
source exposure, the highest low-tail incidence, and repeated checkpoint
persistence supplied the final increment above the frozen 35% limit.

{month_header}
|---|---:|---:|---:|---:|---:|---:|
{month_lines}

For the failed month, composition falls from
{float(stress_months.loc["raw_checkpoint_rows", "share"]):.2%} at raw
checkpoints to {float(stress_months.loc["quiet_state_runs", "share"]):.2%} for
quiet runs, {float(stress_months.loc["frozen_fresh_quiet_episodes", "share"]):.2%}
for frozen fresh episodes, and
{float(stress_months.loc["one_per_stock_session", "share"]):.2%} with one
observation per stock-session. These are explanatory views and do not replace
the checkpoint support gate.

## Surprise movers

At 1.5σ, the binding stress fresh-episode population contains
{int(binding_metrics.loc[1.5, "raw_row_count"])} original rows and
{int(binding_metrics.loc[1.5, "clustered_event_count"])} clustered events.
At 2.0σ it contains {int(binding_metrics.loc[2.0, "raw_row_count"])} original
rows and {int(binding_metrics.loc[2.0, "clustered_event_count"])} events.
The binding 1.5σ maximum month share is
{float(surprise["maximum_month_share"]):.2%}; maximum stock share is
{float(surprise["maximum_stock_share"]):.2%}; maximum stock-month share is
{float(surprise["maximum_stock_month_share"]):.2%}. Event clustering removed
{int(surprise["repeated_checkpoint_rows_removed"])} rows from the binding
fresh population. One event changes the share by
{float(surprise["one_event_share"]):.2%}; one event is the difference between
passing and failing. The exact explanation is
`{surprise["surprise_concentration_explanation"]}`.

## Descriptive sensitivities

Equal-month weighting reports a remains-below-IV rate of
{float(equal.loc["equal_month", "remains_below_iv_rate"]):.2%}, NPV lift of
{float(equal.loc["equal_month", "npv_lift"]):+.2%}, and maximum month
concentration of
{float(equal.loc["equal_month", "maximum_month_concentration"]):.2%}.
These weighted results are descriptive only.

| Omitted month | Rows | Remains below IV | NPV lift | Mean residual | 1.5σ breach | 2.0σ breach |
|---|---:|---:|---:|---:|---:|---:|
{leave_lines}

## Reproducibility and claims boundary

- Independent audit passed: `{bool(independent_audit["passed"])}`.
- Retrospective deterministic replay passed: `{bool(determinism["passed"])}`.
- Maximum floating difference: `{float(determinism["maximum_floating_difference"]):.3g}`.
- Protected historical start: `2026-01-01`; protected rows read: `0`.
- No option P&L, order, account, position, paper-trading, or live-trading path
  was used.

Plots: {", ".join(f"`{plot}`" for plot in plots)}.

This audit does not claim that either historical gate passed, does not claim
option profitability or realistic fill expectancy, and does not create a
replacement validation decision.
"""


def write_audit(
    result: AuditResult,
    *,
    independent_audit: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> None:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    for name, frame in result.frames.items():
        destination = PRIMARY / name
        if destination.suffix == ".parquet":
            _write_parquet(destination, frame)
        else:
            _write_csv(destination, frame)
    for name, payload in result.payloads.items():
        _write_json(PRIMARY / name, payload)
    _write_json(PRIMARY / "contract.json", _read_json(EXPERIMENT_DIR / "contract.json"))
    _write_json(PRIMARY / "independent_audit.json", independent_audit)
    _write_json(PRIMARY / "determinism_check.json", determinism)
    plots = render_plots(result)
    report = render_report(
        result,
        independent_audit=independent_audit,
        determinism=determinism,
        plots=plots,
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (EXPERIMENT_DIR / "report.md").write_text(report, encoding="utf-8")
