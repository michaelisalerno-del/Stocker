"""Research-only first-class profitable-move detector portability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-first-class-profitable-move-detectors-v1.json"
AUDITOR_PATH = HERE / "audit_first_class_profitable_move_detectors_v1.py"
TEST_PATH = HERE / "tests/test_first_class_profitable_move_detectors_v1.py"
RUNNER_PATH = Path(__file__).resolve()

DETECTORS = (
    "H1_downside_expansion_exhaustion_long",
    "H2_failed_breakdown_reclaim_long",
    "H3_two_bar_reversal_confirmation_long",
    "H4_opening_range_failed_breakdown_long",
    "H5_activity_activated_downside_expansion_long",
)
SOURCE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
SEGMENT_KEYS = ["period", "symbol_norm", "session_date", "segment_index"]
ROUND_TRIP_COST_BPS = 10.0
PRIMARY_COST_PER_SIDE_BPS = 5.0
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
SEED = 20260714
RISK_MIN_BPS = 20.0
RISK_MAX_BPS = 250.0
HORIZON_BARS = 24

EVENT_COLUMNS = [
    "event_id",
    "detector",
    "period",
    "symbol_norm",
    "session_date",
    "month",
    "decision_timestamp",
    "entry_timestamp",
    "segment_index",
    "decision_segment_position",
    "bar_ordinal",
    "clock_bucket",
    "entry_open",
    "invalidation_price",
    "risk_bps",
    "target_price",
    "prior_scale_bps",
    "bar_range_bps",
    "signed_body_bps",
    "close_location",
    "lower_wick_fraction",
    "return_3_bps",
    "prior_12_low",
    "opening_range_low",
    "relative_activity_6",
    "historical_volume_activity_proxy",
]

CONTROL_COLUMNS = [
    "control_id",
    "matched_event_id",
    "detector",
    "period",
    "symbol_norm",
    "session_date",
    "month",
    "decision_timestamp",
    "entry_timestamp",
    "segment_index",
    "decision_segment_position",
    "bar_ordinal",
    "clock_bucket",
    "entry_open",
    "risk_bps",
    "target_price",
    "stop_price",
    "prior_scale_bps",
]


def safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text())
    safety = contract["safety"]
    if not (
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["application_code_modification_allowed"] is False
        and safety["repository_write_allowed"] is False
    ):
        raise AssertionError("research safety contract drift")
    if contract["opened_data_status"]["validation_claim_allowed"] is not False:
        raise AssertionError("validation must remain forbidden")
    return contract


def provider_path(contract: dict[str, Any], symbol: str) -> Path:
    return (
        Path(contract["data"]["provider_root"])
        / f"symbol={symbol}"
        / "timeframe=5m"
        / "data.parquet"
    )


def period_symbols(contract: dict[str, Any], period: int) -> list[str]:
    return list(contract["data"][f"symbols_{period}"])


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "auditor": AUDITOR_PATH,
        "tests": TEST_PATH,
    }
    for index, report in enumerate(contract["provenance"]["origin_reports"], start=1):
        paths[f"origin_report_{index}"] = Path(report)
    for period in contract["evaluation"]["periods"]:
        for symbol in period_symbols(contract, int(period)):
            paths[f"provider_{period}_{symbol}"] = provider_path(contract, symbol)
    return paths


def environment_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
    }


def read_period_tape(
    contract: dict[str, Any], period: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    bounds = contract["data"]["period_bounds"][str(period)]
    lower = pd.Timestamp(bounds[0]).to_pydatetime()
    upper = pd.Timestamp(bounds[1]).to_pydatetime()
    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for symbol in period_symbols(contract, period):
        path = provider_path(contract, symbol)
        frame = pd.read_parquet(
            path,
            columns=list(SOURCE_COLUMNS),
            filters=[("timestamp", ">=", lower), ("timestamp", "<", upper)],
            engine="pyarrow",
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        duplicate_rows = int(frame["timestamp"].duplicated(keep=False).sum())
        if duplicate_rows:
            raise AssertionError(f"duplicate provider timestamps: {period} {symbol}")
        prices = frame[["open", "high", "low", "close"]].to_numpy(float)
        finite_positive = np.isfinite(prices).all(axis=1) & (prices > 0.0).all(axis=1)
        order_valid = (prices[:, 2] <= np.minimum(prices[:, 0], prices[:, 3])) & (
            np.maximum(prices[:, 0], prices[:, 3]) <= prices[:, 1]
        )
        valid_ohlc = finite_positive & order_valid
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        local_minutes = local.dt.hour * 60 + local.dt.minute
        regular = local_minutes.ge(570) & local_minutes.lt(960)
        on_grid = ((local_minutes - 570) % 5).eq(0) & local.dt.second.eq(0)
        accepted = valid_ohlc & regular.to_numpy(bool) & on_grid.to_numpy(bool)
        selected = frame.loc[accepted].copy()
        selected_local = selected["timestamp"].dt.tz_convert("America/New_York")
        selected_minutes = selected_local.dt.hour * 60 + selected_local.dt.minute
        selected["period"] = period
        selected["symbol_norm"] = symbol
        selected["session_date"] = selected_local.dt.strftime("%Y-%m-%d")
        selected["month"] = selected["session_date"].str.slice(0, 7)
        selected["bar_ordinal"] = ((selected_minutes - 570) // 5).astype(np.int16)
        selected = selected.sort_values("timestamp", kind="stable").reset_index(drop=True)
        gap = (
            selected.groupby("session_date", sort=False)["timestamp"]
            .diff()
            .ne(pd.Timedelta(minutes=5))
        )
        first = selected.groupby("session_date", sort=False).cumcount().eq(0)
        within_gap = gap & ~first
        selected["segment_index"] = (
            gap.groupby(selected["session_date"], sort=False).cumsum().astype(np.int16) - 1
        )
        selected["segment_position"] = (
            selected.groupby(["session_date", "segment_index"], sort=False)
            .cumcount()
            .astype(np.int16)
        )
        coverage.append(
            {
                "period": period,
                "symbol_norm": symbol,
                "raw_rows": int(len(frame)),
                "accepted_regular_rows": int(len(selected)),
                "sessions": int(selected["session_date"].nunique()),
                "invalid_ohlc_rows": int((~valid_ohlc).sum()),
                "outside_regular_rows": int((~regular).sum()),
                "off_grid_regular_rows": int((regular & ~on_grid).sum()),
                "within_session_gaps": int(within_gap.sum()),
                "first_timestamp": str(selected["timestamp"].min()),
                "last_timestamp": str(selected["timestamp"].max()),
            }
        )
        frames.append(selected)
    tape = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["period", "symbol_norm", "session_date", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    return tape, coverage


def _rolling_prior(frame: pd.DataFrame, column: str, window: int, operation: str) -> pd.Series:
    groups = frame.groupby(SEGMENT_KEYS, sort=False)[column]
    if operation == "median":
        return groups.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).median()
        )
    if operation == "min":
        return groups.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).min()
        )
    if operation == "mean":
        return groups.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
    raise ValueError(operation)


def add_causal_features(tape: pd.DataFrame) -> pd.DataFrame:
    frame = tape.copy()
    span = frame["high"] - frame["low"]
    valid_span = span.gt(0) & np.isfinite(span)
    frame["bar_range_bps"] = 10000.0 * span / frame["open"]
    frame["signed_body_bps"] = 10000.0 * (frame["close"] / frame["open"] - 1.0)
    frame["close_location"] = np.where(valid_span, (frame["close"] - frame["low"]) / span, np.nan)
    frame["lower_wick_fraction"] = np.where(
        valid_span,
        (np.minimum(frame["open"], frame["close"]) - frame["low"]) / span,
        np.nan,
    )
    grouped = frame.groupby(SEGMENT_KEYS, sort=False)
    prior_close = grouped["close"].shift(1)
    true_range = np.maximum.reduce(
        [
            (frame["high"] - frame["low"]).to_numpy(float),
            (frame["high"] - prior_close).abs().to_numpy(float),
            (frame["low"] - prior_close).abs().to_numpy(float),
        ]
    )
    frame["true_range_bps"] = 10000.0 * true_range / prior_close
    frame["prior_scale_bps"] = _rolling_prior(frame, "true_range_bps", 12, "median")
    close_lag3 = grouped["close"].shift(3)
    frame["return_3_bps"] = 10000.0 * (frame["close"] / close_lag3 - 1.0)
    frame["prior_12_low"] = _rolling_prior(frame, "low", 12, "min")

    positive_volume = frame["volume"].where(frame["volume"].gt(0) & np.isfinite(frame["volume"]))
    frame["log_volume"] = np.log(positive_volume)
    prior_log_volume = _rolling_prior(frame, "log_volume", 6, "mean")
    frame["relative_activity_6"] = np.exp(frame["log_volume"] - prior_log_volume)

    opening_rows = frame.loc[frame["bar_ordinal"].between(0, 5)].copy()
    opening = (
        opening_rows.groupby(["period", "symbol_norm", "session_date"], sort=False)
        .agg(
            opening_range_low=("low", "min"),
            opening_count=("bar_ordinal", "nunique"),
            opening_min_ordinal=("bar_ordinal", "min"),
            opening_max_ordinal=("bar_ordinal", "max"),
        )
        .reset_index()
    )
    opening_valid = (
        opening["opening_count"].eq(6)
        & opening["opening_min_ordinal"].eq(0)
        & opening["opening_max_ordinal"].eq(5)
    )
    opening.loc[~opening_valid, "opening_range_low"] = np.nan
    frame = frame.merge(
        opening[["period", "symbol_norm", "session_date", "opening_range_low"]],
        on=["period", "symbol_norm", "session_date"],
        how="left",
        validate="many_to_one",
    )

    grouped = frame.groupby(SEGMENT_KEYS, sort=False)
    frame["entry_open"] = grouped["open"].shift(-1)
    frame["entry_timestamp"] = grouped["timestamp"].shift(-1)
    frame["next_segment_position"] = grouped["segment_position"].shift(-1)
    frame["prior_bar_range_bps"] = grouped["bar_range_bps"].shift(1)
    frame["prior_bar_body_bps"] = grouped["signed_body_bps"].shift(1)
    frame["prior_bar_close_location"] = grouped["close_location"].shift(1)
    frame["prior_bar_scale_bps"] = grouped["prior_scale_bps"].shift(1)
    frame["prior_bar_midpoint"] = (grouped["high"].shift(1) + grouped["low"].shift(1)) / 2.0
    frame["prior_bar_low"] = grouped["low"].shift(1)
    frame["clock_bucket"] = (frame["bar_ordinal"] // 13).clip(0, 5).astype(np.int8)
    return frame.sort_values(
        ["period", "symbol_norm", "session_date", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def detector_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    scale = frame["prior_scale_bps"]
    h1 = (
        scale.gt(0)
        & frame["bar_range_bps"].div(scale).ge(1.25)
        & frame["signed_body_bps"].div(scale).le(-0.75)
        & frame["close_location"].le(0.20)
        & frame["return_3_bps"].div(scale).le(-1.00)
    )
    breach_12_bps = 10000.0 * (frame["prior_12_low"] - frame["low"]) / frame["close"]
    h2 = (
        scale.gt(0)
        & frame["prior_12_low"].gt(0)
        & breach_12_bps.ge(0.10 * scale)
        & frame["close"].gt(frame["prior_12_low"])
        & frame["lower_wick_fraction"].ge(0.40)
        & frame["return_3_bps"].lt(0)
    )
    prior_scale = frame["prior_bar_scale_bps"]
    prior_expansion = (
        prior_scale.gt(0)
        & frame["prior_bar_range_bps"].div(prior_scale).ge(1.25)
        & frame["prior_bar_body_bps"].div(prior_scale).le(-0.75)
        & frame["prior_bar_close_location"].le(0.20)
    )
    h3 = (
        prior_expansion
        & frame["close"].gt(frame["prior_bar_midpoint"])
        & frame["close"].gt(frame["open"])
        & frame["close_location"].ge(0.65)
    )
    opening_breach_bps = 10000.0 * (frame["opening_range_low"] - frame["low"]) / frame["close"]
    h4 = (
        scale.gt(0)
        & frame["bar_ordinal"].between(6, 35)
        & frame["opening_range_low"].gt(0)
        & opening_breach_bps.ge(0.10 * scale)
        & frame["close"].gt(frame["opening_range_low"])
        & frame["lower_wick_fraction"].ge(0.35)
        & frame["return_3_bps"].lt(0)
    )
    h5 = h1 & frame["relative_activity_6"].ge(1.10)
    return dict(zip(DETECTORS, (h1, h2, h3, h4, h5), strict=True))


def _cooldown_select(candidates: pd.DataFrame) -> pd.DataFrame:
    selected_indices: list[int] = []
    for _, group in candidates.groupby(
        ["period", "symbol_norm", "session_date", "detector"], sort=False
    ):
        last_position = -10_000
        for row in group.sort_values("segment_position", kind="stable").itertuples():
            position = int(row.segment_position)
            if position - last_position > HORIZON_BARS:
                selected_indices.append(int(row.Index))
                last_position = position
    return candidates.loc[selected_indices].sort_values(
        ["period", "detector", "symbol_norm", "timestamp"], kind="stable"
    )


def build_events(feature_tape: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    next_is_exact = feature_tape["next_segment_position"].eq(feature_tape["segment_position"] + 1)
    eligible = (
        feature_tape["bar_ordinal"].ge(12)
        & feature_tape["bar_ordinal"].le(53)
        & next_is_exact
        & feature_tape["entry_open"].gt(0)
        & feature_tape["prior_scale_bps"].gt(0)
    )
    masks = detector_masks(feature_tape)
    candidate_frames: list[pd.DataFrame] = []
    for detector, mask in masks.items():
        candidate = feature_tape.loc[eligible & mask].copy()
        candidate["detector"] = detector
        if detector == "H3_two_bar_reversal_confirmation_long":
            candidate["invalidation_price"] = np.minimum(
                candidate["low"], candidate["prior_bar_low"]
            )
        else:
            candidate["invalidation_price"] = candidate["low"]
        candidate["risk_bps"] = (
            10000.0
            * (candidate["entry_open"] - candidate["invalidation_price"])
            / candidate["entry_open"]
        )
        candidate = candidate.loc[candidate["risk_bps"].between(RISK_MIN_BPS, RISK_MAX_BPS)].copy()
        candidate_frames.append(candidate)
    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    selected = _cooldown_select(all_candidates).reset_index(drop=True)
    selected["target_price"] = selected["entry_open"] * (1.0 + selected["risk_bps"] / 10000.0)
    selected["event_id"] = selected.apply(
        lambda row: (
            f"{int(row['period'])}|{row['detector']}|{row['symbol_norm']}|"
            f"{pd.Timestamp(row['timestamp']).isoformat()}"
        ),
        axis=1,
    )
    selected["decision_timestamp"] = selected["timestamp"]
    selected["decision_segment_position"] = selected["segment_position"].astype(int)
    selected["historical_volume_activity_proxy"] = selected["volume"]
    events = selected[EVENT_COLUMNS].sort_values(
        ["period", "detector", "symbol_norm", "decision_timestamp"], kind="stable"
    )
    if events["event_id"].duplicated().any():
        raise AssertionError("duplicate event id")

    eligible_anchors = feature_tape.loc[eligible].copy()
    eligible_anchors["decision_timestamp"] = eligible_anchors["timestamp"]
    eligible_anchors["decision_segment_position"] = eligible_anchors["segment_position"].astype(int)
    return events.reset_index(drop=True), eligible_anchors.reset_index(drop=True)


def build_controls(events: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    controls: list[dict[str, Any]] = []
    for (period, detector), event_group in events.groupby(["period", "detector"], sort=True):
        period_anchors = anchors.loc[anchors["period"].eq(period)].copy()
        event_timestamps = set(
            event_group[["symbol_norm", "decision_timestamp"]].itertuples(index=False, name=None)
        )
        used: set[tuple[str, pd.Timestamp]] = set()
        pools = {
            key: group.copy()
            for key, group in period_anchors.groupby(
                ["symbol_norm", "month", "clock_bucket"], sort=False
            )
        }
        for event in event_group.sort_values(
            ["symbol_norm", "decision_timestamp"], kind="stable"
        ).itertuples(index=False):
            key = (event.symbol_norm, event.month, event.clock_bucket)
            pool = pools.get(key)
            if pool is None:
                raise AssertionError(f"missing control pool for {event.event_id}")
            pool_keys = list(zip(pool["symbol_norm"], pool["decision_timestamp"], strict=True))
            candidates = pool.loc[
                ~pool["session_date"].eq(event.session_date)
                & pd.Series(
                    [item not in event_timestamps for item in pool_keys],
                    index=pool.index,
                )
                & pd.Series([item not in used for item in pool_keys], index=pool.index)
            ].copy()
            if candidates.empty:
                raise AssertionError(f"unmatched event {event.event_id}")
            event_scale = max(float(event.prior_scale_bps), 1e-12)
            candidates["scale_distance"] = np.abs(
                np.log(candidates["prior_scale_bps"].clip(lower=1e-12) / event_scale)
            )
            event_id = event.event_id
            candidates["tie_hash"] = candidates["decision_timestamp"].map(
                lambda timestamp, frozen_event_id=event_id: stable_hash(
                    f"{frozen_event_id}|{pd.Timestamp(timestamp).isoformat()}"
                )
            )
            chosen = candidates.sort_values(["scale_distance", "tie_hash"], kind="stable").iloc[0]
            used_key = (str(chosen["symbol_norm"]), pd.Timestamp(chosen["decision_timestamp"]))
            used.add(used_key)
            risk_bps = float(event.risk_bps)
            entry_open = float(chosen["entry_open"])
            control_id = f"control|{event.event_id}"
            controls.append(
                {
                    "control_id": control_id,
                    "matched_event_id": event.event_id,
                    "detector": detector,
                    "period": int(period),
                    "symbol_norm": str(chosen["symbol_norm"]),
                    "session_date": str(chosen["session_date"]),
                    "month": str(chosen["month"]),
                    "decision_timestamp": chosen["decision_timestamp"],
                    "entry_timestamp": chosen["entry_timestamp"],
                    "segment_index": int(chosen["segment_index"]),
                    "decision_segment_position": int(chosen["decision_segment_position"]),
                    "bar_ordinal": int(chosen["bar_ordinal"]),
                    "clock_bucket": int(chosen["clock_bucket"]),
                    "entry_open": entry_open,
                    "risk_bps": risk_bps,
                    "target_price": entry_open * (1.0 + risk_bps / 10000.0),
                    "stop_price": entry_open * (1.0 - risk_bps / 10000.0),
                    "prior_scale_bps": float(chosen["prior_scale_bps"]),
                }
            )
    control_frame = pd.DataFrame(controls, columns=CONTROL_COLUMNS).sort_values(
        ["period", "detector", "symbol_norm", "decision_timestamp"], kind="stable"
    )
    if len(control_frame) != len(events):
        raise AssertionError("control cardinality drift")
    if control_frame["control_id"].duplicated().any():
        raise AssertionError("duplicate control id")
    return control_frame.reset_index(drop=True)


def prepare_events(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    contract = load_contract()
    all_tapes: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for period in contract["evaluation"]["periods"]:
        tape, coverage = read_period_tape(contract, int(period))
        all_tapes.append(tape)
        coverage_rows.extend(coverage)
    feature_tape = add_causal_features(pd.concat(all_tapes, ignore_index=True))
    events, anchors = build_events(feature_tape)
    controls = build_controls(events, anchors)
    events.to_parquet(out / "pre_outcome_events.parquet", index=False)
    controls.to_parquet(out / "pre_outcome_controls.parquet", index=False)

    coverage = pd.DataFrame(coverage_rows)
    eligible_counts = (
        anchors.groupby(["period", "symbol_norm"], sort=True)
        .size()
        .rename("eligible_anchors")
        .reset_index()
    )
    event_counts = (
        events.groupby(["period", "symbol_norm", "detector"], sort=True)
        .size()
        .rename("events")
        .reset_index()
    )
    coverage = coverage.merge(
        eligible_counts, on=["period", "symbol_norm"], how="left", validate="one_to_one"
    )
    coverage["eligible_anchors"] = coverage["eligible_anchors"].fillna(0).astype(int)
    coverage.to_csv(out / "data_coverage.csv", index=False)
    event_counts.to_csv(out / "event_counts.csv", index=False)

    paths = source_paths(contract)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen source paths: {missing}")
    source_hashes = {name: sha256(path) for name, path in sorted(paths.items())}
    frozen_files = {
        name: sha256(out / name)
        for name in (
            "pre_outcome_events.parquet",
            "pre_outcome_controls.parquet",
            "data_coverage.csv",
            "event_counts.csv",
        )
    }
    manifest = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scientific_status": contract["scientific_status"],
        "environment": environment_manifest(),
        "source_sha256": source_hashes,
        "frozen_file_sha256": frozen_files,
        "event_rows": int(len(events)),
        "control_rows": int(len(controls)),
        "event_rows_by_period_detector": event_counts.to_dict(orient="records"),
    }
    write_json(out / "pre_score_manifest.json", manifest)
    print(
        json.dumps(
            {
                "frozen": str(out),
                "events": len(events),
                "controls": len(controls),
            },
            indent=2,
        )
    )


def tape_groups(tape: pd.DataFrame) -> dict[tuple[int, str, str, int], pd.DataFrame]:
    return {
        (int(period), str(symbol), str(session), int(segment)): group.set_index(
            "segment_position", drop=False
        ).sort_index()
        for (period, symbol, session, segment), group in tape.groupby(SEGMENT_KEYS, sort=False)
    }


def score_path(
    row: Any,
    groups: dict[tuple[int, str, str, int], pd.DataFrame],
    *,
    id_column: str,
    stop_column: str,
) -> dict[str, Any]:
    key = (
        int(row.period),
        str(row.symbol_norm),
        str(row.session_date),
        int(row.segment_index),
    )
    group = groups.get(key)
    decision_position = int(row.decision_segment_position)
    identity = getattr(row, id_column)
    base = {
        id_column: identity,
        "detector": row.detector,
        "period": int(row.period),
        "symbol_norm": row.symbol_norm,
        "session_date": row.session_date,
        "month": row.month,
        "decision_timestamp": row.decision_timestamp,
        "entry_timestamp": row.entry_timestamp,
        "risk_bps": float(row.risk_bps),
    }
    if group is None:
        return {**base, "outcome_status": "missing_segment"}
    positions = np.arange(decision_position + 1, decision_position + HORIZON_BARS + 1)
    if not np.isin(positions, group.index.to_numpy()).all():
        return {**base, "outcome_status": "missing_exact_future_path"}
    future = group.loc[positions]
    if len(future) != HORIZON_BARS:
        return {**base, "outcome_status": "future_path_cardinality_error"}
    entry_open = float(row.entry_open)
    observed_entry = float(future.iloc[0]["open"])
    if not math.isclose(entry_open, observed_entry, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"entry replay mismatch: {identity}")
    target = float(row.target_price)
    stop = float(getattr(row, stop_column))

    hit_type = "no_touch_time_exit"
    hit_step: int | None = None
    exit_price = float(future.iloc[-1]["close"])
    pre_target_low = entry_open
    for step, bar in enumerate(future.itertuples(index=False), start=1):
        open_price = float(bar.open)
        high = float(bar.high)
        low = float(bar.low)
        pre_target_low = min(pre_target_low, low)
        if open_price >= target:
            hit_type = "target_gap_or_open_first"
            hit_step = step
            exit_price = target
            break
        if open_price <= stop:
            hit_type = "stop_gap_or_open_first"
            hit_step = step
            exit_price = stop
            break
        target_touch = high >= target
        stop_touch = low <= stop
        if target_touch and stop_touch:
            hit_type = "dual_touch_conservative_stop"
            hit_step = step
            exit_price = stop
            break
        if target_touch:
            hit_type = "target_first"
            hit_step = step
            exit_price = target
            break
        if stop_touch:
            hit_type = "stop_first"
            hit_step = step
            exit_price = stop
            break
    target_first = hit_type in {"target_gap_or_open_first", "target_first"}
    highs = future["high"].to_numpy(float)
    lows = future["low"].to_numpy(float)
    favorable_bps_path = 10000.0 * (highs / entry_open - 1.0)
    adverse_bps_path = 10000.0 * (1.0 - lows / entry_open)
    mfe_bps = float(np.max(favorable_bps_path))
    mae_bps = float(np.max(adverse_bps_path))
    time_to_mfe = int(np.argmax(favorable_bps_path) + 1)
    time_to_mae = int(np.argmax(adverse_bps_path) + 1)
    pre_target_mae_bps = 10000.0 * (1.0 - pre_target_low / entry_open)
    pre_target_mae_r = pre_target_mae_bps / float(row.risk_bps)
    dynamic_gross_bps = 10000.0 * (exit_price / entry_open - 1.0)
    fixed_gross_bps = 10000.0 * (float(future.iloc[-1]["close"]) / entry_open - 1.0)
    clean_success = bool(
        target_first and hit_step is not None and hit_step <= 6 and pre_target_mae_r <= 0.5
    )
    return {
        **base,
        "outcome_status": "scored",
        "entry_open": entry_open,
        "target_price": target,
        "stop_price": stop,
        "hit_type": hit_type,
        "hit_step": hit_step,
        "target_first": target_first,
        "rapid_target_3": bool(target_first and hit_step is not None and hit_step <= 3),
        "clean_success": clean_success,
        "pre_target_mae_r": pre_target_mae_r,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "mfe_r": mfe_bps / float(row.risk_bps),
        "mae_r": mae_bps / float(row.risk_bps),
        "time_to_mfe": time_to_mfe,
        "time_to_mae": time_to_mae,
        "dynamic_gross_bps": dynamic_gross_bps,
        "dynamic_net_bps": dynamic_gross_bps - ROUND_TRIP_COST_BPS,
        "fixed_h24_gross_bps": fixed_gross_bps,
        "fixed_h24_net_bps": fixed_gross_bps - ROUND_TRIP_COST_BPS,
    }


def score_ledgers(
    contract: dict[str, Any], events: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tapes: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for period in contract["evaluation"]["periods"]:
        tape, period_coverage = read_period_tape(contract, int(period))
        tapes.append(tape)
        coverage.extend(period_coverage)
    tape = pd.concat(tapes, ignore_index=True)
    groups = tape_groups(tape)
    event_rows = [
        score_path(
            row,
            groups,
            id_column="event_id",
            stop_column="invalidation_price",
        )
        for row in events.itertuples(index=False)
    ]
    control_rows = [
        score_path(
            row,
            groups,
            id_column="control_id",
            stop_column="stop_price",
        )
        for row in controls.itertuples(index=False)
    ]
    event_outcomes = pd.DataFrame(event_rows).sort_values(
        ["period", "detector", "symbol_norm", "decision_timestamp"], kind="stable"
    )
    control_outcomes = pd.DataFrame(control_rows).sort_values(
        ["period", "detector", "symbol_norm", "decision_timestamp"], kind="stable"
    )
    return (
        event_outcomes.reset_index(drop=True),
        control_outcomes.reset_index(drop=True),
        pd.DataFrame(coverage),
    )


def paired_outcomes(
    events: pd.DataFrame, controls: pd.DataFrame, control_ledger: pd.DataFrame
) -> pd.DataFrame:
    scored_events = events.loc[events["outcome_status"].eq("scored")].copy()
    scored_controls = controls.loc[controls["outcome_status"].eq("scored")].copy()
    control_ids = control_ledger[["control_id", "matched_event_id"]]
    scored_controls = scored_controls.merge(
        control_ids, on="control_id", how="left", validate="one_to_one"
    )
    control_keep = [
        "matched_event_id",
        "target_first",
        "dynamic_gross_bps",
        "dynamic_net_bps",
        "fixed_h24_net_bps",
        "mfe_bps",
        "mae_bps",
    ]
    paired = scored_events.merge(
        scored_controls[control_keep],
        left_on="event_id",
        right_on="matched_event_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_control"),
    )
    paired["precision_pair"] = paired["target_first"].astype(float)
    paired["precision_lift_pair"] = paired["target_first"].astype(float) - paired[
        "target_first_control"
    ].astype(float)
    paired["dynamic_net_difference_bps"] = (
        paired["dynamic_net_bps"] - paired["dynamic_net_bps_control"]
    )
    return paired


def moving_block_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    seed: int,
    null: float = 0.0,
) -> dict[str, Any]:
    daily = (
        frame.groupby("session_date", sort=True)[value_column].agg(["sum", "count"]).reset_index()
    )
    if daily.empty:
        return {
            "observed_mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_one_sided": None,
            "sessions": 0,
        }
    sums = daily["sum"].to_numpy(float)
    counts = daily["count"].to_numpy(float)
    sessions = len(daily)
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    blocks_needed = math.ceil(sessions / BOOTSTRAP_BLOCK)
    offsets = np.arange(BOOTSTRAP_BLOCK)
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, sessions, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % sessions).ravel()[:sessions]
        samples[draw] = sums[indices].sum() / counts[indices].sum()
    observed = float(frame[value_column].mean())
    return {
        "observed_mean": observed,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "p_one_sided": float((1 + np.sum(samples <= null)) / (BOOTSTRAP_DRAWS + 1)),
        "sessions": sessions,
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def build_metrics(
    paired: pd.DataFrame, event_outcomes: pd.DataFrame, coverage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_by_period = coverage.groupby("period")["eligible_anchors"].sum().to_dict()
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for detector_index, detector in enumerate(DETECTORS):
        for period_index, period in enumerate((2025, 2026)):
            group = paired.loc[paired["detector"].eq(detector) & paired["period"].eq(period)].copy()
            all_emitted = event_outcomes.loc[
                event_outcomes["detector"].eq(detector) & event_outcomes["period"].eq(period)
            ]
            if group.empty:
                continue
            event_precision = float(group["target_first"].mean())
            control_precision = float(group["target_first_control"].mean())
            metric_rows.append(
                {
                    "detector": detector,
                    "period": period,
                    "emitted_rows": int(len(all_emitted)),
                    "scored_paired_rows": int(len(group)),
                    "missing_event_paths": int((~all_emitted["outcome_status"].eq("scored")).sum()),
                    "sessions": int(group["session_date"].nunique()),
                    "stocks": int(group["symbol_norm"].nunique()),
                    "coverage_per_10000_anchors": float(
                        10000.0 * len(all_emitted) / eligible_by_period[int(period)]
                    ),
                    "target_first_precision": event_precision,
                    "control_precision": control_precision,
                    "precision_lift": event_precision - control_precision,
                    "rapid_target_3_rate": float(group["rapid_target_3"].mean()),
                    "clean_success_rate": float(group["clean_success"].mean()),
                    "dynamic_mean_net_bps": float(group["dynamic_net_bps"].mean()),
                    "control_dynamic_mean_net_bps": float(group["dynamic_net_bps_control"].mean()),
                    "paired_dynamic_difference_bps": float(
                        group["dynamic_net_difference_bps"].mean()
                    ),
                    "fixed_h24_mean_net_bps": float(group["fixed_h24_net_bps"].mean()),
                    "mean_risk_bps": float(group["risk_bps"].mean()),
                    "median_risk_bps": float(group["risk_bps"].median()),
                    "mean_mfe_r": float(group["mfe_r"].mean()),
                    "mean_mae_r": float(group["mae_r"].mean()),
                }
            )
            metrics = (
                ("precision", "precision_pair", 0.0),
                ("precision_lift", "precision_lift_pair", 0.0),
                ("dynamic_net", "dynamic_net_bps", 0.0),
            )
            for metric_index, (metric, column, null) in enumerate(metrics):
                result = moving_block_bootstrap(
                    group,
                    column,
                    seed=(SEED + detector_index * 1000 + period_index * 100 + metric_index),
                    null=null,
                )
                bootstrap_rows.append(
                    {
                        "family": metric,
                        "detector": detector,
                        "period": period,
                        "rows": int(len(group)),
                        "draws": BOOTSTRAP_DRAWS,
                        "block_sessions": BOOTSTRAP_BLOCK,
                        **result,
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows).sort_values(["detector", "period"], kind="stable")
    bootstrap_frame = pd.DataFrame(bootstrap_rows).sort_values(
        ["family", "detector", "period"], kind="stable"
    )
    bootstrap_frame["holm_adjusted_p"] = np.nan
    for family in ("precision_lift", "dynamic_net"):
        mask = bootstrap_frame["family"].eq(family)
        bootstrap_frame.loc[mask, "holm_adjusted_p"] = holm_adjust(
            bootstrap_frame.loc[mask, "p_one_sided"].to_numpy(float)
        )
    bootstrap_frame["passes_holm_0_05"] = bootstrap_frame["holm_adjusted_p"].le(0.05).fillna(False)
    return metrics_frame.reset_index(drop=True), bootstrap_frame.reset_index(drop=True)


def build_month_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (detector, period, month), group in paired.groupby(
        ["detector", "period", "month"], sort=True
    ):
        rows.append(
            {
                "detector": detector,
                "period": int(period),
                "month": month,
                "rows": int(len(group)),
                "target_first_precision": float(group["target_first"].mean()),
                "control_precision": float(group["target_first_control"].mean()),
                "precision_lift": float(group["precision_lift_pair"].mean()),
                "dynamic_mean_net_bps": float(group["dynamic_net_bps"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_stock_deletions(paired: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for period in (2025, 2026):
            base = paired.loc[paired["detector"].eq(detector) & paired["period"].eq(period)]
            for symbol in period_symbols(contract, period):
                group = base.loc[~base["symbol_norm"].eq(symbol)]
                rows.append(
                    {
                        "detector": detector,
                        "period": period,
                        "deleted_symbol": symbol,
                        "rows": int(len(group)),
                        "target_first_precision": float(group["target_first"].mean()),
                        "precision_lift": float(group["precision_lift_pair"].mean()),
                        "dynamic_mean_net_bps": float(group["dynamic_net_bps"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def build_cost_sensitivity(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (detector, period), group in paired.groupby(["detector", "period"], sort=True):
        for cost in (2.5, 5.0, 7.5, 10.0):
            rows.append(
                {
                    "detector": detector,
                    "period": int(period),
                    "cost_bps_per_side": cost,
                    "rows": int(len(group)),
                    "dynamic_mean_net_bps": float((group["dynamic_gross_bps"] - 2.0 * cost).mean()),
                    "control_dynamic_mean_net_bps": float(
                        (group["dynamic_gross_bps_control"] - 2.0 * cost).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_path_diagnostics(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (detector, period, hit_type), group in paired.groupby(
        ["detector", "period", "hit_type"], sort=True
    ):
        rows.append(
            {
                "detector": detector,
                "period": int(period),
                "hit_type": hit_type,
                "rows": int(len(group)),
                "share": float(
                    len(group)
                    / len(paired.loc[paired["detector"].eq(detector) & paired["period"].eq(period)])
                ),
                "mean_mfe_r": float(group["mfe_r"].mean()),
                "mean_mae_r": float(group["mae_r"].mean()),
                "mean_dynamic_net_bps": float(group["dynamic_net_bps"].mean()),
            }
        )
    return pd.DataFrame(rows)


def decide(
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    months: pd.DataFrame,
    deletions: pd.DataFrame,
    costs: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    detector_decisions: dict[str, Any] = {}
    for detector in DETECTORS:
        checks_by_period: dict[str, Any] = {}
        for period in (2025, 2026):
            metric_rows = metrics.loc[
                metrics["detector"].eq(detector) & metrics["period"].eq(period)
            ]
            if metric_rows.empty:
                checks_by_period[str(period)] = {
                    "minimum_event_support": False,
                    "minimum_stock_support": False,
                    "precision_at_least_0_60": False,
                    "precision_lower_at_least_0_55": False,
                    "precision_lift_at_least_0_05": False,
                    "precision_lift_lower_positive": False,
                    "precision_lift_holm": False,
                    "dynamic_net_positive": False,
                    "dynamic_net_lower_positive": False,
                    "dynamic_net_holm": False,
                    "survives_7_5_bps_per_side": False,
                    "positive_month_majority": False,
                    "stock_deletion_breadth": False,
                }
                continue
            metric = metric_rows.iloc[0]
            precision_boot = bootstraps.loc[
                bootstraps["family"].eq("precision")
                & bootstraps["detector"].eq(detector)
                & bootstraps["period"].eq(period)
            ].iloc[0]
            lift_boot = bootstraps.loc[
                bootstraps["family"].eq("precision_lift")
                & bootstraps["detector"].eq(detector)
                & bootstraps["period"].eq(period)
            ].iloc[0]
            net_boot = bootstraps.loc[
                bootstraps["family"].eq("dynamic_net")
                & bootstraps["detector"].eq(detector)
                & bootstraps["period"].eq(period)
            ].iloc[0]
            period_months = months.loc[
                months["detector"].eq(detector) & months["period"].eq(period)
            ]
            period_deletions = deletions.loc[
                deletions["detector"].eq(detector) & deletions["period"].eq(period)
            ]
            cost_75 = costs.loc[
                costs["detector"].eq(detector)
                & costs["period"].eq(period)
                & costs["cost_bps_per_side"].eq(7.5)
            ].iloc[0]
            min_events = int(contract["evaluation"]["minimum_events"][str(period)])
            min_stocks = int(contract["evaluation"]["minimum_stocks"][str(period)])
            deletion_required = 16 if period == 2025 else 15
            checks = {
                "minimum_event_support": bool(metric["scored_paired_rows"] >= min_events),
                "minimum_stock_support": bool(metric["stocks"] >= min_stocks),
                "precision_at_least_0_60": bool(metric["target_first_precision"] >= 0.60),
                "precision_lower_at_least_0_55": bool(precision_boot["ci_lower"] >= 0.55),
                "precision_lift_at_least_0_05": bool(metric["precision_lift"] >= 0.05),
                "precision_lift_lower_positive": bool(lift_boot["ci_lower"] > 0.0),
                "precision_lift_holm": bool(lift_boot["passes_holm_0_05"]),
                "dynamic_net_positive": bool(metric["dynamic_mean_net_bps"] > 0.0),
                "dynamic_net_lower_positive": bool(net_boot["ci_lower"] > 0.0),
                "dynamic_net_holm": bool(net_boot["passes_holm_0_05"]),
                "survives_7_5_bps_per_side": bool(cost_75["dynamic_mean_net_bps"] > 0.0),
                "positive_month_majority": bool(
                    period_months["dynamic_mean_net_bps"].gt(0).sum() > len(period_months) / 2
                ),
                "stock_deletion_breadth": bool(
                    (
                        period_deletions["dynamic_mean_net_bps"].gt(0)
                        & period_deletions["precision_lift"].gt(0)
                    ).sum()
                    >= deletion_required
                ),
            }
            checks_by_period[str(period)] = checks
        supported = all(
            value for period_checks in checks_by_period.values() for value in period_checks.values()
        )
        detector_decisions[detector] = {
            "decision": (
                "detector_supported_for_prospective_research_logging_only"
                if supported
                else "detector_rejected_or_descriptive_only"
            ),
            "checks": checks_by_period,
        }
    retained = [
        detector
        for detector, result in detector_decisions.items()
        if result["decision"] == "detector_supported_for_prospective_research_logging_only"
    ]
    return {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scientific_status": contract["scientific_status"],
        "validation_claim": False,
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "application_modified": False,
        "primary_cost_bps_per_side": PRIMARY_COST_PER_SIDE_BPS,
        "detectors": detector_decisions,
        "retained_for_prospective_research_logging_only": retained,
        "overall_decision": (
            "one_or_more_detectors_supported_for_prospective_research_logging_only"
            if retained
            else "all_detectors_rejected_or_descriptive_only"
        ),
    }


def write_artifact_manifest(out: Path) -> None:
    files = []
    for path in sorted(out.iterdir(), key=lambda item: item.name):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(out / "artifact_manifest.json", {"files": files})


def score_experiment(frozen: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    contract = load_contract()
    pre_score = json.loads((frozen / "pre_score_manifest.json").read_text())
    current_sources = {name: sha256(path) for name, path in sorted(source_paths(contract).items())}
    if current_sources != pre_score["source_sha256"]:
        raise AssertionError("source hash drift after event freeze")
    for name, expected in pre_score["frozen_file_sha256"].items():
        if sha256(frozen / name) != expected:
            raise AssertionError(f"frozen event file drift: {name}")
    events = pd.read_parquet(frozen / "pre_outcome_events.parquet")
    controls = pd.read_parquet(frozen / "pre_outcome_controls.parquet")
    if len(events) != pre_score["event_rows"] or len(controls) != pre_score["control_rows"]:
        raise AssertionError("frozen ledger cardinality drift")

    event_outcomes, control_outcomes, raw_coverage = score_ledgers(contract, events, controls)
    paired = paired_outcomes(event_outcomes, control_outcomes, controls)
    prep_coverage = pd.read_csv(frozen / "data_coverage.csv")
    metrics, bootstraps = build_metrics(paired, event_outcomes, prep_coverage)
    months = build_month_metrics(paired)
    deletions = build_stock_deletions(paired, contract)
    costs = build_cost_sensitivity(paired)
    paths = build_path_diagnostics(paired)
    decision = decide(metrics, bootstraps, months, deletions, costs, contract)

    events.to_parquet(out / "pre_outcome_events.parquet", index=False)
    controls.to_parquet(out / "pre_outcome_controls.parquet", index=False)
    prep_coverage.to_csv(out / "data_coverage.csv", index=False)
    pd.read_csv(frozen / "event_counts.csv").to_csv(out / "event_counts.csv", index=False)
    event_outcomes.to_parquet(out / "event_outcomes.parquet", index=False)
    control_outcomes.to_parquet(out / "control_outcomes.parquet", index=False)
    paired.to_parquet(out / "paired_outcomes.parquet", index=False)
    metrics.to_csv(out / "detector_metrics.csv", index=False)
    bootstraps.to_csv(out / "bootstrap_metrics.csv", index=False)
    months.to_csv(out / "monthly_metrics.csv", index=False)
    deletions.to_csv(out / "stock_deletions.csv", index=False)
    costs.to_csv(out / "cost_sensitivity.csv", index=False)
    paths.to_csv(out / "path_diagnostics.csv", index=False)
    raw_coverage.to_csv(out / "raw_data_coverage.csv", index=False)
    write_json(out / "decision.json", decision)
    write_json(out / "source_hashes.json", pre_score)
    summary = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scientific_status": contract["scientific_status"],
        "event_rows": int(len(events)),
        "control_rows": int(len(controls)),
        "scored_event_rows": int(event_outcomes["outcome_status"].eq("scored").sum()),
        "scored_control_rows": int(control_outcomes["outcome_status"].eq("scored").sum()),
        "paired_rows": int(len(paired)),
        "metrics": metrics.to_dict(orient="records"),
        "bootstraps": bootstraps.to_dict(orient="records"),
        "decision": decision,
        "historical_volume_label": contract["data"]["volume_label"],
        "quotes_or_ticks_used": False,
    }
    write_json(out / "summary.json", summary)
    write_artifact_manifest(out)
    print(
        json.dumps(
            {
                "out": str(out),
                "events": len(events),
                "paired_rows": len(paired),
                "overall_decision": decision["overall_decision"],
                "retained": decision["retained_for_prospective_research_logging_only"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-events", type=Path)
    group.add_argument("--score", type=Path)
    parser.add_argument("--frozen-input", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_events is not None:
        if args.frozen_input is not None:
            raise SystemExit("--frozen-input is invalid with --prepare-events")
        prepare_events(args.prepare_events)
        return
    if args.frozen_input is None:
        raise SystemExit("--frozen-input is required with --score")
    score_experiment(args.frozen_input, args.score)


if __name__ == "__main__":
    main()
