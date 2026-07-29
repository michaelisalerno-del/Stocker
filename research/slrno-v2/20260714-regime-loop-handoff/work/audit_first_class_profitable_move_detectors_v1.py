"""Independent audit of first-class profitable-move detector artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-first-class-profitable-move-detectors-v1.json"
RUNNER_PATH = HERE / "run_first_class_profitable_move_detectors_v1.py"
TEST_PATH = HERE / "tests/test_first_class_profitable_move_detectors_v1.py"
DETECTORS = (
    "H1_downside_expansion_exhaustion_long",
    "H2_failed_breakdown_reclaim_long",
    "H3_two_bar_reversal_confirmation_long",
    "H4_opening_range_failed_breakdown_long",
    "H5_activity_activated_downside_expansion_long",
)
ROUND_TRIP_COST_BPS = 10.0
HORIZON_BARS = 24
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
SEED = 20260714


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def provider_path(contract: dict[str, Any], symbol: str) -> Path:
    return (
        Path(contract["data"]["provider_root"])
        / f"symbol={symbol}"
        / "timeframe=5m"
        / "data.parquet"
    )


def symbols(contract: dict[str, Any], period: int) -> list[str]:
    return list(contract["data"][f"symbols_{period}"])


def source_path(name: str, contract: dict[str, Any]) -> Path:
    if name == "contract":
        return CONTRACT_PATH
    if name == "runner":
        return RUNNER_PATH
    if name == "auditor":
        return Path(__file__).resolve()
    if name == "tests":
        return TEST_PATH
    if name.startswith("origin_report_"):
        index = int(name.removeprefix("origin_report_")) - 1
        return Path(contract["provenance"]["origin_reports"][index])
    if name.startswith("provider_"):
        _, period, symbol = name.split("_", 2)
        if symbol not in symbols(contract, int(period)):
            raise KeyError(name)
        return provider_path(contract, symbol)
    raise KeyError(name)


def read_tape(contract: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for period in (2025, 2026):
        lower, upper = contract["data"]["period_bounds"][str(period)]
        for symbol in symbols(contract, period):
            frame = pd.read_parquet(
                provider_path(contract, symbol),
                columns=["timestamp", "open", "high", "low", "close", "volume"],
                filters=[
                    ("timestamp", ">=", pd.Timestamp(lower).to_pydatetime()),
                    ("timestamp", "<", pd.Timestamp(upper).to_pydatetime()),
                ],
                engine="pyarrow",
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            if frame["timestamp"].duplicated().any():
                raise AssertionError("duplicate raw timestamps")
            prices = frame[["open", "high", "low", "close"]].to_numpy(float)
            valid = (
                np.isfinite(prices).all(axis=1)
                & (prices > 0).all(axis=1)
                & (prices[:, 2] <= np.minimum(prices[:, 0], prices[:, 3]))
                & (np.maximum(prices[:, 0], prices[:, 3]) <= prices[:, 1])
            )
            local = frame["timestamp"].dt.tz_convert("America/New_York")
            minute = local.dt.hour * 60 + local.dt.minute
            regular = minute.ge(570) & minute.lt(960)
            on_grid = ((minute - 570) % 5).eq(0) & local.dt.second.eq(0)
            chosen = frame.loc[valid & regular.to_numpy(bool) & on_grid.to_numpy(bool)].copy()
            chosen_local = chosen["timestamp"].dt.tz_convert("America/New_York")
            chosen_minute = chosen_local.dt.hour * 60 + chosen_local.dt.minute
            chosen["period"] = period
            chosen["symbol_norm"] = symbol
            chosen["session_date"] = chosen_local.dt.strftime("%Y-%m-%d")
            chosen["month"] = chosen["session_date"].str[:7]
            chosen["bar_ordinal"] = ((chosen_minute - 570) // 5).astype(int)
            chosen = chosen.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
            is_new = (
                chosen.groupby("session_date", sort=False)["timestamp"]
                .diff()
                .ne(pd.Timedelta(minutes=5))
            )
            chosen["segment_index"] = (
                is_new.groupby(chosen["session_date"], sort=False).cumsum().astype(int) - 1
            )
            chosen["segment_position"] = chosen.groupby(
                ["session_date", "segment_index"], sort=False
            ).cumcount()
            frames.append(chosen)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["period", "symbol_norm", "session_date", "timestamp"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def add_features(tape: pd.DataFrame) -> pd.DataFrame:
    frame = tape.copy()
    keys = ["period", "symbol_norm", "session_date", "segment_index"]
    grouped = frame.groupby(keys, sort=False)
    span = frame["high"] - frame["low"]
    frame["bar_range_bps"] = 10000.0 * span / frame["open"]
    frame["signed_body_bps"] = 10000.0 * (frame["close"] / frame["open"] - 1.0)
    frame["close_location"] = (frame["close"] - frame["low"]) / span
    frame["lower_wick_fraction"] = (np.minimum(frame["open"], frame["close"]) - frame["low"]) / span
    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            span,
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    frame["true_range_bps"] = 10000.0 * true_range / previous_close
    frame["prior_scale_bps"] = frame.groupby(keys, sort=False)["true_range_bps"].transform(
        lambda values: values.shift().rolling(12, min_periods=12).median()
    )
    frame["return_3_bps"] = 10000.0 * (frame["close"] / grouped["close"].shift(3) - 1.0)
    frame["prior_12_low"] = frame.groupby(keys, sort=False)["low"].transform(
        lambda values: values.shift().rolling(12, min_periods=12).min()
    )
    log_volume = np.log(frame["volume"].where(frame["volume"].gt(0)))
    prior_log = log_volume.groupby([frame[key] for key in keys], sort=False).transform(
        lambda values: values.shift().rolling(6, min_periods=6).mean()
    )
    frame["relative_activity_6"] = np.exp(log_volume - prior_log)

    opening = (
        frame.loc[frame["bar_ordinal"].between(0, 5)]
        .groupby(["period", "symbol_norm", "session_date"], sort=False)["low"]
        .agg(["min", "count"])
    )
    opening_low = opening["min"].where(opening["count"].eq(6)).rename("opening_range_low")
    frame = frame.join(
        opening_low,
        on=["period", "symbol_norm", "session_date"],
        validate="many_to_one",
    )
    grouped = frame.groupby(keys, sort=False)
    frame["entry_open"] = grouped["open"].shift(-1)
    frame["entry_timestamp"] = grouped["timestamp"].shift(-1)
    frame["next_position"] = grouped["segment_position"].shift(-1)
    for source, target in (
        ("bar_range_bps", "prior_bar_range_bps"),
        ("signed_body_bps", "prior_bar_body_bps"),
        ("close_location", "prior_bar_close_location"),
        ("prior_scale_bps", "prior_bar_scale_bps"),
        ("low", "prior_bar_low"),
    ):
        frame[target] = grouped[source].shift(1)
    frame["prior_bar_midpoint"] = (grouped["high"].shift(1) + grouped["low"].shift(1)) / 2.0
    frame["clock_bucket"] = (frame["bar_ordinal"] // 13).clip(0, 5).astype(int)
    return frame


def masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    scale = frame["prior_scale_bps"]
    h1 = (
        scale.gt(0)
        & frame["bar_range_bps"].ge(1.25 * scale)
        & frame["signed_body_bps"].le(-0.75 * scale)
        & frame["close_location"].le(0.20)
        & frame["return_3_bps"].le(-1.00 * scale)
    )
    breach = 10000.0 * (frame["prior_12_low"] - frame["low"]) / frame["close"]
    h2 = (
        scale.gt(0)
        & breach.ge(0.10 * scale)
        & frame["close"].gt(frame["prior_12_low"])
        & frame["lower_wick_fraction"].ge(0.40)
        & frame["return_3_bps"].lt(0)
    )
    previous_scale = frame["prior_bar_scale_bps"]
    h3 = (
        previous_scale.gt(0)
        & frame["prior_bar_range_bps"].ge(1.25 * previous_scale)
        & frame["prior_bar_body_bps"].le(-0.75 * previous_scale)
        & frame["prior_bar_close_location"].le(0.20)
        & frame["close"].gt(frame["prior_bar_midpoint"])
        & frame["close"].gt(frame["open"])
        & frame["close_location"].ge(0.65)
    )
    opening_breach = 10000.0 * (frame["opening_range_low"] - frame["low"]) / frame["close"]
    h4 = (
        scale.gt(0)
        & frame["bar_ordinal"].between(6, 35)
        & opening_breach.ge(0.10 * scale)
        & frame["close"].gt(frame["opening_range_low"])
        & frame["lower_wick_fraction"].ge(0.35)
        & frame["return_3_bps"].lt(0)
    )
    return dict(
        zip(DETECTORS, (h1, h2, h3, h4, h1 & frame["relative_activity_6"].ge(1.10)), strict=True)
    )


def rebuild_events(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = (
        features["bar_ordinal"].between(12, 53)
        & features["next_position"].eq(features["segment_position"] + 1)
        & features["entry_open"].gt(0)
        & features["prior_scale_bps"].gt(0)
    )
    candidates: list[pd.DataFrame] = []
    for detector, detector_mask in masks(features).items():
        chosen = features.loc[eligible & detector_mask].copy()
        chosen["detector"] = detector
        chosen["invalidation_price"] = (
            np.minimum(chosen["low"], chosen["prior_bar_low"])
            if detector == "H3_two_bar_reversal_confirmation_long"
            else chosen["low"]
        )
        chosen["risk_bps"] = (
            10000.0 * (chosen["entry_open"] - chosen["invalidation_price"]) / chosen["entry_open"]
        )
        candidates.append(chosen.loc[chosen["risk_bps"].between(20.0, 250.0)])
    raw = pd.concat(candidates, ignore_index=True)
    retained: list[int] = []
    for _, group in raw.groupby(["period", "symbol_norm", "session_date", "detector"], sort=False):
        previous = -10000
        for row in group.sort_values("segment_position", kind="mergesort").itertuples():
            if int(row.segment_position) - previous > 24:
                retained.append(int(row.Index))
                previous = int(row.segment_position)
    events = raw.loc[retained].copy()
    events["decision_timestamp"] = events["timestamp"]
    events["event_id"] = events.apply(
        lambda row: (
            f"{int(row['period'])}|{row['detector']}|{row['symbol_norm']}|"
            f"{pd.Timestamp(row['timestamp']).isoformat()}"
        ),
        axis=1,
    )
    events["target_price"] = events["entry_open"] * (1 + events["risk_bps"] / 10000)
    eligible_anchors = features.loc[eligible].copy()
    eligible_anchors["decision_timestamp"] = eligible_anchors["timestamp"]
    return events, eligible_anchors


def rebuild_controls(events: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (period, _detector), event_group in events.groupby(["period", "detector"], sort=True):
        period_anchors = anchors.loc[anchors["period"].eq(period)].copy()
        event_keys = set(
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
            ["symbol_norm", "decision_timestamp"], kind="mergesort"
        ).itertuples(index=False):
            pool = pools[(event.symbol_norm, event.month, event.clock_bucket)]
            available = []
            for candidate in pool.itertuples(index=False):
                candidate_key = (candidate.symbol_norm, candidate.decision_timestamp)
                if (
                    candidate.session_date != event.session_date
                    and candidate_key not in event_keys
                    and candidate_key not in used
                ):
                    distance = abs(
                        math.log(
                            max(float(candidate.prior_scale_bps), 1e-12)
                            / max(float(event.prior_scale_bps), 1e-12)
                        )
                    )
                    tie = stable_hash(
                        f"{event.event_id}|{pd.Timestamp(candidate.decision_timestamp).isoformat()}"
                    )
                    available.append((distance, tie, candidate))
            if not available:
                raise AssertionError("independent unmatched control")
            chosen = min(available, key=lambda item: (item[0], item[1]))[2]
            used.add((chosen.symbol_norm, chosen.decision_timestamp))
            entry = float(chosen.entry_open)
            risk = float(event.risk_bps)
            rows.append(
                {
                    "control_id": f"control|{event.event_id}",
                    "matched_event_id": event.event_id,
                    "decision_timestamp": chosen.decision_timestamp,
                    "entry_timestamp": chosen.entry_timestamp,
                    "entry_open": entry,
                    "risk_bps": risk,
                    "target_price": entry * (1 + risk / 10000),
                    "stop_price": entry * (1 - risk / 10000),
                }
            )
    return pd.DataFrame(rows)


def groups(tape: pd.DataFrame) -> dict[tuple[int, str, str, int], pd.DataFrame]:
    keys = ["period", "symbol_norm", "session_date", "segment_index"]
    return {
        (int(period), str(symbol), str(session), int(segment)): group.set_index(
            "segment_position", drop=False
        ).sort_index()
        for (period, symbol, session, segment), group in tape.groupby(keys, sort=False)
    }


def replay_outcome(
    row: Any, lookup: dict[tuple[int, str, str, int], pd.DataFrame], stop: float
) -> dict[str, Any]:
    group = lookup[(int(row.period), row.symbol_norm, row.session_date, int(row.segment_index))]
    positions = np.arange(
        int(row.decision_segment_position) + 1,
        int(row.decision_segment_position) + 25,
    )
    if not np.isin(positions, group.index).all():
        return {"outcome_status": "missing_exact_future_path"}
    future = group.loc[positions]
    entry = float(row.entry_open)
    target = float(row.target_price)
    hit = "no_touch_time_exit"
    step_value: int | None = None
    exit_price = float(future.iloc[-1]["close"])
    pre_low = entry
    for step, bar in enumerate(future.itertuples(index=False), start=1):
        pre_low = min(pre_low, float(bar.low))
        if float(bar.open) >= target:
            hit, step_value, exit_price = "target_gap_or_open_first", step, target
            break
        if float(bar.open) <= stop:
            hit, step_value, exit_price = "stop_gap_or_open_first", step, stop
            break
        target_touch = float(bar.high) >= target
        stop_touch = float(bar.low) <= stop
        if target_touch and stop_touch:
            hit, step_value, exit_price = "dual_touch_conservative_stop", step, stop
            break
        if target_touch:
            hit, step_value, exit_price = "target_first", step, target
            break
        if stop_touch:
            hit, step_value, exit_price = "stop_first", step, stop
            break
    success = hit in {"target_gap_or_open_first", "target_first"}
    favorable = 10000 * (future["high"].to_numpy(float) / entry - 1)
    adverse = 10000 * (1 - future["low"].to_numpy(float) / entry)
    pre_mae_r = 10000 * (1 - pre_low / entry) / float(row.risk_bps)
    return {
        "outcome_status": "scored",
        "hit_type": hit,
        "hit_step": step_value,
        "target_first": success,
        "rapid_target_3": bool(success and step_value is not None and step_value <= 3),
        "clean_success": bool(
            success and step_value is not None and step_value <= 6 and pre_mae_r <= 0.5
        ),
        "pre_target_mae_r": pre_mae_r,
        "mfe_bps": float(favorable.max()),
        "mae_bps": float(adverse.max()),
        "mfe_r": float(favorable.max()) / float(row.risk_bps),
        "mae_r": float(adverse.max()) / float(row.risk_bps),
        "time_to_mfe": int(favorable.argmax() + 1),
        "time_to_mae": int(adverse.argmax() + 1),
        "dynamic_gross_bps": 10000 * (exit_price / entry - 1),
        "dynamic_net_bps": 10000 * (exit_price / entry - 1) - ROUND_TRIP_COST_BPS,
        "fixed_h24_gross_bps": 10000 * (float(future.iloc[-1]["close"]) / entry - 1),
        "fixed_h24_net_bps": 10000 * (float(future.iloc[-1]["close"]) / entry - 1)
        - ROUND_TRIP_COST_BPS,
    }


def bootstrap(frame: pd.DataFrame, column: str, seed: int) -> dict[str, float]:
    daily = frame.groupby("session_date", sort=True)[column].agg(["sum", "count"])
    sums = daily["sum"].to_numpy(float)
    counts = daily["count"].to_numpy(float)
    size = len(daily)
    rng = np.random.default_rng(seed)
    result = np.empty(BOOTSTRAP_DRAWS)
    offsets = np.arange(BOOTSTRAP_BLOCK)
    blocks = math.ceil(size / BOOTSTRAP_BLOCK)
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, size, size=blocks)
        chosen = ((starts[:, None] + offsets) % size).ravel()[:size]
        result[draw] = sums[chosen].sum() / counts[chosen].sum()
    return {
        "observed_mean": float(frame[column].mean()),
        "ci_lower": float(np.quantile(result, 0.025)),
        "ci_upper": float(np.quantile(result, 0.975)),
        "p_one_sided": float((1 + (result <= 0).sum()) / (BOOTSTRAP_DRAWS + 1)),
    }


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def maximum_numeric_error(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> float:
    error = 0.0
    for column in columns:
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if finite.any():
            error = max(error, float(np.max(np.abs(left_values[finite] - right_values[finite]))))
        if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
            return float("inf")
    return error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    output = args.artifact / "independent_audit.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads((args.artifact / "source_hashes.json").read_text())
    stored_events = pd.read_parquet(args.artifact / "pre_outcome_events.parquet")
    stored_controls = pd.read_parquet(args.artifact / "pre_outcome_controls.parquet")
    event_outcomes = pd.read_parquet(args.artifact / "event_outcomes.parquet")
    control_outcomes = pd.read_parquet(args.artifact / "control_outcomes.parquet")
    paired = pd.read_parquet(args.artifact / "paired_outcomes.parquet")
    stored_metrics = pd.read_csv(args.artifact / "detector_metrics.csv")
    stored_bootstraps = pd.read_csv(args.artifact / "bootstrap_metrics.csv")

    safety = contract["safety"]
    safety_ok = bool(
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["application_code_modification_allowed"] is False
        and contract["opened_data_status"]["validation_claim_allowed"] is False
    )
    current_hashes = {
        name: sha256(source_path(name, contract)) for name in pre_score["source_sha256"]
    }
    source_hashes_ok = current_hashes == pre_score["source_sha256"]
    frozen_hashes_ok = all(
        sha256(args.artifact / name) == expected
        for name, expected in pre_score["frozen_file_sha256"].items()
        if (args.artifact / name).exists()
    ) and all((args.artifact / name).exists() for name in pre_score["frozen_file_sha256"])
    forbidden_outcomes = {
        "target_first",
        "hit_type",
        "mfe_bps",
        "mae_bps",
        "dynamic_net_bps",
        "fixed_h24_net_bps",
    }
    ledger_schema_ok = forbidden_outcomes.isdisjoint(
        stored_events.columns
    ) and forbidden_outcomes.isdisjoint(stored_controls.columns)

    tape = read_tape(contract)
    features = add_features(tape)
    rebuilt_events, anchors = rebuild_events(features)
    event_id_ok = set(rebuilt_events["event_id"]) == set(stored_events["event_id"])
    event_compare = stored_events.merge(
        rebuilt_events[
            [
                "event_id",
                "entry_open",
                "invalidation_price",
                "risk_bps",
                "target_price",
                "prior_scale_bps",
            ]
        ],
        on="event_id",
        how="outer",
        indicator=True,
        suffixes=("_stored", "_rebuilt"),
    )
    event_numeric_error = maximum_numeric_error(
        event_compare,
        event_compare.rename(
            columns={
                "entry_open_rebuilt": "entry_open_stored",
                "invalidation_price_rebuilt": "invalidation_price_stored",
                "risk_bps_rebuilt": "risk_bps_stored",
                "target_price_rebuilt": "target_price_stored",
                "prior_scale_bps_rebuilt": "prior_scale_bps_stored",
            }
        ),
        [],
    )
    event_numeric_error = 0.0
    for name in ("entry_open", "invalidation_price", "risk_bps", "target_price", "prior_scale_bps"):
        event_numeric_error = max(
            event_numeric_error,
            float(
                np.nanmax(
                    np.abs(
                        event_compare[f"{name}_stored"].to_numpy(float)
                        - event_compare[f"{name}_rebuilt"].to_numpy(float)
                    )
                )
            ),
        )
    events_ok = bool(
        event_id_ok and event_compare["_merge"].eq("both").all() and event_numeric_error <= 1e-10
    )

    rebuilt_controls = rebuild_controls(rebuilt_events, anchors)
    control_compare = stored_controls.merge(
        rebuilt_controls,
        on="control_id",
        how="outer",
        indicator=True,
        suffixes=("_stored", "_rebuilt"),
    )
    control_identity_ok = bool(
        control_compare["_merge"].eq("both").all()
        and control_compare["matched_event_id_stored"]
        .eq(control_compare["matched_event_id_rebuilt"])
        .all()
        and pd.to_datetime(control_compare["decision_timestamp_stored"])
        .eq(pd.to_datetime(control_compare["decision_timestamp_rebuilt"]))
        .all()
    )
    control_numeric_error = 0.0
    for name in ("entry_open", "risk_bps", "target_price", "stop_price"):
        control_numeric_error = max(
            control_numeric_error,
            float(
                np.nanmax(
                    np.abs(
                        control_compare[f"{name}_stored"].to_numpy(float)
                        - control_compare[f"{name}_rebuilt"].to_numpy(float)
                    )
                )
            ),
        )
    controls_ok = control_identity_ok and control_numeric_error <= 1e-10

    lookup = groups(tape)
    event_replay_errors = 0
    outcome_numeric_error = 0.0
    event_ledger_lookup = stored_events.set_index("event_id")
    for stored in event_outcomes.itertuples(index=False):
        ledger = event_ledger_lookup.loc[stored.event_id]
        replay = replay_outcome(SimpleRow(ledger), lookup, float(ledger["invalidation_price"]))
        for field in (
            "outcome_status",
            "hit_type",
            "target_first",
            "rapid_target_3",
            "clean_success",
        ):
            if replay.get(field) != getattr(stored, field):
                event_replay_errors += 1
        for field in (
            "pre_target_mae_r",
            "mfe_bps",
            "mae_bps",
            "mfe_r",
            "mae_r",
            "dynamic_gross_bps",
            "dynamic_net_bps",
            "fixed_h24_gross_bps",
            "fixed_h24_net_bps",
        ):
            outcome_numeric_error = max(
                outcome_numeric_error,
                abs(float(replay[field]) - float(getattr(stored, field))),
            )

    control_replay_errors = 0
    control_ledger_lookup = stored_controls.set_index("control_id")
    for stored in control_outcomes.itertuples(index=False):
        ledger = control_ledger_lookup.loc[stored.control_id]
        replay = replay_outcome(SimpleRow(ledger), lookup, float(ledger["stop_price"]))
        for field in (
            "outcome_status",
            "hit_type",
            "target_first",
            "rapid_target_3",
            "clean_success",
        ):
            if replay.get(field) != getattr(stored, field):
                control_replay_errors += 1
        for field in (
            "pre_target_mae_r",
            "mfe_bps",
            "mae_bps",
            "mfe_r",
            "mae_r",
            "dynamic_gross_bps",
            "dynamic_net_bps",
            "fixed_h24_gross_bps",
            "fixed_h24_net_bps",
        ):
            outcome_numeric_error = max(
                outcome_numeric_error,
                abs(float(replay[field]) - float(getattr(stored, field))),
            )
    outcomes_ok = (
        event_replay_errors == 0 and control_replay_errors == 0 and outcome_numeric_error <= 1e-10
    )

    aggregate_error = 0.0
    for row in stored_metrics.itertuples(index=False):
        group = paired.loc[paired["detector"].eq(row.detector) & paired["period"].eq(row.period)]
        replay_values = {
            "target_first_precision": float(group["target_first"].mean()),
            "control_precision": float(group["target_first_control"].mean()),
            "precision_lift": float(group["precision_lift_pair"].mean()),
            "dynamic_mean_net_bps": float(group["dynamic_net_bps"].mean()),
            "paired_dynamic_difference_bps": float(group["dynamic_net_difference_bps"].mean()),
            "fixed_h24_mean_net_bps": float(group["fixed_h24_net_bps"].mean()),
        }
        for name, value in replay_values.items():
            aggregate_error = max(aggregate_error, abs(value - float(getattr(row, name))))
    aggregates_ok = aggregate_error <= 1e-10

    bootstrap_error = 0.0
    replay_boot_rows: list[dict[str, Any]] = []
    metric_specs = (
        ("precision", "precision_pair"),
        ("precision_lift", "precision_lift_pair"),
        ("dynamic_net", "dynamic_net_bps"),
    )
    for detector_index, detector in enumerate(DETECTORS):
        for period_index, period in enumerate((2025, 2026)):
            group = paired.loc[paired["detector"].eq(detector) & paired["period"].eq(period)]
            if group.empty:
                continue
            for metric_index, (family, column) in enumerate(metric_specs):
                values = bootstrap(
                    group,
                    column,
                    SEED + detector_index * 1000 + period_index * 100 + metric_index,
                )
                replay_boot_rows.append(
                    {"family": family, "detector": detector, "period": period, **values}
                )
    replay_boot = pd.DataFrame(replay_boot_rows)
    replay_boot["holm_adjusted_p"] = np.nan
    for family in ("precision_lift", "dynamic_net"):
        mask = replay_boot["family"].eq(family)
        replay_boot.loc[mask, "holm_adjusted_p"] = holm(
            replay_boot.loc[mask, "p_one_sided"].to_numpy(float)
        )
    boot_compare = stored_bootstraps.merge(
        replay_boot,
        on=["family", "detector", "period"],
        suffixes=("_stored", "_replay"),
        validate="one_to_one",
    )
    for name in ("observed_mean", "ci_lower", "ci_upper", "p_one_sided", "holm_adjusted_p"):
        left = boot_compare[f"{name}_stored"].to_numpy(float)
        right = boot_compare[f"{name}_replay"].to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            bootstrap_error = max(
                bootstrap_error, float(np.max(np.abs(left[finite] - right[finite])))
            )
    bootstraps_ok = bootstrap_error <= 1e-10

    manifest = json.loads((args.artifact / "artifact_manifest.json").read_text())
    manifest_names = {item["name"] for item in manifest["files"]}
    actual_names = {
        path.name
        for path in args.artifact.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json", "independent_audit.json"}
    }
    manifest_errors = int(manifest_names != actual_names)
    for item in manifest["files"]:
        path = args.artifact / item["name"]
        manifest_errors += int(
            not path.exists()
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]
        )
    checks = {
        "research_only_opened_data_status": safety_ok,
        "frozen_source_hashes_match": source_hashes_ok,
        "frozen_pre_outcome_files_match": frozen_hashes_ok,
        "pre_outcome_ledgers_exclude_outcomes": ledger_schema_ok,
        "event_set_and_causal_values_reconstructed": events_ok,
        "matched_controls_reconstructed_without_outcomes": controls_ok,
        "event_and_control_paths_replayed": outcomes_ok,
        "aggregate_metrics_replayed": aggregates_ok,
        "bootstrap_and_holm_replayed": bootstraps_ok,
        "artifact_manifest_complete_and_valid": manifest_errors == 0,
        "aal_2026_excluded": "AAL" not in symbols(contract, 2026),
        "no_regime_or_loop_detector_fields": all(
            forbidden not in json.dumps(contract["shared_features"]).lower()
            for forbidden in ("regime", "loop", "cycle", "route", "child", "morph")
        ),
    }
    payload = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "errors": {
            "event_replay": event_replay_errors,
            "control_replay": control_replay_errors,
            "manifest": manifest_errors,
        },
        "maximum_errors": {
            "event_numeric": event_numeric_error,
            "control_numeric": control_numeric_error,
            "outcome_numeric": outcome_numeric_error,
            "aggregate": aggregate_error,
            "bootstrap_holm": bootstrap_error,
        },
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "passed": payload["passed"],
                "total": payload["total"],
            },
            indent=2,
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)


class SimpleRow:
    """Attribute view over a pandas Series for independent path replay."""

    def __init__(self, series: pd.Series) -> None:
        self.__dict__.update(series.to_dict())


if __name__ == "__main__":
    main()
