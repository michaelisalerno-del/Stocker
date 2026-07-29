"""Independent audit for the research-only selective payoff equation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-selective-payoff-equations-v1.json"
SEED = 20260714
ROUND_TRIP_COST_BPS = 10.0
LOOP_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))
LOOP_NUMERIC = (
    *LOOP_COLUMNS,
    "loop_score_mass",
    "loop_score_entropy",
    "top_loop_score",
    "top_loop_margin",
)
CONTEXT_NUMERIC = (
    "entry_step",
    "decision_session_fraction",
    "base_risk_bps",
    "anchor_range_prior_atr",
    "decision_range_prior_atr",
    "decision_body_fraction",
    "decision_upper_wick_fraction",
    "decision_lower_wick_fraction",
    "decision_close_location",
    "directional_decision_displacement_prior_atr",
    "decision_vwap_distance_prior_atr",
    "decision_activity_ratio",
    "compression_ratio",
    "trend_return_6",
    "source_body_fraction",
    "source_outer_fraction",
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
CONTEXT_CATEGORICAL = (
    "direction_label",
    "state_label",
    "previous_state_label",
    "clock_quartile_label",
    "strong_close_label",
    "trend_aligned_label",
)
PREDICTION_CATEGORICAL = (
    "direction_label",
    "state_label",
    "clock_quartile_label",
    "top_loop_label",
)
SEQUENTIAL_NUMERIC = (
    "checkpoint",
    "sequential_risk_bps",
    "directional_close_return_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "causal_retracement_bps",
    "current_range_prior_atr",
    "current_body_fraction",
    "current_upper_wick_fraction",
    "current_lower_wick_fraction",
    "directional_vwap_distance_prior_atr",
    "current_activity_ratio",
    "favourable_close_fraction",
)
DIRECT_MODELS = {
    "prediction_only": (LOOP_NUMERIC, PREDICTION_CATEGORICAL),
    "context_only": (CONTEXT_NUMERIC, CONTEXT_CATEGORICAL),
    "context_plus_loop_mixture": (
        (*CONTEXT_NUMERIC, *LOOP_NUMERIC),
        (*CONTEXT_CATEGORICAL, "top_loop_label"),
    ),
}
FORBIDDEN = (
    "outcome",
    "target_first",
    "hit_type",
    "gross_return",
    "net_bps",
    "mfe_bps",
    "mae_bps",
    "future_",
    "exit_price",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    return value


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def load_tapes(contract: dict[str, Any]) -> dict[tuple[str, str], pd.DataFrame]:
    root = Path(contract["inputs"]["provider_root_2024"])
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in contract["population"]["symbols"]:
        frame = pd.read_parquet(
            provider_path(root, symbol),
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minute.ge(570) & minute.lt(960) & local.dt.year.eq(2024)].copy()
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        on_grid = local.dt.second.eq(0) & local.dt.microsecond.eq(0) & (minute - 570).mod(5).eq(0)
        frame = frame.loc[on_grid].copy()
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["session_date"] = local.dt.strftime("%Y-%m-%d")
        numeric = ["open", "high", "low", "close", "volume"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        valid = (
            frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & frame["volume"].ge(0)
            & frame["high"].ge(frame[["open", "close"]].max(axis=1))
            & frame["low"].le(frame[["open", "close"]].min(axis=1))
        )
        frame = frame.loc[valid].copy().reset_index(drop=True)
        previous_close = frame["close"].shift(1)
        tr = np.maximum.reduce(
            [
                (frame["high"] - frame["low"]).to_numpy(float),
                (frame["high"] - previous_close).abs().to_numpy(float),
                (frame["low"] - previous_close).abs().to_numpy(float),
            ]
        )
        frame["atr14_prior"] = pd.Series(tr).shift(1).rolling(14, min_periods=14).mean()
        frame["bar_range"] = frame["high"] - frame["low"]
        span = frame["bar_range"].replace(0.0, np.nan)
        frame["body_fraction_calc"] = (frame["close"] - frame["open"]).abs() / span
        frame["upper_wick_fraction_calc"] = (
            frame["high"] - frame[["open", "close"]].max(axis=1)
        ) / span
        frame["lower_wick_fraction_calc"] = (
            frame[["open", "close"]].min(axis=1) - frame["low"]
        ) / span
        frame["close_location_calc"] = (frame["close"] - frame["low"]) / span
        frame["typical"] = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        frame["pv"] = frame["typical"] * frame["volume"]
        cumulative_pv = frame.groupby("session_date", sort=False)["pv"].cumsum()
        cumulative_volume = frame.groupby("session_date", sort=False)["volume"].cumsum()
        frame["session_vwap"] = cumulative_pv / cumulative_volume.replace(0.0, np.nan)
        log_volume = np.log(frame["volume"].where(frame["volume"].gt(0)))
        prior = log_volume.groupby(frame["session_date"], sort=False).transform(
            lambda series: series.shift(1).rolling(6, min_periods=3).mean()
        )
        frame["activity_ratio"] = frame["volume"] / np.exp(prior)
        frame["bar_ordinal"] = frame.groupby("session_date", sort=False).cumcount()
        for session_date, session in frame.groupby("session_date", sort=False):
            groups[(symbol, str(session_date))] = session.sort_values(
                "bar_ordinal", kind="stable"
            ).reset_index(drop=True)
    return groups


def reconstructed_event_ids(
    contract: dict[str, Any], tapes: dict[tuple[str, str], pd.DataFrame]
) -> set[str]:
    signal_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "state",
        "bar_ordinal",
        "clock_quartile",
        "anchor_high",
        "anchor_low",
        "anchor_close",
        "compression_ratio",
        "trend_return_6",
        "confirmed",
        "confirmation_step",
        "strong_close",
        "body_fraction",
        "outer_fraction",
        "trend_aligned",
        "setup",
        "family",
        "horizon",
        "status",
        "direction",
        "entry_step",
    ]
    signals = pd.read_parquet(
        Path(contract["inputs"]["accepted_setup_signals_2024"]), columns=signal_columns
    )
    selected = signals.loc[
        signals["setup"].eq(contract["population"]["setup"])
        & signals["family"].eq(contract["population"]["family"])
        & signals["horizon"].eq(24)
        & signals["status"].eq("filled")
        & signals["symbol_norm"].isin(contract["population"]["symbols"])
    ].copy()
    selected["start_timestamp"] = pd.to_datetime(selected["start_timestamp"], utc=True)
    selected["session_date"] = selected["session_date"].astype(str)
    candidates: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        tape = tapes.get((str(row.symbol_norm), str(row.session_date)))
        if tape is None:
            continue
        anchor_matches = tape.index[
            pd.to_datetime(tape["timestamp"], utc=True).eq(pd.Timestamp(row.start_timestamp))
        ].to_numpy(int)
        if len(anchor_matches) != 1:
            continue
        anchor_ordinal = int(anchor_matches[0])
        decision = anchor_ordinal + int(row.entry_step)
        if decision > 50 or decision + 1 >= len(tape):
            continue
        anchor = tape.iloc[anchor_ordinal]
        decision_bar = tape.iloc[decision]
        if pd.Timestamp(anchor["timestamp"]) != pd.Timestamp(row.start_timestamp):
            raise AssertionError("independent provider anchor mismatch")
        if (
            not math.isfinite(float(anchor["atr14_prior"]))
            or float(anchor["atr14_prior"]) <= 0
            or not math.isfinite(float(decision_bar["atr14_prior"]))
            or float(decision_bar["atr14_prior"]) <= 0
        ):
            continue
        direction = int(row.direction)
        entry = float(tape.iloc[decision + 1]["open"])
        stop = float(row.anchor_low if direction == 1 else row.anchor_high)
        risk_price = direction * (entry - stop)
        risk_bps = 10000.0 * risk_price / entry if risk_price > 0 else math.nan
        if not math.isfinite(risk_bps) or not 20.0 <= risk_bps <= 250.0:
            continue
        event_id = f"eq|2024|{row.symbol_norm}|{row.session_date}|{int(row.anchor_id)}|{direction}"
        candidates.append(
            {
                "event_id": event_id,
                "symbol_norm": row.symbol_norm,
                "session_date": row.session_date,
                "decision_ordinal": decision,
            }
        )
    frame = pd.DataFrame(candidates).sort_values(
        ["symbol_norm", "session_date", "decision_ordinal", "event_id"], kind="stable"
    )
    kept: list[str] = []
    for _, group in frame.groupby(["symbol_norm", "session_date"], sort=False):
        last = -10_000
        for row in group.itertuples(index=False):
            if int(row.decision_ordinal) - last >= 24:
                kept.append(str(row.event_id))
                last = int(row.decision_ordinal)
    return set(kept)


def replay_path(row: Any, tape: pd.DataFrame, sequential: bool) -> dict[str, Any]:
    if sequential:
        entry_ordinal = int(row.sequential_entry_ordinal)
        entry = float(row.sequential_entry_open)
        stop = float(row.sequential_stop_price)
        target = float(row.sequential_target_price)
    else:
        entry_ordinal = int(row.entry_ordinal)
        entry = float(row.entry_open)
        stop = float(row.stop_price)
        target = float(row.target_price)
    future = tape.iloc[entry_ordinal : entry_ordinal + 24]
    direction = int(row.direction)
    hit = "no_touch_time_exit"
    step_value: int | None = None
    exit_price = float(future.iloc[-1]["close"])
    for step, bar in enumerate(future.itertuples(index=False), start=1):
        if direction == 1:
            if float(bar.open) >= target:
                hit, step_value, exit_price = "target_gap_or_open_first", step, target
                break
            if float(bar.open) <= stop:
                hit, step_value, exit_price = "stop_gap_or_open_first", step, float(bar.open)
                break
            target_touch, stop_touch = float(bar.high) >= target, float(bar.low) <= stop
        else:
            if float(bar.open) <= target:
                hit, step_value, exit_price = "target_gap_or_open_first", step, target
                break
            if float(bar.open) >= stop:
                hit, step_value, exit_price = "stop_gap_or_open_first", step, float(bar.open)
                break
            target_touch, stop_touch = float(bar.low) <= target, float(bar.high) >= stop
        if target_touch and stop_touch:
            hit, step_value, exit_price = "dual_touch_conservative_stop", step, stop
            break
        if target_touch:
            hit, step_value, exit_price = "target_first", step, target
            break
        if stop_touch:
            hit, step_value, exit_price = "stop_first", step, stop
            break
    gross = 10000.0 * direction * (exit_price / entry - 1.0)
    return {
        "hit_type": hit,
        "hit_step": step_value,
        "target_first": hit in {"target_gap_or_open_first", "target_first"},
        "gross_bps": gross,
        "net_bps": gross - ROUND_TRIP_COST_BPS,
    }


def model_pipeline(
    numeric: tuple[str, ...], categorical: tuple[str, ...], contract: dict[str, Any]
) -> Pipeline:
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical),
            ),
        ]
    )
    model = LogisticRegression(
        C=float(contract["model"]["C"]),
        solver=str(contract["model"]["solver"]),
        max_iter=int(contract["model"]["maximum_iterations"]),
        random_state=SEED,
    )
    return Pipeline([("features", transform), ("model", model)])


def refit_raw(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    variants: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    id_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    maximum = int(frame["calendar_index"].max())
    for index in range(40, maximum + 1):
        train = frame.loc[
            frame["calendar_index"].ge(max(0, index - 60)) & frame["calendar_index"].lt(index)
        ]
        score = frame.loc[frame["calendar_index"].eq(index)]
        if score.empty or train["session_date"].nunique() < 40 or len(train) < 500:
            continue
        for model_name, (numeric, categorical) in variants.items():
            pipeline = model_pipeline(numeric, categorical, contract)
            pipeline.fit(
                train[list(numeric) + list(categorical)], train["target_first"].astype(int)
            )
            probability = pipeline.predict_proba(score[list(numeric) + list(categorical)])[:, 1]
            for identity, checkpoint, value in zip(
                score[id_column],
                score["checkpoint"] if "checkpoint" in score.columns else np.zeros(len(score)),
                probability,
                strict=True,
            ):
                rows.append(
                    {
                        id_column: identity,
                        "model": model_name,
                        "calendar_index": index,
                        "checkpoint": int(checkpoint),
                        "raw_probability": float(value),
                    }
                )
    return pd.DataFrame(rows)


def refit_calibration(
    raw: pd.DataFrame,
    outcomes: pd.DataFrame,
    contract: dict[str, Any],
    id_column: str,
    sequential: bool,
) -> pd.DataFrame:
    labels = outcomes[[id_column, "target_first"]]
    frame = raw.merge(labels, on=id_column, validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby("model", sort=False):
        for current in group.loc[group["calendar_index"].ge(60)].itertuples(index=False):
            history = group.loc[
                group["calendar_index"].lt(int(current.calendar_index))
                & group["calendar_index"].ge(int(current.calendar_index) - 60)
            ]
            if sequential:
                history = history.loc[history["checkpoint"].eq(int(current.checkpoint))]
                nearest_rows, minimum = 300, 150
            else:
                nearest_rows, minimum = 500, 200
            if len(history) < minimum:
                mean = lower = math.nan
                support = len(history)
            else:
                nearest = history.assign(
                    distance=(history["raw_probability"] - float(current.raw_probability)).abs()
                ).nsmallest(min(nearest_rows, len(history)), "distance", keep="first")
                support = len(nearest)
                wins = int(nearest["target_first"].sum())
                mean = (wins + 0.5) / (support + 1.0)
                lower = float(beta.ppf(0.05, wins + 0.5, support - wins + 0.5))
            rows.append(
                {
                    id_column: getattr(current, id_column),
                    "model": model_name,
                    "checkpoint": int(current.checkpoint),
                    "raw_probability": float(current.raw_probability),
                    "calibrated_probability_mean": mean,
                    "calibrated_probability_lower": lower,
                    "calibration_support": support,
                }
            )
    return pd.DataFrame(rows)


def max_numeric_error(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> float:
    maximum = 0.0
    for column in columns:
        a = left[column].to_numpy(float)
        b = right[column].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(a[finite] - b[finite]))))
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return math.inf
    return maximum


def numeric_difference(left: Any, right: Any) -> float:
    a = float(left)
    b = float(right)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if not math.isfinite(a) or not math.isfinite(b):
        return math.inf
    return abs(a - b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()
    artifact = args.artifact
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    errors: dict[str, Any] = {}
    checks["research_only_opened_status"] = bool(
        contract["safety"]["research_only"]
        and not contract["safety"]["live_ordering_enabled"]
        and contract["safety"]["order_placement"] == "disabled"
        and contract["opened_data_status"]["opened"]
        and not contract["opened_data_status"]["sealed_validation_available"]
    )
    manifest = json.loads((artifact / "pre_score_manifest.json").read_text(encoding="utf-8"))
    source_errors = 0
    for item in manifest["source_hashes"].values():
        path = Path(item["path"])
        source_errors += int(not path.exists() or sha256(path) != item["sha256"])
    checks["frozen_source_hashes_match"] = source_errors == 0
    errors["source_hashes"] = source_errors
    ledger_errors = 0
    for name, item in manifest["frozen_ledgers"].items():
        path = artifact / name
        ledger_errors += int(not path.exists() or sha256(path) != item["sha256"])
    checks["frozen_pre_outcome_files_match"] = ledger_errors == 0
    errors["frozen_ledgers"] = ledger_errors
    base_ledger = pd.read_parquet(artifact / "pre_outcome_base_events.parquet")
    snapshot_ledger = pd.read_parquet(artifact / "pre_outcome_sequential_snapshots.parquet")
    forbidden_columns = []
    for frame, allowed in (
        (base_ledger, set()),
        (snapshot_ledger, {"running_mfe_bps", "running_mae_bps"}),
    ):
        forbidden_columns.extend(
            column
            for column in frame.columns
            if column not in allowed and any(token in column.lower() for token in FORBIDDEN)
        )
    checks["pre_outcome_ledgers_exclude_outcomes"] = not forbidden_columns
    errors["forbidden_pre_outcome_columns"] = forbidden_columns
    tapes = load_tapes(contract)
    rebuilt_ids = reconstructed_event_ids(contract, tapes)
    stored_ids = set(base_ledger["event_id"].astype(str))
    checks["base_population_independently_reconstructed"] = rebuilt_ids == stored_ids
    errors["base_population_symmetric_difference"] = len(rebuilt_ids ^ stored_ids)
    snapshot_identity = {
        f"{event_id}|checkpoint={checkpoint}" for event_id in stored_ids for checkpoint in (1, 2, 3)
    }
    checks["snapshot_identity_complete"] = snapshot_identity == set(
        snapshot_ledger["snapshot_id"].astype(str)
    )
    signal_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "state",
        "bar_ordinal",
        "clock_quartile",
        "anchor_high",
        "anchor_low",
        "anchor_close",
        "compression_ratio",
        "trend_return_6",
        "confirmation_step",
        "strong_close",
        "body_fraction",
        "outer_fraction",
        "trend_aligned",
        "setup",
        "family",
        "horizon",
        "status",
        "direction",
        "entry_step",
    ]
    source_signals = pd.read_parquet(
        Path(contract["inputs"]["accepted_setup_signals_2024"]), columns=signal_columns
    )
    source_signals = source_signals.loc[
        source_signals["setup"].eq(contract["population"]["setup"])
        & source_signals["family"].eq(contract["population"]["family"])
        & source_signals["horizon"].eq(24)
        & source_signals["status"].eq("filled")
        & source_signals["symbol_norm"].isin(contract["population"]["symbols"])
    ].copy()
    source_signals["event_id"] = source_signals.apply(
        lambda row: (
            f"eq|2024|{row.symbol_norm}|{row.session_date}|"
            f"{int(row.anchor_id)}|{int(row.direction)}"
        ),
        axis=1,
    )
    signal_lookup = source_signals.set_index("event_id")
    anchor_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "previous_state_1",
        "current_bar_log_return",
        "return_sum_6",
        "mean_abs_return_12",
        "session_return",
        "bar_range_pct",
        *LOOP_COLUMNS,
    ]
    source_anchors = pd.read_parquet(
        Path(contract["inputs"]["anchor_panel_2024"]), columns=anchor_columns
    )
    anchor_lookup = source_anchors.set_index(["anchor_id", "symbol_norm", "session_date"])
    cycles = pd.read_csv(Path(contract["inputs"]["fixed_cycles"]))["cycle_id"].to_numpy(str)
    base_causal_error = 0.0
    base_categorical_errors = 0
    for row in base_ledger.itertuples(index=False):
        source = signal_lookup.loc[row.event_id]
        anchor = anchor_lookup.loc[(int(row.anchor_id), row.symbol_norm, row.session_date)]
        tape = tapes[(row.symbol_norm, row.session_date)]
        anchor_matches = tape.index[
            pd.to_datetime(tape["timestamp"], utc=True).eq(pd.Timestamp(row.anchor_timestamp))
        ].to_numpy(int)
        if len(anchor_matches) != 1:
            base_categorical_errors += 1
            continue
        anchor_ordinal = int(anchor_matches[0])
        decision_ordinal = anchor_ordinal + int(source.entry_step)
        anchor_bar = tape.iloc[anchor_ordinal]
        decision_bar = tape.iloc[decision_ordinal]
        entry_bar = tape.iloc[decision_ordinal + 1]
        direction = int(source.direction)
        entry = float(entry_bar["open"])
        stop = float(source.anchor_low if direction == 1 else source.anchor_high)
        risk_price = direction * (entry - stop)
        risk_bps = 10000.0 * risk_price / entry
        target = entry + direction * risk_price
        atr_anchor = float(anchor_bar["atr14_prior"])
        atr_decision = float(decision_bar["atr14_prior"])
        expected_numeric = {
            "anchor_ordinal": anchor_ordinal,
            "source_bar_ordinal": int(source.bar_ordinal),
            "decision_ordinal": decision_ordinal,
            "entry_ordinal": decision_ordinal + 1,
            "entry_open": entry,
            "stop_price": stop,
            "target_price": target,
            "base_risk_bps": risk_bps,
            "entry_step": int(source.entry_step),
            "decision_session_fraction": decision_ordinal / 77.0,
            "anchor_range_prior_atr": (float(source.anchor_high) - float(source.anchor_low))
            / atr_anchor,
            "decision_range_prior_atr": float(decision_bar["bar_range"]) / atr_decision,
            "decision_body_fraction": float(decision_bar["body_fraction_calc"]),
            "decision_upper_wick_fraction": float(decision_bar["upper_wick_fraction_calc"]),
            "decision_lower_wick_fraction": float(decision_bar["lower_wick_fraction_calc"]),
            "decision_close_location": float(decision_bar["close_location_calc"]),
            "directional_decision_displacement_prior_atr": direction
            * (float(decision_bar["close"]) - float(source.anchor_close))
            / atr_decision,
            "decision_vwap_distance_prior_atr": direction
            * (float(decision_bar["close"]) - float(decision_bar["session_vwap"]))
            / atr_decision,
            "decision_activity_ratio": float(decision_bar["activity_ratio"]),
            "compression_ratio": float(source.compression_ratio),
            "trend_return_6": float(source.trend_return_6),
            "source_body_fraction": float(source.body_fraction),
            "source_outer_fraction": float(source.outer_fraction),
            "current_bar_log_return": float(anchor.current_bar_log_return),
            "return_sum_6": float(anchor.return_sum_6),
            "mean_abs_return_12": float(anchor.mean_abs_return_12),
            "session_return": float(anchor.session_return),
            "bar_range_pct": float(anchor.bar_range_pct),
        }
        scores = np.asarray([float(anchor[name]) for name in LOOP_COLUMNS])
        mass = float(scores.sum())
        weights = scores / mass
        nonzero = weights[weights > 0]
        order = np.argsort(scores, kind="stable")
        expected_numeric.update(
            {
                **{name: float(value) for name, value in zip(LOOP_COLUMNS, scores, strict=True)},
                "loop_score_mass": mass,
                "loop_score_entropy": float(-(nonzero * np.log(nonzero)).sum()),
                "top_loop_score": float(scores[order[-1]]),
                "top_loop_margin": float(scores[order[-1]] - scores[order[-2]]),
            }
        )
        for name, expected in expected_numeric.items():
            base_causal_error = max(
                base_causal_error, numeric_difference(getattr(row, name), expected)
            )
        expected_timestamps = {
            "decision_timestamp": pd.Timestamp(decision_bar["timestamp"]),
            "entry_timestamp": pd.Timestamp(entry_bar["timestamp"]),
        }
        for name, expected in expected_timestamps.items():
            base_categorical_errors += int(pd.Timestamp(getattr(row, name)) != expected)
        base_categorical_errors += int(row.top_loop_label != str(cycles[int(order[-1])]))
    checks["base_causal_features_independently_reconstructed"] = bool(
        base_causal_error <= 1e-10 and base_categorical_errors == 0
    )
    errors["base_categorical"] = base_categorical_errors
    snapshot_causal_error = 0.0
    snapshot_categorical_errors = 0
    for row in snapshot_ledger.itertuples(index=False):
        tape = tapes[(row.symbol_norm, row.session_date)]
        decision_ordinal = int(row.decision_ordinal) + int(row.checkpoint)
        entry_ordinal = decision_ordinal + 1
        completed = tape.iloc[int(row.entry_ordinal) : decision_ordinal + 1]
        current = tape.iloc[decision_ordinal]
        direction = int(row.direction)
        entry = float(row.entry_open)
        if direction == 1:
            favourable = 10000.0 * (completed["high"].to_numpy(float) / entry - 1.0)
            adverse = 10000.0 * (1.0 - completed["low"].to_numpy(float) / entry)
            invalidated = bool(completed["low"].le(float(row.stop_price)).any())
        else:
            favourable = 10000.0 * (1.0 - completed["low"].to_numpy(float) / entry)
            adverse = 10000.0 * (completed["high"].to_numpy(float) / entry - 1.0)
            invalidated = bool(completed["high"].ge(float(row.stop_price)).any())
        sequential_entry = float(tape.iloc[entry_ordinal]["open"])
        risk_price = direction * (sequential_entry - float(row.stop_price))
        risk_bps = 10000.0 * risk_price / sequential_entry if risk_price > 0 else math.nan
        eligible = bool(
            not invalidated
            and math.isfinite(risk_bps)
            and 20.0 <= risk_bps <= 250.0
            and entry_ordinal + 23 < len(tape)
        )
        closes = completed["close"].to_numpy(float)
        favourable_close_fraction = (
            float(direction * (closes[-1] / entry - 1.0) > 0.0)
            if len(closes) <= 1
            else float(np.mean(direction * np.diff(closes) > 0.0))
        )
        close_return = 10000.0 * direction * (float(current["close"]) / entry - 1.0)
        mfe = float(favourable.max())
        atr = float(current["atr14_prior"])
        expected_numeric = {
            "checkpoint_decision_ordinal": decision_ordinal,
            "sequential_entry_ordinal": entry_ordinal,
            "sequential_entry_open": sequential_entry,
            "sequential_stop_price": float(row.stop_price),
            "sequential_target_price": sequential_entry + direction * risk_price,
            "sequential_risk_bps": risk_bps,
            "directional_close_return_bps": close_return,
            "running_mfe_bps": mfe,
            "running_mae_bps": float(adverse.max()),
            "causal_retracement_bps": mfe - close_return,
            "current_range_prior_atr": float(current["bar_range"]) / atr,
            "current_body_fraction": float(current["body_fraction_calc"]),
            "current_upper_wick_fraction": float(current["upper_wick_fraction_calc"]),
            "current_lower_wick_fraction": float(current["lower_wick_fraction_calc"]),
            "directional_vwap_distance_prior_atr": direction
            * (float(current["close"]) - float(current["session_vwap"]))
            / atr,
            "current_activity_ratio": float(current["activity_ratio"]),
            "favourable_close_fraction": favourable_close_fraction,
        }
        for name, expected in expected_numeric.items():
            snapshot_causal_error = max(
                snapshot_causal_error, numeric_difference(getattr(row, name), expected)
            )
        snapshot_categorical_errors += int(bool(row.eligible_for_sequential_admission) != eligible)
        snapshot_categorical_errors += int(
            bool(row.invalidation_observed_before_checkpoint) != invalidated
        )
        snapshot_categorical_errors += int(
            pd.Timestamp(row.checkpoint_decision_timestamp) != pd.Timestamp(current["timestamp"])
        )
        snapshot_categorical_errors += int(
            pd.Timestamp(row.sequential_entry_timestamp)
            != pd.Timestamp(tape.iloc[entry_ordinal]["timestamp"])
        )
    checks["sequential_causal_features_independently_reconstructed"] = bool(
        snapshot_causal_error <= 1e-10 and snapshot_categorical_errors == 0
    )
    errors["snapshot_categorical"] = snapshot_categorical_errors
    base_outcomes = pd.read_parquet(artifact / "base_outcomes.parquet")
    snapshot_outcomes = pd.read_parquet(artifact / "snapshot_outcomes.parquet")
    base_errors = 0
    base_numeric_error = 0.0
    for row in base_outcomes.itertuples(index=False):
        replay = replay_path(row, tapes[(row.symbol_norm, row.session_date)], False)
        base_errors += int(replay["hit_type"] != row.hit_type)
        base_errors += int(bool(replay["target_first"]) != bool(row.target_first))
        base_numeric_error = max(
            base_numeric_error,
            abs(float(replay["gross_bps"]) - float(row.gross_bps)),
            abs(float(replay["net_bps"]) - float(row.net_bps)),
        )
    snapshot_errors = 0
    snapshot_numeric_error = 0.0
    for row in snapshot_outcomes.itertuples(index=False):
        replay = replay_path(row, tapes[(row.symbol_norm, row.session_date)], True)
        snapshot_errors += int(replay["hit_type"] != row.hit_type)
        snapshot_errors += int(bool(replay["target_first"]) != bool(row.target_first))
        snapshot_numeric_error = max(
            snapshot_numeric_error,
            abs(float(replay["gross_bps"]) - float(row.gross_bps)),
            abs(float(replay["net_bps"]) - float(row.net_bps)),
        )
    checks["all_base_and_snapshot_paths_replayed"] = bool(
        base_errors == 0
        and snapshot_errors == 0
        and base_numeric_error <= 1e-10
        and snapshot_numeric_error <= 1e-10
    )
    errors["base_path"] = base_errors
    errors["snapshot_path"] = snapshot_errors
    direct_stored = pd.read_parquet(artifact / "direct_predictions.parquet")
    direct_raw = refit_raw(base_outcomes, contract, DIRECT_MODELS, "event_id")
    direct_rebuilt = refit_calibration(direct_raw, base_outcomes, contract, "event_id", False)
    direct_compare = direct_stored.merge(
        direct_rebuilt,
        on=["event_id", "model"],
        suffixes=("_stored", "_rebuilt"),
        validate="one_to_one",
    )
    direct_probability_error = max_numeric_error(
        direct_compare,
        direct_compare.rename(
            columns={
                "raw_probability_stored": "raw_probability",
                "calibrated_probability_mean_stored": "calibrated_probability_mean",
                "calibrated_probability_lower_stored": "calibrated_probability_lower",
            }
        ),
        [],
    )
    direct_probability_error = 0.0
    for name in (
        "raw_probability",
        "calibrated_probability_mean",
        "calibrated_probability_lower",
    ):
        direct_probability_error = max(
            direct_probability_error,
            float(
                np.nanmax(
                    np.abs(
                        direct_compare[f"{name}_stored"].to_numpy(float)
                        - direct_compare[f"{name}_rebuilt"].to_numpy(float)
                    )
                )
            ),
        )
    checks["direct_equations_refit_and_recalibrated"] = bool(
        len(direct_compare) == len(direct_stored) and direct_probability_error <= 1e-12
    )
    sequential_stored = pd.read_parquet(artifact / "sequential_predictions.parquet")
    sequential_variants = {
        "sequential_confirmation": (
            (*CONTEXT_NUMERIC, *LOOP_NUMERIC, *SEQUENTIAL_NUMERIC),
            (*CONTEXT_CATEGORICAL, "top_loop_label"),
        )
    }
    sequential_raw = refit_raw(snapshot_outcomes, contract, sequential_variants, "snapshot_id")
    sequential_rebuilt = refit_calibration(
        sequential_raw, snapshot_outcomes, contract, "snapshot_id", True
    )
    sequential_compare = sequential_stored.merge(
        sequential_rebuilt,
        on=["snapshot_id", "model", "checkpoint"],
        suffixes=("_stored", "_rebuilt"),
        validate="one_to_one",
    )
    sequential_probability_error = 0.0
    for name in (
        "raw_probability",
        "calibrated_probability_mean",
        "calibrated_probability_lower",
    ):
        sequential_probability_error = max(
            sequential_probability_error,
            float(
                np.nanmax(
                    np.abs(
                        sequential_compare[f"{name}_stored"].to_numpy(float)
                        - sequential_compare[f"{name}_rebuilt"].to_numpy(float)
                    )
                )
            ),
        )
    checks["sequential_equation_refit_and_recalibrated"] = bool(
        len(sequential_compare) == len(sequential_stored) and sequential_probability_error <= 1e-12
    )
    policy = pd.read_parquet(artifact / "policy_rows.parquet")
    policy_logic_errors = 0
    for (event_id, policy_type), group in policy.loc[
        policy["model"].eq("sequential_confirmation")
    ].groupby(["event_id", "policy_type"], sort=False):
        predictions = sequential_stored.loc[sequential_stored["event_id"].eq(event_id)]
        column = (
            "conservative_selected"
            if policy_type == "conservative_uncertainty_aware"
            else "point_selected"
        )
        selected = predictions.loc[predictions[column].astype(bool)].sort_values("checkpoint")
        stored = group.iloc[0]
        expected_checkpoint = int(selected.iloc[0]["checkpoint"]) if len(selected) else -1
        policy_logic_errors += int(int(stored["selected_checkpoint"]) != expected_checkpoint)
    checks["earliest_sequential_policy_replayed"] = policy_logic_errors == 0
    errors["policy_logic"] = policy_logic_errors
    artifact_manifest = json.loads(
        (artifact / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    manifest_errors = 0
    for item in artifact_manifest["files"]:
        path = artifact / item["name"]
        manifest_errors += int(
            not path.exists()
            or path.stat().st_size != int(item["bytes"])
            or sha256(path) != item["sha256"]
        )
    checks["artifact_manifest_complete_and_valid"] = manifest_errors == 0
    errors["artifact_manifest"] = manifest_errors
    allowed_path_features = {"running_mfe_bps", "running_mae_bps"}
    checks["no_loop_or_context_outcome_leakage"] = bool(
        not any(
            token in feature.lower()
            for feature in (*CONTEXT_NUMERIC, *LOOP_NUMERIC, *SEQUENTIAL_NUMERIC)
            if feature not in allowed_path_features
            for token in FORBIDDEN
        )
    )
    result = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "errors": errors,
        "maximum_errors": {
            "base_causal": base_causal_error,
            "snapshot_causal": snapshot_causal_error,
            "base_path_numeric": base_numeric_error,
            "snapshot_path_numeric": snapshot_numeric_error,
            "direct_probability": direct_probability_error,
            "sequential_probability": sequential_probability_error,
        },
    }
    (artifact / "independent_audit.json").write_text(
        json.dumps(safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all(checks.values()):
        raise AssertionError(json.dumps(safe(result), indent=2, sort_keys=True))
    print(
        json.dumps(
            {"artifact": str(artifact), "passed": result["passed"], "total": result["total"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
