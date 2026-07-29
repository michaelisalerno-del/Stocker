"""Research-only causal tests of selective payoff equations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-selective-payoff-equations-v1.json"
AUDITOR_PATH = HERE / "audit_selective_payoff_equations_v1.py"
TEST_PATH = HERE / "tests/test_selective_payoff_equations_v1.py"

SEED = 20260714
ROUND_TRIP_COST_BPS = 10.0
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
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

DIRECT_MODELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "prediction_only": (LOOP_NUMERIC, PREDICTION_CATEGORICAL),
    "context_only": (CONTEXT_NUMERIC, CONTEXT_CATEGORICAL),
    "context_plus_loop_mixture": (
        (*CONTEXT_NUMERIC, *LOOP_NUMERIC),
        (*CONTEXT_CATEGORICAL, "top_loop_label"),
    ),
}
SEQUENTIAL_MODEL = "sequential_confirmation"
POLICIES = ("conservative_uncertainty_aware", "calibrated_point_mean_nonqualifying")
FORBIDDEN_LEDGER_TOKENS = (
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    safety = contract["safety"]
    if not safety["research_only"] or safety["live_ordering_enabled"]:
        raise AssertionError("research-only safety contract drift")
    if safety["order_placement"] != "disabled":
        raise AssertionError("order placement must remain disabled")
    return contract


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    inputs = contract["inputs"]
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "auditor": AUDITOR_PATH,
        "tests": TEST_PATH,
        "accepted_setup_signals_2024": Path(inputs["accepted_setup_signals_2024"]),
        "anchor_panel_2024": Path(inputs["anchor_panel_2024"]),
        "fixed_cycles": Path(inputs["fixed_cycles"]),
    }
    parent = (
        HERE
        / "artifacts/20260714-first-class-profitable-move-detectors-v1/primary"
        / "artifact_manifest.json"
    ).resolve()
    paths["parent_first_class_artifact_manifest"] = parent
    root = Path(inputs["provider_root_2024"])
    for symbol in contract["population"]["symbols"]:
        paths[f"provider_2024_{symbol}"] = provider_path(root, symbol)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen sources: {missing}")
    return paths


def load_tapes(
    contract: dict[str, Any],
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    root = Path(contract["inputs"]["provider_root_2024"])
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    for symbol in contract["population"]["symbols"]:
        path = provider_path(root, symbol)
        frame = pd.read_parquet(
            path, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if frame["timestamp"].duplicated().any():
            raise AssertionError(f"duplicate provider timestamp for {symbol}")
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        regular = minute.ge(570) & minute.lt(960) & local.dt.year.eq(2024)
        period_rows = frame.loc[regular].copy()
        period_local = period_rows["timestamp"].dt.tz_convert("America/New_York")
        period_minute = period_local.dt.hour * 60 + period_local.dt.minute
        on_grid = (
            period_local.dt.second.eq(0)
            & period_local.dt.microsecond.eq(0)
            & (period_minute - 570).mod(5).eq(0)
        )
        off_grid_rows = int((~on_grid).sum())
        period_rows = period_rows.loc[on_grid].copy()
        local = period_rows["timestamp"].dt.tz_convert("America/New_York")
        period_rows["session_date"] = local.dt.strftime("%Y-%m-%d")
        numeric = ["open", "high", "low", "close", "volume"]
        period_rows[numeric] = period_rows[numeric].apply(pd.to_numeric, errors="coerce")
        valid = (
            period_rows[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & period_rows["volume"].ge(0)
            & period_rows["high"].ge(period_rows[["open", "close"]].max(axis=1))
            & period_rows["low"].le(period_rows[["open", "close"]].min(axis=1))
        )
        invalid_rows = int((~valid).sum())
        period_rows = period_rows.loc[valid].copy().reset_index(drop=True)
        previous_close = period_rows["close"].shift(1)
        true_range = np.maximum.reduce(
            [
                (period_rows["high"] - period_rows["low"]).to_numpy(float),
                (period_rows["high"] - previous_close).abs().to_numpy(float),
                (period_rows["low"] - previous_close).abs().to_numpy(float),
            ]
        )
        period_rows["true_range"] = true_range
        period_rows["atr14_prior"] = (
            pd.Series(true_range, index=period_rows.index)
            .shift(1)
            .rolling(14, min_periods=14)
            .mean()
        )
        period_rows["bar_range"] = period_rows["high"] - period_rows["low"]
        safe_range = period_rows["bar_range"].replace(0.0, np.nan)
        period_rows["body_fraction_calc"] = (
            period_rows["close"] - period_rows["open"]
        ).abs() / safe_range
        period_rows["upper_wick_fraction_calc"] = (
            period_rows["high"] - period_rows[["open", "close"]].max(axis=1)
        ) / safe_range
        period_rows["lower_wick_fraction_calc"] = (
            period_rows[["open", "close"]].min(axis=1) - period_rows["low"]
        ) / safe_range
        period_rows["close_location_calc"] = (
            period_rows["close"] - period_rows["low"]
        ) / safe_range
        period_rows["typical_price"] = (
            period_rows["high"] + period_rows["low"] + period_rows["close"]
        ) / 3.0
        period_rows["pv"] = period_rows["typical_price"] * period_rows["volume"]
        cumulative_pv = period_rows.groupby("session_date", sort=False)["pv"].cumsum()
        cumulative_volume = period_rows.groupby("session_date", sort=False)["volume"].cumsum()
        period_rows["session_vwap"] = cumulative_pv / cumulative_volume.replace(0.0, np.nan)
        log_volume = np.log(period_rows["volume"].where(period_rows["volume"].gt(0)))
        prior_log_mean = log_volume.groupby(period_rows["session_date"], sort=False).transform(
            lambda series: series.shift(1).rolling(6, min_periods=3).mean()
        )
        period_rows["activity_ratio"] = period_rows["volume"] / np.exp(prior_log_mean)
        period_rows["bar_ordinal"] = period_rows.groupby("session_date", sort=False).cumcount()
        for session_date, session in period_rows.groupby("session_date", sort=False):
            clean = session.sort_values("bar_ordinal", kind="stable").reset_index(drop=True)
            if not np.array_equal(clean["bar_ordinal"].to_numpy(int), np.arange(len(clean))):
                raise AssertionError("provider session ordinal drift")
            groups[(symbol, str(session_date))] = clean
        coverage_rows.append(
            {
                "symbol": symbol,
                "rows": len(period_rows),
                "sessions": period_rows["session_date"].nunique(),
                "first_timestamp": period_rows["timestamp"].min(),
                "last_timestamp": period_rows["timestamp"].max(),
                "off_grid_rows_excluded": off_grid_rows,
                "invalid_rows_excluded": invalid_rows,
                "atr14_available": int(period_rows["atr14_prior"].notna().sum()),
                "activity_ratio_available": int(period_rows["activity_ratio"].notna().sum()),
            }
        )
    return groups, pd.DataFrame(coverage_rows)


def labelled(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)


def event_features(
    signal: Any,
    anchor: Any,
    tape: pd.DataFrame,
    cycles: np.ndarray,
) -> dict[str, Any] | None:
    anchor_matches = tape.index[
        pd.to_datetime(tape["timestamp"], utc=True).eq(pd.Timestamp(signal.start_timestamp))
    ].to_numpy(int)
    if len(anchor_matches) != 1:
        return None
    anchor_ordinal = int(anchor_matches[0])
    decision_ordinal = anchor_ordinal + int(signal.entry_step)
    if int(signal.entry_step) != int(signal.confirmation_step):
        raise AssertionError("entry and confirmation step drift")
    if decision_ordinal > 50 or decision_ordinal + 1 >= len(tape):
        return None
    anchor_bar = tape.iloc[anchor_ordinal]
    decision_bar = tape.iloc[decision_ordinal]
    entry_bar = tape.iloc[decision_ordinal + 1]
    if pd.Timestamp(anchor_bar["timestamp"]) != pd.Timestamp(signal.start_timestamp):
        raise AssertionError("exact provider anchor mapping drift")
    direction = int(signal.direction)
    if direction not in (-1, 1):
        raise AssertionError("invalid source direction")
    stop = float(signal.anchor_low if direction == 1 else signal.anchor_high)
    entry = float(entry_bar["open"])
    risk_price = direction * (entry - stop)
    if risk_price <= 0:
        return None
    risk_bps = 10000.0 * risk_price / entry
    if not 20.0 <= risk_bps <= 250.0:
        return None
    atr_anchor = float(anchor_bar["atr14_prior"])
    atr_decision = float(decision_bar["atr14_prior"])
    if (
        not math.isfinite(atr_anchor)
        or atr_anchor <= 0
        or not math.isfinite(atr_decision)
        or atr_decision <= 0
    ):
        return None
    scores = np.asarray([float(getattr(anchor, name)) for name in LOOP_COLUMNS], dtype=float)
    if (scores < 0).any() or not math.isfinite(float(scores.sum())) or scores.sum() <= 0:
        raise AssertionError("invalid loop score vector")
    mass = float(scores.sum())
    weights = scores / mass
    nonzero = weights[weights > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    order = np.argsort(scores, kind="stable")
    top_index = int(order[-1])
    top_score = float(scores[top_index])
    margin = top_score - float(scores[order[-2]])
    target = entry + direction * risk_price
    session_fraction = decision_ordinal / 77.0
    source_values = {
        "compression_ratio": float(signal.compression_ratio),
        "trend_return_6": float(signal.trend_return_6),
        "source_body_fraction": float(signal.body_fraction),
        "source_outer_fraction": float(signal.outer_fraction),
        "current_bar_log_return": float(anchor.current_bar_log_return),
        "return_sum_6": float(anchor.return_sum_6),
        "mean_abs_return_12": float(anchor.mean_abs_return_12),
        "session_return": float(anchor.session_return),
        "bar_range_pct": float(anchor.bar_range_pct),
    }
    event_id = (
        f"eq|2024|{signal.symbol_norm}|{signal.session_date}|"
        f"{int(signal.anchor_id)}|{direction}"
    )
    row: dict[str, Any] = {
        "event_id": event_id,
        "symbol_norm": str(signal.symbol_norm),
        "session_date": str(signal.session_date),
        "month": str(signal.session_date)[:7],
        "anchor_id": int(signal.anchor_id),
        "anchor_timestamp": pd.Timestamp(signal.start_timestamp),
        "decision_timestamp": pd.Timestamp(decision_bar["timestamp"]),
        "entry_timestamp": pd.Timestamp(entry_bar["timestamp"]),
        "anchor_ordinal": anchor_ordinal,
        "source_bar_ordinal": int(signal.bar_ordinal),
        "decision_ordinal": decision_ordinal,
        "entry_ordinal": decision_ordinal + 1,
        "direction": direction,
        "direction_label": "long" if direction == 1 else "short",
        "state_label": labelled(signal.state),
        "previous_state_label": labelled(anchor.previous_state_1),
        "clock_quartile_label": labelled(signal.clock_quartile),
        "strong_close_label": labelled(bool(signal.strong_close)),
        "trend_aligned_label": labelled(bool(signal.trend_aligned)),
        "entry_step": int(signal.entry_step),
        "decision_session_fraction": session_fraction,
        "entry_open": entry,
        "stop_price": stop,
        "target_price": target,
        "base_risk_bps": risk_bps,
        "anchor_range_prior_atr": (float(signal.anchor_high) - float(signal.anchor_low))
        / atr_anchor,
        "decision_range_prior_atr": float(decision_bar["bar_range"]) / atr_decision,
        "decision_body_fraction": float(decision_bar["body_fraction_calc"]),
        "decision_upper_wick_fraction": float(decision_bar["upper_wick_fraction_calc"]),
        "decision_lower_wick_fraction": float(decision_bar["lower_wick_fraction_calc"]),
        "decision_close_location": float(decision_bar["close_location_calc"]),
        "directional_decision_displacement_prior_atr": direction
        * (float(decision_bar["close"]) - float(signal.anchor_close))
        / atr_decision,
        "decision_vwap_distance_prior_atr": direction
        * (float(decision_bar["close"]) - float(decision_bar["session_vwap"]))
        / atr_decision,
        "decision_activity_ratio": float(decision_bar["activity_ratio"]),
        "loop_score_mass": mass,
        "loop_score_entropy": entropy,
        "top_loop_score": top_score,
        "top_loop_margin": margin,
        "top_loop_label": str(cycles[top_index]),
        **source_values,
    }
    row.update({name: float(value) for name, value in zip(LOOP_COLUMNS, scores, strict=True)})
    return row


def apply_cooldown(events: pd.DataFrame, bars: int) -> pd.DataFrame:
    kept: list[int] = []
    for _, group in events.sort_values(
        ["symbol_norm", "session_date", "decision_ordinal", "event_id"], kind="stable"
    ).groupby(["symbol_norm", "session_date"], sort=False):
        last = -10_000
        for index, row in group.iterrows():
            decision = int(row["decision_ordinal"])
            if decision - last >= bars:
                kept.append(index)
                last = decision
    return (
        events.loc[kept]
        .sort_values(["session_date", "symbol_norm", "decision_ordinal", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def directional_excursions(
    path: pd.DataFrame, entry: float, direction: int
) -> tuple[np.ndarray, np.ndarray]:
    if direction == 1:
        favourable = 10000.0 * (path["high"].to_numpy(float) / entry - 1.0)
        adverse = 10000.0 * (1.0 - path["low"].to_numpy(float) / entry)
    else:
        favourable = 10000.0 * (1.0 - path["low"].to_numpy(float) / entry)
        adverse = 10000.0 * (path["high"].to_numpy(float) / entry - 1.0)
    return favourable, adverse


def snapshot_features(event: Any, tape: pd.DataFrame, checkpoint: int) -> dict[str, Any]:
    decision_ordinal = int(event.decision_ordinal) + checkpoint
    entry_ordinal = decision_ordinal + 1
    direction = int(event.direction)
    completed = tape.iloc[int(event.entry_ordinal) : decision_ordinal + 1]
    current = tape.iloc[decision_ordinal]
    entry = float(event.entry_open)
    favourable, adverse = directional_excursions(completed, entry, direction)
    stop = float(event.stop_price)
    invalidated = bool(
        completed["low"].le(stop).any() if direction == 1 else completed["high"].ge(stop).any()
    )
    sequential_entry = float(tape.iloc[entry_ordinal]["open"])
    risk_price = direction * (sequential_entry - stop)
    risk_bps = 10000.0 * risk_price / sequential_entry if risk_price > 0 else math.nan
    eligible = bool(
        not invalidated
        and math.isfinite(risk_bps)
        and 20.0 <= risk_bps <= 250.0
        and entry_ordinal + 23 < len(tape)
    )
    atr = float(current["atr14_prior"])
    closes = completed["close"].to_numpy(float)
    if len(closes) <= 1:
        favourable_close_fraction = float(direction * (closes[-1] / entry - 1.0) > 0.0)
    else:
        changes = direction * np.diff(closes)
        favourable_close_fraction = float(np.mean(changes > 0.0))
    close_return = 10000.0 * direction * (float(current["close"]) / entry - 1.0)
    mfe = float(favourable.max())
    row = event._asdict()
    row.update(
        {
            "snapshot_id": f"{event.event_id}|checkpoint={checkpoint}",
            "checkpoint": checkpoint,
            "checkpoint_decision_ordinal": decision_ordinal,
            "checkpoint_decision_timestamp": pd.Timestamp(current["timestamp"]),
            "sequential_entry_ordinal": entry_ordinal,
            "sequential_entry_timestamp": pd.Timestamp(tape.iloc[entry_ordinal]["timestamp"]),
            "sequential_entry_open": sequential_entry,
            "sequential_stop_price": stop,
            "sequential_target_price": sequential_entry + direction * risk_price,
            "sequential_risk_bps": risk_bps,
            "eligible_for_sequential_admission": eligible,
            "invalidation_observed_before_checkpoint": invalidated,
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
    )
    return row


def build_pre_outcome_ledgers(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tapes, coverage = load_tapes(contract)
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
        & signals["horizon"].eq(contract["population"]["source_horizon"])
        & signals["status"].eq(contract["population"]["source_status"])
        & signals["symbol_norm"].isin(contract["population"]["symbols"])
    ].copy()
    selected["start_timestamp"] = pd.to_datetime(selected["start_timestamp"], utc=True)
    selected["session_date"] = selected["session_date"].astype(str)
    calendar = sorted(selected["session_date"].unique())
    if len(calendar) != int(contract["population"]["surface_sessions"]):
        raise AssertionError("surface calendar drift")
    calendar_index = {date: index for index, date in enumerate(calendar)}
    anchor_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "state",
        "previous_state_1",
        "current_bar_log_return",
        "return_sum_6",
        "mean_abs_return_12",
        "session_return",
        "bar_range_pct",
        *LOOP_COLUMNS,
    ]
    anchors = pd.read_parquet(Path(contract["inputs"]["anchor_panel_2024"]), columns=anchor_columns)
    anchors["start_timestamp"] = pd.to_datetime(anchors["start_timestamp"], utc=True)
    anchors["session_date"] = anchors["session_date"].astype(str)
    joined = selected.merge(
        anchors,
        on=["anchor_id", "symbol_norm", "session_date", "start_timestamp", "state"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_anchor"),
    )
    if joined[LOOP_COLUMNS[0]].isna().any():
        raise AssertionError("anchor loop-score join drift")
    cycles = pd.read_csv(Path(contract["inputs"]["fixed_cycles"]))["cycle_id"].to_numpy(str)
    if len(cycles) != 20:
        raise AssertionError("cycle dictionary drift")
    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for signal in joined.itertuples(index=False):
        counters["selected_source_rows"] += 1
        tape = tapes.get((str(signal.symbol_norm), str(signal.session_date)))
        if tape is None:
            counters["missing_provider_session"] += 1
            continue
        features = event_features(signal, signal, tape, cycles)
        if features is None:
            counters["failed_clock_atr_or_risk_filter"] += 1
            continue
        features["calendar_index"] = calendar_index[str(signal.session_date)]
        rows.append(features)
    events = pd.DataFrame(rows)
    before_cooldown = len(events)
    events = apply_cooldown(events, int(contract["population"]["cooldown_bars_per_symbol_session"]))
    counters["pre_cooldown_rows"] = before_cooldown
    counters["post_cooldown_rows"] = len(events)
    snapshots: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        tape = tapes[(str(event.symbol_norm), str(event.session_date))]
        for checkpoint in contract["causal_clock"]["sequential_checkpoints"]:
            snapshots.append(snapshot_features(event, tape, int(checkpoint)))
    snapshot_frame = (
        pd.DataFrame(snapshots)
        .sort_values(
            ["session_date", "symbol_norm", "decision_ordinal", "checkpoint"], kind="stable"
        )
        .reset_index(drop=True)
    )
    counts = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in sorted(counters.items())]
        + [
            {"metric": "base_sessions", "value": events["session_date"].nunique()},
            {"metric": "base_symbols", "value": events["symbol_norm"].nunique()},
            {"metric": "sequential_snapshots", "value": len(snapshot_frame)},
            {
                "metric": "eligible_sequential_snapshots",
                "value": int(snapshot_frame["eligible_for_sequential_admission"].sum()),
            },
        ]
    )
    return events, snapshot_frame, coverage, counts


def assert_pre_outcome_columns(
    frame: pd.DataFrame, *, allow_completed_path_excursions: bool = False
) -> None:
    allowed = (
        {"running_mfe_bps", "running_mae_bps"}
        if allow_completed_path_excursions
        else set()
    )
    bad = [
        column
        for column in frame.columns
        if column not in allowed
        and any(token in column.lower() for token in FORBIDDEN_LEDGER_TOKENS)
    ]
    if bad:
        raise AssertionError(f"pre-outcome ledger contains forbidden columns: {bad}")


def pre_score_manifest(out: Path, contract: dict[str, Any]) -> dict[str, Any]:
    sources = source_paths(contract)
    return {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scientific_status": contract["scientific_status"],
        "source_hashes": {
            name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in sorted(sources.items())
        },
        "frozen_ledgers": {
            name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
            for name in (
                "pre_outcome_base_events.parquet",
                "pre_outcome_sequential_snapshots.parquet",
                "data_coverage.csv",
                "population_counts.csv",
            )
        },
    }


def prepare_events(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    contract = load_contract()
    out.mkdir(parents=True)
    events, snapshots, coverage, counts = build_pre_outcome_ledgers(contract)
    assert_pre_outcome_columns(events)
    assert_pre_outcome_columns(snapshots, allow_completed_path_excursions=True)
    events.to_parquet(out / "pre_outcome_base_events.parquet", index=False)
    snapshots.to_parquet(out / "pre_outcome_sequential_snapshots.parquet", index=False)
    coverage.to_csv(out / "data_coverage.csv", index=False)
    counts.to_csv(out / "population_counts.csv", index=False)
    write_json(out / "pre_score_manifest.json", pre_score_manifest(out, contract))
    print(
        json.dumps(
            {
                "frozen": str(out),
                "base_events": len(events),
                "sequential_snapshots": len(snapshots),
                "eligible_sequential_snapshots": int(
                    snapshots["eligible_for_sequential_admission"].sum()
                ),
            },
            indent=2,
        )
    )


def verify_frozen(frozen: Path, contract: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((frozen / "pre_score_manifest.json").read_text(encoding="utf-8"))
    if manifest["contract_id"] != contract["contract_id"]:
        raise AssertionError("frozen contract id drift")
    for name, item in manifest["source_hashes"].items():
        path = Path(item["path"])
        if not path.exists() or sha256(path) != item["sha256"]:
            raise AssertionError(f"frozen source hash drift: {name}")
    for name, item in manifest["frozen_ledgers"].items():
        path = frozen / name
        if not path.exists() or sha256(path) != item["sha256"]:
            raise AssertionError(f"frozen ledger hash drift: {name}")
    return manifest


def score_path(
    row: Any,
    tape: pd.DataFrame,
    *,
    entry_ordinal_field: str,
    entry_open_field: str,
    stop_field: str,
    target_field: str,
    risk_field: str,
) -> dict[str, Any]:
    entry_ordinal = int(getattr(row, entry_ordinal_field))
    positions = np.arange(entry_ordinal, entry_ordinal + 24)
    if positions[-1] >= len(tape):
        return {"outcome_status": "missing_exact_future_path"}
    future = tape.iloc[positions]
    entry = float(getattr(row, entry_open_field))
    if not math.isclose(entry, float(future.iloc[0]["open"]), rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("next-open replay mismatch")
    direction = int(row.direction)
    stop = float(getattr(row, stop_field))
    target = float(getattr(row, target_field))
    risk_bps = float(getattr(row, risk_field))
    hit_type = "no_touch_time_exit"
    hit_step: int | None = None
    exit_price = float(future.iloc[-1]["close"])
    for step, bar in enumerate(future.itertuples(index=False), start=1):
        open_price = float(bar.open)
        if direction == 1:
            if open_price >= target:
                hit_type, hit_step, exit_price = "target_gap_or_open_first", step, target
                break
            if open_price <= stop:
                hit_type, hit_step, exit_price = "stop_gap_or_open_first", step, open_price
                break
            target_touch = float(bar.high) >= target
            stop_touch = float(bar.low) <= stop
        else:
            if open_price <= target:
                hit_type, hit_step, exit_price = "target_gap_or_open_first", step, target
                break
            if open_price >= stop:
                hit_type, hit_step, exit_price = "stop_gap_or_open_first", step, open_price
                break
            target_touch = float(bar.low) <= target
            stop_touch = float(bar.high) >= stop
        if target_touch and stop_touch:
            hit_type, hit_step, exit_price = "dual_touch_conservative_stop", step, stop
            break
        if target_touch:
            hit_type, hit_step, exit_price = "target_first", step, target
            break
        if stop_touch:
            hit_type, hit_step, exit_price = "stop_first", step, stop
            break
    target_first = hit_type in {"target_gap_or_open_first", "target_first"}
    gross = 10000.0 * direction * (exit_price / entry - 1.0)
    favourable, adverse = directional_excursions(future, entry, direction)
    return {
        "outcome_status": "scored",
        "hit_type": hit_type,
        "hit_step": hit_step,
        "target_first": target_first,
        "gross_bps": gross,
        "net_bps": gross - ROUND_TRIP_COST_BPS,
        "mfe_bps": float(favourable.max()),
        "mae_bps": float(adverse.max()),
        "time_to_mfe": int(favourable.argmax() + 1),
        "time_to_mae": int(adverse.argmax() + 1),
        "risk_bps_replayed": risk_bps,
    }


def attach_outcomes(
    ledger: pd.DataFrame,
    tapes: dict[tuple[str, str], pd.DataFrame],
    *,
    id_column: str,
    entry_ordinal_field: str,
    entry_open_field: str,
    stop_field: str,
    target_field: str,
    risk_field: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in ledger.itertuples(index=False):
        outcome = score_path(
            row,
            tapes[(str(row.symbol_norm), str(row.session_date))],
            entry_ordinal_field=entry_ordinal_field,
            entry_open_field=entry_open_field,
            stop_field=stop_field,
            target_field=target_field,
            risk_field=risk_field,
        )
        rows.append({id_column: getattr(row, id_column), **outcome})
    return ledger.merge(pd.DataFrame(rows), on=id_column, how="left", validate="one_to_one")


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
    spec = contract["model"]
    model = LogisticRegression(
        C=float(spec["C"]),
        solver=str(spec["solver"]),
        max_iter=int(spec["maximum_iterations"]),
        random_state=SEED,
    )
    return Pipeline([("features", transform), ("model", model)])


def prediction_identity_columns(id_column: str) -> list[str]:
    columns = [id_column]
    if id_column != "event_id":
        columns.append("event_id")
    return columns


def prequential_predictions(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    variants: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    *,
    id_column: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    spec = contract["model"]
    first = int(spec["first_raw_prediction_session_index"])
    rolling = int(spec["rolling_training_sessions"])
    minimum_sessions = int(spec["minimum_training_sessions"])
    minimum_rows = int(spec["minimum_training_rows"])
    max_index = int(frame["calendar_index"].max())
    for calendar_index in range(first, max_index + 1):
        start = max(0, calendar_index - rolling)
        train = frame.loc[
            frame["calendar_index"].ge(start) & frame["calendar_index"].lt(calendar_index)
        ].copy()
        score = frame.loc[frame["calendar_index"].eq(calendar_index)].copy()
        if score.empty or train.empty:
            continue
        if train["session_date"].nunique() < minimum_sessions or len(train) < minimum_rows:
            continue
        if train["target_first"].nunique() != 2:
            continue
        for model_name, (numeric, categorical) in variants.items():
            pipeline = model_pipeline(numeric, categorical, contract)
            pipeline.fit(
                train[list(numeric) + list(categorical)], train["target_first"].astype(int)
            )
            probability = pipeline.predict_proba(score[list(numeric) + list(categorical)])[:, 1]
            output_columns = [
                *prediction_identity_columns(id_column),
                "symbol_norm",
                "session_date",
                "month",
                "calendar_index",
                "target_first",
                "gross_bps",
                "net_bps",
            ]
            risk_column = "base_risk_bps" if id_column == "event_id" else "sequential_risk_bps"
            output = score[output_columns + [risk_column]].copy()
            output = output.rename(columns={risk_column: "risk_bps"})
            if "checkpoint" in score.columns:
                output["checkpoint"] = score["checkpoint"].to_numpy(int)
            output["model"] = model_name
            output["raw_probability"] = probability
            output["train_first_calendar_index"] = start
            output["train_last_calendar_index"] = calendar_index - 1
            output["train_sessions"] = train["session_date"].nunique()
            output["train_rows"] = len(train)
            rows.append(output)
    if not rows:
        return pd.DataFrame()
    sort_columns = ["model", "calendar_index", "event_id"]
    if id_column != "event_id":
        sort_columns.append("checkpoint")
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(sort_columns, kind="stable")
        .reset_index(drop=True)
    )


def calibration_values(
    current_probability: float,
    history: pd.DataFrame,
    *,
    nearest_rows: int,
    minimum_support: int,
) -> tuple[float, float, int]:
    if len(history) < minimum_support:
        return math.nan, math.nan, len(history)
    nearest = history.assign(
        distance=(history["raw_probability"] - current_probability).abs()
    ).nsmallest(min(nearest_rows, len(history)), "distance", keep="first")
    support = len(nearest)
    wins = int(nearest["target_first"].sum())
    mean = (wins + 0.5) / (support + 1.0)
    lower = float(beta.ppf(0.05, wins + 0.5, support - wins + 0.5))
    return float(mean), lower, support


def calibrate_predictions(
    predictions: pd.DataFrame,
    contract: dict[str, Any],
    *,
    sequential: bool,
) -> pd.DataFrame:
    calibration = contract["causal_probability_calibration"]
    history_sessions = int(calibration["maximum_history_sessions"])
    if sequential:
        nearest_rows = int(calibration["sequential_nearest_probability_rows_per_checkpoint"])
        minimum_support = int(calibration["sequential_minimum_support_per_checkpoint"])
    else:
        nearest_rows = int(calibration["direct_nearest_probability_rows"])
        minimum_support = int(calibration["direct_minimum_support"])
    main_first = int(contract["model"]["first_primary_score_session_index"])
    output_rows: list[dict[str, Any]] = []
    for _model_name, model_frame in predictions.groupby("model", sort=False):
        for current in model_frame.loc[model_frame["calendar_index"].ge(main_first)].itertuples(
            index=False
        ):
            history = model_frame.loc[
                model_frame["calendar_index"].lt(int(current.calendar_index))
                & model_frame["calendar_index"].ge(int(current.calendar_index) - history_sessions)
            ]
            if sequential:
                history = history.loc[history["checkpoint"].eq(int(current.checkpoint))]
            mean, lower, support = calibration_values(
                float(current.raw_probability),
                history,
                nearest_rows=nearest_rows,
                minimum_support=minimum_support,
            )
            risk = float(current.risk_bps)
            mean_ev = (
                (2.0 * mean - 1.0) * risk - ROUND_TRIP_COST_BPS if math.isfinite(mean) else math.nan
            )
            lower_ev = (
                (2.0 * lower - 1.0) * risk - ROUND_TRIP_COST_BPS
                if math.isfinite(lower)
                else math.nan
            )
            row = current._asdict()
            row.update(
                {
                    "calibrated_probability_mean": mean,
                    "calibrated_probability_lower": lower,
                    "calibration_support": support,
                    "point_expected_net_bps": mean_ev,
                    "conservative_expected_net_bps": lower_ev,
                    "point_selected": bool(math.isfinite(mean_ev) and mean_ev > 0.0),
                    "conservative_selected": bool(math.isfinite(lower_ev) and lower_ev > 0.0),
                }
            )
            output_rows.append(row)
    return (
        pd.DataFrame(output_rows)
        .sort_values(
            ["model", "calendar_index", "event_id"] + (["checkpoint"] if sequential else []),
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_policy_rows(
    base: pd.DataFrame,
    direct: pd.DataFrame,
    sequential: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    first = int(contract["model"]["first_primary_score_session_index"])
    base_score = base.loc[base["calendar_index"].ge(first)].copy()
    base_lookup = base_score.set_index("event_id")
    rows: list[dict[str, Any]] = []
    selection_columns = {
        "conservative_uncertainty_aware": "conservative_selected",
        "calibrated_point_mean_nonqualifying": "point_selected",
    }
    for prediction in direct.itertuples(index=False):
        base_row = base_lookup.loc[prediction.event_id]
        for policy_type, selected_column in selection_columns.items():
            selected = bool(getattr(prediction, selected_column))
            rows.append(
                {
                    "event_id": prediction.event_id,
                    "symbol_norm": prediction.symbol_norm,
                    "session_date": prediction.session_date,
                    "month": prediction.month,
                    "calendar_index": int(prediction.calendar_index),
                    "model": prediction.model,
                    "policy_type": policy_type,
                    "selected": selected,
                    "selected_checkpoint": 0 if selected else -1,
                    "target_first_if_selected": bool(base_row["target_first"])
                    if selected
                    else False,
                    "selected_gross_bps": float(base_row["gross_bps"]) if selected else 0.0,
                    "selected_net_bps": float(base_row["net_bps"]) if selected else 0.0,
                    "selector_return_per_opportunity_bps": (
                        float(base_row["net_bps"]) if selected else 0.0
                    ),
                    "unfiltered_base_net_bps": float(base_row["net_bps"]),
                    "risk_bps_if_selected": float(base_row["base_risk_bps"])
                    if selected
                    else math.nan,
                }
            )
    sequential_groups = {
        event_id: group.sort_values("checkpoint", kind="stable")
        for event_id, group in sequential.groupby("event_id", sort=False)
    }
    for event in base_score.itertuples(index=False):
        group = sequential_groups.get(event.event_id, pd.DataFrame())
        for policy_type, selected_column in selection_columns.items():
            candidates = (
                group.loc[group[selected_column].astype(bool)] if not group.empty else group
            )
            selected = not candidates.empty
            chosen = candidates.iloc[0] if selected else None
            rows.append(
                {
                    "event_id": event.event_id,
                    "symbol_norm": event.symbol_norm,
                    "session_date": event.session_date,
                    "month": event.month,
                    "calendar_index": int(event.calendar_index),
                    "model": SEQUENTIAL_MODEL,
                    "policy_type": policy_type,
                    "selected": selected,
                    "selected_checkpoint": int(chosen["checkpoint"]) if selected else -1,
                    "target_first_if_selected": bool(chosen["target_first"]) if selected else False,
                    "selected_gross_bps": float(chosen["gross_bps"]) if selected else 0.0,
                    "selected_net_bps": float(chosen["net_bps"]) if selected else 0.0,
                    "selector_return_per_opportunity_bps": (
                        float(chosen["net_bps"]) if selected else 0.0
                    ),
                    "unfiltered_base_net_bps": float(event.net_bps),
                    "risk_bps_if_selected": float(chosen["risk_bps"]) if selected else math.nan,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["policy_type", "model", "calendar_index", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def moving_block_bootstrap(frame: pd.DataFrame, value_column: str, *, seed: int) -> dict[str, Any]:
    daily = frame.groupby("session_date", sort=True)[value_column].agg(["sum", "count"])
    if daily.empty:
        return {
            "observed_mean": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "p_one_sided": math.nan,
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
    return {
        "observed_mean": float(frame[value_column].mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "p_one_sided": float((1 + np.sum(samples <= 0.0)) / (BOOTSTRAP_DRAWS + 1)),
        "sessions": sessions,
    }


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    valid = output["p_one_sided"].notna()
    output["holm_adjusted_p"] = math.nan
    if not valid.any():
        output["passes_holm_0_05"] = False
        return output
    indices = output.index[valid].to_numpy()
    values = output.loc[indices, "p_one_sided"].to_numpy(float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, ordered_index in enumerate(order):
        running = max(running, (len(values) - rank) * values[ordered_index])
        adjusted[ordered_index] = min(1.0, running)
    output.loc[indices, "holm_adjusted_p"] = adjusted
    output["passes_holm_0_05"] = output["holm_adjusted_p"].lt(0.05) & output["ci_lower"].gt(0.0)
    return output


def precision_lower(successes: int, rows: int) -> float:
    if rows <= 0:
        return math.nan
    return float(beta.ppf(0.05, successes + 0.5, rows - successes + 0.5))


def evaluate(
    base: pd.DataFrame,
    direct: pd.DataFrame,
    sequential: pd.DataFrame,
    policy_rows: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    for frame in (direct, sequential):
        for model_name, group in frame.groupby("model", sort=True):
            actual = group["target_first"].astype(int)
            probability = np.clip(group["raw_probability"].to_numpy(float), 1e-9, 1 - 1e-9)
            model_rows.append(
                {
                    "model": model_name,
                    "rows": len(group),
                    "sessions": group["session_date"].nunique(),
                    "target_rate": float(actual.mean()),
                    "raw_log_loss": float(log_loss(actual, probability, labels=[0, 1])),
                    "raw_brier": float(brier_score_loss(actual, probability)),
                    "raw_auc": (
                        float(roc_auc_score(actual, probability))
                        if actual.nunique() == 2
                        else math.nan
                    ),
                    "mean_raw_probability": float(probability.mean()),
                    "mean_calibrated_probability": float(
                        group["calibrated_probability_mean"].mean()
                    ),
                    "mean_calibration_support": float(group["calibration_support"].mean()),
                }
            )
    policy_metric_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for (policy_type, model_name), group in policy_rows.groupby(
        ["policy_type", "model"], sort=True
    ):
        selected = group.loc[group["selected"]].copy()
        successes = int(selected["target_first_if_selected"].sum())
        policy_metric_rows.append(
            {
                "policy_type": policy_type,
                "model": model_name,
                "base_rows": len(group),
                "selected_rows": len(selected),
                "coverage": float(len(selected) / len(group)),
                "selected_sessions": selected["session_date"].nunique(),
                "selected_stocks": selected["symbol_norm"].nunique(),
                "selected_target_first_precision": (
                    float(successes / len(selected)) if len(selected) else math.nan
                ),
                "selected_precision_lower": precision_lower(successes, len(selected)),
                "selected_mean_gross_bps": (
                    float(selected["selected_gross_bps"].mean()) if len(selected) else math.nan
                ),
                "selected_mean_net_bps": (
                    float(selected["selected_net_bps"].mean()) if len(selected) else math.nan
                ),
                "selector_mean_net_per_opportunity_bps": float(
                    group["selector_return_per_opportunity_bps"].mean()
                ),
                "unfiltered_base_mean_net_bps": float(group["unfiltered_base_net_bps"].mean()),
            }
        )
        for month, month_group in group.groupby("month", sort=True):
            month_selected = month_group.loc[month_group["selected"]]
            monthly_rows.append(
                {
                    "policy_type": policy_type,
                    "model": model_name,
                    "month": month,
                    "base_rows": len(month_group),
                    "selected_rows": len(month_selected),
                    "coverage": float(len(month_selected) / len(month_group)),
                    "selected_mean_net_bps": (
                        float(month_selected["selected_net_bps"].mean())
                        if len(month_selected)
                        else math.nan
                    ),
                    "selector_mean_net_per_opportunity_bps": float(
                        month_group["selector_return_per_opportunity_bps"].mean()
                    ),
                }
            )
        for symbol in sorted(group["symbol_norm"].unique()):
            deleted = group.loc[~group["symbol_norm"].eq(symbol)]
            deleted_selected = deleted.loc[deleted["selected"]]
            deletion_rows.append(
                {
                    "policy_type": policy_type,
                    "model": model_name,
                    "deleted_symbol": symbol,
                    "base_rows": len(deleted),
                    "selected_rows": len(deleted_selected),
                    "selected_mean_net_bps": (
                        float(deleted_selected["selected_net_bps"].mean())
                        if len(deleted_selected)
                        else math.nan
                    ),
                    "selector_mean_net_per_opportunity_bps": float(
                        deleted["selector_return_per_opportunity_bps"].mean()
                    ),
                }
            )
        for cost in (2.5, 5.0, 7.5, 10.0):
            selected_net = selected["selected_gross_bps"] - 2.0 * cost
            per_opportunity = np.where(
                group["selected"], group["selected_gross_bps"] - 2.0 * cost, 0.0
            )
            cost_rows.append(
                {
                    "policy_type": policy_type,
                    "model": model_name,
                    "cost_bps_per_side": cost,
                    "selected_rows": len(selected),
                    "selected_mean_net_bps": (
                        float(selected_net.mean()) if len(selected) else math.nan
                    ),
                    "selector_mean_net_per_opportunity_bps": float(np.mean(per_opportunity)),
                }
            )
    primary = policy_rows.loc[
        policy_rows["policy_type"].eq("conservative_uncertainty_aware")
    ].copy()
    pivot = primary.pivot(
        index=["event_id", "symbol_norm", "session_date", "month"],
        columns="model",
        values="selector_return_per_opportunity_bps",
    ).reset_index()
    comparisons = {
        "context_only_minus_prediction_only": ("context_only", "prediction_only"),
        "context_plus_loop_minus_context_only": (
            "context_plus_loop_mixture",
            "context_only",
        ),
        "sequential_confirmation_minus_context_plus_loop": (
            "sequential_confirmation",
            "context_plus_loop_mixture",
        ),
    }
    incremental_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for name, (new_model, base_model) in comparisons.items():
        values = pivot[new_model] - pivot[base_model]
        incremental_rows.append(
            {
                "comparison": name,
                "rows": len(pivot),
                "new_model_mean_per_opportunity_bps": float(pivot[new_model].mean()),
                "base_model_mean_per_opportunity_bps": float(pivot[base_model].mean()),
                "incremental_mean_bps": float(values.mean()),
            }
        )
        test = pivot[["session_date"]].copy()
        test["value"] = values.to_numpy(float)
        bootstrap_rows.append(
            {
                "family": "incremental_return",
                "endpoint": name,
                "rows": len(test),
                **moving_block_bootstrap(test, "value", seed=SEED + seed_offset),
            }
        )
        seed_offset += 1
    for model_name, group in primary.groupby("model", sort=True):
        selected = group.loc[group["selected"]].copy()
        result = (
            moving_block_bootstrap(selected, "selected_net_bps", seed=SEED + 100 + seed_offset)
            if len(selected)
            else {
                "observed_mean": math.nan,
                "ci_lower": math.nan,
                "ci_upper": math.nan,
                "p_one_sided": math.nan,
                "sessions": 0,
            }
        )
        bootstrap_rows.append(
            {
                "family": "absolute_selected_net",
                "endpoint": model_name,
                "rows": len(selected),
                **result,
            }
        )
        seed_offset += 1
    direct_wide = direct.pivot(
        index=["event_id", "session_date"],
        columns="model",
        values=["raw_probability", "target_first"],
    )
    actual = direct_wide[("target_first", "context_only")].astype(int)
    loss_by_model: dict[str, np.ndarray] = {}
    for model_name in DIRECT_MODELS:
        probability = np.clip(
            direct_wide[("raw_probability", model_name)].to_numpy(float), 1e-9, 1 - 1e-9
        )
        loss_by_model[model_name] = -(
            actual.to_numpy(float) * np.log(probability)
            + (1.0 - actual.to_numpy(float)) * np.log(1.0 - probability)
        )
    predictive = direct_wide.reset_index()[["event_id", "session_date"]].copy()
    predictive_tests = {
        "context_loss_improvement_vs_prediction": loss_by_model["prediction_only"]
        - loss_by_model["context_only"],
        "loop_loss_improvement_vs_context": loss_by_model["context_only"]
        - loss_by_model["context_plus_loop_mixture"],
    }
    for name, values in predictive_tests.items():
        test = predictive[["session_date"]].copy()
        test["value"] = values
        bootstrap_rows.append(
            {
                "family": "predictive_log_loss",
                "endpoint": name,
                "rows": len(test),
                **moving_block_bootstrap(test, "value", seed=SEED + 200 + seed_offset),
            }
        )
        seed_offset += 1
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame = pd.concat(
        [holm(group) for _, group in bootstrap_frame.groupby("family", sort=True)],
        ignore_index=True,
    )
    return {
        "model_metrics": pd.DataFrame(model_rows),
        "policy_metrics": pd.DataFrame(policy_metric_rows),
        "incremental_metrics": pd.DataFrame(incremental_rows),
        "bootstrap_metrics": bootstrap_frame,
        "monthly_metrics": pd.DataFrame(monthly_rows),
        "stock_deletions": pd.DataFrame(deletion_rows),
        "cost_sensitivity": pd.DataFrame(cost_rows),
    }


def decision(metrics: dict[str, pd.DataFrame], contract: dict[str, Any]) -> dict[str, Any]:
    gates = contract["evaluation"]["qualification_gates"]
    policy = metrics["policy_metrics"]
    bootstrap = metrics["bootstrap_metrics"]
    monthly = metrics["monthly_metrics"]
    deletions = metrics["stock_deletions"]
    costs = metrics["cost_sensitivity"]
    increment_by_model = {
        "context_only": "context_only_minus_prediction_only",
        "context_plus_loop_mixture": "context_plus_loop_minus_context_only",
        "sequential_confirmation": "sequential_confirmation_minus_context_plus_loop",
    }
    results: dict[str, Any] = {}
    retained: list[str] = []
    for model_name in (*DIRECT_MODELS.keys(), SEQUENTIAL_MODEL):
        row = policy.loc[
            policy["policy_type"].eq("conservative_uncertainty_aware")
            & policy["model"].eq(model_name)
        ].iloc[0]
        absolute = bootstrap.loc[
            bootstrap["family"].eq("absolute_selected_net") & bootstrap["endpoint"].eq(model_name)
        ].iloc[0]
        month = monthly.loc[
            monthly["policy_type"].eq("conservative_uncertainty_aware")
            & monthly["model"].eq(model_name)
        ]
        deletion = deletions.loc[
            deletions["policy_type"].eq("conservative_uncertainty_aware")
            & deletions["model"].eq(model_name)
        ]
        cost = costs.loc[
            costs["policy_type"].eq("conservative_uncertainty_aware")
            & costs["model"].eq(model_name)
            & costs["cost_bps_per_side"].eq(7.5)
        ].iloc[0]
        incremental_ok = True
        incremental_endpoint = increment_by_model.get(model_name)
        if incremental_endpoint:
            incremental = bootstrap.loc[
                bootstrap["family"].eq("incremental_return")
                & bootstrap["endpoint"].eq(incremental_endpoint)
            ].iloc[0]
            incremental_ok = bool(incremental["passes_holm_0_05"])
        checks = {
            "selected_rows": int(row["selected_rows"]) >= int(gates["selected_rows_minimum"]),
            "coverage": float(row["coverage"]) >= float(gates["coverage_minimum"]),
            "selected_sessions": int(row["selected_sessions"])
            >= int(gates["selected_sessions_minimum"]),
            "selected_stocks": int(row["selected_stocks"]) >= int(gates["selected_stocks_minimum"]),
            "selected_mean_net_positive": bool(float(row["selected_mean_net_bps"]) > 0.0),
            "selected_net_block_and_holm": bool(absolute["passes_holm_0_05"]),
            "incremental_return_block_and_holm": incremental_ok,
            "positive_at_7_5_bps_per_side": bool(float(cost["selected_mean_net_bps"]) > 0.0),
            "positive_month_majority": bool(
                month["selected_mean_net_bps"].gt(0).sum() > len(month) / 2
            ),
            "positive_leave_one_stock_out": int(deletion["selected_mean_net_bps"].gt(0).sum())
            >= int(gates["positive_leave_one_stock_out_minimum"]),
        }
        qualified = all(checks.values())
        if qualified:
            retained.append(model_name)
        results[model_name] = {
            "decision": (
                "descriptive_equation_candidate_for_prospective_research_only"
                if qualified
                else "rejected_or_unknown"
            ),
            "checks": checks,
        }
    return {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scientific_status": contract["scientific_status"],
        "models": results,
        "retained_for_prospective_research_only": retained,
        "overall_decision": (
            "descriptive_equation_candidates_exist_no_validation_or_promotion"
            if retained
            else "all_equations_rejected_or_unknown"
        ),
        "validation_claim": False,
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "application_modified": False,
    }


def artifact_manifest(out: Path) -> dict[str, Any]:
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(out.iterdir())
            if path.is_file()
            and path.name not in {"artifact_manifest.json", "independent_audit.json"}
        ],
    }


def score_experiment(frozen: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    contract = load_contract()
    frozen_manifest = verify_frozen(frozen, contract)
    out.mkdir(parents=True)
    for name in (
        "pre_outcome_base_events.parquet",
        "pre_outcome_sequential_snapshots.parquet",
        "data_coverage.csv",
        "population_counts.csv",
        "pre_score_manifest.json",
    ):
        shutil.copy2(frozen / name, out / name)
    base_ledger = pd.read_parquet(frozen / "pre_outcome_base_events.parquet")
    snapshot_ledger = pd.read_parquet(frozen / "pre_outcome_sequential_snapshots.parquet")
    assert_pre_outcome_columns(base_ledger)
    assert_pre_outcome_columns(snapshot_ledger, allow_completed_path_excursions=True)
    tapes, raw_coverage = load_tapes(contract)
    base = attach_outcomes(
        base_ledger,
        tapes,
        id_column="event_id",
        entry_ordinal_field="entry_ordinal",
        entry_open_field="entry_open",
        stop_field="stop_price",
        target_field="target_price",
        risk_field="base_risk_bps",
    )
    eligible_snapshots = snapshot_ledger.loc[
        snapshot_ledger["eligible_for_sequential_admission"]
    ].copy()
    snapshots = attach_outcomes(
        eligible_snapshots,
        tapes,
        id_column="snapshot_id",
        entry_ordinal_field="sequential_entry_ordinal",
        entry_open_field="sequential_entry_open",
        stop_field="sequential_stop_price",
        target_field="sequential_target_price",
        risk_field="sequential_risk_bps",
    )
    base = base.loc[base["outcome_status"].eq("scored")].reset_index(drop=True)
    snapshots = snapshots.loc[snapshots["outcome_status"].eq("scored")].reset_index(drop=True)
    direct_raw = prequential_predictions(base, contract, DIRECT_MODELS, id_column="event_id")
    sequential_features = {
        SEQUENTIAL_MODEL: (
            (*CONTEXT_NUMERIC, *LOOP_NUMERIC, *SEQUENTIAL_NUMERIC),
            (*CONTEXT_CATEGORICAL, "top_loop_label"),
        )
    }
    sequential_raw = prequential_predictions(
        snapshots,
        contract,
        sequential_features,
        id_column="snapshot_id",
    )
    direct = calibrate_predictions(direct_raw, contract, sequential=False)
    sequential = calibrate_predictions(sequential_raw, contract, sequential=True)
    policies = build_policy_rows(base, direct, sequential, contract)
    metrics = evaluate(base, direct, sequential, policies)
    decision_value = decision(metrics, contract)
    base.to_parquet(out / "base_outcomes.parquet", index=False)
    snapshots.to_parquet(out / "snapshot_outcomes.parquet", index=False)
    direct.to_parquet(out / "direct_predictions.parquet", index=False)
    sequential.to_parquet(out / "sequential_predictions.parquet", index=False)
    policies.to_parquet(out / "policy_rows.parquet", index=False)
    for name, frame in metrics.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    raw_coverage.to_csv(out / "raw_data_coverage.csv", index=False)
    write_json(out / "decision.json", decision_value)
    write_json(
        out / "summary.json",
        {
            "contract_id": contract["contract_id"],
            "scientific_status": contract["scientific_status"],
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "base_rows": len(base),
            "snapshot_rows": len(snapshots),
            "direct_prediction_rows": len(direct),
            "sequential_prediction_rows": len(sequential),
            "decision": decision_value,
            "model_metrics": metrics["model_metrics"].to_dict("records"),
            "policy_metrics": metrics["policy_metrics"].to_dict("records"),
            "incremental_metrics": metrics["incremental_metrics"].to_dict("records"),
            "bootstrap_metrics": metrics["bootstrap_metrics"].to_dict("records"),
            "frozen_manifest_sha256": sha256(frozen / "pre_score_manifest.json"),
        },
    )
    write_json(out / "source_hashes.json", frozen_manifest)
    write_json(out / "artifact_manifest.json", artifact_manifest(out))
    print(
        json.dumps(
            {
                "out": str(out),
                "base_rows": len(base),
                "snapshot_rows": len(snapshots),
                "overall_decision": decision_value["overall_decision"],
                "retained": decision_value["retained_for_prospective_research_only"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-events", type=Path)
    group.add_argument("--score", type=Path)
    parser.add_argument("--frozen-input", type=Path)
    args = parser.parse_args()
    if args.prepare_events is not None:
        if args.frozen_input is not None:
            raise ValueError("--frozen-input cannot accompany --prepare-events")
        prepare_events(args.prepare_events)
        return
    if args.frozen_input is None:
        raise ValueError("--frozen-input is required with --score")
    score_experiment(args.frozen_input, args.score)


if __name__ == "__main__":
    main()
