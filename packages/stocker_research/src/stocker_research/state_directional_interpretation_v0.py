"""Role-aware directional interpretation for state-event-detector reports.

This module updates a completed ``state_event_detector_v0`` report with metrics
that score event states according to their intended use: long entry, short or
long-blocker, or no-trade blocker. It does not alter event detection.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT_BASE_DIR = Path("data/reports/research/state_event_detector_v0")

EVENT_STATE_ROLES: dict[str, str] = {
    "controlled_pullback_after_bullish_impulse": "long_entry",
    "liquidation_failed_low_reclaim": "long_entry_or_reversal",
    "slow_snapback_after_dip": "long_entry_or_reversal",
    "failed_bounce_active_liquidation": "long_blocker_or_short",
    "failed_bullish_impulse_recoil": "long_blocker_or_short",
    "failed_open_down_continuation": "long_blocker_or_short",
    "dead_chop_blocker": "no_trade_blocker",
}

ROLE_DEFAULT_DIRECTIONS: dict[str, int] = {
    "long_entry": 1,
    "long_entry_or_reversal": 1,
    "long_blocker_or_short": -1,
    "no_trade_blocker": 0,
}

REQUIRED_STATE_EVENT_FILES = (
    "event_rows.csv",
    "summary.md",
    "summary.json",
    "decision.json",
)


@dataclass(frozen=True)
class DirectionalInterpretationConfig:
    """Configuration for role-aware event-state interpretation."""

    horizons: tuple[int, ...] = (6, 9, 12, 24)
    train_fraction: float = 0.60
    min_events: int = 30
    min_symbols: int = 3
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    min_aligned_median_return_bps: float = 0.0
    min_aligned_win_rate: float = 0.50
    min_random_excess_bps: float = 0.0
    low_movement_threshold: float = 0.001
    random_seed: int = 1337
    random_iterations: int = 100


@dataclass(frozen=True)
class DirectionalInterpretationResult:
    """Result paths from updating a state-event-detector report."""

    input_dir: Path
    output_dir: Path
    summary_markdown_path: Path
    decision_json_path: Path
    directional_state_summary_csv_path: Path
    blocker_quality_summary_csv_path: Path
    short_candidate_summary_csv_path: Path
    no_trade_quality_summary_csv_path: Path
    oos_directional_state_response_csv_path: Path
    decision: str
    state_decision_count: int


def _return_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_return"


def _mfe_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mfe"


def _mae_col(horizon: int) -> str:
    return f"forward_{horizon}_bar_mae"


def _role_for_state(event_state: str) -> str:
    return EVENT_STATE_ROLES.get(str(event_state), "unknown")


def _default_direction_for_role(role: str) -> int:
    return ROLE_DEFAULT_DIRECTIONS.get(role, 0)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 16) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def find_latest_state_event_detector_run(
    base_dir: Path = DEFAULT_INPUT_BASE_DIR,
) -> Path:
    """Find the latest complete state-event-detector report directory."""

    if not base_dir.exists():
        raise FileNotFoundError(f"Missing state event detector base directory: {base_dir}")
    candidates: list[Path] = []
    for path in base_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("state_event_detector_v0_"):
            continue
        if all((path / name).exists() for name in REQUIRED_STATE_EVENT_FILES):
            candidates.append(path)
    if not candidates:
        expected = base_dir / "state_event_detector_v0_<run_id>"
        raise FileNotFoundError(
            f"No complete state_event_detector_v0 run found; expected {expected}"
        )
    return sorted(candidates, key=lambda item: (item.stat().st_mtime, item.name))[-1]


def _validate_input_dir(input_dir: Path) -> None:
    missing = [name for name in REQUIRED_STATE_EVENT_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required state-event report files: {missing}")


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _split_train_test(
    rows: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return rows, rows
    data = rows.copy()
    data["_timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.sort_values(["_timestamp", "symbol", "event_state"], kind="mergesort")
    if len(data) < 2:
        return data.drop(columns=["_timestamp"]), data.iloc[0:0].drop(columns=["_timestamp"])
    train_count = max(1, min(len(data) - 1, int(len(data) * train_fraction)))
    train = data.iloc[:train_count].drop(columns=["_timestamp"])
    test = data.iloc[train_count:].drop(columns=["_timestamp"])
    return train, test


def _single_symbol_share(rows: pd.DataFrame) -> float:
    if rows.empty or "symbol" not in rows:
        return math.nan
    return float(rows["symbol"].astype(str).value_counts(normalize=True).max())


def _single_session_share(rows: pd.DataFrame) -> float:
    if rows.empty or "session_date" not in rows:
        return math.nan
    return float(rows["session_date"].astype(str).value_counts(normalize=True).max())


def _symbol_count(rows: pd.DataFrame) -> int:
    return int(rows["symbol"].nunique()) if "symbol" in rows and not rows.empty else 0


def estimate_expected_direction(
    *,
    event_state: str,
    train_rows: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    """Estimate expected direction using train rows only plus explicit role defaults."""

    role = _role_for_state(event_state)
    default_direction = _default_direction_for_role(role)
    return_column = _return_col(horizon)
    returns = pd.to_numeric(train_rows.get(return_column, pd.Series(dtype=float)), errors="coerce")
    train_median = float(returns.median()) if returns.notna().any() else math.nan
    evidence_direction = 0
    if not math.isnan(train_median):
        if train_median > 0.0:
            evidence_direction = 1
        elif train_median < 0.0:
            evidence_direction = -1
    if role == "no_trade_blocker":
        expected_direction = 0
    else:
        expected_direction = default_direction if default_direction else evidence_direction
    conflict = bool(
        default_direction != 0
        and evidence_direction != 0
        and evidence_direction != default_direction
    )
    return {
        "event_state": event_state,
        "role": role,
        "horizon": int(horizon),
        "default_direction": int(default_direction),
        "expected_direction": int(expected_direction),
        "train_median_return": train_median,
        "train_evidence_direction": int(evidence_direction),
        "role_evidence_conflict": conflict,
    }


def add_aligned_return_columns(
    rows: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
) -> pd.DataFrame:
    """Add aligned return and consistency columns for one horizon."""

    data = rows.copy()
    returns = pd.to_numeric(data[_return_col(horizon)], errors="coerce")
    aligned_column = f"aligned_{horizon}_bar_return"
    consistency_column = f"directional_consistent_{horizon}"
    if expected_direction == 0:
        data[aligned_column] = -returns.abs()
        data[consistency_column] = (
            returns.abs() <= returns.abs().median()
        ).map(bool).astype(object)
        return data
    aligned = returns * expected_direction
    data[aligned_column] = aligned
    data[consistency_column] = (aligned > 0.0).map(bool).astype(object)
    return data


def _random_sample_metric(
    rows: pd.DataFrame,
    *,
    retained_count: int,
    horizon: int,
    expected_direction: int,
    seed: int,
    iterations: int,
) -> dict[str, float | int | str]:
    if rows.empty or retained_count <= 0:
        return {
            "retained_count": int(retained_count),
            "random_aligned_median_return": math.nan,
            "random_directional_consistency": math.nan,
        }
    rng = np.random.default_rng(seed)
    sample_size = min(retained_count, len(rows))
    indices = np.array(rows.index.tolist())
    medians: list[float] = []
    consistency: list[float] = []
    for _ in range(max(1, iterations)):
        sampled = rows.loc[rng.choice(indices, size=sample_size, replace=False)]
        aligned = add_aligned_return_columns(
            sampled,
            horizon=horizon,
            expected_direction=expected_direction,
        )
        aligned_returns = pd.to_numeric(aligned[f"aligned_{horizon}_bar_return"], errors="coerce")
        medians.append(float(aligned_returns.median()))
        consistency.append(float((aligned_returns > 0.0).mean()))
    return {
        "retained_count": int(retained_count),
        "random_aligned_median_return": float(np.nanmedian(medians)),
        "random_directional_consistency": float(np.nanmedian(consistency)),
    }


def _different_event_same_bucket_baseline(
    source_rows: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
) -> float:
    if source_rows.empty or pool.empty or "time_of_day_bucket" not in source_rows:
        return math.nan
    samples: list[pd.DataFrame] = []
    for bucket, bucket_rows in source_rows.groupby("time_of_day_bucket"):
        candidate_pool = pool[
            pool["time_of_day_bucket"].astype(str).eq(str(bucket))
            & ~pool["event_state"].astype(str).isin(bucket_rows["event_state"].astype(str).unique())
        ]
        if candidate_pool.empty:
            continue
        samples.append(
            candidate_pool.sample(
                n=min(len(bucket_rows), len(candidate_pool)),
                random_state=17,
            )
        )
    if not samples:
        return math.nan
    sampled = pd.concat(samples, ignore_index=True)
    aligned = add_aligned_return_columns(
        sampled,
        horizon=horizon,
        expected_direction=expected_direction,
    )
    return float(pd.to_numeric(aligned[f"aligned_{horizon}_bar_return"], errors="coerce").median())


def summarize_blocker_quality(
    rows: pd.DataFrame,
    *,
    event_state: str,
    horizon: int,
) -> dict[str, Any]:
    """Summarize long-blocker quality from raw forward long returns."""

    returns = pd.to_numeric(rows.get(_return_col(horizon), pd.Series(dtype=float)), errors="coerce")
    valid = returns.dropna()
    if valid.empty:
        return {
            "event_state": event_state,
            "horizon": int(horizon),
            "event_count": 0,
            "bad_long_capture_rate": math.nan,
            "good_long_false_block_rate": math.nan,
            "avoided_long_loss_bps": math.nan,
            "missed_long_profit_bps": math.nan,
            "blocker_net_value_bps": math.nan,
        }
    bad = valid <= 0.0
    good = valid > 0.0
    avoided = float((-valid[bad]).median() * 10_000) if bad.any() else 0.0
    missed = float(valid[good].median() * 10_000) if good.any() else 0.0
    return {
        "event_state": event_state,
        "horizon": int(horizon),
        "event_count": int(len(valid)),
        "bad_long_capture_rate": float(bad.mean()),
        "good_long_false_block_rate": float(good.mean()),
        "avoided_long_loss_bps": avoided,
        "missed_long_profit_bps": missed,
        "blocker_net_value_bps": avoided - missed,
    }


def summarize_short_candidates(
    rows: pd.DataFrame,
    *,
    event_state: str,
    horizon: int,
) -> dict[str, Any]:
    """Summarize short-entry interpretation of blocker/short states."""

    returns = pd.to_numeric(rows.get(_return_col(horizon), pd.Series(dtype=float)), errors="coerce")
    valid = rows[returns.notna()]
    returns = returns[returns.notna()]
    short_returns = -returns
    mfe = pd.to_numeric(
        valid.get(_mfe_col(horizon), pd.Series(np.nan, index=valid.index)),
        errors="coerce",
    )
    mae = pd.to_numeric(
        valid.get(_mae_col(horizon), pd.Series(np.nan, index=valid.index)),
        errors="coerce",
    )
    return {
        "event_state": event_state,
        "horizon": int(horizon),
        "event_count": int(len(valid)),
        "short_median_return": float(short_returns.median())
        if not short_returns.empty
        else math.nan,
        "short_win_rate": float((returns < 0.0).mean()) if not returns.empty else math.nan,
        "short_mfe": float((-mae).median()) if mae.notna().any() else math.nan,
        "short_mae": float((-mfe).median()) if mfe.notna().any() else math.nan,
        "short_directional_accuracy": float((returns < 0.0).mean())
        if not returns.empty
        else math.nan,
    }


def summarize_no_trade_quality(
    rows: pd.DataFrame,
    *,
    event_state: str,
    horizon: int,
    low_movement_threshold: float,
) -> dict[str, Any]:
    """Summarize no-trade blocker quality by low movement and big-move false blocks."""

    returns = pd.to_numeric(rows.get(_return_col(horizon), pd.Series(dtype=float)), errors="coerce")
    valid = rows[returns.notna()]
    returns = returns[returns.notna()]
    abs_returns = returns.abs()
    mfe = pd.to_numeric(
        valid.get(_mfe_col(horizon), pd.Series(np.nan, index=valid.index)),
        errors="coerce",
    )
    mae = pd.to_numeric(
        valid.get(_mae_col(horizon), pd.Series(np.nan, index=valid.index)),
        errors="coerce",
    )
    if abs_returns.empty:
        low_rate = math.nan
        big_rate = math.nan
    else:
        low_rate = float((abs_returns <= low_movement_threshold).mean())
        big_rate = float((abs_returns > low_movement_threshold).mean())
    return {
        "event_state": event_state,
        "horizon": int(horizon),
        "event_count": int(len(valid)),
        "median_abs_forward_return": float(abs_returns.median())
        if not abs_returns.empty
        else math.nan,
        "median_mfe": float(mfe.median()) if mfe.notna().any() else math.nan,
        "median_mae": float(mae.median()) if mae.notna().any() else math.nan,
        "low_movement_rate": low_rate,
        "false_block_big_move_rate": big_rate,
        "no_trade_quality_score": low_rate - big_rate
        if not math.isnan(low_rate) and not math.isnan(big_rate)
        else math.nan,
    }


def run_random_blocker_baseline(
    rows: pd.DataFrame,
    *,
    retained_count: int,
    horizon: int,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    """Random same-count blocker baseline."""

    if rows.empty or retained_count <= 0:
        return {
            "baseline": "random_blocker_same_count",
            "retained_count": int(retained_count),
            "bad_long_capture_rate": math.nan,
            "good_long_false_block_rate": math.nan,
            "blocker_net_value_bps": math.nan,
        }
    rng = np.random.default_rng(seed)
    count = min(retained_count, len(rows))
    indices = np.array(rows.index.tolist())
    bad_rates: list[float] = []
    false_rates: list[float] = []
    net_values: list[float] = []
    for _ in range(max(1, iterations)):
        sample = rows.loc[rng.choice(indices, size=count, replace=False)]
        quality = summarize_blocker_quality(sample, event_state="random", horizon=horizon)
        bad_rates.append(float(quality["bad_long_capture_rate"]))
        false_rates.append(float(quality["good_long_false_block_rate"]))
        net_values.append(float(quality["blocker_net_value_bps"]))
    return {
        "baseline": "random_blocker_same_count",
        "retained_count": int(count),
        "bad_long_capture_rate": float(np.nanmedian(bad_rates)),
        "good_long_false_block_rate": float(np.nanmedian(false_rates)),
        "blocker_net_value_bps": float(np.nanmedian(net_values)),
    }

def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def _decision_evaluation(
    row: pd.Series,
    oos_lookup: dict[tuple[str, int], pd.Series],
) -> dict[str, Any]:
    event_state = str(row["event_state"])
    horizon = int(row["horizon"])
    aligned = _as_float(row.get("aligned_median_return", math.nan))
    win_rate = _as_float(row.get("aligned_win_rate", math.nan))
    random_median = _as_float(row.get("random_aligned_median_return", math.nan))
    generic_median = _as_float(row.get("generic_aligned_median_return", math.nan))
    basis = "full_sample"
    oos_row = oos_lookup.get((event_state, horizon))
    if oos_row is not None:
        oos_aligned = _as_float(oos_row.get("test_aligned_median_return", math.nan))
        if not math.isnan(oos_aligned):
            aligned = oos_aligned
            win_rate = _as_float(oos_row.get("test_aligned_win_rate", math.nan))
            random_median = _as_float(
                oos_row.get("random_aligned_median_return", random_median)
            )
            generic_median = _as_float(
                oos_row.get("generic_aligned_median_return", generic_median)
            )
            basis = "oos_60_40"
    random_excess = (
        (aligned - random_median) * 10_000
        if not math.isnan(aligned) and not math.isnan(random_median)
        else math.nan
    )
    generic_excess = (
        (aligned - generic_median) * 10_000
        if not math.isnan(aligned) and not math.isnan(generic_median)
        else math.nan
    )
    return {
        "state_decision_basis": basis,
        "decision_aligned_median_return": aligned,
        "decision_aligned_win_rate": win_rate,
        "decision_random_aligned_median_return": random_median,
        "decision_generic_aligned_median_return": generic_median,
        "decision_aligned_excess_vs_random_bps": random_excess,
        "decision_aligned_excess_vs_generic_bps": generic_excess,
    }


def _directional_summary_for_group(
    *,
    event_state: str,
    horizon: int,
    rows: pd.DataFrame,
    pool: pd.DataFrame,
    config: DirectionalInterpretationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train, test = _split_train_test(rows, config.train_fraction)
    direction = estimate_expected_direction(
        event_state=event_state,
        train_rows=train,
        horizon=horizon,
    )
    expected_direction = int(direction["expected_direction"])
    aligned_all = add_aligned_return_columns(
        rows,
        horizon=horizon,
        expected_direction=expected_direction,
    )
    aligned_test = add_aligned_return_columns(
        test,
        horizon=horizon,
        expected_direction=expected_direction,
    )
    returns = pd.to_numeric(rows[_return_col(horizon)], errors="coerce")
    aligned_returns = pd.to_numeric(
        aligned_all[f"aligned_{horizon}_bar_return"],
        errors="coerce",
    )
    test_aligned = pd.to_numeric(
        aligned_test[f"aligned_{horizon}_bar_return"],
        errors="coerce",
    )
    random = _random_sample_metric(
        pool[pd.to_numeric(pool[_return_col(horizon)], errors="coerce").notna()],
        retained_count=len(rows),
        horizon=horizon,
        expected_direction=expected_direction,
        seed=config.random_seed + horizon + len(event_state),
        iterations=config.random_iterations,
    )
    generic_aligned = add_aligned_return_columns(
        pool,
        horizon=horizon,
        expected_direction=expected_direction,
    )
    generic_returns = pd.to_numeric(
        generic_aligned[f"aligned_{horizon}_bar_return"],
        errors="coerce",
    )
    different_event_median = _different_event_same_bucket_baseline(
        rows,
        pool,
        horizon=horizon,
        expected_direction=expected_direction,
    )
    summary = {
        **direction,
        "event_count": int(len(rows)),
        "symbol_count": _symbol_count(rows),
        "session_count": int(rows["session_date"].nunique()) if "session_date" in rows else 0,
        "raw_median_return": float(returns.median()) if returns.notna().any() else math.nan,
        "aligned_median_return": float(aligned_returns.median())
        if aligned_returns.notna().any()
        else math.nan,
        "aligned_win_rate": float((aligned_returns > 0.0).mean())
        if aligned_returns.notna().any()
        else math.nan,
        "directional_consistency": float((aligned_returns > 0.0).mean())
        if aligned_returns.notna().any()
        else math.nan,
        "wrong_way_rate": float((aligned_returns < 0.0).mean())
        if aligned_returns.notna().any()
        else math.nan,
        "single_symbol_share": _single_symbol_share(rows),
        "single_session_share": _single_session_share(rows),
        "random_aligned_median_return": random["random_aligned_median_return"],
        "random_directional_consistency": random["random_directional_consistency"],
        "generic_aligned_median_return": float(generic_returns.median())
        if generic_returns.notna().any()
        else math.nan,
        "different_event_same_time_bucket_aligned_median_return": different_event_median,
    }
    oos = {
        **direction,
        "split_mode": "walk_forward",
        "fold": "time_split_60_40",
        "train_event_count": int(len(train)),
        "test_event_count": int(len(test)),
        "test_raw_median_return": float(
            pd.to_numeric(test[_return_col(horizon)], errors="coerce").median()
        )
        if not test.empty
        else math.nan,
        "test_aligned_median_return": float(test_aligned.median())
        if test_aligned.notna().any()
        else math.nan,
        "test_aligned_win_rate": float((test_aligned > 0.0).mean())
        if test_aligned.notna().any()
        else math.nan,
        "random_aligned_median_return": random["random_aligned_median_return"],
        "generic_aligned_median_return": summary["generic_aligned_median_return"],
        "aligned_excess_vs_random_bps": (
            (float(test_aligned.median()) - float(random["random_aligned_median_return"])) * 10_000
            if test_aligned.notna().any()
            and not math.isnan(float(random["random_aligned_median_return"]))
            else math.nan
        ),
        "aligned_excess_vs_generic_bps": (
            (float(test_aligned.median()) - float(summary["generic_aligned_median_return"]))
            * 10_000
            if test_aligned.notna().any()
            and not math.isnan(float(summary["generic_aligned_median_return"]))
            else math.nan
        ),
    }
    return summary, oos


def build_directional_summaries(
    event_rows: pd.DataFrame,
    *,
    config: DirectionalInterpretationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all directional interpretation tables."""

    directional_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []
    short_rows: list[dict[str, Any]] = []
    no_trade_rows: list[dict[str, Any]] = []
    data = event_rows[event_rows["event_state"].astype(str).isin(EVENT_STATE_ROLES)].copy()
    for horizon in config.horizons:
        return_column = _return_col(horizon)
        if return_column not in data:
            continue
        horizon_pool = data[pd.to_numeric(data[return_column], errors="coerce").notna()]
        if horizon_pool.empty:
            continue
        for event_state, rows in horizon_pool.groupby("event_state"):
            summary, oos = _directional_summary_for_group(
                event_state=str(event_state),
                horizon=horizon,
                rows=rows.copy(),
                pool=horizon_pool,
                config=config,
            )
            directional_rows.append(summary)
            oos_rows.append(oos)
            role = _role_for_state(str(event_state))
            if role == "long_blocker_or_short":
                blocker_rows.append(
                    summarize_blocker_quality(rows, event_state=str(event_state), horizon=horizon)
                )
                short_rows.append(
                    summarize_short_candidates(rows, event_state=str(event_state), horizon=horizon)
                )
            elif role == "no_trade_blocker":
                no_trade_rows.append(
                    summarize_no_trade_quality(
                        rows,
                        event_state=str(event_state),
                        horizon=horizon,
                        low_movement_threshold=config.low_movement_threshold,
                    )
                )
    return (
        pd.DataFrame(directional_rows),
        pd.DataFrame(blocker_rows),
        pd.DataFrame(short_rows),
        pd.DataFrame(no_trade_rows),
        pd.DataFrame(oos_rows),
    )


def _state_decision(
    row: pd.Series,
    config: DirectionalInterpretationConfig,
    no_trade_quality: dict[tuple[str, int], float],
) -> str:
    if int(row["event_count"]) < config.min_events:
        return "reject_low_sample"
    if int(row["symbol_count"]) < config.min_symbols:
        return "reject_low_sample"
    if float(row["single_symbol_share"]) > config.max_single_symbol_share:
        return "reject_concentrated"
    if float(row["single_session_share"]) > config.max_single_session_share:
        return "reject_concentrated"
    role = str(row["role"])
    if role == "no_trade_blocker":
        quality = no_trade_quality.get((str(row["event_state"]), int(row["horizon"])), math.nan)
        if math.isnan(quality) or quality <= 0.0:
            return "reject_no_directional_consistency"
        return "continue_research_no_trade_filter"
    aligned_bps = float(row["decision_aligned_median_return"]) * 10_000
    random_excess_bps = float(row["decision_aligned_excess_vs_random_bps"])
    generic_excess_bps = float(row["decision_aligned_excess_vs_generic_bps"])
    aligned_win_rate = float(row["decision_aligned_win_rate"])
    if random_excess_bps < config.min_random_excess_bps:
        return "reject_random_baseline_better"
    if generic_excess_bps < config.min_random_excess_bps:
        return "reject_random_baseline_better"
    if aligned_bps < config.min_aligned_median_return_bps:
        return "reject_no_directional_consistency"
    if aligned_win_rate < config.min_aligned_win_rate:
        return "reject_no_directional_consistency"
    if role == "long_entry":
        return "continue_research_long_candidate"
    if role == "long_entry_or_reversal":
        return "continue_research_long_candidate"
    if role == "long_blocker_or_short":
        return "continue_research_short_candidate"
    if role == "no_trade_blocker":
        return "continue_research_no_trade_filter"
    return "reject_no_directional_consistency"


def build_directional_decision(
    *,
    directional_summary: pd.DataFrame,
    blocker_quality: pd.DataFrame,
    short_summary: pd.DataFrame,
    no_trade_summary: pd.DataFrame,
    oos_response: pd.DataFrame | None = None,
    config: DirectionalInterpretationConfig,
) -> dict[str, Any]:
    """Build role-aware decisions without treating negative raw return as failure."""

    if directional_summary.empty:
        return {
            "decision": "reject_low_sample",
            "decision_reasons": ["no directional state rows were available"],
            "state_decisions": [],
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
        }
    rows = directional_summary.copy()
    no_trade_quality_lookup = {
        (str(row["event_state"]), int(row["horizon"])): float(row["no_trade_quality_score"])
        for _, row in no_trade_summary.iterrows()
    } if not no_trade_summary.empty else {}
    oos_lookup = {
        (str(row["event_state"]), int(row["horizon"])): row
        for _, row in (oos_response if oos_response is not None else pd.DataFrame()).iterrows()
    }
    evaluation_rows = rows.apply(
        lambda row: pd.Series(_decision_evaluation(row, oos_lookup)),
        axis=1,
    )
    rows = pd.concat(
        [rows.reset_index(drop=True), evaluation_rows.reset_index(drop=True)],
        axis=1,
    )
    rows["state_decision"] = rows.apply(
        lambda row: _state_decision(row, config, no_trade_quality_lookup),
        axis=1,
    )
    state_decisions = rows[
        [
            "event_state",
            "horizon",
            "role",
            "expected_direction",
            "raw_median_return",
            "aligned_median_return",
            "aligned_win_rate",
            "random_aligned_median_return",
            "state_decision_basis",
            "decision_aligned_median_return",
            "decision_aligned_win_rate",
            "decision_aligned_excess_vs_random_bps",
            "decision_aligned_excess_vs_generic_bps",
            "role_evidence_conflict",
            "state_decision",
        ]
    ].to_dict("records")
    continue_rows = rows[rows["state_decision"].astype(str).str.startswith("continue_research")]
    if continue_rows.empty:
        if rows["state_decision"].eq("reject_concentrated").any():
            decision = "reject_concentrated"
        elif rows["state_decision"].eq("reject_low_sample").all():
            decision = "reject_low_sample"
        elif rows["state_decision"].eq("reject_random_baseline_better").any():
            decision = "reject_random_baseline_better"
        else:
            decision = "reject_no_directional_consistency"
        reasons = ["no state/horizon passed role-aware directional gates"]
    else:
        priority = [
            "continue_research_short_candidate",
            "continue_research_blocker",
            "continue_research_long_candidate",
            "continue_research_no_trade_filter",
        ]
        decisions = set(continue_rows["state_decision"].astype(str))
        decision = next(item for item in priority if item in decisions)
        reasons = ["at least one state/horizon passed role-aware directional gates"]
    return {
        "decision": decision,
        "decision_reasons": reasons,
        "state_decisions": state_decisions,
        "continue_state_count": int(len(continue_rows)),
        "role_mapping": EVENT_STATE_ROLES,
        "blocker_rows": blocker_quality.to_dict("records") if not blocker_quality.empty else [],
        "short_candidate_rows": short_summary.to_dict("records") if not short_summary.empty else [],
        "no_trade_rows": no_trade_summary.to_dict("records") if not no_trade_summary.empty else [],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
    }


def _summary_markdown(
    *,
    input_dir: Path,
    decision: dict[str, Any],
    directional_summary: pd.DataFrame,
    blocker_quality: pd.DataFrame,
    short_summary: pd.DataFrame,
    no_trade_summary: pd.DataFrame,
    oos_response: pd.DataFrame,
) -> str:
    long_states = [
        state for state, role in EVENT_STATE_ROLES.items() if role.startswith("long_entry")
    ]
    blocker_states = [
        state for state, role in EVENT_STATE_ROLES.items() if role == "long_blocker_or_short"
    ]
    no_trade_states = [
        state for state, role in EVENT_STATE_ROLES.items() if role == "no_trade_blocker"
    ]
    negative_useful = directional_summary[
        directional_summary["expected_direction"].eq(-1)
        & (directional_summary["raw_median_return"] < 0.0)
        & (directional_summary["aligned_median_return"] > 0.0)
    ] if not directional_summary.empty else pd.DataFrame()
    rejected = directional_summary.copy()
    if not rejected.empty and decision.get("state_decisions"):
        decision_frame = pd.DataFrame(decision["state_decisions"])
        rejected = decision_frame[
            ~decision_frame["state_decision"].astype(str).str.startswith("continue_research")
        ]
    summary_columns = [
        "event_state",
        "horizon",
        "role",
        "expected_direction",
        "raw_median_return",
        "aligned_median_return",
        "aligned_win_rate",
        "role_evidence_conflict",
    ]
    blocker_columns = [
        "event_state",
        "horizon",
        "bad_long_capture_rate",
        "good_long_false_block_rate",
        "avoided_long_loss_bps",
        "missed_long_profit_bps",
        "blocker_net_value_bps",
    ]
    short_columns = [
        "event_state",
        "horizon",
        "short_median_return",
        "short_win_rate",
        "short_directional_accuracy",
    ]
    no_trade_columns = [
        "event_state",
        "horizon",
        "median_abs_forward_return",
        "low_movement_rate",
        "false_block_big_move_rate",
        "no_trade_quality_score",
    ]
    oos_columns = [
        "event_state",
        "horizon",
        "expected_direction",
        "test_raw_median_return",
        "test_aligned_median_return",
        "aligned_excess_vs_random_bps",
        "aligned_excess_vs_generic_bps",
    ]
    return f"""# State Event Detector V0

Research-only report. No edge is claimed. Order placement is disabled.

## Directional Interpretation

input_report: {input_dir}
decision: {decision["decision"]}

Long-entry candidates: {", ".join(long_states)}

Blocker/short candidates: {", ".join(blocker_states)}

No-trade blockers: {", ".join(no_trade_states)}

States with consistent negative response that should not be treated as failures:

{_markdown_table(negative_useful, summary_columns)}

States inconsistent or rejected by directional gates:

{_markdown_table(rejected, ["event_state", "horizon", "role", "state_decision"])}

## Directional State Summary

{_markdown_table(directional_summary, summary_columns)}

## Blocker Quality

{_markdown_table(blocker_quality, blocker_columns)}

## Short Candidate Summary

{_markdown_table(short_summary, short_columns)}

## No-Trade Quality

{_markdown_table(no_trade_summary, no_trade_columns)}

## OOS Directional Response

{_markdown_table(oos_response, oos_columns)}
"""


def run_state_directional_interpretation_report(
    *,
    input_dir: Path | None = None,
    input_base_dir: Path = DEFAULT_INPUT_BASE_DIR,
    config: DirectionalInterpretationConfig | None = None,
) -> DirectionalInterpretationResult:
    """Update a state-event-detector report with role-aware directional metrics."""

    cfg = config or DirectionalInterpretationConfig()
    resolved_input = input_dir or find_latest_state_event_detector_run(input_base_dir)
    _validate_input_dir(resolved_input)
    event_rows = pd.read_csv(resolved_input / "event_rows.csv")
    (
        directional_summary,
        blocker_quality,
        short_summary,
        no_trade_summary,
        oos_response,
    ) = build_directional_summaries(event_rows, config=cfg)
    previous_decision = _load_json_if_present(resolved_input / "decision.json")
    decision = build_directional_decision(
        directional_summary=directional_summary,
        blocker_quality=blocker_quality,
        short_summary=short_summary,
        no_trade_summary=no_trade_summary,
        oos_response=oos_response,
        config=cfg,
    )
    decision["previous_state_event_detector_decision"] = previous_decision
    decision["directional_interpretation_updated_at"] = datetime.now(tz=UTC).isoformat()

    directional_path = resolved_input / "directional_state_summary.csv"
    blocker_path = resolved_input / "blocker_quality_summary.csv"
    short_path = resolved_input / "short_candidate_summary.csv"
    no_trade_path = resolved_input / "no_trade_quality_summary.csv"
    oos_path = resolved_input / "oos_directional_state_response.csv"
    decision_path = resolved_input / "decision.json"
    summary_path = resolved_input / "summary.md"
    _write_csv(directional_path, directional_summary)
    _write_csv(blocker_path, blocker_quality)
    _write_csv(short_path, short_summary)
    _write_csv(no_trade_path, no_trade_summary)
    _write_csv(oos_path, oos_response)
    _write_json(decision_path, decision)
    summary_path.write_text(
        _summary_markdown(
            input_dir=resolved_input,
            decision=decision,
            directional_summary=directional_summary,
            blocker_quality=blocker_quality,
            short_summary=short_summary,
            no_trade_summary=no_trade_summary,
            oos_response=oos_response,
        ),
        encoding="utf-8",
    )
    summary_json_path = resolved_input / "summary.json"
    summary_payload = _load_json_if_present(summary_json_path)
    summary_payload["directional_interpretation"] = {
        "config": asdict(cfg),
        "decision": decision,
        "files": {
            "directional_state_summary": str(directional_path),
            "blocker_quality_summary": str(blocker_path),
            "short_candidate_summary": str(short_path),
            "no_trade_quality_summary": str(no_trade_path),
            "oos_directional_state_response": str(oos_path),
        },
    }
    _write_json(summary_json_path, summary_payload)
    return DirectionalInterpretationResult(
        input_dir=resolved_input,
        output_dir=resolved_input,
        summary_markdown_path=summary_path,
        decision_json_path=decision_path,
        directional_state_summary_csv_path=directional_path,
        blocker_quality_summary_csv_path=blocker_path,
        short_candidate_summary_csv_path=short_path,
        no_trade_quality_summary_csv_path=no_trade_path,
        oos_directional_state_response_csv_path=oos_path,
        decision=str(decision["decision"]),
        state_decision_count=int(len(decision["state_decisions"])),
    )


__all__ = [
    "DirectionalInterpretationConfig",
    "DirectionalInterpretationResult",
    "EVENT_STATE_ROLES",
    "add_aligned_return_columns",
    "build_directional_decision",
    "build_directional_summaries",
    "estimate_expected_direction",
    "find_latest_state_event_detector_run",
    "run_random_blocker_baseline",
    "run_state_directional_interpretation_report",
    "summarize_blocker_quality",
    "summarize_no_trade_quality",
    "summarize_short_candidates",
]
