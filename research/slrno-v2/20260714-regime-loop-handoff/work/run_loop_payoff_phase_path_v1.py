#!/usr/bin/env python3
"""Research-only loop payoff phase and path diagnostic.

Consumes frozen forecasts, executions, state runs, and provider OHLC. It cannot
connect to a broker, place orders, mutate positions, deploy, or edit app code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260713-loop-payoff-phase-path-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260713-loop-payoff-phase-path-v1-pre-score.json"
LOOP_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))
PERIODS = (2023, 2025)
SEED = 20260713
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
ROUND_TRIP_COST_BPS = 10.0


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "auditor": HERE / "audit_loop_payoff_phase_path_v1.py",
        "anchor_2023": Path(contract["inputs"]["anchor_panels"]["2023"]),
        "anchor_2025": Path(contract["inputs"]["anchor_panels"]["2025"]),
        "ledger": Path(contract["inputs"]["accepted_signal_ledger"]),
        "fixed_cycles": Path(contract["inputs"]["fixed_cycles"]),
        "runs_2023": Path(contract["inputs"]["runs"]["2023"]),
        "runs_2024": Path(contract["inputs"]["runs"]["2024_hazard_fit"]),
        "runs_2025": Path(contract["inputs"]["runs"]["2025"]),
        "parent_report": Path(contract["inputs"]["parent_report"]),
        "parent_handoff": Path(contract["inputs"]["parent_handoff"]),
    }
    for period in PERIODS:
        root = Path(contract["inputs"]["provider_roots"][str(period)])
        for symbol in contract["population"]["symbols"]:
            paths[f"provider_{period}_{symbol}"] = provider_path(root, symbol)
    return paths


def load_and_verify_contract() -> tuple[dict[str, Any], dict[str, str]]:
    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    safety = (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["broker_connection_enabled"] is False
        and contract["paper_or_demo_execution_enabled"] is False
        and contract["deployment_enabled"] is False
        and contract["strategy_promotion_allowed"] is False
        and contract["application_code_modification_allowed"] is False
        and contract["sealed_data_status"]["genuinely_unseen_sessions_available"] is False
    )
    if not safety:
        raise AssertionError("research-only safety boundary drift")
    paths = source_paths(contract)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    actual = {name: sha256(path) for name, path in paths.items()}
    expected = pre_score["sha256"]
    if actual != expected:
        changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise AssertionError(f"pre-score source hash mismatch: {changed}")
    return contract, actual


def load_cycles(path: Path) -> pd.DataFrame:
    cycles = pd.read_csv(path)
    if len(cycles) != 20 or set(cycles.columns) != {"cycle_id", "cycle", "transition_length"}:
        raise AssertionError("fixed cycle dictionary drift")
    return cycles


def load_anchor_panel(path: Path, period: int, cycles: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "anchor_id", "symbol_norm", "session_date", "state", "start_pos",
        "start_timestamp", "previous_state_1", *LOOP_COLUMNS,
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["period"] = period
    if frame["anchor_id"].duplicated().any():
        raise AssertionError(f"duplicate anchor ids in {period}")
    values = frame.loc[:, LOOP_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise AssertionError("invalid loop score")
    top = np.argmax(values, axis=1)
    frame["top_loop"] = cycles["cycle_id"].to_numpy(str)[top]
    frame["top_loop_probability"] = values[np.arange(len(frame)), top]
    frame = frame.rename(columns={"state": "anchor_state", "start_pos": "anchor_start_pos"})
    return frame[[
        "period", "anchor_id", "symbol_norm", "session_date", "anchor_state",
        "anchor_start_pos", "start_timestamp", "previous_state_1", "top_loop",
        "top_loop_probability",
    ]]


def load_runs(path: Path, period: int) -> pd.DataFrame:
    columns = [
        "symbol_norm", "session_date", "state", "duration", "start_pos",
        "end_pos", "start_timestamp", "end_timestamp", "next_state", "has_next_state",
    ]
    frame = pd.read_parquet(path, columns=columns) if path.suffix == ".parquet" else pd.read_csv(path, usecols=columns)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["state"] = pd.to_numeric(frame["state"], errors="raise").astype(int)
    frame["duration"] = pd.to_numeric(frame["duration"], errors="raise").astype(int)
    frame["start_pos"] = pd.to_numeric(frame["start_pos"], errors="raise").astype(int)
    frame["end_pos"] = pd.to_numeric(frame["end_pos"], errors="raise").astype(int)
    frame["period"] = period
    return frame.sort_values(["symbol_norm", "session_date", "start_pos"], kind="stable").reset_index(drop=True)


def hazard_training(runs_2024: pd.DataFrame) -> dict[int, np.ndarray]:
    completed = runs_2024.loc[runs_2024["has_next_state"].eq(True)].copy()
    result: dict[int, np.ndarray] = {}
    for state, group in completed.groupby("state", sort=True):
        durations = np.sort(group["duration"].to_numpy(int))
        if len(durations) < 1000:
            raise AssertionError(f"insufficient 2024 duration support for state {state}")
        result[int(state)] = durations
    if set(result) != set(range(8)):
        raise AssertionError("missing state in 2024 hazard fit")
    return result


def duration_features(durations: np.ndarray, age: int, required_age: int) -> tuple[float, float, int]:
    at_risk = durations[durations >= age]
    if not len(at_risk):
        return math.nan, math.nan, 0
    percentile = float(np.mean(durations <= age))
    hazard = float(np.mean(at_risk < required_age))
    return percentile, hazard, int(len(at_risk))


def load_tape(root: Path, symbols: list[str], period: int) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for symbol in symbols:
        path = provider_path(root, symbol)
        frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["session_date"] = local.dt.strftime("%Y-%m-%d")
        previous_close = frame["close"].shift(1)
        tr = np.maximum.reduce([
            (frame["high"] - frame["low"]).to_numpy(float),
            (frame["high"] - previous_close).abs().to_numpy(float),
            (frame["low"] - previous_close).abs().to_numpy(float),
        ])
        frame["true_range"] = tr
        frame["atr14_prior"] = pd.Series(tr, index=frame.index).shift(1).rolling(14, min_periods=14).mean()
        target = frame.loc[pd.to_datetime(frame["session_date"]).dt.year.eq(period)].copy()
        target["bar_ordinal"] = target.groupby("session_date", sort=False).cumcount()
        if target.empty or target[["open", "high", "low", "close"]].le(0).any().any():
            raise AssertionError(f"invalid provider tape {period} {symbol}")
        for session_date, group in target.groupby("session_date", sort=False):
            clean = group.sort_values("bar_ordinal", kind="stable").reset_index(drop=True)
            if not np.array_equal(clean["bar_ordinal"].to_numpy(int), np.arange(len(clean))):
                raise AssertionError("non-contiguous session ordinal")
            groups[(symbol, str(session_date))] = clean
        audits.append({
            "period": period,
            "symbol": symbol,
            "rows": int(len(target)),
            "sessions": int(target["session_date"].nunique()),
            "first_timestamp": target["timestamp"].min(),
            "last_timestamp": target["timestamp"].max(),
            "atr14_available": int(target["atr14_prior"].notna().sum()),
        })
    return groups, audits


def run_lookup(runs: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (str(symbol), str(session)): group.reset_index(drop=True)
        for (symbol, session), group in runs.groupby(["symbol_norm", "session_date"], sort=False)
    }


def covering_run(group: pd.DataFrame, ordinal: int) -> pd.Series:
    starts = group["start_pos"].to_numpy(int)
    index = int(np.searchsorted(starts, ordinal, side="right") - 1)
    if index < 0:
        raise AssertionError("no state run before admission")
    row = group.iloc[index]
    if not (int(row.start_pos) <= ordinal <= int(row.end_pos)):
        raise AssertionError("admission not covered by state run")
    return row


def to_state_position(anchor_start_pos: int, anchor_bar_ordinal: int, session_ordinal: int) -> int:
    """Translate a session-local bar ordinal into the global state-run coordinate."""
    return int(anchor_start_pos) - int(anchor_bar_ordinal) + int(session_ordinal)


def execution_ordinals(anchor_tape_ordinal: int, entry_step: int, horizon: int) -> tuple[int, int]:
    return int(anchor_tape_ordinal) + int(entry_step), int(anchor_tape_ordinal) + int(horizon)


def age_bin(age: int) -> str:
    if age == 1:
        return "1"
    if age <= 3:
        return "2-3"
    if age <= 6:
        return "4-6"
    if age <= 12:
        return "7-12"
    return "13+"


def percentile_bin(value: float) -> str:
    if value < 0.50:
        return "<0.50"
    if value < 0.80:
        return "0.50-0.80"
    return ">=0.80"


def hazard_bin(value: float) -> str:
    if value < 0.50:
        return "<0.50"
    if value < 0.80:
        return "0.50-0.80"
    return ">=0.80"


def enrich_signals(
    signals: pd.DataFrame,
    period: int,
    tape: dict[tuple[str, str], pd.DataFrame],
    runs: dict[tuple[str, str], pd.DataFrame],
    hazard_fit: dict[int, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in signals.itertuples(index=False):
        key = (str(row.symbol_norm), str(row.session_date))
        bars = tape[key]
        state_runs = runs[key]
        anchor_timestamp = pd.Timestamp(row.start_timestamp)
        anchor_matches = np.flatnonzero(bars["timestamp"].eq(anchor_timestamp).to_numpy(bool))
        if len(anchor_matches) != 1:
            raise AssertionError("provider anchor timestamp match failure")
        tape_anchor_ordinal = int(anchor_matches[0])
        entry_ordinal, exit_ordinal = execution_ordinals(tape_anchor_ordinal, row.entry_step, 24)
        if not (0 <= tape_anchor_ordinal < entry_ordinal <= exit_ordinal < len(bars)):
            raise AssertionError("invalid frozen path ordinals")
        entry_state_position = int(row.anchor_start_pos) + int(row.entry_step)
        exit_state_position = int(row.anchor_start_pos) + 24
        admission = covering_run(state_runs, entry_state_position)
        admission_state = int(admission.state)
        admission_age = entry_state_position - int(admission.start_pos) + 1
        required_age = exit_state_position - int(admission.start_pos) + 1
        duration_percentile, exit_hazard, hazard_support = duration_features(
            hazard_fit[admission_state], admission_age, required_age
        )
        path = bars.iloc[entry_ordinal : exit_ordinal + 1]
        entry = float(row.entry_price)
        direction = int(row.direction)
        if direction == 1:
            favorable = 10000.0 * (path["high"].to_numpy(float) / entry - 1.0)
            adverse = 10000.0 * (path["low"].to_numpy(float) / entry - 1.0)
        elif direction == -1:
            favorable = -10000.0 * (path["low"].to_numpy(float) / entry - 1.0)
            adverse = -10000.0 * (path["high"].to_numpy(float) / entry - 1.0)
        else:
            raise AssertionError("filled signal without direction")
        mfe = float(np.max(favorable))
        mae = float(np.min(adverse))
        time_mfe = int(np.argmax(favorable))
        time_mae = int(np.argmin(adverse))
        if len(path) > 1:
            post_mfe = float(np.max(favorable[1:]))
            post_mae = float(np.min(adverse[1:]))
            post_time_mfe = int(np.argmax(favorable[1:]) + 1)
            post_time_mae = int(np.argmin(adverse[1:]) + 1)
        else:
            post_mfe = math.nan
            post_mae = math.nan
            post_time_mfe = -1
            post_time_mae = -1
        exit_close = float(path.iloc[-1]["close"])
        replay_gross = 10000.0 * direction * (exit_close / entry - 1.0)
        if not np.isclose(replay_gross, float(row.gross_return_bps), atol=1e-8, rtol=1e-8):
            raise AssertionError("frozen gross payoff replay mismatch")
        net_bps = float(row.gross_return_bps) - ROUND_TRIP_COST_BPS
        final_positive = net_bps > 0.0
        if final_positive:
            path_class = "final_positive"
        elif mfe > ROUND_TRIP_COST_BPS:
            path_class = "timing_failure"
        else:
            path_class = "no_usable_move"
        atr = float(bars.iloc[entry_ordinal]["atr14_prior"])
        atr_valid = math.isfinite(atr) and atr > 0.0
        orientation_survived = (
            admission_state == int(row.anchor_state)
            and int(admission.start_pos) == int(row.anchor_start_pos)
        )
        payload = row._asdict()
        payload.update({
            "entry_ordinal": entry_ordinal,
            "tape_anchor_ordinal": tape_anchor_ordinal,
            "entry_state_position": entry_state_position,
            "frozen_exit_state_position": exit_state_position,
            "admission_state": admission_state,
            "admission_regime_start_pos": int(admission.start_pos),
            "admission_regime_age_bars": admission_age,
            "required_regime_age_at_frozen_close": required_age,
            "orientation_survived_to_admission": bool(orientation_survived),
            "duration_percentile_2024": duration_percentile,
            "exit_before_frozen_close_hazard_2024": exit_hazard,
            "hazard_support_2024": hazard_support,
            "realized_regime_exited_before_frozen_close": bool(int(admission.end_pos) < exit_state_position),
            "age_bin": age_bin(admission_age),
            "duration_percentile_bin": percentile_bin(duration_percentile),
            "exit_hazard_bin": hazard_bin(exit_hazard),
            "atr14_prior": atr if atr_valid else math.nan,
            "mfe_bps": mfe,
            "mae_bps": mae,
            "time_to_mfe_bars": time_mfe,
            "time_to_mae_bars": time_mae,
            "post_entry_mfe_bps": post_mfe,
            "post_entry_mae_bps": post_mae,
            "post_entry_time_to_mfe_bars": post_time_mfe,
            "post_entry_time_to_mae_bars": post_time_mae,
            "mfe_atr": (mfe / 10000.0 * entry / atr) if atr_valid else math.nan,
            "mae_atr": (mae / 10000.0 * entry / atr) if atr_valid else math.nan,
            "net_return_bps": net_bps,
            "final_positive": bool(final_positive),
            "path_class": path_class,
        })
        rows.append(payload)
    result = pd.DataFrame(rows)
    if result.empty or not np.isfinite(result["exit_before_frozen_close_hazard_2024"]).all():
        raise AssertionError(f"empty or invalid enriched signal population {period}")
    return result


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(score, method="average")
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def moving_block_sample(sessions: list[str], rng: np.random.Generator) -> list[str]:
    n = len(sessions)
    blocks = int(math.ceil(n / BOOTSTRAP_BLOCK))
    starts = rng.integers(0, n, size=blocks)
    sampled = [sessions[(int(start) + offset) % n] for start in starts for offset in range(BOOTSTRAP_BLOCK)]
    return sampled[:n]


def hazard_bootstrap(group: pd.DataFrame, seed_offset: int) -> dict[str, Any]:
    sessions = sorted(group["session_date"].astype(str).unique())
    observed = auc_score(
        group["final_positive"].to_numpy(bool),
        -group["exit_before_frozen_close_hazard_2024"].to_numpy(float),
    )
    by_session = {date: block for date, block in group.groupby("session_date", sort=False)}
    rng = np.random.default_rng(SEED + seed_offset)
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = moving_block_sample(sessions, rng)
        replay = pd.concat([by_session[date] for date in sampled], ignore_index=True)
        values[draw] = auc_score(
            replay["final_positive"].to_numpy(bool),
            -replay["exit_before_frozen_close_hazard_2024"].to_numpy(float),
        )
    values = values[np.isfinite(values)]
    return {
        "auc_negative_hazard": observed,
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_one_sided": float((1 + np.sum(values <= 0.5)) / (len(values) + 1)),
        "draws_valid": int(len(values)),
        "sessions": len(sessions),
    }


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    p = output["p_one_sided"].to_numpy(float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    output["holm_adjusted_p"] = adjusted
    output["passes_holm_0_05"] = output["holm_adjusted_p"].lt(0.05) & output["ci_lower"].gt(0.5)
    return output


def summarize_subset(name: str, period: int, subset: pd.DataFrame) -> dict[str, Any]:
    positive = subset["final_positive"].to_numpy(bool)
    survive = subset["orientation_survived_to_admission"].to_numpy(bool)
    survive_diff = math.nan
    if survive.any() and (~survive).any():
        survive_diff = float(subset.loc[survive, "net_return_bps"].mean() - subset.loc[~survive, "net_return_bps"].mean())
    return {
        "cohort": name,
        "period": period,
        "rows": int(len(subset)),
        "stocks": int(subset["symbol_norm"].nunique()),
        "sessions": int(subset["session_date"].nunique()),
        "mean_net_bps": float(subset["net_return_bps"].mean()),
        "median_net_bps": float(subset["net_return_bps"].median()),
        "positive_rate": float(positive.mean()),
        "mean_mfe_bps": float(subset["mfe_bps"].mean()),
        "mean_mae_bps": float(subset["mae_bps"].mean()),
        "orientation_survival_rate": float(survive.mean()),
        "survival_mean_net_difference_bps": survive_diff,
        "negative_hazard_auc": auc_score(positive, -subset["exit_before_frozen_close_hazard_2024"].to_numpy(float)),
        "timing_failure_share_all": float(subset["path_class"].eq("timing_failure").mean()),
        "no_usable_move_share_all": float(subset["path_class"].eq("no_usable_move").mean()),
        "timing_failure_share_losses": float(subset.loc[~positive, "path_class"].eq("timing_failure").mean()) if (~positive).any() else math.nan,
    }


def candidate_mask(frame: pd.DataFrame, loop: str, state: int) -> pd.Series:
    return frame["top_loop"].eq(loop) & frame["anchor_state"].eq(state)


def artifact_manifest(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "artifact_manifest.json"):
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    contract, source_hashes = load_and_verify_contract()
    args.out.mkdir(parents=True)
    cycles = load_cycles(Path(contract["inputs"]["fixed_cycles"]))
    anchors = pd.concat([
        load_anchor_panel(Path(contract["inputs"]["anchor_panels"][str(period)]), period, cycles)
        for period in PERIODS
    ], ignore_index=True)
    ledger = pd.read_parquet(Path(contract["inputs"]["accepted_signal_ledger"]))
    ledger["period"] = pd.to_numeric(ledger["period"], errors="raise").astype(int)
    ledger["session_date"] = ledger["session_date"].astype(str)
    source = ledger.loc[
        ledger["period"].isin(PERIODS)
        & ledger["strategy"].eq(contract["population"]["source_strategy"])
        & ledger["horizon"].eq(contract["population"]["source_horizon_bars_from_anchor"])
        & ledger["status"].eq(contract["population"]["source_status"])
    ].copy()
    source = source.merge(
        anchors,
        on=["period", "anchor_id", "symbol_norm", "session_date", "start_timestamp"],
        how="left",
        validate="many_to_one",
    )
    if source["top_loop"].isna().any():
        raise AssertionError("anchor context join failure")
    runs_2024 = load_runs(Path(contract["inputs"]["runs"]["2024_hazard_fit"]), 2024)
    hazard_fit = hazard_training(runs_2024)
    enriched: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for period in PERIODS:
        symbols = list(contract["population"]["symbols"])
        tape, tape_audit = load_tape(Path(contract["inputs"]["provider_roots"][str(period)]), symbols, period)
        coverage.extend(tape_audit)
        runs = load_runs(Path(contract["inputs"]["runs"][str(period)]), period)
        enriched.append(enrich_signals(source.loc[source["period"].eq(period)].copy(), period, tape, run_lookup(runs), hazard_fit))
    scored = pd.concat(enriched, ignore_index=True)
    scored = scored.sort_values(["period", "session_date", "symbol_norm", "bar_ordinal"], kind="stable").reset_index(drop=True)
    scored.to_parquet(args.out / "signal_level_path_diagnostics.parquet", index=False)
    pd.DataFrame(coverage).to_csv(args.out / "data_coverage.csv", index=False)

    cohort_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    candidate_specs = [("cycle_04", 4, 2), ("cycle_07", 5, 6)]
    seed_offset = 0
    for period in PERIODS:
        period_frame = scored.loc[scored["period"].eq(period)].copy()
        cohort_rows.append(summarize_subset("unfiltered_frozen_signal", period, period_frame))
        for loop, candidate_state, control_state in candidate_specs:
            loop_frame = period_frame.loc[period_frame["top_loop"].eq(loop)].copy()
            candidate = period_frame.loc[candidate_mask(period_frame, loop, candidate_state)].copy()
            control = period_frame.loc[candidate_mask(period_frame, loop, control_state)].copy()
            cohort_rows.extend([
                summarize_subset(f"{loop}|loop_only", period, loop_frame),
                summarize_subset(f"{loop}|state{candidate_state}|candidate", period, candidate),
                summarize_subset(f"{loop}|state{control_state}|matched_control", period, control),
            ])
            if len(candidate) < int(contract["inference"]["minimum_candidate_period_rows"]):
                raise AssertionError(f"candidate support below contract: {loop} {period}")
            for outcome, group in candidate.groupby("path_class", sort=True):
                path_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}", "path_class": outcome,
                    "rows": int(len(group)), "share": float(len(group) / len(candidate)),
                    "mean_net_bps": float(group["net_return_bps"].mean()),
                    "mean_mfe_bps": float(group["mfe_bps"].mean()),
                    "mean_mae_bps": float(group["mae_bps"].mean()),
                    "median_time_to_mfe_bars": float(group["time_to_mfe_bars"].median()),
                    "median_time_to_mae_bars": float(group["time_to_mae_bars"].median()),
                    "mean_mfe_atr": float(group["mfe_atr"].mean()),
                    "mean_mae_atr": float(group["mae_atr"].mean()),
                })
            for profitable, group in candidate.groupby("final_positive", sort=True):
                feature_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}",
                    "outcome": "positive" if profitable else "nonpositive", "rows": int(len(group)),
                    "mean_net_bps": float(group["net_return_bps"].mean()),
                    "median_regime_age_bars": float(group["admission_regime_age_bars"].median()),
                    "mean_duration_percentile": float(group["duration_percentile_2024"].mean()),
                    "mean_exit_hazard": float(group["exit_before_frozen_close_hazard_2024"].mean()),
                    "orientation_survival_rate": float(group["orientation_survived_to_admission"].mean()),
                    "mean_mfe_bps": float(group["mfe_bps"].mean()),
                    "mean_mae_bps": float(group["mae_bps"].mean()),
                })
            for quarter, group in candidate.groupby("quarter", sort=True):
                quarter_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}", "quarter": quarter,
                    "rows": int(len(group)), "mean_net_bps": float(group["net_return_bps"].mean()),
                    "negative_hazard_auc": auc_score(group["final_positive"].to_numpy(bool), -group["exit_before_frozen_close_hazard_2024"].to_numpy(float)),
                    "orientation_survival_rate": float(group["orientation_survived_to_admission"].mean()),
                })
            boot = hazard_bootstrap(candidate, seed_offset)
            seed_offset += 1
            bootstrap_rows.append({"period": period, "candidate": f"{loop}|state{candidate_state}", "rows": len(candidate), **boot})
            for deleted in contract["population"]["symbols"]:
                subset = candidate.loc[~candidate["symbol_norm"].eq(deleted)]
                survive = subset["orientation_survived_to_admission"].to_numpy(bool)
                survival_diff = math.nan
                if survive.any() and (~survive).any():
                    survival_diff = float(subset.loc[survive, "net_return_bps"].mean() - subset.loc[~survive, "net_return_bps"].mean())
                deletion_rows.append({
                    "period": period, "candidate": f"{loop}|state{candidate_state}", "deleted_symbol": deleted,
                    "rows": int(len(subset)),
                    "negative_hazard_auc": auc_score(subset["final_positive"].to_numpy(bool), -subset["exit_before_frozen_close_hazard_2024"].to_numpy(float)),
                    "orientation_survival_mean_net_difference_bps": survival_diff,
                })

    cohorts = pd.DataFrame(cohort_rows)
    features = pd.DataFrame(feature_rows)
    paths = pd.DataFrame(path_rows)
    quarters = pd.DataFrame(quarter_rows)
    bootstraps = holm_adjust(pd.DataFrame(bootstrap_rows))
    deletions = pd.DataFrame(deletion_rows)
    cohorts.to_csv(args.out / "cohort_metrics.csv", index=False)
    features.to_csv(args.out / "profitable_vs_losing_features.csv", index=False)
    paths.to_csv(args.out / "path_class_metrics.csv", index=False)
    quarters.to_csv(args.out / "quarter_metrics.csv", index=False)
    bootstraps.to_csv(args.out / "hazard_auc_bootstraps.csv", index=False)
    deletions.to_csv(args.out / "stock_deletion_metrics.csv", index=False)

    candidate_cohorts = cohorts.loc[cohorts["cohort"].str.endswith("|candidate")]
    deletion_checks = deletions.assign(
        hazard_positive=deletions["negative_hazard_auc"].gt(0.5),
        survival_positive=deletions["orientation_survival_mean_net_difference_bps"].gt(0.0),
    ).groupby(["period", "candidate"], as_index=False).agg(
        hazard_positive_deletions=("hazard_positive", "sum"),
        survival_positive_deletions=("survival_positive", "sum"),
    )
    checks = {
        "support_at_least_50_all_four_cells": bool(candidate_cohorts["rows"].ge(50).all() and len(candidate_cohorts) == 4),
        "negative_hazard_auc_above_half_all_four_cells": bool(bootstraps["auc_negative_hazard"].gt(0.5).all() and len(bootstraps) == 4),
        "all_four_hazard_endpoints_pass_holm": bool(bootstraps["passes_holm_0_05"].all() and len(bootstraps) == 4),
        "orientation_survival_difference_positive_all_four_cells": bool(candidate_cohorts["survival_mean_net_difference_bps"].gt(0.0).all() and len(candidate_cohorts) == 4),
        "hazard_leave_one_stock_out_at_least_16_all_four_cells": bool(deletion_checks["hazard_positive_deletions"].ge(16).all() and len(deletion_checks) == 4),
    }
    supported = bool(all(checks.values()))
    decision = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "application_modified": False,
        "sealed_validation_performed": False,
        "prospective_validation_claim": False,
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "primary_cost_bps_per_side": 5,
        "checks": checks,
        "decision": "phase_hazard_features_supported_for_prospective_logging_only" if supported else "phase_hazard_features_not_supported_as_payoff_admission_discriminator",
    }
    write_json(args.out / "decision.json", decision)
    summary = {
        "contract_id": contract["contract_id"],
        "scientific_status": contract["scientific_status"],
        "sealed_data_status": contract["sealed_data_status"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": contract["evidence_labels"]["volume"],
        "quotes_or_ticks_used": False,
        "score_rows": int(len(scored)),
        "candidate_metrics": candidate_cohorts.to_dict("records"),
        "hazard_bootstraps": bootstraps.to_dict("records"),
        "deletion_checks": deletion_checks.to_dict("records"),
        "path_metrics": paths.to_dict("records"),
        "decision": decision,
    }
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "source_hashes.json", {
        "contract_id": contract["contract_id"],
        "frozen_before_scoring": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "sha256": source_hashes,
    })
    write_json(args.out / "artifact_manifest.json", artifact_manifest(args.out))
    print(json.dumps({
        "out": str(args.out),
        "decision": decision["decision"],
        "rows": len(scored),
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
