#!/usr/bin/env python3
"""Independent replay audit for dynamic-loop-context-edge-v1.

This file intentionally does not import the research runner.
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
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260713-dynamic-loop-context-edge-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260713-dynamic-loop-context-edge-v1-pre-score.json"
RUNNER_PATH = HERE / "run_dynamic_loop_context_edge_v1.py"
PERIODS = (2025, 2023)
HORIZONS = (6, 12, 24)
PRIMARY = "loop_current_regime"
SOURCE_STRATEGY = "breakout_loop_scores_range_p75"
LOOP_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))
WINDOW = 60
MIN_SUPPORT = 20
PSEUDOCOUNT = 50.0
ROUND_TRIP_COST_BPS = 10.0
UNIVERSE_SIZE = 20
BLOCK_SESSIONS = 20
BLOCK_MIN_SUPPORT = 5
SEED = 20260713
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "anchor_panel_2023": Path(contract["inputs"]["anchor_panels"]["2023"]),
        "anchor_panel_2024": Path(
            contract["inputs"]["anchor_panels"]["2024_threshold_fit"]
        ),
        "anchor_panel_2025": Path(contract["inputs"]["anchor_panels"]["2025"]),
        "accepted_signal_ledger": Path(contract["inputs"]["accepted_signal_ledger"]),
        "fixed_cycles": Path(contract["inputs"]["fixed_cycles"]),
        "execution_manifest": Path(contract["inputs"]["execution_manifest"]),
    }
    for period in PERIODS:
        root = Path(contract["inputs"]["provider_roots"][str(period)])
        for symbol in contract["population"]["symbols"]:
            paths[f"provider_{period}_{symbol}"] = provider_path(root, symbol)
    return paths


def regular_sessions(path: Path, period: int) -> list[str]:
    frame = pd.read_parquet(path, columns=["timestamp"])
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    timestamp = timestamp.loc[
        timestamp.ge(pd.Timestamp(f"{period}-01-01", tz="UTC"))
        & timestamp.lt(pd.Timestamp(f"{period + 1}-01-01", tz="UTC"))
    ]
    local = timestamp.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    return sorted(local.loc[minute.ge(570) & minute.lt(960)].dt.strftime("%Y-%m-%d").unique())


def top_loop_context(
    anchor_path: Path, cycles: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "anchor_id",
        "state",
        "previous_state_1",
        "history_token",
        *LOOP_COLUMNS,
    ]
    frame = pd.read_parquet(anchor_path, columns=columns)
    values = frame.loc[:, LOOP_COLUMNS].to_numpy(float)
    top_index = np.argmax(values, axis=1)
    frame["top_loop"] = cycles["cycle_id"].to_numpy(str)[top_index]
    frame["top_loop_probability_replay"] = values[np.arange(len(frame)), top_index]
    frame["cell_key"] = (
        frame["top_loop"] + "|c" + frame["state"].astype(int).astype(str)
    )
    members = [
        {int(item) for item in value.split("->")} for value in cycles["cycle"].astype(str)
    ]
    compatible = [
        int(state) in members[index]
        for state, index in zip(frame["state"], top_index, strict=True)
    ]
    if not all(compatible):
        raise AssertionError("auditor top-loop compatibility failure")
    decoded_previous = (frame["history_token"].to_numpy(int) % 72) // 8
    decoded_state = frame["history_token"].to_numpy(int) % 8
    if not np.array_equal(decoded_previous, frame["previous_state_1"].to_numpy(int)):
        raise AssertionError("auditor history previous-state decode failure")
    if not np.array_equal(decoded_state, frame["state"].to_numpy(int)):
        raise AssertionError("auditor history state decode failure")
    return frame.loc[
        :,
        [
            "anchor_id",
            "state",
            "previous_state_1",
            "history_token",
            "top_loop",
            "top_loop_probability_replay",
            "cell_key",
        ],
    ]


def build_primary_source(
    contract: dict[str, Any], cycles: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    source = pd.read_parquet(Path(contract["inputs"]["accepted_signal_ledger"]))
    source = source.loc[
        source["strategy"].eq(SOURCE_STRATEGY)
        & source["horizon"].isin(HORIZONS)
        & source["period"].astype(str).isin([str(period) for period in PERIODS])
    ].copy()
    sessions_by_period: dict[int, list[str]] = {}
    rows: list[pd.DataFrame] = []
    first_symbol = contract["population"]["symbols"][0]
    for period in PERIODS:
        root = Path(contract["inputs"]["provider_roots"][str(period)])
        sessions = regular_sessions(provider_path(root, first_symbol), period)
        if len(sessions) != 250:
            raise AssertionError("auditor session count failure")
        sessions_by_period[period] = sessions
        context = top_loop_context(
            Path(contract["inputs"]["anchor_panels"][str(period)]), cycles
        )
        selected = source.loc[source["period"].astype(str).eq(str(period))].copy()
        selected = selected.merge(context, on="anchor_id", how="left", validate="many_to_one")
        if selected["top_loop"].isna().any():
            raise AssertionError("auditor source join failure")
        selected["session_index"] = selected["session_date"].map(
            {date: index for index, date in enumerate(sessions)}
        )
        selected["net_return_bps_replay"] = np.where(
            selected["status"].eq("filled"),
            selected["gross_return_bps"].to_numpy(float) - ROUND_TRIP_COST_BPS,
            np.nan,
        )
        rows.append(selected)
    return pd.concat(rows, ignore_index=True), sessions_by_period


def replay_primary_selector(
    source: pd.DataFrame, sessions_by_period: dict[int, list[str]]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for period in PERIODS:
        sessions = sessions_by_period[period]
        for horizon in HORIZONS:
            frame = source.loc[
                source["period"].astype(str).eq(str(period))
                & source["horizon"].eq(horizon)
            ].copy()
            keys = sorted(frame["cell_key"].unique())
            key_to_loop = dict(
                frame[["cell_key", "top_loop"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            ages = {key: 0 for key in keys}
            for score_index in range(WINDOW, len(sessions)):
                past = frame.loc[
                    frame["status"].eq("filled")
                    & frame["session_index"].ge(score_index - WINDOW)
                    & frame["session_index"].lt(score_index)
                ]
                current = frame.loc[frame["session_index"].eq(score_index)].copy()
                global_mean = float(past["net_return_bps_replay"].mean())
                loops = past.groupby("top_loop")["net_return_bps_replay"].agg(["size", "sum"])
                cells = past.groupby("cell_key")["net_return_bps_replay"].agg(["size", "sum"])
                loop_estimates: dict[str, float] = {}
                loop_support: dict[str, int] = {}
                for loop in sorted(frame["top_loop"].unique()):
                    n_loop = int(loops.loc[loop, "size"]) if loop in loops.index else 0
                    sum_loop = float(loops.loc[loop, "sum"]) if loop in loops.index else 0.0
                    loop_support[loop] = n_loop
                    loop_estimates[loop] = (
                        (sum_loop + PSEUDOCOUNT * global_mean) / (n_loop + PSEUDOCOUNT)
                        if n_loop
                        else global_mean
                    )
                state: dict[str, tuple[float, int, int, bool, bool, int]] = {}
                for key in keys:
                    loop = key_to_loop[key]
                    support = int(cells.loc[key, "size"]) if key in cells.index else 0
                    total = float(cells.loc[key, "sum"]) if key in cells.index else 0.0
                    individualized = support >= MIN_SUPPORT
                    estimate = (
                        (total + PSEUDOCOUNT * loop_estimates[loop])
                        / (support + PSEUDOCOUNT)
                        if individualized
                        else loop_estimates[loop]
                    )
                    active = bool(loop_support[loop] >= MIN_SUPPORT and estimate > 0.0)
                    ages[key] = ages[key] + 1 if active else 0
                    state[key] = (
                        estimate,
                        support,
                        loop_support[loop],
                        individualized,
                        active,
                        ages[key],
                    )
                if current.empty:
                    continue
                values = current["cell_key"].map(state)
                current["estimate_replay"] = values.map(lambda value: value[0])
                current["support_replay"] = values.map(lambda value: value[1])
                current["loop_support_replay"] = values.map(lambda value: value[2])
                current["individualized_replay"] = values.map(lambda value: value[3])
                current["active_replay"] = values.map(lambda value: value[4])
                current["age_replay"] = values.map(lambda value: value[5])
                rows.append(current)
    return pd.concat(rows, ignore_index=True)


def daily_return(
    frame: pd.DataFrame, sessions: list[str], selector: str
) -> pd.Series:
    selected = frame.loc[frame["status"].eq("filled")].copy()
    if selector != "unfiltered":
        selected = selected.loc[
            selected[f"selector__{selector}__active"].astype(bool)
        ].copy()
    selected["net_return"] = selected["net_return_bps"].to_numpy(float) / 10000.0
    selected["log_growth"] = np.log1p(selected["net_return"].to_numpy(float))
    sleeve = np.expm1(
        selected.groupby(["session_date", "symbol_norm"])["log_growth"].sum()
    )
    return (sleeve.groupby("session_date").sum() / UNIVERSE_SIZE).reindex(
        sessions, fill_value=0.0
    )


def cumulative(values: np.ndarray) -> float:
    return float(np.prod(1.0 + np.asarray(values, dtype=float)) - 1.0)


def moving_samples(values: np.ndarray, seed_offset: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED + seed_offset)
    starts = np.arange(len(data) - BOOTSTRAP_BLOCK + 1)
    blocks = math.ceil(len(data) / BOOTSTRAP_BLOCK)
    selected = rng.choice(starts, size=(BOOTSTRAP_DRAWS, blocks), replace=True)
    positions = (
        selected[:, :, None] + np.arange(BOOTSTRAP_BLOCK)[None, None, :]
    ).reshape(BOOTSTRAP_DRAWS, -1)[:, : len(data)]
    return data[positions].mean(axis=1)


def holm(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(p) - rank) * p[position])
        adjusted[position] = min(1.0, running)
    return adjusted


def volume_replay_for_symbol(path: Path, period: int) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["timestamp", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "volume"])
    frame = frame.loc[
        frame["timestamp"].ge(pd.Timestamp(f"{period - 2}-01-01", tz="UTC"))
        & frame["timestamp"].lt(pd.Timestamp(f"{period + 1}-01-01", tz="UTC"))
    ].copy()
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["session_date"] = local.dt.strftime("%Y-%m-%d")
    frame["ordinal"] = frame.groupby("session_date").cumcount()
    baseline = frame.groupby("ordinal")["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).median()
    )
    frame["volume_ratio_replay"] = np.where(
        baseline.to_numpy(float) > 0.0,
        frame["volume"].to_numpy(float) / baseline.to_numpy(float),
        np.nan,
    )
    return frame.loc[
        pd.to_datetime(frame["session_date"]).dt.year.eq(period),
        ["timestamp", "volume_ratio_replay"],
    ]


def manifest_for(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            )
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}

    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    decision = json.loads((root / "decision.json").read_text())
    initial_manifest = json.loads((root / "artifact_manifest.json").read_text())
    checks["safety_contract"] = bool(
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
    )
    checks["safety_decision"] = bool(
        decision["research_only"] is True
        and decision["live_ordering_enabled"] is False
        and decision["order_placement"] == "disabled"
        and decision["live_or_paper_use_authorized"] is False
    )
    manifest_map = {row["name"]: row for row in initial_manifest["files"]}
    checks["initial_artifact_manifest"] = all(
        (root / name).exists()
        and (root / name).stat().st_size == row["bytes"]
        and digest(root / name) == row["sha256"]
        for name, row in manifest_map.items()
    )
    actual_sources = {name: digest(path) for name, path in source_paths(contract).items()}
    checks["source_hashes"] = actual_sources == pre_score["sha256"]

    scored = pd.read_parquet(root / "scored_signal_ledger.parquet")
    forbidden_fragments = (
        "loop_occurs",
        "future_state",
        "direction_target",
        "return_bps_target",
        "realized_loop",
        "next_state",
    )
    checks["no_forbidden_future_columns"] = not any(
        fragment in column.lower()
        for column in scored.columns
        for fragment in forbidden_fragments
    )
    checks["net_return_identity"] = bool(
        np.allclose(
            scored.loc[scored["status"].eq("filled"), "net_return_bps"],
            scored.loc[scored["status"].eq("filled"), "gross_return_bps"]
            - ROUND_TRIP_COST_BPS,
            atol=1e-12,
            rtol=0.0,
        )
        and scored.loc[~scored["status"].eq("filled"), "net_return_bps"].isna().all()
    )
    chronology = pd.read_csv(root / "chronology_audit.csv")
    checks["chronology"] = bool(
        len(chronology) == 6 * 190
        and not chronology["same_session_outcomes_used"].astype(bool).any()
        and (
            pd.to_datetime(chronology["training_last_session"])
            < pd.to_datetime(chronology["score_session"])
        ).all()
    )

    cycles = pd.read_csv(Path(contract["inputs"]["fixed_cycles"]))
    primary_source, sessions_by_period = build_primary_source(contract, cycles)
    replay = replay_primary_selector(primary_source, sessions_by_period)
    compare_columns = ["period", "horizon", "anchor_id"]
    comparison = scored.merge(
        replay[
            [
                *compare_columns,
                "top_loop",
                "top_loop_probability_replay",
                "cell_key",
                "net_return_bps_replay",
                "estimate_replay",
                "support_replay",
                "loop_support_replay",
                "individualized_replay",
                "active_replay",
                "age_replay",
            ]
        ],
        on=compare_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_replayed"),
    )
    checks["primary_replay_population"] = bool(
        len(comparison) == len(scored) and comparison["_merge"].eq("both").all()
    )
    probability_error = float(
        np.max(
            np.abs(
                comparison["top_loop_probability"].to_numpy(float)
                - comparison["top_loop_probability_replay"].to_numpy(float)
            )
        )
    )
    estimate_error = float(
        np.max(
            np.abs(
                comparison[f"selector__{PRIMARY}__estimate_net_bps"].to_numpy(float)
                - comparison["estimate_replay"].to_numpy(float)
            )
        )
    )
    metrics["maximum_top_loop_probability_error"] = probability_error
    metrics["maximum_primary_estimate_error_bps"] = estimate_error
    checks["top_loop_identity_replay"] = bool(
        comparison["top_loop"].eq(comparison["top_loop_replayed"]).all()
        and comparison[f"key__{PRIMARY}"].eq(comparison["cell_key"]).all()
        and probability_error <= 1e-15
    )
    checks["primary_numeric_replay"] = bool(
        estimate_error <= 1e-12
        and comparison[f"selector__{PRIMARY}__support"].eq(
            comparison["support_replay"]
        ).all()
        and comparison[f"selector__{PRIMARY}__loop_support"].eq(
            comparison["loop_support_replay"]
        ).all()
        and comparison[f"selector__{PRIMARY}__active_age_sessions"].eq(
            comparison["age_replay"]
        ).all()
    )
    checks["primary_boolean_replay"] = bool(
        comparison[f"selector__{PRIMARY}__individualized"].astype(bool).eq(
            comparison["individualized_replay"].astype(bool)
        ).all()
        and comparison[f"selector__{PRIMARY}__active"].astype(bool).eq(
            comparison["active_replay"].astype(bool)
        ).all()
    )

    daily_artifact = pd.read_parquet(root / "daily_portfolio_returns.parquet")
    selector_metrics = pd.read_csv(root / "selector_metrics.csv")
    max_daily_error = 0.0
    max_metric_error = 0.0
    for period in PERIODS:
        sessions = sessions_by_period[period][WINDOW:]
        for horizon in HORIZONS:
            surface = scored.loc[
                scored["period"].astype(str).eq(str(period))
                & scored["horizon"].eq(horizon)
            ]
            for selector in ("unfiltered", "loop_only", PRIMARY):
                replay_daily = daily_return(surface, sessions, selector)
                stored_daily = daily_artifact.loc[
                    daily_artifact["period"].eq(period)
                    & daily_artifact["horizon"].eq(horizon)
                    & daily_artifact["selector"].eq(selector)
                ].sort_values("session_date")
                max_daily_error = max(
                    max_daily_error,
                    float(
                        np.max(
                            np.abs(
                                replay_daily.to_numpy(float)
                                - stored_daily["daily_return"].to_numpy(float)
                            )
                        )
                    ),
                )
                active = (
                    np.ones(len(surface), dtype=bool)
                    if selector == "unfiltered"
                    else surface[f"selector__{selector}__active"].astype(bool).to_numpy()
                )
                selected = surface.loc[active & surface["status"].eq("filled")]
                stored = selector_metrics.loc[
                    selector_metrics["period"].eq(period)
                    & selector_metrics["horizon"].eq(horizon)
                    & selector_metrics["selector"].eq(selector)
                ].iloc[0]
                values = selected["net_return_bps"].to_numpy(float)
                candidates = [
                    abs(float(stored["filled_trades"]) - len(selected)),
                    abs(float(stored["active_signal_fraction"]) - float(active.mean())),
                    abs(float(stored["mean_net_trade_bps"]) - float(values.mean())),
                    abs(float(stored["cumulative_return"]) - cumulative(replay_daily)),
                ]
                if selector == PRIMARY:
                    candidates.append(
                        abs(
                            float(stored["individualized_decision_fraction"])
                            - float(
                                surface[f"selector__{PRIMARY}__individualized"]
                                .astype(bool)
                                .mean()
                            )
                        )
                    )
                max_metric_error = max(max_metric_error, max(candidates))
    metrics["maximum_daily_return_error"] = max_daily_error
    metrics["maximum_selector_metric_error"] = max_metric_error
    checks["daily_return_replay"] = max_daily_error <= 1e-15
    checks["selector_metric_replay"] = max_metric_error <= 1e-12

    bootstrap_stored = pd.read_csv(root / "primary_bootstraps.csv")
    bootstrap_rows: list[dict[str, float]] = []
    seed_offset = 0
    for period in PERIODS:
        for horizon in HORIZONS:
            surface = daily_artifact.loc[
                daily_artifact["period"].eq(period)
                & daily_artifact["horizon"].eq(horizon)
            ]
            candidate = surface.loc[surface["selector"].eq(PRIMARY)].sort_values(
                "session_date"
            )["daily_return"].to_numpy(float)
            for comparison_name, baseline_name in (
                ("absolute", None),
                ("versus_unfiltered", "unfiltered"),
                ("versus_loop_only", "loop_only"),
            ):
                values = candidate.copy()
                if baseline_name is not None:
                    baseline = surface.loc[
                        surface["selector"].eq(baseline_name)
                    ].sort_values("session_date")["daily_return"].to_numpy(float)
                    values -= baseline
                samples = moving_samples(values, seed_offset)
                seed_offset += 1
                lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
                bootstrap_rows.append(
                    {
                        "mean_daily_difference": float(values.mean()),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "p_one_sided": (1.0 + float((samples <= 0.0).sum()))
                        / (BOOTSTRAP_DRAWS + 1.0),
                    }
                )
    bootstrap_replay = pd.DataFrame(bootstrap_rows)
    bootstrap_replay["holm_adjusted_p"] = holm(
        bootstrap_replay["p_one_sided"].to_numpy(float)
    )
    bootstrap_error = float(
        np.max(
            np.abs(
                bootstrap_replay[
                    [
                        "mean_daily_difference",
                        "ci_lower",
                        "ci_upper",
                        "p_one_sided",
                        "holm_adjusted_p",
                    ]
                ].to_numpy(float)
                - bootstrap_stored[
                    [
                        "mean_daily_difference",
                        "ci_lower",
                        "ci_upper",
                        "p_one_sided",
                        "holm_adjusted_p",
                    ]
                ].to_numpy(float)
            )
        )
    )
    metrics["maximum_bootstrap_error"] = bootstrap_error
    checks["bootstrap_replay"] = bootstrap_error <= 1e-15

    lifetime_stored = pd.read_csv(root / "lifetime_comparisons.csv")
    lifetime_rows = []
    for period in PERIODS:
        for horizon in HORIZONS:
            selected = scored.loc[
                scored["period"].astype(str).eq(str(period))
                & scored["horizon"].eq(horizon)
                & scored[f"selector__{PRIMARY}__active"].astype(bool)
                & scored["status"].eq("filled")
            ]
            age = selected[f"selector__{PRIMARY}__active_age_sessions"].to_numpy(int)
            values = selected["net_return_bps"].to_numpy(float)
            early = values[(age >= 1) & (age <= 10)]
            late = values[age >= 21]
            lifetime_rows.append(
                [
                    len(early),
                    float(early.mean()),
                    len(late),
                    float(late.mean()),
                    float(late.mean() - early.mean()),
                ]
            )
    lifetime_error = float(
        np.max(
            np.abs(
                np.asarray(lifetime_rows, dtype=float)
                - lifetime_stored[
                    [
                        "early_1_10_trades",
                        "early_1_10_mean_net_bps",
                        "late_21_plus_trades",
                        "late_21_plus_mean_net_bps",
                        "late_minus_early_mean_net_bps",
                    ]
                ].to_numpy(float)
            )
        )
    )
    metrics["maximum_lifetime_error"] = lifetime_error
    checks["lifetime_replay"] = lifetime_error <= 1e-12

    cell_stored = pd.read_csv(root / "cell_block_profitability.csv")
    cell_replays: list[pd.DataFrame] = []
    for period in PERIODS:
        sessions = sessions_by_period[period]
        for horizon in HORIZONS:
            frame = primary_source.loc[
                primary_source["period"].astype(str).eq(str(period))
                & primary_source["horizon"].eq(horizon)
                & primary_source["status"].eq("filled")
            ].copy()
            frame["block_index"] = frame["session_index"] // BLOCK_SESSIONS
            group = (
                frame.groupby(["block_index", "cell_key", "top_loop", "state"])[
                    "net_return_bps_replay"
                ]
                .agg(["size", "mean", "median"])
                .reset_index()
                .rename(
                    columns={
                        "size": "filled_trades",
                        "mean": "mean_net_trade_bps",
                        "median": "median_net_trade_bps",
                    }
                )
            )
            group["period"] = period
            group["horizon"] = horizon
            group["supported"] = group["filled_trades"].ge(BLOCK_MIN_SUPPORT)
            group["profitable"] = group["mean_net_trade_bps"].gt(0.0)
            group["block_start_session"] = group["block_index"].map(
                lambda value: sessions[int(value) * BLOCK_SESSIONS]
            )
            group["block_end_session"] = group["block_index"].map(
                lambda value: sessions[
                    min((int(value) + 1) * BLOCK_SESSIONS, len(sessions)) - 1
                ]
            )
            cell_replays.append(group)
    cell_replay = pd.concat(cell_replays, ignore_index=True)
    key_columns = ["period", "horizon", "block_index", "cell_key"]
    cell_compare = cell_stored.merge(
        cell_replay,
        on=key_columns,
        how="outer",
        validate="one_to_one",
        suffixes=("", "_replay"),
        indicator=True,
    )
    cell_error = float(
        np.max(
            np.abs(
                cell_compare[
                    ["filled_trades", "mean_net_trade_bps", "median_net_trade_bps"]
                ].to_numpy(float)
                - cell_compare[
                    [
                        "filled_trades_replay",
                        "mean_net_trade_bps_replay",
                        "median_net_trade_bps_replay",
                    ]
                ].to_numpy(float)
            )
        )
    )
    metrics["maximum_cell_block_error"] = cell_error
    checks["cell_block_replay"] = bool(
        cell_compare["_merge"].eq("both").all()
        and cell_compare["supported"].eq(cell_compare["supported_replay"]).all()
        and cell_compare["profitable"].eq(cell_compare["profitable_replay"]).all()
        and cell_error <= 1e-12
    )

    volume_errors = []
    for period in PERIODS:
        root_path = Path(contract["inputs"]["provider_roots"][str(period)])
        for symbol in ("AAL", "MSTR", "WULF"):
            replay_volume = volume_replay_for_symbol(
                provider_path(root_path, symbol), period
            )
            stored_volume = scored.loc[
                scored["period"].astype(str).eq(str(period))
                & scored["symbol_norm"].eq(symbol),
                ["start_timestamp", "volume_ratio", "volume_bucket"],
            ].drop_duplicates("start_timestamp")
            joined = stored_volume.merge(
                replay_volume,
                left_on="start_timestamp",
                right_on="timestamp",
                how="left",
                validate="one_to_one",
            )
            finite = np.isfinite(joined["volume_ratio_replay"].to_numpy(float))
            if finite.any():
                volume_errors.append(
                    float(
                        np.max(
                            np.abs(
                                joined.loc[finite, "volume_ratio"].to_numpy(float)
                                - joined.loc[finite, "volume_ratio_replay"].to_numpy(float)
                            )
                        )
                    )
                )
            expected_bucket = np.where(
                ~finite,
                "unknown",
                np.where(
                    joined["volume_ratio_replay"].to_numpy(float) >= 1.0,
                    "high",
                    "low",
                ),
            )
            if not np.array_equal(joined["volume_bucket"].astype(str), expected_bucket):
                volume_errors.append(math.inf)
    maximum_volume_error = max(volume_errors, default=0.0)
    metrics["maximum_sampled_volume_ratio_error"] = maximum_volume_error
    checks["sampled_historical_volume_replay"] = maximum_volume_error <= 1e-12

    checks["frozen_rejection_decision"] = bool(
        decision["decision"] == "dynamic_loop_context_profitability_hypothesis_not_supported"
        and decision["checks"]["overall_hypothesis_supported"] is False
        and decision["economic_edge_claim"] is False
        and decision["strategy_promotion"] is False
    )

    passed = all(checks.values())
    result = {
        "audit": "independent_dynamic_loop_context_edge_v1",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "auditor_sha256": digest(Path(__file__).resolve()),
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "metrics": metrics,
        "pass": passed,
        "rejection_verified": checks["frozen_rejection_decision"],
    }
    (root / "independent_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest_for(root), indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"independent audit failed: {failed}")


if __name__ == "__main__":
    main()
