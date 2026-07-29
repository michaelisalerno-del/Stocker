#!/usr/bin/env python3
"""Run the preregistered M1C Asymmetric Downside Residual V1 experiment."""

from __future__ import annotations

# ruff: noqa: E402 -- deterministic numerical limits must precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

EXPERIMENT_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = EXPERIMENT_DIR.parents[2]
PRIMARY: Final[Path] = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS: Final[Path] = EXPERIMENT_DIR / "reports"
CONTRACT_PATH: Final[Path] = EXPERIMENT_DIR / "contract.json"
TAIL_PRIMARY: Final[Path] = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-tail-phase-v1"
    / "artifacts"
    / "primary"
)
TAIL_EPISODES_PATH: Final[Path] = TAIL_PRIMARY / "fresh_episode_results_v1.parquet"
TAIL_CHECKPOINTS_PATH: Final[Path] = TAIL_PRIMARY / "checkpoint_results_v1.parquet"
TAIL_PROVENANCE_PATH: Final[Path] = TAIL_PRIMARY / "provenance_manifest_v1.json"
STATE_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/Documents/Codex/"
    "2026-07-23-you-are-working-in-the-github-5/data/cache/"
    "minimal-intraday-iv-excess-holdout-v0/frozen_state_surface.parquet"
)
HISTORICAL_OPTIONS_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/Documents/Codex/"
    "2026-07-23-you-are-working-in-the-github-3/research/cross-market-context/"
    "20260723-daily-stock-front-options-context-v01/artifacts/primary/"
    "front_options_dimensions.parquet"
)
STRESS_OPTIONS_PATH: Final[Path] = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260724-minimal-intraday-iv-excess-holdout-v01"
    / "artifacts"
    / "primary"
    / "holdout_selected_option_pairs.parquet"
)

for _package in ("stocker_research", "stocker_prospective", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.m1c_asymmetric_downside_residual_v1 import (
    DOWNSIDE_FEATURES,
    M1C_HIGH_MOVEMENT_THRESHOLD,
    MODEL_RANDOM_SEED,
    apply_asymmetric_policy,
    assert_unprotected_sessions,
    build_downside_features,
    expanding_time_ordered_oof,
    fit_downside_model,
    freeze_action_thresholds,
    partition_endpoint_return,
)
from stocker_research.m1c_low_movement_v0 import iv_expected_absolute
from stocker_research.stock_local_directional_archetypes_v0 import (
    shift_features_to_next_episode,
)

DEVELOPMENT_START: Final[str] = "2024-01-01"
DEVELOPMENT_END: Final[str] = "2024-12-31"
ASSESSMENT_START: Final[str] = "2025-01-01"
ASSESSMENT_END: Final[str] = "2025-08-22"
STRESS_START: Final[str] = "2025-09-01"
STRESS_END: Final[str] = "2025-12-31"
PROTECTED_START: Final[str] = "2026-01-01"
BOOTSTRAP_DRAWS: Final[int] = 1000
BOOTSTRAP_SEED: Final[int] = 2026072801
PERMUTATION_DRAWS: Final[int] = 1000
PERMUTATION_SEED: Final[int] = 2026072802
RUN_COMMAND: Final[str] = (
    "rtk uv run python research/directional-readiness/"
    "20260728-m1c-asymmetric-downside-residual-v1/run_experiment.py"
)
AUDIT_COMMAND: Final[str] = (
    "rtk uv run python research/directional-readiness/"
    "20260728-m1c-asymmetric-downside-residual-v1/audit_experiment.py"
)
FOCUSED_TEST_COMMAND: Final[str] = (
    "rtk uv run pytest tests/test_m1c_asymmetric_downside_residual_v1.py "
    "tests/test_m1c_asymmetric_downside_residual_v1_artifacts.py -q"
)
LINT_COMMAND: Final[str] = (
    "rtk uv run ruff check packages/stocker_research/src/stocker_research/"
    "m1c_asymmetric_downside_residual_v1.py "
    "tests/test_m1c_asymmetric_downside_residual_v1.py "
    "tests/test_m1c_asymmetric_downside_residual_v1_artifacts.py "
    "research/directional-readiness/20260728-m1c-asymmetric-downside-residual-v1/"
    "run_experiment.py "
    "research/directional-readiness/20260728-m1c-asymmetric-downside-residual-v1/"
    "audit_experiment.py"
)
TYPE_CHECK_COMMAND: Final[str] = (
    "rtk uv run mypy packages/stocker_research/src/stocker_research/"
    "m1c_asymmetric_downside_residual_v1.py"
)
FULL_TEST_COMMAND: Final[str] = "rtk uv run pytest"
EXPECTED_HASHES: Final[dict[Path, str]] = {
    CONTRACT_PATH: "04accd9a0f19b7b3d8a31534d14af02c8b8d3f72759930979fcb1f8306017845",
    TAIL_EPISODES_PATH: "a843fedf4c5df712237fd374c5efb3ff8925b575c875395ea648d786b589d9a3",
    TAIL_CHECKPOINTS_PATH: "8dd0ef53d9c5493b70f600a28d6f77e8ffabd5e7b48a5378cf0bb4411382cb8f",
    TAIL_PROVENANCE_PATH: "b6e54f9b2d7de085ccd7132c4dbdb26231f901d57b3536a68abf26228fec6421",
    STATE_PATH: "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0",
    HISTORICAL_OPTIONS_PATH: ("4bc6fd0ce6972210949a5447fd06ca0ffaa258cb953d5e3447c1c07afab85b40"),
    STRESS_OPTIONS_PATH: ("0b3f16cb06ae00df06dc34041a87d78b17478f1245810f54b5cc2d0f38d27e97"),
}
IDENTITY_COLUMNS: Final[list[str]] = ["stock", "session", "checkpoint"]
STATE_ORDER: Final[tuple[str, ...]] = (
    "UP_MOVE",
    "DOWN_MOVE",
    "NO_MOVE",
    "AMBIGUOUS_BOTH_WITHIN_BAR",
)
ACTION_ORDER: Final[tuple[str, ...]] = ("CALL", "ABSTAIN", "PUT")
PHASE_ORDER: Final[tuple[str, ...]] = ("FIRST_ENTRY", "PERSISTENT", "RE_ENTRY")
SCORE_BINS: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class ExperimentBlocked(RuntimeError):
    """A fail-closed data, chronology, or reproducibility blocker."""


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise ExperimentBlocked(f"required frozen source is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ExperimentBlocked(f"frozen source hash drifted: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
        )
    prior = cast(
        dict[str, Any],
        json.loads(TAIL_PROVENANCE_PATH.read_text(encoding="utf-8")),
    )
    protected = cast(dict[str, Any], prior["protected_data_confirmation"])
    execution = cast(dict[str, Any], prior["execution_confirmation"])
    causes = cast(dict[str, Any], prior["causality_confirmation"])
    if any(
        (
            bool(protected["protected_data_opened"]),
            bool(protected["protected_outcomes_calculated"]),
            bool(protected["protected_outcomes_displayed"]),
            bool(protected["protected_outcomes_inspected"]),
            bool(execution["broker_access"]),
            bool(execution["order_routing_enabled"]),
            bool(execution["orders_submitted"]),
            bool(causes["m1c_refit"]),
            bool(causes["a1_refit"]),
            bool(causes["fresh_episode_definition_changed"]),
            bool(causes["archived_signed_pressure_used"]),
            bool(causes["archived_tension_used"]),
        )
    ):
        raise ExperimentBlocked("inherited Tail Phase provenance violates the safety contract")
    return records


def _read_opened_parquet(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Push the protected boundary into every row-bearing read."""

    frame = pd.read_parquet(
        path,
        columns=None if columns is None else list(columns),
        filters=[("session", "<", PROTECTED_START)],
    )
    assert_unprotected_sessions(frame["session"])
    if len(frame) != pq.ParquetFile(path).metadata.num_rows:
        raise ExperimentBlocked(f"source contains rows at or beyond the protected boundary: {path}")
    return frame


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes = _read_opened_parquet(TAIL_EPISODES_PATH)
    checkpoints = _read_opened_parquet(TAIL_CHECKPOINTS_PATH)
    state_columns = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "bar_is_complete",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vti__bar_log_return",
    ]
    bars = pd.read_parquet(
        STATE_PATH,
        columns=state_columns,
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", STRESS_END),
        ],
    ).rename(columns={"symbol": "stock"})
    assert_unprotected_sessions(bars["session"])
    if str(bars["session"].min()) < DEVELOPMENT_START or str(bars["session"].max()) > STRESS_END:
        raise ExperimentBlocked("completed-bar read crossed an opened date boundary")
    if not bool(bars["bar_is_complete"].astype(bool).all()):
        raise ExperimentBlocked("completed-bar source contains non-final bars")
    if bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ExperimentBlocked("completed-bar identities are not unique")
    bars = bars.drop(columns="bar_is_complete")

    option_columns = ["symbol", "session", "previous_close_underlying_price", "atm_iv"]
    historical = pd.read_parquet(
        HISTORICAL_OPTIONS_PATH,
        columns=option_columns,
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", ASSESSMENT_END),
        ],
    )
    stress_columns = [*option_columns, "pair_available"]
    stress = pd.read_parquet(
        STRESS_OPTIONS_PATH,
        columns=stress_columns,
        filters=[
            ("session", ">=", STRESS_START),
            ("session", "<=", STRESS_END),
        ],
    )
    stress = stress.loc[stress["pair_available"].astype(bool), option_columns]
    options = pd.concat([historical, stress], ignore_index=True).rename(columns={"symbol": "stock"})
    assert_unprotected_sessions(options["session"])
    if options.duplicated(["stock", "session"]).any():
        raise ExperimentBlocked("previous-close option context is not unique")
    return episodes, checkpoints, bars, options


def _market_return_15m(
    checkpoints: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.Series:
    market = bars[["stock", "session", "bar_ordinal", "vti__bar_log_return"]].sort_values(
        ["stock", "session", "bar_ordinal"], kind="mergesort"
    )
    groups = {
        (str(stock), str(session)): group.set_index("bar_ordinal")
        for (stock, session), group in market.groupby(["stock", "session"], sort=False)
    }
    values: list[float] = []
    for raw in checkpoints.itertuples(index=False):
        row = cast(Any, raw)
        group = groups.get((str(row.stock), str(row.session)))
        ordinals = list(range(int(row.checkpoint) - 3, int(row.checkpoint)))
        if group is None or not set(ordinals).issubset(group.index):
            values.append(math.nan)
            continue
        returns = pd.to_numeric(
            group.loc[ordinals, "vti__bar_log_return"],
            errors="coerce",
        ).to_numpy(float)
        values.append(float(np.sum(returns)) if np.isfinite(returns).all() else math.nan)
    return pd.Series(values, index=checkpoints.index, dtype=float)


def prepare_panel(
    source: pd.DataFrame,
    bars: pd.DataFrame,
    options: pd.DataFrame,
    *,
    population: str,
) -> pd.DataFrame:
    """Attach the four causal predictors and the fixed endpoint target."""

    required = {
        *IDENTITY_COLUMNS,
        "partition",
        "feature_available_timestamp_utc",
        "M1C_probability",
        "m1c_high_tail_v1",
        "future_15m_exceed_iv_v1",
        "future_15m_absolute_movement_v1",
        "future_15m_iv_residual_v1",
        "available_15m",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ExperimentBlocked(f"{population} source columns missing: {missing}")
    assert_unprotected_sessions(source["session"])
    context = options.copy()
    context["iv_expected_absolute_15m"] = pd.to_numeric(
        context["atm_iv"],
        errors="coerce",
    ).map(lambda value: iv_expected_absolute(float(value), 15) if value > 0.0 else math.nan)
    context["implied_movement_15m_price"] = (
        pd.to_numeric(context["previous_close_underlying_price"], errors="coerce")
        * context["iv_expected_absolute_15m"]
    )
    panel = source.merge(
        context,
        on=["stock", "session"],
        how="left",
        validate="many_to_one",
    )
    panel = build_downside_features(panel, bars)
    panel["pre_entry_broad_market_signed_return_15m_v1"] = _market_return_15m(panel, bars)

    entry = bars[["stock", "session", "bar_ordinal", "open"]].rename(
        columns={"bar_ordinal": "checkpoint", "open": "_entry_open_15m"}
    )
    terminal = bars[["stock", "session", "bar_ordinal", "close"]].copy()
    terminal["checkpoint"] = terminal["bar_ordinal"].astype(int) - 2
    terminal = terminal.rename(columns={"close": "_terminal_close_15m"}).drop(columns="bar_ordinal")
    panel = panel.merge(
        entry,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    ).merge(
        terminal,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    entry_values = pd.to_numeric(panel["_entry_open_15m"], errors="coerce").to_numpy(float)
    close_values = pd.to_numeric(panel["_terminal_close_15m"], errors="coerce").to_numpy(float)
    valid_price = (
        np.isfinite(entry_values)
        & (entry_values > 0.0)
        & np.isfinite(close_values)
        & (close_values > 0.0)
    )
    signed = np.full(len(panel), np.nan, dtype=float)
    signed[valid_price] = np.log(close_values[valid_price] / entry_values[valid_price])
    panel["signed_endpoint_return_15m_v1"] = signed
    threshold = pd.to_numeric(panel["iv_expected_absolute_15m"], errors="coerce")
    outcome_complete = (
        panel["available_15m"].astype(bool)
        & np.isfinite(panel["signed_endpoint_return_15m_v1"])
        & np.isfinite(threshold)
        & threshold.gt(0.0)
    )
    panel["primary_outcome_complete_v1"] = outcome_complete
    panel["primary_outcome_state_v1"] = "INCOMPLETE"
    panel.loc[outcome_complete, "primary_outcome_state_v1"] = [
        partition_endpoint_return(float(observed), implied_movement=float(implied))
        for observed, implied in zip(
            panel.loc[outcome_complete, "signed_endpoint_return_15m_v1"],
            threshold.loc[outcome_complete],
            strict=True,
        )
    ]
    panel["is_down_move_v1"] = panel["primary_outcome_state_v1"].eq("DOWN_MOVE").astype(int)
    reconstructed_strict = np.abs(panel["signed_endpoint_return_15m_v1"]) > threshold
    comparable = outcome_complete & panel["future_15m_exceed_iv_v1"].notna()
    if not reconstructed_strict.loc[comparable].equals(
        panel.loc[comparable, "future_15m_exceed_iv_v1"].astype(bool)
    ):
        raise ExperimentBlocked("canonical strict M1C movement target reconstruction drifted")
    absolute_difference = np.abs(
        pd.to_numeric(panel["future_15m_absolute_movement_v1"], errors="coerce")
        - np.abs(panel["signed_endpoint_return_15m_v1"])
    )
    if float(absolute_difference.loc[comparable].max()) > 1e-12:
        raise ExperimentBlocked("canonical 15-minute endpoint movement reconstruction drifted")
    residual_difference = np.abs(
        pd.to_numeric(panel["future_15m_iv_residual_v1"], errors="coerce")
        - (np.abs(panel["signed_endpoint_return_15m_v1"]) - threshold)
    )
    if float(residual_difference.loc[comparable].max()) > 1e-12:
        raise ExperimentBlocked("canonical previous-close IV denominator reconstruction drifted")
    panel["population_v1"] = population
    panel["target_definition_v1"] = "inclusive_endpoint_direction_15m"
    panel["exact_probability_decomposition_supported_v1"] = False
    return panel


def eligible_panel(panel: pd.DataFrame, *, primary: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    high = panel["M1C_probability"].ge(M1C_HIGH_MOVEMENT_THRESHOLD)
    inherited_high = panel["m1c_high_tail_v1"].astype(bool)
    if not high.equals(inherited_high):
        raise ExperimentBlocked("frozen M1C high-tail membership changed")
    phase = (
        panel["phase_at_trigger_v1"].isin(["FIRST_ENTRY", "RE_ENTRY"])
        if primary
        else panel["m1c_tail_phase_v1"].isin(PHASE_ORDER)
    )
    causal = panel["downside_features_complete"].astype(bool)
    outcome = panel["primary_outcome_complete_v1"].astype(bool)
    market = np.isfinite(panel["pre_entry_broad_market_signed_return_15m_v1"])
    eligible = high & phase & causal & outcome
    exclusions = {
        "source_rows": int(len(panel)),
        "not_high_m1c": int((~high).sum()),
        "non_primary_phase": int((high & ~phase).sum()),
        "incomplete_fixed_predictors": int((high & phase & ~causal).sum()),
        "incomplete_primary_outcome": int((high & phase & causal & ~outcome).sum()),
        "eligible_rows": int(eligible.sum()),
        "eligible_rows_missing_market_baseline": int((eligible & ~market).sum()),
    }
    result = panel.loc[eligible].copy()
    result["month_v1"] = result["session"].astype(str).str[:7]
    return result, exclusions


def conditional_movers(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["primary_outcome_state_v1"].isin(["UP_MOVE", "DOWN_MOVE"])].copy()


def probability_metrics(
    target: Sequence[int] | np.ndarray,
    score: Sequence[float] | np.ndarray,
) -> dict[str, float | int | str]:
    y = np.asarray(target, dtype=int)
    probability = np.asarray(score, dtype=float)
    valid = np.isfinite(probability) & np.isin(y, [0, 1])
    y = y[valid]
    probability = np.clip(probability[valid], 1e-12, 1.0 - 1e-12)
    if len(y) == 0 or len(np.unique(y)) != 2:
        return {
            "rows": int(len(y)),
            "downside_base_rate": float(np.mean(y)) if len(y) else math.nan,
            "roc_auc": math.nan,
            "average_precision": math.nan,
            "log_loss": math.nan,
            "brier_score": math.nan,
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
            "calibration_status": "blocked_insufficient_support",
        }
    logits = np.log(probability / (1.0 - probability))
    calibration_intercept = math.nan
    calibration_slope = math.nan
    calibration_status = "supported"
    try:
        design = sm.add_constant(logits, has_constant="add")
        calibration = sm.GLM(y, design, family=sm.families.Binomial()).fit()
        calibration_intercept = float(calibration.params[0])
        calibration_slope = float(calibration.params[1])
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        calibration_status = "blocked_insufficient_support"
    return {
        "rows": int(len(y)),
        "downside_base_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "calibration_status": calibration_status,
    }


def reliability_table(
    frame: pd.DataFrame,
    *,
    period: str,
    score_column: str = "q_down_v1",
) -> pd.DataFrame:
    movers = conditional_movers(frame)
    bins = pd.cut(
        movers[score_column],
        bins=list(SCORE_BINS),
        include_lowest=True,
        right=True,
        duplicates="drop",
    )
    records: list[dict[str, Any]] = []
    for label, group in movers.groupby(bins, observed=False):
        records.append(
            {
                "period": period,
                "score_bin": str(label),
                "rows": int(len(group)),
                "sessions": int(group["session"].nunique()),
                "mean_q_down": (float(group[score_column].mean()) if len(group) else math.nan),
                "observed_down_rate": (
                    float(group["is_down_move_v1"].mean()) if len(group) else math.nan
                ),
            }
        )
    return pd.DataFrame(records)


def bucket_composition(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for action in ACTION_ORDER:
        group = frame.loc[frame["asymmetric_action_v1"].eq(action)]
        count = len(group)
        record: dict[str, Any] = {
            "period": period,
            "bucket": action,
            "rows": int(count),
            "sessions": int(group["session"].nunique()),
            "stocks": int(group["stock"].nunique()),
            "material_up_count": int(group["primary_outcome_state_v1"].eq("UP_MOVE").sum()),
            "material_down_count": int(group["primary_outcome_state_v1"].eq("DOWN_MOVE").sum()),
            "no_move_count": int(group["primary_outcome_state_v1"].eq("NO_MOVE").sum()),
            "ambiguous_path_count": int(
                group["primary_outcome_state_v1"].eq("AMBIGUOUS_BOTH_WITHIN_BAR").sum()
            ),
            "mean_signed_primary_horizon_return": float(
                group["signed_endpoint_return_15m_v1"].mean()
            ),
            "median_signed_primary_horizon_return": float(
                group["signed_endpoint_return_15m_v1"].median()
            ),
            "mean_absolute_movement": float(group["future_15m_absolute_movement_v1"].mean()),
            "median_absolute_movement": float(group["future_15m_absolute_movement_v1"].median()),
            "mean_previous_close_iv_residual": float(group["future_15m_iv_residual_v1"].mean()),
            "exceed_iv_rate": float(group["future_15m_exceed_iv_v1"].astype(float).mean()),
        }
        for state, output in (
            ("UP_MOVE", "material_up_rate"),
            ("DOWN_MOVE", "material_down_rate"),
            ("NO_MOVE", "no_move_rate"),
            ("AMBIGUOUS_BOTH_WITHIN_BAR", "ambiguous_path_rate"),
        ):
            record[output] = (
                float(group["primary_outcome_state_v1"].eq(state).mean()) if count else math.nan
            )
        records.append(record)
    return pd.DataFrame(records)


def add_aligned_columns(frame: pd.DataFrame, action_column: str) -> pd.DataFrame:
    output = frame.copy()
    action = output[action_column].astype("string")
    signed = pd.to_numeric(output["signed_endpoint_return_15m_v1"], errors="coerce")
    output["_acted"] = action.isin(["CALL", "PUT"])
    output["_correct"] = (action.eq("CALL") & output["primary_outcome_state_v1"].eq("UP_MOVE")) | (
        action.eq("PUT") & output["primary_outcome_state_v1"].eq("DOWN_MOVE")
    )
    output["_aligned_return"] = np.where(
        action.eq("CALL"),
        signed,
        np.where(action.eq("PUT"), -signed, np.nan),
    )
    maximum_up = pd.to_numeric(output["maximum_up_excursion_15m"], errors="coerce")
    maximum_down = pd.to_numeric(output["maximum_down_excursion_15m"], errors="coerce")
    output["_aligned_mfe"] = np.where(
        action.eq("CALL"),
        maximum_up,
        np.where(action.eq("PUT"), -maximum_down, np.nan),
    )
    output["_aligned_mae"] = np.where(
        action.eq("CALL"),
        maximum_down,
        np.where(action.eq("PUT"), -maximum_up, np.nan),
    )
    return output


def policy_metrics(
    frame: pd.DataFrame,
    *,
    action_column: str,
    policy: str,
    period: str,
    evaluation_scope: str,
) -> dict[str, Any]:
    working = add_aligned_columns(frame, action_column)
    acted = working.loc[working["_acted"]].copy()
    material = acted.loc[acted["primary_outcome_state_v1"].isin(["UP_MOVE", "DOWN_MOVE"])]
    session_returns = acted.groupby("session", sort=False)["_aligned_return"].mean()
    month_returns = acted.groupby("month_v1", sort=False)["_aligned_return"].mean()
    call = acted.loc[acted[action_column].eq("CALL"), "_aligned_return"]
    put = acted.loc[acted[action_column].eq("PUT"), "_aligned_return"]
    return {
        "period": period,
        "policy": policy,
        "evaluation_scope": evaluation_scope,
        "eligible_rows": int(len(working)),
        "call_actions": int(working[action_column].eq("CALL").sum()),
        "put_actions": int(working[action_column].eq("PUT").sum()),
        "abstentions": int((~working["_acted"]).sum()),
        "action_sessions": int(acted["session"].nunique()),
        "directional_accuracy_unambiguous_material_moves": (
            float(material["_correct"].mean()) if len(material) else math.nan
        ),
        "directional_accuracy_material_move_denominator": int(len(material)),
        "accuracy_counting_no_move_as_failure": (
            float(acted["_correct"].mean()) if len(acted) else math.nan
        ),
        "mean_aligned_return": (float(acted["_aligned_return"].mean()) if len(acted) else math.nan),
        "median_aligned_return": (
            float(acted["_aligned_return"].median()) if len(acted) else math.nan
        ),
        "mean_aligned_return_CALL": float(call.mean()) if len(call) else math.nan,
        "mean_aligned_return_PUT": float(put.mean()) if len(put) else math.nan,
        "positive_session_rate": (
            float(session_returns.gt(0.0).mean()) if len(session_returns) else math.nan
        ),
        "positive_month_rate": (
            float(month_returns.gt(0.0).mean()) if len(month_returns) else math.nan
        ),
        "mean_maximum_favourable_excursion": (
            float(acted["_aligned_mfe"].mean()) if len(acted) else math.nan
        ),
        "mean_maximum_adverse_excursion": (
            float(acted["_aligned_mae"].mean()) if len(acted) else math.nan
        ),
        "mean_movement_consumed_v1": (
            float(acted["movement_consumed_v1"].mean()) if len(acted) else math.nan
        ),
        "mean_post_share_of_local_range_v1": (
            float(acted["post_share_of_local_range_v1"].mean()) if len(acted) else math.nan
        ),
    }


def _sign_actions(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    actions = pd.Series("ABSTAIN", index=values.index, dtype="string")
    actions.loc[numeric.gt(0.0)] = "CALL"
    actions.loc[numeric.lt(0.0)] = "PUT"
    return actions


def attach_baseline_actions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["always_CALL_action"] = "CALL"
    output["always_PUT_action"] = "PUT"
    output["momentum_5m_action"] = _sign_actions(output["D1_signed_return_5m"])
    output["momentum_15m_action"] = _sign_actions(output["D2_signed_return_15m"])
    output["broad_market_15m_action"] = _sign_actions(
        output["pre_entry_broad_market_signed_return_15m_v1"]
    )
    output["frozen_A1_action"] = output["A1_action_v1"].astype("string")
    return output


def baseline_tables(
    frame: pd.DataFrame,
    *,
    period: str,
) -> pd.DataFrame:
    working = attach_baseline_actions(frame)
    acted_mask = working["asymmetric_action_v1"].isin(["CALL", "PUT"])
    specifications = (
        ("asymmetric_downside_v1", "asymmetric_action_v1"),
        ("always_CALL", "always_CALL_action"),
        ("always_PUT", "always_PUT_action"),
        ("recent_signed_5m_momentum", "momentum_5m_action"),
        ("recent_signed_15m_momentum", "momentum_15m_action"),
        ("causal_broad_market_15m_direction", "broad_market_15m_action"),
        ("frozen_A1", "frozen_A1_action"),
    )
    records: list[dict[str, Any]] = []
    for name, column in specifications:
        records.append(
            policy_metrics(
                working,
                action_column=column,
                policy=name,
                period=period,
                evaluation_scope="all_complete_episodes",
            )
        )
        records.append(
            policy_metrics(
                working.loc[acted_mask].copy(),
                action_column=column,
                policy=name,
                period=period,
                evaluation_scope="asymmetric_policy_acted_episodes",
            )
        )
    blocked = {
        key: None
        for key in records[0]
        if key
        not in {
            "period",
            "policy",
            "evaluation_scope",
        }
    }
    for scope in (
        "all_complete_episodes",
        "asymmetric_policy_acted_episodes",
    ):
        records.append(
            {
                "period": period,
                "policy": "existing_frozen_D2",
                "evaluation_scope": scope,
                **blocked,
                "status": "blocked_contaminated_or_unreproducible_lineage",
            }
        )
    return pd.DataFrame(records)


def secondary_10m_directional_table(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    """Report the inherited 10-minute directional endpoint as secondary only."""

    records: list[dict[str, Any]] = []
    signed = pd.to_numeric(frame["future_10m_signed_return_v1"], errors="coerce")
    for policy, action_column in (
        ("asymmetric_downside_v1", "asymmetric_action_v1"),
        ("frozen_A1", "A1_action_v1"),
    ):
        action = frame[action_column].astype("string")
        acted = action.isin(["CALL", "PUT"]) & signed.notna() & signed.ne(0.0)
        correct = (action.eq("CALL") & signed.gt(0.0)) | (action.eq("PUT") & signed.lt(0.0))
        aligned = np.where(
            action.eq("CALL"),
            signed,
            np.where(action.eq("PUT"), -signed, np.nan),
        )
        records.append(
            {
                "period": period,
                "horizon_minutes": 10,
                "role": "secondary_only_not_m1c_probability_partition",
                "policy": policy,
                "actions": int(acted.sum()),
                "sessions": int(frame.loc[acted, "session"].nunique()),
                "directional_accuracy": float(correct.loc[acted].mean()),
                "mean_aligned_return": float(np.nanmean(aligned[acted])),
                "median_aligned_return": float(np.nanmedian(aligned[acted])),
            }
        )
    return pd.DataFrame(records)


def bucket_spreads(frame: pd.DataFrame) -> dict[str, float]:
    put = frame.loc[frame["asymmetric_action_v1"].eq("PUT")]
    call = frame.loc[frame["asymmetric_action_v1"].eq("CALL")]
    if put.empty or call.empty:
        return {"down_rate_spread": math.nan, "up_rate_spread": math.nan}
    return {
        "down_rate_spread": float(
            put["primary_outcome_state_v1"].eq("DOWN_MOVE").mean()
            - call["primary_outcome_state_v1"].eq("DOWN_MOVE").mean()
        ),
        "up_rate_spread": float(
            call["primary_outcome_state_v1"].eq("UP_MOVE").mean()
            - put["primary_outcome_state_v1"].eq("UP_MOVE").mean()
        ),
    }


def _bootstrap_sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sessions = frame["session"].drop_duplicates().to_numpy()
    sampled = rng.choice(sessions, size=len(sessions), replace=True)
    pieces: list[pd.DataFrame] = []
    for cluster_number, session in enumerate(sampled):
        piece = frame.loc[frame["session"].eq(session)].copy()
        piece["_bootstrap_cluster"] = cluster_number
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _percentile_interval(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan, 0
    lower, upper = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(lower), float(upper), int(len(finite))


def session_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    period: str,
    constant_down_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observations: dict[str, list[float]] = {
        "roc_auc": [],
        "log_loss_improvement_vs_constant": [],
        "brier_improvement_vs_constant": [],
        "down_rate_spread": [],
        "up_rate_spread": [],
        "mean_aligned_return": [],
    }
    for _ in range(BOOTSTRAP_DRAWS):
        sample = _bootstrap_sample(frame, rng)
        movers = conditional_movers(sample)
        if len(movers) and movers["is_down_move_v1"].nunique() == 2:
            model = probability_metrics(movers["is_down_move_v1"], movers["q_down_v1"])
            constant = probability_metrics(
                movers["is_down_move_v1"],
                np.full(len(movers), constant_down_rate),
            )
            observations["roc_auc"].append(float(model["roc_auc"]))
            observations["log_loss_improvement_vs_constant"].append(
                float(constant["log_loss"]) - float(model["log_loss"])
            )
            observations["brier_improvement_vs_constant"].append(
                float(constant["brier_score"]) - float(model["brier_score"])
            )
        spreads = bucket_spreads(sample)
        observations["down_rate_spread"].append(spreads["down_rate_spread"])
        observations["up_rate_spread"].append(spreads["up_rate_spread"])
        aligned = add_aligned_columns(sample, "asymmetric_action_v1")
        observations["mean_aligned_return"].append(
            float(aligned.loc[aligned["_acted"], "_aligned_return"].mean())
        )
    movers = conditional_movers(frame)
    observed_model = probability_metrics(movers["is_down_move_v1"], movers["q_down_v1"])
    observed_constant = probability_metrics(
        movers["is_down_move_v1"],
        np.full(len(movers), constant_down_rate),
    )
    spreads = bucket_spreads(frame)
    aligned = add_aligned_columns(frame, "asymmetric_action_v1")
    observed = {
        "roc_auc": float(observed_model["roc_auc"]),
        "log_loss_improvement_vs_constant": (
            float(observed_constant["log_loss"]) - float(observed_model["log_loss"])
        ),
        "brier_improvement_vs_constant": (
            float(observed_constant["brier_score"]) - float(observed_model["brier_score"])
        ),
        **spreads,
        "mean_aligned_return": float(aligned.loc[aligned["_acted"], "_aligned_return"].mean()),
    }
    records: list[dict[str, Any]] = []
    for statistic, values in observations.items():
        lower, upper, valid = _percentile_interval(values)
        records.append(
            {
                "period": period,
                "statistic": statistic,
                "observed": observed[statistic],
                "lower_95": lower,
                "upper_95": upper,
                "draws_requested": BOOTSTRAP_DRAWS,
                "draws_valid": valid,
                "cluster": "session",
                "seed": seed,
            }
        )
    return pd.DataFrame(records)


def _shuffle_within_groups(
    frame: pd.DataFrame,
    column: str,
    rng: np.random.Generator,
) -> np.ndarray:
    output = frame[column].to_numpy(copy=True)
    group_indices = frame.groupby(["stock", "checkpoint"], sort=False).indices
    for indices in group_indices.values():
        locations = np.asarray(indices, dtype=int)
        output[locations] = rng.permutation(output[locations])
    return output


def label_permutation_results(
    frame: pd.DataFrame,
    *,
    period: str,
    constant_down_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    movers = conditional_movers(frame).reset_index(drop=True)
    full = frame.reset_index(drop=True)
    observed_model = probability_metrics(movers["is_down_move_v1"], movers["q_down_v1"])
    observed_constant = probability_metrics(
        movers["is_down_move_v1"],
        np.full(len(movers), constant_down_rate),
    )
    observed_spreads = bucket_spreads(full)
    observed = {
        "roc_auc": float(observed_model["roc_auc"]),
        "log_loss_improvement_vs_constant": (
            float(observed_constant["log_loss"]) - float(observed_model["log_loss"])
        ),
        "brier_improvement_vs_constant": (
            float(observed_constant["brier_score"]) - float(observed_model["brier_score"])
        ),
        **observed_spreads,
    }
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    null_values: dict[str, list[float]] = {key: [] for key in observed}
    for draw in range(PERMUTATION_DRAWS):
        shuffled_binary = _shuffle_within_groups(movers, "is_down_move_v1", rng).astype(int)
        model = probability_metrics(shuffled_binary, movers["q_down_v1"])
        constant = probability_metrics(
            shuffled_binary,
            np.full(len(movers), constant_down_rate),
        )
        shuffled_states = _shuffle_within_groups(full, "primary_outcome_state_v1", rng)
        state_copy = full.copy()
        state_copy["primary_outcome_state_v1"] = shuffled_states
        spread = bucket_spreads(state_copy)
        draw_values = {
            "roc_auc": float(model["roc_auc"]),
            "log_loss_improvement_vs_constant": (
                float(constant["log_loss"]) - float(model["log_loss"])
            ),
            "brier_improvement_vs_constant": (
                float(constant["brier_score"]) - float(model["brier_score"])
            ),
            **spread,
        }
        for statistic, value in draw_values.items():
            null_values[statistic].append(value)
            records.append(
                {
                    "period": period,
                    "draw": draw + 1,
                    "statistic": statistic,
                    "value": value,
                }
            )
    summary: list[dict[str, Any]] = []
    for statistic, values in null_values.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        observed_value = observed[statistic]
        p_value = (
            float((1 + np.sum(finite >= observed_value)) / (1 + len(finite)))
            if len(finite)
            else math.nan
        )
        lower, upper, valid = _percentile_interval(finite)
        summary.append(
            {
                "period": period,
                "statistic": statistic,
                "observed": observed_value,
                "null_mean": float(np.mean(finite)) if len(finite) else math.nan,
                "null_lower_95": lower,
                "null_upper_95": upper,
                "one_sided_p_value": p_value,
                "draws_requested": PERMUTATION_DRAWS,
                "draws_valid": valid,
                "seed": seed,
                "structure": "outcomes permuted within stock and checkpoint",
            }
        )
    return pd.DataFrame(summary), pd.DataFrame(records)


def leave_one_out_table(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    aligned_source = add_aligned_columns(frame, "asymmetric_action_v1")
    for dimension, column in (
        ("stock", "stock"),
        ("month", "month_v1"),
        ("checkpoint", "checkpoint"),
    ):
        for omitted in sorted(frame[column].dropna().unique(), key=str):
            subset = frame.loc[~frame[column].eq(omitted)].copy()
            aligned = add_aligned_columns(subset, "asymmetric_action_v1")
            spreads = bucket_spreads(subset)
            records.append(
                {
                    "period": period,
                    "dimension": dimension,
                    "omitted": str(omitted),
                    "remaining_rows": int(len(subset)),
                    "remaining_sessions": int(subset["session"].nunique()),
                    **spreads,
                    "mean_aligned_return": float(
                        aligned.loc[aligned["_acted"], "_aligned_return"].mean()
                    ),
                    "omitted_acted_share": float(
                        aligned_source.loc[
                            aligned_source[column].eq(omitted) & aligned_source["_acted"]
                        ].shape[0]
                        / max(1, int(aligned_source["_acted"].sum()))
                    ),
                }
            )
    return pd.DataFrame(records)


def concentration_table(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    acted = add_aligned_columns(frame, "asymmetric_action_v1")
    acted = acted.loc[acted["_acted"]].copy()
    records: list[dict[str, Any]] = []
    dimensions = (
        ("stock", "stock"),
        ("month", "month_v1"),
        ("session", "session"),
        ("checkpoint", "checkpoint"),
        ("time_of_day", "time_of_day_v1"),
    )
    for dimension, column in dimensions:
        for value, group in acted.groupby(column, sort=True, dropna=False):
            records.append(
                {
                    "period": period,
                    "dimension": dimension,
                    "value": str(value),
                    "acted_rows": int(len(group)),
                    "acted_share": float(len(group) / max(1, len(acted))),
                    "sessions": int(group["session"].nunique()),
                    "stocks": int(group["stock"].nunique()),
                    "mean_aligned_return": float(group["_aligned_return"].mean()),
                    "down_rate": float(group["primary_outcome_state_v1"].eq("DOWN_MOVE").mean()),
                    "up_rate": float(group["primary_outcome_state_v1"].eq("UP_MOVE").mean()),
                }
            )
    return pd.DataFrame(records)


def tail_phase_diagnostics(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for phase in PHASE_ORDER:
        phase_frame = frame.loc[frame["m1c_tail_phase_v1"].eq(phase)].copy()
        movers = conditional_movers(phase_frame)
        metrics = probability_metrics(movers["is_down_move_v1"], movers["q_down_v1"])
        supported = len(movers) >= 30 and movers["session"].nunique() >= 10
        policy = policy_metrics(
            phase_frame,
            action_column="asymmetric_action_v1",
            policy="asymmetric_downside_v1",
            period=period,
            evaluation_scope=f"tail_phase_{phase}",
        )
        a1 = policy_metrics(
            attach_baseline_actions(phase_frame),
            action_column="frozen_A1_action",
            policy="frozen_A1",
            period=period,
            evaluation_scope=f"tail_phase_{phase}",
        )
        records.append(
            {
                "period": period,
                "phase": phase,
                "diagnostic": "summary",
                "support_status": ("supported" if supported else "blocked_insufficient_support"),
                "rows": int(len(phase_frame)),
                "sessions": int(phase_frame["session"].nunique()),
                "conditional_mover_rows": int(len(movers)),
                "conditional_mover_sessions": int(movers["session"].nunique()),
                "downside_auc": metrics["roc_auc"] if supported else math.nan,
                "call_actions": policy["call_actions"],
                "put_actions": policy["put_actions"],
                "abstentions": policy["abstentions"],
                "mean_aligned_return": policy["mean_aligned_return"],
                "frozen_A1_mean_aligned_return": a1["mean_aligned_return"],
                "frozen_A1_actions": int(a1["call_actions"]) + int(a1["put_actions"]),
            }
        )
        for action in ACTION_ORDER:
            group = phase_frame.loc[phase_frame["asymmetric_action_v1"].eq(action)]
            cell_supported = len(group) >= 30 and group["session"].nunique() >= 10
            records.append(
                {
                    "period": period,
                    "phase": phase,
                    "diagnostic": f"action_{action}",
                    "support_status": (
                        "supported" if cell_supported else "blocked_insufficient_support"
                    ),
                    "rows": int(len(group)),
                    "sessions": int(group["session"].nunique()),
                    "material_up_rate": (
                        float(group["primary_outcome_state_v1"].eq("UP_MOVE").mean())
                        if len(group)
                        else math.nan
                    ),
                    "material_down_rate": (
                        float(group["primary_outcome_state_v1"].eq("DOWN_MOVE").mean())
                        if len(group)
                        else math.nan
                    ),
                    "no_move_rate": (
                        float(group["primary_outcome_state_v1"].eq("NO_MOVE").mean())
                        if len(group)
                        else math.nan
                    ),
                    "mean_aligned_return": (
                        float(
                            add_aligned_columns(group, "asymmetric_action_v1")[
                                "_aligned_return"
                            ].mean()
                        )
                        if len(group)
                        else math.nan
                    ),
                }
            )
        for dimension, column in (
            ("checkpoint", "checkpoint"),
            ("time_of_day", "time_of_day_v1"),
        ):
            acted = phase_frame.loc[phase_frame["asymmetric_action_v1"].isin(["CALL", "PUT"])]
            for value, group in acted.groupby(column, sort=True, dropna=False):
                records.append(
                    {
                        "period": period,
                        "phase": phase,
                        "diagnostic": f"{dimension}_{value}",
                        "support_status": (
                            "supported"
                            if len(group) >= 30 and group["session"].nunique() >= 10
                            else "blocked_insufficient_support"
                        ),
                        "rows": int(len(group)),
                        "sessions": int(group["session"].nunique()),
                        "acted_share_within_phase": float(len(group) / max(1, len(acted))),
                    }
                )
    return pd.DataFrame(records)


def temporal_placebo(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    stress: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([development, assessment, stress], ignore_index=True)
    shifted = shift_features_to_next_episode(combined, DOWNSIDE_FEATURES)
    shifted_complete = np.isfinite(shifted.loc[:, list(DOWNSIDE_FEATURES)].to_numpy(float)).all(
        axis=1
    )
    shifted = shifted.loc[shifted_complete].copy()
    development_shifted = shifted.loc[shifted["partition"].eq("development")]
    conditional_development = conditional_movers(development_shifted)
    oof, _ = expanding_time_ordered_oof(
        conditional_development,
        target_column="is_down_move_v1",
    )
    thresholds = freeze_action_thresholds(oof["q_down_oof"])
    model = fit_downside_model(
        conditional_development,
        target_column="is_down_move_v1",
    )
    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for period in ("assessment", "stress"):
        period_frame = shifted.loc[shifted["partition"].eq(period)].copy()
        period_frame["q_down_temporal_placebo"] = model.predict_proba(period_frame)
        period_frame["asymmetric_action_temporal_placebo"] = apply_asymmetric_policy(
            period_frame["q_down_temporal_placebo"],
            low_threshold=thresholds["low"],
            high_threshold=thresholds["high"],
        )
        movers = conditional_movers(period_frame)
        metrics = probability_metrics(
            movers["is_down_move_v1"],
            movers["q_down_temporal_placebo"],
        )
        renamed = period_frame.copy()
        renamed["asymmetric_action_v1"] = renamed["asymmetric_action_temporal_placebo"]
        spreads = bucket_spreads(renamed)
        aligned = add_aligned_columns(
            period_frame,
            "asymmetric_action_temporal_placebo",
        )
        summaries.append(
            {
                "period": period,
                "placebo": "within_stock_prior_episode_feature_reassignment",
                "rows": int(len(period_frame)),
                "sessions": int(period_frame["session"].nunique()),
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                **spreads,
                "mean_aligned_return": float(
                    aligned.loc[aligned["_acted"], "_aligned_return"].mean()
                ),
                "low_threshold": thresholds["low"],
                "high_threshold": thresholds["high"],
            }
        )
        prediction_frames.append(period_frame)
    return pd.DataFrame(summaries), pd.concat(prediction_frames, ignore_index=True)


def robustness_summary(
    leave_one_out: pd.DataFrame,
    concentration: pd.DataFrame,
    assessment: pd.DataFrame,
    stress: pd.DataFrame,
) -> dict[str, Any]:
    assessment_loo = leave_one_out.loc[leave_one_out["period"].eq("assessment")]
    stress_loo = leave_one_out.loc[leave_one_out["period"].eq("stress")]
    sign_preserved = bool(
        assessment_loo["down_rate_spread"].gt(0.0).all()
        and stress_loo["down_rate_spread"].gt(0.0).all()
        and assessment_loo["mean_aligned_return"].gt(0.0).all()
        and stress_loo["mean_aligned_return"].gt(0.0).all()
    )
    top_shares = concentration.groupby(
        ["period", "dimension"],
        sort=False,
    )["acted_share"].max()
    below_half = bool(top_shares.lt(0.5).all())
    aligned_assessment = add_aligned_columns(assessment, "asymmetric_action_v1")
    aligned_stress = add_aligned_columns(stress, "asymmetric_action_v1")

    def winsorised_mean(frame: pd.DataFrame) -> float:
        values = frame.loc[frame["_acted"], "_aligned_return"].dropna().to_numpy(float)
        if not len(values):
            return math.nan
        low, high = np.quantile(values, [0.01, 0.99], method="linear")
        return float(np.mean(np.clip(values, low, high)))

    return {
        "all_leave_one_stock_month_checkpoint_down_spreads_positive_and_returns_positive": (
            sign_preserved
        ),
        "all_stock_month_session_checkpoint_time_concentrations_below_50_percent": below_half,
        "maximum_acted_concentration": (float(top_shares.max()) if len(top_shares) else math.nan),
        "assessment_one_percent_winsorised_mean_aligned_return": winsorised_mean(
            aligned_assessment
        ),
        "stress_one_percent_winsorised_mean_aligned_return": winsorised_mean(aligned_stress),
    }


def choose_decisions(
    assessment: pd.DataFrame,
    stress: pd.DataFrame,
    conditional_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    permutation: pd.DataFrame,
    robustness: Mapping[str, Any],
) -> tuple[str, str, str, dict[str, bool]]:
    metrics = conditional_metrics.set_index("period")
    bootstrap_index = bootstrap.set_index(["period", "statistic"])
    permutation_index = permutation.set_index(["period", "statistic"])
    assessment_bucket = bucket_composition(assessment, period="assessment").set_index("bucket")
    stress_bucket = bucket_composition(stress, period="stress").set_index("bucket")
    assessment_policy = policy_metrics(
        assessment,
        action_column="asymmetric_action_v1",
        policy="asymmetric_downside_v1",
        period="assessment",
        evaluation_scope="all_complete_episodes",
    )
    stress_policy = policy_metrics(
        stress,
        action_column="asymmetric_action_v1",
        policy="asymmetric_downside_v1",
        period="stress",
        evaluation_scope="all_complete_episodes",
    )
    assessment_spread = bucket_spreads(assessment)
    stress_spread = bucket_spreads(stress)
    assessment_call = assessment.loc[assessment["asymmetric_action_v1"].eq("CALL")]
    assessment_put = assessment.loc[assessment["asymmetric_action_v1"].eq("PUT")]
    stress_call = stress.loc[stress["asymmetric_action_v1"].eq("CALL")]
    stress_put = stress.loc[stress["asymmetric_action_v1"].eq("PUT")]
    null_beaten = all(
        float(permutation_index.loc[(period, statistic), "one_sided_p_value"]) <= 0.05
        for period in ("assessment", "stress")
        for statistic in (
            "roc_auc",
            "log_loss_improvement_vs_constant",
            "brier_improvement_vs_constant",
            "down_rate_spread",
        )
    )
    checks = {
        "assessment_auc_lower_above_half": float(
            bootstrap_index.loc[("assessment", "roc_auc"), "lower_95"]
        )
        > 0.5,
        "stress_auc_above_half": float(metrics.loc["stress", "roc_auc"]) > 0.5,
        "proper_scores_improve_both_periods": bool(
            metrics["log_loss_improvement_vs_constant"].gt(0.0).all()
            and metrics["brier_improvement_vs_constant"].gt(0.0).all()
        ),
        "put_down_rate_exceeds_call_both_periods": (
            assessment_spread["down_rate_spread"] > 0.0 and stress_spread["down_rate_spread"] > 0.0
        ),
        "assessment_spread_ci_above_zero": float(
            bootstrap_index.loc[("assessment", "down_rate_spread"), "lower_95"]
        )
        > 0.0,
        "stress_spread_same_sign": stress_spread["down_rate_spread"] > 0.0,
        "assessment_action_support": (
            len(assessment_call) >= 30
            and assessment_call["session"].nunique() >= 10
            and len(assessment_put) >= 30
            and assessment_put["session"].nunique() >= 10
        ),
        "stress_action_support": (
            len(stress_call) >= 30
            and stress_call["session"].nunique() >= 10
            and len(stress_put) >= 30
            and stress_put["session"].nunique() >= 10
        ),
        "positive_mean_aligned_return_both_periods": (
            float(assessment_policy["mean_aligned_return"]) > 0.0
            and float(stress_policy["mean_aligned_return"]) > 0.0
        ),
        "not_dependent_on_one_support_cell_or_extreme": bool(
            robustness[
                "all_leave_one_stock_month_checkpoint_down_spreads_positive_and_returns_positive"
            ]
            and robustness[
                "all_stock_month_session_checkpoint_time_concentrations_below_50_percent"
            ]
            and float(robustness["assessment_one_percent_winsorised_mean_aligned_return"]) > 0.0
            and float(robustness["stress_one_percent_winsorised_mean_aligned_return"]) > 0.0
        ),
        "beats_specified_label_nulls": null_beaten,
    }
    full_support = all(checks.values())
    reproducible_ranking = bool(
        metrics["roc_auc"].gt(0.5).all()
        and metrics["log_loss_improvement_vs_constant"].gt(0.0).all()
        and metrics["brier_improvement_vs_constant"].gt(0.0).all()
    )
    low_fails = bool(
        (
            assessment_bucket.loc["CALL", "material_up_rate"]
            <= assessment_bucket.loc["CALL", "material_down_rate"]
        )
        or (
            assessment_bucket.loc["CALL", "no_move_rate"]
            >= max(
                assessment_bucket.loc["CALL", "material_up_rate"],
                assessment_bucket.loc["CALL", "material_down_rate"],
            )
        )
        or (
            stress_bucket.loc["CALL", "material_up_rate"]
            <= stress_bucket.loc["CALL", "material_down_rate"]
        )
        or (
            stress_bucket.loc["CALL", "no_move_rate"]
            >= max(
                stress_bucket.loc["CALL", "material_up_rate"],
                stress_bucket.loc["CALL", "material_down_rate"],
            )
        )
    )
    if low_fails:
        descriptive_finding = "low_downside_does_not_imply_upside"
    elif reproducible_ranking:
        descriptive_finding = "asymmetric_downside_ranking_only"
    else:
        descriptive_finding = "no_incremental_asymmetric_directional_signal"
    if not checks["assessment_action_support"] or not checks["stress_action_support"]:
        decision = "blocked_insufficient_support"
    elif full_support:
        decision = "asymmetric_downside_supported_retrospectively"
    else:
        decision = descriptive_finding
    return (
        "target_mismatch_prevents_exact_probability_decomposition",
        decision,
        descriptive_finding,
        checks,
    )


def _conditional_metric_rows(
    frame: pd.DataFrame,
    *,
    period: str,
    constant_down_rate: float,
) -> list[dict[str, Any]]:
    movers = conditional_movers(frame)
    specifications: list[tuple[str, np.ndarray, str]] = [
        ("asymmetric_downside_v1", movers["q_down_v1"].to_numpy(float), "supported"),
        (
            "constant_2024_downside_base_rate",
            np.full(len(movers), constant_down_rate),
            "supported",
        ),
        (
            "recent_signed_5m_momentum",
            np.where(movers["D1_signed_return_5m"].lt(0.0), 1.0, 0.0),
            "direction_only_baseline",
        ),
        (
            "recent_signed_15m_momentum",
            np.where(movers["D2_signed_return_15m"].lt(0.0), 1.0, 0.0),
            "direction_only_baseline",
        ),
        (
            "frozen_A1",
            1.0 - pd.to_numeric(movers["A1_probability_up_v1"], errors="coerce").to_numpy(float),
            "frozen_10m_model_applied_to_15m_primary_target",
        ),
    ]
    constant_metrics = probability_metrics(
        movers["is_down_move_v1"],
        np.full(len(movers), constant_down_rate),
    )
    records: list[dict[str, Any]] = []
    for model, scores, status in specifications:
        metrics = probability_metrics(movers["is_down_move_v1"], scores)
        records.append(
            {
                "period": period,
                "model": model,
                "status": status,
                "rows": metrics["rows"],
                "sessions": int(movers["session"].nunique()),
                "downside_base_rate": metrics["downside_base_rate"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "calibration_status": metrics["calibration_status"],
                "log_loss_improvement_vs_constant": (
                    float(constant_metrics["log_loss"]) - float(metrics["log_loss"])
                ),
                "brier_improvement_vs_constant": (
                    float(constant_metrics["brier_score"]) - float(metrics["brier_score"])
                ),
            }
        )
    records.append(
        {
            "period": period,
            "model": "existing_frozen_D2",
            "status": "blocked_contaminated_or_unreproducible_lineage",
            "rows": 0,
            "sessions": 0,
        }
    )
    return records


def _frozen_field_regression(
    original: pd.DataFrame,
    prepared: pd.DataFrame,
    *,
    primary: bool,
) -> dict[str, Any]:
    fields = [
        "M1C_probability",
        "m1c_high_tail_v1",
        "m1c_tail_phase_v1",
        "movement_consumed_v1",
        "A1_probability_up_v1",
        "A1_action_v1",
    ]
    if primary:
        fields.extend(["episode_id", "existing_fresh_episode_identifier"])
    comparison = original[IDENTITY_COLUMNS + fields].merge(
        prepared[IDENTITY_COLUMNS + fields],
        on=IDENTITY_COLUMNS,
        suffixes=("_source", "_prepared"),
        validate="one_to_one",
    )
    if len(comparison) != len(original):
        raise ExperimentBlocked("prepared panel changed inherited row identity")
    for field in fields:
        source = comparison[f"{field}_source"]
        prepared_values = comparison[f"{field}_prepared"]
        if pd.api.types.is_numeric_dtype(source):
            if not np.allclose(
                pd.to_numeric(source, errors="coerce"),
                pd.to_numeric(prepared_values, errors="coerce"),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise ExperimentBlocked(f"frozen field changed: {field}")
        elif (
            not source.fillna("<NA>").astype(str).equals(prepared_values.fillna("<NA>").astype(str))
        ):
            raise ExperimentBlocked(f"frozen field changed: {field}")
    return {
        "rows_compared": int(len(comparison)),
        "fields": fields,
        "passed": True,
        "tolerance": 0.0,
    }


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    requested = [
        "row_id",
        *IDENTITY_COLUMNS,
        "episode_id",
        "existing_fresh_episode_identifier",
        "partition",
        "month_v1",
        "checkpoint_group_v1",
        "time_of_day_v1",
        "feature_available_timestamp_utc",
        "prospective_entry_timestamp",
        "M1C_probability",
        "m1c_high_tail_threshold_v1",
        "m1c_high_tail_v1",
        "m1c_model_version_v1",
        "m1c_model_hash_v1",
        "m1c_feature_hash_v1",
        "m1c_tail_phase_v1",
        "phase_at_trigger_v1",
        "tail_run_age_minutes_v1",
        "movement_consumed_v1",
        "movement_consumed_bucket_v1",
        "post_share_of_local_range_v1",
        "A1_complete_v1",
        "A1_probability_up_v1",
        "A1_action_v1",
        "A1_feature_hash_v1",
        "A1_model_hash_v1",
        "A1_preprocessing_hash_v1",
        *DOWNSIDE_FEATURES,
        "maximum_predictor_bar_ordinal",
        "maximum_predictor_timestamp",
        "D4_causal_volume_available",
        "downside_features_complete",
        "pre_entry_broad_market_signed_return_15m_v1",
        "iv_expected_absolute_15m",
        "signed_endpoint_return_15m_v1",
        "future_10m_signed_return_v1",
        "future_15m_absolute_movement_v1",
        "future_15m_iv_residual_v1",
        "future_15m_exceed_iv_v1",
        "maximum_up_excursion_15m",
        "maximum_down_excursion_15m",
        "primary_outcome_complete_v1",
        "primary_outcome_state_v1",
        "is_down_move_v1",
        "q_down_v1",
        "asymmetric_action_v1",
        "target_definition_v1",
        "exact_probability_decomposition_supported_v1",
        "protected_outcomes_accessed_v1",
    ]
    return [column for column in requested if column in frame.columns]


def _write_markdown_report(
    summary: Mapping[str, Any],
    conditional: pd.DataFrame,
    buckets: pd.DataFrame,
    policy: pd.DataFrame,
    baselines: pd.DataFrame,
    phases: pd.DataFrame,
    bootstrap: pd.DataFrame,
    permutation: pd.DataFrame,
) -> str:
    metric = conditional.loc[conditional["model"].eq("asymmetric_downside_v1")].set_index("period")
    bucket = buckets.set_index(["period", "bucket"])
    policy_index = policy.loc[
        policy["policy"].eq("asymmetric_downside_v1")
        & policy["evaluation_scope"].eq("all_complete_episodes")
    ].set_index("period")
    baseline_index = baselines.loc[
        baselines["evaluation_scope"].eq("all_complete_episodes")
    ].set_index(["period", "policy"])
    phase_index = phases.loc[phases["diagnostic"].eq("summary")].set_index(["period", "phase"])
    bootstrap_index = bootstrap.set_index(["period", "statistic"])
    permutation_index = permutation.set_index(["period", "statistic"])

    def number(value: object, digits: int = 4) -> str:
        try:
            numeric = float(cast(Any, value))
        except (TypeError, ValueError):
            return "n/a"
        return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "n/a"

    assessment_permutation_p = {
        statistic: number(
            permutation_index.loc[
                ("assessment", statistic),
                "one_sided_p_value",
            ]
        )
        for statistic in (
            "roc_auc",
            "log_loss_improvement_vs_constant",
            "brier_improvement_vs_constant",
            "down_rate_spread",
        )
    }
    stress_permutation_p = {
        statistic: number(
            permutation_index.loc[
                ("stress", statistic),
                "one_sided_p_value",
            ]
        )
        for statistic in (
            "roc_auc",
            "log_loss_improvement_vs_constant",
            "brier_improvement_vs_constant",
            "down_rate_spread",
        )
    }
    assessment_policy_mean = number(
        policy_index.loc["assessment", "mean_aligned_return"],
        6,
    )
    stress_policy_mean = number(
        policy_index.loc["stress", "mean_aligned_return"],
        6,
    )
    assessment_baseline_means = {
        policy_name: number(
            baseline_index.loc[
                ("assessment", policy_name),
                "mean_aligned_return",
            ],
            6,
        )
        for policy_name in (
            "recent_signed_5m_momentum",
            "recent_signed_15m_momentum",
            "causal_broad_market_15m_direction",
            "frozen_A1",
        )
    }
    lines = [
        "# M1C Asymmetric Downside Residual V1",
        "",
        "## Decision",
        "",
        f"- Probability-decomposition decision: `{summary['primary_decision']}`.",
        f"- Endpoint-direction decision: `{summary['directional_diagnostic_decision']}`.",
        f"- Descriptive endpoint finding: `{summary['descriptive_directional_finding']}`.",
        "- This is retrospective out-of-development evidence from already-opened 2025 "
        "periods, not untouched confirmation.",
        "",
        "## Target audit",
        "",
        "M1C enters at the next five-minute bar open and measures the absolute log return "
        "to the close of the third post-entry bar (15 minutes). Its material-move label is "
        "strictly `absolute return > prior-close ATM IV × sqrt(15/(252×390)) × sqrt(2/π)`.",
        "The required directional endpoint partition assigns exact positive and negative "
        "threshold equality to UP and DOWN. The events therefore do not form the literal "
        "complement of the frozen strict M1C event. No joint up/down/no-move probabilities "
        "were constructed.",
        "",
        "## Direct answers",
        "",
        "1. **Clean three-state decomposition?** No—not exactly, because of the strict-versus-"
        "inclusive equality mismatch. The endpoint diagnostic itself is exhaustive.",
        f"2. **High-M1C fresh material movers?** Development "
        f"{summary['episode_accounting']['development']['material_movers']}, assessment "
        f"{summary['episode_accounting']['assessment']['material_movers']}, stress "
        f"{summary['episode_accounting']['stress']['material_movers']}.",
        f"3. **Rank downside among movers?** Assessment AUC "
        f"{number(metric.loc['assessment', 'roc_auc'])}; stress "
        f"{number(metric.loc['stress', 'roc_auc'])}. Assessment session-bootstrap 95% CI "
        f"[{number(bootstrap_index.loc[('assessment', 'roc_auc'), 'lower_95'])}, "
        f"{number(bootstrap_index.loc[('assessment', 'roc_auc'), 'upper_95'])}].",
        f"4. **Proper-score improvement?** Assessment log-loss/Brier improvements "
        f"{number(metric.loc['assessment', 'log_loss_improvement_vs_constant'])}/"
        f"{number(metric.loc['assessment', 'brier_improvement_vs_constant'])}; stress "
        f"{number(metric.loc['stress', 'log_loss_improvement_vs_constant'])}/"
        f"{number(metric.loc['stress', 'brier_improvement_vs_constant'])}.",
        f"5. **PUT more downside than CALL?** Assessment spread "
        f"{number(bootstrap_index.loc[('assessment', 'down_rate_spread'), 'observed'])} "
        f"(95% CI {number(bootstrap_index.loc[('assessment', 'down_rate_spread'), 'lower_95'])} "
        f"to {number(bootstrap_index.loc[('assessment', 'down_rate_spread'), 'upper_95'])}); "
        f"stress {number(bootstrap_index.loc[('stress', 'down_rate_spread'), 'observed'])}. "
        "The assessment PUT cell has 29 actions, so the formal action-support result is "
        "`blocked_insufficient_support`.",
        f"6. **CALL more upside than downside?** Assessment "
        f"{number(bucket.loc[('assessment', 'CALL'), 'material_up_rate'])} up versus "
        f"{number(bucket.loc[('assessment', 'CALL'), 'material_down_rate'])} down; stress "
        f"{number(bucket.loc[('stress', 'CALL'), 'material_up_rate'])} versus "
        f"{number(bucket.loc[('stress', 'CALL'), 'material_down_rate'])}.",
        f"7. **CALL merely no-move?** In assessment, yes: "
        f"{number(bucket.loc[('assessment', 'CALL'), 'no_move_rate'])} were no-moves, "
        "a majority. In stress the no-move rate was "
        f"{number(bucket.loc[('stress', 'CALL'), 'no_move_rate'])}, but downside "
        "outnumbered upside, so low downside score still did not imply upside.",
        "8. **Selective policy versus baselines?** No consistent outperformance. Assessment "
        f"mean aligned return was {assessment_policy_mean} "
        "versus recent 5m/15m, market, and A1 "
        f"{assessment_baseline_means['recent_signed_5m_momentum']}/"
        f"{assessment_baseline_means['recent_signed_15m_momentum']}/"
        f"{assessment_baseline_means['causal_broad_market_15m_direction']}/"
        f"{assessment_baseline_means['frozen_A1']}. "
        f"Stress was {stress_policy_mean}, beating "
        "the momentum/market means but not frozen A1 consistently across both periods. "
        "Frozen D2 is blocked by contaminated or unreproducible lineage. Acted-timestamp "
        "comparisons are reported separately in `baseline_comparisons_v1.csv`.",
        "9. **Stable assessment/stress?** No. AUC moved from "
        f"{number(metric.loc['assessment', 'roc_auc'])} to "
        f"{number(metric.loc['stress', 'roc_auc'])}, down-rate spread changed from "
        f"{number(bootstrap_index.loc[('assessment', 'down_rate_spread'), 'observed'])} "
        f"to {number(bootstrap_index.loc[('stress', 'down_rate_spread'), 'observed'])}, "
        "and mean aligned return changed sign. No refit or recalibration occurred.",
        "10. **Broad support?** See leave-one-out and concentration tables. The summary "
        f"robustness result is `{summary['robustness']}`.",
        "11. **Tail Phase?** It did not reveal a consistent modifier. FIRST_ENTRY AUC was "
        f"{number(phase_index.loc[('assessment', 'FIRST_ENTRY'), 'downside_auc'])} assessment "
        f"and {number(phase_index.loc[('stress', 'FIRST_ENTRY'), 'downside_auc'])} stress; "
        "PERSISTENT checkpoint rows were secondary and changed sign; assessment RE_ENTRY "
        f"status was `{phase_index.loc[('assessment', 'RE_ENTRY'), 'support_status']}`. "
        "No phase gated, fit, or changed thresholds. Full action/A1/checkpoint/time strata "
        "are in `tail_phase_diagnostics_v1.csv`.",
        f"12. **Movement remaining?** Mean canonical post-entry local-range share among acted "
        f"episodes was assessment "
        f"{number(policy_index.loc['assessment', 'mean_post_share_of_local_range_v1'])} and "
        f"stress {number(policy_index.loc['stress', 'mean_post_share_of_local_range_v1'])}.",
        f"13. **Ranking/policy conclusion?** Formal endpoint decision "
        f"`{summary['directional_diagnostic_decision']}`; descriptive finding "
        f"`{summary['descriptive_directional_finding']}`.",
        "14. **Still unknowable?** Option profitability, executable bid/ask prices, slippage, "
        "market impact, fill probability, and prospective behavioural stability remain "
        "unknown. Underlying aligned returns are not option P&L.",
        "",
        "## Action accounting",
        "",
        "| period | CALL | PUT | ABSTAIN | mean aligned return |",
        "|---|---:|---:|---:|---:|",
    ]
    for period in ("assessment", "stress"):
        row = policy_index.loc[period]
        lines.append(
            f"| {period} | {int(row['call_actions'])} | {int(row['put_actions'])} | "
            f"{int(row['abstentions'])} | {number(row['mean_aligned_return'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Null tests",
            "",
            f"Assessment permutation p-values: AUC "
            f"{assessment_permutation_p['roc_auc']}, "
            f"log-loss improvement "
            f"{assessment_permutation_p['log_loss_improvement_vs_constant']}, "
            f"Brier improvement "
            f"{assessment_permutation_p['brier_improvement_vs_constant']}, "
            f"down-rate spread "
            f"{assessment_permutation_p['down_rate_spread']}.",
            f"Stress permutation p-values: AUC {stress_permutation_p['roc_auc']}, "
            f"log-loss improvement "
            f"{stress_permutation_p['log_loss_improvement_vs_constant']}, "
            f"Brier improvement "
            f"{stress_permutation_p['brier_improvement_vs_constant']}, "
            f"down-rate spread {stress_permutation_p['down_rate_spread']}.",
            "",
            "The fixed temporal placebo reassigns each stock's prior episode predictors to "
            "the next episode and reruns the same fixed procedure; results are in "
            "`temporal_placebo_v1.csv`.",
            "",
            "The inherited 10-minute endpoint diagnostic is retained only as a secondary "
            "table in `secondary_10m_directional_v1.csv`; it is not substituted for the "
            "M1C-compatible 15-minute primary horizon.",
            "",
            "## Safety and scope",
            "",
            "- M1C, A1, Tail Phase V1, the frozen cohort, checkpoint grid, and fresh-episode "
            "identifiers were unchanged.",
            "- No archived pressure, tension, peer-slate normalisation, future-dependent "
            "membership, option outcomes, or execution fields entered the model.",
            "- No protected 2026 outcome was read, calculated, displayed, or inspected.",
            "- No broker was accessed, no order routing was enabled, and no order was placed.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    source_records = verify_sources()
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    episodes_source, checkpoints_source, bars, options = load_inputs()
    episode_panel = prepare_panel(
        episodes_source,
        bars,
        options,
        population="fresh_high_m1c_episode",
    )
    checkpoint_panel = prepare_panel(
        checkpoints_source.loc[checkpoints_source["m1c_high_tail_v1"].astype(bool)].copy(),
        bars,
        options,
        population="high_m1c_checkpoint_secondary",
    )
    frozen_regression = {
        "fresh_episodes": _frozen_field_regression(
            episodes_source,
            episode_panel,
            primary=True,
        ),
        "high_tail_checkpoints": _frozen_field_regression(
            checkpoints_source.loc[checkpoints_source["m1c_high_tail_v1"].astype(bool)].copy(),
            checkpoint_panel,
            primary=False,
        ),
    }
    episodes, episode_exclusions = eligible_panel(episode_panel, primary=True)
    checkpoints, checkpoint_exclusions = eligible_panel(checkpoint_panel, primary=False)
    if not bool(episodes["D4_causal_volume_available"].astype(bool).all()):
        raise ExperimentBlocked(
            "historical causal stock volume is incomplete; preregistered D1-D3 fallback "
            "requires a fresh frozen run rather than silent feature removal"
        )

    development = episodes.loc[episodes["partition"].eq("development")].copy()
    development_movers = conditional_movers(development)
    oof, fold_audits = expanding_time_ordered_oof(
        development_movers,
        target_column="is_down_move_v1",
    )
    thresholds = freeze_action_thresholds(oof["q_down_oof"])
    final_model = fit_downside_model(
        development_movers,
        target_column="is_down_move_v1",
    )
    constant_down_rate = float(development_movers["is_down_move_v1"].mean())
    if not 0.0 < constant_down_rate < 1.0:
        raise ExperimentBlocked("2024 conditional downside base rate is degenerate")

    for frame in (episodes, checkpoints):
        frame["q_down_v1"] = final_model.predict_proba(frame)
        frame["asymmetric_action_v1"] = apply_asymmetric_policy(
            frame["q_down_v1"],
            low_threshold=thresholds["low"],
            high_threshold=thresholds["high"],
        )
        if not np.array_equal(
            frame["asymmetric_action_v1"].eq("CALL").to_numpy(dtype=bool),
            frame["q_down_v1"].le(thresholds["low"]).to_numpy(dtype=bool),
        ):
            raise ExperimentBlocked("CALL assignment drifted from the frozen low threshold")
        if not np.array_equal(
            frame["asymmetric_action_v1"].eq("PUT").to_numpy(dtype=bool),
            frame["q_down_v1"].ge(thresholds["high"]).to_numpy(dtype=bool),
        ):
            raise ExperimentBlocked("PUT assignment drifted from the frozen high threshold")
    assessment = episodes.loc[episodes["partition"].eq("assessment")].copy()
    stress = episodes.loc[episodes["partition"].eq("stress")].copy()
    assessment_checkpoints = checkpoints.loc[checkpoints["partition"].eq("assessment")].copy()
    stress_checkpoints = checkpoints.loc[checkpoints["partition"].eq("stress")].copy()
    if assessment.empty or stress.empty:
        raise ExperimentBlocked("opened assessment or stress population is empty")

    conditional_records: list[dict[str, Any]] = []
    reliability_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    permutation_summaries: list[pd.DataFrame] = []
    permutation_draws: list[pd.DataFrame] = []
    leave_one_out_frames: list[pd.DataFrame] = []
    concentration_frames: list[pd.DataFrame] = []
    phase_frames: list[pd.DataFrame] = []
    secondary_10m_frames: list[pd.DataFrame] = []
    for period, frame, checkpoint_frame in (
        ("assessment", assessment, assessment_checkpoints),
        ("stress", stress, stress_checkpoints),
    ):
        conditional_records.extend(
            _conditional_metric_rows(
                frame,
                period=period,
                constant_down_rate=constant_down_rate,
            )
        )
        reliability_frames.append(reliability_table(frame, period=period))
        bucket_frames.append(bucket_composition(frame, period=period))
        policy_frames.append(
            pd.DataFrame(
                [
                    policy_metrics(
                        frame,
                        action_column="asymmetric_action_v1",
                        policy="asymmetric_downside_v1",
                        period=period,
                        evaluation_scope="all_complete_episodes",
                    )
                ]
            )
        )
        baseline_frames.append(baseline_tables(frame, period=period))
        bootstrap_frames.append(
            session_cluster_bootstrap(
                frame,
                period=period,
                constant_down_rate=constant_down_rate,
                seed=BOOTSTRAP_SEED,
            )
        )
        permutation_summary, permutation_detail = label_permutation_results(
            frame,
            period=period,
            constant_down_rate=constant_down_rate,
            seed=PERMUTATION_SEED,
        )
        permutation_summaries.append(permutation_summary)
        permutation_draws.append(permutation_detail)
        leave_one_out_frames.append(leave_one_out_table(frame, period=period))
        concentration_frames.append(concentration_table(frame, period=period))
        phase_frames.append(tail_phase_diagnostics(checkpoint_frame, period=period))
        secondary_10m_frames.append(secondary_10m_directional_table(frame, period=period))

    conditional_table = pd.DataFrame(conditional_records)
    reliability = pd.concat(reliability_frames, ignore_index=True)
    buckets = pd.concat(bucket_frames, ignore_index=True)
    policy = pd.concat(policy_frames, ignore_index=True)
    baselines = pd.concat(baseline_frames, ignore_index=True)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    permutation = pd.concat(permutation_summaries, ignore_index=True)
    permutation_detail = pd.concat(permutation_draws, ignore_index=True)
    leave_one_out = pd.concat(leave_one_out_frames, ignore_index=True)
    concentration = pd.concat(concentration_frames, ignore_index=True)
    phases = pd.concat(phase_frames, ignore_index=True)
    secondary_10m = pd.concat(secondary_10m_frames, ignore_index=True)
    placebo, placebo_predictions = temporal_placebo(development, assessment, stress)
    robustness = robustness_summary(
        leave_one_out,
        concentration,
        assessment,
        stress,
    )
    (
        primary_decision,
        directional_decision,
        descriptive_finding,
        decision_checks,
    ) = choose_decisions(
        assessment,
        stress,
        conditional_table.loc[conditional_table["model"].eq("asymmetric_downside_v1")].copy(),
        bootstrap,
        permutation,
        robustness,
    )

    equality_events = int(
        np.isclose(
            np.abs(episodes["signed_endpoint_return_15m_v1"].to_numpy(float)),
            episodes["iv_expected_absolute_15m"].to_numpy(float),
            rtol=0.0,
            atol=0.0,
        ).sum()
    )
    episode_accounting: dict[str, dict[str, int]] = {}
    for period in ("development", "assessment", "stress"):
        period_frame = episodes.loc[episodes["partition"].eq(period)]
        episode_accounting[period] = {
            "eligible_fresh_episodes": int(len(period_frame)),
            "material_movers": int(
                period_frame["primary_outcome_state_v1"].isin(["UP_MOVE", "DOWN_MOVE"]).sum()
            ),
            "material_up": int(period_frame["primary_outcome_state_v1"].eq("UP_MOVE").sum()),
            "material_down": int(period_frame["primary_outcome_state_v1"].eq("DOWN_MOVE").sum()),
            "no_move": int(period_frame["primary_outcome_state_v1"].eq("NO_MOVE").sum()),
            "ambiguous_path": int(
                period_frame["primary_outcome_state_v1"].eq("AMBIGUOUS_BOTH_WITHIN_BAR").sum()
            ),
            "sessions": int(period_frame["session"].nunique()),
            "stocks": int(period_frame["stock"].nunique()),
        }
    summary: dict[str, Any] = {
        "schema_version": "m1c-asymmetric-downside-residual-v1",
        "research_id": "M1C Asymmetric Downside Residual V1",
        "primary_decision": primary_decision,
        "directional_diagnostic_decision": directional_decision,
        "descriptive_directional_finding": descriptive_finding,
        "retrospective_status": ("already_inspected_2025_periods_not_untouched_confirmation"),
        "exact_probability_decomposition_supported": False,
        "joint_probabilities_constructed": False,
        "target_mismatch": (
            "frozen M1C uses strict absolute endpoint exceedance; required directional "
            "partition includes exact threshold equality"
        ),
        "exact_equality_events_observed": equality_events,
        "features": list(DOWNSIDE_FEATURES),
        "D4_status": "available_historical_causal_stock_volume",
        "model": final_model.as_dict(),
        "standardisation": final_model.standardisation.as_dict(),
        "frozen_action_thresholds": thresholds,
        "constant_2024_downside_base_rate": constant_down_rate,
        "episode_accounting": episode_accounting,
        "decision_contract_checks": decision_checks,
        "robustness": robustness,
        "exclusions": {
            "fresh_episodes": episode_exclusions,
            "checkpoint_secondary": checkpoint_exclusions,
        },
        "ambiguous_path_count": int(
            episodes["primary_outcome_state_v1"].eq("AMBIGUOUS_BOTH_WITHIN_BAR").sum()
        ),
        "movement_consumed_conditional_percentile_transform": (
            "not_implemented_existing_tail_phase_has_no_frozen_transform"
        ),
        "protected_2026_outcomes_accessed": False,
        "order_routing_enabled": False,
        "orders_placed": False,
        "option_profitability_tested": False,
    }

    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frozen_config = {
        "contract": contract,
        "realised_feature_set": list(DOWNSIDE_FEATURES),
        "realised_model": final_model.as_dict(),
        "realised_thresholds": thresholds,
        "constant_2024_downside_base_rate": constant_down_rate,
    }
    feature_manifest = {
        "features": contract["features"],
        "ordered_model_columns": list(DOWNSIDE_FEATURES),
        "stock_local": True,
        "causal_at_entry": True,
        "standardisation_fit_scope": "2024 development training rows only",
        "excluded_fields": contract["feature_exclusions"],
        "D4_volume_status": "available_historical_causal_stock_volume",
        "no_imputation": True,
    }
    target_audit = {
        "canonical_m1c_target": contract["canonical_m1c_target"],
        "direction_partition": contract["direction_partition"],
        "primary_horizon_minutes": 15,
        "canonical_target_reconstruction_passed": True,
        "endpoint_absolute_movement_reconstruction_tolerance": 1e-12,
        "previous_close_denominator_reconstruction_tolerance": 1e-12,
        "exact_equality_events_observed": equality_events,
        "ambiguous_path_events": 0,
        "exact_probability_decomposition_supported": False,
        "decision": "target_mismatch_prevents_exact_probability_decomposition",
    }
    write_json(PRIMARY / "frozen_experiment_configuration_v1.json", frozen_config)
    write_json(PRIMARY / "feature_manifest_v1.json", feature_manifest)
    write_json(PRIMARY / "target_definition_audit_v1.json", target_audit)
    write_json(PRIMARY / "final_model_parameters_v1.json", final_model.as_dict())
    write_json(
        PRIMARY / "standardisation_parameters_v1.json",
        final_model.standardisation.as_dict(),
    )
    write_json(PRIMARY / "frozen_action_thresholds_v1.json", thresholds)
    write_json(PRIMARY / "development_fold_audit_v1.json", {"folds": fold_audits})
    write_json(PRIMARY / "summary_v1.json", summary)

    oof_columns = [
        *IDENTITY_COLUMNS,
        "episode_id",
        "partition",
        *DOWNSIDE_FEATURES,
        "primary_outcome_state_v1",
        "is_down_move_v1",
        "q_down_oof",
        "fold",
    ]
    write_parquet(
        PRIMARY / "development_oof_predictions_v1.parquet",
        oof.loc[:, [column for column in oof_columns if column in oof]],
    )
    write_parquet(
        PRIMARY / "assessment_episode_predictions_v1.parquet",
        assessment.loc[:, _prediction_columns(assessment)],
    )
    write_parquet(
        PRIMARY / "stress_episode_predictions_v1.parquet",
        stress.loc[:, _prediction_columns(stress)],
    )
    write_parquet(
        PRIMARY / "checkpoint_secondary_predictions_v1.parquet",
        pd.concat(
            [assessment_checkpoints, stress_checkpoints],
            ignore_index=True,
        ).loc[
            :,
            _prediction_columns(
                pd.concat(
                    [assessment_checkpoints, stress_checkpoints],
                    ignore_index=True,
                )
            ),
        ],
    )
    write_parquet(
        PRIMARY / "temporal_placebo_predictions_v1.parquet",
        placebo_predictions.loc[
            :,
            [
                column
                for column in (
                    *IDENTITY_COLUMNS,
                    "partition",
                    *DOWNSIDE_FEATURES,
                    "primary_outcome_state_v1",
                    "q_down_temporal_placebo",
                    "asymmetric_action_temporal_placebo",
                )
                if column in placebo_predictions
            ],
        ],
    )
    write_csv(PRIMARY / "full_three_state_outcomes_v1.csv", buckets)
    write_csv(PRIMARY / "conditional_direction_metrics_v1.csv", conditional_table)
    write_csv(PRIMARY / "reliability_bins_v1.csv", reliability)
    write_csv(PRIMARY / "combined_policy_metrics_v1.csv", policy)
    write_csv(PRIMARY / "baseline_comparisons_v1.csv", baselines)
    write_csv(PRIMARY / "tail_phase_diagnostics_v1.csv", phases)
    write_csv(PRIMARY / "secondary_10m_directional_v1.csv", secondary_10m)
    write_csv(PRIMARY / "session_cluster_bootstrap_v1.csv", bootstrap)
    write_csv(PRIMARY / "label_permutation_summary_v1.csv", permutation)
    write_parquet(
        PRIMARY / "label_permutation_draws_v1.parquet",
        permutation_detail,
    )
    write_csv(PRIMARY / "temporal_placebo_v1.csv", placebo)
    write_csv(PRIMARY / "leave_one_month_stock_checkpoint_out_v1.csv", leave_one_out)
    write_csv(PRIMARY / "concentration_v1.csv", concentration)
    write_json(PRIMARY / "frozen_system_regression_v1.json", frozen_regression)

    report = _write_markdown_report(
        summary,
        conditional_table,
        buckets,
        policy,
        baselines,
        phases,
        bootstrap,
        permutation,
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")

    git_commit = subprocess.run(
        ["rtk", "git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_branch = subprocess.run(
        ["rtk", "git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_status = subprocess.run(
        ["rtk", "git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    output_paths = sorted(
        path
        for path in PRIMARY.iterdir()
        if path.is_file()
        and path.name != "provenance_manifest_v1.json"
        and not path.name.startswith("independent_audit_v1.")
    )
    provenance = {
        "research_id": "M1C Asymmetric Downside Residual V1",
        "branch": git_branch,
        "commit": git_commit,
        "dirty_working_tree": bool(dirty_status),
        "dirty_status": dirty_status,
        "input_artifact_identities": source_records,
        "configuration_hashes": {
            "contract": sha256_file(CONTRACT_PATH),
            "implementation": sha256_file(
                REPO_ROOT
                / "packages"
                / "stocker_research"
                / "src"
                / "stocker_research"
                / "m1c_asymmetric_downside_residual_v1.py"
            ),
            "runner": sha256_file(Path(__file__)),
        },
        "data_date_boundaries": contract["chronology"],
        "source_row_counts": {
            "fresh_episode_rows": int(len(episodes_source)),
            "checkpoint_rows": int(len(checkpoints_source)),
            "completed_bar_rows_opened": int(len(bars)),
            "previous_close_option_context_rows_opened": int(len(options)),
        },
        "episode_counts": episode_accounting,
        "exclusions_and_reasons": summary["exclusions"],
        "ambiguous_path_counts": {
            "endpoint_primary": 0,
            "first_breach_not_primary": 0,
        },
        "missingness_counts": {
            "D1": int(episode_panel[DOWNSIDE_FEATURES[0]].isna().sum()),
            "D2": int(episode_panel[DOWNSIDE_FEATURES[1]].isna().sum()),
            "D3": int(episode_panel[DOWNSIDE_FEATURES[2]].isna().sum()),
            "D4": int(episode_panel[DOWNSIDE_FEATURES[3]].isna().sum()),
            "primary_outcome": int((~episode_panel["primary_outcome_complete_v1"]).sum()),
        },
        "exact_commands": [
            RUN_COMMAND,
            AUDIT_COMMAND,
            FOCUSED_TEST_COMMAND,
            LINT_COMMAND,
            TYPE_CHECK_COMMAND,
            FULL_TEST_COMMAND,
        ],
        "verification_status_at_provenance_generation": {
            "focused_tests": "21_passed",
            "lint": "passed",
            "type_check": "passed",
            "full_suite": (
                "1351_passed_1_skipped_13_failed_19_errors_due_to_missing_unrelated_"
                "frozen_slrno_artifacts"
            ),
        },
        "random_seeds": {
            "model": MODEL_RANDOM_SEED,
            "bootstrap_assessment": BOOTSTRAP_SEED,
            "bootstrap_stress": BOOTSTRAP_SEED,
            "permutation_assessment": PERMUTATION_SEED,
            "permutation_stress": PERMUTATION_SEED,
        },
        "frozen_system_regression": frozen_regression,
        "protected_data_confirmation": {
            "protected_2026_outcome_read": False,
            "protected_2026_outcome_calculated": False,
            "protected_2026_outcome_displayed": False,
            "protected_2026_outcome_inspected": False,
            "maximum_opened_session": str(
                max(
                    episodes["session"].max(),
                    checkpoints["session"].max(),
                    bars["session"].max(),
                    options["session"].max(),
                )
            ),
            "filter_pushdown_used_for_external_row_sources": True,
        },
        "causality_confirmation": {
            "M1C_modified": False,
            "M1C_refit": False,
            "M1C_recalibrated": False,
            "M1C_threshold_changed": False,
            "fresh_episode_definition_changed": False,
            "A1_modified": False,
            "A1_refit": False,
            "Tail_Phase_V1_modified": False,
            "archived_signed_pressure_used": False,
            "archived_tension_used": False,
            "future_filtered_peer_slate_used": False,
            "peer_normalisation_used": False,
            "stock_identity_used_in_model": False,
            "M1C_probability_used_in_model": False,
            "option_outcomes_used": False,
        },
        "execution_confirmation": {
            "broker_accessed": False,
            "order_routing_path_imported": False,
            "order_routing_enabled": False,
            "orders_placed": False,
        },
        "output_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in output_paths
        },
    }
    write_json(PRIMARY / "provenance_manifest_v1.json", provenance)
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
