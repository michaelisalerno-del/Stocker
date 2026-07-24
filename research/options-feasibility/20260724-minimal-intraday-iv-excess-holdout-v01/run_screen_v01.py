#!/usr/bin/env python3
"""Run the strictly resumed minimal intraday-H0 IV-excess holdout V0.1."""

from __future__ import annotations

# ruff: noqa: E402 -- deterministic numerical limits precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-minimal-intraday-iv-excess-v01-mpl")

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
V0_EXPERIMENT_DIR = (
    REPO_ROOT / "research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0"
)
V0_PRIMARY = V0_EXPERIMENT_DIR / "artifacts" / "primary"
V0_RUNNER = V0_EXPERIMENT_DIR / "run_screen_v0.py"
DEFAULT_PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
DEFAULT_OPTIONS_CACHE = (
    REPO_ROOT
    / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v01"
    / "canonical/exact_holdout_options.parquet"
)
DEFAULT_STOCK_CACHE = (
    REPO_ROOT
    / "data/cache/minimal-intraday-iv-excess-holdout-v0"
    / "frozen_h0_stock_surface.parquet"
)
DEFAULT_STATE_CACHE = DEFAULT_STOCK_CACHE.with_name("frozen_state_surface.parquet")
DENSE_PANEL = (
    REPO_ROOT
    / "research/route-competition/20260722-broad-conflict-advance-hazard-v02"
    / "artifacts/primary/dense_advance_panel.parquet"
)
EXPECTED_STOCK_CACHE_SHA256 = "d81655b54a5c5e2e8b2d324e2e6520716700965612404202793ec1b3d9b0e846"
EXPECTED_STATE_CACHE_SHA256 = "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0"
EXPECTED_DENSE_PANEL_SHA256 = "a916b792e15e8630dadc09bed64d71be5533ce9f3b2bd93af06605d0faaa0cc3"
STARTING_SHA = "8b612d1def5198d64bc3927c09467f64bbe0841b"

for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.minimal_intraday_iv_excess_holdout_v01 import (
    MAXIMUM_ADDITIONAL_BYTES,
    MAXIMUM_ADDITIONAL_RECORDS,
    MAXIMUM_CUMULATIVE_RECORDS,
    SAFETY_FLAGS,
    add_movement_outcomes_with_optional_30m,
    assert_v01_safety_flags,
    attach_movement_prices_with_optional_30m,
    authorize_outcome_access,
    coverage_preflight,
    movement_timing_metrics_with_optional_30m,
)


class ScreenBlocked(RuntimeError):
    """One authorized fail-closed V0.1 experiment decision."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_identity(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    hashes = pd.util.hash_pandas_object(selected, index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"cannot load frozen V0 runner: {path}",
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def configure_v0_runner() -> ModuleType:
    v0 = load_module(V0_RUNNER, "minimal_holdout_v0_runner_for_v01")
    v0.PRIMARY = PRIMARY
    v0.REPORTS = REPORTS
    v0.SAFETY_FLAGS = SAFETY_FLAGS
    return v0


def load_contract(v0: ModuleType) -> dict[str, Any]:
    contract = cast(
        dict[str, Any],
        json.loads((EXPERIMENT_DIR / "contract.json").read_text(encoding="utf-8")),
    )
    assert_v01_safety_flags(contract)
    expected_limits = {
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
        "primary_logistic_models": 2,
        "binding_tails": 1,
        "session_bootstrap_draws": 10,
        "intraday_h0_bundle_null_refits": 3,
        "maximum_additional_eodhd_records": MAXIMUM_ADDITIONAL_RECORDS,
        "maximum_cumulative_eodhd_records": MAXIMUM_CUMULATIVE_RECORDS,
        "maximum_additional_raw_bytes": MAXIMUM_ADDITIONAL_BYTES,
        "maximum_plots": 2,
        "full_repository_test_suite": False,
    }
    limits = cast(Mapping[str, Any], contract["hard_limits"])
    if any(limits.get(name) != value for name, value in expected_limits.items()):
        raise ScreenBlocked(
            "blocked_resume_resource_limit",
            "V0.1 contract resource limits differ from the authorized resume",
        )
    if tuple(int(value) for value in contract["checkpoints"]) != tuple(v0.CHECKPOINTS):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "structural checkpoints differ from V0",
        )
    return contract


def load_h0_feature_surface(
    stock_cache: Path,
    v0: ModuleType,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reload the frozen feature-only H0 surface without movement prices."""

    if not stock_cache.is_file() or sha256_file(stock_cache) != EXPECTED_STOCK_CACHE_SHA256:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            "frozen V0 H0 cache is missing or drifted",
        )
    if not DENSE_PANEL.is_file() or sha256_file(DENSE_PANEL) != EXPECTED_DENSE_PANEL_SHA256:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            "frozen predecessor H0 panel is missing or drifted",
        )
    columns = [
        "row_id",
        "symbol",
        "session",
        "period",
        "checkpoint",
        "row_weight",
        *v0.GROUP_I,
    ]
    current = pd.read_parquet(stock_cache, columns=columns)
    predecessor = pd.read_parquet(
        DENSE_PANEL,
        columns=[
            "row_id",
            "session",
            "advance_eligible",
            "row_weight",
            *v0.GROUP_I,
        ],
    )
    predecessor = predecessor.loc[predecessor["advance_eligible"].astype(int).eq(1)].copy()
    historical = current.loc[
        pd.to_datetime(current["session"], errors="raise").le("2025-08-22")
    ].copy()
    first = predecessor.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    second = historical.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    mismatches = int(v0.row_identity_mismatches(first, second))
    difference = float(
        v0.maximum_numeric_difference(
            first,
            second,
            [*v0.GROUP_I, "row_weight"],
        )
    )
    if mismatches or difference > 1e-12:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            "reused H0 feature surface no longer reconstructs its predecessor",
        )
    holdout = current.loc[current["period"].astype(str).eq("holdout")]
    manifest = {
        **SAFETY_FLAGS,
        "stock_cache_path": str(stock_cache),
        "stock_cache_sha256": sha256_file(stock_cache),
        "historical_row_identity_mismatches": mismatches,
        "maximum_historical_h0_feature_weight_difference": difference,
        "holdout_candidate_rows_before_options": int(len(holdout)),
        "holdout_candidate_sessions": int(holdout["session"].nunique()),
        "holdout_candidate_stocks": int(holdout["symbol"].nunique()),
        "movement_columns_read": False,
        "passed": True,
    }
    return current, manifest


def coverage_gap_rows(option_audit: pd.DataFrame) -> pd.DataFrame:
    gaps = option_audit.loc[~option_audit["pair_available"].astype(bool)].copy()
    columns = [
        "request_id",
        "symbol",
        "holdout_session",
        "required_options_date",
        "gap_type",
        "gap_reason",
    ]
    if gaps.empty:
        return pd.DataFrame(columns=columns)
    gaps = gaps.rename(columns={"session": "holdout_session", "pair_reason": "gap_reason"})
    gaps["request_id"] = ""
    gaps["gap_type"] = "front_pair_selection"
    return gaps.loc[:, columns]


def perform_coverage_preflight(
    *,
    h0_surface: pd.DataFrame,
    provider_root: Path,
    options_cache: Path,
    v0: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select exact-D-1 pairs and gate the prospective join before outcomes."""

    context, pair_coverage, option_audit, option_manifest = v0.build_holdout_option_context(
        h0_surface,
        provider_root=provider_root,
        options_cache=options_cache,
    )
    feature_panel, structural_audit = v0.join_holdout_panel(h0_surface, context)
    planned_stock_sessions = int(h0_surface[["symbol", "session"]].drop_duplicates().shape[0])
    planned_session_count = int(h0_surface["session"].nunique())
    preflight = coverage_preflight(
        feature_panel,
        context,
        planned_stock_sessions=planned_stock_sessions,
        planned_session_count=planned_session_count,
        planned_stock_month_cells=80,
    )
    expected = (
        feature_panel.assign(month=feature_panel["session"].astype(str).str[:7])
        .groupby(["symbol", "month"], sort=True, observed=True)
        .agg(
            expected_joined_rows=("row_id", "size"),
            expected_weight=("row_weight", "sum"),
        )
        .reset_index()
    )
    coverage = pair_coverage.merge(
        expected,
        on=["symbol", "month"],
        how="left",
        validate="one_to_one",
    )
    coverage["expected_joined_rows"] = (
        pd.to_numeric(coverage["expected_joined_rows"], errors="coerce").fillna(0).astype(int)
    )
    coverage["expected_weight"] = pd.to_numeric(
        coverage["expected_weight"], errors="coerce"
    ).fillna(0.0)
    by_checkpoint = (
        feature_panel.groupby("checkpoint", sort=True, observed=True)
        .agg(
            expected_joined_rows=("row_id", "size"),
            expected_sessions=("session", "nunique"),
            expected_stocks=("symbol", "nunique"),
            expected_weight=("row_weight", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    preflight.update(
        {
            "exact_previous_close_chains": int(option_manifest["exact_chain_sessions"]),
            "valid_atm_pairs": int(option_manifest["selected_pair_sessions"]),
            "pair_coverage_by_stock_month": coverage.to_dict(orient="records"),
            "pair_coverage_by_structural_checkpoint": by_checkpoint,
            "same_day_or_future_observations": int(
                option_manifest["same_day_or_future_observations"]
            ),
            "non_exact_previous_session_observations": int(
                option_manifest["non_exact_previous_session_observations"]
            ),
            "front_pair_rule": {
                "expiry": "nearest_valid_7_to_45_DTE",
                "strike": "common_call_put_min_abs_log_strike_over_prior_close",
                "substitute_expiry_after_quality_failure": False,
            },
        }
    )
    write_parquet(PRIMARY / "holdout_selected_option_pairs.parquet", context)
    write_csv(PRIMARY / "holdout_options_coverage.csv", coverage)
    write_csv(PRIMARY / "remaining_options_gap.csv", coverage_gap_rows(option_audit))
    structural = structural_audit.copy()
    structural["audit_type"] = "prospective_structural_join"
    write_csv(PRIMARY / "holdout_join_audit.csv", structural)
    write_json(PRIMARY / "holdout_coverage_preflight.json", preflight)
    return feature_panel, context, preflight


def freeze_models(
    *,
    feature_panel: pd.DataFrame,
    v0: ModuleType,
) -> tuple[Any, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Refit exactly M0/M1 on 2024 and freeze their 2024 thresholds."""

    historical_path = v0.historical_panel_path()
    historical = pd.read_parquet(historical_path)
    models, development, _reference, reconstruction = v0.reconstruct_historical_models(historical)
    thresholds = cast(dict[str, Any], reconstruction.pop("thresholds"))
    predecessor_thresholds = cast(
        dict[str, Any],
        json.loads((V0_PRIMARY / "frozen_tail_thresholds.json").read_text(encoding="utf-8")),
    )
    threshold_difference = max(
        abs(
            float(thresholds["M0_top_5_percent_threshold"])
            - float(predecessor_thresholds["M0_top_5_percent_threshold"])
        ),
        abs(
            float(thresholds["M1_top_5_percent_threshold"])
            - float(predecessor_thresholds["M1_top_5_percent_threshold"])
        ),
    )
    reconstruction.update(
        {
            **SAFETY_FLAGS,
            "strict_resume_from_v0": True,
            "maximum_v0_threshold_difference": threshold_difference,
            "predecessor_frozen_thresholds_reproduced": threshold_difference <= 1e-12,
            "passed": bool(reconstruction["passed"] and threshold_difference <= 1e-12),
        }
    )
    if not reconstruction["passed"]:
        raise ScreenBlocked(
            "blocked_historical_model_reconstruction_failure",
            "M0/G0 or frozen V0 thresholds failed exact reconstruction",
        )
    feature_manifest = cast(dict[str, Any], v0.minimal_feature_manifest())
    feature_manifest.update(SAFETY_FLAGS)
    write_json(PRIMARY / "minimal_feature_manifest.json", feature_manifest)
    configurations = {
        **SAFETY_FLAGS,
        "M0": {
            "numeric_features": list(v0.GROUP_O),
            "categorical_controls": ["stock"],
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
            "fitted_period": "2024_only",
        },
        "M1": {
            "numeric_features": [*v0.GROUP_O, *v0.GROUP_I],
            "categorical_controls": ["stock"],
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
            "fitted_period": "2024_only",
        },
        "primary_models_fitted": 2,
    }
    coefficients = {
        **SAFETY_FLAGS,
        "M0": v0.model_specification(models.m0),
        "M1": v0.model_specification(models.m1),
    }
    if int(coefficients["M0"]["iterations"]) >= 300 or int(coefficients["M1"]["iterations"]) >= 300:
        raise ScreenBlocked(
            "blocked_model_convergence_failure",
            "a frozen logistic fit reached max_iter",
        )
    write_json(PRIMARY / "historical_model_reconstruction.json", reconstruction)
    write_json(PRIMARY / "model_configurations.json", configurations)
    write_json(PRIMARY / "model_coefficients.json", coefficients)
    write_json(
        PRIMARY / "frozen_tail_thresholds.json",
        {
            **SAFETY_FLAGS,
            **thresholds,
            "written_before_holdout_outcomes": True,
            "holdout_rank_forcing": False,
        },
    )
    feature_panel["M0_probability"] = models.m0.predict(feature_panel)
    feature_panel["M1_probability"] = models.m1.predict(feature_panel)
    feature_panel["M0_top_5pct"] = v0.frozen_tail_membership(
        feature_panel["M0_probability"].to_numpy(float),
        float(thresholds["M0_top_5_percent_threshold"]),
    )
    feature_panel["M1_top_5pct"] = v0.frozen_tail_membership(
        feature_panel["M1_probability"].to_numpy(float),
        float(thresholds["M1_top_5_percent_threshold"]),
    )
    freeze_manifest = {
        **SAFETY_FLAGS,
        "frozen": True,
        "holdout_outcomes_read_before_freeze": False,
        "coverage_preflight_passed_before_freeze": True,
        "feature_manifest_sha256": sha256_file(PRIMARY / "minimal_feature_manifest.json"),
        "model_configurations_sha256": sha256_file(PRIMARY / "model_configurations.json"),
        "model_coefficients_sha256": sha256_file(PRIMARY / "model_coefficients.json"),
        "frozen_tail_thresholds_sha256": sha256_file(PRIMARY / "frozen_tail_thresholds.json"),
        "model_classes": {
            "M0": type(models.m0).__name__,
            "M1": type(models.m1).__name__,
        },
        "pre_outcome_joined_row_identity": dataframe_identity(
            feature_panel.sort_values("row_id", kind="mergesort"),
            ["row_id", "symbol", "session", "checkpoint", "row_weight"],
        ),
        "pre_outcome_feature_identity": dataframe_identity(
            feature_panel.sort_values("row_id", kind="mergesort"),
            ["row_id", *v0.GROUP_O, *v0.GROUP_I],
        ),
        "threshold_values": {
            "M0_top_5_percent_threshold": thresholds["M0_top_5_percent_threshold"],
            "M1_top_5_percent_threshold": thresholds["M1_top_5_percent_threshold"],
        },
    }
    write_json(PRIMARY / "pre_outcome_freeze_manifest.json", freeze_manifest)
    return models, development, thresholds, reconstruction


def append_metric_months(metrics: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["months"] = np.nan
    output.loc[output["model"].isin(["M0", "M1"]), "months"] = int(
        pd.to_datetime(holdout["session"]).dt.to_period("M").nunique()
    )
    return output


def build_report(
    *,
    decision: Mapping[str, Any],
    download: Mapping[str, Any],
    preflight: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    model_metrics: pd.DataFrame,
    increment: Mapping[str, float],
    tail_table: pd.DataFrame,
    comparison: Mapping[str, float],
    overlap: Mapping[str, Any],
    timing: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
) -> str:
    indexed = model_metrics.loc[model_metrics["model"].isin(["M0", "M1"])].set_index("model")
    m0_tail = tail_table.loc[tail_table["model"].eq("M0")].iloc[0]
    m1_tail = tail_table.loc[tail_table["model"].eq("M1")].iloc[0]
    intervals = bootstrap.loc[bootstrap["record_type"].eq("interval")]
    monthly = pd.read_csv(PRIMARY / "holdout_monthly_metrics.csv")
    stability = pd.read_csv(PRIMARY / "holdout_checkpoint_metrics.csv")
    joined_support = cast(Mapping[str, Any], decision["joined_holdout_support"])
    tail_support = cast(Mapping[str, Any], decision["frozen_tail_support"])
    lines = [
        "# Minimal Intraday Stock → IV-Excess Holdout Validation V0.1",
        "",
        f"Overall decision: `{decision['overall_decision']}`.",
        "",
        "Retrospective, research-only frozen holdout validation. This is not option "
        "P&L, prospective trading validation, or a deployable strategy.",
        "",
        "## Resume and freeze",
        "",
        f"- Complete V0 receipts reused: {download['complete_receipts_reused']}.",
        f"- Interrupted request repaired: {download['interrupted_request_repaired']}.",
        f"- New complete logical requests: {download['new_requests_completed']}; "
        f"provider records: {download['new_provider_records']}; bytes: "
        f"{download['new_bytes_downloaded']}.",
        f"- New exact-date records: {download['new_exact_date_records']}; extra-date "
        f"records rejected: {download['new_extra_date_records_rejected']}; 2026-or-later "
        f"records rejected: {download['new_2026_or_later_records_rejected']}.",
        f"- Requests remaining: {download['requests_remaining']}; unauthorized rows "
        f"materialized: {download['protected_or_unauthorised_records_materialised']}.",
        f"- Coverage preflight passed before outcomes: `{preflight['passed']}`; "
        f"planned stock-sessions/trading sessions: "
        f"{preflight['total_holdout_stock_sessions']}/{preflight['planned_session_count']}; "
        f"exact chains/valid pairs: {preflight['exact_previous_close_chains']}/"
        f"{preflight['valid_atm_pairs']}; "
        f"expected rows/sessions/stocks: {preflight['expected_joined_rows']}/"
        f"{preflight['expected_session_count']}/{preflight['expected_stock_count']}.",
        f"- Historical M0/G0 reconstruction passed: `{reconstruction['passed']}`.",
        f"- Frozen M0 threshold: {float(thresholds['M0_top_5_percent_threshold']):.12g}.",
        f"- Frozen M1 threshold: {float(thresholds['M1_top_5_percent_threshold']):.12g}.",
        "",
        "## Holdout forecast",
        "",
    ]
    for model in ("M0", "M1"):
        row = indexed.loc[model]
        lines.append(
            f"- {model}: log loss {float(row['log_loss']):.8f}; Brier "
            f"{float(row['brier_score']):.8f}; AUC {float(row['auc']):.8f}; "
            f"average precision {float(row['average_precision']):.8f}; ECE "
            f"{float(row['expected_calibration_error']):.8f}; calibration intercept/slope "
            f"{float(row['calibration_intercept']):.8f}/"
            f"{float(row['calibration_slope']):.8f}; base rate "
            f"{float(row['base_rate']):.8f}; realised-class probability "
            f"{float(row['mean_probability_realised_class']):.8f}."
        )
    lines.extend(
        [
            f"- M1−M0: log-loss {increment['log_loss_improvement']:.8f}; Brier "
            f"{increment['brier_improvement']:.8f}; AUC "
            f"{increment['auc_improvement']:.8f}; average precision "
            f"{increment['average_precision_improvement']:.8f}; ECE "
            f"{increment['expected_calibration_error_improvement']:.8f}.",
            f"- Joined support: rows/sessions/stocks/months/positives "
            f"{joined_support['rows']}/{joined_support['sessions']}/"
            f"{joined_support['stocks']}/{joined_support['months']}/"
            f"{joined_support['positive_outcomes']}; passed "
            f"`{joined_support['passed']}`.",
            "",
            "## Stability",
            "",
        ]
    )
    for group in ("2025-09", "2025-10", "2025-11", "2025-12"):
        row = monthly.loc[monthly["record_type"].eq("increment") & monthly["group"].eq(group)].iloc[
            0
        ]
        tail = monthly.loc[
            monthly["record_type"].eq("tail")
            & monthly["group"].eq(group)
            & monthly["model"].eq("M1")
        ].iloc[0]
        lines.append(
            f"- {group}: M1−M0 log-loss/Brier/AUC/AP "
            f"{float(row['log_loss_improvement']):.8f}/"
            f"{float(row['brier_improvement']):.8f}/"
            f"{float(row['auc_improvement']):.8f}/"
            f"{float(row['average_precision_improvement']):.8f}; M1-tail "
            f"mean/median residual and exceed rate "
            f"{float(tail['mean_iv_residual']):.8f}/"
            f"{float(tail['median_iv_residual']):.8f}/"
            f"{float(tail['exceed_iv_rate']):.6f}."
        )
    for scope, group in (
        ("checkpoint_group", "early_6_14"),
        ("checkpoint_group", "middle_16_24"),
        ("checkpoint_group", "late_26_34"),
        ("prior_close_atm_iv", "low"),
        ("prior_close_atm_iv", "high"),
        ("intraday_transition_probability", "low"),
        ("intraday_transition_probability", "high"),
    ):
        row = stability.loc[
            stability["scope"].eq(scope)
            & stability["group"].eq(group)
            & stability["record_type"].eq("increment")
        ].iloc[0]
        lines.append(
            f"- {scope}:{group}: M1−M0 log-loss/Brier/AUC/AP "
            f"{float(row['log_loss_improvement']):.8f}/"
            f"{float(row['brier_improvement']):.8f}/"
            f"{float(row['auc_improvement']):.8f}/"
            f"{float(row['average_precision_improvement']):.8f}."
        )
    lines.extend(
        [
            "",
            "## Frozen M1 top-5% tail",
            "",
            f"- Rows/sessions/stocks/months: {int(m1_tail['rows'])}/"
            f"{int(m1_tail['sessions'])}/{int(m1_tail['stocks'])}/"
            f"{int(m1_tail['months'])}.",
            f"- Mean/median movement {float(m1_tail['mean_absolute_movement']):.8f}/"
            f"{float(m1_tail['median_absolute_movement']):.8f}; mean IV expectation "
            f"{float(m1_tail['mean_iv_expectation']):.8f}.",
            f"- Mean/median/trimmed residual "
            f"{float(m1_tail['mean_iv_residual']):.8f}/"
            f"{float(m1_tail['median_iv_residual']):.8f}/"
            f"{float(m1_tail['trimmed_10pct_mean_iv_residual']):.8f}; exceed-IV "
            f"and positive-residual rates {float(m1_tail['exceed_iv_rate']):.6f}/"
            f"{float(m1_tail['positive_residual_rate']):.6f}; IV sigma ratio "
            f"{float(m1_tail['iv_sigma_ratio']):.6f}.",
            f"- Residual p05/p25/p75/p95 "
            f"{float(m1_tail['iv_residual_percentile_05']):.8f}/"
            f"{float(m1_tail['iv_residual_percentile_25']):.8f}/"
            f"{float(m1_tail['iv_residual_percentile_75']):.8f}/"
            f"{float(m1_tail['iv_residual_percentile_95']):.8f}.",
            f"- Concentration: largest-5%-row contribution "
            f"{float(m1_tail['top_5pct_positive_residual_contribution']):.6f}; max "
            f"stock/month/session shares {float(m1_tail['maximum_stock_share']):.6f}/"
            f"{float(m1_tail['maximum_month_share']):.6f}/"
            f"{float(m1_tail['maximum_session_share']):.6f}; support passed "
            f"`{tail_support['passed']}`.",
            "",
            "## Frozen M0 versus M1 tails",
            "",
            f"- M0 rows and mean/median residual/exceed rate: {int(m0_tail['rows'])}; "
            f"{float(m0_tail['mean_iv_residual']):.8f}/"
            f"{float(m0_tail['median_iv_residual']):.8f}/"
            f"{float(m0_tail['exceed_iv_rate']):.6f}.",
            f"- M1−M0 mean/median residual, exceed-rate, movement, IV-ratio, "
            f"positive-rate, concentration differences: "
            f"{comparison['mean_iv_residual_difference']:.8f}/"
            f"{comparison['median_iv_residual_difference']:.8f}/"
            f"{comparison['exceed_iv_rate_difference']:.8f}/"
            f"{comparison['absolute_movement_difference']:.8f}/"
            f"{comparison['iv_sigma_ratio_difference']:.8f}/"
            f"{comparison['positive_residual_rate_difference']:.8f}/"
            f"{comparison['tail_concentration_difference']:.8f}.",
            f"- Intersection/union/Jaccard/M1-only/M0-only: "
            f"{int(overlap['intersection_rows'])}/{int(overlap['union_rows'])}/"
            f"{float(overlap['jaccard_overlap']):.6f}/"
            f"{int(overlap['M1_only_rows'])}/{int(overlap['M0_only_rows'])}.",
            "",
            "## Movement timing",
            "",
        ]
    )
    for row in timing.itertuples(index=False):
        lines.append(
            f"- {int(row.horizon_minutes)}m: mean residual "
            f"{float(row.mean_iv_residual):.8f}; median residual "
            f"{float(row.median_iv_residual):.8f}; exceed rate "
            f"{float(row.exceed_iv_rate):.6f}; 30m movement realised "
            f"{float(row.percent_eventual_30m_movement_realized):.6f}; maximum-excursion "
            f"bucket share {float(row.maximum_absolute_excursion_bucket_share):.6f}."
        )
    lines.extend(["", "## Coarse fixed-prediction bootstrap", ""])
    for statistic in intervals["statistic"].drop_duplicates():
        selected = intervals.loc[intervals["statistic"].eq(statistic)].sort_values("interval_level")
        rendered = "; ".join(
            f"{int(float(row.interval_level) * 100)}% "
            f"[{float(row.lower):.8f}, {float(row.upper):.8f}]"
            for row in selected.itertuples(index=False)
        )
        lines.append(f"- {statistic}: {rendered}.")
    lines.extend(
        [
            "",
            f"Real increment exceeded H0 nulls for log loss "
            f"{int(nulls['real_exceeds_null_log_loss_improvement'].sum())}/3, "
            f"Brier {int(nulls['real_exceeds_null_brier_improvement'].sum())}/3, "
            f"AUC {int(nulls['real_exceeds_null_auc_improvement'].sum())}/3, and "
            f"average precision "
            f"{int(nulls['real_exceeds_null_average_precision_improvement'].sum())}/3.",
            "",
            "## Status and reproducibility",
            "",
            f"- Component statuses: minimal model `{decision['minimal_model_status']}`; "
            f"frozen tail `{decision['frozen_top_5pct_status']}`; options-only "
            f"comparison `{decision['options_only_tail_comparison_status']}`; timing "
            f"`{decision['movement_timing_status']}`; coverage "
            f"`{decision['holdout_options_coverage_status']}`; download "
            f"`{decision['download_resume_status']}`.",
            "- Independent audit and determinism are recorded in `lightweight_audit.json` "
            "and `determinism_check.json`; the post-run auditor must pass before delivery.",
            "",
            "Ten bootstrap draws are a coarse whole-session diagnostic. The binding "
            "target remains 15-minute underlying movement versus exact previous-close "
            "ATM-IV expectation.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(
    *,
    provider_root: Path,
    options_cache: Path,
    stock_cache: Path,
    state_cache: Path,
) -> dict[str, Any]:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    v0 = configure_v0_runner()
    load_contract(v0)
    download = cast(
        dict[str, Any],
        json.loads((PRIMARY / "resume_download_manifest.json").read_text(encoding="utf-8")),
    )
    assert_v01_safety_flags(download)
    if (
        download.get("status") != "complete"
        or int(download.get("cumulative_complete_requests", 0)) != 1_700
        or int(download.get("requests_remaining", -1)) != 0
    ):
        raise ScreenBlocked(
            "blocked_holdout_options_download_failure",
            "resume download is not complete",
        )

    h0_surface, h0_manifest = load_h0_feature_surface(stock_cache, v0)
    feature_panel, option_context, preflight = perform_coverage_preflight(
        h0_surface=h0_surface,
        provider_root=provider_root,
        options_cache=options_cache,
        v0=v0,
    )
    if not bool(preflight["passed"]):
        raise ScreenBlocked(
            "blocked_insufficient_holdout_options_coverage",
            "prospective joined coverage failed before outcomes were opened",
        )

    _models, development, thresholds, reconstruction = freeze_models(
        feature_panel=feature_panel,
        v0=v0,
    )
    freeze_manifest = cast(
        dict[str, Any],
        json.loads((PRIMARY / "pre_outcome_freeze_manifest.json").read_text(encoding="utf-8")),
    )
    authorize_outcome_access(preflight, freeze_manifest)

    if not state_cache.is_file() or sha256_file(state_cache) != EXPECTED_STATE_CACHE_SHA256:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "frozen state cache is missing or drifted after pre-outcome freeze",
        )
    states = pd.read_parquet(
        state_cache,
        columns=["symbol", "session", "bar_ordinal", "bar_start_timestamp", "open", "close"],
    )
    timestamps = pd.to_datetime(states["bar_start_timestamp"], errors="raise", utc=True)
    if bool(timestamps.ge("2026-01-01T00:00:00Z").any()):
        raise ScreenBlocked(
            "blocked_protected_boundary_failure",
            "protected state row was materialized",
        )
    holdout = attach_movement_prices_with_optional_30m(feature_panel, states)
    holdout = add_movement_outcomes_with_optional_30m(holdout)
    v0.validate_holdout_dates(holdout["session"])
    if len(holdout) != len(feature_panel):
        raise ScreenBlocked(
            "blocked_insufficient_holdout_support",
            "a prospective joined row lacked the frozen movement horizons",
        )
    holdout = holdout.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    write_parquet(PRIMARY / "holdout_panel.parquet", holdout)

    prediction_columns = [
        "row_id",
        "symbol",
        "session",
        "checkpoint",
        "row_weight",
        "M0_probability",
        "M1_probability",
        "M0_top_5pct",
        "M1_top_5pct",
        v0.TARGET_COLUMN,
        *(f"absolute_log_return_{horizon}m" for horizon in v0.HORIZONS),
        *(f"iv_expected_absolute_{horizon}m" for horizon in v0.HORIZONS),
        *(f"iv_absolute_residual_{horizon}m" for horizon in v0.HORIZONS),
    ]
    write_parquet(
        PRIMARY / "holdout_predictions.parquet",
        holdout.loc[:, prediction_columns],
    )

    model_metrics, increment = v0.prediction_metrics(holdout)
    model_metrics = append_metric_months(model_metrics, holdout)
    write_csv(PRIMARY / "holdout_model_metrics.csv", model_metrics)
    monthly, checkpoint = v0.stability_tables(holdout, thresholds)
    write_csv(PRIMARY / "holdout_monthly_metrics.csv", monthly)
    write_csv(PRIMARY / "holdout_checkpoint_metrics.csv", checkpoint)

    m1_tail_frame = holdout.loc[holdout["M1_top_5pct"].astype(bool)].copy()
    m0_tail_frame = holdout.loc[holdout["M0_top_5pct"].astype(bool)].copy()
    if m1_tail_frame.empty or m0_tail_frame.empty:
        raise ScreenBlocked(
            "blocked_insufficient_frozen_tail_support",
            "a frozen top-5% threshold selected no holdout rows",
        )
    m1_tail = v0.tail_metrics(m1_tail_frame, model="M1")
    m0_tail = v0.tail_metrics(m0_tail_frame, model="M0")
    tail_table = pd.DataFrame([m0_tail, m1_tail])
    comparison = v0.tail_comparison_metrics(m1_tail, m0_tail)
    overlap = v0.tail_overlap_metrics(
        holdout["M1_top_5pct"].to_numpy(bool),
        holdout["M0_top_5pct"].to_numpy(bool),
    )
    timing = movement_timing_metrics_with_optional_30m(m1_tail_frame)
    concentration = v0.concentration_table(holdout)
    write_csv(PRIMARY / "tail_metrics.csv", tail_table)
    write_csv(PRIMARY / "tail_comparison_metrics.csv", pd.DataFrame([comparison]))
    write_csv(PRIMARY / "tail_overlap_metrics.csv", pd.DataFrame([overlap]))
    write_csv(PRIMARY / "movement_timing_metrics.csv", timing)
    write_csv(PRIMARY / "concentration_metrics.csv", concentration)

    bootstrap = v0.bootstrap_statistics(holdout)
    write_csv(PRIMARY / "bootstrap_metrics.csv", bootstrap)
    nulls, null_evidence = v0.h0_null_refits(
        development,
        holdout,
        real_increment=increment,
    )
    if not null_evidence["row_id"].astype(str).equals(holdout["row_id"].astype(str)):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "H0-null evidence row ordering drifted",
        )
    null_columns = [column for column in null_evidence if column != "row_id"]
    for column in null_columns:
        holdout[column] = null_evidence[column].to_numpy()
    prediction_columns.extend(null_columns)
    write_parquet(
        PRIMARY / "holdout_predictions.parquet",
        holdout.loc[:, prediction_columns],
    )
    write_csv(PRIMARY / "intraday_h0_null_metrics.csv", nulls)

    joined_gate = v0.joined_support(holdout)
    tail_support = v0.frozen_tail_support(m1_tail_frame)
    positive_log_months, positive_mean_months, positive_median_months = v0.monthly_positive_counts(
        monthly
    )
    adverse = v0.adverse_checkpoint_groups(checkpoint)
    real_exceeds_null = bool(
        nulls["real_exceeds_null_log_loss_improvement"].sum() == 3
        or nulls["real_exceeds_null_brier_improvement"].sum() == 3
    )
    model_gate = v0.ModelGateInputs(
        log_loss_improvement=increment["log_loss_improvement"],
        brier_improvement=increment["brier_improvement"],
        auc_improvement=increment["auc_improvement"],
        average_precision_improvement=increment["average_precision_improvement"],
        bootstrap_80_log_loss_lower=v0.bootstrap_lower(
            bootstrap, "M1_minus_M0_log_loss_improvement"
        ),
        bootstrap_80_brier_lower=v0.bootstrap_lower(bootstrap, "M1_minus_M0_brier_improvement"),
        bootstrap_80_average_precision_lower=v0.bootstrap_lower(
            bootstrap, "M1_minus_M0_average_precision_improvement"
        ),
        positive_log_loss_months=positive_log_months,
        materially_adverse_checkpoint_groups=adverse,
        real_exceeds_all_nulls=real_exceeds_null,
        support_passed=bool(joined_gate["passed"]),
    )
    concentration_passed = bool(
        tail_support["passed"]
        and math.isfinite(float(m1_tail["top_5pct_positive_residual_contribution"]))
        and float(m1_tail["top_5pct_positive_residual_contribution"]) <= 0.50
    )
    tail_gate = v0.TailGateInputs(
        mean_iv_residual=float(m1_tail["mean_iv_residual"]),
        median_iv_residual=float(m1_tail["median_iv_residual"]),
        exceed_iv_rate=float(m1_tail["exceed_iv_rate"]),
        bootstrap_80_mean_lower=v0.bootstrap_lower(bootstrap, "M1_top_5pct_mean_iv_residual"),
        bootstrap_80_median_lower=v0.bootstrap_lower(bootstrap, "M1_top_5pct_median_iv_residual"),
        positive_mean_months=positive_mean_months,
        positive_median_months=positive_median_months,
        m1_minus_m0_mean=comparison["mean_iv_residual_difference"],
        bootstrap_80_difference_lower=v0.bootstrap_lower(
            bootstrap, "M1_minus_M0_top_5pct_mean_iv_residual"
        ),
        concentration_passed=concentration_passed,
        support_passed=bool(tail_support["passed"]),
    )
    decision = cast(dict[str, Any], v0.decide_experiment(model=model_gate, tail=tail_gate))
    decision.update(SAFETY_FLAGS)
    decision["download_resume_status"] = "supported"
    decision["holdout_options_coverage_status"] = "supported"
    decision["joined_holdout_support"] = joined_gate
    decision["frozen_tail_support"] = tail_support
    decision["coverage_preflight_passed_before_outcomes"] = True
    decision["pre_outcome_freeze_manifest_sha256"] = sha256_file(
        PRIMARY / "pre_outcome_freeze_manifest.json"
    )
    if not joined_gate["passed"]:
        decision["overall_decision"] = "blocked_insufficient_holdout_support"
        decision["minimal_model_status"] = "insufficient_support"
    elif not tail_support["passed"]:
        decision["overall_decision"] = "blocked_insufficient_frozen_tail_support"
        decision["frozen_top_5pct_status"] = "insufficient_support"
        decision["options_only_tail_comparison_status"] = "insufficient_support"
        decision["movement_timing_status"] = "insufficient_support"
    assert_v01_safety_flags(decision)
    write_json(PRIMARY / "decision.json", decision)

    protected = {
        **SAFETY_FLAGS,
        "protected_start": "2026-01-01",
        "protected_underlying_rows_materialised": 0,
        "protected_option_rows_materialised": 0,
        "protected_or_unauthorised_records_materialised": 0,
        "protected_option_records_rejected": int(download["new_2026_or_later_records_rejected"]),
        "holdout_rows_outside_authorized_dates": 0,
        "same_day_or_future_option_observations": int(preflight["same_day_or_future_observations"]),
        "non_exact_previous_session_option_observations": int(
            preflight["non_exact_previous_session_observations"]
        ),
        "passed": True,
    }
    write_json(PRIMARY / "protected_boundary_audit.json", protected)
    authorization = {
        **SAFETY_FLAGS,
        "authorized_calendar_range": ["2025-09-01", "2025-12-31"],
        "actual_xnys_sessions": int(preflight["planned_session_count"]),
        "joined_outcome_sessions": int(holdout["session"].nunique()),
        "protected_start": "2026-01-01",
        "other_protected_period_opened": False,
        "economic_option_pnl_outcome_opened": False,
        "coverage_preflight_passed_before_outcomes": True,
        "holdout_outcomes_read_before_freeze": False,
        "holdout_outcomes_opened": True,
        "threshold_artifact_sha256": sha256_file(PRIMARY / "frozen_tail_thresholds.json"),
        "passed": True,
    }
    write_json(PRIMARY / "holdout_data_authorisation.json", authorization)
    source_manifest = {
        **SAFETY_FLAGS,
        "starting_branch": "agent/minimal-intraday-iv-excess-holdout-v0",
        "starting_sha": STARTING_SHA,
        "final_branch": "agent/minimal-intraday-iv-excess-holdout-v01",
        "strict_continuation": True,
        "v0_artifacts_modified": False,
        "historical_panel_path": str(v0.historical_panel_path()),
        "historical_panel_sha256": sha256_file(v0.historical_panel_path()),
        "h0_feature_surface": h0_manifest,
        "state_cache_path": str(state_cache),
        "state_cache_sha256": sha256_file(state_cache),
        "options_cache_path": str(options_cache),
        "options_cache_sha256": sha256_file(options_cache),
        "resume_download_manifest_sha256": sha256_file(PRIMARY / "resume_download_manifest.json"),
        "coverage_preflight_sha256": sha256_file(PRIMARY / "holdout_coverage_preflight.json"),
        "pre_outcome_freeze_manifest_sha256": sha256_file(
            PRIMARY / "pre_outcome_freeze_manifest.json"
        ),
        "raw_vendor_data_committed": False,
        "canonical_vendor_data_committed": False,
        "protected_rows_materialised": 0,
    }
    write_json(PRIMARY / "source_manifest.json", source_manifest)
    write_json(
        PRIMARY / "lightweight_audit.json",
        {
            **SAFETY_FLAGS,
            "status": "pending_independent_audit",
            "runner_self_checks_passed": True,
            "resume_scope_passed": True,
            "coverage_preflight_passed_before_outcomes": True,
            "historical_reconstruction_passed": True,
            "protected_boundary_passed": True,
            "exact_previous_session_chronology_passed": True,
        },
    )

    v0.create_plots(holdout, m0_tail_frame, m1_tail_frame, timing)
    report = build_report(
        decision=decision,
        download=download,
        preflight=preflight,
        reconstruction=reconstruction,
        thresholds=thresholds,
        model_metrics=model_metrics,
        increment=increment,
        tail_table=tail_table,
        comparison=comparison,
        overlap=overlap,
        timing=timing,
        bootstrap=bootstrap,
        nulls=nulls,
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    return {
        "decision": decision,
        "rows": len(holdout),
        "sessions": int(holdout["session"].nunique()),
        "stocks": int(holdout["symbol"].nunique()),
        "M0_threshold": thresholds["M0_top_5_percent_threshold"],
        "M1_threshold": thresholds["M1_top_5_percent_threshold"],
        "M1_tail": m1_tail,
        "increment": increment,
    }


def write_blocked_decision(error: ScreenBlocked) -> None:
    decision = {
        **SAFETY_FLAGS,
        "overall_decision": error.decision,
        "minimal_model_status": "blocked",
        "frozen_top_5pct_status": "blocked",
        "options_only_tail_comparison_status": "blocked",
        "movement_timing_status": "blocked",
        "holdout_options_coverage_status": "blocked",
        "download_resume_status": (
            "supported" if (PRIMARY / "resume_download_manifest.json").is_file() else "blocked"
        ),
        "blocker_detail": error.detail,
    }
    write_json(PRIMARY / "decision.json", decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    parser.add_argument("--options-cache", type=Path, default=DEFAULT_OPTIONS_CACHE)
    parser.add_argument("--stock-cache", type=Path, default=DEFAULT_STOCK_CACHE)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_STATE_CACHE)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        result = execute(
            provider_root=arguments.provider_root,
            options_cache=arguments.options_cache,
            stock_cache=arguments.stock_cache,
            state_cache=arguments.state_cache,
        )
    except ScreenBlocked as error:
        write_blocked_decision(error)
        print(error.decision)
        return 2
    except Exception as error:
        decision = getattr(error, "decision", "blocked_reproducibility_or_audit_failure")
        detail = getattr(error, "detail", f"{type(error).__name__}: {error}")
        blocked = ScreenBlocked(str(decision), str(detail))
        write_blocked_decision(blocked)
        print(blocked.decision)
        return 2
    print(
        f"{result['decision']['overall_decision']}: {result['rows']} rows, "
        f"{result['sessions']} sessions, {result['stocks']} stocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
