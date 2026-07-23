#!/usr/bin/env python3
"""Run the bounded Daily Stock + Front-Options Context Quick Screen V0.1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "cross-market-context"
    / "20260723-daily-stock-options-regime-context-v0"
)
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
PREDECESSOR_RUNNER = PREDECESSOR_DIR / "run_screen_v0.py"
DENSE_PANEL = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
    / "dense_advance_panel.parquet"
)
TRACE_PANEL = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-hazard-quick-v0"
    / "artifacts"
    / "primary"
    / "causal_state_trace.parquet"
)
DEFAULT_PROVIDER_ROOT = (
    Path.home() / "StockerLocal" / "data" / "processed" / "source=eodhd" / "instrument_type=stock"
)
STARTING_BRANCH = "agent/daily-stock-options-regime-context-quick-v0"
STARTING_SHA = "a0e0a74658663c36d7bd5d5754490c3f17d9bf7a"
FINAL_BRANCH = "agent/daily-stock-front-options-context-v01"

for package in ("stocker_research", "stocker_data"):
    package_path = str(REPO_ROOT / "packages" / package / "src")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from stocker_research.broad_conflict_advance_hazard_v02 import (  # noqa: E402
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    ROUTE_FEATURES,
)
from stocker_research.daily_soft_regimes_v0 import (  # noqa: E402
    DAILY_STOCK_DIMENSIONS,
    FrozenDimensionParameters,
    FrozenSoftRegime,
    RobustValueScale,
    apply_stock_dimensions,
)
from stocker_research.daily_stock_front_options_context_v01 import (  # noqa: E402
    ASSESSMENT_END,
    DEVELOPMENT_START,
    FRONT_MISMATCH_FEATURES,
    FROZEN_COHORT,
    PROTECTED_START,
    SAFETY_FLAGS,
    MeanScale,
    add_front_mismatch_features,
    assert_safety_flags,
    choose_overall_decision,
    fit_front_mismatch_standardization,
    iv_excess_15m_frame,
    prepare_front_options_raw,
    route_state_iv_metrics,
    weighted_quantile,
)
from stocker_research.daily_stock_options_context_v0 import (  # noqa: E402
    permute_bundle_within_slates,
    select_daily_options_surface,
)
from stocker_research.front_options_soft_regimes_v01 import (  # noqa: E402
    FRONT_OPTIONS_CANONICAL_DIMENSIONS,
    FRONT_OPTIONS_DIMENSIONS,
    FRONT_OPTIONS_MISSING_INDICATORS,
    FRONT_OPTIONS_RAW_FEATURES,
    apply_front_options_dimensions,
    apply_front_options_regime,
    apply_serialized_diag_regime,
    fit_front_options_dimension_parameters,
    fit_front_options_regime,
    front_options_regime_mapping,
)
from stocker_research.stock_options_cross_market_quick_v0 import (  # noqa: E402
    FrozenCrossMarketModel,
    binary_metrics,
    fit_cross_market_model,
    fixed_session_bootstrap_multiplicities,
    manual_model_prediction,
    probability_quantile_boundaries,
    reconstruct_clean_structural_panel,
)

CHECKPOINT_FEATURES = tuple(f"checkpoint_{value}" for value in DENSE_CHECKPOINTS)
H0_NON_CLOCK_FEATURES = tuple(
    value for value in DENSE_H0_FEATURES if value not in CHECKPOINT_FEATURES
)
STOCK_REGIME_FEATURES = (
    *(f"daily_stock_regime_p_{value}" for value in range(4)),
    "daily_stock_regime_entropy",
    "daily_stock_regime_margin",
)
FRONT_REGIME_FEATURES = (
    *(f"front_options_regime_p_{value}" for value in range(4)),
    "front_options_regime_entropy",
    "front_options_regime_margin",
)
STOCK_CONTEXT_FEATURES = (*DAILY_STOCK_DIMENSIONS, *STOCK_REGIME_FEATURES)
FRONT_CONTEXT_FEATURES = (
    *FRONT_OPTIONS_DIMENSIONS,
    *FRONT_REGIME_FEATURES,
    *FRONT_OPTIONS_MISSING_INDICATORS,
)
A0_FEATURES = (*DENSE_H0_FEATURES, *ROUTE_FEATURES)
A1_FEATURES = (*A0_FEATURES, *STOCK_CONTEXT_FEATURES)
B0_FEATURES = A1_FEATURES
B1_FEATURES = (*B0_FEATURES, *FRONT_CONTEXT_FEATURES, *FRONT_MISMATCH_FEATURES)
C0_FEATURES = (*FRONT_CONTEXT_FEATURES, *CHECKPOINT_FEATURES)
C1_FEATURES = (
    *C0_FEATURES,
    *STOCK_CONTEXT_FEATURES,
    *H0_NON_CLOCK_FEATURES,
    *ROUTE_FEATURES,
    *FRONT_MISMATCH_FEATURES,
)

FRONT_OPTIONS_NULL_SEEDS = (20260723, 20260724, 20260725)
STOCK_STRUCTURE_NULL_SEEDS = (20260726, 20260727, 20260728)
TARGET_COMPLETION = "registered_completion_clean_bars_2_or_3"
TARGET_IV_EXCESS = "movement_exceeds_prior_close_iv_15m"


@dataclass(frozen=True)
class DevelopmentCutoffs:
    """Development-frozen subgroup boundaries."""

    mismatch_compression_vs_front_iv: float
    mismatch_complacent_broad_conflict: float


class BranchBlocker(RuntimeError):
    """A fail-closed blocker limited to one independent branch."""

    def __init__(self, branch: str, status: str, detail: str) -> None:
        super().__init__(detail)
        self.branch = branch
        self.status = status
        self.detail = detail


REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "structural_panel_reconstruction.json",
    "daily_stock_reconstruction.json",
    "front_options_pair_reconstruction.json",
    "front_options_raw_features.parquet",
    "front_options_dimensions.parquet",
    "front_options_feature_manifest.json",
    "front_options_regime_mapping.json",
    "front_options_regime_diagnostics.csv",
    "front_options_coverage.csv",
    "front_options_cross_market_panel.parquet",
    "mismatch_feature_manifest.json",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "branch_a_metrics.csv",
    "branch_a_monthly_metrics.csv",
    "branch_a_regime_metrics.csv",
    "branch_b_metrics.csv",
    "branch_b_monthly_metrics.csv",
    "branch_b_regime_metrics.csv",
    "branch_c_metrics.csv",
    "branch_c_monthly_metrics.csv",
    "branch_c_route_state_metrics.csv",
    "bootstrap_metrics.csv",
    "front_options_null_metrics.csv",
    "stock_structure_null_metrics.csv",
    "concentration_metrics.csv",
    "back_expiry_schema_preflight.json",
    "back_expiry_future_request_plan.json",
    "decision.json",
    "lightweight_audit.json",
    "determinism_check.json",
    "report.md",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predecessor_runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "_daily_context_predecessor_v0",
        PREDECESSOR_RUNNER,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the predecessor runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_contract() -> dict[str, Any]:
    value = read_json(EXPERIMENT_DIR / "contract.json")
    assert_safety_flags(value)
    if (
        value.get("development_start") != "2024-01-01"
        or value.get("development_end") != "2024-12-31"
    ):
        raise ValueError("development dates differ from the V0.1 contract")
    if value.get("assessment_end") != "2025-08-22":
        raise ValueError("assessment dates differ from the V0.1 contract")
    return value


def load_structural_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    dense = pd.read_parquet(DENSE_PANEL)
    structural, reconstruction_value = reconstruct_clean_structural_panel(dense)
    structural["checkpoint_group"] = np.select(
        [
            structural["checkpoint"].astype(int).between(6, 14),
            structural["checkpoint"].astype(int).between(16, 24),
            structural["checkpoint"].astype(int).between(26, 34),
        ],
        ["early_6_14", "middle_16_24", "late_26_34"],
        default="invalid",
    )
    if structural["checkpoint_group"].eq("invalid").any():
        raise BranchBlocker(
            "all",
            "blocked_structural_panel_reconstruction_failure",
            "a checkpoint is outside the frozen reporting groups",
        )
    reconstruction = {**reconstruction_value, **SAFETY_FLAGS}
    required_checkpoints = tuple(int(value) for value in DENSE_CHECKPOINTS)
    observed_checkpoints = tuple(sorted(structural["checkpoint"].astype(int).unique()))
    passed = bool(
        int(reconstruction["row_identity_mismatches"]) == 0
        and int(reconstruction["route_state_mismatches"]) == 0
        and int(reconstruction["target_mismatches"]) == 0
        and float(reconstruction["maximum_shared_feature_difference"]) <= 1e-12
        and observed_checkpoints == required_checkpoints
    )
    reconstruction.update(
        {
            "movement_outcomes_reconstructed_later_from_frozen_underlying_bars": True,
            "passed": passed,
        }
    )
    if not passed:
        raise BranchBlocker(
            "all",
            "blocked_structural_panel_reconstruction_failure",
            "the frozen dense clean-advance panel did not reconstruct exactly",
        )
    return structural, reconstruction


def _row_identity_mismatches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
) -> int:
    left_keys = left.loc[:, list(columns)].astype(str).agg("|".join, axis=1).tolist()
    right_keys = right.loc[:, list(columns)].astype(str).agg("|".join, axis=1).tolist()
    return abs(len(left_keys) - len(right_keys)) + sum(
        left_value != right_value
        for left_value, right_value in zip(left_keys, right_keys, strict=False)
    )


def _maximum_numeric_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    if len(left) != len(right):
        return math.inf
    maximum = 0.0
    for column in columns:
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        both_nan = np.isnan(left_values) & np.isnan(right_values)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if bool((~both_nan & ~finite).any()):
            return math.inf
        if bool(finite.any()):
            maximum = max(
                maximum,
                float(np.max(np.abs(left_values[finite] - right_values[finite]))),
            )
    return maximum


def reconstruct_daily_stock_context() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path = PREDECESSOR_PRIMARY / "daily_stock_raw_features.parquet"
    dimensions_path = PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
    manifest_path = PREDECESSOR_PRIMARY / "daily_stock_feature_manifest.json"
    mapping_path = PREDECESSOR_PRIMARY / "daily_stock_regime_mapping.json"
    raw = pd.read_parquet(raw_path)
    frozen = pd.read_parquet(dimensions_path).sort_values(["symbol", "session"], kind="mergesort")
    manifest = read_json(manifest_path)
    mapping = read_json(mapping_path)
    scales_value = cast(Mapping[str, Mapping[str, Any]], manifest["scales"])
    parameters = FrozenDimensionParameters(
        kind="daily_stock",
        scales={
            name: RobustValueScale(
                center=float(value["center"]),
                scale=float(value["scale"]),
            )
            for name, value in scales_value.items()
        },
        imputation_medians={},
    )
    rebuilt_all = apply_stock_dimensions(raw, parameters)
    rebuilt = rebuilt_all.loc[
        rebuilt_all.loc[:, list(DAILY_STOCK_DIMENSIONS)].notna().all(axis=1)
    ].sort_values(["symbol", "session"], kind="mergesort")
    identity_mismatches = _row_identity_mismatches(
        frozen,
        rebuilt,
        ("symbol", "session", "stock_information_date"),
    )
    maximum_dimension_difference = _maximum_numeric_difference(
        frozen,
        rebuilt,
        DAILY_STOCK_DIMENSIONS,
    )
    reassigned = apply_serialized_diag_regime(
        rebuilt,
        mapping,
        prefix="daily_stock_regime",
    )
    posterior_columns = (
        *(f"daily_stock_regime_p_{value}" for value in range(4)),
        "daily_stock_regime_entropy",
        "daily_stock_regime_top_probability",
        "daily_stock_regime_margin",
        "daily_stock_regime_mahalanobis_to_nearest_centroid",
    )
    maximum_posterior_difference = _maximum_numeric_difference(
        frozen,
        reassigned,
        posterior_columns,
    )
    hard_mismatches = int(
        np.sum(
            frozen["daily_stock_regime"].to_numpy(int)
            != reassigned["daily_stock_regime"].to_numpy(int)
        )
    )
    canonical_dimensions = tuple(str(value) for value in mapping["canonical_dimensions"])
    centroids = cast(Sequence[Mapping[str, Any]], mapping["canonical_centroids"])
    ordering_keys = [
        tuple(float(centroid[column]) for column in canonical_dimensions) for centroid in centroids
    ]
    mapping_mismatches = int(ordering_keys != sorted(ordering_keys)) + hard_mismatches
    assessment = frozen.loc[frozen["period"].astype(str).eq("assessment")]
    predecessor_support = cast(Mapping[str, Any], manifest["support"])
    support = {
        "assessment_rows": len(assessment),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_months": int(pd.to_datetime(assessment["session"]).dt.to_period("M").nunique()),
        "daily_stock_feature_retention": float(
            predecessor_support["daily_stock_feature_retention"]
        ),
    }
    support["passed"] = bool(
        support["assessment_stocks"] == 20
        and support["assessment_sessions"] >= 140
        and support["assessment_months"] == 8
        and support["daily_stock_feature_retention"] >= 0.95
    )
    regime_support_value: dict[int, dict[str, Any]] = {}
    for regime in range(4):
        hard = assessment.loc[assessment["daily_stock_regime"].astype(int).eq(regime)]
        mass = float(assessment[f"daily_stock_regime_p_{regime}"].mean())
        regime_support_value[regime] = {
            "posterior_mass": mass,
            "hard_rows": len(hard),
            "stocks": int(hard["symbol"].nunique()),
            "sessions": int(hard["session"].nunique()),
            "months": int(pd.to_datetime(hard["session"]).dt.to_period("M").nunique()),
            "supported": bool(
                mass >= 0.05
                and hard["symbol"].nunique() >= 8
                and pd.to_datetime(hard["session"]).dt.to_period("M").nunique() >= 4
            ),
        }
    reconstruction = {
        **SAFETY_FLAGS,
        "predecessor_experiment": "Daily Stock × Options Regime Context Quick Screen V0",
        "predecessor_daily_stock_raw_sha256": sha256_file(raw_path),
        "predecessor_daily_stock_dimensions_sha256": sha256_file(dimensions_path),
        "predecessor_daily_stock_mapping_sha256": sha256_file(mapping_path),
        "rows": len(frozen),
        "row_identity_mismatches": identity_mismatches,
        "maximum_daily_stock_dimension_difference": maximum_dimension_difference,
        "maximum_daily_stock_posterior_difference": maximum_posterior_difference,
        "daily_stock_regime_mapping_mismatches": mapping_mismatches,
        "frozen_model_reused_without_refit": True,
        "support": support,
        "regime_support": regime_support_value,
        "canonical_centroids": mapping["canonical_centroids"],
        "canonical_to_original": mapping["canonical_to_original"],
    }
    reconstruction["passed"] = bool(
        identity_mismatches == 0
        and maximum_dimension_difference <= 1e-12
        and maximum_posterior_difference <= 1e-12
        and mapping_mismatches == 0
        and support["passed"]
    )
    if not reconstruction["passed"]:
        raise BranchBlocker(
            "daily_stock",
            "blocked_daily_stock_reconstruction_failure",
            f"daily stock reconstruction failed: {reconstruction}",
        )
    return frozen.reset_index(drop=True), reconstruction


def _dimension_manifest(
    parameters: FrozenDimensionParameters,
    *,
    front_raw: pd.DataFrame,
) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "kind": parameters.kind,
        "fitted_period": parameters.fitted_period,
        "raw_features": list(FRONT_OPTIONS_RAW_FEATURES),
        "dimensions": list(FRONT_OPTIONS_DIMENSIONS),
        "missing_indicators": list(FRONT_OPTIONS_MISSING_INDICATORS),
        "forbidden_features": ["front_term_urgency", "back_atm_iv", "term_structure"],
        "scales": {name: asdict(value) for name, value in parameters.scales.items()},
        "imputation_medians": dict(parameters.imputation_medians),
        "development_rows": int(front_raw["period"].eq("development").sum()),
        "assessment_rows": int(front_raw["period"].eq("assessment").sum()),
        "development_only_scaling_and_imputation": True,
    }


def front_regime_diagnostics(
    frame: pd.DataFrame,
    fitted: FrozenSoftRegime,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prefix = "front_options_regime"
    for period, period_frame in frame.groupby("period", sort=True, observed=True):
        months = pd.to_datetime(period_frame["session"]).dt.to_period("M").astype(str)
        for regime in range(4):
            posterior = pd.to_numeric(period_frame[f"{prefix}_p_{regime}"], errors="raise")
            hard = period_frame.loc[period_frame[prefix].astype(int).eq(regime)]
            stock_mass = (
                period_frame.assign(_mass=posterior).groupby("symbol", observed=True)["_mass"].sum()
            )
            month_mass = (
                period_frame.assign(_mass=posterior, _month=months)
                .groupby("_month", observed=True)["_mass"]
                .sum()
            )
            row: dict[str, Any] = {
                "diagnostic_type": "regime_summary",
                "period": period,
                "regime": regime,
                "rows": len(period_frame),
                "posterior_mass": float(posterior.sum() / len(period_frame)),
                "hard_top_regime_rows": len(hard),
                "stocks": int(hard["symbol"].nunique()),
                "sessions": int(hard["session"].nunique()),
                "months": int(pd.to_datetime(hard["session"]).dt.to_period("M").nunique()),
                "maximum_stock_share": (
                    float(stock_mass.max() / stock_mass.sum())
                    if float(stock_mass.sum()) > 0.0
                    else math.nan
                ),
                "maximum_month_share": (
                    float(month_mass.max() / month_mass.sum())
                    if float(month_mass.sum()) > 0.0
                    else math.nan
                ),
                "entropy_mean": float(period_frame[f"{prefix}_entropy"].mean()),
                "entropy_q10": float(period_frame[f"{prefix}_entropy"].quantile(0.10)),
                "entropy_q50": float(period_frame[f"{prefix}_entropy"].quantile(0.50)),
                "entropy_q90": float(period_frame[f"{prefix}_entropy"].quantile(0.90)),
                "margin_mean": float(period_frame[f"{prefix}_margin"].mean()),
                "margin_q10": float(period_frame[f"{prefix}_margin"].quantile(0.10)),
                "margin_q50": float(period_frame[f"{prefix}_margin"].quantile(0.50)),
                "margin_q90": float(period_frame[f"{prefix}_margin"].quantile(0.90)),
                "mahalanobis_mean": float(
                    period_frame[f"{prefix}_mahalanobis_to_nearest_centroid"].mean()
                ),
                "mahalanobis_q90": float(
                    period_frame[f"{prefix}_mahalanobis_to_nearest_centroid"].quantile(0.90)
                ),
            }
            for dimension in FRONT_OPTIONS_DIMENSIONS:
                row[f"mean_{dimension}"] = float(hard[dimension].mean()) if len(hard) else math.nan
            rows.append(row)
        for symbol, stock_frame in period_frame.groupby("symbol", sort=True, observed=True):
            rows.append(
                {
                    "diagnostic_type": "support_by_stock",
                    "period": period,
                    "group": symbol,
                    **{
                        f"posterior_mass_regime_{regime}": float(
                            stock_frame[f"{prefix}_p_{regime}"].mean()
                        )
                        for regime in range(4)
                    },
                }
            )
        for month, month_frame in period_frame.assign(_month=months).groupby(
            "_month", sort=True, observed=True
        ):
            rows.append(
                {
                    "diagnostic_type": "support_by_month",
                    "period": period,
                    "group": str(month),
                    **{
                        f"posterior_mass_regime_{regime}": float(
                            month_frame[f"{prefix}_p_{regime}"].mean()
                        )
                        for regime in range(4)
                    },
                }
            )
    development = frame.loc[frame["period"].eq("development")]
    assessment = frame.loc[frame["period"].eq("assessment")]
    for dimension in FRONT_OPTIONS_DIMENSIONS:
        rows.append(
            {
                "diagnostic_type": "development_assessment_dimension_drift",
                "group": dimension,
                "development_mean": float(development[dimension].mean()),
                "assessment_mean": float(assessment[dimension].mean()),
                "mean_difference": float(
                    assessment[dimension].mean() - development[dimension].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def front_regime_support(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    assessment = frame.loc[frame["period"].eq("assessment")]
    evidence: dict[int, dict[str, Any]] = {}
    for regime in range(4):
        hard = assessment.loc[assessment["front_options_regime"].astype(int).eq(regime)]
        mass = float(assessment[f"front_options_regime_p_{regime}"].mean())
        evidence[regime] = {
            "posterior_mass": mass,
            "hard_rows": len(hard),
            "stocks": int(hard["symbol"].nunique()),
            "months": int(pd.to_datetime(hard["session"]).dt.to_period("M").nunique()),
            "supported": bool(
                mass >= 0.05
                and hard["symbol"].nunique() >= 8
                and pd.to_datetime(hard["session"]).dt.to_period("M").nunique() >= 4
            ),
        }
    return evidence


def load_repaired_exact_date_options_cache() -> tuple[pd.DataFrame, Path, str]:
    """Load only pre-boundary rows from the repaired canonical options cache."""

    predecessor_source = read_json(PREDECESSOR_PRIMARY / "source_manifest.json")
    sources = cast(Mapping[str, Any], predecessor_source["sources"])
    cache_path = Path(str(sources["repaired_exact_date_options_cache"])).resolve()
    expected_sha256 = str(sources["repaired_exact_date_options_cache_sha256"])
    if not cache_path.is_file():
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            f"repaired exact-date cache is unavailable: {cache_path}",
        )
    observed_sha256 = sha256_file(cache_path)
    if observed_sha256 != expected_sha256:
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            "repaired exact-date cache hash differs from the frozen predecessor source",
        )
    required = (
        "underlying_symbol",
        "trade_date",
        "expiration_date",
        "option_type",
        "strike",
        "contract_id",
        "bid",
        "ask",
        "midpoint",
        "implied_volatility",
        "delta",
        "open_interest",
        "request_id",
    )
    cache = pd.read_parquet(
        cache_path,
        columns=list(required),
        filters=[("trade_date", "<", PROTECTED_START)],
    )
    if missing := sorted(set(required).difference(cache.columns)):
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            f"repaired options cache schema is missing: {missing}",
        )
    cache["trade_date"] = pd.to_datetime(cache["trade_date"], errors="raise").dt.date
    if cache["trade_date"].ge(PROTECTED_START).any():
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            "protected option observations entered the cache reconstruction",
        )
    cache["underlying_symbol"] = cache["underlying_symbol"].astype(str)
    cache = cache.loc[cache["underlying_symbol"].isin(FROZEN_COHORT)].copy()
    return cache, cache_path, observed_sha256


def reconstruct_front_options_raw() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild every selected front pair from the repaired cached chains."""

    stock_raw = pd.read_parquet(PREDECESSOR_PRIMARY / "daily_stock_raw_features.parquet")
    cache, cache_path, cache_sha256 = load_repaired_exact_date_options_cache()
    groups = {
        (str(symbol), cast(date, observation_date)): group.copy()
        for (symbol, observation_date), group in cache.groupby(
            ["underlying_symbol", "trade_date"],
            sort=False,
            observed=True,
        )
    }
    rows: list[dict[str, Any]] = []
    reused_records = 0
    for source in stock_raw.itertuples(index=False):
        information_date = date.fromisoformat(str(source.stock_information_date))
        base = {
            "symbol": str(source.symbol),
            "session": str(source.session),
            "period": str(source.period),
            "required_options_date": information_date.isoformat(),
        }
        chain = groups.get((str(source.symbol), information_date))
        realised = float(source.realised_volatility_20d)
        previous_close = float(source.unadjusted_close)
        if chain is None:
            result: dict[str, object] = {
                "pair_available": False,
                "pair_reason": "missing_exact_chain",
            }
        elif not math.isfinite(realised) or not math.isfinite(previous_close):
            result = {
                "pair_available": False,
                "pair_reason": "missing_daily_stock_volatility_or_close",
            }
        else:
            reused_records += len(chain)
            try:
                result = select_daily_options_surface(
                    chain,
                    previous_close=previous_close,
                    realised_volatility_20d=realised,
                )
            except ValueError as error:
                raise BranchBlocker(
                    "front_options",
                    "blocked_front_options_regime_failure",
                    f"cached-chain pair selection failed for {source.symbol}/{information_date}: "
                    f"{error}",
                ) from error
        rows.append({**base, **result})
    rebuilt = prepare_front_options_raw(pd.DataFrame(rows))

    predecessor_path = PREDECESSOR_PRIMARY / "daily_options_raw_features.parquet"
    predecessor = prepare_front_options_raw(pd.read_parquet(predecessor_path))
    identity_columns = (
        "symbol",
        "session",
        "options_observation_date",
        "front_expiration_date",
        "front_strike",
        "front_call_contract_id",
        "front_put_contract_id",
        "skew_put_contract_id",
        "skew_call_contract_id",
    )
    contract_mismatches = _row_identity_mismatches(predecessor, rebuilt, identity_columns)
    maximum_feature_difference = _maximum_numeric_difference(
        predecessor,
        rebuilt,
        (*FRONT_OPTIONS_RAW_FEATURES, *FRONT_OPTIONS_MISSING_INDICATORS),
    )
    reconstruction = {
        **SAFETY_FLAGS,
        "source_experiment": "Daily Stock × Options Regime Context Quick Screen V0",
        "source_sha256": sha256_file(predecessor_path),
        "repaired_exact_date_cache": str(cache_path),
        "repaired_exact_date_cache_sha256": cache_sha256,
        "cache_rows_loaded_before_protected_boundary": len(cache),
        "cache_records_reused": reused_records,
        "selected_pairs": len(rebuilt),
        "development_selected_pairs": int(rebuilt["period"].eq("development").sum()),
        "assessment_selected_pairs": int(rebuilt["period"].eq("assessment").sum()),
        "selected_contract_mismatches": contract_mismatches,
        "maximum_front_raw_feature_difference": maximum_feature_difference,
        "exact_previous_session_rows": int(
            rebuilt["required_options_date"]
            .astype(str)
            .eq(rebuilt["options_observation_date"].astype(str))
            .sum()
        ),
        "same_day_or_future_observations": int(
            pd.to_datetime(rebuilt["options_observation_date"])
            .ge(pd.to_datetime(rebuilt["session"]))
            .sum()
        ),
        "selection_rebuilt_from_cached_chains": True,
        "back_expiry_required": False,
        "front_term_urgency_referenced": False,
    }
    reconstruction["passed"] = bool(
        contract_mismatches == 0
        and maximum_feature_difference <= 1e-12
        and reconstruction["exact_previous_session_rows"] == len(rebuilt)
        and reconstruction["same_day_or_future_observations"] == 0
    )
    if not reconstruction["passed"]:
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            f"cached-chain front-pair reconstruction failed: {reconstruction}",
        )
    return rebuilt, reconstruction


def build_front_options_context() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    FrozenDimensionParameters,
    FrozenSoftRegime,
    dict[str, Any],
    dict[str, Any],
]:
    raw, pair_reconstruction = reconstruct_front_options_raw()
    if raw.duplicated(["symbol", "session"]).any():
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            "front-options context is not one row per stock-session",
        )
    development = raw.loc[raw["period"].eq("development")].copy()
    try:
        parameters = fit_front_options_dimension_parameters(development)
        dimensions = apply_front_options_dimensions(raw, parameters)
        fitted = fit_front_options_regime(
            dimensions.loc[dimensions["period"].eq("development")].copy()
        )
        dimensions = apply_front_options_regime(dimensions, fitted)
    except (RuntimeError, ValueError) as error:
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            f"front-options regime failed: {type(error).__name__}: {error}",
        ) from error
    support = front_regime_support(dimensions)
    regime_mapping = front_options_regime_mapping(
        fitted,
        safety_flags=SAFETY_FLAGS,
    )
    regime_mapping["support"] = support
    return (
        raw,
        dimensions,
        parameters,
        fitted,
        pair_reconstruction,
        regime_mapping,
    )


def join_daily_stock_panel(
    structural: pd.DataFrame,
    stock_context: pd.DataFrame,
) -> pd.DataFrame:
    panel = structural.merge(
        stock_context,
        on=["symbol", "session", "period"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_daily"),
    )
    return panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def join_front_options_panel(
    daily_stock_panel: pd.DataFrame,
    front_context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, MeanScale], DevelopmentCutoffs]:
    panel = daily_stock_panel.merge(
        front_context,
        on=["symbol", "session", "period"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_front"),
    )
    if (
        not pd.to_datetime(panel["options_observation_date"])
        .lt(pd.to_datetime(panel["session"]))
        .all()
    ):
        raise BranchBlocker(
            "front_options",
            "blocked_front_options_regime_failure",
            "same-day or future front-options data entered the joined panel",
        )
    development = panel.loc[panel["period"].eq("development")]
    standardization = fit_front_mismatch_standardization(development)
    panel = add_front_mismatch_features(panel, standardization)
    cutoffs = DevelopmentCutoffs(
        mismatch_compression_vs_front_iv=float(
            panel.loc[
                panel["period"].eq("development"),
                "mismatch_compression_vs_front_iv",
            ].median()
        ),
        mismatch_complacent_broad_conflict=float(
            panel.loc[
                panel["period"].eq("development"),
                "mismatch_complacent_broad_conflict",
            ].median()
        ),
    )
    return (
        panel.sort_values("row_id", kind="mergesort").reset_index(drop=True),
        standardization,
        cutoffs,
    )


def attach_15m_movement(
    panel: pd.DataFrame,
    full_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = full_bars.loc[
        :,
        ["symbol", "session", "bar_ordinal", "open", "close"],
    ].copy()
    working = panel.copy()
    working["_entry_bar_ordinal"] = working["checkpoint_bar_ordinal_zero_based"].astype(int) + 1
    working["_third_future_bar_ordinal"] = (
        working["checkpoint_bar_ordinal_zero_based"].astype(int) + 3
    )
    entry = lookup.rename(
        columns={
            "bar_ordinal": "_entry_bar_ordinal",
            "open": "entry_price",
            "close": "_entry_close_unused",
        }
    )
    third = lookup.rename(
        columns={
            "bar_ordinal": "_third_future_bar_ordinal",
            "open": "_third_open_unused",
            "close": "close_15m",
        }
    )
    working = working.merge(
        entry.loc[:, ["symbol", "session", "_entry_bar_ordinal", "entry_price"]],
        on=["symbol", "session", "_entry_bar_ordinal"],
        how="left",
        validate="many_to_one",
    ).merge(
        third.loc[
            :,
            ["symbol", "session", "_third_future_bar_ordinal", "close_15m"],
        ],
        on=["symbol", "session", "_third_future_bar_ordinal"],
        how="left",
        validate="many_to_one",
    )
    finite = np.isfinite(
        working.loc[:, ["entry_price", "close_15m", "atm_iv"]].to_numpy(float)
    ).all(axis=1)
    positive = working.loc[:, ["entry_price", "close_15m", "atm_iv"]].gt(0.0).all(axis=1)
    valid = finite & positive
    dropped = int((~valid).sum())
    working = working.loc[valid].copy()
    outcomes = iv_excess_15m_frame(
        entry_price=working["entry_price"].to_numpy(float),
        close_15m=working["close_15m"].to_numpy(float),
        atm_iv=working["atm_iv"].to_numpy(float),
    )
    outcomes.index = working.index
    for column in (
        "absolute_log_return_15m",
        "iv_sigma_15m",
        "iv_expected_absolute_15m",
        TARGET_IV_EXCESS,
        "iv_absolute_residual_15m",
    ):
        working[column] = outcomes[column]
    working = working.drop(columns=["_entry_bar_ordinal", "_third_future_bar_ordinal"])
    audit = {
        "input_joined_rows": len(panel),
        "valid_15m_outcome_rows": len(working),
        "missing_or_invalid_outcome_rows": dropped,
        "entry_price_definition": "open of first completed five-minute bar after checkpoint",
        "exit_price_definition": "close of third future completed five-minute bar",
        "target_definition": TARGET_IV_EXCESS,
    }
    return working.reset_index(drop=True), audit


@dataclass
class ModelPairResult:
    """One independently fitted two-model branch."""

    branch: str
    target: str
    development: pd.DataFrame
    assessment: pd.DataFrame
    models: dict[str, FrozenCrossMarketModel]
    boundaries: dict[str, dict[str, float]]


def _weighted_stock_share(frame: pd.DataFrame) -> float:
    assessment = frame.loc[frame["period"].eq("assessment")]
    grouped = assessment.groupby("symbol", observed=True)["row_weight"].sum()
    return float(grouped.max() / grouped.sum())


def branch_support(
    panel: pd.DataFrame,
    *,
    branch: str,
) -> dict[str, Any]:
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    common = {
        "rows": len(assessment),
        "sessions": int(assessment["session"].nunique()),
        "stocks": int(assessment["symbol"].nunique()),
        "months": int(pd.to_datetime(assessment["session"]).dt.to_period("M").nunique()),
        "maximum_weighted_stock_share": _weighted_stock_share(panel),
    }
    if branch == "A":
        evidence = {
            **common,
            "positive_outcomes": int(assessment[TARGET_COMPLETION].sum()),
        }
        evidence["passed"] = bool(
            evidence["rows"] >= 30_000
            and evidence["sessions"] >= 140
            and evidence["stocks"] >= 15
            and evidence["months"] == 8
            and evidence["positive_outcomes"] >= 400
            and evidence["maximum_weighted_stock_share"] <= 0.10
        )
        return evidence
    if branch not in {"B", "C"}:
        raise ValueError(f"unknown branch support request: {branch}")
    target = TARGET_COMPLETION if branch == "B" else TARGET_IV_EXCESS
    minimum_positives = 100 if branch == "B" else 300
    evidence = {
        **common,
        "positive_outcomes": int(assessment[target].sum()),
        "broad_conflict_rows": int(assessment["BROAD_CONFLICT"].sum()),
        "low_route_support_rows": int(assessment["LOW_ROUTE_SUPPORT"].sum()),
    }
    evidence["passed"] = bool(
        evidence["rows"] >= 5_000
        and evidence["sessions"] >= 100
        and evidence["stocks"] >= 10
        and evidence["months"] >= 5
        and evidence["positive_outcomes"] >= minimum_positives
        and evidence["broad_conflict_rows"] >= 200
        and evidence["low_route_support_rows"] >= 200
        and evidence["maximum_weighted_stock_share"] <= 0.15
    )
    return evidence


def fit_model_pair(
    panel: pd.DataFrame,
    *,
    branch: str,
    model_specs: Sequence[tuple[str, Sequence[str], Sequence[str]]],
    target: str,
) -> ModelPairResult:
    development = panel.loc[panel["period"].eq("development")].copy()
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    models: dict[str, FrozenCrossMarketModel] = {}
    boundaries: dict[str, dict[str, float]] = {}
    for model_id, features, controls in model_specs:
        try:
            model = fit_cross_market_model(
                development,
                model_id=model_id,
                numeric_features=features,
                category_control_names=controls,
                target_column=target,
                kind="logistic",
            )
        except (RuntimeError, ValueError, ConvergenceWarning) as error:
            raise BranchBlocker(
                branch,
                "blocked_model_convergence_failure",
                f"{model_id} fit failed: {type(error).__name__}: {error}",
            ) from error
        models[model_id] = model
        development[f"{model_id}_prediction"] = model.predict(development)
        assessment[f"{model_id}_prediction"] = model.predict(assessment)
        boundaries[model_id] = probability_quantile_boundaries(
            development[f"{model_id}_prediction"]
        )
    return ModelPairResult(
        branch=branch,
        target=target,
        development=development,
        assessment=assessment,
        models=models,
        boundaries=boundaries,
    )


def _metric_row(
    frame: pd.DataFrame,
    *,
    model_id: str,
    target: str,
    boundaries: Mapping[str, float],
    scope: str = "overall",
    group: str = "all",
) -> dict[str, Any]:
    return {
        "model": model_id,
        "scope": scope,
        "group": group,
        **binary_metrics(
            frame,
            target_column=target,
            probability_column=f"{model_id}_prediction",
            boundaries=boundaries,
        ),
    }


def _monthly_metrics(result: ModelPairResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    month_values = pd.to_datetime(result.assessment["session"]).dt.to_period("M").astype(str)
    for month, group in result.assessment.assign(_month=month_values).groupby(
        "_month", sort=True, observed=True
    ):
        for model_id in result.models:
            rows.append(
                _metric_row(
                    group,
                    model_id=model_id,
                    target=result.target,
                    boundaries=result.boundaries[model_id],
                    scope="month",
                    group=str(month),
                )
            )
    return pd.DataFrame(rows)


def _supported_stock_regimes(stock_context: pd.DataFrame) -> set[int]:
    assessment = stock_context.loc[stock_context["period"].eq("assessment")]
    supported: set[int] = set()
    for regime in range(4):
        hard = assessment.loc[assessment["daily_stock_regime"].astype(int).eq(regime)]
        mass = float(assessment[f"daily_stock_regime_p_{regime}"].mean())
        if (
            mass >= 0.05
            and hard["symbol"].nunique() >= 8
            and pd.to_datetime(hard["session"]).dt.to_period("M").nunique() >= 4
        ):
            supported.add(regime)
    return supported


def subgroup_metrics(
    result: ModelPairResult,
    *,
    definitions: Sequence[tuple[str, pd.Series]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, mask in definitions:
        group = result.assessment.loc[mask]
        if group.empty:
            continue
        for model_id in result.models:
            rows.append(
                _metric_row(
                    group,
                    model_id=model_id,
                    target=result.target,
                    boundaries=result.boundaries[model_id],
                    scope="subgroup",
                    group=label,
                )
            )
    return pd.DataFrame(rows)


def branch_metric_tables(
    result: ModelPairResult,
    *,
    supported_stock_regimes: set[int],
    supported_front_regimes: set[int],
    cutoffs: DevelopmentCutoffs | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            _metric_row(
                result.assessment,
                model_id=model_id,
                target=result.target,
                boundaries=result.boundaries[model_id],
            )
            for model_id in result.models
        ]
    )
    definitions: list[tuple[str, pd.Series]] = [
        *[
            (
                f"checkpoint_group={group}",
                result.assessment["checkpoint_group"].eq(group),
            )
            for group in ("early_6_14", "middle_16_24", "late_26_34")
        ],
        (
            "route_state=BROAD_CONFLICT",
            result.assessment["BROAD_CONFLICT"].eq(1),
        ),
        (
            "route_state=LOW_ROUTE_SUPPORT",
            result.assessment["LOW_ROUTE_SUPPORT"].eq(1),
        ),
        *[
            (
                f"daily_stock_regime={regime}",
                result.assessment["daily_stock_regime"].astype(int).eq(regime),
            )
            for regime in sorted(supported_stock_regimes)
        ],
    ]
    if result.branch == "B":
        definitions.extend(
            [
                *[
                    (
                        f"front_options_regime={regime}",
                        result.assessment["front_options_regime"].astype(int).eq(regime),
                    )
                    for regime in sorted(supported_front_regimes)
                ],
            ]
        )
        if cutoffs is None:
            raise ValueError("Branch B subgroup metrics require frozen cutoffs")
        for feature, cutoff in (
            (
                "mismatch_compression_vs_front_iv",
                cutoffs.mismatch_compression_vs_front_iv,
            ),
            (
                "mismatch_complacent_broad_conflict",
                cutoffs.mismatch_complacent_broad_conflict,
            ),
        ):
            definitions.extend(
                [
                    (
                        f"{feature}=high",
                        result.assessment[feature].ge(cutoff),
                    ),
                    (
                        f"{feature}=low",
                        result.assessment[feature].lt(cutoff),
                    ),
                ]
            )
    return (
        overall,
        _monthly_metrics(result),
        subgroup_metrics(
            result,
            definitions=definitions,
        ),
    )


def _metric_map(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["model"]): cast(dict[str, Any], row) for row in frame.to_dict(orient="records")}


def metric_improvement(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "log_loss_improvement": float(old["log_loss"]) - float(new["log_loss"]),
        "brier_improvement": float(old["brier_score"]) - float(new["brier_score"]),
        "auc_improvement": float(new["auc"]) - float(old["auc"]),
        "average_precision_improvement": float(new["average_precision"])
        - float(old["average_precision"]),
    }


def _bootstrap_lower(
    bootstrap: pd.DataFrame,
    statistic: str,
    confidence: float = 0.80,
) -> float:
    row = bootstrap.loc[
        bootstrap["statistic"].eq(statistic)
        & np.isclose(bootstrap["confidence"].to_numpy(float), confidence)
    ]
    if len(row) != 1:
        raise ValueError(f"bootstrap interval unavailable: {statistic}/{confidence}")
    return float(row.iloc[0]["lower"])


def evaluate_increment_status(
    *,
    prefix: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    bootstrap: pd.DataFrame,
    positive_months: int,
    adverse_checkpoint_groups: int | None,
    null_metrics: pd.DataFrame | None,
) -> tuple[str, dict[str, Any]]:
    """Apply the exact preregistered classifier increment gates."""

    increment = metric_improvement(old, new)
    gates: dict[str, Any] = {
        **increment,
        "log_loss_improved": increment["log_loss_improvement"] > 0.0,
        "brier_improved": increment["brier_improvement"] > 0.0,
        "auc_not_reduced": increment["auc_improvement"] >= 0.0,
        "average_precision_improved": (increment["average_precision_improvement"] > 0.0),
        "bootstrap_80_log_loss_lower": _bootstrap_lower(
            bootstrap,
            f"{prefix}_log_loss_improvement",
        ),
        "bootstrap_80_brier_lower": _bootstrap_lower(
            bootstrap,
            f"{prefix}_brier_improvement",
        ),
        "positive_assessment_months": positive_months,
        "positive_in_at_least_four_months": positive_months >= 4,
    }
    gates["bootstrap_80_proper_score_lowers_non_negative"] = bool(
        gates["bootstrap_80_log_loss_lower"] >= 0.0 and gates["bootstrap_80_brier_lower"] >= 0.0
    )
    if adverse_checkpoint_groups is not None:
        gates["materially_adverse_checkpoint_groups"] = adverse_checkpoint_groups
        gates["no_checkpoint_group_materially_adverse"] = adverse_checkpoint_groups == 0
    if null_metrics is not None:
        expected_columns = {
            "real_exceeds_null_log_loss_improvement",
            "real_exceeds_null_brier_improvement",
        }
        if missing := sorted(expected_columns.difference(null_metrics.columns)):
            raise ValueError(f"null proper-score comparisons missing: {missing}")
        gates["proper_score_increment_exceeds_all_three_nulls"] = bool(
            len(null_metrics) == 3
            and null_metrics[
                [
                    "real_exceeds_null_log_loss_improvement",
                    "real_exceeds_null_brier_improvement",
                ]
            ]
            .astype(bool)
            .all(axis=None)
        )
    required = [
        "log_loss_improved",
        "brier_improved",
        "auc_not_reduced",
        "average_precision_improved",
        "bootstrap_80_proper_score_lowers_non_negative",
        "positive_in_at_least_four_months",
    ]
    if adverse_checkpoint_groups is not None:
        required.append("no_checkpoint_group_materially_adverse")
    if null_metrics is not None:
        required.append("proper_score_increment_exceeds_all_three_nulls")
    gates["passed"] = all(bool(gates[name]) for name in required)
    return ("supported" if gates["passed"] else "not_supported"), gates


def positive_month_count(
    monthly: pd.DataFrame,
    *,
    old_model: str,
    new_model: str,
) -> int:
    old = monthly.loc[monthly["model"].eq(old_model)].set_index("group")
    new = monthly.loc[monthly["model"].eq(new_model)].set_index("group")
    shared = old.index.intersection(new.index)
    return int(
        sum(
            float(old.loc[group, "log_loss"]) - float(new.loc[group, "log_loss"]) > 0.0
            for group in shared
        )
    )


def adverse_checkpoint_group_count(
    subgroup: pd.DataFrame,
    *,
    old_model: str,
    new_model: str,
) -> int:
    adverse = 0
    for group in (
        "checkpoint_group=early_6_14",
        "checkpoint_group=middle_16_24",
        "checkpoint_group=late_26_34",
    ):
        old = subgroup.loc[subgroup["model"].eq(old_model) & subgroup["group"].eq(group)]
        new = subgroup.loc[subgroup["model"].eq(new_model) & subgroup["group"].eq(group)]
        if len(old) != 1 or len(new) != 1:
            raise ValueError(f"checkpoint-group metric missing: {group}")
        if (
            float(old.iloc[0]["log_loss"]) - float(new.iloc[0]["log_loss"]) < -1e-12
            or float(old.iloc[0]["brier_score"]) - float(new.iloc[0]["brier_score"]) < -1e-12
        ):
            adverse += 1
    return adverse


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_rate(frame: pd.DataFrame, column: str) -> float:
    return _weighted_mean(frame, column)


def _weighted_median(frame: pd.DataFrame, column: str) -> float:
    return weighted_quantile(
        pd.to_numeric(frame[column], errors="raise").to_numpy(float),
        pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float),
        0.5,
    )


def _bootstrap_metric_increment(
    result: ModelPairResult,
    multiplicity: Mapping[str, int],
    old_model: str,
    new_model: str,
) -> dict[str, float]:
    sampled = result.assessment.copy()
    sampled["row_weight"] = sampled["row_weight"].to_numpy(float) * sampled["session"].astype(
        str
    ).map(multiplicity).fillna(0).to_numpy(float)
    sampled = sampled.loc[sampled["row_weight"].gt(0.0)]
    old = binary_metrics(
        sampled,
        target_column=result.target,
        probability_column=f"{old_model}_prediction",
        boundaries=result.boundaries[old_model],
    )
    new = binary_metrics(
        sampled,
        target_column=result.target,
        probability_column=f"{new_model}_prediction",
        boundaries=result.boundaries[new_model],
    )
    return metric_improvement(old, new)


def bootstrap_intervals(
    branch_a: ModelPairResult | None,
    branch_b: ModelPairResult | None,
    branch_c: ModelPairResult | None,
) -> pd.DataFrame:
    results = [result for result in (branch_a, branch_b, branch_c) if result is not None]
    sessions = pd.Series(
        sorted(set().union(*[set(result.assessment["session"].astype(str)) for result in results])),
        dtype="string",
    )
    session_draws = fixed_session_bootstrap_multiplicities(
        sessions,
        draws=10,
        seed=20260723,
    )
    names: list[str] = []
    if branch_a is not None:
        names.extend(
            [
                "A1_minus_A0_log_loss_improvement",
                "A1_minus_A0_brier_improvement",
                "A1_minus_A0_auc_improvement",
                "A1_minus_A0_average_precision_improvement",
            ]
        )
    if branch_b is not None:
        names.extend(
            [
                "B1_minus_B0_log_loss_improvement",
                "B1_minus_B0_brier_improvement",
                "B1_minus_B0_auc_improvement",
                "B1_minus_B0_average_precision_improvement",
            ]
        )
    if branch_c is not None:
        names.extend(
            [
                "C1_minus_C0_log_loss_improvement",
                "C1_minus_C0_brier_improvement",
                "C1_minus_C0_auc_improvement",
                "C1_minus_C0_average_precision_improvement",
                "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_mean_iv_residual",
                "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_median_iv_residual",
                "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_exceed_iv_rate",
            ]
        )
    statistics: dict[str, list[float]] = {name: [] for name in names}
    for draw in session_draws:
        multiplicity = {
            str(session): int(count) for session, count in zip(sessions, draw, strict=True)
        }
        comparisons: list[tuple[str, ModelPairResult, str, str]] = []
        if branch_a is not None:
            comparisons.append(("A1_minus_A0", branch_a, "A0", "A1"))
        if branch_b is not None:
            comparisons.append(("B1_minus_B0", branch_b, "B0", "B1"))
        if branch_c is not None:
            comparisons.append(("C1_minus_C0", branch_c, "C0", "C1"))
        for prefix, result, old, new in comparisons:
            increment = _bootstrap_metric_increment(
                result,
                multiplicity,
                old,
                new,
            )
            for metric, value in increment.items():
                statistics[f"{prefix}_{metric}"].append(value)
        if branch_c is None:
            continue
        sampled = branch_c.assessment.copy()
        sampled["row_weight"] = sampled["row_weight"].to_numpy(float) * sampled["session"].astype(
            str
        ).map(multiplicity).fillna(0).to_numpy(float)
        sampled = sampled.loc[sampled["row_weight"].gt(0.0)]
        broad = sampled.loc[sampled["BROAD_CONFLICT"].eq(1)]
        low = sampled.loc[sampled["LOW_ROUTE_SUPPORT"].eq(1)]
        statistics["BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_mean_iv_residual"].append(
            _weighted_mean(broad, "iv_absolute_residual_15m")
            - _weighted_mean(low, "iv_absolute_residual_15m")
        )
        statistics["BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_median_iv_residual"].append(
            _weighted_median(broad, "iv_absolute_residual_15m")
            - _weighted_median(low, "iv_absolute_residual_15m")
        )
        statistics["BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_exceed_iv_rate"].append(
            _weighted_rate(broad, TARGET_IV_EXCESS) - _weighted_rate(low, TARGET_IV_EXCESS)
        )
    rows: list[dict[str, Any]] = []
    for statistic, values in statistics.items():
        array = np.asarray(values, dtype=float)
        for confidence in (0.80, 0.90, 0.95):
            tail = (1.0 - confidence) / 2.0
            rows.append(
                {
                    "statistic": statistic,
                    "confidence": confidence,
                    "lower": float(np.quantile(array, tail)),
                    "upper": float(np.quantile(array, 1.0 - tail)),
                    "draws": 10,
                    "fixed_prediction": True,
                    "whole_session_resampling": True,
                    "shared_session_draws": True,
                    "coarse_quick_screen_diagnostic": True,
                }
            )
    return pd.DataFrame(rows)


def _front_bundle_columns(panel: pd.DataFrame) -> list[str]:
    candidates = (
        *FRONT_OPTIONS_RAW_FEATURES,
        *FRONT_CONTEXT_FEATURES,
        "front_options_regime",
        "front_options_regime_top_probability",
        "front_options_regime_mahalanobis_to_nearest_centroid",
        "required_options_date",
        "options_observation_date",
        "previous_close_underlying_price",
        "front_expiration_date",
        "front_strike",
        "front_call_contract_id",
        "front_put_contract_id",
        "skew_put_contract_id",
        "skew_call_contract_id",
        "previous_close_chain_request_ids",
    )
    return list(dict.fromkeys(column for column in candidates if column in panel.columns))


def front_options_null_refits(
    result: ModelPairResult,
    *,
    standardization: Mapping[str, MeanScale],
) -> pd.DataFrame:
    b0 = binary_metrics(
        result.assessment,
        target_column=result.target,
        probability_column="B0_prediction",
        boundaries=result.boundaries["B0"],
    )
    real_b1 = binary_metrics(
        result.assessment,
        target_column=result.target,
        probability_column="B1_prediction",
        boundaries=result.boundaries["B1"],
    )
    real_increment = metric_improvement(b0, real_b1)
    complete = pd.concat(
        [result.development, result.assessment],
        ignore_index=True,
    )
    bundle = _front_bundle_columns(complete)
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(FRONT_OPTIONS_NULL_SEEDS):
        permuted = permute_bundle_within_slates(
            complete,
            columns=bundle,
            seed=seed,
        )
        permuted = add_front_mismatch_features(permuted, standardization)
        model = fit_cross_market_model(
            permuted.loc[permuted["period"].eq("development")],
            model_id=f"B1_front_options_null_{index}",
            numeric_features=B1_FEATURES,
            category_control_names=("stock", "route_state"),
            target_column=TARGET_COMPLETION,
            kind="logistic",
        )
        assessment = permuted.loc[permuted["period"].eq("assessment")].copy()
        assessment["_null_prediction"] = model.predict(assessment)
        null_metric = binary_metrics(
            assessment,
            target_column=TARGET_COMPLETION,
            probability_column="_null_prediction",
            boundaries=result.boundaries["B1"],
        )
        increment = metric_improvement(b0, null_metric)
        rows.append(
            {
                "null_refit": index,
                "seed": seed,
                **increment,
                **{
                    f"real_exceeds_null_{metric}": real_increment[metric] > value
                    for metric, value in increment.items()
                },
            }
        )
    return pd.DataFrame(rows)


def _stock_structure_bundle_columns(panel: pd.DataFrame) -> list[str]:
    candidates = (
        *STOCK_CONTEXT_FEATURES,
        "daily_stock_regime",
        "daily_stock_regime_top_probability",
        "daily_stock_regime_mahalanobis_to_nearest_centroid",
        *H0_NON_CLOCK_FEATURES,
        *ROUTE_FEATURES,
        "route_resolution_state",
    )
    return list(dict.fromkeys(column for column in candidates if column in panel.columns))


def stock_structure_null_refits(
    result: ModelPairResult,
    *,
    standardization: Mapping[str, MeanScale],
) -> pd.DataFrame:
    c0 = binary_metrics(
        result.assessment,
        target_column=result.target,
        probability_column="C0_prediction",
        boundaries=result.boundaries["C0"],
    )
    real_c1 = binary_metrics(
        result.assessment,
        target_column=result.target,
        probability_column="C1_prediction",
        boundaries=result.boundaries["C1"],
    )
    real_increment = metric_improvement(c0, real_c1)
    complete = pd.concat(
        [result.development, result.assessment],
        ignore_index=True,
    )
    bundle = _stock_structure_bundle_columns(complete)
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(STOCK_STRUCTURE_NULL_SEEDS):
        permuted = permute_bundle_within_slates(
            complete,
            columns=bundle,
            seed=seed,
        )
        permuted["BROAD_CONFLICT"] = (
            permuted["route_resolution_state"].astype(str).eq("BROAD_CONFLICT").astype(int)
        )
        permuted["LOW_ROUTE_SUPPORT"] = (
            permuted["route_resolution_state"].astype(str).eq("LOW_ROUTE_SUPPORT").astype(int)
        )
        permuted = add_front_mismatch_features(permuted, standardization)
        model = fit_cross_market_model(
            permuted.loc[permuted["period"].eq("development")],
            model_id=f"C1_stock_structure_null_{index}",
            numeric_features=C1_FEATURES,
            category_control_names=("stock", "route_state"),
            target_column=TARGET_IV_EXCESS,
            kind="logistic",
        )
        assessment = permuted.loc[permuted["period"].eq("assessment")].copy()
        assessment["_null_prediction"] = model.predict(assessment)
        null_metric = binary_metrics(
            assessment,
            target_column=TARGET_IV_EXCESS,
            probability_column="_null_prediction",
            boundaries=result.boundaries["C1"],
        )
        increment = metric_improvement(c0, null_metric)
        rows.append(
            {
                "null_refit": index,
                "seed": seed,
                **increment,
                **{
                    f"real_exceeds_null_{metric}": real_increment[metric] > value
                    for metric, value in increment.items()
                },
            }
        )
    return pd.DataFrame(rows)


def concentration_metrics(
    branch_a_panel: pd.DataFrame,
    joined_panel: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, panel in (
        ("branch_a", branch_a_panel),
        ("joined_front_options", joined_panel),
    ):
        assessment = panel.loc[panel["period"].eq("assessment")]
        total = float(assessment["row_weight"].sum())
        for kind in ("stock", "month"):
            groups = (
                assessment["symbol"].astype(str)
                if kind == "stock"
                else pd.to_datetime(assessment["session"]).dt.to_period("M").astype(str)
            )
            grouped = (
                assessment.assign(_group=groups)
                .groupby("_group", sort=True, observed=True)["row_weight"]
                .sum()
            )
            for group, weight in grouped.items():
                rows.append(
                    {
                        "population": population,
                        "concentration_type": kind,
                        "group": str(group),
                        "weighted_rows": float(weight),
                        "share": float(weight / total),
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class CoreRun:
    """All independently staged model-branch outputs."""

    structural: pd.DataFrame
    structural_reconstruction: dict[str, Any]
    stock_context: pd.DataFrame
    stock_reconstruction: dict[str, Any]
    branch_a_panel: pd.DataFrame
    branch_a_support: dict[str, Any]
    branch_a: ModelPairResult | None
    front_raw: pd.DataFrame | None
    front_dimensions: pd.DataFrame | None
    front_parameters: FrozenDimensionParameters | None
    front_regime: FrozenSoftRegime | None
    front_pair_reconstruction: dict[str, Any] | None
    front_regime_mapping: dict[str, Any] | None
    joined_panel: pd.DataFrame | None
    mismatch_standardization: dict[str, MeanScale] | None
    cutoffs: DevelopmentCutoffs | None
    branch_b_support: dict[str, Any] | None
    branch_b: ModelPairResult | None
    movement_panel: pd.DataFrame | None
    movement_audit: dict[str, Any] | None
    full_bar_manifest: dict[str, Any] | None
    branch_c_support: dict[str, Any] | None
    branch_c: ModelPairResult | None
    blockers: dict[str, dict[str, str]]


def execute_core(provider_root: Path) -> CoreRun:
    """Build and fit the three branches without coupling their terminal states."""

    structural, structural_reconstruction = load_structural_panel()
    stock_context, stock_reconstruction = reconstruct_daily_stock_context()
    branch_a_panel = join_daily_stock_panel(structural, stock_context)
    branch_a_support_value = branch_support(branch_a_panel, branch="A")
    blockers: dict[str, dict[str, str]] = {}
    branch_a_result: ModelPairResult | None = None
    if bool(branch_a_support_value["passed"]):
        try:
            branch_a_result = fit_model_pair(
                branch_a_panel,
                branch="A",
                model_specs=(
                    ("A0", A0_FEATURES, ("stock", "route_state")),
                    ("A1", A1_FEATURES, ("stock", "route_state")),
                ),
                target=TARGET_COMPLETION,
            )
        except BranchBlocker as blocker:
            blockers["A"] = {
                "status": blocker.status,
                "detail": blocker.detail,
            }
    else:
        blockers["A"] = {
            "status": "insufficient_support",
            "detail": f"Branch A support gate failed: {branch_a_support_value}",
        }

    front_raw: pd.DataFrame | None = None
    front_dimensions: pd.DataFrame | None = None
    front_parameters: FrozenDimensionParameters | None = None
    front_regime: FrozenSoftRegime | None = None
    pair_reconstruction: dict[str, Any] | None = None
    regime_mapping_value: dict[str, Any] | None = None
    joined_panel: pd.DataFrame | None = None
    mismatch_standardization: dict[str, MeanScale] | None = None
    cutoffs: DevelopmentCutoffs | None = None
    branch_b_support_value: dict[str, Any] | None = None
    branch_b_result: ModelPairResult | None = None
    movement_panel: pd.DataFrame | None = None
    movement_audit: dict[str, Any] | None = None
    full_bar_manifest: dict[str, Any] | None = None
    branch_c_support_value: dict[str, Any] | None = None
    branch_c_result: ModelPairResult | None = None

    try:
        (
            front_raw,
            front_dimensions,
            front_parameters,
            front_regime,
            pair_reconstruction,
            regime_mapping_value,
        ) = build_front_options_context()
        joined_panel, mismatch_standardization, cutoffs = join_front_options_panel(
            branch_a_panel,
            front_dimensions,
        )
    except BranchBlocker as blocker:
        blockers["front_options"] = {
            "status": blocker.status,
            "detail": blocker.detail,
        }
        blockers["B"] = {
            "status": "blocked",
            "detail": "front-options context is unavailable",
        }
        blockers["C"] = {
            "status": "blocked",
            "detail": "front-options context is unavailable",
        }
        return CoreRun(
            structural=structural,
            structural_reconstruction=structural_reconstruction,
            stock_context=stock_context,
            stock_reconstruction=stock_reconstruction,
            branch_a_panel=branch_a_panel,
            branch_a_support=branch_a_support_value,
            branch_a=branch_a_result,
            front_raw=front_raw,
            front_dimensions=front_dimensions,
            front_parameters=front_parameters,
            front_regime=front_regime,
            front_pair_reconstruction=pair_reconstruction,
            front_regime_mapping=regime_mapping_value,
            joined_panel=joined_panel,
            mismatch_standardization=mismatch_standardization,
            cutoffs=cutoffs,
            branch_b_support=branch_b_support_value,
            branch_b=branch_b_result,
            movement_panel=movement_panel,
            movement_audit=movement_audit,
            full_bar_manifest=full_bar_manifest,
            branch_c_support=branch_c_support_value,
            branch_c=branch_c_result,
            blockers=blockers,
        )

    if joined_panel is None:
        raise AssertionError("front-options join did not return a panel")
    branch_b_support_value = branch_support(joined_panel, branch="B")
    if bool(branch_b_support_value["passed"]):
        try:
            branch_b_result = fit_model_pair(
                joined_panel,
                branch="B",
                model_specs=(
                    ("B0", B0_FEATURES, ("stock", "route_state")),
                    ("B1", B1_FEATURES, ("stock", "route_state")),
                ),
                target=TARGET_COMPLETION,
            )
        except BranchBlocker as blocker:
            blockers["B"] = {
                "status": blocker.status,
                "detail": blocker.detail,
            }
    else:
        blockers["B"] = {
            "status": "insufficient_support",
            "detail": f"Branch B support gate failed: {branch_b_support_value}",
        }

    try:
        predecessor = load_predecessor_runner()
        full_bars, full_bar_manifest_value = predecessor.load_full_regular_session_bars(
            provider_root
        )
        full_bar_manifest = cast(dict[str, Any], full_bar_manifest_value)
        movement_panel, movement_audit = attach_15m_movement(joined_panel, full_bars)
        structural_reconstruction["fifteen_minute_underlying_outcome_rows"] = len(movement_panel)
        structural_reconstruction["fifteen_minute_outcome_missing_rows"] = int(
            movement_audit["missing_or_invalid_outcome_rows"]
        )
        branch_c_support_value = branch_support(movement_panel, branch="C")
        if bool(branch_c_support_value["passed"]):
            branch_c_result = fit_model_pair(
                movement_panel,
                branch="C",
                model_specs=(
                    ("C0", C0_FEATURES, ("stock",)),
                    ("C1", C1_FEATURES, ("stock", "route_state")),
                ),
                target=TARGET_IV_EXCESS,
            )
        else:
            blockers["C"] = {
                "status": "insufficient_support",
                "detail": f"Branch C support gate failed: {branch_c_support_value}",
            }
    except BranchBlocker as blocker:
        blockers["C"] = {
            "status": blocker.status,
            "detail": blocker.detail,
        }
    except Exception as error:
        blockers["C"] = {
            "status": "blocked",
            "detail": f"15-minute movement reconstruction failed: {type(error).__name__}: {error}",
        }

    primary_fits = sum(
        len(result.models)
        for result in (branch_a_result, branch_b_result, branch_c_result)
        if result is not None
    )
    if primary_fits > 6:
        raise BranchBlocker(
            "all",
            "blocked_quick_resource_limit",
            f"primary classifier fits exceeded six: {primary_fits}",
        )
    return CoreRun(
        structural=structural,
        structural_reconstruction=structural_reconstruction,
        stock_context=stock_context,
        stock_reconstruction=stock_reconstruction,
        branch_a_panel=branch_a_panel,
        branch_a_support=branch_a_support_value,
        branch_a=branch_a_result,
        front_raw=front_raw,
        front_dimensions=front_dimensions,
        front_parameters=front_parameters,
        front_regime=front_regime,
        front_pair_reconstruction=pair_reconstruction,
        front_regime_mapping=regime_mapping_value,
        joined_panel=joined_panel,
        mismatch_standardization=mismatch_standardization,
        cutoffs=cutoffs,
        branch_b_support=branch_b_support_value,
        branch_b=branch_b_result,
        movement_panel=movement_panel,
        movement_audit=movement_audit,
        full_bar_manifest=full_bar_manifest,
        branch_c_support=branch_c_support_value,
        branch_c=branch_c_result,
        blockers=blockers,
    )


def front_coverage_table(
    structural: pd.DataFrame,
    front_dimensions: pd.DataFrame,
) -> pd.DataFrame:
    joined = structural.merge(
        front_dimensions.loc[
            :,
            [
                "symbol",
                "session",
                "front_options_regime",
                "options_observation_date",
            ],
        ],
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for period, period_frame in joined.groupby("period", sort=True, observed=True):
        rows.append(
            {
                "period": period,
                "coverage_type": "overall",
                "group": "all",
                "rows": len(period_frame),
                "sessions": int(period_frame["session"].nunique()),
                "stocks": int(period_frame["symbol"].nunique()),
                "months": int(pd.to_datetime(period_frame["session"]).dt.to_period("M").nunique()),
                "broad_conflict_rows": int(period_frame["BROAD_CONFLICT"].sum()),
                "low_route_support_rows": int(period_frame["LOW_ROUTE_SUPPORT"].sum()),
            }
        )
    assessment = joined.loc[joined["period"].eq("assessment")].copy()
    assessment["_month"] = pd.to_datetime(assessment["session"]).dt.to_period("M").astype(str)
    grouping = (
        ("stock", "symbol"),
        ("month", "_month"),
        ("checkpoint", "checkpoint"),
        ("route_resolution_state", "route_resolution_state"),
        ("front_options_regime", "front_options_regime"),
    )
    for kind, column in grouping:
        for group, frame in assessment.groupby(column, sort=True, observed=True):
            rows.append(
                {
                    "period": "assessment",
                    "coverage_type": kind,
                    "group": str(group),
                    "rows": len(frame),
                    "sessions": int(frame["session"].nunique()),
                    "stocks": int(frame["symbol"].nunique()),
                    "months": int(pd.to_datetime(frame["session"]).dt.to_period("M").nunique()),
                    "broad_conflict_rows": int(frame["BROAD_CONFLICT"].sum()),
                    "low_route_support_rows": int(frame["LOW_ROUTE_SUPPORT"].sum()),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class MetricOutputs:
    """All non-resampled assessment metric tables."""

    branch_a: pd.DataFrame
    branch_a_monthly: pd.DataFrame
    branch_a_subgroup: pd.DataFrame
    branch_b: pd.DataFrame
    branch_b_monthly: pd.DataFrame
    branch_b_subgroup: pd.DataFrame
    branch_c: pd.DataFrame
    branch_c_monthly: pd.DataFrame
    branch_c_subgroup: pd.DataFrame
    branch_c_route_state: pd.DataFrame


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=["model", "scope", "group", "log_loss", "brier_score"])


def build_metric_outputs(core: CoreRun) -> MetricOutputs:
    supported_stock = _supported_stock_regimes(core.stock_context)
    supported_front: set[int] = set()
    if core.front_dimensions is not None:
        supported_front = {
            regime
            for regime, evidence in front_regime_support(core.front_dimensions).items()
            if bool(evidence["supported"])
        }

    def tables(
        result: ModelPairResult | None,
        *,
        use_cutoffs: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if result is None:
            return _empty_metrics(), _empty_metrics(), _empty_metrics()
        return branch_metric_tables(
            result,
            supported_stock_regimes=supported_stock,
            supported_front_regimes=supported_front,
            cutoffs=core.cutoffs if use_cutoffs else None,
        )

    a, a_monthly, a_subgroup = tables(core.branch_a, use_cutoffs=False)
    b, b_monthly, b_subgroup = tables(core.branch_b, use_cutoffs=True)
    c, c_monthly, c_subgroup = tables(core.branch_c, use_cutoffs=False)
    route_state = (
        route_state_iv_metrics(core.branch_c.assessment)
        if core.branch_c is not None
        else pd.DataFrame()
    )
    return MetricOutputs(
        branch_a=a,
        branch_a_monthly=a_monthly,
        branch_a_subgroup=a_subgroup,
        branch_b=b,
        branch_b_monthly=b_monthly,
        branch_b_subgroup=b_subgroup,
        branch_c=c,
        branch_c_monthly=c_monthly,
        branch_c_subgroup=c_subgroup,
        branch_c_route_state=route_state,
    )


def _missing_branch_status(core: CoreRun, branch: str) -> str:
    blocker = core.blockers.get(branch)
    if blocker is None:
        return "blocked"
    return "insufficient_support" if blocker["status"] == "insufficient_support" else "blocked"


def build_decision(
    core: CoreRun,
    metrics: MetricOutputs,
    bootstrap: pd.DataFrame,
    front_null: pd.DataFrame,
    stock_null: pd.DataFrame,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}

    if core.branch_a is None:
        daily_stock_status = _missing_branch_status(core, "A")
    else:
        mapped = _metric_map(metrics.branch_a)
        daily_stock_status, gates["daily_stock_context"] = evaluate_increment_status(
            prefix="A1_minus_A0",
            old=mapped["A0"],
            new=mapped["A1"],
            bootstrap=bootstrap,
            positive_months=positive_month_count(
                metrics.branch_a_monthly,
                old_model="A0",
                new_model="A1",
            ),
            adverse_checkpoint_groups=adverse_checkpoint_group_count(
                metrics.branch_a_subgroup,
                old_model="A0",
                new_model="A1",
            ),
            null_metrics=None,
        )

    if core.front_dimensions is None:
        front_regime_status = "blocked"
        front_regime_support_value: dict[int, dict[str, Any]] = {}
    else:
        front_regime_support_value = front_regime_support(core.front_dimensions)
        front_regime_status = (
            "supported"
            if all(bool(value["supported"]) for value in front_regime_support_value.values())
            else "descriptive_only"
        )

    if core.branch_b is None:
        front_completion_status = _missing_branch_status(core, "B")
    else:
        mapped = _metric_map(metrics.branch_b)
        front_completion_status, gates["front_options_completion"] = evaluate_increment_status(
            prefix="B1_minus_B0",
            old=mapped["B0"],
            new=mapped["B1"],
            bootstrap=bootstrap,
            positive_months=positive_month_count(
                metrics.branch_b_monthly,
                old_model="B0",
                new_model="B1",
            ),
            adverse_checkpoint_groups=None,
            null_metrics=front_null,
        )

    if core.branch_c is None:
        stock_to_iv_status = _missing_branch_status(core, "C")
    else:
        mapped = _metric_map(metrics.branch_c)
        stock_to_iv_status, gates["stock_to_iv_excess"] = evaluate_increment_status(
            prefix="C1_minus_C0",
            old=mapped["C0"],
            new=mapped["C1"],
            bootstrap=bootstrap,
            positive_months=positive_month_count(
                metrics.branch_c_monthly,
                old_model="C0",
                new_model="C1",
            ),
            adverse_checkpoint_groups=None,
            null_metrics=stock_null,
        )

    broad_status = "insufficient_support"
    broad_gates: dict[str, Any] = {}
    if core.branch_c is not None and not metrics.branch_c_route_state.empty:
        indexed = metrics.branch_c_route_state.set_index("route_state")
        broad = indexed.loc["BROAD_CONFLICT"]
        low = indexed.loc["LOW_ROUTE_SUPPORT"]
        broad_gates = {
            "broad_rows": int(broad["rows"]),
            "low_route_support_rows": int(low["rows"]),
            "mean_iv_residual_difference": float(broad["mean_iv_residual"])
            - float(low["mean_iv_residual"]),
            "median_iv_residual_difference": float(broad["median_iv_residual"])
            - float(low["median_iv_residual"]),
            "exceed_iv_rate_difference": float(broad["exceed_iv_rate"])
            - float(low["exceed_iv_rate"]),
            "bootstrap_80_mean_residual_lower": _bootstrap_lower(
                bootstrap,
                "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_mean_iv_residual",
            ),
        }
        broad_gates["supported"] = bool(
            broad_gates["broad_rows"] >= 200
            and broad_gates["low_route_support_rows"] >= 200
            and broad_gates["mean_iv_residual_difference"] > 0.0
            and broad_gates["median_iv_residual_difference"] > 0.0
            and broad_gates["exceed_iv_rate_difference"] > 0.0
            and broad_gates["bootstrap_80_mean_residual_lower"] >= 0.0
        )
        broad_status = "supported" if broad_gates["supported"] else "descriptive_only"
    gates["broad_conflict_iv_residual"] = broad_gates

    preflight_path = PRIMARY / "back_expiry_schema_preflight.json"
    preflight = (
        read_json(preflight_path)
        if preflight_path.is_file()
        else {"status": "blocked_missing_eodhd_api_token"}
    )
    preflight_status = str(preflight.get("status", "blocked_schema_or_endpoint_failure"))

    if core.stock_reconstruction.get("passed") is not True:
        overall = "blocked_daily_stock_reconstruction_failure"
    elif core.structural_reconstruction.get("passed") is not True:
        overall = "blocked_structural_panel_reconstruction_failure"
    elif front_regime_status == "blocked" and daily_stock_status not in {
        "supported",
        "descriptive_only",
    }:
        overall = "blocked_front_options_regime_failure"
    else:
        overall = choose_overall_decision(
            daily_stock_context_status=daily_stock_status,
            front_options_completion_status=front_completion_status,
            stock_to_iv_excess_status=stock_to_iv_status,
        )
    decision = {
        **SAFETY_FLAGS,
        "overall_decision": overall,
        "daily_stock_context_status": daily_stock_status,
        "front_options_regime_status": front_regime_status,
        "front_options_completion_status": front_completion_status,
        "stock_to_iv_excess_status": stock_to_iv_status,
        "broad_conflict_iv_residual_status": broad_status,
        "back_expiry_preflight_status": preflight_status,
        "front_options_regime_support": front_regime_support_value,
        "branch_support": {
            "A": core.branch_a_support,
            "B": core.branch_b_support,
            "C": core.branch_c_support,
        },
        "gates": gates,
        "branch_blockers": core.blockers,
        "primary_classifier_fits": sum(
            len(result.models)
            for result in (core.branch_a, core.branch_b, core.branch_c)
            if result is not None
        ),
        "ridge_fits": 0,
        "option_pnl_calculated": False,
    }
    return decision


def assessment_predictions(core: CoreRun) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for result in (core.branch_a, core.branch_b, core.branch_c):
        if result is None:
            continue
        prediction_columns = [f"{model_id}_prediction" for model_id in result.models]
        pieces.append(
            result.assessment.loc[
                :,
                [
                    "row_id",
                    "symbol",
                    "session",
                    "checkpoint",
                    "row_weight",
                    result.target,
                    *prediction_columns,
                ],
            ].copy()
        )
    if not pieces:
        return pd.DataFrame({"status": ["no_models_fitted"]})
    output = pieces[0]
    for piece in pieces[1:]:
        shared_nonkeys = [
            column for column in piece.columns if column != "row_id" and column in output.columns
        ]
        piece = piece.drop(columns=shared_nonkeys)
        output = output.merge(piece, on="row_id", how="outer", validate="one_to_one")
    return output.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def model_artifacts(
    core: CoreRun,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configurations: dict[str, Any] = {
        **SAFETY_FLAGS,
        "primary_classifier_fit_limit": 6,
        "primary_classifier_fits": 0,
        "models": {},
    }
    coefficients: dict[str, Any] = {**SAFETY_FLAGS, "models": {}}
    population = {
        "A": "full_structural_stock_context",
        "B": "joined_front_options",
        "C": "joined_front_options_with_15m_outcome",
    }
    for result in (core.branch_a, core.branch_b, core.branch_c):
        if result is None:
            continue
        configurations["primary_classifier_fits"] += len(result.models)
        for model_id, model in result.models.items():
            specification = model.as_dict()
            cast(dict[str, Any], configurations["models"])[model_id] = {
                "branch": result.branch,
                "population": population[result.branch],
                "target": result.target,
                "numeric_features": list(model.numeric_features),
                "category_controls": list(model.category_controls),
                "development_rows": len(result.development),
                "assessment_rows": len(result.assessment),
                "penalty": "l2",
                "C": 0.25,
                "solver": "liblinear",
                "max_iter": 300,
                "class_weight": None,
                "n_jobs": 1,
            }
            cast(dict[str, Any], coefficients["models"])[model_id] = specification
    return configurations, coefficients


def _maximum_prediction_difference(
    first: ModelPairResult | None,
    second: ModelPairResult | None,
) -> float:
    if first is None or second is None:
        return 0.0 if first is second else math.inf
    if set(first.models) != set(second.models):
        return math.inf
    maximum = 0.0
    for model_id in first.models:
        left = first.assessment.loc[:, ["row_id", f"{model_id}_prediction"]].sort_values(
            "row_id", kind="mergesort"
        )
        right = second.assessment.loc[:, ["row_id", f"{model_id}_prediction"]].sort_values(
            "row_id", kind="mergesort"
        )
        if _row_identity_mismatches(left, right, ("row_id",)):
            return math.inf
        maximum = max(
            maximum,
            _maximum_numeric_difference(
                left,
                right,
                (f"{model_id}_prediction",),
            ),
        )
    return maximum


def determinism_rebuild(
    first: CoreRun,
    *,
    provider_root: Path,
    first_decision: Mapping[str, Any],
    frozen_bootstrap: pd.DataFrame,
    frozen_front_null: pd.DataFrame,
    frozen_stock_null: pd.DataFrame,
) -> dict[str, Any]:
    second = execute_core(provider_root)
    if first.front_raw is None or second.front_raw is None:
        selected_contract_mismatches = int(first.front_raw is not second.front_raw)
        maximum_front_raw_feature_difference = (
            0.0 if first.front_raw is second.front_raw else math.inf
        )
    else:
        first_front = first.front_raw.sort_values(["symbol", "session"], kind="mergesort")
        second_front = second.front_raw.sort_values(["symbol", "session"], kind="mergesort")
        selected_contract_mismatches = _row_identity_mismatches(
            first_front,
            second_front,
            (
                "symbol",
                "session",
                "options_observation_date",
                "front_expiration_date",
                "front_strike",
                "front_call_contract_id",
                "front_put_contract_id",
            ),
        )
        maximum_front_raw_feature_difference = _maximum_numeric_difference(
            first_front,
            second_front,
            (*FRONT_OPTIONS_RAW_FEATURES, *FRONT_OPTIONS_MISSING_INDICATORS),
        )
    pair_reconstruction_applicable = bool(
        first.front_raw is not None or second.front_raw is not None
    )
    pair_selection_rebuilt = bool(
        pair_reconstruction_applicable
        and first.front_pair_reconstruction is not None
        and second.front_pair_reconstruction is not None
        and first.front_pair_reconstruction.get("selection_rebuilt_from_cached_chains") is True
        and second.front_pair_reconstruction.get("selection_rebuilt_from_cached_chains") is True
    )
    if first.joined_panel is None or second.joined_panel is None:
        joined_row_mismatches = int(first.joined_panel is not second.joined_panel)
        maximum_feature_difference = 0.0 if first.joined_panel is second.joined_panel else math.inf
    else:
        first_joined = first.joined_panel.sort_values("row_id", kind="mergesort")
        second_joined = second.joined_panel.sort_values("row_id", kind="mergesort")
        joined_row_mismatches = _row_identity_mismatches(
            first_joined,
            second_joined,
            ("row_id",),
        )
        feature_columns = tuple(
            dict.fromkeys(
                (
                    *STOCK_CONTEXT_FEATURES,
                    *FRONT_OPTIONS_RAW_FEATURES,
                    *FRONT_CONTEXT_FEATURES,
                    *FRONT_MISMATCH_FEATURES,
                    *DENSE_H0_FEATURES,
                    *ROUTE_FEATURES,
                    "row_weight",
                    TARGET_COMPLETION,
                )
            )
        )
        maximum_feature_difference = _maximum_numeric_difference(
            first_joined,
            second_joined,
            feature_columns,
        )
    first_mapping = first.front_regime_mapping or {}
    second_mapping = second.front_regime_mapping or {}
    mapping_mismatches = int(
        first_mapping.get("canonical_to_original") != second_mapping.get("canonical_to_original")
        or first_mapping.get("canonical_centroids") != second_mapping.get("canonical_centroids")
    )
    maximum_probability_difference = max(
        _maximum_prediction_difference(first.branch_a, second.branch_a),
        _maximum_prediction_difference(first.branch_b, second.branch_b),
        _maximum_prediction_difference(first.branch_c, second.branch_c),
    )
    second_metrics = build_metric_outputs(second)
    second_decision = build_decision(
        second,
        second_metrics,
        frozen_bootstrap,
        frozen_front_null,
        frozen_stock_null,
    )
    decision_mismatches = int(
        second_decision["overall_decision"] != first_decision["overall_decision"]
        or any(
            second_decision[key] != first_decision[key]
            for key in (
                "daily_stock_context_status",
                "front_options_regime_status",
                "front_options_completion_status",
                "stock_to_iv_excess_status",
                "broad_conflict_iv_residual_status",
                "back_expiry_preflight_status",
            )
        )
    )
    result = {
        **SAFETY_FLAGS,
        "selected_contract_mismatches": selected_contract_mismatches,
        "pair_reconstruction_applicable": pair_reconstruction_applicable,
        "pair_selection_rebuilt_from_cached_chains": pair_selection_rebuilt,
        "maximum_front_raw_feature_difference": (maximum_front_raw_feature_difference),
        "joined_row_mismatches": joined_row_mismatches,
        "front_options_regime_mapping_mismatches": mapping_mismatches,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "decision_mismatches": decision_mismatches,
        "bootstrap_repeated": False,
        "null_draws_repeated": False,
        "options_redownloaded": False,
    }
    result["passed"] = bool(
        selected_contract_mismatches == 0
        and (not pair_reconstruction_applicable or pair_selection_rebuilt)
        and maximum_front_raw_feature_difference <= 1e-12
        and joined_row_mismatches == 0
        and mapping_mismatches == 0
        and maximum_feature_difference <= 1e-12
        and maximum_probability_difference <= 1e-12
        and decision_mismatches == 0
    )
    return result


def chronology_audit(front_raw: pd.DataFrame | None) -> pd.DataFrame:
    if front_raw is None:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session",
                "stock_information_date",
                "required_options_date",
                "options_observation_date",
                "stock_d_minus_1_exact",
                "options_d_minus_1_exact",
                "same_day_or_future_options_used",
                "chronology_passed",
            ]
        )
    full_stock_chronology = pd.read_parquet(
        PREDECESSOR_PRIMARY / "daily_stock_raw_features.parquet",
        columns=["symbol", "session", "stock_information_date"],
    )
    chronology = front_raw.merge(
        full_stock_chronology,
        on=["symbol", "session"],
        how="left",
        validate="one_to_one",
    )
    chronology["stock_d_minus_1_exact"] = (
        chronology["stock_information_date"]
        .astype(str)
        .eq(chronology["required_options_date"].astype(str))
    )
    chronology["options_d_minus_1_exact"] = (
        chronology["options_observation_date"]
        .astype(str)
        .eq(chronology["required_options_date"].astype(str))
    )
    chronology["same_day_or_future_options_used"] = pd.to_datetime(
        chronology["options_observation_date"]
    ).ge(pd.to_datetime(chronology["session"]))
    chronology["chronology_passed"] = (
        chronology["stock_d_minus_1_exact"]
        & chronology["options_d_minus_1_exact"]
        & ~chronology["same_day_or_future_options_used"]
    )
    return chronology.loc[
        :,
        [
            "symbol",
            "session",
            "stock_information_date",
            "required_options_date",
            "options_observation_date",
            "stock_d_minus_1_exact",
            "options_d_minus_1_exact",
            "same_day_or_future_options_used",
            "chronology_passed",
        ],
    ]


def mismatch_manifest(
    panel: pd.DataFrame,
    standardization: Mapping[str, MeanScale],
) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for period, period_frame in panel.groupby("period", sort=True, observed=True):
        distributions[str(period)] = {
            feature: {
                "rows": int(period_frame[feature].notna().sum()),
                "mean": float(period_frame[feature].mean()),
                "standard_deviation": float(period_frame[feature].std(ddof=0)),
                "minimum": float(period_frame[feature].min()),
                "q10": float(period_frame[feature].quantile(0.10)),
                "median": float(period_frame[feature].median()),
                "q90": float(period_frame[feature].quantile(0.90)),
                "maximum": float(period_frame[feature].max()),
            }
            for feature in FRONT_MISMATCH_FEATURES
        }
    return {
        **SAFETY_FLAGS,
        "features": list(FRONT_MISMATCH_FEATURES),
        "feature_count": 5,
        "additional_interactions_created": 0,
        "standardization_fitted_period": "development_2024_only",
        "standardization": {name: asdict(value) for name, value in standardization.items()},
        "formulas": {
            "mismatch_compression_vs_front_iv": (
                "z(daily_compression) - z(front_options_implied_tension)"
            ),
            "mismatch_daily_volatility_vs_front_iv": (
                "z(daily_volatility_acceleration) - z(front_options_implied_tension)"
            ),
            "mismatch_route_vs_front_premium": (
                "z(prefix_family_entropy) - z(front_options_premium_richness)"
            ),
            "mismatch_direction_agreement": (
                "z(signed_pressure) * z(front_options_directional_positioning)"
            ),
            "mismatch_complacent_broad_conflict": (
                "indicator(BROAD_CONFLICT) * -z(front_options_implied_tension)"
            ),
        },
        "distributions": distributions,
    }


def protected_boundary_audit(core: CoreRun) -> dict[str, Any]:
    market_dates: list[pd.Series] = [
        pd.to_datetime(core.structural["session"]),
        pd.to_datetime(core.stock_context["stock_information_date"]),
    ]
    protected_market_rows = sum(
        int(values.dt.date.ge(PROTECTED_START).sum()) for values in market_dates
    )
    model_option_rows = (
        0
        if core.front_raw is None
        else int(
            pd.to_datetime(core.front_raw["options_observation_date"])
            .dt.date.ge(PROTECTED_START)
            .sum()
        )
    )
    preflight = (
        read_json(PRIMARY / "back_expiry_schema_preflight.json")
        if (PRIMARY / "back_expiry_schema_preflight.json").is_file()
        else {}
    )
    preflight_protected_rows = int(preflight.get("protected_records_persisted", 0))
    protected_option_rows = model_option_rows + preflight_protected_rows
    result = {
        **SAFETY_FLAGS,
        "protected_start": PROTECTED_START.isoformat(),
        "boundary_applies_to": "observation_date_not_contract_expiration",
        "maximum_market_observation_date": max(str(values.max().date()) for values in market_dates),
        "maximum_option_observation_date": (
            None
            if core.front_raw is None
            else str(pd.to_datetime(core.front_raw["options_observation_date"]).max().date())
        ),
        "protected_market_rows_materialised": protected_market_rows,
        "protected_option_observations_materialised": protected_option_rows,
        "model_branch_protected_option_observations_materialised": model_option_rows,
        "preflight_protected_option_observations_materialised": (preflight_protected_rows),
        "preflight_raw_response_persisted": bool(preflight.get("raw_response_persisted", False)),
        "preflight_protected_records_returned_and_rejected": int(
            preflight.get("protected_records_returned", 0)
        ),
    }
    result["passed"] = bool(
        protected_market_rows == 0
        and protected_option_rows == 0
        and result["preflight_raw_response_persisted"] is False
    )
    return result


def source_manifest(core: CoreRun) -> dict[str, Any]:
    predecessor_source = read_json(PREDECESSOR_PRIMARY / "source_manifest.json")
    preflight = (
        read_json(PRIMARY / "back_expiry_schema_preflight.json")
        if (PRIMARY / "back_expiry_schema_preflight.json").is_file()
        else {"status": "blocked_missing_eodhd_api_token", "request_count": 0}
    )
    source_paths = cast(Mapping[str, Any], predecessor_source["sources"])
    return {
        **SAFETY_FLAGS,
        "starting_branch": STARTING_BRANCH,
        "starting_sha": STARTING_SHA,
        "final_branch": FINAL_BRANCH,
        "dates": {
            "development_start": DEVELOPMENT_START.isoformat(),
            "development_end": "2024-12-31",
            "assessment_start": "2025-01-01",
            "assessment_end": ASSESSMENT_END.isoformat(),
            "protected_start": PROTECTED_START.isoformat(),
        },
        "cohort": list(FROZEN_COHORT),
        "sources": {
            "dense_panel": str(DENSE_PANEL),
            "dense_panel_sha256": sha256_file(DENSE_PANEL),
            "predecessor_daily_stock_dimensions": str(
                PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
            ),
            "predecessor_daily_stock_dimensions_sha256": sha256_file(
                PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
            ),
            "predecessor_front_options_raw": str(
                PREDECESSOR_PRIMARY / "daily_options_raw_features.parquet"
            ),
            "predecessor_front_options_raw_sha256": sha256_file(
                PREDECESSOR_PRIMARY / "daily_options_raw_features.parquet"
            ),
            "repaired_exact_date_options_cache": source_paths["repaired_exact_date_options_cache"],
            "repaired_exact_date_options_cache_sha256": source_paths[
                "repaired_exact_date_options_cache_sha256"
            ],
            "full_regular_session_underlying_root": (
                None
                if core.full_bar_manifest is None
                else core.full_bar_manifest.get("provider_root")
            ),
        },
        "structural_rows": len(core.structural),
        "daily_stock_rows_reused": len(core.stock_context),
        "front_options_stock_sessions_reused": (
            0 if core.front_raw is None else len(core.front_raw)
        ),
        "cached_canonical_options_records_available": int(
            cast(Mapping[str, Any], predecessor_source["options_cache_reprocessing"])[
                "cache_rows_loaded"
            ]
        ),
        "cached_provider_records_reused_by_predecessor_pair_reconstruction": int(
            0
            if core.front_pair_reconstruction is None
            else core.front_pair_reconstruction["cache_records_reused"]
        ),
        "cached_pair_selection_rebuilt_in_v01": bool(
            core.front_pair_reconstruction is not None
            and core.front_pair_reconstruction["selection_rebuilt_from_cached_chains"]
        ),
        "model_branch_network_requests": 0,
        "model_branch_newly_downloaded_records": 0,
        "model_branch_newly_downloaded_bytes": 0,
        "back_expiry_preflight": {
            "status": preflight.get("status"),
            "request_count": int(preflight.get("request_count", 0)),
            "returned_records": int(preflight.get("record_count", 0)),
            "protected_records_returned_and_rejected": int(
                preflight.get("protected_records_returned", 0)
            ),
            "protected_records_persisted": int(preflight.get("protected_records_persisted", 0)),
            "canonical_records_admitted": 0,
        },
        "failed_compact_recovery_records_admitted": 0,
        "full_regular_session_underlying": core.full_bar_manifest,
    }


def create_plots(metrics: MetricOutputs, reports: Path) -> list[str]:
    reports.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if not metrics.branch_a.empty and not metrics.branch_b.empty:
        comparison = pd.concat(
            [
                metrics.branch_a.assign(branch="A"),
                metrics.branch_b.assign(branch="B"),
            ],
            ignore_index=True,
        )
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        for axis, metric, title in (
            (axes[0], "log_loss", "Completion log loss"),
            (axes[1], "brier_score", "Completion Brier score"),
        ):
            labels = comparison["model"].astype(str).tolist()
            axis.bar(labels, comparison[metric].to_numpy(float), color="#406c80")
            axis.set_title(title)
            axis.set_ylabel("Lower is better")
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle("Daily stock and front-options completion models")
        figure.tight_layout()
        path = reports / "completion_model_proper_scores.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    if not metrics.branch_c.empty and not metrics.branch_c_route_state.empty:
        figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
        axes[0].bar(
            metrics.branch_c["model"].astype(str),
            metrics.branch_c["log_loss"].to_numpy(float),
            color="#765a8c",
        )
        axes[0].set_title("IV-excess forecast log loss")
        axes[0].set_ylabel("Lower is better")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(
            metrics.branch_c_route_state["route_state"].astype(str),
            metrics.branch_c_route_state["mean_iv_residual"].to_numpy(float),
            color="#a35f45",
        )
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_title("Mean 15-minute IV residual")
        axes[1].tick_params(axis="x", rotation=25)
        figure.tight_layout()
        path = reports / "iv_excess_and_route_state_residuals.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    if len(paths) > 2:
        raise AssertionError("plot ceiling exceeded")
    return paths


def _markdown_metric_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No model was fitted for this branch."
    columns = [
        "model",
        "log_loss",
        "brier_score",
        "auc",
        "average_precision",
        "base_rate",
        "rows",
        "sessions",
        "stocks",
        "positive_outcomes",
    ]
    available = [column for column in columns if column in frame.columns]
    return markdown_table(frame.loc[:, available], float_digits=8)


def markdown_table(frame: pd.DataFrame, *, float_digits: int = 6) -> str:
    """Render a small Markdown table without an optional third-party dependency."""

    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_report_from_artifacts() -> str:
    """Resume only report rendering from already-frozen numerical artifacts."""

    def load_csv(name: str) -> pd.DataFrame:
        path = PRIMARY / name
        if not path.is_file():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        return (
            pd.DataFrame()
            if tuple(frame.columns) == ("status",)
            and frame["status"].astype(str).eq("unavailable").all()
            else frame
        )

    def load_json(name: str) -> dict[str, Any]:
        path = PRIMARY / name
        if not path.is_file():
            return {}
        value = read_json(path)
        return {} if value.get("status") == "unavailable" else value

    decision = read_json(PRIMARY / "decision.json")
    front_context_available = decision.get("front_options_regime_status") in {
        "supported",
        "descriptive_only",
    }
    determinism = load_json("determinism_check.json")
    mapping = load_json("front_options_regime_mapping.json") if front_context_available else {}
    branch_a = load_csv("branch_a_metrics.csv")
    branch_b = load_csv("branch_b_metrics.csv")
    branch_c = load_csv("branch_c_metrics.csv")
    route = load_csv("branch_c_route_state_metrics.csv")
    bootstrap = load_csv("bootstrap_metrics.csv")
    front_null = load_csv("front_options_null_metrics.csv")
    stock_null = load_csv("stock_structure_null_metrics.csv")
    structural = read_json(PRIMARY / "structural_panel_reconstruction.json")
    stock_reconstruction = read_json(PRIMARY / "daily_stock_reconstruction.json")
    pair_reconstruction = (
        load_json("front_options_pair_reconstruction.json") if front_context_available else {}
    )
    feature_manifest = (
        load_json("front_options_feature_manifest.json") if front_context_available else {}
    )
    mismatch = load_json("mismatch_feature_manifest.json") if front_context_available else {}
    preflight = read_json(PRIMARY / "back_expiry_schema_preflight.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    source = read_json(PRIMARY / "source_manifest.json")
    audit = (
        read_json(PRIMARY / "independent_audit.json")
        if decision.get("independent_audit_status") in {"passed", "failed"}
        and (PRIMARY / "independent_audit.json").is_file()
        else {
            "passed": False,
            "checks_passed": 0,
            "checks_run": 0,
            "status": "pending",
        }
    )
    concentration = load_csv("concentration_metrics.csv")
    branch_a_monthly = load_csv("branch_a_monthly_metrics.csv")
    branch_b_monthly = load_csv("branch_b_monthly_metrics.csv")
    branch_c_monthly = load_csv("branch_c_monthly_metrics.csv")
    branch_a_subgroup = load_csv("branch_a_regime_metrics.csv")
    branch_b_subgroup = load_csv("branch_b_regime_metrics.csv")
    centroid_rows = [
        {
            "regime": regime,
            **{
                dimension: float(cast(Mapping[str, Any], centroid)[dimension])
                for dimension in FRONT_OPTIONS_CANONICAL_DIMENSIONS
            },
        }
        for regime, centroid in enumerate(
            cast(Sequence[Mapping[str, Any]], mapping.get("canonical_centroids", []))
        )
    ]
    plots = sorted(str(path) for path in REPORTS.glob("*.png"))
    plot_lines = "\n".join(f"- `{path}`" for path in plots) or "- None"
    branch_support_value = cast(
        Mapping[str, Any],
        decision.get("branch_support", {}),
    )
    increment_rows: list[dict[str, Any]] = []
    for branch, frame, old_model, new_model in (
        ("A", branch_a, "A0", "A1"),
        ("B", branch_b, "B0", "B1"),
        ("C", branch_c, "C0", "C1"),
    ):
        if "model" not in frame:
            continue
        old_rows = frame.loc[frame["model"].eq(old_model)]
        new_rows = frame.loc[frame["model"].eq(new_model)]
        if len(old_rows) != 1 or len(new_rows) != 1:
            continue
        old = old_rows.iloc[0]
        new = new_rows.iloc[0]
        increment_rows.append(
            {
                "branch": branch,
                "comparison": f"{new_model}-{old_model}",
                "log_loss_improvement": float(old["log_loss"]) - float(new["log_loss"]),
                "brier_improvement": float(old["brier_score"]) - float(new["brier_score"]),
                "auc_improvement": float(new["auc"]) - float(old["auc"]),
                "average_precision_improvement": float(new["average_precision"])
                - float(old["average_precision"]),
            }
        )
    monthly_rows: list[dict[str, Any]] = []
    for branch, frame, old_model, new_model in (
        ("A", branch_a_monthly, "A0", "A1"),
        ("B", branch_b_monthly, "B0", "B1"),
        ("C", branch_c_monthly, "C0", "C1"),
    ):
        if not {"model", "group", "log_loss"}.issubset(frame.columns):
            continue
        old = frame.loc[frame["model"].eq(old_model)].set_index("group")
        new = frame.loc[frame["model"].eq(new_model)].set_index("group")
        shared = old.index.intersection(new.index)
        if shared.empty:
            continue
        improvement = old.loc[shared, "log_loss"] - new.loc[shared, "log_loss"]
        monthly_rows.append(
            {
                "branch": branch,
                "positive_log_loss_months": int(improvement.gt(0.0).sum()),
                "months": len(improvement),
                "minimum_monthly_log_loss_improvement": float(improvement.min()),
                "maximum_monthly_log_loss_improvement": float(improvement.max()),
            }
        )
    checkpoint_rows: list[dict[str, Any]] = []
    for branch, frame, old_model, new_model in (
        ("A", branch_a_subgroup, "A0", "A1"),
        ("B", branch_b_subgroup, "B0", "B1"),
    ):
        if not {"model", "group", "log_loss", "brier_score"}.issubset(frame.columns):
            continue
        checkpoint = frame.loc[frame["group"].astype(str).str.startswith("checkpoint_group=")]
        old = checkpoint.loc[checkpoint["model"].eq(old_model)].set_index("group")
        new = checkpoint.loc[checkpoint["model"].eq(new_model)].set_index("group")
        for group in old.index:
            checkpoint_rows.append(
                {
                    "branch": branch,
                    "checkpoint_group": str(group).removeprefix("checkpoint_group="),
                    "log_loss_improvement": float(old.loc[group, "log_loss"])
                    - float(new.loc[group, "log_loss"]),
                    "brier_improvement": float(old.loc[group, "brier_score"])
                    - float(new.loc[group, "brier_score"]),
                }
            )
    interval_80 = (
        bootstrap.loc[
            np.isclose(bootstrap["confidence"].to_numpy(float), 0.80),
            ["statistic", "lower", "upper"],
        ]
        if {"statistic", "confidence", "lower", "upper"}.issubset(bootstrap.columns)
        else pd.DataFrame()
    )

    def null_summary_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"null": name, "refits": len(frame)}
        for metric in (
            "log_loss_improvement",
            "brier_improvement",
            "auc_improvement",
            "average_precision_improvement",
        ):
            column = f"real_exceeds_null_{metric}"
            row[metric] = int(frame[column].astype(bool).sum()) if column in frame else 0
        return row

    null_summary = pd.DataFrame(
        [
            null_summary_row("front_options_bundle", front_null),
            null_summary_row("stock_structure_bundle", stock_null),
        ]
    )
    stock_centroids = pd.DataFrame(
        [
            {"regime": regime, **cast(Mapping[str, Any], centroid)}
            for regime, centroid in enumerate(
                cast(
                    Sequence[Mapping[str, Any]],
                    stock_reconstruction["canonical_centroids"],
                )
            )
        ]
    )
    stock_support = pd.DataFrame(
        [
            {"regime": int(regime), **cast(Mapping[str, Any], evidence)}
            for regime, evidence in cast(
                Mapping[str, Mapping[str, Any]],
                stock_reconstruction.get("regime_support", {}),
            ).items()
        ]
    )
    front_support = pd.DataFrame(
        [
            {"regime": int(regime), **cast(Mapping[str, Any], evidence)}
            for regime, evidence in cast(
                Mapping[str, Mapping[str, Any]], mapping.get("support", {})
            ).items()
        ]
    )
    mismatch_distributions = cast(
        Mapping[str, Mapping[str, Mapping[str, Any]]],
        mismatch.get("distributions", {}),
    )
    mismatch_assessment = pd.DataFrame(
        [
            {"feature": feature, **cast(Mapping[str, Any], values)}
            for feature, values in mismatch_distributions.get("assessment", {}).items()
        ]
    )
    concentration_max = (
        concentration.sort_values("share", ascending=False)
        .groupby(["population", "concentration_type"], observed=True)
        .head(1)
        .sort_values(["population", "concentration_type"])
        if {"share", "population", "concentration_type"}.issubset(concentration.columns)
        else pd.DataFrame()
    )
    route_indexed = route.set_index("route_state") if "route_state" in route else pd.DataFrame()
    broad_low_mean_residual = (
        float(route_indexed.loc["BROAD_CONFLICT", "mean_iv_residual"])
        - float(route_indexed.loc["LOW_ROUTE_SUPPORT", "mean_iv_residual"])
        if "mean_iv_residual" in route_indexed
        and {"BROAD_CONFLICT", "LOW_ROUTE_SUPPORT"}.issubset(route_indexed.index)
        else math.nan
    )
    stock_support_values = cast(Mapping[str, Any], stock_reconstruction["support"])
    stock_feature_retention = float(stock_support_values["daily_stock_feature_retention"])
    pair_summary = (
        f"Front pairs reconstructed: `{pair_reconstruction['selected_pairs']}` "
        "stock-sessions "
        f"(`{pair_reconstruction['development_selected_pairs']}` development, "
        f"`{pair_reconstruction['assessment_selected_pairs']}` assessment). Same-day "
        "or future observations: "
        f"`{pair_reconstruction['same_day_or_future_observations']}`. Selection was "
        "rebuilt from the repaired cached chains: "
        f"`{pair_reconstruction['selection_rebuilt_from_cached_chains']}`; "
        "selected-contract mismatches: "
        f"`{pair_reconstruction['selected_contract_mismatches']}`. All eight front "
        "raw features had finite development support: "
        f"`{feature_manifest.get('front_feature_support_passed')}`."
        if pair_reconstruction
        else "Front-options pair reconstruction was unavailable for this branch."
    )
    bootstrap_statistic_count = (
        int(bootstrap["statistic"].nunique()) if "statistic" in bootstrap else 0
    )
    broad_low_text = (
        f"`{broad_low_mean_residual:.9f}`"
        if math.isfinite(broad_low_mean_residual)
        else "unavailable"
    )
    return f"""# Daily Stock + Front-Options Context Quick Screen V0.1

## Result

Overall decision: `{decision["overall_decision"]}`.

Branches were run independently. Branch A used the full frozen structural panel;
Branch B used exact previous-close front options without term structure; Branch C
tested underlying 15-minute movement relative to prior-close ATM IV. No option P&L
was calculated.

Component statuses:

- Daily stock context: `{decision["daily_stock_context_status"]}`
- Front-options regimes: `{decision["front_options_regime_status"]}`
- Front-options completion context: `{decision["front_options_completion_status"]}`
- Stock structure to IV excess: `{decision["stock_to_iv_excess_status"]}`
- Broad-conflict IV residual: `{decision["broad_conflict_iv_residual_status"]}`
- Back-expiry schema preflight: `{decision["back_expiry_preflight_status"]}`

Structural reconstruction passed: `{structural["passed"]}`; clean rows:
`{structural["clean_advance_rows"]}`; assessment rows:
`{structural["assessment_clean_rows"]}`; row, route-state, and target mismatches:
`{structural["row_identity_mismatches"]}`,
`{structural["route_state_mismatches"]}`, `{structural["target_mismatches"]}`.

## Branch A

Support: `{json.dumps(branch_support_value.get("A"), sort_keys=True)}`.

{_markdown_metric_table(branch_a)}

## Branch B

Support: `{json.dumps(branch_support_value.get("B"), sort_keys=True)}`.

{_markdown_metric_table(branch_b)}

## Branch C

Support: `{json.dumps(branch_support_value.get("C"), sort_keys=True)}`.

{_markdown_metric_table(branch_c)}

### Primary increments

{markdown_table(pd.DataFrame(increment_rows), float_digits=9)}

### IV-relative movement by frozen route state

{markdown_table(route, float_digits=8)}

The binding BROAD_CONFLICT-minus-LOW_ROUTE_SUPPORT mean residual is
{broad_low_text}.

## Daily-stock reconstruction and regimes

Daily-stock reconstruction passed: `{stock_reconstruction["passed"]}`. Assessment
feature retention: `{stock_feature_retention:.6%}`.
Maximum dimension difference:
`{stock_reconstruction["maximum_daily_stock_dimension_difference"]}`; maximum
posterior difference:
`{stock_reconstruction["maximum_daily_stock_posterior_difference"]}`.

{markdown_table(stock_support, float_digits=6)}

{markdown_table(stock_centroids, float_digits=4)}

## Front-options regime centroids

{markdown_table(pd.DataFrame(centroid_rows), float_digits=4)}

{markdown_table(front_support, float_digits=6)}

{pair_summary}

## Mismatch distributions

{markdown_table(mismatch_assessment, float_digits=6)}

## Monthly, checkpoint, bootstrap, and null stability

{markdown_table(pd.DataFrame(monthly_rows), float_digits=9)}

{markdown_table(pd.DataFrame(checkpoint_rows), float_digits=9)}

The complete 80% interval surface is:

{markdown_table(interval_80, float_digits=9)}

{markdown_table(null_summary, float_digits=0)}

## Resampling and nulls

The bootstrap contains `{bootstrap_statistic_count}` statistics and exactly
ten fixed-seed, fixed-prediction, whole-session draws with 80%, 90%, and 95%
intervals. Front-options null refits: `{len(front_null)}`. Stock-structure null
refits: `{len(stock_null)}`. No bootstrap draw refit a model.

Maximum concentration:

{markdown_table(concentration_max, float_digits=6)}

## Back-expiry schema preflight

Status: `{preflight["status"]}`. Endpoint: `{preflight.get("endpoint", "unavailable")}`.
Exactly `{preflight.get("request_count", 0)}` request returned
`{preflight.get("record_count", 0)}` records;
`{preflight.get("exact_requested_date_records", 0)}` were exact-date rows and
`{preflight.get("back_expiry_dte_records", 0)}` were 46–90 DTE rows. Exact-date
filtering was possible: `{preflight.get("exact_date_filtering_possible", False)}`.
Canonical cache modified: `{preflight.get("canonical_cache_modified", False)}`.
The provider returned
`{preflight.get("protected_records_returned", 0)}` protected-date records despite the exact
request; they were rejected without persistence, and protected records persisted
were `{preflight.get("protected_records_persisted", 0)}`. The future plan preserves
non-compact identities and exact-date filtering; it is not a DTE recommendation.

Model branches reused
`{source.get("cached_provider_records_reused_by_predecessor_pair_reconstruction", 0)}`
cached provider records and downloaded zero model-branch records.

## Chronology and protection

All admitted option observations are from the exact prior US trading session. No
same-day or future option record was used. Protected market and option observations
dated 2025-08-23 or later are zero.
Materialised protected market rows:
`{protected["protected_market_rows_materialised"]}`; protected option observations:
`{protected["protected_option_observations_materialised"]}`.

## Reproducibility

Determinism passed: `{determinism.get("passed", False)}`. Joined-row mismatches:
`{determinism.get("joined_row_mismatches", "unavailable")}`. Maximum feature
difference: `{determinism.get("maximum_feature_difference", "unavailable")}`.
Maximum probability difference:
`{determinism.get("maximum_probability_difference", "unavailable")}`. Options were
not redownloaded, and bootstrap/null draws were not repeated.

Independent audit passed: `{audit["passed"]}`
(`{audit["checks_passed"]}/{audit["checks_run"]}` checks). Audit status:
`{decision.get("independent_audit_status", "pending")}`. The auditor made zero
provider requests and refit zero null models.

## Plots

{plot_lines}

## Scientific boundary

This retrospective screen does not establish option profitability, intraday option
fills, economic or directional edge, prospective validity, trading utility, or a
deployable strategy.
"""


def _write_empty_required_artifact(path: Path) -> None:
    if path.suffix == ".csv":
        write_csv(path, pd.DataFrame({"status": ["unavailable"]}))
    elif path.suffix == ".parquet":
        write_parquet(path, pd.DataFrame({"status": ["unavailable"]}))
    elif path.suffix == ".json":
        write_json(path, {**SAFETY_FLAGS, "status": "unavailable"})
    elif path.suffix == ".md":
        path.write_text("# Unavailable\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=DEFAULT_PROVIDER_ROOT,
        help="Local EODHD underlying five-minute parquet root.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the two bounded report plots.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Render reports from frozen artifacts without rebuilding models.",
    )
    parser.add_argument(
        "--chronology-only",
        action="store_true",
        help="Rebuild only chronology_audit.csv from frozen artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if arguments.report_only:
        report = render_report_from_artifacts()
        (PRIMARY / "report.md").write_text(report, encoding="utf-8")
        (REPORTS / "report.md").write_text(report, encoding="utf-8")
        print(str(read_json(PRIMARY / "decision.json")["overall_decision"]))
        return
    if arguments.chronology_only:
        front_raw = pd.read_parquet(PRIMARY / "front_options_raw_features.parquet")
        write_csv(PRIMARY / "chronology_audit.csv", chronology_audit(front_raw))
        print("chronology_audit_rebuilt")
        return
    contract_value = load_contract()
    write_json(PRIMARY / "contract.json", contract_value)
    if not (PRIMARY / "back_expiry_schema_preflight.json").is_file():
        write_json(
            PRIMARY / "back_expiry_schema_preflight.json",
            {
                **SAFETY_FLAGS,
                "status": "blocked_missing_eodhd_api_token",
                "request_attempted": False,
                "request_count": 0,
                "record_count": 0,
                "credential_recorded": False,
            },
        )
    if not (PRIMARY / "back_expiry_future_request_plan.json").is_file():
        write_json(
            PRIMARY / "back_expiry_future_request_plan.json",
            {
                **SAFETY_FLAGS,
                "preflight_status": "blocked_missing_eodhd_api_token",
                "future_acquisition_supported": False,
                "current_experiment_bulk_download": False,
            },
        )

    core = execute_core(arguments.provider_root)
    metrics = build_metric_outputs(core)
    if any(result is not None for result in (core.branch_a, core.branch_b, core.branch_c)):
        bootstrap = bootstrap_intervals(
            core.branch_a,
            core.branch_b,
            core.branch_c,
        )
    else:
        bootstrap = pd.DataFrame()
    if core.branch_b is not None and core.mismatch_standardization is not None:
        front_null = front_options_null_refits(
            core.branch_b,
            standardization=core.mismatch_standardization,
        )
    else:
        front_null = pd.DataFrame()
    if core.branch_c is not None and core.mismatch_standardization is not None:
        stock_null = stock_structure_null_refits(
            core.branch_c,
            standardization=core.mismatch_standardization,
        )
    else:
        stock_null = pd.DataFrame()
    decision = build_decision(
        core,
        metrics,
        bootstrap,
        front_null,
        stock_null,
    )

    write_json(PRIMARY / "structural_panel_reconstruction.json", core.structural_reconstruction)
    write_json(PRIMARY / "daily_stock_reconstruction.json", core.stock_reconstruction)
    if core.front_pair_reconstruction is not None:
        write_json(
            PRIMARY / "front_options_pair_reconstruction.json",
            core.front_pair_reconstruction,
        )
    if core.front_raw is not None:
        write_parquet(PRIMARY / "front_options_raw_features.parquet", core.front_raw)
    if core.front_dimensions is not None:
        write_parquet(
            PRIMARY / "front_options_dimensions.parquet",
            core.front_dimensions,
        )
    if core.front_parameters is not None and core.front_raw is not None:
        manifest = _dimension_manifest(
            core.front_parameters,
            front_raw=core.front_raw,
        )
        manifest["development_finite_raw_feature_counts"] = {
            feature: int(
                pd.to_numeric(
                    core.front_raw.loc[
                        core.front_raw["period"].eq("development"),
                        feature,
                    ],
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .sum()
            )
            for feature in FRONT_OPTIONS_RAW_FEATURES
        }
        manifest["front_feature_support_passed"] = all(
            int(value) > 0
            for value in cast(
                Mapping[str, Any],
                manifest["development_finite_raw_feature_counts"],
            ).values()
        )
        write_json(PRIMARY / "front_options_feature_manifest.json", manifest)
    if core.front_regime_mapping is not None:
        write_json(
            PRIMARY / "front_options_regime_mapping.json",
            core.front_regime_mapping,
        )
    if core.front_dimensions is not None and core.front_regime is not None:
        write_csv(
            PRIMARY / "front_options_regime_diagnostics.csv",
            front_regime_diagnostics(
                core.front_dimensions,
                core.front_regime,
            ),
        )
        write_csv(
            PRIMARY / "front_options_coverage.csv",
            front_coverage_table(core.structural, core.front_dimensions),
        )
    if core.joined_panel is not None:
        joined_to_write = core.joined_panel.copy()
        if core.movement_panel is not None:
            movement_columns = [
                "row_id",
                "entry_price",
                "close_15m",
                "absolute_log_return_15m",
                "iv_sigma_15m",
                "iv_expected_absolute_15m",
                TARGET_IV_EXCESS,
                "iv_absolute_residual_15m",
            ]
            joined_to_write = joined_to_write.merge(
                core.movement_panel.loc[:, movement_columns],
                on="row_id",
                how="left",
                validate="one_to_one",
            )
        write_parquet(
            PRIMARY / "front_options_cross_market_panel.parquet",
            joined_to_write,
        )
        if core.mismatch_standardization is None:
            raise AssertionError("joined panel lacks mismatch standardization")
        write_json(
            PRIMARY / "mismatch_feature_manifest.json",
            mismatch_manifest(joined_to_write, core.mismatch_standardization),
        )

    configurations, coefficients = model_artifacts(core)
    write_json(PRIMARY / "model_configurations.json", configurations)
    write_json(PRIMARY / "model_coefficients.json", coefficients)
    write_parquet(
        PRIMARY / "assessment_predictions.parquet",
        assessment_predictions(core),
    )
    write_csv(PRIMARY / "branch_a_metrics.csv", metrics.branch_a)
    write_csv(PRIMARY / "branch_a_monthly_metrics.csv", metrics.branch_a_monthly)
    write_csv(PRIMARY / "branch_a_regime_metrics.csv", metrics.branch_a_subgroup)
    write_csv(PRIMARY / "branch_b_metrics.csv", metrics.branch_b)
    write_csv(PRIMARY / "branch_b_monthly_metrics.csv", metrics.branch_b_monthly)
    write_csv(PRIMARY / "branch_b_regime_metrics.csv", metrics.branch_b_subgroup)
    write_csv(PRIMARY / "branch_c_metrics.csv", metrics.branch_c)
    write_csv(PRIMARY / "branch_c_monthly_metrics.csv", metrics.branch_c_monthly)
    write_csv(
        PRIMARY / "branch_c_route_state_metrics.csv",
        metrics.branch_c_route_state,
    )
    write_csv(PRIMARY / "bootstrap_metrics.csv", bootstrap)
    write_csv(PRIMARY / "front_options_null_metrics.csv", front_null)
    write_csv(PRIMARY / "stock_structure_null_metrics.csv", stock_null)
    if core.joined_panel is not None:
        write_csv(
            PRIMARY / "concentration_metrics.csv",
            concentration_metrics(core.branch_a_panel, core.joined_panel),
        )
    write_csv(PRIMARY / "chronology_audit.csv", chronology_audit(core.front_raw))
    protected = protected_boundary_audit(core)
    if not protected["passed"]:
        raise BranchBlocker(
            "all",
            "blocked_protected_boundary_failure",
            f"protected-boundary audit failed: {protected}",
        )
    write_json(PRIMARY / "protected_boundary_audit.json", protected)
    write_json(PRIMARY / "source_manifest.json", source_manifest(core))
    decision["determinism_status"] = "pending"
    decision["independent_audit_status"] = "pending"
    write_json(PRIMARY / "decision.json", decision)

    determinism = determinism_rebuild(
        core,
        provider_root=arguments.provider_root,
        first_decision=decision,
        frozen_bootstrap=bootstrap,
        frozen_front_null=front_null,
        frozen_stock_null=stock_null,
    )
    write_json(PRIMARY / "determinism_check.json", determinism)
    decision["determinism_status"] = "passed" if determinism["passed"] else "failed"
    if not determinism["passed"]:
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
    write_json(PRIMARY / "decision.json", decision)

    all_models = [
        model
        for result in (core.branch_a, core.branch_b, core.branch_c)
        if result is not None
        for model in result.models.values()
    ]
    manual_differences: dict[str, float] = {}
    for result in (core.branch_a, core.branch_b, core.branch_c):
        if result is None:
            continue
        sample = result.assessment.sort_values("row_id", kind="mergesort").head(100)
        for model_id, model in result.models.items():
            manual = manual_model_prediction(sample, model.as_dict())
            manual_differences[model_id] = float(np.max(np.abs(model.predict(sample) - manual)))
    chronology = chronology_audit(core.front_raw)
    lightweight = {
        **SAFETY_FLAGS,
        "audit_kind": "runner_lightweight_pre_audit",
        "branch_isolation": {
            "branch_a_does_not_require_front_options": True,
            "branch_b_does_not_require_movement_outcomes": True,
            "back_expiry_preflight_does_not_block_models": True,
        },
        "chronology_rows": len(chronology),
        "chronology_failures": int((~chronology["chronology_passed"].astype(bool)).sum()),
        "manual_probability_rows_per_model": 100 if all_models else 0,
        "manual_probability_maximum_difference_by_model": manual_differences,
        "maximum_manual_probability_difference": max(manual_differences.values(), default=0.0),
        "primary_classifier_fits": len(all_models),
        "ridge_fits": 0,
        "bootstrap_draws": (0 if bootstrap.empty else int(bootstrap["draws"].max())),
        "front_options_null_refits": len(front_null),
        "stock_structure_null_refits": len(stock_null),
        "determinism_passed": bool(determinism["passed"]),
        "independent_audit_pending": True,
    }
    lightweight["passed"] = bool(
        lightweight["chronology_failures"] == 0
        and lightweight["maximum_manual_probability_difference"] <= 1e-12
        and lightweight["primary_classifier_fits"] <= 6
        and lightweight["ridge_fits"] == 0
        and lightweight["bootstrap_draws"] in {0, 10}
        and lightweight["front_options_null_refits"] in {0, 3}
        and lightweight["stock_structure_null_refits"] in {0, 3}
        and lightweight["determinism_passed"]
    )
    write_json(PRIMARY / "lightweight_audit.json", lightweight)

    if not arguments.skip_plots:
        create_plots(metrics, REPORTS)
    report = render_report_from_artifacts()
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    for artifact in REQUIRED_ARTIFACTS:
        path = PRIMARY / artifact
        if not path.exists():
            _write_empty_required_artifact(path)
    missing = [artifact for artifact in REQUIRED_ARTIFACTS if not (PRIMARY / artifact).exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    print(str(decision["overall_decision"]))


if __name__ == "__main__":
    main()
