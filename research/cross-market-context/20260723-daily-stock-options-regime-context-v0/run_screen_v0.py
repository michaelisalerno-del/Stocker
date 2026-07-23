#!/usr/bin/env python3
"""Run the bounded Daily Stock x Options Regime Context Quick Screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
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
OPTIONS_V01_SOURCE = (
    REPO_ROOT
    / "research"
    / "options-strategies"
    / "20260723-eodhd-fixed-overnight-options-v01"
    / "artifacts"
    / "primary"
    / "source_manifest.json"
)
STARTING_BRANCH = "agent/eodhd-fixed-options-strategy-quick-v01"
STARTING_SHA = "274a1b7"
FINAL_BRANCH = "agent/daily-stock-options-regime-context-quick-v0"

for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))

from stocker_data.calendars import get_market_calendar  # noqa: E402
from stocker_research.broad_conflict_advance_hazard_v02 import (  # noqa: E402
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    ROUTE_FEATURES,
)
from stocker_research.daily_soft_regimes_v0 import (  # noqa: E402
    DAILY_OPTIONS_DIMENSIONS,
    DAILY_STOCK_DIMENSIONS,
    FrozenDimensionParameters,
    FrozenSoftRegime,
    apply_options_dimensions,
    apply_soft_regime,
    apply_stock_dimensions,
    fit_options_dimension_parameters,
    fit_soft_regime,
    fit_stock_dimension_parameters,
)
from stocker_research.daily_stock_options_context_v0 import (  # noqa: E402
    ASSESSMENT_END,
    DAILY_OPTIONS_MISSING_INDICATORS,
    DAILY_OPTIONS_RAW_FEATURES,
    DAILY_STOCK_RAW_FEATURES,
    DEVELOPMENT_START,
    FROZEN_COHORT,
    MISMATCH_FEATURES,
    PROTECTED_START,
    SAFETY_FLAGS,
    MeanStandardization,
    add_mismatch_features,
    assert_safety_flags,
    calculate_daily_stock_raw_features,
    choose_daily_context_decision,
    fit_mismatch_standardization,
    iv_horizon_outcomes,
    permute_bundle_within_slates,
    previous_us_trading_session,
    reject_protected_observations,
    select_daily_options_surface,
    validate_daily_context_chronology,
)
from stocker_research.stock_options_cross_market_quick_v0 import (  # noqa: E402
    FrozenCrossMarketModel,
    binary_metrics,
    continuous_residual_metrics,
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
OPTIONS_REGIME_FEATURES = (
    *(f"daily_options_regime_p_{value}" for value in range(4)),
    "daily_options_regime_entropy",
    "daily_options_regime_margin",
)
STOCK_CONTEXT_FEATURES = (*DAILY_STOCK_DIMENSIONS, *STOCK_REGIME_FEATURES)
OPTIONS_CONTEXT_FEATURES = (
    *DAILY_OPTIONS_DIMENSIONS,
    *OPTIONS_REGIME_FEATURES,
    *DAILY_OPTIONS_MISSING_INDICATORS,
)
S0_FEATURES = (*DENSE_H0_FEATURES, *ROUTE_FEATURES)
S1_FEATURES = (*S0_FEATURES, *STOCK_CONTEXT_FEATURES)
S2_FEATURES = (*S1_FEATURES, *OPTIONS_CONTEXT_FEATURES, *MISMATCH_FEATURES)
O0_FEATURES = (*OPTIONS_CONTEXT_FEATURES, *CHECKPOINT_FEATURES)
O1_FEATURES = (*O0_FEATURES, *STOCK_CONTEXT_FEATURES)
O2_FEATURES = (
    *O1_FEATURES,
    *H0_NON_CLOCK_FEATURES,
    *ROUTE_FEATURES,
    *MISMATCH_FEATURES,
)
R0_FEATURES = O0_FEATURES
R1_FEATURES = O2_FEATURES
OPTIONS_NULL_SEEDS = (20260723, 20260724, 20260725)
ROUTE_NULL_SEEDS = (20260726, 20260727, 20260728)
DAILY_HISTORY_START = date(2023, 10, 1)
DEFAULT_PROVIDER_ROOT = (
    Path.home() / "StockerLocal" / "data" / "processed" / "source=eodhd" / "instrument_type=stock"
)
DOWNLOAD_RECEIPT = PRIMARY / "download_gap_receipt.json"


@dataclass(frozen=True)
class DevelopmentCutoffs:
    """Development-frozen subgroup thresholds used unchanged in assessment."""

    mismatch_compression_vs_iv: float
    mismatch_complacent_conflict: float
    options_implied_tension: float
    options_front_urgency: float
    mismatch_route_vs_premium: float


REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "chronology_audit.csv",
    "structural_panel_reconstruction.json",
    "daily_stock_raw_features.parquet",
    "daily_stock_dimensions.parquet",
    "daily_stock_feature_manifest.json",
    "daily_stock_regime_mapping.json",
    "daily_stock_regime_diagnostics.csv",
    "daily_options_raw_features.parquet",
    "daily_options_dimensions.parquet",
    "daily_options_feature_manifest.json",
    "daily_options_regime_mapping.json",
    "daily_options_regime_diagnostics.csv",
    "daily_options_coverage_gap.csv",
    "daily_cross_market_panel.parquet",
    "mismatch_feature_manifest.json",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "test_a_metrics.csv",
    "test_a_monthly_metrics.csv",
    "test_a_regime_metrics.csv",
    "test_b_metrics.csv",
    "test_b_monthly_metrics.csv",
    "test_b_regime_metrics.csv",
    "continuous_residual_metrics.csv",
    "persistence_horizon_metrics.csv",
    "regime_pair_persistence_metrics.csv",
    "dte_horizon_mapping.csv",
    "bootstrap_metrics.csv",
    "options_null_metrics.csv",
    "route_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "lightweight_audit.json",
    "determinism_check.json",
    "report.md",
)


class ScreenBlocker(RuntimeError):
    """A preregistered fail-closed terminal condition."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (date, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
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


def contract() -> dict[str, Any]:
    value = read_json(EXPERIMENT_DIR / "contract.json")
    assert_safety_flags(value)
    return value


def completed_download_receipt() -> dict[str, Any] | None:
    """Return a credential-free completed recovery receipt, if one is reusable."""

    if not DOWNLOAD_RECEIPT.is_file():
        return None
    receipt = read_json(DOWNLOAD_RECEIPT)
    if receipt.get("status") != "completed":
        return None
    records = int(receipt.get("newly_downloaded_records", 0))
    raw_bytes = int(receipt.get("newly_downloaded_bytes", 0))
    if records > 500_000 or raw_bytes > 5 * 1024**3:
        raise ScreenBlocker(
            "blocked_quick_resource_limit",
            "completed recovery receipt exceeds the preregistered download cap",
        )
    output_cache = receipt.get("output_cache")
    if not isinstance(output_cache, str) or not Path(output_cache).is_file():
        return None
    return receipt


def discover_options_cache(explicit: Path | None) -> tuple[Path, dict[str, Any] | None]:
    if explicit is not None:
        path = explicit.resolve()
        candidate_receipt = completed_download_receipt()
        receipt = (
            candidate_receipt
            if candidate_receipt is not None
            and Path(str(candidate_receipt["output_cache"])).resolve() == path
            else None
        )
    else:
        receipt = completed_download_receipt()
        if receipt is not None:
            path = Path(str(receipt["output_cache"])).resolve()
        else:
            source = read_json(OPTIONS_V01_SOURCE)
            cache = cast(Mapping[str, Any], source["options_cache"])
            path = Path(str(cache["canonical_cache_path"])).resolve()
    if not path.is_file():
        raise ScreenBlocker(
            "blocked_insufficient_daily_options_coverage",
            f"repaired exact-date canonical cache is unavailable: {path}",
        )
    return path, receipt


def load_structural_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    dense = pd.read_parquet(DENSE_PANEL)
    structural, reconstruction = reconstruct_clean_structural_panel(dense)
    reconstruction = {**SAFETY_FLAGS, **reconstruction}
    if (
        int(reconstruction["row_identity_mismatches"]) != 0
        or int(reconstruction["route_state_mismatches"]) != 0
        or int(reconstruction["target_mismatches"]) != 0
        or float(reconstruction["maximum_shared_feature_difference"]) > 1e-12
    ):
        raise ScreenBlocker(
            "blocked_structural_panel_reconstruction_failure",
            "frozen dense clean-advance panel did not reconstruct exactly",
        )
    return structural, reconstruction


def load_trace() -> pd.DataFrame:
    trace = pd.read_parquet(TRACE_PANEL)
    trace["session"] = trace["session"].astype(str)
    reject_protected_observations(
        trace.loc[
            pd.to_datetime(trace["session"]).dt.date.le(ASSESSMENT_END)
            | pd.to_datetime(trace["session"]).dt.date.ge(PROTECTED_START)
        ],
        date_columns=("session",),
    )
    trace = trace.loc[
        trace["symbol"].isin(FROZEN_COHORT)
        & pd.to_datetime(trace["session"]).dt.date.le(ASSESSMENT_END)
    ].copy()
    if trace.empty:
        raise ScreenBlocker("blocked_daily_stock_feature_failure", "intraday trace is empty")
    if pd.to_datetime(trace["session"]).dt.date.ge(PROTECTED_START).any():
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "protected trace rows were materialised"
        )
    return trace


def load_full_regular_session_bars(
    provider_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only complete, exchange-calendar regular-session five-minute bars."""

    calendar = get_market_calendar("XNYS")
    schedule = (
        calendar.schedule(
            start_date=DAILY_HISTORY_START.isoformat(),
            end_date=ASSESSMENT_END.isoformat(),
        )
        .rename_axis("session")
        .reset_index()
    )
    schedule["session"] = pd.to_datetime(schedule["session"], errors="raise").dt.date
    pieces: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    off_grid_rows_discarded = 0
    protected_rows_materialised = 0
    for symbol in FROZEN_COHORT:
        path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        if not path.is_file():
            raise ScreenBlocker(
                "blocked_daily_stock_feature_failure",
                f"full five-minute underlying source is unavailable for {symbol}: {path}",
            )
        source_hashes[symbol] = sha256_file(path)
        raw = pd.read_parquet(
            path,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
            filters=[
                ("timestamp", ">=", pd.Timestamp(DAILY_HISTORY_START, tz="UTC")),
                ("timestamp", "<", pd.Timestamp(PROTECTED_START, tz="UTC")),
            ],
        )
        timestamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
        protected_rows_materialised += int(
            timestamps.ge(pd.Timestamp(PROTECTED_START, tz="UTC")).sum()
        )
        working = raw.assign(
            timestamp=timestamps,
            symbol=symbol,
            session=timestamps.dt.tz_convert("America/New_York").dt.date,
        ).merge(schedule, on="session", how="inner", validate="many_to_one")
        working = working.loc[
            working["timestamp"].ge(working["market_open"])
            & working["timestamp"].lt(working["market_close"])
        ].copy()
        elapsed_seconds = (working["timestamp"] - working["market_open"]).dt.total_seconds()
        on_grid = elapsed_seconds.mod(300.0).eq(0.0)
        off_grid_rows_discarded += int((~on_grid).sum())
        working = working.loc[on_grid].copy()
        working["bar_ordinal"] = (elapsed_seconds.loc[on_grid] / 300.0).astype(int)
        expected_bars = (
            (working["market_close"] - working["market_open"]).dt.total_seconds() / 300.0
        ).astype(int)
        working["expected_session_bars"] = expected_bars
        pieces.append(working)
    if protected_rows_materialised:
        raise ScreenBlocker(
            "blocked_protected_boundary_failure",
            f"full-bar loader materialised {protected_rows_materialised} protected rows",
        )
    bars = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort")
        .reset_index(drop=True)
    )
    if bars.duplicated(["symbol", "session", "bar_ordinal"]).any():
        raise ScreenBlocker(
            "blocked_daily_stock_feature_failure",
            "full regular-session source has duplicate stock-session bar ordinals",
        )
    observed_counts = bars.groupby(["symbol", "session"], observed=True)["bar_ordinal"].size()
    expected_counts = bars.groupby(["symbol", "session"], observed=True)[
        "expected_session_bars"
    ].first()
    incomplete = observed_counts.ne(expected_counts)
    if bool(incomplete.any()):
        raise ScreenBlocker(
            "blocked_daily_stock_feature_failure",
            f"full regular-session source has {int(incomplete.sum())} incomplete stock-sessions",
        )
    nonfinite_ohlc_rows = int(
        (~np.isfinite(bars[["open", "high", "low", "close"]].to_numpy(float)).all(axis=1)).sum()
    )
    complete_daily_groups = int(
        bars.assign(
            _valid=np.isfinite(bars[["open", "high", "low", "close"]].to_numpy(float)).all(axis=1)
        )
        .groupby(["symbol", "session"], observed=True)["_valid"]
        .all()
        .sum()
    )
    manifest = {
        "provider_root": str(provider_root),
        "timeframe": "5m",
        "calendar": "XNYS",
        "history_read_start": DAILY_HISTORY_START.isoformat(),
        "maximum_observation_date": str(bars["session"].max()),
        "regular_session_rows": len(bars),
        "stock_sessions": int(observed_counts.size),
        "off_grid_rows_discarded": off_grid_rows_discarded,
        "regular_session_activity_missing_rows": int(bars["volume"].isna().sum()),
        "regular_session_ohlc_missing_rows": nonfinite_ohlc_rows,
        "complete_daily_stock_sessions": complete_daily_groups,
        "minimum_session_bars": int(observed_counts.min()),
        "maximum_session_bars": int(observed_counts.max()),
        "source_sha256_by_stock": source_hashes,
        "protected_rows_materialised": protected_rows_materialised,
    }
    bars["session"] = bars["session"].astype(str)
    return bars, manifest


def audit_full_bar_overlap(trace: pd.DataFrame, full_bars: pd.DataFrame) -> dict[str, Any]:
    """Verify the full-session source exactly matches the frozen causal trace."""

    columns = ["symbol", "session", "bar_ordinal", "open", "high", "low", "close", "volume"]
    left = trace.loc[:, columns].copy()
    right = full_bars.loc[:, columns].copy()
    joined = left.merge(
        right,
        on=["symbol", "session", "bar_ordinal"],
        how="left",
        validate="one_to_one",
        suffixes=("_trace", "_full"),
        indicator=True,
    )
    missing = int(joined["_merge"].ne("both").sum())
    differences: list[float] = []
    for column in ("open", "high", "low", "close", "volume"):
        trace_values = pd.to_numeric(joined[f"{column}_trace"], errors="coerce").to_numpy(float)
        full_values = pd.to_numeric(joined[f"{column}_full"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(trace_values) & np.isnan(full_values)
        finite = np.isfinite(trace_values) & np.isfinite(full_values)
        if bool((~both_nan & ~finite).any()):
            differences.append(math.inf)
        elif bool(finite.any()):
            differences.append(float(np.max(np.abs(trace_values[finite] - full_values[finite]))))
    maximum_difference = max(differences, default=0.0)
    result = {
        "trace_rows": len(trace),
        "matched_rows": int(joined["_merge"].eq("both").sum()),
        "missing_rows": missing,
        "maximum_ohlcv_difference": maximum_difference,
    }
    if missing or maximum_difference > 1e-12:
        raise ScreenBlocker(
            "blocked_daily_stock_feature_failure",
            f"full-bar source differs from frozen causal trace: {result}",
        )
    return result


def aggregate_daily_bars(full_bars: pd.DataFrame) -> pd.DataFrame:
    ordered = full_bars.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort")
    ordered["_ohlc_valid"] = np.isfinite(
        ordered[["open", "high", "low", "close"]].to_numpy(float)
    ).all(axis=1)
    ordered["_activity_valid"] = np.isfinite(
        pd.to_numeric(ordered["volume"], errors="coerce").to_numpy(float)
    )
    complete = (
        ordered.groupby(["symbol", "session"], observed=True)["_ohlc_valid"]
        .transform("all")
        .astype(bool)
    )
    ordered = ordered.loc[complete].copy()
    if ordered.empty:
        raise ScreenBlocker(
            "blocked_daily_stock_feature_failure",
            "no complete regular-session daily bars are available",
        )
    daily = (
        ordered.groupby(["symbol", "session"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            activity=("volume", "sum"),
            activity_complete=("_activity_valid", "all"),
            bars=("bar_ordinal", "size"),
        )
        .reset_index()
    )
    daily.loc[~daily["activity_complete"].astype(bool), "activity"] = np.nan
    daily = daily.drop(columns="activity_complete")
    return daily


def build_stock_context(
    structural: pd.DataFrame, daily_bars: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    FrozenDimensionParameters,
    FrozenSoftRegime,
    dict[str, Any],
]:
    raw_by_information_date = calculate_daily_stock_raw_features(daily_bars)
    required = (
        structural.loc[:, ["symbol", "session", "period"]]
        .drop_duplicates()
        .sort_values(["symbol", "session"], kind="mergesort")
        .reset_index(drop=True)
    )
    previous_dates = {
        signal: previous_us_trading_session(date.fromisoformat(signal)).isoformat()
        for signal in sorted(set(required["session"].astype(str)))
    }
    required["stock_information_date"] = required["session"].astype(str).map(previous_dates)
    raw = raw_by_information_date.rename(columns={"session": "stock_information_date"})
    raw["stock_information_date"] = raw["stock_information_date"].astype(str)
    context = required.merge(
        raw,
        on=["symbol", "stock_information_date"],
        how="left",
        validate="many_to_one",
    )
    development = context.loc[context["period"].eq("development")]
    parameters = fit_stock_dimension_parameters(development)
    dimensions = apply_stock_dimensions(context, parameters)
    complete = dimensions.loc[
        dimensions.loc[:, list(DAILY_STOCK_DIMENSIONS)].notna().all(axis=1)
    ].copy()
    fit_rows = complete.loc[complete["period"].eq("development")].copy()
    try:
        regime = fit_soft_regime(
            fit_rows,
            dimensions=DAILY_STOCK_DIMENSIONS,
            missing_indicators=(),
            canonical_dimensions=(
                "daily_compression",
                "daily_volatility_acceleration",
                "daily_directional_efficiency",
                "daily_extension",
                "daily_rejection",
                "daily_relative_strength",
            ),
            prefix="daily_stock_regime",
        )
        assigned = apply_soft_regime(complete, regime)
    except (RuntimeError, ValueError) as error:
        raise ScreenBlocker(
            "blocked_daily_stock_regime_failure",
            f"daily stock regime fit failed: {type(error).__name__}: {error}",
        ) from error
    retention = float(
        dimensions.loc[dimensions["period"].eq("assessment"), list(DAILY_STOCK_RAW_FEATURES)]
        .notna()
        .all(axis=1)
        .mean()
    )
    assessment = assigned.loc[assigned["period"].eq("assessment")]
    support = {
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_months": int(pd.to_datetime(assessment["session"]).dt.to_period("M").nunique()),
        "daily_stock_feature_retention": retention,
    }
    support["passed"] = bool(
        support["assessment_stocks"] >= 15
        and support["assessment_sessions"] >= 140
        and support["assessment_months"] == 8
        and retention >= 0.95
    )
    if not support["passed"]:
        raise ScreenBlocker(
            "blocked_daily_stock_feature_failure", f"daily stock support failed: {support}"
        )
    return context, assigned, parameters, regime, support


def load_options_cache(path: Path) -> pd.DataFrame:
    cache = pd.read_parquet(path)
    required = {
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
    }
    if missing := sorted(required.difference(cache.columns)):
        raise ScreenBlocker(
            "blocked_daily_options_schema_failure", f"canonical options schema missing: {missing}"
        )
    cache["trade_date"] = pd.to_datetime(cache["trade_date"], errors="raise").dt.date
    try:
        reject_protected_observations(cache, date_columns=("trade_date",))
    except ValueError as error:
        raise ScreenBlocker("blocked_protected_boundary_failure", str(error)) from error
    cache["underlying_symbol"] = cache["underlying_symbol"].astype(str)
    return cache


def build_options_context(
    stock_context: pd.DataFrame,
    cache: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    groups = {
        (str(symbol), cast(date, observation_date)): group.copy()
        for (symbol, observation_date), group in cache.groupby(
            ["underlying_symbol", "trade_date"], sort=False, observed=True
        )
    }
    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    reused_records = 0
    for source in stock_context.itertuples(index=False):
        signal_date = date.fromisoformat(str(source.session))
        information_date = date.fromisoformat(str(source.stock_information_date))
        validate_daily_context_chronology(
            signal_date=signal_date,
            stock_information_date=information_date,
            options_observation_date=information_date,
        )
        base = {
            "symbol": str(source.symbol),
            "session": signal_date.isoformat(),
            "period": str(source.period),
            "required_options_date": information_date.isoformat(),
        }
        chain = groups.get((str(source.symbol), information_date))
        realised = getattr(source, "realised_volatility_20d", math.nan)
        previous_close = getattr(source, "unadjusted_close", math.nan)
        if chain is None:
            result: dict[str, object] = {
                "pair_available": False,
                "pair_reason": "missing_exact_chain",
            }
        elif not math.isfinite(float(realised)) or not math.isfinite(float(previous_close)):
            result = {
                "pair_available": False,
                "pair_reason": "missing_daily_stock_volatility_or_close",
            }
        else:
            reused_records += len(chain)
            try:
                result = select_daily_options_surface(
                    chain,
                    previous_close=float(previous_close),
                    realised_volatility_20d=float(realised),
                )
            except ValueError as error:
                raise ScreenBlocker(
                    "blocked_daily_options_schema_failure",
                    f"options surface failed for {source.symbol}/{information_date}: {error}",
                ) from error
        row = {**base, **result}
        rows.append(row)
        if not bool(result["pair_available"]):
            gaps.append(
                {
                    **base,
                    "gap_component": "front_atm_pair",
                    "gap_reason": str(result["pair_reason"]),
                    "bounded_download_required": str(result["pair_reason"])
                    == "missing_exact_chain",
                }
            )
            continue
        optional_gaps = (
            ("front_25_delta_skew", "skew_missing"),
            ("back_atm_pair", "back_expiry_missing"),
            ("front_expiry_open_interest", "oi_concentration_missing"),
            ("front_expiry_open_interest", "call_put_oi_imbalance_missing"),
        )
        for component, missing_column in optional_gaps:
            if bool(result[missing_column]):
                gaps.append(
                    {
                        **base,
                        "gap_component": component,
                        "gap_reason": f"missing_{missing_column.removesuffix('_missing')}",
                        "bounded_download_required": True,
                    }
                )
    raw = pd.DataFrame(rows)
    gap = pd.DataFrame(
        gaps,
        columns=[
            "symbol",
            "session",
            "period",
            "required_options_date",
            "gap_component",
            "gap_reason",
            "bounded_download_required",
        ],
    )
    return raw, gap, reused_records


def fit_options_context(
    raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    FrozenDimensionParameters,
    FrozenSoftRegime,
]:
    available = raw.loc[raw["pair_available"].astype(bool)].copy()
    if available.empty:
        raise ScreenBlocker(
            "blocked_insufficient_daily_options_coverage", "no valid previous-close front pairs"
        )
    parameters = fit_options_dimension_parameters(
        available.loc[available["period"].eq("development")]
    )
    dimensions = apply_options_dimensions(available, parameters)
    try:
        regime = fit_soft_regime(
            dimensions.loc[dimensions["period"].eq("development")],
            dimensions=DAILY_OPTIONS_DIMENSIONS,
            missing_indicators=DAILY_OPTIONS_MISSING_INDICATORS,
            canonical_dimensions=(
                "options_implied_tension",
                "options_premium_richness",
                "options_front_urgency",
                "options_downside_asymmetry",
                "options_liquidity_stress",
                "options_positioning_concentration",
            ),
            prefix="daily_options_regime",
        )
        assigned = apply_soft_regime(dimensions, regime)
    except (RuntimeError, ValueError) as error:
        raise ScreenBlocker(
            "blocked_daily_options_regime_failure",
            f"daily options regime fit failed: {type(error).__name__}: {error}",
        ) from error
    return assigned, parameters, regime


def regime_mapping(fitted: FrozenSoftRegime) -> dict[str, Any]:
    weights = fitted.estimator.weights_[list(fitted.canonical_to_original)]
    covariances = fitted.estimator.covariances_[list(fitted.canonical_to_original)]
    return {
        **SAFETY_FLAGS,
        "prefix": fitted.prefix,
        "fitted_period": fitted.fitted_period,
        "n_components": 4,
        "covariance_type": "diag",
        "reg_covar": 1e-5,
        "n_init": 5,
        "max_iter": 300,
        "random_state": 20260723,
        "input_columns": fitted.input_columns,
        "canonical_dimensions": fitted.canonical_dimensions,
        "canonical_to_original": fitted.canonical_to_original,
        "original_to_canonical": fitted.original_to_canonical,
        "canonical_centroids": fitted.canonical_centroids,
        "canonical_input_means": fitted.estimator.means_[list(fitted.canonical_to_original)],
        "input_medians": fitted.input_medians,
        "canonical_weights": weights,
        "canonical_covariances": covariances,
        "iterations": int(fitted.estimator.n_iter_),
        "converged": bool(fitted.estimator.converged_),
    }


def dimension_manifest(parameters: FrozenDimensionParameters) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "kind": parameters.kind,
        "fitted_period": parameters.fitted_period,
        "scales": {name: asdict(value) for name, value in parameters.scales.items()},
        "imputation_medians": parameters.imputation_medians,
    }


def regime_diagnostics(frame: pd.DataFrame, fitted: FrozenSoftRegime) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prefix = fitted.prefix
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
                    float(stock_mass.max() / stock_mass.sum()) if stock_mass.sum() > 0 else math.nan
                ),
                "maximum_month_share": (
                    float(month_mass.max() / month_mass.sum()) if month_mass.sum() > 0 else math.nan
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
            for dimension in fitted.dimensions:
                row[f"mean_{dimension}"] = float(hard[dimension].mean()) if len(hard) else math.nan
            rows.append(row)
        for stock, stock_frame in period_frame.groupby("symbol", sort=True, observed=True):
            rows.append(
                {
                    "diagnostic_type": "support_by_stock",
                    "period": period,
                    "group": stock,
                    **{
                        f"posterior_mass_regime_{regime}": float(
                            stock_frame[f"{prefix}_p_{regime}"].sum() / len(stock_frame)
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
                    "group": month,
                    **{
                        f"posterior_mass_regime_{regime}": float(
                            month_frame[f"{prefix}_p_{regime}"].sum() / len(month_frame)
                        )
                        for regime in range(4)
                    },
                }
            )
    development = frame.loc[frame["period"].eq("development")]
    assessment = frame.loc[frame["period"].eq("assessment")]
    for dimension in fitted.dimensions:
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


def regime_support(frame: pd.DataFrame, prefix: str) -> tuple[bool, dict[int, dict[str, Any]]]:
    assessment = frame.loc[frame["period"].eq("assessment")].copy()
    months = pd.to_datetime(assessment["session"]).dt.to_period("M")
    evidence: dict[int, dict[str, Any]] = {}
    for regime in range(4):
        hard = assessment.loc[assessment[prefix].astype(int).eq(regime)]
        mass = float(assessment[f"{prefix}_p_{regime}"].sum() / len(assessment))
        evidence[regime] = {
            "posterior_mass": mass,
            "stocks": int(hard["symbol"].nunique()),
            "months": int(months.loc[hard.index].nunique()),
            "supported": bool(
                mass >= 0.05
                and hard["symbol"].nunique() >= 8
                and months.loc[hard.index].nunique() >= 4
            ),
        }
    return all(bool(value["supported"]) for value in evidence.values()), evidence


def attach_movement_horizons(
    panel: pd.DataFrame, full_bars: pd.DataFrame, daily_raw: pd.DataFrame
) -> pd.DataFrame:
    """Attach exact post-checkpoint movements through the third following close."""

    groups = {
        (str(symbol), str(session)): group.sort_values("bar_ordinal", kind="mergesort").set_index(
            "bar_ordinal", drop=False
        )
        for (symbol, session), group in full_bars.groupby(["symbol", "session"], sort=False)
    }
    calendar = get_market_calendar("XNYS")
    schedule = calendar.schedule(
        start_date=str(full_bars["session"].min()),
        end_date=ASSESSMENT_END.isoformat(),
    )
    market_sessions = [
        value.date().isoformat() for value in pd.to_datetime(schedule.index, errors="raise")
    ]
    session_positions = {session: index for index, session in enumerate(market_sessions)}
    split_map = {
        (str(row.symbol), str(row.session)): bool(row.inferred_corporate_action_boundary)
        for row in daily_raw.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for source in panel.to_dict(orient="records"):
        symbol = str(source["symbol"])
        session = str(source["session"])
        bars = groups[(symbol, session)]
        checkpoint = int(source["checkpoint_bar_ordinal_zero_based"])
        entry_ordinal = checkpoint + 1
        third_future_ordinal = checkpoint + 3
        if entry_ordinal not in bars.index or third_future_ordinal not in bars.index:
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure",
                f"future 15-minute bars unavailable: {source['row_id']}",
            )
        entry_price = float(bars.loc[entry_ordinal, "open"])
        close_15m = float(bars.loc[third_future_ordinal, "close"])
        same_close_value = float(bars.iloc[-1]["close"])
        same_close = same_close_value if math.isfinite(same_close_value) else None
        remaining_minutes = (int(bars["bar_ordinal"].max()) - checkpoint) * 5
        if session not in session_positions:
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure",
                f"signal session is not an XNYS session: {session}",
            )
        position = session_positions[session]
        next_session = (
            market_sessions[position + 1] if position + 1 < len(market_sessions) else None
        )
        third_session = (
            market_sessions[position + 3] if position + 3 < len(market_sessions) else None
        )
        next_close: float | None = None
        third_close: float | None = None
        next_split = False
        third_split = False
        if next_session is not None:
            next_split = bool(split_map.get((symbol, next_session), False))
            if not next_split and (symbol, next_session) in groups:
                value = float(groups[(symbol, next_session)].iloc[-1]["close"])
                next_close = value if math.isfinite(value) else None
        if third_session is not None:
            intervening = market_sessions[position + 1 : position + 4]
            third_split = any(bool(split_map.get((symbol, value), False)) for value in intervening)
            if not third_split and (symbol, third_session) in groups:
                value = float(groups[(symbol, third_session)].iloc[-1]["close"])
                third_close = value if math.isfinite(value) else None
        outcomes = iv_horizon_outcomes(
            entry_price=entry_price,
            atm_iv=float(source["atm_iv"]),
            close_15m=close_15m,
            same_session_close=same_close,
            next_session_close=next_close,
            third_session_close=third_close,
            remaining_regular_session_minutes=remaining_minutes,
        )
        rows.append(
            {
                **source,
                **outcomes,
                "next_close_session": next_session,
                "third_close_session": third_session,
                "next_close_corporate_action_excluded": next_split,
                "third_close_corporate_action_excluded": third_split,
            }
        )
    return pd.DataFrame(rows)


def join_cross_market_panel(
    structural: pd.DataFrame,
    stock_dimensions: pd.DataFrame,
    options_dimensions: pd.DataFrame,
    full_bars: pd.DataFrame,
    daily_raw_information: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, MeanStandardization]]:
    stock_columns = [
        "symbol",
        "session",
        "stock_information_date",
        "unadjusted_close",
        "realised_volatility_20d",
        "inferred_corporate_action_boundary",
        *DAILY_STOCK_RAW_FEATURES,
        *(f"{feature}_missing" for feature in DAILY_STOCK_RAW_FEATURES),
        *STOCK_CONTEXT_FEATURES,
        "daily_stock_regime",
        "daily_stock_regime_top_probability",
        "daily_stock_regime_mahalanobis_to_nearest_centroid",
    ]
    option_columns = [
        "symbol",
        "session",
        "required_options_date",
        "options_observation_date",
        "previous_close_underlying_price",
        "front_expiration_date",
        "front_strike",
        "front_call_contract_id",
        "front_put_contract_id",
        "back_expiration_date",
        "back_strike",
        "back_call_contract_id",
        "back_put_contract_id",
        "skew_put_contract_id",
        "skew_call_contract_id",
        "previous_close_chain_request_ids",
        *DAILY_OPTIONS_RAW_FEATURES,
        *OPTIONS_CONTEXT_FEATURES,
        "daily_options_regime",
        "daily_options_regime_top_probability",
        "daily_options_regime_mahalanobis_to_nearest_centroid",
    ]
    joined = structural.merge(
        stock_dimensions.loc[:, stock_columns],
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    ).merge(
        options_dimensions.loc[:, option_columns],
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    )
    joined["options_observation_date"] = joined["options_observation_date"].astype(str)
    for row in (
        joined.loc[:, ["session", "stock_information_date", "options_observation_date"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        validate_daily_context_chronology(
            signal_date=date.fromisoformat(str(row.session)),
            stock_information_date=date.fromisoformat(str(row.stock_information_date)),
            options_observation_date=date.fromisoformat(str(row.options_observation_date)),
        )
    standardization = fit_mismatch_standardization(joined.loc[joined["period"].eq("development")])
    joined = add_mismatch_features(joined, standardization)
    joined = attach_movement_horizons(joined, full_bars, daily_raw_information)
    joined["checkpoint_group"] = pd.cut(
        joined["checkpoint"],
        bins=[5, 14, 24, 34],
        labels=["early_6_14", "middle_16_24", "late_26_34"],
        include_lowest=True,
    ).astype(str)
    return joined.sort_values("row_id", kind="mergesort").reset_index(drop=True), standardization


def options_coverage(structural: pd.DataFrame, options_dimensions: pd.DataFrame) -> dict[str, Any]:
    assessment_pairs = options_dimensions.loc[options_dimensions["period"].eq("assessment")]
    joined = structural.merge(
        assessment_pairs.loc[:, ["symbol", "session"]],
        on=["symbol", "session"],
        how="inner",
        validate="many_to_one",
    )
    assessment = joined.loc[joined["period"].eq("assessment")]
    evidence = {
        "assessment_clean_checkpoint_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(pd.to_datetime(assessment["session"]).dt.to_period("M").nunique()),
        "broad_conflict_rows": int(assessment["BROAD_CONFLICT"].sum()),
        "low_route_support_rows": int(assessment["LOW_ROUTE_SUPPORT"].sum()),
    }
    evidence["passed"] = bool(
        evidence["assessment_clean_checkpoint_rows"] >= 3_000
        and evidence["assessment_sessions"] >= 100
        and evidence["assessment_stocks"] >= 10
        and evidence["assessment_months"] >= 5
        and evidence["broad_conflict_rows"] >= 200
        and evidence["low_route_support_rows"] >= 200
    )
    return evidence


def raw_options_support(
    structural: pd.DataFrame,
    options_raw: pd.DataFrame,
) -> dict[str, Any]:
    """Summarise front-pair coverage and whether every frozen dimension is estimable."""

    available = options_raw.loc[options_raw["pair_available"].astype(bool)].copy()
    coverage = options_coverage(structural, available)
    development = available.loc[available["period"].eq("development")]
    assessment = available.loc[available["period"].eq("assessment")]
    finite_counts = {
        feature: int(
            pd.to_numeric(development[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .sum()
        )
        for feature in DAILY_OPTIONS_RAW_FEATURES
    }
    unsupported_features = sorted(feature for feature, count in finite_counts.items() if count == 0)
    return {
        **coverage,
        "required_stock_sessions": len(options_raw),
        "development_front_pair_stock_sessions": len(development),
        "assessment_front_pair_stock_sessions": len(assessment),
        "development_finite_raw_feature_counts": finite_counts,
        "unsupported_development_raw_features": unsupported_features,
        "dimension_fit_supported": not unsupported_features,
        "back_pair_stock_sessions": int(available["front_term_urgency"].notna().sum()),
        "skew_stock_sessions": int(available["skew_25d"].notna().sum()),
        "open_interest_surface_stock_sessions": int(
            (
                available["near_spot_oi_concentration"].notna()
                & available["call_put_oi_imbalance"].notna()
            ).sum()
        ),
        "passed": bool(coverage["passed"] and not unsupported_features),
    }


def write_chronology_audit(
    stock_raw: pd.DataFrame,
    options_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Write the exact D-1 stock/options chronology before any model fitting."""

    chronology = options_raw.merge(
        stock_raw.loc[:, ["symbol", "session", "stock_information_date"]],
        on=["symbol", "session"],
        how="left",
        validate="one_to_one",
    )
    chronology["actual_options_observation_date"] = chronology.get(
        "options_observation_date", pd.Series(index=chronology.index, dtype=object)
    )
    chronology["stock_d_minus_1_exact"] = chronology["stock_information_date"].astype(
        str
    ) == chronology["required_options_date"].astype(str)
    chronology["options_d_minus_1_exact"] = (
        chronology["actual_options_observation_date"].astype(str)
        == chronology["required_options_date"].astype(str)
    ) | ~chronology["pair_available"].astype(bool)
    chronology["same_day_options_used"] = chronology["actual_options_observation_date"].astype(
        str
    ) == chronology["session"].astype(str)
    chronology["chronology_passed"] = (
        chronology["stock_d_minus_1_exact"]
        & chronology["options_d_minus_1_exact"]
        & ~chronology["same_day_options_used"]
    )
    columns = [
        "symbol",
        "session",
        "stock_information_date",
        "required_options_date",
        "actual_options_observation_date",
        "pair_available",
        "stock_d_minus_1_exact",
        "options_d_minus_1_exact",
        "same_day_options_used",
        "chronology_passed",
    ]
    write_csv(PRIMARY / "chronology_audit.csv", chronology.loc[:, columns])
    if not bool(chronology["chronology_passed"].all()):
        raise ScreenBlocker("blocked_chronology_or_leakage_failure", "D-1 chronology audit failed")
    return chronology


def fit_all_models(
    panel: pd.DataFrame,
) -> tuple[
    dict[str, FrozenCrossMarketModel],
    dict[str, dict[str, float]],
    pd.DataFrame,
    pd.DataFrame,
]:
    development = panel.loc[panel["period"].eq("development")].copy()
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    configurations = {
        "S0": (
            S0_FEATURES,
            ("stock", "route_state"),
            "registered_completion_clean_bars_2_or_3",
            "logistic",
        ),
        "S1": (
            S1_FEATURES,
            ("stock", "route_state"),
            "registered_completion_clean_bars_2_or_3",
            "logistic",
        ),
        "S2": (
            S2_FEATURES,
            ("stock", "route_state"),
            "registered_completion_clean_bars_2_or_3",
            "logistic",
        ),
        "O0": (O0_FEATURES, ("stock",), "movement_exceeds_prior_close_iv_15m", "logistic"),
        "O1": (O1_FEATURES, ("stock",), "movement_exceeds_prior_close_iv_15m", "logistic"),
        "O2": (
            O2_FEATURES,
            ("stock", "route_state"),
            "movement_exceeds_prior_close_iv_15m",
            "logistic",
        ),
        "R0": (R0_FEATURES, ("stock",), "iv_absolute_residual_15m", "ridge"),
        "R1": (R1_FEATURES, ("stock", "route_state"), "iv_absolute_residual_15m", "ridge"),
    }
    models: dict[str, FrozenCrossMarketModel] = {}
    boundaries: dict[str, dict[str, float]] = {}
    for model_id, (features, controls, target, kind) in configurations.items():
        model = fit_cross_market_model(
            development,
            model_id=model_id,
            numeric_features=features,
            category_control_names=controls,
            target_column=target,
            kind=kind,
        )
        models[model_id] = model
        development[f"{model_id}_prediction"] = model.predict(development)
        assessment[f"{model_id}_prediction"] = model.predict(assessment)
        if kind == "logistic":
            boundaries[model_id] = probability_quantile_boundaries(
                development[f"{model_id}_prediction"]
            )
    return models, boundaries, development, assessment


def _metric_row(
    frame: pd.DataFrame,
    *,
    model_id: str,
    target: str,
    boundaries: Mapping[str, float],
    scope: str = "overall",
    group: str = "all",
) -> dict[str, Any]:
    metrics = binary_metrics(
        frame,
        target_column=target,
        probability_column=f"{model_id}_prediction",
        boundaries=boundaries,
    )
    return {"model": model_id, "scope": scope, "group": group, **metrics}


def model_metric_tables(
    assessment: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
    cutoffs: DevelopmentCutoffs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_a_target = "registered_completion_clean_bars_2_or_3"
    test_b_target = "movement_exceeds_prior_close_iv_15m"
    test_a = pd.DataFrame(
        [
            _metric_row(
                assessment,
                model_id=model,
                target=test_a_target,
                boundaries=boundaries[model],
            )
            for model in ("S0", "S1", "S2")
        ]
    )
    test_b = pd.DataFrame(
        [
            _metric_row(
                assessment,
                model_id=model,
                target=test_b_target,
                boundaries=boundaries[model],
            )
            for model in ("O0", "O1", "O2")
        ]
    )

    def monthly(models: Sequence[str], target: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        month = pd.to_datetime(assessment["session"]).dt.to_period("M").astype(str)
        for value, group in assessment.assign(_month=month).groupby(
            "_month", sort=True, observed=True
        ):
            for model in models:
                rows.append(
                    _metric_row(
                        group,
                        model_id=model,
                        target=target,
                        boundaries=boundaries[model],
                        scope="month",
                        group=str(value),
                    )
                )
        return pd.DataFrame(rows)

    development_medians = {
        "mismatch_compression_vs_iv": cutoffs.mismatch_compression_vs_iv,
        "mismatch_complacent_conflict": cutoffs.mismatch_complacent_conflict,
        "options_implied_tension": cutoffs.options_implied_tension,
        "options_front_urgency": cutoffs.options_front_urgency,
        "mismatch_route_vs_premium": cutoffs.mismatch_route_vs_premium,
    }

    def subgroup(
        models: Sequence[str], target: str, definitions: Sequence[tuple[str, pd.Series]]
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for label, mask in definitions:
            group = assessment.loc[mask]
            if group.empty:
                continue
            for model in models:
                rows.append(
                    _metric_row(
                        group,
                        model_id=model,
                        target=target,
                        boundaries=boundaries[model],
                        scope="subgroup",
                        group=label,
                    )
                )
        return pd.DataFrame(rows)

    common: list[tuple[str, pd.Series]] = [
        *[
            (f"checkpoint_group={value}", assessment["checkpoint_group"].eq(value))
            for value in ("early_6_14", "middle_16_24", "late_26_34")
        ],
        ("route_state=BROAD_CONFLICT", assessment["BROAD_CONFLICT"].eq(1)),
        ("route_state=LOW_ROUTE_SUPPORT", assessment["LOW_ROUTE_SUPPORT"].eq(1)),
        *[
            (
                f"daily_stock_regime={value}",
                assessment["daily_stock_regime"].astype(int).eq(value),
            )
            for value in range(4)
        ],
        *[
            (
                f"daily_options_regime={value}",
                assessment["daily_options_regime"].astype(int).eq(value),
            )
            for value in range(4)
        ],
    ]
    test_a_definitions = [
        *common,
        *[
            (
                f"mismatch_compression_vs_iv={label}",
                assessment["mismatch_compression_vs_iv"].ge(
                    development_medians["mismatch_compression_vs_iv"]
                )
                if label == "high"
                else assessment["mismatch_compression_vs_iv"].lt(
                    development_medians["mismatch_compression_vs_iv"]
                ),
            )
            for label in ("high", "low")
        ],
        *[
            (
                f"mismatch_complacent_conflict={label}",
                assessment["mismatch_complacent_conflict"].ge(
                    development_medians["mismatch_complacent_conflict"]
                )
                if label == "high"
                else assessment["mismatch_complacent_conflict"].lt(
                    development_medians["mismatch_complacent_conflict"]
                ),
            )
            for label in ("high", "low")
        ],
    ]
    test_b_definitions = [
        *common,
        *[
            (
                f"options_implied_tension={label}",
                assessment["options_implied_tension"].ge(
                    development_medians["options_implied_tension"]
                )
                if label == "high"
                else assessment["options_implied_tension"].lt(
                    development_medians["options_implied_tension"]
                ),
            )
            for label in ("high", "low")
        ],
        *[
            (
                f"options_front_urgency={label}",
                assessment["options_front_urgency"].ge(development_medians["options_front_urgency"])
                if label == "high"
                else assessment["options_front_urgency"].lt(
                    development_medians["options_front_urgency"]
                ),
            )
            for label in ("high", "low")
        ],
        *[
            (
                f"mismatch_compression_vs_iv={label}",
                assessment["mismatch_compression_vs_iv"].ge(
                    development_medians["mismatch_compression_vs_iv"]
                )
                if label == "high"
                else assessment["mismatch_compression_vs_iv"].lt(
                    development_medians["mismatch_compression_vs_iv"]
                ),
            )
            for label in ("high", "low")
        ],
        *[
            (
                f"mismatch_route_vs_premium={label}",
                assessment["mismatch_route_vs_premium"].ge(
                    development_medians["mismatch_route_vs_premium"]
                )
                if label == "high"
                else assessment["mismatch_route_vs_premium"].lt(
                    development_medians["mismatch_route_vs_premium"]
                ),
            )
            for label in ("high", "low")
        ],
    ]
    return (
        test_a,
        monthly(("S0", "S1", "S2"), test_a_target),
        subgroup(("S0", "S1", "S2"), test_a_target, test_a_definitions),
        test_b,
        monthly(("O0", "O1", "O2"), test_b_target),
        subgroup(("O0", "O1", "O2"), test_b_target, test_b_definitions),
    )


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    valid = frame[column].notna() & frame["row_weight"].notna()
    if not bool(valid.any()):
        return math.nan
    values = pd.to_numeric(frame.loc[valid, column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame.loc[valid, "row_weight"], errors="raise").to_numpy(float)
    return float(np.sum(values * weights) / np.sum(weights))


def persistence_summary(frame: pd.DataFrame, *, group: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": group,
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(pd.to_datetime(frame["session"]).dt.to_period("M").nunique()),
    }
    for horizon in ("15m", "to_close", "next_close", "third_close"):
        residual = f"iv_absolute_residual_{horizon}"
        valid = frame[residual].notna()
        row[f"valid_rows_{horizon}"] = int(valid.sum())
        row[f"mean_iv_residual_{horizon}"] = _weighted_mean(frame, residual)
        row[f"mean_absolute_log_return_{horizon}"] = _weighted_mean(
            frame, f"absolute_log_return_{horizon}"
        )
        row[f"mean_iv_expected_absolute_{horizon}"] = _weighted_mean(
            frame, f"iv_expected_absolute_{horizon}"
        )
        row[f"positive_residual_rate_{horizon}"] = _weighted_mean(
            frame.assign(_positive=frame[residual].gt(0.0).where(valid).astype(float)),
            "_positive",
        )
    return row


def persistence_tables(
    development: pd.DataFrame, assessment: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compression_median = float(development["mismatch_compression_vs_iv"].median())
    complacent_median = float(development["mismatch_complacent_conflict"].median())
    rows = [
        persistence_summary(assessment, group="all_joined_rows"),
        persistence_summary(
            assessment.loc[assessment["BROAD_CONFLICT"].eq(1)], group="BROAD_CONFLICT"
        ),
        persistence_summary(
            assessment.loc[assessment["LOW_ROUTE_SUPPORT"].eq(1)],
            group="LOW_ROUTE_SUPPORT",
        ),
        persistence_summary(
            assessment.loc[assessment["mismatch_compression_vs_iv"].ge(compression_median)],
            group="high_mismatch_compression_vs_iv",
        ),
        persistence_summary(
            assessment.loc[assessment["mismatch_complacent_conflict"].ge(complacent_median)],
            group="high_mismatch_complacent_conflict",
        ),
    ]
    for regime in range(4):
        rows.append(
            persistence_summary(
                assessment.loc[assessment["daily_stock_regime"].astype(int).eq(regime)],
                group=f"daily_stock_regime={regime}",
            )
        )
        rows.append(
            persistence_summary(
                assessment.loc[assessment["daily_options_regime"].astype(int).eq(regime)],
                group=f"daily_options_regime={regime}",
            )
        )
    pair_rows: list[dict[str, Any]] = []
    for (stock_regime, options_regime), group in assessment.groupby(
        ["daily_stock_regime", "daily_options_regime"], sort=True, observed=True
    ):
        support = bool(
            len(group) >= 200
            and group["session"].nunique() >= 50
            and group["symbol"].nunique() >= 8
            and pd.to_datetime(group["session"]).dt.to_period("M").nunique() >= 4
        )
        summary = persistence_summary(
            group,
            group=f"daily_stock_regime={int(stock_regime)}|daily_options_regime={int(options_regime)}",
        )
        summary["daily_stock_regime"] = int(stock_regime)
        summary["daily_options_regime"] = int(options_regime)
        summary["reportable_support"] = support
        pair_rows.append(summary)
    persistence = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)

    def mapping(row: Mapping[str, Any]) -> str:
        residual_15m = float(row["mean_iv_residual_15m"])
        residual_next = float(row["mean_iv_residual_next_close"])
        residual_third = float(row["mean_iv_residual_third_close"])
        if not math.isfinite(residual_15m) or residual_15m <= 0.0:
            return "no_iv_excess_persistence"
        if math.isfinite(residual_third) and residual_next > 0.0 and residual_third > 0.0:
            return "multi_session_persistent"
        if math.isfinite(residual_next) and residual_next > 0.0:
            return "overnight_persistent"
        return "intraday_only"

    mapping_groups = persistence.loc[
        persistence["group"].isin(
            [
                "BROAD_CONFLICT",
                "LOW_ROUTE_SUPPORT",
                "high_mismatch_compression_vs_iv",
            ]
        )
    ].copy()
    supported_pairs = pairs.loc[pairs["reportable_support"].astype(bool)].copy()
    mapping_source = pd.concat([mapping_groups, supported_pairs], ignore_index=True)
    mapping_frame = pd.DataFrame(
        [
            {
                "group": row["group"],
                "persistence_horizon_classification": mapping(row),
                "research_classification_only": True,
                "dte_recommendation": False,
                "trading_instruction": False,
            }
            for row in mapping_source.to_dict(orient="records")
        ]
    )
    return persistence, pairs, mapping_frame


def _improvement(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, float]:
    return {
        "log_loss_improvement": float(old["log_loss"]) - float(new["log_loss"]),
        "brier_improvement": float(old["brier_score"]) - float(new["brier_score"]),
        "auc_improvement": float(new["auc"]) - float(old["auc"]),
        "average_precision_improvement": float(new["average_precision"])
        - float(old["average_precision"]),
    }


def bootstrap_intervals(
    assessment: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
    cutoffs: DevelopmentCutoffs,
) -> pd.DataFrame:
    draws = fixed_session_bootstrap_multiplicities(assessment["session"], draws=10, seed=20260723)
    statistics: dict[str, list[float]] = {
        name: []
        for name in (
            "S1_minus_S0_log_loss_improvement",
            "S1_minus_S0_brier_improvement",
            "S2_minus_S1_log_loss_improvement",
            "S2_minus_S1_brier_improvement",
            "S2_minus_S1_auc_improvement",
            "S2_minus_S1_average_precision_improvement",
            "O1_minus_O0_log_loss_improvement",
            "O1_minus_O0_brier_improvement",
            "O2_minus_O1_log_loss_improvement",
            "O2_minus_O1_brier_improvement",
            "O2_minus_O1_auc_improvement",
            "O2_minus_O1_average_precision_improvement",
            "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_15m_iv_residual",
            "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_next_close_iv_residual",
            "high_minus_low_mismatch_compression_vs_iv_15m_iv_residual",
            "high_minus_low_mismatch_compression_vs_iv_next_close_iv_residual",
        )
    }
    compression_median = cutoffs.mismatch_compression_vs_iv
    for multiplicity in draws:
        sampled = assessment.copy()
        sampled["row_weight"] = sampled["row_weight"].to_numpy(float) * multiplicity
        sampled = sampled.loc[sampled["row_weight"].gt(0.0)]
        metrics: dict[str, dict[str, Any]] = {}
        for model, target in (
            ("S0", "registered_completion_clean_bars_2_or_3"),
            ("S1", "registered_completion_clean_bars_2_or_3"),
            ("S2", "registered_completion_clean_bars_2_or_3"),
            ("O0", "movement_exceeds_prior_close_iv_15m"),
            ("O1", "movement_exceeds_prior_close_iv_15m"),
            ("O2", "movement_exceeds_prior_close_iv_15m"),
        ):
            metrics[model] = binary_metrics(
                sampled,
                target_column=target,
                probability_column=f"{model}_prediction",
                boundaries=boundaries[model],
            )
        s10 = _improvement(metrics["S0"], metrics["S1"])
        s21 = _improvement(metrics["S1"], metrics["S2"])
        o10 = _improvement(metrics["O0"], metrics["O1"])
        o21 = _improvement(metrics["O1"], metrics["O2"])
        for prefix, values in (
            ("S1_minus_S0", s10),
            ("S2_minus_S1", s21),
            ("O1_minus_O0", o10),
            ("O2_minus_O1", o21),
        ):
            for metric, value in values.items():
                key = f"{prefix}_{metric}"
                if key in statistics:
                    statistics[key].append(value)
        broad = sampled.loc[sampled["BROAD_CONFLICT"].eq(1)]
        low = sampled.loc[sampled["LOW_ROUTE_SUPPORT"].eq(1)]
        high = sampled.loc[sampled["mismatch_compression_vs_iv"].ge(compression_median)]
        low_mismatch = sampled.loc[sampled["mismatch_compression_vs_iv"].lt(compression_median)]
        for horizon in ("15m", "next_close"):
            statistics[f"BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_{horizon}_iv_residual"].append(
                _weighted_mean(broad, f"iv_absolute_residual_{horizon}")
                - _weighted_mean(low, f"iv_absolute_residual_{horizon}")
            )
            statistics[f"high_minus_low_mismatch_compression_vs_iv_{horizon}_iv_residual"].append(
                _weighted_mean(high, f"iv_absolute_residual_{horizon}")
                - _weighted_mean(low_mismatch, f"iv_absolute_residual_{horizon}")
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
                    "coarse_quick_screen_diagnostic": True,
                }
            )
    return pd.DataFrame(rows)


def null_refits(
    panel: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
    standardization: Mapping[str, MeanStandardization],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    s1_metrics = binary_metrics(
        assessment,
        target_column="registered_completion_clean_bars_2_or_3",
        probability_column="S1_prediction",
        boundaries=boundaries["S1"],
    )
    o1_metrics = binary_metrics(
        assessment,
        target_column="movement_exceeds_prior_close_iv_15m",
        probability_column="O1_prediction",
        boundaries=boundaries["O1"],
    )
    real_s2 = binary_metrics(
        assessment,
        target_column="registered_completion_clean_bars_2_or_3",
        probability_column="S2_prediction",
        boundaries=boundaries["S2"],
    )
    real_o2 = binary_metrics(
        assessment,
        target_column="movement_exceeds_prior_close_iv_15m",
        probability_column="O2_prediction",
        boundaries=boundaries["O2"],
    )
    real_s_increment = _improvement(s1_metrics, real_s2)
    real_o_increment = _improvement(o1_metrics, real_o2)
    options_columns = [
        column
        for column in (
            *DAILY_OPTIONS_RAW_FEATURES,
            *OPTIONS_CONTEXT_FEATURES,
            "daily_options_regime",
            "daily_options_regime_top_probability",
            "daily_options_regime_mahalanobis_to_nearest_centroid",
            "options_observation_date",
            "front_expiration_date",
            "front_strike",
            "front_call_contract_id",
            "front_put_contract_id",
            "back_expiration_date",
            "back_strike",
            "back_call_contract_id",
            "back_put_contract_id",
            "skew_put_contract_id",
            "skew_call_contract_id",
        )
        if column in panel.columns
    ]
    options_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(OPTIONS_NULL_SEEDS):
        permuted = permute_bundle_within_slates(panel, columns=options_columns, seed=seed)
        permuted = add_mismatch_features(permuted, standardization)
        model = fit_cross_market_model(
            permuted.loc[permuted["period"].eq("development")],
            model_id=f"S2_options_null_{index}",
            numeric_features=S2_FEATURES,
            category_control_names=("stock", "route_state"),
            target_column="registered_completion_clean_bars_2_or_3",
            kind="logistic",
        )
        permuted_assessment = permuted.loc[permuted["period"].eq("assessment")].copy()
        permuted_assessment["_null_prediction"] = model.predict(permuted_assessment)
        null_metrics = binary_metrics(
            permuted_assessment,
            target_column="registered_completion_clean_bars_2_or_3",
            probability_column="_null_prediction",
            boundaries=boundaries["S2"],
        )
        increment = _improvement(s1_metrics, null_metrics)
        options_rows.append(
            {
                "null_refit": index,
                "seed": seed,
                **increment,
                **{
                    f"real_exceeds_null_{metric}": real_s_increment[metric] > value
                    for metric, value in increment.items()
                },
            }
        )
    route_columns = (*ROUTE_FEATURES, "route_resolution_state")
    route_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(ROUTE_NULL_SEEDS):
        permuted = permute_bundle_within_slates(panel, columns=route_columns, seed=seed)
        permuted["BROAD_CONFLICT"] = (
            permuted["route_resolution_state"].astype(str).eq("BROAD_CONFLICT").astype(int)
        )
        permuted["LOW_ROUTE_SUPPORT"] = (
            permuted["route_resolution_state"].astype(str).eq("LOW_ROUTE_SUPPORT").astype(int)
        )
        permuted = add_mismatch_features(permuted, standardization)
        model = fit_cross_market_model(
            permuted.loc[permuted["period"].eq("development")],
            model_id=f"O2_route_null_{index}",
            numeric_features=O2_FEATURES,
            category_control_names=("stock", "route_state"),
            target_column="movement_exceeds_prior_close_iv_15m",
            kind="logistic",
        )
        permuted_assessment = permuted.loc[permuted["period"].eq("assessment")].copy()
        permuted_assessment["_null_prediction"] = model.predict(permuted_assessment)
        null_metrics = binary_metrics(
            permuted_assessment,
            target_column="movement_exceeds_prior_close_iv_15m",
            probability_column="_null_prediction",
            boundaries=boundaries["O2"],
        )
        increment = _improvement(o1_metrics, null_metrics)
        route_rows.append(
            {
                "null_refit": index,
                "seed": seed,
                **increment,
                **{
                    f"real_exceeds_null_{metric}": real_o_increment[metric] > value
                    for metric, value in increment.items()
                },
            }
        )
    return pd.DataFrame(options_rows), pd.DataFrame(route_rows)


def _metric_map(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["model"]): cast(dict[str, Any], row) for row in frame.to_dict(orient="records")}


def _bootstrap_lower(bootstrap: pd.DataFrame, statistic: str, confidence: float = 0.80) -> float:
    row = bootstrap.loc[
        bootstrap["statistic"].eq(statistic)
        & np.isclose(bootstrap["confidence"].to_numpy(float), confidence)
    ]
    if len(row) != 1:
        raise ValueError(f"bootstrap interval unavailable: {statistic}/{confidence}")
    return float(row.iloc[0]["lower"])


def _positive_months(monthly: pd.DataFrame, *, old_model: str, new_model: str) -> int:
    old = monthly.loc[monthly["model"].eq(old_model)].set_index("group")
    new = monthly.loc[monthly["model"].eq(new_model)].set_index("group")
    shared = old.index.intersection(new.index)
    return int(
        sum(
            float(old.loc[group, "log_loss"]) - float(new.loc[group, "log_loss"]) > 0.0
            for group in shared
        )
    )


def _adverse_checkpoint_groups(subgroup: pd.DataFrame, *, old_model: str, new_model: str) -> int:
    groups = [
        "checkpoint_group=early_6_14",
        "checkpoint_group=middle_16_24",
        "checkpoint_group=late_26_34",
    ]
    adverse = 0
    for group in groups:
        old = subgroup.loc[subgroup["model"].eq(old_model) & subgroup["group"].eq(group)].iloc[0]
        new = subgroup.loc[subgroup["model"].eq(new_model) & subgroup["group"].eq(group)].iloc[0]
        if (
            float(old["log_loss"]) - float(new["log_loss"]) < -1e-12
            or float(old["brier_score"]) - float(new["brier_score"]) < -1e-12
        ):
            adverse += 1
    return adverse


def component_decision(
    *,
    panel: pd.DataFrame,
    test_a: pd.DataFrame,
    test_a_monthly: pd.DataFrame,
    test_a_subgroup: pd.DataFrame,
    test_b: pd.DataFrame,
    test_b_monthly: pd.DataFrame,
    bootstrap: pd.DataFrame,
    options_null: pd.DataFrame,
    route_null: pd.DataFrame,
    stock_regime_supported: bool,
    options_regime_supported: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    weight_by_stock = assessment.groupby("symbol", observed=True)["row_weight"].sum()
    max_stock_share = float(weight_by_stock.max() / weight_by_stock.sum())
    support_a = bool(
        assessment["registered_completion_clean_bars_2_or_3"].sum() >= 100
        and max_stock_share <= 0.15
    )
    support_b = bool(
        assessment["movement_exceeds_prior_close_iv_15m"].sum() >= 300 and max_stock_share <= 0.15
    )
    a = _metric_map(test_a)
    b = _metric_map(test_b)
    s10 = _improvement(a["S0"], a["S1"])
    s21 = _improvement(a["S1"], a["S2"])
    o10 = _improvement(b["O0"], b["O1"])
    o21 = _improvement(b["O1"], b["O2"])
    test_a_stock_pass = bool(
        support_a
        and s10["log_loss_improvement"] > 0.0
        and s10["brier_improvement"] > 0.0
        and s10["auc_improvement"] >= 0.0
        and _bootstrap_lower(bootstrap, "S1_minus_S0_log_loss_improvement") >= 0.0
        and _bootstrap_lower(bootstrap, "S1_minus_S0_brier_improvement") >= 0.0
        and _positive_months(test_a_monthly, old_model="S0", new_model="S1") >= 4
        and _adverse_checkpoint_groups(test_a_subgroup, old_model="S0", new_model="S1") == 0
    )
    options_real_exceeds = bool(
        options_null["real_exceeds_null_log_loss_improvement"].all()
        and options_null["real_exceeds_null_brier_improvement"].all()
    )
    test_a_options_pass = bool(
        support_a
        and s21["log_loss_improvement"] > 0.0
        and s21["brier_improvement"] > 0.0
        and s21["auc_improvement"] >= 0.0
        and s21["average_precision_improvement"] > 0.0
        and _bootstrap_lower(bootstrap, "S2_minus_S1_log_loss_improvement") >= 0.0
        and _bootstrap_lower(bootstrap, "S2_minus_S1_brier_improvement") >= 0.0
        and options_real_exceeds
        and _positive_months(test_a_monthly, old_model="S1", new_model="S2") >= 4
    )
    test_b_stock_pass = bool(
        support_b
        and o10["log_loss_improvement"] > 0.0
        and o10["brier_improvement"] > 0.0
        and o10["auc_improvement"] >= 0.0
        and _bootstrap_lower(bootstrap, "O1_minus_O0_log_loss_improvement") >= 0.0
        and _bootstrap_lower(bootstrap, "O1_minus_O0_brier_improvement") >= 0.0
        and _positive_months(test_b_monthly, old_model="O0", new_model="O1") >= 4
    )
    route_real_exceeds = bool(
        route_null["real_exceeds_null_log_loss_improvement"].all()
        and route_null["real_exceeds_null_brier_improvement"].all()
    )
    test_b_route_pass = bool(
        support_b
        and o21["log_loss_improvement"] > 0.0
        and o21["brier_improvement"] > 0.0
        and o21["auc_improvement"] >= 0.0
        and o21["average_precision_improvement"] > 0.0
        and _bootstrap_lower(bootstrap, "O2_minus_O1_log_loss_improvement") >= 0.0
        and _bootstrap_lower(bootstrap, "O2_minus_O1_brier_improvement") >= 0.0
        and route_real_exceeds
        and _positive_months(test_b_monthly, old_model="O1", new_model="O2") >= 4
    )

    def increment_status(supported: bool, support: bool) -> str:
        if not support:
            return "insufficient_support"
        return "supported" if supported else "not_supported"

    statuses = {
        "daily_stock_regime_status": (
            "supported" if stock_regime_supported else "descriptive_only"
        ),
        "daily_options_regime_status": (
            "supported" if options_regime_supported else "descriptive_only"
        ),
        "test_a_daily_stock_increment_status": increment_status(test_a_stock_pass, support_a),
        "test_a_daily_options_increment_status": increment_status(test_a_options_pass, support_a),
        "test_b_daily_stock_increment_status": increment_status(test_b_stock_pass, support_b),
        "test_b_intraday_route_increment_status": increment_status(test_b_route_pass, support_b),
        "mismatch_status": (
            "descriptive_only" if test_a_options_pass or test_b_route_pass else "not_supported"
        ),
        "persistence_horizon_status": "descriptive_only",
    }
    gates = {
        "test_a_support": support_a,
        "test_b_support": support_b,
        "maximum_stock_weight_share": max_stock_share,
        "S1_minus_S0": s10,
        "S2_minus_S1": s21,
        "O1_minus_O0": o10,
        "O2_minus_O1": o21,
        "test_a_daily_stock_passed": test_a_stock_pass,
        "test_a_daily_options_passed": test_a_options_pass,
        "test_b_daily_stock_passed": test_b_stock_pass,
        "test_b_intraday_route_passed": test_b_route_pass,
        "options_real_proper_scores_exceed_all_nulls": options_real_exceeds,
        "route_real_proper_scores_exceed_all_nulls": route_real_exceeds,
        "test_a_daily_stock_positive_months": _positive_months(
            test_a_monthly, old_model="S0", new_model="S1"
        ),
        "test_a_daily_options_positive_months": _positive_months(
            test_a_monthly, old_model="S1", new_model="S2"
        ),
        "test_b_daily_stock_positive_months": _positive_months(
            test_b_monthly, old_model="O0", new_model="O1"
        ),
        "test_b_intraday_route_positive_months": _positive_months(
            test_b_monthly, old_model="O1", new_model="O2"
        ),
    }
    return statuses, gates


def concentration_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    assessment = panel.loc[panel["period"].eq("assessment")]
    total = float(assessment["row_weight"].sum())
    rows: list[dict[str, Any]] = []
    for kind, group_column in (("stock", "symbol"), ("month", "session")):
        values = (
            pd.to_datetime(assessment[group_column]).dt.to_period("M").astype(str)
            if kind == "month"
            else assessment[group_column].astype(str)
        )
        grouped = (
            assessment.assign(_group=values)
            .groupby("_group", sort=True, observed=True)["row_weight"]
            .sum()
        )
        for group, weight in grouped.items():
            rows.append(
                {
                    "concentration_type": kind,
                    "group": group,
                    "weighted_rows": float(weight),
                    "share": float(weight / total),
                }
            )
    return pd.DataFrame(rows)


def determinism_rebuild(
    *,
    structural: pd.DataFrame,
    full_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    cache: pd.DataFrame,
    first_panel: pd.DataFrame,
    first_models: Mapping[str, FrozenCrossMarketModel],
    first_stock_regime: FrozenSoftRegime,
    first_options_regime: FrozenSoftRegime,
    first_persistence: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    first_decision: Mapping[str, Any],
    frozen_bootstrap: pd.DataFrame,
    frozen_options_null: pd.DataFrame,
    frozen_route_null: pd.DataFrame,
) -> dict[str, Any]:
    daily_raw = calculate_daily_stock_raw_features(daily_bars)
    stock_raw, stock_dimensions, _stock_parameters, stock_regime, _support = build_stock_context(
        structural, daily_bars
    )
    options_raw, _gap, _records = build_options_context(stock_raw, cache)
    options_dimensions, _options_parameters, options_regime = fit_options_context(options_raw)
    second_panel, _standardization = join_cross_market_panel(
        structural, stock_dimensions, options_dimensions, full_bars, daily_raw
    )
    second_models, second_boundaries, second_development, second_assessment = fit_all_models(
        second_panel
    )
    first_assessment = first_panel.loc[first_panel["period"].eq("assessment")].copy()
    for model_id in first_models:
        first_assessment[f"{model_id}_prediction"] = first_models[model_id].predict(
            first_assessment
        )
    row_mismatches = abs(len(first_panel) - len(second_panel)) + sum(
        left != right
        for left, right in zip(
            first_panel["row_id"].astype(str),
            second_panel["row_id"].astype(str),
            strict=False,
        )
    )
    numeric_features = (
        *DAILY_STOCK_RAW_FEATURES,
        *STOCK_CONTEXT_FEATURES,
        *DAILY_OPTIONS_RAW_FEATURES,
        *OPTIONS_CONTEXT_FEATURES,
        *MISMATCH_FEATURES,
    )
    maximum_feature_difference = (
        math.inf
        if row_mismatches
        else float(
            np.nanmax(
                np.abs(
                    first_panel.loc[:, list(numeric_features)].to_numpy(float)
                    - second_panel.loc[:, list(numeric_features)].to_numpy(float)
                )
            )
        )
    )
    maximum_probability_difference = float(
        max(
            np.max(
                np.abs(
                    first_assessment[f"{model_id}_prediction"].to_numpy(float)
                    - second_assessment[f"{model_id}_prediction"].to_numpy(float)
                )
            )
            for model_id in first_models
        )
    )
    stock_mapping_mismatches = int(
        first_stock_regime.canonical_to_original != stock_regime.canonical_to_original
        or first_stock_regime.original_to_canonical != stock_regime.original_to_canonical
    )
    options_mapping_mismatches = int(
        first_options_regime.canonical_to_original != options_regime.canonical_to_original
        or first_options_regime.original_to_canonical != options_regime.original_to_canonical
    )
    second_persistence = persistence_tables(second_development, second_assessment)

    def frame_difference(left: pd.DataFrame, right: pd.DataFrame) -> tuple[int, float]:
        if set(left.columns) != set(right.columns) or len(left) != len(right):
            return 1, math.inf
        columns = sorted(left.columns)
        left_sorted = (
            left.loc[:, columns].sort_values(columns, kind="mergesort").reset_index(drop=True)
        )
        right_sorted = (
            right.loc[:, columns].sort_values(columns, kind="mergesort").reset_index(drop=True)
        )
        numeric = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(left_sorted[column])
            and pd.api.types.is_numeric_dtype(right_sorted[column])
        ]
        nonnumeric = [column for column in columns if column not in numeric]
        mismatches = int(
            any(
                not left_sorted[column]
                .fillna("<NA>")
                .astype(str)
                .equals(right_sorted[column].fillna("<NA>").astype(str))
                for column in nonnumeric
            )
        )
        maximum = 0.0
        for column in numeric:
            left_values = pd.to_numeric(left_sorted[column], errors="coerce").to_numpy(float)
            right_values = pd.to_numeric(right_sorted[column], errors="coerce").to_numpy(float)
            both_nan = np.isnan(left_values) & np.isnan(right_values)
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            mismatches += int(bool((~both_nan & ~finite).any()))
            if bool(finite.any()):
                maximum = max(
                    maximum,
                    float(np.max(np.abs(left_values[finite] - right_values[finite]))),
                )
        return mismatches, maximum

    persistence_mismatches = 0
    maximum_persistence_difference = 0.0
    for first_frame, second_frame in zip(first_persistence, second_persistence, strict=True):
        mismatches, difference = frame_difference(first_frame, second_frame)
        persistence_mismatches += mismatches
        maximum_persistence_difference = max(maximum_persistence_difference, difference)

    second_cutoffs = DevelopmentCutoffs(
        mismatch_compression_vs_iv=float(second_development["mismatch_compression_vs_iv"].median()),
        mismatch_complacent_conflict=float(
            second_development["mismatch_complacent_conflict"].median()
        ),
        options_implied_tension=float(second_development["options_implied_tension"].median()),
        options_front_urgency=float(second_development["options_front_urgency"].median()),
        mismatch_route_vs_premium=float(second_development["mismatch_route_vs_premium"].median()),
    )
    (
        second_test_a,
        second_test_a_monthly,
        second_test_a_subgroup,
        second_test_b,
        second_test_b_monthly,
        _second_test_b_subgroup,
    ) = model_metric_tables(second_assessment, second_boundaries, second_cutoffs)
    second_stock_supported, _ = regime_support(stock_dimensions, "daily_stock_regime")
    second_options_supported, _ = regime_support(options_dimensions, "daily_options_regime")
    second_statuses, _second_gates = component_decision(
        panel=second_panel,
        test_a=second_test_a,
        test_a_monthly=second_test_a_monthly,
        test_a_subgroup=second_test_a_subgroup,
        test_b=second_test_b,
        test_b_monthly=second_test_b_monthly,
        bootstrap=frozen_bootstrap,
        options_null=frozen_options_null,
        route_null=frozen_route_null,
        stock_regime_supported=second_stock_supported,
        options_regime_supported=second_options_supported,
    )
    second_overall = choose_daily_context_decision(
        blocker=None,
        test_a_daily_stock_supported=second_statuses["test_a_daily_stock_increment_status"]
        == "supported",
        test_a_daily_options_supported=second_statuses["test_a_daily_options_increment_status"]
        == "supported",
        test_b_daily_stock_supported=second_statuses["test_b_daily_stock_increment_status"]
        == "supported",
        test_b_intraday_route_supported=second_statuses["test_b_intraday_route_increment_status"]
        == "supported",
        mismatch_supported=second_statuses["mismatch_status"] == "supported",
        descriptive=any(value == "descriptive_only" for value in second_statuses.values()),
    )
    decision_mismatches = int(
        second_overall != first_decision.get("overall_decision")
        or any(first_decision.get(key) != value for key, value in second_statuses.items())
    )
    result = {
        **SAFETY_FLAGS,
        "stock_regime_mapping_mismatches": stock_mapping_mismatches,
        "options_regime_mapping_mismatches": options_mapping_mismatches,
        "joined_row_mismatches": int(row_mismatches),
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "persistence_result_mismatches": persistence_mismatches,
        "maximum_persistence_difference": maximum_persistence_difference,
        "decision_mismatches": decision_mismatches,
        "bootstrap_repeated": False,
        "null_draws_repeated": False,
    }
    result["passed"] = bool(
        stock_mapping_mismatches == 0
        and options_mapping_mismatches == 0
        and row_mismatches == 0
        and maximum_feature_difference <= 1e-12
        and maximum_probability_difference <= 1e-12
        and persistence_mismatches == 0
        and maximum_persistence_difference <= 1e-12
        and decision_mismatches == 0
    )
    return result


def create_plots(
    *,
    stock_regime: FrozenSoftRegime,
    options_regime: FrozenSoftRegime,
    test_a: pd.DataFrame,
    test_b: pd.DataFrame,
    persistence: pd.DataFrame,
) -> None:
    stock_values = np.asarray(
        [
            [centroid[name] for name in stock_regime.canonical_dimensions]
            for centroid in stock_regime.canonical_centroids
        ],
        dtype=float,
    )
    option_values = np.asarray(
        [
            [centroid[name] for name in options_regime.canonical_dimensions]
            for centroid in options_regime.canonical_centroids
        ],
        dtype=float,
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, values, names, title in (
        (
            axes[0],
            stock_values,
            stock_regime.canonical_dimensions,
            "Daily stock regime canonical centroids",
        ),
        (
            axes[1],
            option_values,
            options_regime.canonical_dimensions,
            "Daily options regime canonical centroids",
        ),
    ):
        image = axis.imshow(values, aspect="auto", cmap="coolwarm")
        axis.set_yticks(range(4), labels=[f"R{value}" for value in range(4)])
        axis.set_xticks(range(len(names)), labels=names, rotation=25, ha="right")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.02)
    figure.savefig(PRIMARY / "daily_regime_centroid_summary.png", dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    model_rows = pd.concat(
        [
            test_a.loc[:, ["model", "log_loss", "brier_score"]],
            test_b.loc[:, ["model", "log_loss", "brier_score"]],
        ],
        ignore_index=True,
    )
    x = np.arange(len(model_rows))
    axes[0].bar(x - 0.18, model_rows["log_loss"], 0.36, label="log loss")
    axes[0].bar(x + 0.18, model_rows["brier_score"], 0.36, label="Brier")
    axes[0].set_xticks(x, labels=model_rows["model"])
    axes[0].set_title("Assessment proper scores")
    axes[0].legend()
    residual = persistence.loc[
        persistence["group"].isin(
            [
                "high_mismatch_compression_vs_iv",
                "BROAD_CONFLICT",
                "LOW_ROUTE_SUPPORT",
            ]
        )
    ]
    axes[1].bar(
        np.arange(len(residual)),
        residual["mean_iv_residual_15m"],
        color=["#3b82f6", "#f97316", "#64748b"][: len(residual)],
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(residual)), labels=residual["group"], rotation=20, ha="right")
    axes[1].set_title("15-minute IV residual by frozen state")
    figure.savefig(PRIMARY / "proper_scores_and_iv_residual.png", dpi=130)
    plt.close(figure)


def render_report(
    *,
    decision: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    stock_support: Mapping[str, Any],
    options_coverage_value: Mapping[str, Any],
    reused_records: int,
    newly_downloaded_records: int,
    newly_downloaded_bytes: int,
    panel: pd.DataFrame,
    test_a: pd.DataFrame,
    test_b: pd.DataFrame,
    continuous: pd.DataFrame,
    persistence: pd.DataFrame,
    bootstrap: pd.DataFrame,
    options_null: pd.DataFrame,
    route_null: pd.DataFrame,
    determinism: Mapping[str, Any],
) -> str:
    a = _metric_map(test_a)
    b = _metric_map(test_b)
    s10 = _improvement(a["S0"], a["S1"])
    s21 = _improvement(a["S1"], a["S2"])
    o10 = _improvement(b["O0"], b["O1"])
    o21 = _improvement(b["O1"], b["O2"])
    assessment = panel.loc[panel["period"].eq("assessment")]
    broad = persistence.loc[persistence["group"].eq("BROAD_CONFLICT")].iloc[0]
    low = persistence.loc[persistence["group"].eq("LOW_ROUTE_SUPPORT")].iloc[0]

    def metric_line(model: str, values: Mapping[str, Any]) -> str:
        return (
            f"- {model}: log loss {float(values['log_loss']):.8f}; "
            f"Brier {float(values['brier_score']):.8f}; AUC {float(values['auc']):.8f}; "
            f"average precision {float(values['average_precision']):.8f}."
        )

    return f"""# Daily Stock × Options Regime Context Quick Screen V0

Overall decision: `{decision["overall_decision"]}`.

This was a retrospective research screen using previous-close options only. It calculated no
option P&L, did not search strategies or DTE rules, and did not access execution or brokers.

## Support and chronology

- Structural rows: {int(reconstruction["clean_advance_rows"]):,}; assessment rows:
  {int(reconstruction["assessment_clean_rows"]):,}; reconstruction passed:
  `{bool(reconstruction["passed"])}`.
- Daily stock assessment retention: {float(stock_support["daily_stock_feature_retention"]):.4%};
  stocks {int(stock_support["assessment_stocks"])}; sessions
  {int(stock_support["assessment_sessions"])}; months {int(stock_support["assessment_months"])}.
- Reused exact-date option records: {reused_records:,}; newly downloaded records:
  {newly_downloaded_records:,}; newly downloaded raw bytes: {newly_downloaded_bytes:,}.
- Valid-pair assessment clean rows:
  {int(options_coverage_value["assessment_clean_checkpoint_rows"]):,}; sessions
  {int(options_coverage_value["assessment_sessions"])}; stocks
  {int(options_coverage_value["assessment_stocks"])}; months
  {int(options_coverage_value["assessment_months"])}.
- Joined assessment rows: {len(assessment):,}; protected market rows materialised: 0;
  protected option observations materialised: 0.

## Test A — clean registered-loop completion

{metric_line("S0", a["S0"])}
{metric_line("S1", a["S1"])}
{metric_line("S2", a["S2"])}

- S1−S0: `{json.dumps(_json_safe(s10), sort_keys=True)}`.
- S2−S1: `{json.dumps(_json_safe(s21), sort_keys=True)}`.

## Test B — movement above previous-close IV expectation

{metric_line("O0", b["O0"])}
{metric_line("O1", b["O1"])}
{metric_line("O2", b["O2"])}

- O1−O0: `{json.dumps(_json_safe(o10), sort_keys=True)}`.
- O2−O1: `{json.dumps(_json_safe(o21), sort_keys=True)}`.
- Ridge metrics: `{continuous.to_json(orient="records")}`.

## Persistence census

- BROAD_CONFLICT mean IV residuals: 15m
  {float(broad["mean_iv_residual_15m"]):.8f}; same close
  {float(broad["mean_iv_residual_to_close"]):.8f}; next close
  {float(broad["mean_iv_residual_next_close"]):.8f}; third close
  {float(broad["mean_iv_residual_third_close"]):.8f}.
- LOW_ROUTE_SUPPORT mean IV residuals: 15m
  {float(low["mean_iv_residual_15m"]):.8f}; same close
  {float(low["mean_iv_residual_to_close"]):.8f}; next close
  {float(low["mean_iv_residual_next_close"]):.8f}; third close
  {float(low["mean_iv_residual_third_close"]):.8f}.

The horizon mapping is descriptive research classification only, not a DTE recommendation or
trading instruction.

## Stability, nulls, and reproducibility

- Bootstrap rows: {len(bootstrap)} from exactly 10 fixed-prediction session draws.
- Options null refits: {len(options_null)}; route null refits: {len(route_null)}.
- Determinism passed: `{bool(determinism["passed"])}`; maximum feature difference
  {float(determinism["maximum_feature_difference"]):.1e}; maximum probability difference
  {float(determinism["maximum_probability_difference"]):.1e}.
- Independent audit: pending standalone auditor execution.

No result is option profitability, an intraday option fill, economic or directional edge,
prospective validation, trading utility, or a deployable strategy.
"""


def _empty_artifact(path: Path) -> None:
    if path.exists():
        return
    if path.suffix == ".json":
        write_json(path, {"status": "not_produced"})
    elif path.suffix == ".csv":
        write_csv(path, pd.DataFrame({"status": ["not_produced"]}))
    elif path.suffix == ".parquet":
        pd.DataFrame({"status": pd.Series(dtype=str)}).to_parquet(path, index=False)
    elif path.suffix == ".md":
        path.write_text("# Not produced\n", encoding="utf-8")


def finalize_blocker(
    blocker: ScreenBlocker,
    *,
    contract_value: Mapping[str, Any],
    daily_stock_regime_status: str = "blocked",
    daily_options_regime_status: str = "blocked",
) -> None:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(PRIMARY / "contract.json", contract_value)
    statuses = {
        "daily_stock_regime_status": daily_stock_regime_status,
        "daily_options_regime_status": daily_options_regime_status,
        "test_a_daily_stock_increment_status": "blocked",
        "test_a_daily_options_increment_status": "blocked",
        "test_b_daily_stock_increment_status": "blocked",
        "test_b_intraday_route_increment_status": "blocked",
        "mismatch_status": "blocked",
        "persistence_horizon_status": "blocked",
    }
    decision = {
        **SAFETY_FLAGS,
        "overall_decision": blocker.decision,
        "blocker_detail": blocker.detail,
        **statuses,
    }
    write_json(PRIMARY / "decision.json", decision)
    write_json(
        PRIMARY / "lightweight_audit.json",
        {**SAFETY_FLAGS, "passed": False, "blocker": blocker.decision},
    )
    write_json(
        PRIMARY / "determinism_check.json",
        {**SAFETY_FLAGS, "passed": False, "blocker": blocker.decision},
    )
    write_json(
        PRIMARY / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "protected_market_rows_materialised": 0,
            "protected_option_observations_materialised": 0,
        },
    )
    report = (
        "# Daily Stock × Options Regime Context Quick Screen V0\n\n"
        f"Overall decision: `{blocker.decision}`.\n\n{blocker.detail}\n\n"
        "No cross-market model result was promoted or interpreted as trading utility.\n"
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    for name in REQUIRED_ARTIFACTS:
        _empty_artifact(PRIMARY / name)


def run(
    *,
    provider_root: Path = DEFAULT_PROVIDER_ROOT,
    options_cache_path: Path | None = None,
) -> str:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    contract_value = contract()
    write_json(PRIMARY / "contract.json", contract_value)
    daily_stock_regime_status = "blocked"
    daily_options_regime_status = "blocked"
    try:
        structural, reconstruction = load_structural_panel()
        write_json(PRIMARY / "structural_panel_reconstruction.json", reconstruction)
        trace = load_trace()
        full_bars, full_bar_source = load_full_regular_session_bars(provider_root.resolve())
        full_bar_overlap = audit_full_bar_overlap(trace, full_bars)
        daily_bars = aggregate_daily_bars(full_bars)
        daily_raw_information = calculate_daily_stock_raw_features(daily_bars)
        (
            stock_raw,
            stock_dimensions,
            stock_parameters,
            stock_regime,
            stock_support,
        ) = build_stock_context(structural, daily_bars)
        stock_raw.to_parquet(PRIMARY / "daily_stock_raw_features.parquet", index=False)
        stock_dimensions.to_parquet(PRIMARY / "daily_stock_dimensions.parquet", index=False)
        write_json(
            PRIMARY / "daily_stock_feature_manifest.json",
            {
                **dimension_manifest(stock_parameters),
                "raw_features": DAILY_STOCK_RAW_FEATURES,
                "missing_indicators": tuple(
                    f"{feature}_missing" for feature in DAILY_STOCK_RAW_FEATURES
                ),
                "dimensions": DAILY_STOCK_DIMENSIONS,
                "activity_field": "EODHD historical activity proxy",
                "activity_is_confirmed_exchange_volume": False,
                "minimum_valid_trailing_sessions_for_20": 15,
                "corporate_action_handling": (
                    "continuity adjustment at inferred 0.55/1.80 boundaries"
                ),
                "support": stock_support,
            },
        )
        write_json(PRIMARY / "daily_stock_regime_mapping.json", regime_mapping(stock_regime))
        stock_diagnostics = regime_diagnostics(stock_dimensions, stock_regime)
        write_csv(PRIMARY / "daily_stock_regime_diagnostics.csv", stock_diagnostics)
        stock_regime_supported_at_blocker, _stock_regime_evidence_at_blocker = regime_support(
            stock_dimensions, "daily_stock_regime"
        )
        daily_stock_regime_status = (
            "supported" if stock_regime_supported_at_blocker else "descriptive_only"
        )

        cache_path, recovery_receipt = discover_options_cache(options_cache_path)
        options_cache = load_options_cache(cache_path)
        options_raw, coverage_gap, reused_records = build_options_context(stock_raw, options_cache)
        options_raw.to_parquet(PRIMARY / "daily_options_raw_features.parquet", index=False)
        write_csv(PRIMARY / "daily_options_coverage_gap.csv", coverage_gap)
        chronology = write_chronology_audit(stock_raw, options_raw)
        coverage = raw_options_support(structural, options_raw)
        prior_options_source = read_json(OPTIONS_V01_SOURCE)
        protected = {
            **SAFETY_FLAGS,
            "protected_start": PROTECTED_START.isoformat(),
            "protected_market_rows_materialised": 0,
            "protected_option_observations_materialised": 0,
            "maximum_market_observation_date": str(full_bars["session"].max()),
            "maximum_option_observation_date": str(options_cache["trade_date"].max()),
            "maximum_contract_expiration_metadata": str(
                pd.to_datetime(options_cache["expiration_date"]).dt.date.max()
            ),
            "expiration_metadata_may_cross_boundary": True,
        }
        write_json(PRIMARY / "protected_boundary_audit.json", protected)
        preliminary_source_manifest = {
            **SAFETY_FLAGS,
            "starting_branch": STARTING_BRANCH,
            "starting_sha": STARTING_SHA,
            "final_branch": FINAL_BRANCH,
            "dates": {
                "development_start": DEVELOPMENT_START.isoformat(),
                "assessment_end": ASSESSMENT_END.isoformat(),
                "protected_start": PROTECTED_START.isoformat(),
            },
            "cohort": FROZEN_COHORT,
            "sources": {
                "dense_panel": str(DENSE_PANEL),
                "dense_panel_sha256": sha256_file(DENSE_PANEL),
                "trace_panel": str(TRACE_PANEL),
                "trace_panel_sha256": sha256_file(TRACE_PANEL),
                "full_regular_session_underlying_root": str(provider_root.resolve()),
                "repaired_exact_date_options_cache": str(cache_path),
                "repaired_exact_date_options_cache_sha256": sha256_file(cache_path),
            },
            "full_regular_session_underlying": full_bar_source,
            "full_bar_overlap_with_frozen_trace": full_bar_overlap,
            "options_cache_reprocessing": {
                "source_experiment": "EODHD Fixed Overnight Options Strategy Quick Screen V0.1",
                "repaired_exact_observation_date_filtering": True,
                "cached_exact_date_records_recovered": cast(
                    Mapping[str, Any], prior_options_source.get("options_cache", {})
                ).get("cached_exact_date_records_recovered"),
                "cached_responses_examined": cast(
                    Mapping[str, Any], prior_options_source.get("options_cache", {})
                ).get("cached_responses_examined"),
                "cache_rows_loaded": len(options_cache),
                "cache_stock_dates": int(
                    options_cache.groupby(
                        ["underlying_symbol", "trade_date"], observed=True
                    ).ngroups
                ),
                "maximum_cached_dte": int(options_cache["dte"].max()),
            },
            "options_records_reused": reused_records,
            "newly_downloaded_records": (
                int(recovery_receipt["newly_downloaded_records"])
                if recovery_receipt is not None
                else 0
            ),
            "newly_downloaded_bytes": (
                int(recovery_receipt["newly_downloaded_bytes"])
                if recovery_receipt is not None
                else 0
            ),
            "network_requests_made": (
                int(recovery_receipt["network_requests_made"])
                if recovery_receipt is not None
                else 0
            ),
            "bounded_download": (
                {
                    key: recovery_receipt.get(key)
                    for key in (
                        "status",
                        "planned_exact_stock_date_requests",
                        "planned_gap_rows",
                        "network_requests_made",
                        "newly_downloaded_records",
                        "newly_downloaded_bytes",
                        "protected_option_observations_materialised",
                        "output_cache",
                    )
                }
                if recovery_receipt is not None
                else (
                    {
                        key: read_json(DOWNLOAD_RECEIPT).get(key)
                        for key in (
                            "status",
                            "planned_exact_stock_date_requests",
                            "planned_gap_rows",
                            "network_requests_made",
                            "newly_downloaded_records",
                            "newly_downloaded_bytes",
                            "protected_option_observations_materialised",
                            "output_cache",
                        )
                    }
                    if DOWNLOAD_RECEIPT.is_file()
                    else {"status": "not_run"}
                )
            ),
            "options_coverage": coverage,
            "options_gap_rows": len(coverage_gap),
            "bounded_download_required_gap_rows": int(
                coverage_gap["bounded_download_required"].sum()
            ),
            "stock_support": stock_support,
        }
        write_json(PRIMARY / "source_manifest.json", preliminary_source_manifest)
        base_options_manifest = {
            **SAFETY_FLAGS,
            "raw_features": DAILY_OPTIONS_RAW_FEATURES,
            "missing_indicators": DAILY_OPTIONS_MISSING_INDICATORS,
            "dimensions": DAILY_OPTIONS_DIMENSIONS,
            "front_pair_rule": "nearest 7-45 DTE common call/put strike by abs(log(strike/close))",
            "back_pair_rule": "nearest 46-90 DTE common call/put strike",
            "skew_delta_targets": {"put": -0.25, "call": 0.25, "maximum_error": 0.10},
            "newly_downloaded_records": preliminary_source_manifest["newly_downloaded_records"],
            "newly_downloaded_bytes": preliminary_source_manifest["newly_downloaded_bytes"],
            "support": coverage,
        }
        if not bool(coverage["passed"]):
            write_json(
                PRIMARY / "daily_options_feature_manifest.json",
                {
                    **base_options_manifest,
                    "status": "blocked_before_dimension_fit",
                    "fitted_period": "not_fitted",
                    "reason": (
                        "front coverage or development support for a frozen raw "
                        "options feature is insufficient"
                    ),
                },
            )
            pd.DataFrame({"status": ["blocked_insufficient_daily_options_coverage"]}).to_parquet(
                PRIMARY / "daily_options_dimensions.parquet", index=False
            )
            write_json(
                PRIMARY / "daily_options_regime_mapping.json",
                {
                    **SAFETY_FLAGS,
                    "status": "blocked_insufficient_daily_options_coverage",
                    "n_components": 4,
                    "fitted": False,
                },
            )
            write_csv(
                PRIMARY / "daily_options_regime_diagnostics.csv",
                pd.DataFrame({"status": ["blocked_insufficient_daily_options_coverage"]}),
            )
            raise ScreenBlocker(
                "blocked_insufficient_daily_options_coverage",
                (
                    "previous-close front-pair coverage or frozen options-dimension "
                    f"support is inadequate: {coverage}"
                ),
            )
        options_dimensions, options_parameters, options_regime = fit_options_context(options_raw)
        options_dimensions.to_parquet(PRIMARY / "daily_options_dimensions.parquet", index=False)
        write_json(
            PRIMARY / "daily_options_feature_manifest.json",
            {
                **base_options_manifest,
                **dimension_manifest(options_parameters),
                "status": "fitted",
            },
        )
        write_json(PRIMARY / "daily_options_regime_mapping.json", regime_mapping(options_regime))
        options_diagnostics = regime_diagnostics(options_dimensions, options_regime)
        write_csv(PRIMARY / "daily_options_regime_diagnostics.csv", options_diagnostics)
        options_regime_supported_at_blocker, _options_regime_evidence_at_blocker = regime_support(
            options_dimensions, "daily_options_regime"
        )
        daily_options_regime_status = (
            "supported" if options_regime_supported_at_blocker else "descriptive_only"
        )
        panel, mismatch_standardization = join_cross_market_panel(
            structural,
            stock_dimensions,
            options_dimensions,
            full_bars,
            daily_raw_information,
        )
        panel.to_parquet(PRIMARY / "daily_cross_market_panel.parquet", index=False)
        write_json(
            PRIMARY / "mismatch_feature_manifest.json",
            {
                **SAFETY_FLAGS,
                "features": MISMATCH_FEATURES,
                "standardization_fitted_period": "development_2024_only",
                "standardization": {
                    name: asdict(value) for name, value in mismatch_standardization.items()
                },
                "additional_interactions": 0,
            },
        )
        stock_regime_supported, stock_regime_evidence = regime_support(
            stock_dimensions, "daily_stock_regime"
        )
        options_regime_supported, options_regime_evidence = regime_support(
            options_dimensions, "daily_options_regime"
        )

        try:
            (
                models,
                boundaries,
                development_predictions,
                assessment_predictions,
            ) = fit_all_models(panel)
        except (RuntimeError, ConvergenceWarning) as error:
            raise ScreenBlocker(
                "blocked_model_convergence_failure",
                f"primary model convergence failed: {error}",
            ) from error
        development_cutoffs = DevelopmentCutoffs(
            mismatch_compression_vs_iv=float(
                development_predictions["mismatch_compression_vs_iv"].median()
            ),
            mismatch_complacent_conflict=float(
                development_predictions["mismatch_complacent_conflict"].median()
            ),
            options_implied_tension=float(
                development_predictions["options_implied_tension"].median()
            ),
            options_front_urgency=float(development_predictions["options_front_urgency"].median()),
            mismatch_route_vs_premium=float(
                development_predictions["mismatch_route_vs_premium"].median()
            ),
        )
        assessment_predictions.to_parquet(PRIMARY / "assessment_predictions.parquet", index=False)
        write_json(
            PRIMARY / "model_configurations.json",
            {
                **SAFETY_FLAGS,
                "primary_classifier_fit_count": 6,
                "ridge_fit_count": 2,
                "models": {name: model.as_dict() for name, model in models.items()},
                "development_prediction_boundaries": boundaries,
                "development_subgroup_cutoffs": asdict(development_cutoffs),
            },
        )
        write_json(
            PRIMARY / "model_coefficients.json",
            {
                **SAFETY_FLAGS,
                "models": {
                    name: {
                        "intercept": model.intercept,
                        "design_columns": model.design_columns,
                        "coefficients": model.coefficients,
                    }
                    for name, model in models.items()
                },
            },
        )
        (
            test_a,
            test_a_monthly,
            test_a_regime,
            test_b,
            test_b_monthly,
            test_b_regime,
        ) = model_metric_tables(
            assessment_predictions,
            boundaries,
            development_cutoffs,
        )
        write_csv(PRIMARY / "test_a_metrics.csv", test_a)
        write_csv(PRIMARY / "test_a_monthly_metrics.csv", test_a_monthly)
        write_csv(PRIMARY / "test_a_regime_metrics.csv", test_a_regime)
        write_csv(PRIMARY / "test_b_metrics.csv", test_b)
        write_csv(PRIMARY / "test_b_monthly_metrics.csv", test_b_monthly)
        write_csv(PRIMARY / "test_b_regime_metrics.csv", test_b_regime)

        continuous_rows: list[dict[str, Any]] = []
        for model_id in ("R0", "R1"):
            continuous_rows.append(
                {
                    "model": model_id,
                    **continuous_residual_metrics(
                        assessment_predictions,
                        target_column="iv_absolute_residual_15m",
                        prediction_column=f"{model_id}_prediction",
                    ),
                    "mean_predicted_residual": float(
                        np.average(
                            assessment_predictions[f"{model_id}_prediction"],
                            weights=assessment_predictions["row_weight"],
                        )
                    ),
                }
            )
        continuous = pd.DataFrame(continuous_rows)
        r0 = continuous.loc[continuous["model"].eq("R0")].iloc[0]
        r1 = continuous.loc[continuous["model"].eq("R1")].iloc[0]
        continuous["R1_minus_R0_MAE_improvement"] = float(r0["weighted_mae"] - r1["weighted_mae"])
        continuous["R1_minus_R0_RMSE_improvement"] = float(
            r0["weighted_rmse"] - r1["weighted_rmse"]
        )
        write_csv(PRIMARY / "continuous_residual_metrics.csv", continuous)

        persistence, regime_pairs, horizon_mapping = persistence_tables(
            development_predictions, assessment_predictions
        )
        write_csv(PRIMARY / "persistence_horizon_metrics.csv", persistence)
        write_csv(PRIMARY / "regime_pair_persistence_metrics.csv", regime_pairs)
        write_csv(PRIMARY / "dte_horizon_mapping.csv", horizon_mapping)
        bootstrap = bootstrap_intervals(
            assessment_predictions,
            boundaries,
            development_cutoffs,
        )
        write_csv(PRIMARY / "bootstrap_metrics.csv", bootstrap)
        panel_with_predictions = pd.concat(
            [development_predictions, assessment_predictions], ignore_index=True
        )
        try:
            options_null, route_null = null_refits(
                panel_with_predictions,
                boundaries,
                mismatch_standardization,
            )
        except (RuntimeError, ConvergenceWarning) as error:
            raise ScreenBlocker(
                "blocked_model_convergence_failure",
                f"null-refit model convergence failed: {error}",
            ) from error
        write_csv(PRIMARY / "options_null_metrics.csv", options_null)
        write_csv(PRIMARY / "route_null_metrics.csv", route_null)
        concentration = concentration_metrics(panel)
        write_csv(PRIMARY / "concentration_metrics.csv", concentration)

        statuses, gates = component_decision(
            panel=panel,
            test_a=test_a,
            test_a_monthly=test_a_monthly,
            test_a_subgroup=test_a_regime,
            test_b=test_b,
            test_b_monthly=test_b_monthly,
            bootstrap=bootstrap,
            options_null=options_null,
            route_null=route_null,
            stock_regime_supported=stock_regime_supported,
            options_regime_supported=options_regime_supported,
        )
        overall = choose_daily_context_decision(
            blocker=None,
            test_a_daily_stock_supported=statuses["test_a_daily_stock_increment_status"]
            == "supported",
            test_a_daily_options_supported=statuses["test_a_daily_options_increment_status"]
            == "supported",
            test_b_daily_stock_supported=statuses["test_b_daily_stock_increment_status"]
            == "supported",
            test_b_intraday_route_supported=statuses["test_b_intraday_route_increment_status"]
            == "supported",
            mismatch_supported=statuses["mismatch_status"] == "supported",
            descriptive=any(value == "descriptive_only" for value in statuses.values()),
        )
        decision = {
            **SAFETY_FLAGS,
            "overall_decision": overall,
            **statuses,
            "gates": gates,
            "daily_stock_regime_support": stock_regime_evidence,
            "daily_options_regime_support": options_regime_evidence,
        }
        write_json(PRIMARY / "decision.json", decision)

        determinism = determinism_rebuild(
            structural=structural,
            full_bars=full_bars,
            daily_bars=daily_bars,
            cache=options_cache,
            first_panel=panel,
            first_models=models,
            first_stock_regime=stock_regime,
            first_options_regime=options_regime,
            first_persistence=(persistence, regime_pairs, horizon_mapping),
            first_decision=decision,
            frozen_bootstrap=bootstrap,
            frozen_options_null=options_null,
            frozen_route_null=route_null,
        )
        if not bool(determinism["passed"]):
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"determinism rebuild failed: {determinism}",
            )
        write_json(PRIMARY / "determinism_check.json", determinism)
        manual_differences: dict[str, float] = {}
        manual_sample = assessment_predictions.head(100)
        for model_id, model in models.items():
            manual = manual_model_prediction(manual_sample, model.as_dict())
            manual_differences[model_id] = float(
                np.max(np.abs(manual - model.predict(manual_sample)))
            )
        lightweight = {
            **SAFETY_FLAGS,
            "passed": bool(max(manual_differences.values()) <= 1e-12),
            "manual_probability_rows_per_model": len(manual_sample),
            "maximum_manual_prediction_difference_by_model": manual_differences,
            "structural_reconstruction_passed": bool(reconstruction["passed"]),
            "chronology_passed": bool(chronology["chronology_passed"].all()),
            "protected_boundary_passed": True,
            "independent_auditor_pending": True,
        }
        write_json(PRIMARY / "lightweight_audit.json", lightweight)
        source_manifest = {
            **preliminary_source_manifest,
            "joined_rows": len(panel),
            "joined_assessment_rows": int(panel["period"].eq("assessment").sum()),
        }
        write_json(PRIMARY / "source_manifest.json", source_manifest)
        create_plots(
            stock_regime=stock_regime,
            options_regime=options_regime,
            test_a=test_a,
            test_b=test_b,
            persistence=persistence,
        )
        report = render_report(
            decision=decision,
            reconstruction=reconstruction,
            stock_support=stock_support,
            options_coverage_value=coverage,
            reused_records=reused_records,
            newly_downloaded_records=int(preliminary_source_manifest["newly_downloaded_records"]),
            newly_downloaded_bytes=int(preliminary_source_manifest["newly_downloaded_bytes"]),
            panel=panel,
            test_a=test_a,
            test_b=test_b,
            continuous=continuous,
            persistence=persistence,
            bootstrap=bootstrap,
            options_null=options_null,
            route_null=route_null,
            determinism=determinism,
        )
        (PRIMARY / "report.md").write_text(report, encoding="utf-8")
        (REPORTS / "report.md").write_text(report, encoding="utf-8")
        return overall
    except ScreenBlocker as blocker:
        finalize_blocker(
            blocker,
            contract_value=contract_value,
            daily_stock_regime_status=daily_stock_regime_status,
            daily_options_regime_status=daily_options_regime_status,
        )
        return blocker.decision
    except Exception as error:
        blocker = ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"unexplained runner failure: {type(error).__name__}: {error}",
        )
        finalize_blocker(
            blocker,
            contract_value=contract_value,
            daily_stock_regime_status=daily_stock_regime_status,
            daily_options_regime_status=daily_options_regime_status,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=DEFAULT_PROVIDER_ROOT,
        help="Existing EODHD processed-stock root containing full regular sessions.",
    )
    parser.add_argument(
        "--options-cache",
        type=Path,
        default=None,
        help="Optional repaired exact-date canonical cache path.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    print(
        run(
            provider_root=arguments.provider_root,
            options_cache_path=arguments.options_cache,
        )
    )


if __name__ == "__main__":
    main()
