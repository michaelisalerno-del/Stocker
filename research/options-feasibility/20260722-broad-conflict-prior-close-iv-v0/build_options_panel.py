#!/usr/bin/env python3
"""Build the causal prior-close options/movement panel and bounded V0 analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from datetime import time as datetime_time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from run_screen_v0 import (  # noqa: E402
    PRIMARY,
    TRACE_PANEL,
    default_provider_root,
    load_clean_advance_panel,
    options_data_dir,
    write_csv,
    write_json,
    write_parquet,
)

from stocker_research.broad_conflict_options_iv_screen_v0 import (  # noqa: E402
    DENSE_H0_FEATURES,
    FROZEN_COHORT,
    OPTIONS_PRIMARY_FEATURES,
    OPTIONS_SCREEN_DECISIONS,
    ROUTE_FEATURES,
    SAFETY_FLAGS,
    add_iv_relative_outcomes,
    assign_development_frozen_iv_deciles,
    broad_conflict_iv_gate_passes,
    build_matched_control_relations,
    calculate_optional_option_features,
    calculate_primary_option_features,
    choose_options_movement_decision,
    compute_underlying_movement_outcomes,
    coverage_gates_pass,
    fit_options_linear_model,
    fixed_session_bootstrap_multiplicities,
    iv_movement_approximations,
    o1_model_gate_passes,
    permute_intact_route_bundle,
    select_primary_atm_pair,
    validate_exact_previous_session_join,
)

MATERIALLY_ADVERSE_LOG_LOSS = -0.005
MATERIALLY_ADVERSE_BRIER = -0.002
MATERIALLY_ADVERSE_IV_RESIDUAL = -0.0005


def classify_analysis_blocker(error: Exception) -> str:
    message = str(error)
    if message in OPTIONS_SCREEN_DECISIONS and message.startswith("blocked_"):
        return message
    if "failed to converge" in message:
        return "blocked_model_convergence_failure"
    if "protected" in message.casefold():
        return "blocked_protected_boundary_failure"
    if "chronology" in message.casefold() or "previous trading session" in message.casefold():
        return "blocked_chronology_or_leakage_failure"
    return "blocked_reproducibility_or_audit_failure"


def write_analysis_blocker(output: Path, blocker: str) -> dict[str, Any]:
    decision = {
        **SAFETY_FLAGS,
        "decision": blocker,
        "options_download_status": (
            "blocked" if blocker == "blocked_options_download_incomplete" else "supported"
        ),
        "options_coverage_status": "blocked",
        "iv_excess_model_status": "blocked",
        "broad_conflict_movement_status": "blocked",
        "matched_control_status": "blocked",
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "lightweight_audit.json",
        {
            "passed": False,
            "status": blocker,
            "audit_scope": "analysis_failed_before_independent_audit_completion",
        },
    )
    write_json(
        output / "determinism_check.json",
        {
            "status": blocker,
            "passed": False,
            "reason": "analysis_failed_before_determinism_completion",
        },
    )
    (output / "report.md").write_text(
        "# Prior-Close Options IV Movement Screen V0\n\n"
        f"Primary decision: `{blocker}`\n\n"
        "The fail-closed analysis did not produce an options-movement inference. "
        "No intraday option fill or option P&L was calculated.\n",
        encoding="utf-8",
    )
    return decision


def _load_full_underlying_bars(provider_root: Path, *, sessions: set[str]) -> pd.DataFrame:
    """Load protected, regular-session unadjusted bars and verify the frozen overlap."""

    rows: list[pd.DataFrame] = []
    minimum_session = min(sessions)
    for symbol in FROZEN_COHORT:
        path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        raw = pd.read_parquet(
            path,
            columns=["timestamp", "open", "high", "low", "close"],
            filters=[
                ("timestamp", ">=", pd.Timestamp(f"{minimum_session}T00:00:00Z")),
                ("timestamp", "<", pd.Timestamp("2025-08-23T00:00:00Z")),
            ],
        )
        timestamp = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
        local = timestamp.dt.tz_convert("America/New_York")
        session = local.dt.strftime("%Y-%m-%d")
        clock = local.dt.time
        regular = (
            session.isin(sessions) & clock.ge(datetime_time(9, 30)) & clock.lt(datetime_time(16, 0))
        )
        selected = raw.loc[regular, ["open", "high", "low", "close"]].copy()
        selected["symbol"] = symbol
        selected["session"] = session.loc[regular].to_numpy()
        selected["bar_start_timestamp"] = timestamp.loc[regular].to_numpy()
        selected["bar_complete_timestamp"] = (
            timestamp.loc[regular] + pd.Timedelta(minutes=5)
        ).to_numpy()
        selected = selected.sort_values(["session", "bar_start_timestamp"], kind="mergesort")
        selected["bar_ordinal"] = selected.groupby("session", sort=False).cumcount()
        rows.append(selected)
    bars = pd.concat(rows, ignore_index=True)
    if bars.duplicated(["symbol", "session", "bar_ordinal"]).any():
        raise RuntimeError("blocked_options_data_integrity_failure")
    trace = pd.read_parquet(
        TRACE_PANEL,
        columns=[
            "symbol",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
            "open",
            "high",
            "low",
            "close",
        ],
    )
    trace = trace.loc[trace["session"].astype(str).isin(sessions)]
    overlap = bars.loc[bars["bar_ordinal"].le(37)].merge(
        trace,
        on=["symbol", "session", "bar_ordinal"],
        how="outer",
        suffixes=("_raw", "_trace"),
        indicator=True,
        validate="one_to_one",
    )
    numeric_differences = [
        np.abs(
            overlap[f"{column}_raw"].to_numpy(float) - overlap[f"{column}_trace"].to_numpy(float)
        )
        for column in ("open", "high", "low", "close")
    ]
    timestamp_matches = bool(
        pd.to_datetime(overlap["bar_start_timestamp_raw"], utc=True).equals(
            pd.to_datetime(overlap["bar_start_timestamp_trace"], utc=True)
        )
        and pd.to_datetime(overlap["bar_complete_timestamp_raw"], utc=True).equals(
            pd.to_datetime(overlap["bar_complete_timestamp_trace"], utc=True)
        )
    )
    if (
        not overlap["_merge"].eq("both").all()
        or max(float(np.max(value)) for value in numeric_differences) > 1e-12
        or not timestamp_matches
    ):
        raise RuntimeError("blocked_structural_panel_reconstruction_failure")
    return bars


def _read_canonical(data_dir: Path, *, expected_chunk_ids: set[str] | None = None) -> pd.DataFrame:
    paths = sorted((data_dir / "canonical").glob("*.parquet"))
    if expected_chunk_ids is not None:
        paths = [path for path in paths if path.stem in expected_chunk_ids]
        observed_chunk_ids = {path.stem for path in paths}
        if observed_chunk_ids != expected_chunk_ids:
            raise RuntimeError("blocked_options_download_incomplete")
    if not paths:
        raise RuntimeError("blocked_options_download_incomplete")
    frames = [pd.read_parquet(path) for path in paths]
    canonical = pd.concat(frames, ignore_index=True)
    if canonical.empty:
        raise RuntimeError("blocked_options_download_incomplete")
    canonical["trade_date"] = pd.to_datetime(canonical["trade_date"], errors="raise").dt.date
    canonical["expiration_date"] = pd.to_datetime(
        canonical["expiration_date"], errors="raise"
    ).dt.date
    return canonical


def _quality_summary(canonical: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    bid = pd.to_numeric(canonical["bid"], errors="coerce")
    ask = pd.to_numeric(canonical["ask"], errors="coerce")
    iv = pd.to_numeric(canonical["implied_volatility"], errors="coerce")
    delta = pd.to_numeric(canonical["delta"], errors="coerce")
    gamma = pd.to_numeric(canonical["gamma"], errors="coerce")
    open_interest = pd.to_numeric(canonical["open_interest"], errors="coerce")
    metrics: dict[str, int | float] = {
        "raw_records": int(manifest.get("raw_records", 0)),
        "canonical_records": len(canonical),
        "duplicate_records": int(manifest.get("duplicate_records", 0)),
        "rejected_records": int(manifest.get("rejected_records", 0)),
        "calls": int(canonical["option_type"].eq("call").sum()),
        "puts": int(canonical["option_type"].eq("put").sum()),
        "missing_iv": int(iv.isna().sum()),
        "iv_outside_0_005_to_5": int((iv.notna() & ~iv.between(0.005, 5.0)).sum()),
        "missing_bid": int(bid.isna().sum()),
        "missing_ask": int(ask.isna().sum()),
        "crossed_quotes": int((bid.notna() & ask.notna() & ask.lt(bid)).sum()),
        "zero_bids": int(bid.eq(0.0).sum()),
        "missing_open_interest": int(open_interest.isna().sum()),
        "implausible_delta": int((delta.notna() & delta.abs().gt(1.05)).sum()),
        "implausible_gamma": int((gamma.notna() & gamma.lt(0.0)).sum()),
        "expiration_before_trade": int(
            (canonical["expiration_date"] < canonical["trade_date"]).sum()
        ),
        "dte_minimum": float(pd.to_numeric(canonical["dte"], errors="raise").min()),
        "dte_median": float(pd.to_numeric(canonical["dte"], errors="raise").median()),
        "dte_maximum": float(pd.to_numeric(canonical["dte"], errors="raise").max()),
    }
    rows = [{"metric": key, "value": value, "status": "observed"} for key, value in metrics.items()]
    for symbol, count in canonical.groupby("underlying_symbol", sort=True).size().items():
        rows.append(
            {"metric": f"records_by_symbol:{symbol}", "value": int(count), "status": "observed"}
        )
    months = pd.to_datetime(canonical["trade_date"], errors="raise").dt.strftime("%Y-%m")
    for month, count in months.value_counts(sort=False).sort_index().items():
        rows.append(
            {"metric": f"records_by_month:{month}", "value": int(count), "status": "observed"}
        )
    for trade_date, count in canonical.groupby("trade_date", sort=True).size().items():
        rows.append(
            {
                "metric": f"records_by_trade_date:{trade_date}",
                "value": int(count),
                "status": "observed",
            }
        )
    dte = pd.to_numeric(canonical["dte"], errors="raise")
    for quantile in (0.25, 0.75):
        rows.append(
            {
                "metric": f"dte_q{int(quantile * 100):02d}",
                "value": float(dte.quantile(quantile)),
                "status": "observed",
            }
        )
    return pd.DataFrame(rows)


def _provider_moneyness_audit(canonical: pd.DataFrame, price_audit: pd.DataFrame) -> pd.DataFrame:
    """Check provider moneyness ratios against each unadjusted prior close."""

    output = price_audit.copy()
    output["provider_moneyness_check"] = "not_available"
    output["provider_moneyness_maximum_relative_difference"] = math.nan
    chains = {
        (str(symbol), cast(date, trade_date)): group
        for (symbol, trade_date), group in canonical.groupby(
            ["underlying_symbol", "trade_date"], sort=False
        )
    }
    for index, source in output.iterrows():
        symbol = str(source["symbol"])
        trade_date = date.fromisoformat(str(source["required_options_date"]))
        chain = chains.get((symbol, trade_date))
        if chain is None:
            continue
        provider = pd.to_numeric(chain["moneyness"], errors="coerce")
        strike = pd.to_numeric(chain["strike"], errors="coerce")
        valid = provider.notna() & strike.gt(0.0)
        if not valid.any():
            continue
        close = float(source["previous_close_underlying_price"])
        strike_over_spot = strike.loc[valid].to_numpy(float) / close
        spot_over_strike = close / strike.loc[valid].to_numpy(float)
        observed = provider.loc[valid].to_numpy(float)
        difference = np.minimum(
            np.abs(observed - strike_over_spot) / np.maximum(strike_over_spot, 1e-12),
            np.abs(observed - spot_over_strike) / np.maximum(spot_over_strike, 1e-12),
        )
        maximum = float(np.max(difference))
        output.at[index, "provider_moneyness_maximum_relative_difference"] = maximum
        output.at[index, "provider_moneyness_check"] = (
            "consistent_ratio_orientation_audited" if maximum <= 0.05 else "inconsistent"
        )
    return output


def _select_pairs(
    canonical: pd.DataFrame,
    price_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    chains = {
        (str(symbol), cast(date, trade_date)): group.copy()
        for (symbol, trade_date), group in canonical.groupby(
            ["underlying_symbol", "trade_date"], sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    ordered = price_audit.sort_values(["symbol", "signal_date"], kind="mergesort")
    for source in cast(list[dict[str, Any]], ordered.to_dict(orient="records")):
        symbol = str(source["symbol"])
        signal_date = date.fromisoformat(str(source["signal_date"]))
        required_date = date.fromisoformat(str(source["required_options_date"]))
        validate_exact_previous_session_join(
            signal_date=signal_date,
            required_options_date=required_date,
            actual_options_date=required_date,
        )
        identity = {
            "symbol": symbol,
            "signal_date": signal_date.isoformat(),
            "required_options_date": required_date.isoformat(),
        }
        chain = chains.get((symbol, required_date))
        exact_chain_available = chain is not None and not chain.empty
        if bool(source["split_boundary_ambiguous"]):
            manifest.append(
                {
                    **identity,
                    "exact_chain_available": exact_chain_available,
                    "available": False,
                    "reason": "split_boundary_ambiguous",
                }
            )
            continue
        if str(source.get("provider_moneyness_check", "not_available")) == "inconsistent":
            manifest.append(
                {
                    **identity,
                    "exact_chain_available": exact_chain_available,
                    "available": False,
                    "reason": "provider_moneyness_inconsistent",
                }
            )
            continue
        if not exact_chain_available:
            manifest.append(
                {
                    **identity,
                    "exact_chain_available": False,
                    "available": False,
                    "reason": "exact_prior_chain_missing",
                }
            )
            continue
        assert chain is not None
        previous_close = float(source["previous_close_underlying_price"])
        selection = select_primary_atm_pair(chain, previous_close=previous_close)
        selection_record = {
            **identity,
            "exact_chain_available": True,
            "available": selection.available,
            "reason": selection.reason,
            "selected_expiry": (
                None if selection.expiration_date is None else selection.expiration_date.isoformat()
            ),
            "selected_strike": selection.strike,
            "call_contract_id": selection.call_contract_id,
            "put_contract_id": selection.put_contract_id,
        }
        manifest.append(selection_record)
        if not selection.available:
            continue
        features = calculate_primary_option_features(selection, previous_close=previous_close)
        features.update(
            calculate_optional_option_features(
                chain,
                front_selection=selection,
                previous_close=previous_close,
            )
        )
        features.update(iv_movement_approximations(float(features["atm_iv"])))
        rows.append(
            {
                **identity,
                "previous_close_underlying_price": previous_close,
                **features,
            }
        )
    return pd.DataFrame(rows), manifest


def _coverage(
    clean: pd.DataFrame,
    pairs: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    join = clean.copy()
    join["signal_date"] = pd.to_datetime(join["session"], errors="raise").dt.strftime("%Y-%m-%d")
    join = join.merge(
        pairs[["symbol", "signal_date"]].assign(valid_pair=1),
        on=["symbol", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    join["valid_pair"] = join["valid_pair"].fillna(0).astype(int)
    coverage_rows: list[dict[str, object]] = []
    for symbol in FROZEN_COHORT:
        for period in ("development", "assessment"):
            group = join.loc[join["symbol"].eq(symbol) & join["period"].eq(period)]
            coverage_rows.append(
                {
                    "symbol": symbol,
                    "period": period,
                    "required_rows": len(group),
                    "valid_pair_rows": int(group["valid_pair"].sum()),
                    "coverage": float(group["valid_pair"].mean()) if len(group) else 0.0,
                    "status": "supported" if bool(group["valid_pair"].any()) else "not_supported",
                }
            )
    coverage_frame = pd.DataFrame(coverage_rows)
    assessment = join.loc[join["period"].eq("assessment") & join["valid_pair"].eq(1)].copy()
    development = join.loc[join["period"].eq("development") & join["valid_pair"].eq(1)].copy()
    shares = (
        assessment.groupby("symbol", sort=True)["row_weight"].sum() / assessment["row_weight"].sum()
        if not assessment.empty
        else pd.Series(dtype=float)
    )
    concentration = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "weighted_share": float(shares.get(symbol, 0.0)),
                "status": "supported"
                if float(shares.get(symbol, 0.0)) <= 0.12
                else "not_supported",
            }
            for symbol in FROZEN_COHORT
        ]
    )
    paired_both = coverage_frame.pivot(index="symbol", columns="period", values="valid_pair_rows")
    both_count = int((paired_both.min(axis=1) > 0).sum())
    mapping_available = mapping["coverage_available"].astype(str).str.casefold().eq(
        "true"
    ) & pd.to_numeric(mapping["records_returned"], errors="coerce").fillna(0).gt(0)
    evidence = {
        "historical_symbols": int(mapping_available.sum()),
        "paired_symbols_development": both_count,
        "paired_symbols_assessment": both_count,
        "development_row_coverage": len(development)
        / max(int(clean["period"].eq("development").sum()), 1),
        "assessment_row_coverage": len(assessment)
        / max(int(clean["period"].eq("assessment").sum()), 1),
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "assessment_broad_conflict_rows": int(
            assessment["route_resolution_state"].eq("BROAD_CONFLICT").sum()
        ),
        "assessment_low_route_support_rows": int(
            assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT").sum()
        ),
        "maximum_stock_weight_share": float(shares.max()) if len(shares) else 1.0,
        "download_integrity_passed": True,
    }
    evidence["passed"] = coverage_gates_pass(evidence)
    return evidence, coverage_frame, concentration


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    weight = pd.to_numeric(weights, errors="raise").to_numpy(float)
    valid = np.isfinite(numeric) & np.isfinite(weight) & (weight > 0.0)
    if not valid.any():
        return math.nan
    return float(np.average(numeric[valid], weights=weight[valid]))


def _finite_or_zero(value: float) -> float:
    """Encode an unavailable non-binding support statistic without non-JSON NaN."""

    return float(value) if math.isfinite(float(value)) else 0.0


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "weight": pd.to_numeric(weights, errors="coerce"),
        }
    ).dropna()
    frame = frame.loc[frame["weight"].gt(0.0)].sort_values("value", kind="mergesort")
    if frame.empty:
        return math.nan
    cutoff = float(frame["weight"].sum()) / 2.0
    return float(frame.loc[frame["weight"].cumsum().ge(cutoff), "value"].iloc[0])


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="raise"),
            "weight": pd.to_numeric(weights, errors="raise"),
        }
    ).sort_values("value", kind="mergesort")
    cumulative = frame["weight"].cumsum() / frame["weight"].sum()
    return float(frame.loc[cumulative.ge(quantile), "value"].iloc[0])


def _calibration_fit(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Fit the standard weighted logistic calibration intercept and slope diagnostic."""

    if set(target.astype(int)) != {0, 1}:
        return math.nan, math.nan
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(len(target), dtype=float), logits])
    beta = np.asarray([0.0, 1.0], dtype=float)
    for _ in range(50):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35.0, 35.0)))
        gradient = design.T @ (weights * (target - fitted))
        curvature = weights * fitted * (1.0 - fitted)
        information = design.T @ (curvature[:, None] * design)
        information += np.eye(2, dtype=float) * 1e-12
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        beta += step
        if float(np.max(np.abs(step))) <= 1e-10:
            break
    return float(beta[0]), float(beta[1])


def _binary_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    probability_column: str,
    decile_boundary: float,
    quintile_boundary: float,
) -> dict[str, Any]:
    target = frame["movement_exceeds_iv_expected_absolute"].to_numpy(int)
    probability = frame[probability_column].to_numpy(float)
    weights = frame["row_weight"].to_numpy(float)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    base_rate = float(np.average(target, weights=weights))
    assigned = target * probability + (1 - target) * (1.0 - probability)
    bins = np.minimum((probability * 10.0).astype(int), 9)
    ece = 0.0
    total_weight = float(weights.sum())
    for bin_id in range(10):
        mask = bins == bin_id
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        ece += (
            bin_weight
            / total_weight
            * abs(
                float(np.average(target[mask], weights=weights[mask]))
                - float(np.average(probability[mask], weights=weights[mask]))
            )
        )
    calibration_intercept, calibration_slope = _calibration_fit(target, probability, weights)
    top_decile = probability >= decile_boundary
    top_quintile = probability >= quintile_boundary

    def top_precision(mask: np.ndarray[Any, np.dtype[np.bool_]]) -> float:
        return float(np.average(target[mask], weights=weights[mask])) if mask.any() else math.nan

    decile_precision = top_precision(top_decile)
    quintile_precision = top_precision(top_quintile)
    return {
        "model": model,
        "log_loss": float(log_loss(target, clipped, sample_weight=weights, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability, sample_weight=weights)),
        "auc": float(roc_auc_score(target, probability, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(target, probability, sample_weight=weights)
        ),
        "expected_calibration_error": ece,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "base_rate": base_rate,
        "mean_probability_assigned_to_realised_class": float(np.average(assigned, weights=weights)),
        "rows": len(frame),
        "unique_stock_sessions": int(
            (frame["symbol"].astype(str) + "|" + frame["session"].astype(str)).nunique()
        ),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "positive_outcomes": int(target.sum()),
        "top_decile_probability_boundary": decile_boundary,
        "top_decile_precision": decile_precision,
        "top_decile_lift": decile_precision / base_rate,
        "top_quintile_probability_boundary": quintile_boundary,
        "top_quintile_precision": quintile_precision,
        "top_quintile_lift": quintile_precision / base_rate,
    }


def _route_state_metrics(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in ("BROAD_CONFLICT", "LOW_ROUTE_SUPPORT", "OTHER", "NARROWING", "DOMINANT_ROUTE"):
        group = assessment.loc[assessment["route_resolution_state"].eq(state)]
        if group.empty:
            continue
        weights = group["row_weight"]
        rows.append(
            {
                "route_resolution_state": state,
                "rows": len(group),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "months": int(group["year_month"].nunique()),
                "mean_absolute_return": _weighted_mean(group["absolute_log_return_15m"], weights),
                "mean_absolute_return_10m": _weighted_mean(
                    group["absolute_log_return_10m"], weights
                ),
                "mean_absolute_return_30m": _weighted_mean(
                    group["absolute_log_return_30m"], weights
                ),
                "mean_absolute_return_60m": _weighted_mean(
                    group["absolute_log_return_60m"], weights
                ),
                "median_absolute_return": _weighted_median(
                    group["absolute_log_return_15m"], weights
                ),
                "mean_iv_expected_absolute_movement": _weighted_mean(
                    group["iv_expected_absolute_15m"], weights
                ),
                "mean_iv_absolute_residual": _weighted_mean(
                    group["iv_absolute_residual_15m"], weights
                ),
                "median_iv_absolute_residual": _weighted_median(
                    group["iv_absolute_residual_15m"], weights
                ),
                "mean_iv_sigma_ratio": _weighted_mean(group["iv_sigma_ratio_15m"], weights),
                "percentage_exceeding_iv_expected_absolute": _weighted_mean(
                    group["movement_exceeds_iv_expected_absolute"], weights
                ),
                "percentage_exceeding_one_iv_sigma": _weighted_mean(
                    group["movement_exceeds_one_iv_sigma"], weights
                ),
                "mean_realised_range": _weighted_mean(group["realised_range_15m"], weights),
                "mean_maximum_absolute_excursion": _weighted_mean(
                    group["maximum_absolute_excursion_15m"], weights
                ),
                "mean_realised_variance": _weighted_mean(group["realised_variance_15m"], weights),
                "registered_completion_bars_2_or_3_rate": _weighted_mean(
                    group["registered_completion_in_bars_2_or_3"], weights
                ),
                "mean_movement_before_completion": _weighted_mean(
                    group["movement_before_completion"], weights
                ),
                "mean_movement_completion_to_horizon_end": _weighted_mean(
                    group["movement_from_completion_to_horizon_end"], weights
                ),
            }
        )
    output = pd.DataFrame(rows)
    if {"BROAD_CONFLICT", "LOW_ROUTE_SUPPORT"}.issubset(set(output["route_resolution_state"])):
        indexed = output.set_index("route_resolution_state")
        broad = indexed.loc["BROAD_CONFLICT"]
        low = indexed.loc["LOW_ROUTE_SUPPORT"]
        output.loc[len(output)] = {
            "route_resolution_state": "BROAD_CONFLICT_MINUS_LOW_ROUTE_SUPPORT",
            "rows": int(broad["rows"]),
            "sessions": int(broad["sessions"]),
            "stocks": int(broad["stocks"]),
            "months": int(broad["months"]),
            "mean_iv_absolute_residual": float(broad["mean_iv_absolute_residual"])
            - float(low["mean_iv_absolute_residual"]),
            "mean_iv_sigma_ratio": float(broad["mean_iv_sigma_ratio"])
            - float(low["mean_iv_sigma_ratio"]),
            "percentage_exceeding_iv_expected_absolute": float(
                broad["percentage_exceeding_iv_expected_absolute"]
            )
            - float(low["percentage_exceeding_iv_expected_absolute"]),
        }
    return output


def _matched_metrics(
    assessment: pd.DataFrame, relations: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    indexed = assessment.set_index("row_id", drop=False)
    metrics = {
        "absolute_log_return_15m": "absolute return",
        "iv_absolute_residual_15m": "IV absolute residual",
        "iv_sigma_ratio_15m": "IV sigma ratio",
        "movement_exceeds_iv_expected_absolute": "exceed-IV rate",
        "realised_range_15m": "realised range",
        "maximum_absolute_excursion_15m": "maximum absolute excursion",
    }
    rows: list[dict[str, Any]] = []
    effects: dict[str, float] = {}
    eligible_broad_rows = int(assessment["route_resolution_state"].eq("BROAD_CONFLICT").sum())
    matched_treated_rows = int(relations["treated_row_id"].nunique()) if len(relations) else 0
    for column, label in metrics.items():
        treated_effects: list[float] = []
        treated_weights: list[float] = []
        for treated_id, group in relations.groupby("treated_row_id", sort=True):
            treated = indexed.loc[str(treated_id)]
            controls = indexed.loc[group["control_row_id"].astype(str)]
            control_value = float(
                np.average(
                    controls[column].to_numpy(float),
                    weights=group["match_weight"].to_numpy(float),
                )
            )
            treated_effects.append(float(treated[column]) - control_value)
            treated_weights.append(float(treated["row_weight"]))
        effect = (
            float(np.average(treated_effects, weights=treated_weights))
            if treated_effects
            else math.nan
        )
        effects[column] = effect
        rows.append(
            {
                "contrast": "broad_conflict_minus_matched_control",
                "metric": label,
                "value": effect,
                "treated_rows": len(treated_effects),
                "eligible_broad_rows": eligible_broad_rows,
                "matched_treated_rows": matched_treated_rows,
                "treated_coverage": matched_treated_rows / max(eligible_broad_rows, 1),
                "status": (
                    "supported"
                    if matched_treated_rows == eligible_broad_rows and eligible_broad_rows > 0
                    else "insufficient_support"
                ),
            }
        )
    return pd.DataFrame(rows), effects


def _ridge_metrics(frame: pd.DataFrame, *, prediction_column: str, model: str) -> dict[str, Any]:
    target = frame["iv_absolute_residual_15m"].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    weights = frame["row_weight"].to_numpy(float)
    return {
        "model": model,
        "weighted_mae": float(mean_absolute_error(target, prediction, sample_weight=weights)),
        "weighted_rmse": float(
            mean_squared_error(target, prediction, sample_weight=weights) ** 0.5
        ),
        "weighted_r2": float(r2_score(target, prediction, sample_weight=weights)),
        "mean_residual_prediction": float(np.average(prediction, weights=weights)),
    }


def _metrics_long(frame: pd.DataFrame, id_columns: Sequence[str]) -> pd.DataFrame:
    value_columns = [column for column in frame.columns if column not in id_columns]
    return frame.melt(
        id_vars=list(id_columns),
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    )


def _increment_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["movement_exceeds_iv_expected_absolute"].to_numpy(int)
    weights = frame["row_weight"].to_numpy(float)
    o0 = frame["O0_probability"].to_numpy(float)
    o1 = frame["O1_probability"].to_numpy(float)
    return {
        "log_loss_improvement": float(
            log_loss(target, o0, sample_weight=weights, labels=[0, 1])
            - log_loss(target, o1, sample_weight=weights, labels=[0, 1])
        ),
        "brier_improvement": float(
            brier_score_loss(target, o0, sample_weight=weights)
            - brier_score_loss(target, o1, sample_weight=weights)
        ),
        "auc_improvement": float(
            roc_auc_score(target, o1, sample_weight=weights)
            - roc_auc_score(target, o0, sample_weight=weights)
        ),
        "average_precision_improvement": float(
            average_precision_score(target, o1, sample_weight=weights)
            - average_precision_score(target, o0, sample_weight=weights)
        ),
        "top_decile_precision_improvement": _weighted_mean(
            frame.loc[frame["O1_top_decile"].astype(bool), "movement_exceeds_iv_expected_absolute"],
            frame.loc[frame["O1_top_decile"].astype(bool), "row_weight"],
        )
        - _weighted_mean(
            frame.loc[frame["O0_top_decile"].astype(bool), "movement_exceeds_iv_expected_absolute"],
            frame.loc[frame["O0_top_decile"].astype(bool), "row_weight"],
        ),
    }


def _bootstrap_analysis(
    assessment: pd.DataFrame,
    relations: pd.DataFrame,
) -> pd.DataFrame:
    multiplicities = fixed_session_bootstrap_multiplicities(assessment, draws=25, seed=20260722)
    indexed = assessment.set_index("row_id", drop=False)
    draw_rows: list[dict[str, Any]] = []
    statistics: dict[str, list[float]] = {}
    for draw, multiplicity in enumerate(multiplicities):
        sample = assessment.copy()
        sample["row_weight"] = sample["row_weight"].to_numpy(float) * multiplicity
        sample = sample.loc[sample["row_weight"].gt(0.0)]
        increments = _increment_metrics(sample)
        broad = sample.loc[sample["route_resolution_state"].eq("BROAD_CONFLICT")]
        low = sample.loc[sample["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
        values = {
            **increments,
            "broad_conflict_mean_iv_absolute_residual": _weighted_mean(
                broad["iv_absolute_residual_15m"], broad["row_weight"]
            ),
            "broad_minus_low_iv_absolute_residual": _weighted_mean(
                broad["iv_absolute_residual_15m"], broad["row_weight"]
            )
            - _weighted_mean(low["iv_absolute_residual_15m"], low["row_weight"]),
        }
        row_multiplicity = dict(zip(assessment["row_id"].astype(str), multiplicity, strict=True))
        matched_values: dict[str, list[float]] = {
            "iv_absolute_residual_15m": [],
            "movement_exceeds_iv_expected_absolute": [],
            "iv_sigma_ratio_15m": [],
        }
        matched_weights: list[float] = []
        for treated_id, group in relations.groupby("treated_row_id", sort=True):
            treated_multiplier = int(row_multiplicity.get(str(treated_id), 0))
            if treated_multiplier == 0:
                continue
            control_multiplicity = group["control_row_id"].astype(str).map(row_multiplicity)
            control_weights = group["match_weight"].to_numpy(float) * control_multiplicity.to_numpy(
                float
            )
            if control_weights.sum() <= 0.0:
                continue
            treated = indexed.loc[str(treated_id)]
            controls = indexed.loc[group["control_row_id"].astype(str)]
            for column in matched_values:
                control = float(
                    np.average(controls[column].to_numpy(float), weights=control_weights)
                )
                matched_values[column].append(float(treated[column]) - control)
            matched_weights.append(float(treated["row_weight"]) * treated_multiplier)
        for column, metric in (
            ("iv_absolute_residual_15m", "broad_minus_matched_iv_absolute_residual"),
            (
                "movement_exceeds_iv_expected_absolute",
                "broad_minus_matched_exceed_iv_rate",
            ),
            ("iv_sigma_ratio_15m", "broad_minus_matched_iv_sigma_ratio"),
        ):
            values[metric] = (
                float(np.average(matched_values[column], weights=matched_weights))
                if matched_weights
                else math.nan
            )
        for statistic, value in values.items():
            statistics.setdefault(statistic, []).append(value)
            draw_rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": statistic,
                    "value": value,
                    "interval_level": math.nan,
                    "lower": math.nan,
                    "upper": math.nan,
                    "status": "supported",
                }
            )
    for statistic, values in statistics.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        for level, tail in ((0.80, 0.10), (0.90, 0.05), (0.95, 0.025)):
            draw_rows.append(
                {
                    "record_type": "interval",
                    "draw": math.nan,
                    "metric": statistic,
                    "value": math.nan,
                    "interval_level": level,
                    "lower": (
                        float(np.quantile(finite, tail, method="linear"))
                        if len(finite)
                        else math.nan
                    ),
                    "upper": (
                        float(np.quantile(finite, 1.0 - tail, method="linear"))
                        if len(finite)
                        else math.nan
                    ),
                    "status": "supported",
                }
            )
    return pd.DataFrame(draw_rows)


def _bootstrap_lower(frame: pd.DataFrame, metric: str, level: float = 0.80) -> float:
    row = frame.loc[
        frame["record_type"].eq("interval")
        & frame["metric"].eq(metric)
        & frame["interval_level"].eq(level)
    ]
    if len(row) != 1:
        raise RuntimeError(f"bootstrap interval unavailable: {metric}/{level}")
    return float(row.iloc[0]["lower"])


def _derive_decision_gates(
    assessment: pd.DataFrame,
    *,
    state_metrics: pd.DataFrame,
    matched_effects: Mapping[str, float],
    bootstrap: pd.DataFrame,
    route_null: pd.DataFrame,
    coverage_passed: bool,
    matched_support_passed: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Derive the two frozen gates and sole decision from completed predictions."""

    real_increment = _increment_metrics(assessment)
    monthly_increments = [
        _increment_metrics(group) for _month, group in assessment.groupby("year_month", sort=True)
    ]
    checkpoint_increments = [
        _increment_metrics(group)
        for _checkpoint, group in assessment.groupby("checkpoint_group", sort=True)
    ]
    comparison_rows = route_null.loc[route_null["record_type"].eq("comparison")]
    null_counts = dict(
        zip(
            comparison_rows["metric"].astype(str),
            comparison_rows["real_exceeds_null_count"].astype(int),
            strict=True,
        )
    )
    o1_gates = {
        **real_increment,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(bootstrap, "log_loss_improvement"),
        "bootstrap_80_brier_lower": _bootstrap_lower(bootstrap, "brier_improvement"),
        "bootstrap_80_average_precision_lower": _bootstrap_lower(
            bootstrap, "average_precision_improvement"
        ),
        "positive_months": sum(value["log_loss_improvement"] > 0.0 for value in monthly_increments),
        "materially_adverse_checkpoint_groups": sum(
            value["log_loss_improvement"] < MATERIALLY_ADVERSE_LOG_LOSS
            or value["brier_improvement"] < MATERIALLY_ADVERSE_BRIER
            for value in checkpoint_increments
        ),
        "real_exceeds_matching_nulls": max(
            int(null_counts.get("log_loss_improvement", 0)),
            int(null_counts.get("brier_improvement", 0)),
        ),
        "coverage_and_concentration_passed": coverage_passed,
    }
    state_index = state_metrics.set_index("route_resolution_state")
    broad = state_index.loc["BROAD_CONFLICT"]
    low = state_index.loc["LOW_ROUTE_SUPPORT"]
    broad_month_differences = [
        _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("BROAD_CONFLICT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT"), "row_weight"],
        )
        - _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "row_weight"],
        )
        for _month, group in assessment.groupby("year_month", sort=True)
    ]
    broad_checkpoint_differences = [
        _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("BROAD_CONFLICT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT"), "row_weight"],
        )
        - _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "row_weight"],
        )
        for _checkpoint, group in assessment.groupby("checkpoint_group", sort=True)
    ]
    broad_gates = {
        "mean_residual": float(broad["mean_iv_absolute_residual"]),
        "minus_low_route_support_residual": float(broad["mean_iv_absolute_residual"])
        - float(low["mean_iv_absolute_residual"]),
        "minus_matched_residual": _finite_or_zero(matched_effects["iv_absolute_residual_15m"]),
        "minus_matched_exceed_rate": _finite_or_zero(
            matched_effects["movement_exceeds_iv_expected_absolute"]
        ),
        "bootstrap_80_minus_low_residual_lower": _bootstrap_lower(
            bootstrap, "broad_minus_low_iv_absolute_residual"
        ),
        "bootstrap_80_minus_matched_residual_lower": _finite_or_zero(
            _bootstrap_lower(bootstrap, "broad_minus_matched_iv_absolute_residual")
        ),
        "bootstrap_80_minus_matched_exceed_lower": _finite_or_zero(
            _bootstrap_lower(bootstrap, "broad_minus_matched_exceed_iv_rate")
        ),
        "positive_months": sum(value > 0.0 for value in broad_month_differences),
        "materially_adverse_checkpoint_groups": sum(
            value < MATERIALLY_ADVERSE_IV_RESIDUAL for value in broad_checkpoint_differences
        ),
        "support_and_concentration_passed": coverage_passed and matched_support_passed,
    }
    o1_passed = o1_model_gate_passes(o1_gates)
    broad_passed = broad_conflict_iv_gate_passes(broad_gates)
    decision = choose_options_movement_decision(
        blocker=None,
        o1_passed=o1_passed,
        broad_conflict_passed=broad_passed,
        descriptive_only=False,
    )
    return o1_gates, broad_gates, decision


def _model_dict(model: Any) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "kind": model.kind,
        "numeric_features": list(model.numeric_features),
        "numeric_medians": model.numeric_medians.tolist(),
        "numeric_means": model.numeric_means.tolist(),
        "numeric_scales": model.numeric_scales.tolist(),
        "category_levels": {key: list(value) for key, value in model.category_levels.items()},
        "design_columns": list(model.design_columns),
        "coefficients": model.coefficients.tolist(),
        "intercept": model.intercept,
        "iterations": model.iterations,
        "preprocessing_fitted_period": model.preprocessing_fitted_period,
    }


def _write_completed_report(
    output: Path,
    *,
    decision: Mapping[str, Any],
    pooled: pd.DataFrame,
    state_metrics: pd.DataFrame,
    ridge: pd.DataFrame,
    matched: pd.DataFrame,
) -> None:
    """Replace the pre-download blocker report after a supported cached-data run."""

    table_names = (
        "options_data_quality.csv",
        "options_coverage.csv",
        "options_structural_join_audit.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "subgroup_metrics.csv",
        "bootstrap_metrics.csv",
        "route_null_metrics.csv",
        "concentration_metrics.csv",
    )
    complete_tables = {
        name: pd.read_csv(output / name).to_dict(orient="records") for name in table_names
    }
    download_summary = json.loads(
        (output / "options_download_manifest.json").read_text(encoding="utf-8")
    )
    download_summary.pop("manifest_rows", None)
    audit_and_determinism = {
        name: (
            json.loads((output / name).read_text(encoding="utf-8"))
            if (output / name).is_file()
            else {"status": "not_yet_written"}
        )
        for name in ("determinism_check.json", "lightweight_audit.json")
    }
    report = "\n".join(
        [
            "# Prior-Close Options IV Movement Screen V0",
            "",
            f"Primary decision: `{decision['decision']}`",
            "",
            "This retrospective screen uses only exact previous-session EOD options",
            "information and 15-minute underlying movement. It does not simulate an intraday",
            "option fill or calculate",
            "option P&L, establish option profitability, or claim prospective/trading utility.",
            "",
            "## O0/O1 pooled metrics",
            "",
            "```json",
            json.dumps(pooled.to_dict(orient="records"), indent=2, default=str),
            "```",
            "",
            "## Route-state underlying movement",
            "",
            "```json",
            json.dumps(state_metrics.to_dict(orient="records"), indent=2, default=str),
            "```",
            "",
            "## Continuous residual and matched-control diagnostics",
            "",
            "```json",
            json.dumps(
                {
                    "continuous": ridge.to_dict(orient="records"),
                    "matched": matched.to_dict(orient="records"),
                },
                indent=2,
                default=str,
            ),
            "```",
            "",
            "## Download, coverage, stability, resampling and audit artifacts",
            "",
            "```json",
            json.dumps(
                {
                    "download_summary": download_summary,
                    "tables": complete_tables,
                    "audit_and_determinism": audit_and_determinism,
                },
                indent=2,
                default=str,
            ),
            "```",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")


def _stability_tables(
    assessment: pd.DataFrame,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for key, destination in (("year_month", monthly), ("checkpoint", checkpoints)):
        for level, group in assessment.groupby(key, sort=True):
            for model in ("O0", "O1"):
                result = _binary_metrics(
                    group,
                    model=model,
                    probability_column=f"{model}_probability",
                    decile_boundary=float(thresholds[model]["decile"]),
                    quintile_boundary=float(thresholds[model]["quintile"]),
                )
                destination.append({key: level, **result})
            broad = group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT")]
            low = group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
            for state in ("BROAD_CONFLICT", "LOW_ROUTE_SUPPORT", "OTHER", "NARROWING"):
                state_group = group.loc[group["route_resolution_state"].eq(state)]
                if state_group.empty:
                    continue
                weight = state_group["row_weight"]
                destination.append(
                    {
                        key: level,
                        "model": f"STATE_{state}",
                        "rows": len(state_group),
                        "mean_absolute_return": _weighted_mean(
                            state_group["absolute_log_return_15m"], weight
                        ),
                        "mean_iv_absolute_residual": _weighted_mean(
                            state_group["iv_absolute_residual_15m"], weight
                        ),
                        "mean_iv_sigma_ratio": _weighted_mean(
                            state_group["iv_sigma_ratio_15m"], weight
                        ),
                        "exceed_iv_rate": _weighted_mean(
                            state_group["movement_exceeds_iv_expected_absolute"], weight
                        ),
                        "exceed_one_sigma_rate": _weighted_mean(
                            state_group["movement_exceeds_one_iv_sigma"], weight
                        ),
                        "mean_realised_range": _weighted_mean(
                            state_group["realised_range_15m"], weight
                        ),
                        "mean_maximum_absolute_excursion": _weighted_mean(
                            state_group["maximum_absolute_excursion_15m"], weight
                        ),
                        "mean_realised_variance": _weighted_mean(
                            state_group["realised_variance_15m"], weight
                        ),
                    }
                )
            destination.append(
                {
                    key: level,
                    "model": "BROAD_CONFLICT_MINUS_LOW_ROUTE_SUPPORT",
                    "mean_iv_residual_difference": _weighted_mean(
                        broad["iv_absolute_residual_15m"], broad["row_weight"]
                    )
                    - _weighted_mean(low["iv_absolute_residual_15m"], low["row_weight"]),
                    "exceed_iv_rate_difference": _weighted_mean(
                        broad["movement_exceeds_iv_expected_absolute"], broad["row_weight"]
                    )
                    - _weighted_mean(
                        low["movement_exceeds_iv_expected_absolute"], low["row_weight"]
                    ),
                    "iv_sigma_ratio_difference": _weighted_mean(
                        broad["iv_sigma_ratio_15m"], broad["row_weight"]
                    )
                    - _weighted_mean(low["iv_sigma_ratio_15m"], low["row_weight"]),
                    "broad_rows": len(broad),
                    "low_rows": len(low),
                }
            )
    development = assessment.attrs.get("development")
    if not isinstance(development, pd.DataFrame):
        raise RuntimeError("development frame missing from stability context")
    medians = {
        "atm_iv": _weighted_median(development["atm_iv"], development["row_weight"]),
        "transition_probability": _weighted_median(
            development["transition_probability"], development["row_weight"]
        ),
        "posterior_entropy": _weighted_median(
            development["posterior_entropy"], development["row_weight"]
        ),
        "combined_relative_spread": _weighted_median(
            development["combined_relative_spread"], development["row_weight"]
        ),
    }
    subgroup_specs: dict[str, pd.Series] = {
        "checkpoint_era": pd.cut(
            assessment["checkpoint"],
            bins=[5, 14, 24, 34],
            labels=["early_6_14", "middle_16_24", "later_26_34"],
        ).astype(str),
        "atm_iv": np.where(assessment["atm_iv"].le(medians["atm_iv"]), "low", "high"),
        "transition_probability": np.where(
            assessment["transition_probability"].le(medians["transition_probability"]),
            "low",
            "high",
        ),
        "posterior_entropy": np.where(
            assessment["posterior_entropy"].le(medians["posterior_entropy"]),
            "low",
            "high",
        ),
        "front_dte": pd.cut(
            assessment["front_dte"],
            bins=[6.999, 14, 30, 45],
            labels=["7-14", "15-30", "31-45"],
        ).astype(str),
        "option_pair_spread": np.where(
            assessment["combined_relative_spread"].le(medians["combined_relative_spread"]),
            "tight",
            "wide",
        ),
        "registered_completion_bars_2_or_3": np.where(
            assessment["registered_completion_in_bars_2_or_3"].astype(bool),
            "completion",
            "no_completion",
        ),
    }
    subgroup_rows: list[dict[str, Any]] = []
    for subgroup, labels in subgroup_specs.items():
        for level in sorted(set(str(value) for value in labels)):
            mask = pd.Series(labels, index=assessment.index).astype(str).eq(level)
            group = assessment.loc[mask]
            for model in ("O0", "O1"):
                result = _binary_metrics(
                    group,
                    model=model,
                    probability_column=f"{model}_probability",
                    decile_boundary=float(thresholds[model]["decile"]),
                    quintile_boundary=float(thresholds[model]["quintile"]),
                )
                subgroup_rows.append({"subgroup": subgroup, "level": level, **result})
            broad = group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT")]
            low = group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
            for state in ("BROAD_CONFLICT", "LOW_ROUTE_SUPPORT", "OTHER", "NARROWING"):
                state_group = group.loc[group["route_resolution_state"].eq(state)]
                if state_group.empty:
                    continue
                weight = state_group["row_weight"]
                subgroup_rows.append(
                    {
                        "subgroup": subgroup,
                        "level": level,
                        "model": f"STATE_{state}",
                        "rows": len(state_group),
                        "mean_absolute_return": _weighted_mean(
                            state_group["absolute_log_return_15m"], weight
                        ),
                        "mean_iv_absolute_residual": _weighted_mean(
                            state_group["iv_absolute_residual_15m"], weight
                        ),
                        "mean_iv_sigma_ratio": _weighted_mean(
                            state_group["iv_sigma_ratio_15m"], weight
                        ),
                        "exceed_iv_rate": _weighted_mean(
                            state_group["movement_exceeds_iv_expected_absolute"], weight
                        ),
                        "mean_realised_range": _weighted_mean(
                            state_group["realised_range_15m"], weight
                        ),
                        "mean_maximum_absolute_excursion": _weighted_mean(
                            state_group["maximum_absolute_excursion_15m"], weight
                        ),
                        "mean_realised_variance": _weighted_mean(
                            state_group["realised_variance_15m"], weight
                        ),
                    }
                )
            subgroup_rows.append(
                {
                    "subgroup": subgroup,
                    "level": level,
                    "model": "BROAD_CONFLICT_MINUS_LOW_ROUTE_SUPPORT",
                    "mean_iv_residual_difference": _weighted_mean(
                        broad["iv_absolute_residual_15m"], broad["row_weight"]
                    )
                    - _weighted_mean(low["iv_absolute_residual_15m"], low["row_weight"]),
                    "exceed_iv_rate_difference": _weighted_mean(
                        broad["movement_exceeds_iv_expected_absolute"], broad["row_weight"]
                    )
                    - _weighted_mean(
                        low["movement_exceeds_iv_expected_absolute"], low["row_weight"]
                    ),
                    "iv_sigma_ratio_difference": _weighted_mean(
                        broad["iv_sigma_ratio_15m"], broad["row_weight"]
                    )
                    - _weighted_mean(low["iv_sigma_ratio_15m"], low["row_weight"]),
                    "broad_rows": len(broad),
                    "low_rows": len(low),
                }
            )
    return pd.DataFrame(monthly), pd.DataFrame(checkpoints), pd.DataFrame(subgroup_rows)


def _write_blocked_coverage_decision(output: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    decision = {
        **SAFETY_FLAGS,
        "decision": "blocked_insufficient_options_chain_coverage",
        "options_download_status": "supported",
        "options_coverage_status": "blocked",
        "iv_excess_model_status": "blocked",
        "broad_conflict_movement_status": "blocked",
        "matched_control_status": "blocked",
        "coverage_evidence": dict(evidence),
    }
    write_json(output / "decision.json", decision)
    _write_completed_report(
        output,
        decision=decision,
        pooled=pd.DataFrame(),
        state_metrics=pd.DataFrame(),
        ridge=pd.DataFrame(),
        matched=pd.DataFrame(),
    )
    write_json(
        output / "determinism_check.json",
        {
            "status": "blocked_before_model_fit",
            "reason": "blocked_insufficient_options_chain_coverage",
            "bootstrap_repeated": False,
            "route_null_refits_repeated": False,
            "selected_contract_mismatches": None,
            "joined_row_mismatches": None,
            "maximum_option_feature_difference": None,
            "maximum_probability_difference": None,
            "maximum_movement_difference": None,
        },
    )
    return decision


def _maximum_numeric_difference(
    left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str]
) -> float:
    if not columns:
        return 0.0
    first = left.loc[:, list(columns)].to_numpy(float)
    second = right.loc[:, list(columns)].to_numpy(float)
    if first.shape != second.shape:
        return math.inf
    if np.logical_xor(np.isnan(first), np.isnan(second)).any():
        return math.inf
    difference = np.abs(first - second)
    difference[np.isnan(first) & np.isnan(second)] = 0.0
    return float(np.nanmax(difference)) if difference.size else 0.0


def _run_determinism_check(
    *,
    data_dir: Path,
    provider_root: Path,
    output: Path,
    price_audit: pd.DataFrame,
    original_pairs: pd.DataFrame,
    original_movement: pd.DataFrame,
    original_assessment: pd.DataFrame,
    original_models: Mapping[str, Any],
    expected_decision: str,
    expected_chunk_ids: set[str],
) -> dict[str, Any]:
    """Reload cached canonical data and repeat only selection, joins, models, and predictions."""

    canonical = _read_canonical(data_dir, expected_chunk_ids=expected_chunk_ids)
    pairs, _manifest = _select_pairs(canonical, price_audit)
    pair_keys = ["symbol", "signal_date"]
    original_pair_order = original_pairs.sort_values(pair_keys, kind="mergesort").reset_index(
        drop=True
    )
    pair_order = pairs.sort_values(pair_keys, kind="mergesort").reset_index(drop=True)
    contract_columns = ["call_contract_id", "put_contract_id", "selected_expiry", "selected_strike"]
    selected_contract_mismatches = abs(len(original_pair_order) - len(pair_order))
    if selected_contract_mismatches == 0:
        selected_contract_mismatches += int(
            original_pair_order[contract_columns]
            .astype(str)
            .ne(pair_order[contract_columns].astype(str))
            .any(axis=1)
            .sum()
        )
    pair_numeric = [
        column
        for column in original_pair_order.columns
        if column in pair_order
        and pd.api.types.is_numeric_dtype(original_pair_order[column])
        and column not in {"selected_strike"}
    ]
    maximum_option_difference = _maximum_numeric_difference(
        original_pair_order, pair_order, pair_numeric
    )
    clean, _reconstruction = load_clean_advance_panel()
    clean["signal_date"] = pd.to_datetime(clean["session"], errors="raise").dt.strftime("%Y-%m-%d")
    joined = clean.merge(
        pairs,
        on=["symbol", "signal_date"],
        how="inner",
        validate="many_to_one",
    )
    movement = add_iv_relative_outcomes(
        compute_underlying_movement_outcomes(
            joined,
            _load_full_underlying_bars(
                provider_root,
                sessions=set(clean["session"].astype(str)),
            ),
        )
    )
    original_rows = original_movement.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    rebuilt_rows = movement.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    joined_row_mismatches = abs(len(original_rows) - len(rebuilt_rows))
    if joined_row_mismatches == 0:
        joined_row_mismatches += int(
            original_rows["row_id"].astype(str).ne(rebuilt_rows["row_id"].astype(str)).sum()
        )
    movement_columns = [
        "entry_price",
        "absolute_log_return_10m",
        "absolute_log_return_15m",
        "absolute_log_return_30m",
        "absolute_log_return_60m",
        "realised_range_15m",
        "maximum_absolute_excursion_15m",
        "realised_variance_15m",
        "iv_absolute_residual_15m",
        "iv_sigma_ratio_15m",
    ]
    maximum_movement_difference = _maximum_numeric_difference(
        original_rows, rebuilt_rows, movement_columns
    )
    development = movement.loc[movement["period"].eq("development")].copy()
    assessment = movement.loc[movement["period"].eq("assessment")].copy()
    o0_features = (*OPTIONS_PRIMARY_FEATURES, *DENSE_H0_FEATURES)
    o1_features = (*o0_features, *ROUTE_FEATURES)
    rebuilt_models = {
        "O0": fit_options_linear_model(
            development, numeric_features=o0_features, model_id="O0", kind="logistic"
        ),
        "O1": fit_options_linear_model(
            development, numeric_features=o1_features, model_id="O1", kind="logistic"
        ),
        "R0": fit_options_linear_model(
            development, numeric_features=o0_features, model_id="R0", kind="ridge"
        ),
        "R1": fit_options_linear_model(
            development, numeric_features=o1_features, model_id="R1", kind="ridge"
        ),
    }
    coefficient_differences: list[float] = []
    probability_differences: list[float] = []
    original_prediction_order = original_assessment.sort_values(
        "row_id", kind="mergesort"
    ).reset_index(drop=True)
    assessment_order = assessment.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    for model_id, rebuilt_model in rebuilt_models.items():
        original_model = original_models[model_id]
        coefficient_differences.extend(
            np.abs(original_model.coefficients - rebuilt_model.coefficients).tolist()
        )
        coefficient_differences.append(abs(original_model.intercept - rebuilt_model.intercept))
        rebuilt_prediction = rebuilt_model.predict(assessment_order)
        development_prediction = rebuilt_model.predict(development)
        if model_id in {"O0", "O1"}:
            assessment_order[f"{model_id}_probability"] = rebuilt_prediction
            development[f"{model_id}_probability"] = development_prediction
        else:
            assessment_order[f"{model_id}_prediction"] = rebuilt_prediction
            development[f"{model_id}_prediction"] = development_prediction
        probability_differences.extend(
            np.abs(
                original_prediction_order[f"{model_id}_probability"].to_numpy(float)
                - rebuilt_prediction
            ).tolist()
            if model_id in {"O0", "O1"}
            else np.abs(
                original_prediction_order[f"{model_id}_prediction"].to_numpy(float)
                - rebuilt_prediction
            ).tolist()
        )
    maximum_coefficient_difference = max(coefficient_differences, default=0.0)
    maximum_probability_difference = max(probability_differences, default=0.0)
    thresholds = {
        model_id: {
            "decile": _weighted_quantile(
                development[f"{model_id}_probability"], development["row_weight"], 0.90
            ),
            "quintile": _weighted_quantile(
                development[f"{model_id}_probability"], development["row_weight"], 0.80
            ),
        }
        for model_id in ("O0", "O1")
    }
    for model_id in ("O0", "O1"):
        assessment_order[f"{model_id}_top_decile"] = assessment_order[f"{model_id}_probability"].ge(
            thresholds[model_id]["decile"]
        )
        assessment_order[f"{model_id}_top_quintile"] = assessment_order[
            f"{model_id}_probability"
        ].ge(thresholds[model_id]["quintile"])
    rebuilt_pooled = pd.DataFrame(
        [
            _binary_metrics(
                assessment_order,
                model=model_id,
                probability_column=f"{model_id}_probability",
                decile_boundary=thresholds[model_id]["decile"],
                quintile_boundary=thresholds[model_id]["quintile"],
            )
            for model_id in ("O0", "O1")
        ]
    )
    rebuilt_pooled_long = _metrics_long(rebuilt_pooled, ["model"]).sort_values(
        ["model", "metric"], kind="mergesort"
    )
    stored_pooled_long = pd.read_csv(output / "pooled_metrics.csv").sort_values(
        ["model", "metric"], kind="mergesort"
    )
    if (
        rebuilt_pooled_long[["model", "metric"]]
        .reset_index(drop=True)
        .equals(stored_pooled_long[["model", "metric"]].reset_index(drop=True))
    ):
        pooled_values = rebuilt_pooled_long["value"].to_numpy(float)
        stored_values = stored_pooled_long["value"].to_numpy(float)
        if np.logical_xor(np.isnan(pooled_values), np.isnan(stored_values)).any():
            maximum_pooled_metric_difference = math.inf
        else:
            pooled_difference = np.abs(pooled_values - stored_values)
            pooled_difference[np.isnan(pooled_values) & np.isnan(stored_values)] = 0.0
            maximum_pooled_metric_difference = float(np.nanmax(pooled_difference))
    else:
        maximum_pooled_metric_difference = math.inf
    assessment_order["atm_iv_decile"], _ = assign_development_frozen_iv_deciles(
        development["atm_iv"], assessment_order["atm_iv"]
    )
    rebuilt_state_metrics = _route_state_metrics(assessment_order)
    rebuilt_relations = build_matched_control_relations(assessment_order)
    _rebuilt_matched_frame, rebuilt_matched_effects = _matched_metrics(
        assessment_order, rebuilt_relations
    )
    rebuilt_matched_support = int(rebuilt_relations["treated_row_id"].nunique()) == int(
        assessment_order["route_resolution_state"].eq("BROAD_CONFLICT").sum()
    )
    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    route_null = pd.read_csv(output / "route_null_metrics.csv")
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    coverage_evidence, _coverage_frame, _concentration = _coverage(clean, pairs, mapping)
    _rebuilt_o1_gates, _rebuilt_broad_gates, rebuilt_decision = _derive_decision_gates(
        assessment_order,
        state_metrics=rebuilt_state_metrics,
        matched_effects=rebuilt_matched_effects,
        bootstrap=bootstrap,
        route_null=route_null,
        coverage_passed=bool(coverage_evidence["passed"]),
        matched_support_passed=rebuilt_matched_support,
    )
    pooled_metrics_match = maximum_pooled_metric_difference <= 1e-12
    final_decision_match = rebuilt_decision == expected_decision
    passed = bool(
        selected_contract_mismatches == 0
        and joined_row_mismatches == 0
        and maximum_option_difference <= 1e-12
        and maximum_coefficient_difference <= 1e-12
        and maximum_probability_difference <= 1e-12
        and maximum_movement_difference <= 1e-12
        and pooled_metrics_match
        and final_decision_match
    )
    return {
        "status": "supported" if passed else "blocked_reproducibility_or_audit_failure",
        "passed": passed,
        "canonical_cache_reloaded": True,
        "download_repeated": False,
        "bootstrap_repeated": False,
        "route_null_refits_repeated": False,
        "selected_contract_mismatches": selected_contract_mismatches,
        "joined_row_mismatches": joined_row_mismatches,
        "maximum_option_feature_difference": maximum_option_difference,
        "maximum_coefficient_difference": maximum_coefficient_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_movement_difference": maximum_movement_difference,
        "maximum_pooled_metric_difference": maximum_pooled_metric_difference,
        "pooled_metrics_match": pooled_metrics_match,
        "expected_final_decision": expected_decision,
        "rebuilt_final_decision": rebuilt_decision,
        "final_decision_match": final_decision_match,
        "output_path": str(output),
    }


def build_and_analyse_cached_options(*, provider_root: Path, output: Path) -> dict[str, Any]:
    """Build selected pairs, causal movement panel, fixed models, resampling, and decision."""

    data_dir = options_data_dir()
    download_manifest = json.loads(
        (output / "options_download_manifest.json").read_text(encoding="utf-8")
    )
    if download_manifest.get("status") != "supported" or not download_manifest.get(
        "pagination_complete"
    ):
        raise RuntimeError("blocked_options_download_incomplete")
    request_plan = json.loads((output / "options_request_plan.json").read_text(encoding="utf-8"))
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    mapped_mask = mapping["coverage_available"].astype(str).str.casefold().eq("true")
    mapped_symbols = set(mapping.loc[mapped_mask, "stocker_symbol"].astype(str))
    expected_chunk_ids = {
        str(chunk["chunk_id"])
        for chunk in cast(list[dict[str, Any]], request_plan["chunks"])
        if str(chunk["underlying_symbol"]) in mapped_symbols
    }
    canonical = _read_canonical(data_dir, expected_chunk_ids=expected_chunk_ids)
    quality = _quality_summary(canonical, download_manifest)
    price_audit = pd.read_csv(output / "option_underlying_price_audit.csv")
    price_audit = _provider_moneyness_audit(canonical, price_audit)
    write_csv(output / "option_underlying_price_audit.csv", price_audit)
    pairs, pair_manifest = _select_pairs(canonical, price_audit)
    write_json(
        output / "option_pair_selection_manifest.json",
        {"status": "supported", "pairs": pair_manifest},
    )
    write_parquet(output / "selected_option_pairs.parquet", pairs)
    for column in (
        "front_dte",
        "atm_iv",
        "combined_relative_spread",
        "combined_open_interest",
    ):
        values = (
            pd.to_numeric(pairs[column], errors="coerce")
            if column in pairs
            else pd.Series(dtype=float)
        )
        for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
            quality.loc[len(quality)] = {
                "metric": f"selected_pair_{column}_q{int(quantile * 100):02d}",
                "value": float(values.quantile(quantile)) if len(values) else math.nan,
                "status": "observed",
            }
    write_csv(output / "options_data_quality.csv", quality)
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    for symbol in FROZEN_COHORT:
        observed = canonical.loc[canonical["underlying_symbol"].eq(symbol)]
        mask = mapping["stocker_symbol"].eq(symbol)
        if observed.empty:
            continue
        mapping.loc[mask, "earliest_option_date"] = min(observed["trade_date"]).isoformat()
        mapping.loc[mask, "latest_option_date"] = max(observed["trade_date"]).isoformat()
        mapping.loc[mask, "records_returned"] = len(observed)
    write_csv(output / "underlying_symbol_mapping.csv", mapping)
    clean, _reconstruction = load_clean_advance_panel()
    evidence, coverage, concentration = _coverage(clean, pairs, mapping)
    write_csv(output / "options_coverage.csv", coverage)
    write_csv(output / "concentration_metrics.csv", concentration)
    if not evidence["passed"]:
        decision = _write_blocked_coverage_decision(output, evidence)
        from audit_screen_v0 import run_audit

        try:
            audit = run_audit(output)
        except AssertionError:
            decision.update(
                {
                    "decision": "blocked_reproducibility_or_audit_failure",
                    "iv_excess_model_status": "blocked",
                    "broad_conflict_movement_status": "blocked",
                    "matched_control_status": "blocked",
                    "independent_audit_passed": False,
                }
            )
        else:
            decision["independent_audit_passed"] = bool(audit["passed"])
        write_json(output / "decision.json", decision)
        return decision
    clean["signal_date"] = pd.to_datetime(clean["session"], errors="raise").dt.strftime("%Y-%m-%d")
    joined = clean.merge(
        pairs,
        on=["symbol", "signal_date"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise RuntimeError("blocked_option_pair_selection_failure")
    if (
        pd.to_datetime(joined["required_options_date"]).dt.date
        >= pd.to_datetime(joined["session"]).dt.date
    ).any():
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    bars = _load_full_underlying_bars(
        provider_root,
        sessions=set(clean["session"].astype(str)),
    )
    movement = compute_underlying_movement_outcomes(joined, bars)
    movement = add_iv_relative_outcomes(movement)
    write_parquet(output / "options_movement_panel.parquet", movement)
    session_status = pd.DataFrame(pair_manifest)[
        ["symbol", "signal_date", "exact_chain_available", "available", "reason"]
    ]
    clean_status = clean.copy()
    clean_status["signal_date"] = pd.to_datetime(
        clean_status["session"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    clean_status = clean_status.merge(
        session_status,
        on=["symbol", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    audit_rows: list[dict[str, Any]] = []
    for key, group in [
        ("all", clean_status),
        *list(clean_status.groupby("period", sort=True)),
    ]:
        valid_rows = int(group["available"].astype(bool).sum())
        audit_rows.append(
            {
                "scope": "pooled_or_period",
                "key": key,
                "rows": len(group),
                "exact_chain_rows": int(group["exact_chain_available"].astype(bool).sum()),
                "valid_pair_rows": valid_rows,
                "coverage": valid_rows / len(group),
                "status": "supported",
            }
        )
    for reason, group in clean_status.loc[~clean_status["available"].astype(bool)].groupby(
        "reason", sort=True
    ):
        audit_rows.append(
            {
                "scope": "missing_reason",
                "key": reason,
                "rows": len(group),
                "exact_chain_rows": int(group["exact_chain_available"].astype(bool).sum()),
                "valid_pair_rows": 0,
                "coverage": 0.0,
                "status": "not_supported",
            }
        )
    for scope, column in (
        ("stock", "symbol"),
        ("month", "year_month"),
        ("checkpoint", "checkpoint"),
        ("route_state", "route_resolution_state"),
    ):
        for key, group in clean.groupby(column, sort=True):
            joined_rows = int(movement[column].eq(key).sum())
            status_group = clean_status.loc[clean_status[column].eq(key)]
            audit_rows.append(
                {
                    "scope": scope,
                    "key": key,
                    "rows": len(group),
                    "exact_chain_rows": int(
                        status_group["exact_chain_available"].astype(bool).sum()
                    ),
                    "valid_pair_rows": joined_rows,
                    "coverage": joined_rows / len(group),
                    "status": "supported",
                }
            )
    write_csv(output / "options_structural_join_audit.csv", pd.DataFrame(audit_rows))
    development = movement.loc[movement["period"].eq("development")].copy()
    assessment = movement.loc[movement["period"].eq("assessment")].copy()
    assessment["atm_iv_decile"], iv_decile_edges = assign_development_frozen_iv_deciles(
        development["atm_iv"], assessment["atm_iv"]
    )
    development["atm_iv_decile"], _ = assign_development_frozen_iv_deciles(
        development["atm_iv"], development["atm_iv"]
    )
    o0_features = (*OPTIONS_PRIMARY_FEATURES, *DENSE_H0_FEATURES)
    o1_features = (*o0_features, *ROUTE_FEATURES)
    o0 = fit_options_linear_model(
        development,
        numeric_features=o0_features,
        model_id="O0",
        kind="logistic",
    )
    o1 = fit_options_linear_model(
        development,
        numeric_features=o1_features,
        model_id="O1",
        kind="logistic",
    )
    r0 = fit_options_linear_model(
        development,
        numeric_features=o0_features,
        model_id="R0",
        kind="ridge",
    )
    r1 = fit_options_linear_model(
        development,
        numeric_features=o1_features,
        model_id="R1",
        kind="ridge",
    )
    for frame in (development, assessment):
        frame["O0_probability"] = o0.predict(frame)
        frame["O1_probability"] = o1.predict(frame)
        frame["R0_prediction"] = r0.predict(frame)
        frame["R1_prediction"] = r1.predict(frame)
    thresholds = {
        "O0": {
            "decile": _weighted_quantile(
                development["O0_probability"], development["row_weight"], 0.90
            ),
            "quintile": _weighted_quantile(
                development["O0_probability"], development["row_weight"], 0.80
            ),
        },
        "O1": {
            "decile": _weighted_quantile(
                development["O1_probability"], development["row_weight"], 0.90
            ),
            "quintile": _weighted_quantile(
                development["O1_probability"], development["row_weight"], 0.80
            ),
        },
    }
    for model in ("O0", "O1"):
        assessment[f"{model}_top_decile"] = assessment[f"{model}_probability"].ge(
            thresholds[model]["decile"]
        )
        assessment[f"{model}_top_quintile"] = assessment[f"{model}_probability"].ge(
            thresholds[model]["quintile"]
        )
    write_parquet(output / "assessment_predictions.parquet", assessment)
    pooled = pd.DataFrame(
        [
            _binary_metrics(
                assessment,
                model=model,
                probability_column=f"{model}_probability",
                decile_boundary=thresholds[model]["decile"],
                quintile_boundary=thresholds[model]["quintile"],
            )
            for model in ("O0", "O1")
        ]
    )
    write_csv(output / "pooled_metrics.csv", _metrics_long(pooled, ["model"]))
    state_metrics = _route_state_metrics(assessment)
    write_csv(output / "route_state_movement_metrics.csv", state_metrics)
    relations = build_matched_control_relations(assessment)
    matched_support_passed = int(relations["treated_row_id"].nunique()) == int(
        assessment["route_resolution_state"].eq("BROAD_CONFLICT").sum()
    )
    matched_frame, matched_effects = _matched_metrics(assessment, relations)
    write_csv(output / "matched_control_metrics.csv", matched_frame)
    ridge = pd.DataFrame(
        [
            _ridge_metrics(assessment, prediction_column="R0_prediction", model="R0"),
            _ridge_metrics(assessment, prediction_column="R1_prediction", model="R1"),
        ]
    )
    ridge.loc[len(ridge)] = {
        "model": "R1_MINUS_R0",
        "weighted_mae": float(ridge.loc[0, "weighted_mae"] - ridge.loc[1, "weighted_mae"]),
        "weighted_rmse": float(ridge.loc[0, "weighted_rmse"] - ridge.loc[1, "weighted_rmse"]),
        "weighted_r2": math.nan,
        "mean_residual_prediction": math.nan,
    }
    write_csv(output / "continuous_residual_metrics.csv", _metrics_long(ridge, ["model"]))
    assessment.attrs["development"] = development
    monthly, checkpoints, subgroups = _stability_tables(assessment, thresholds=thresholds)
    write_csv(output / "monthly_metrics.csv", monthly)
    write_csv(output / "checkpoint_metrics.csv", checkpoints)
    write_csv(output / "subgroup_metrics.csv", subgroups)
    bootstrap = _bootstrap_analysis(assessment, relations)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    combined = pd.concat([development, assessment], ignore_index=True)
    real_increment = _increment_metrics(assessment)
    null_rows: list[dict[str, Any]] = []
    null_models: dict[str, Any] = {}
    for draw in range(5):
        permuted = permute_intact_route_bundle(combined, seed=20260722 + draw)
        null_development = permuted.loc[permuted["period"].eq("development")]
        null_assessment = permuted.loc[permuted["period"].eq("assessment")].copy()
        null_model = fit_options_linear_model(
            null_development,
            numeric_features=o1_features,
            model_id=f"O1_ROUTE_NULL_{draw}",
            kind="logistic",
        )
        null_assessment["O1_probability"] = null_model.predict(null_assessment)
        null_assessment["O0_probability"] = assessment["O0_probability"].to_numpy(float)
        null_assessment["O0_top_decile"] = assessment["O0_top_decile"].to_numpy(bool)
        null_assessment["O1_top_decile"] = null_assessment["O1_probability"].ge(
            _weighted_quantile(
                pd.Series(null_model.predict(null_development)),
                null_development["row_weight"],
                0.90,
            )
        )
        increments = _increment_metrics(null_assessment)
        for metric in (
            "log_loss_improvement",
            "brier_improvement",
            "auc_improvement",
            "average_precision_improvement",
        ):
            null_rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "real_increment": real_increment[metric],
                    "null_increment": increments[metric],
                    "real_exceeds_null": real_increment[metric] > increments[metric],
                    "status": "supported",
                }
            )
        null_models[str(draw)] = _model_dict(null_model)
    null_frame = pd.DataFrame(null_rows)
    comparisons = (
        null_frame.groupby("metric", sort=True)["real_exceeds_null"]
        .sum()
        .astype(int)
        .reset_index(name="real_exceeds_null_count")
    )
    for row in comparisons.itertuples(index=False):
        null_rows.append(
            {
                "record_type": "comparison",
                "draw": math.nan,
                "metric": row.metric,
                "real_increment": real_increment[str(row.metric)],
                "null_increment": math.nan,
                "real_exceeds_null": math.nan,
                "real_exceeds_null_count": int(row.real_exceeds_null_count),
                "status": "supported",
            }
        )
    route_null = pd.DataFrame(null_rows)
    write_csv(output / "route_null_metrics.csv", route_null)
    write_json(
        output / "model_coefficients.json",
        {
            "status": "supported",
            "primary_models": {
                "O0": _model_dict(o0),
                "O1": _model_dict(o1),
                "R0": _model_dict(r0),
                "R1": _model_dict(r1),
            },
            "route_null_models": null_models,
            "atm_iv_decile_edges": list(iv_decile_edges),
            "development_prediction_thresholds": thresholds,
        },
    )
    monthly_increments = [
        _increment_metrics(group) for _month, group in assessment.groupby("year_month", sort=True)
    ]
    checkpoint_increments = [
        _increment_metrics(group)
        for _checkpoint, group in assessment.groupby("checkpoint_group", sort=True)
    ]
    null_counts = dict(
        zip(comparisons["metric"].astype(str), comparisons["real_exceeds_null_count"], strict=True)
    )
    o1_gates = {
        **real_increment,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(bootstrap, "log_loss_improvement"),
        "bootstrap_80_brier_lower": _bootstrap_lower(bootstrap, "brier_improvement"),
        "bootstrap_80_average_precision_lower": _bootstrap_lower(
            bootstrap, "average_precision_improvement"
        ),
        "positive_months": sum(value["log_loss_improvement"] > 0.0 for value in monthly_increments),
        "materially_adverse_checkpoint_groups": sum(
            value["log_loss_improvement"] < MATERIALLY_ADVERSE_LOG_LOSS
            or value["brier_improvement"] < MATERIALLY_ADVERSE_BRIER
            for value in checkpoint_increments
        ),
        "real_exceeds_matching_nulls": max(
            int(null_counts.get("log_loss_improvement", 0)),
            int(null_counts.get("brier_improvement", 0)),
        ),
        "coverage_and_concentration_passed": bool(evidence["passed"]),
    }
    state_index = state_metrics.set_index("route_resolution_state")
    broad = state_index.loc["BROAD_CONFLICT"]
    low = state_index.loc["LOW_ROUTE_SUPPORT"]
    broad_month_differences = [
        _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("BROAD_CONFLICT"), "iv_absolute_residual_15m"
            ],
            group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT"), "row_weight"],
        )
        - _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "iv_absolute_residual_15m"
            ],
            group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "row_weight"],
        )
        for _month, group in assessment.groupby("year_month", sort=True)
    ]
    broad_checkpoint_differences = [
        _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("BROAD_CONFLICT"), "iv_absolute_residual_15m"
            ],
            group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT"), "row_weight"],
        )
        - _weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "iv_absolute_residual_15m"
            ],
            group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "row_weight"],
        )
        for _checkpoint, group in assessment.groupby("checkpoint_group", sort=True)
    ]
    broad_gates = {
        "mean_residual": float(broad["mean_iv_absolute_residual"]),
        "minus_low_route_support_residual": float(broad["mean_iv_absolute_residual"])
        - float(low["mean_iv_absolute_residual"]),
        "minus_matched_residual": _finite_or_zero(matched_effects["iv_absolute_residual_15m"]),
        "minus_matched_exceed_rate": _finite_or_zero(
            matched_effects["movement_exceeds_iv_expected_absolute"]
        ),
        "bootstrap_80_minus_low_residual_lower": _bootstrap_lower(
            bootstrap, "broad_minus_low_iv_absolute_residual"
        ),
        "bootstrap_80_minus_matched_residual_lower": _finite_or_zero(
            _bootstrap_lower(bootstrap, "broad_minus_matched_iv_absolute_residual")
        ),
        "bootstrap_80_minus_matched_exceed_lower": _finite_or_zero(
            _bootstrap_lower(bootstrap, "broad_minus_matched_exceed_iv_rate")
        ),
        "positive_months": sum(value > 0.0 for value in broad_month_differences),
        "materially_adverse_checkpoint_groups": sum(
            value < MATERIALLY_ADVERSE_IV_RESIDUAL for value in broad_checkpoint_differences
        ),
        "support_and_concentration_passed": bool(evidence["passed"]) and matched_support_passed,
    }
    o1_passed = o1_model_gate_passes(o1_gates)
    broad_passed = broad_conflict_iv_gate_passes(broad_gates)
    decision_value = choose_options_movement_decision(
        blocker=None,
        o1_passed=o1_passed,
        broad_conflict_passed=broad_passed,
        descriptive_only=False,
    )
    decision = {
        **SAFETY_FLAGS,
        "decision": decision_value,
        "options_download_status": "supported",
        "options_coverage_status": "supported",
        "iv_excess_model_status": "supported" if o1_passed else "not_supported",
        "broad_conflict_movement_status": "supported" if broad_passed else "not_supported",
        "matched_control_status": (
            "supported" if matched_support_passed else "insufficient_support"
        ),
        "coverage_evidence": evidence,
        "O1_gate": {**o1_gates, "passed": o1_passed},
        "BROAD_CONFLICT_gate": {**broad_gates, "passed": broad_passed},
        "bootstrap_draws_executed": 25,
        "route_null_refits_executed": 5,
    }
    write_json(output / "decision.json", decision)
    determinism = _run_determinism_check(
        data_dir=data_dir,
        provider_root=provider_root,
        output=output,
        price_audit=price_audit,
        original_pairs=pairs,
        original_movement=movement,
        original_assessment=assessment,
        original_models={"O0": o0, "O1": o1, "R0": r0, "R1": r1},
        expected_decision=decision_value,
        expected_chunk_ids=expected_chunk_ids,
    )
    write_json(output / "determinism_check.json", determinism)
    if not determinism["passed"]:
        decision.update(
            {
                "decision": "blocked_reproducibility_or_audit_failure",
                "iv_excess_model_status": "blocked",
                "broad_conflict_movement_status": "blocked",
                "matched_control_status": "blocked",
                "determinism_passed": False,
            }
        )
        write_json(output / "decision.json", decision)
        write_json(
            output / "lightweight_audit.json",
            {
                "passed": False,
                "status": "blocked_reproducibility_or_audit_failure",
                "audit_scope": "independent_audit_not_run_after_determinism_failure",
                "determinism_passed": False,
            },
        )
        _write_completed_report(
            output,
            decision=decision,
            pooled=pooled,
            state_metrics=state_metrics,
            ridge=ridge,
            matched=matched_frame,
        )
        return decision
    from audit_screen_v0 import run_audit

    try:
        audit = run_audit(output)
    except AssertionError:
        audit = {"passed": False}
    decision["determinism_passed"] = True
    decision["independent_audit_passed"] = bool(audit["passed"])
    if not audit["passed"]:
        decision.update(
            {
                "decision": "blocked_reproducibility_or_audit_failure",
                "iv_excess_model_status": "blocked",
                "broad_conflict_movement_status": "blocked",
                "matched_control_status": "blocked",
            }
        )
    write_json(output / "decision.json", decision)
    _write_completed_report(
        output,
        decision=decision,
        pooled=pooled,
        state_metrics=state_metrics,
        ridge=ridge,
        matched=matched_frame,
    )
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-root", type=Path, default=default_provider_root())
    parser.add_argument("--output", type=Path, default=PRIMARY)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = arguments.output.expanduser().resolve()
    try:
        decision = build_and_analyse_cached_options(
            provider_root=arguments.provider_root.expanduser().resolve(),
            output=output,
        )
    except Exception as error:
        decision = write_analysis_blocker(output, classify_analysis_blocker(error))
        print(decision["decision"])
        return 1
    print(decision.get("decision", decision.get("status", "unknown")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
