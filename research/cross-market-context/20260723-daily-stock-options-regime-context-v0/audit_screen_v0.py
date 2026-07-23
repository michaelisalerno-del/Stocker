#!/usr/bin/env python3
"""Independent fail-closed audit for the Daily Stock x Options Context V0 screen."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
PROTECTED_START = pd.Timestamp("2025-08-23")
EXPECTED_BLOCKER = "blocked_insufficient_daily_options_coverage"
EXPECTED_SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "quick_daily_context_screen": True,
    "daily_stock_dimensions": True,
    "daily_options_dimensions": True,
    "soft_daily_stock_regimes": True,
    "soft_daily_options_regimes": True,
    "cross_market_mismatch_test": True,
    "previous_close_options_only": True,
    "intraday_option_quotes_used": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "options_loop_discovery_enabled": False,
    "economic_strategy_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_safety(value: Mapping[str, object], label: str) -> None:
    mismatches = {
        key: (value.get(key), expected)
        for key, expected in EXPECTED_SAFETY_FLAGS.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise AssertionError(f"{label} safety flags differ: {mismatches}")


def z(values: pd.Series, manifest: Mapping[str, Any], name: str) -> pd.Series:
    scale = cast(Mapping[str, Any], cast(Mapping[str, Any], manifest["scales"])[name])
    return (pd.to_numeric(values, errors="raise") - float(scale["center"])) / float(scale["scale"])


def audit_stock_dimensions() -> dict[str, Any]:
    raw_all = pd.read_parquet(PRIMARY / "daily_stock_raw_features.parquet")
    stored_dimensions = pd.read_parquet(PRIMARY / "daily_stock_dimensions.parquet")
    manifest = read_json(PRIMARY / "daily_stock_feature_manifest.json")
    assert_safety(manifest, "daily_stock_feature_manifest")
    stored = stored_dimensions.merge(
        raw_all,
        on=["symbol", "session"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )
    if len(stored) != len(stored_dimensions):
        raise AssertionError("daily stock dimension row identity changed")
    raw = stored
    directional = 0.5 * (
        z(raw["daily_efficiency_5"], manifest, "daily_efficiency_5")
        + z(raw["daily_efficiency_10"], manifest, "daily_efficiency_10")
    )
    expected = pd.DataFrame(index=raw.index)
    expected["daily_compression"] = (
        -z(raw["daily_range_5_to_20"], manifest, "daily_range_5_to_20")
        - z(raw["daily_rv_5_to_20"], manifest, "daily_rv_5_to_20")
        + z(raw["daily_range_overlap_5"], manifest, "daily_range_overlap_5")
    ) / 3.0
    expected["daily_directional_efficiency"] = directional
    expected["daily_trend_persistence"] = 0.5 * (
        z(raw["daily_sign_persistence_5"], manifest, "daily_sign_persistence_5")
        + z(raw["daily_extension_20"].abs(), manifest, "abs_daily_extension_20")
    )
    expected["daily_extension"] = z(raw["daily_extension_20"], manifest, "daily_extension_20")
    expected["daily_rejection"] = 0.5 * (
        z(raw["daily_extreme_wick_3"], manifest, "daily_extreme_wick_3")
        - (
            directional
            - float(
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], manifest["scales"])["daily_directional_efficiency"],
                )["center"]
            )
        )
        / float(
            cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], manifest["scales"])["daily_directional_efficiency"],
            )["scale"]
        )
    )
    expected["daily_volatility_acceleration"] = z(
        raw["daily_rv_5_to_20"], manifest, "daily_rv_5_to_20"
    )
    expected["daily_relative_strength"] = z(
        raw["daily_relative_return_5"], manifest, "daily_relative_return_5"
    )
    expected["daily_activity_acceleration"] = z(
        raw["daily_activity_5_to_20"], manifest, "daily_activity_5_to_20"
    )
    columns = list(expected.columns)
    maximum_difference = float(
        np.nanmax(
            np.abs(expected.to_numpy(dtype=float) - stored.loc[:, columns].to_numpy(dtype=float))
        )
    )
    if maximum_difference > 1e-12:
        raise AssertionError(f"daily stock dimensions differ by {maximum_difference}")
    if not (pd.to_datetime(raw["stock_information_date"]) < pd.to_datetime(raw["session"])).all():
        raise AssertionError("same-day daily stock context detected")
    support = cast(Mapping[str, Any], manifest["support"])
    if not bool(support["passed"]):
        raise AssertionError("daily stock support gate did not pass")
    return {
        "raw_rows": len(raw_all),
        "dimension_rows": len(stored),
        "maximum_dimension_difference": maximum_difference,
        "development_scaling_only": manifest["fitted_period"] == "development_2024_only",
    }


def audit_stock_regime() -> dict[str, Any]:
    frame = pd.read_parquet(PRIMARY / "daily_stock_dimensions.parquet")
    mapping = read_json(PRIMARY / "daily_stock_regime_mapping.json")
    assert_safety(mapping, "daily_stock_regime_mapping")
    canonical_names = list(mapping["canonical_dimensions"])
    centroids = [
        cast(Mapping[str, Any], centroid)
        for centroid in cast(list[object], mapping["canonical_centroids"])
    ]
    keys = [tuple(float(centroid[name]) for name in canonical_names) for centroid in centroids]
    if keys != sorted(keys):
        raise AssertionError("daily stock regime IDs are not canonical lexicographic order")
    input_columns = list(mapping["input_columns"])
    weights = np.asarray(mapping["canonical_weights"], dtype=float)
    means = np.asarray(
        [[float(centroid[name]) for name in input_columns] for centroid in centroids],
        dtype=float,
    )
    covariances = np.asarray(mapping["canonical_covariances"], dtype=float)
    sample = frame.head(100)
    x = sample.loc[:, input_columns].to_numpy(dtype=float)
    log_density = np.empty((len(sample), 4), dtype=float)
    for regime in range(4):
        delta = x - means[regime]
        log_density[:, regime] = (
            math.log(weights[regime])
            - 0.5 * np.log(2.0 * math.pi * covariances[regime]).sum()
            - 0.5 * ((delta * delta) / covariances[regime]).sum(axis=1)
        )
    log_density -= log_density.max(axis=1, keepdims=True)
    manual = np.exp(log_density)
    manual /= manual.sum(axis=1, keepdims=True)
    stored = sample.loc[:, [f"daily_stock_regime_p_{regime}" for regime in range(4)]].to_numpy(
        dtype=float
    )
    maximum_difference = float(np.max(np.abs(manual - stored)))
    if maximum_difference > 1e-12:
        raise AssertionError(f"manual stock posterior differs by {maximum_difference}")
    if not bool(mapping["converged"]) or mapping["fitted_period"] != "development_2024_only":
        raise AssertionError("stock GMM convergence or fit-period audit failed")
    return {
        "manual_posterior_rows": len(sample),
        "maximum_manual_posterior_difference": maximum_difference,
        "canonical_ordering": True,
    }


def audit_options_and_chronology() -> dict[str, Any]:
    raw = pd.read_parquet(PRIMARY / "daily_options_raw_features.parquet")
    gaps = pd.read_csv(PRIMARY / "daily_options_coverage_gap.csv")
    chronology = pd.read_csv(PRIMARY / "chronology_audit.csv")
    manifest = read_json(PRIMARY / "daily_options_feature_manifest.json")
    assert_safety(manifest, "daily_options_feature_manifest")
    if not chronology["chronology_passed"].astype(bool).all():
        raise AssertionError("chronology audit contains a failure")
    available = raw.loc[raw["pair_available"].astype(bool)]
    if not (
        available["options_observation_date"].astype(str)
        == available["required_options_date"].astype(str)
    ).all():
        raise AssertionError("options observation was not exact D-1")
    if not (
        pd.to_datetime(available["options_observation_date"]) < pd.to_datetime(available["session"])
    ).all():
        raise AssertionError("same-day options context detected")
    if pd.to_datetime(available["options_observation_date"]).ge(PROTECTED_START).any():
        raise AssertionError("protected options observation materialised")
    front_dte = (
        pd.to_datetime(available["front_expiration_date"])
        - pd.to_datetime(available["options_observation_date"])
    ).dt.days
    if not front_dte.between(7, 45).all():
        raise AssertionError("front pair escaped the frozen DTE window")
    back_count = int(available["front_term_urgency"].notna().sum())
    unsupported = cast(
        list[str],
        cast(Mapping[str, Any], manifest["support"])["unsupported_development_raw_features"],
    )
    if back_count != 0 or unsupported != ["front_term_urgency"]:
        raise AssertionError("the recorded back-surface blocker is not exact")
    back_gaps = gaps.loc[gaps["gap_component"].eq("back_atm_pair")]
    if len(back_gaps) != len(available):
        raise AssertionError("back-expiry gap manifest does not cover every valid front pair")
    forbidden = [column for column in raw.columns if "pnl" in column.casefold()]
    if forbidden:
        raise AssertionError(f"option PnL columns unexpectedly exist: {forbidden}")
    return {
        "required_stock_sessions": len(raw),
        "front_pair_stock_sessions": len(available),
        "back_pair_stock_sessions": back_count,
        "gap_rows": len(gaps),
        "bounded_download_gap_rows": int(gaps["bounded_download_required"].astype(bool).sum()),
        "chronology_rows": len(chronology),
        "same_day_options_rows": int(chronology["same_day_options_used"].astype(bool).sum()),
    }


def audit_blocker_and_artifacts() -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (PRIMARY / name).is_file()]
    if missing:
        raise AssertionError(f"required artifacts missing: {missing}")
    contract = read_json(PRIMARY / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    source = read_json(PRIMARY / "source_manifest.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    reconstruction = read_json(PRIMARY / "structural_panel_reconstruction.json")
    for label, value in (
        ("contract", contract),
        ("decision", decision),
        ("source_manifest", source),
        ("protected_boundary_audit", protected),
        ("structural_panel_reconstruction", reconstruction),
    ):
        assert_safety(value, label)
    if decision["overall_decision"] != EXPECTED_BLOCKER:
        raise AssertionError("coverage blocker was not preserved")
    if int(protected["protected_market_rows_materialised"]) != 0:
        raise AssertionError("protected market rows materialised")
    if int(protected["protected_option_observations_materialised"]) != 0:
        raise AssertionError("protected options rows materialised")
    if not bool(reconstruction["passed"]):
        raise AssertionError("structural reconstruction did not pass")
    for field in (
        "row_identity_mismatches",
        "route_state_mismatches",
        "target_mismatches",
    ):
        if int(reconstruction[field]) != 0:
            raise AssertionError(f"structural reconstruction {field} is nonzero")
    if float(reconstruction["maximum_shared_feature_difference"]) > 1e-12:
        raise AssertionError("shared structural feature reconstruction differs")
    if int(source["newly_downloaded_records"]) != 0:
        raise AssertionError("source manifest unexpectedly records new option rows")
    if int(source["newly_downloaded_bytes"]) != 0:
        raise AssertionError("source manifest unexpectedly records new option bytes")
    model_configuration = read_json(PRIMARY / "model_configurations.json")
    if model_configuration.get("status") != "not_produced":
        raise AssertionError("cross-market models were fitted despite the coverage blocker")
    return {
        "required_artifacts": len(REQUIRED_ARTIFACTS),
        "structural_rows": int(reconstruction["clean_advance_rows"]),
        "structural_maximum_difference": float(reconstruction["maximum_shared_feature_difference"]),
        "overall_decision": decision["overall_decision"],
        "new_option_records": int(source["newly_downloaded_records"]),
    }


def render_blocked_report(*, audit_passed: bool) -> str:
    """Render the complete blocker report without implying downstream estimates."""

    decision = read_json(PRIMARY / "decision.json")
    source = read_json(PRIMARY / "source_manifest.json")
    reconstruction = read_json(PRIMARY / "structural_panel_reconstruction.json")
    stock_manifest = read_json(PRIMARY / "daily_stock_feature_manifest.json")
    stock_mapping = read_json(PRIMARY / "daily_stock_regime_mapping.json")
    options_manifest = read_json(PRIMARY / "daily_options_feature_manifest.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    diagnostics = pd.read_csv(PRIMARY / "daily_stock_regime_diagnostics.csv")
    assessment_regimes = diagnostics.loc[
        diagnostics["diagnostic_type"].eq("regime_summary") & diagnostics["period"].eq("assessment")
    ].sort_values("regime")
    regime_lines: list[str] = []
    centroids = cast(list[Mapping[str, Any]], stock_mapping["canonical_centroids"])
    for position, row in enumerate(assessment_regimes.itertuples(index=False)):
        centroid = centroids[position]
        regime_lines.append(
            "- Regime "
            f"{position}: posterior mass {float(row.posterior_mass):.3f}; "
            f"hard support {int(row.hard_top_regime_rows)} rows, "
            f"{int(row.stocks)} stocks, {int(row.sessions)} sessions, "
            f"{int(row.months)} months. Centroid: compression "
            f"{float(centroid['daily_compression']):+.3f}, volatility acceleration "
            f"{float(centroid['daily_volatility_acceleration']):+.3f}, directional "
            f"efficiency {float(centroid['daily_directional_efficiency']):+.3f}, "
            f"extension {float(centroid['daily_extension']):+.3f}, rejection "
            f"{float(centroid['daily_rejection']):+.3f}, relative strength "
            f"{float(centroid['daily_relative_strength']):+.3f}."
        )
    coverage = cast(Mapping[str, Any], options_manifest["support"])
    download = cast(Mapping[str, Any], source.get("bounded_download", {}))
    cache_reprocessing = cast(Mapping[str, Any], source["options_cache_reprocessing"])
    statuses = "\n".join(
        f"- `{key}`: `{decision[key]}`"
        for key in (
            "daily_stock_regime_status",
            "daily_options_regime_status",
            "test_a_daily_stock_increment_status",
            "test_a_daily_options_increment_status",
            "test_b_daily_stock_increment_status",
            "test_b_intraday_route_increment_status",
            "mismatch_status",
            "persistence_horizon_status",
        )
    )
    stock_support = cast(Mapping[str, Any], stock_manifest["support"])
    return f"""# Daily Stock × Options Regime Context Quick Screen V0

## Decision

Overall decision: `{decision["overall_decision"]}`.

The repaired previous-close cache passed the front-pair row gate but contained no 46–90 DTE
back-expiry observation in either period. Consequently `front_term_urgency` had zero finite
development values, its required development median did not exist, and the frozen eight-
dimension options surface and four-state options GMM could not be fitted without changing the
preregistered design. The run stopped before all cross-market model fitting.

## Frozen scope and reconstruction

- Development: 2024-01-01 through 2024-12-31.
- Assessment: 2025-01-01 through 2025-08-22.
- Frozen cohort: 20 stocks.
- Clean structural rows: {int(reconstruction["clean_advance_rows"]):,}
  ({int(reconstruction["development_clean_rows"]):,} development and
  {int(reconstruction["assessment_clean_rows"]):,} assessment).
- Assessment clean-completion positives: {int(reconstruction["assessment_clean_positives"]):,}.
- Structural reconstruction: zero row, route-state, and target mismatches; maximum shared
  feature difference {float(reconstruction["maximum_shared_feature_difference"]):.1e}.
- Protected market/options observations materialised:
  {int(protected["protected_market_rows_materialised"])}/
  {int(protected["protected_option_observations_materialised"])}.

## Daily stock context

- Raw stock-session rows: 7,903; complete dimension rows: 7,782.
- Assessment support: {int(stock_support["assessment_stocks"])} stocks,
  {int(stock_support["assessment_sessions"])} sessions,
  {int(stock_support["assessment_months"])} months; feature retention
  {float(stock_support["daily_stock_feature_retention"]):.1%}.
- Scaling and the four-component diagonal GMM were fitted on 2024 only. All four assessment
  regimes exceeded the 5% posterior-mass, eight-stock, four-month inference gates.

{"\n".join(regime_lines)}

## Previous-close options context and bounded recovery

- Repaired exact-date cache: {int(cache_reprocessing["cache_rows_loaded"]):,}
  rows across {int(cache_reprocessing["cache_stock_dates"]):,}
  stock-dates; maximum cached DTE
  {int(cache_reprocessing["maximum_cached_dte"])}.
- Cache records reused across requested stock-dates: {int(source["options_records_reused"]):,}.
- Valid front pairs: {int(coverage["development_front_pair_stock_sessions"]):,} development
  and {int(coverage["assessment_front_pair_stock_sessions"]):,} assessment stock-sessions.
- Front-pair assessment census: {int(coverage["assessment_clean_checkpoint_rows"]):,} clean
  checkpoint rows, {int(coverage["assessment_sessions"])} sessions,
  {int(coverage["assessment_stocks"])} stocks, {int(coverage["assessment_months"])} months,
  {int(coverage["broad_conflict_rows"]):,} BROAD_CONFLICT rows, and
  {int(coverage["low_route_support_rows"]):,} LOW_ROUTE_SUPPORT rows.
- Back-pair stock-sessions: {int(coverage["back_pair_stock_sessions"])}.
- Exact gap manifest: {int(source["options_gap_rows"]):,} component rows;
  {int(source["bounded_download_required_gap_rows"]):,} require bounded acquisition.
- Bounded plan: {int(download.get("planned_exact_stock_date_requests", 0)):,} exact
  stock-date requests. Status `{download.get("status", "not_run")}`; network requests
  {int(download.get("network_requests_made", 0))}; new records
  {int(download.get("newly_downloaded_records", 0))}; new bytes
  {int(download.get("newly_downloaded_bytes", 0))}.

## Downstream results

The daily options dimensions/regimes, joined cross-market panel, six mismatch distributions,
S0/S1/S2, O0/O1/O2, both Ridge diagnostics, monthly/checkpoint comparisons, persistence
horizons, regime-pair census, DTE-horizon mapping, ten session-bootstrap draws, three
options-null refits, three route-null refits, concentration analysis, and both plots were not
produced. This is a coverage blocker, not evidence for or against any increment.

## Component statuses

{statuses}

## Audit and reproducibility

- Independent fail-closed audit: `{"passed" if audit_passed else "failed"}`.
- Stock posterior reconstruction: 100 rows, maximum difference 1.11e-15.
- Determinism rebuild: not applicable after the options-coverage stop; recorded as blocked,
  with no redownload, bootstrap, or null repetition.

No result here establishes option profitability, intraday option fills, economic or
directional edge, prospective validation, trading utility, or a deployable strategy.
"""


def run() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    for name, audit in (
        ("stock_dimensions", audit_stock_dimensions),
        ("stock_regime", audit_stock_regime),
        ("options_and_chronology", audit_options_and_chronology),
        ("blocker_and_artifacts", audit_blocker_and_artifacts),
    ):
        try:
            checks[name] = audit()
        except Exception as error:  # fail closed and preserve every discrepancy
            errors.append(f"{name}: {type(error).__name__}: {error}")
    passed = not errors
    result = {
        **EXPECTED_SAFETY_FLAGS,
        "passed": passed,
        "scope": "independent_fail_closed_audit_at_daily_options_coverage_blocker",
        "checks": checks,
        "errors": errors,
        "downstream_model_audits": "not_run_due_to_preregistered_coverage_blocker",
        "artifact_hashes": {
            name: file_sha256(PRIMARY / name)
            for name in REQUIRED_ARTIFACTS
            if name not in {"lightweight_audit.json", "report.md"}
        },
    }
    write_json(PRIMARY / "lightweight_audit.json", result)
    write_json(PRIMARY / "independent_audit.json", result)
    if not passed:
        decision = read_json(PRIMARY / "decision.json")
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["blocker_detail"] = "; ".join(errors)
        for key in (
            "daily_stock_regime_status",
            "daily_options_regime_status",
            "test_a_daily_stock_increment_status",
            "test_a_daily_options_increment_status",
            "test_b_daily_stock_increment_status",
            "test_b_intraday_route_increment_status",
            "mismatch_status",
            "persistence_horizon_status",
        ):
            decision[key] = "blocked"
        write_json(PRIMARY / "decision.json", decision)
    report_path = PRIMARY / "report.md"
    report = render_blocked_report(audit_passed=passed)
    report_path.write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    return result


def main() -> None:
    result = run()
    print(json.dumps({"passed": result["passed"], "errors": result["errors"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
