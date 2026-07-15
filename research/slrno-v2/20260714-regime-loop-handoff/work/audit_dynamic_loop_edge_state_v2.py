#!/usr/bin/env python3
"""Independent causal and exact-rerun audit for dynamic loop edge-state V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CORE_ARTIFACTS = (
    "session_payoff_panel.parquet",
    "causal_edge_state_forecasts.parquet",
    "trade_decisions.parquet",
    "model_comparison_metrics.csv",
    "calibration_results.csv",
    "change_point_diagnostics.csv",
    "hindsight_episode_diagnostics.parquet",
    "stress_test_results.csv",
    "causal_feature_panel.parquet",
    "prequential_scored_targets.parquet",
    "hindsight_episode_states.parquet",
    "trading_slices.csv",
    "concentration_analysis.csv",
)
EXPECTED_MODELS = {
    "v1_60_session_selector",
    "ewma_short_memory",
    "payoff_only_change_point",
    "hierarchical_payoff_history_change_point",
    "hierarchical_change_point",
}


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _manifest_hashes_are_valid(root: Path) -> tuple[bool, str]:
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    failures: list[str] = []
    for item in manifest["files"]:
        candidate = Path(str(item["name"]))
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists() or sha256(candidate) != str(item["sha256"]):
            failures.append(str(item["name"]))
    return not failures, "invalid=" + ",".join(
        failures
    ) if failures else "all recorded hashes match"


def _timestamps_strictly_before(
    frame: pd.DataFrame,
    availability_column: str,
    decision_column: str,
) -> bool:
    availability = pd.to_datetime(frame[availability_column], utc=True, errors="coerce")
    decision = pd.to_datetime(frame[decision_column], utc=True, errors="raise")
    return bool((availability.isna() | availability.lt(decision)).all())


def audit(primary: Path, exact_rerun: Path) -> dict[str, Any]:
    checks: list[AuditCheck] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(AuditCheck(name=name, passed=bool(passed), detail=detail))

    for label, root in (("primary", primary), ("exact_rerun", exact_rerun)):
        missing = [
            name
            for name in (*CORE_ARTIFACTS, "run_metadata.json", "artifact_manifest.json")
            if not (root / name).exists()
        ]
        check(f"{label}_artifact_completeness", not missing, f"missing={missing}")
        if not missing:
            valid, detail = _manifest_hashes_are_valid(root)
            check(f"{label}_manifest_hashes", valid, detail)

    primary_metadata = json.loads((primary / "run_metadata.json").read_text())
    exact_metadata = json.loads((exact_rerun / "run_metadata.json").read_text())
    for label, metadata in (("primary", primary_metadata), ("exact_rerun", exact_metadata)):
        check(
            f"{label}_research_safety",
            metadata.get("research_only") is True
            and metadata.get("live_ordering_enabled") is False
            and metadata.get("order_placement") == "disabled",
            "research_only/live/order flags",
        )
        recovery = metadata.get("recovery_equivalence", {})
        check(
            f"{label}_v1_recovery_equivalence",
            recovery.get("top_loop_exact") is True
            and recovery.get("top_probability_exact") is True
            and recovery.get("state_and_history_token_exact") is True,
            json.dumps(recovery, sort_keys=True),
        )

    for field in (
        "run_id",
        "git_sha",
        "repository_branch",
        "data_snapshot_identifier",
        "universe_snapshot_identifier",
        "configuration_hash",
        "model_version",
        "cost_model_version",
        "feature_schema_version",
        "fixed_horizon_bars",
        "random_seed",
    ):
        check(
            f"metadata_exact_{field}",
            primary_metadata.get(field) == exact_metadata.get(field),
            f"primary={primary_metadata.get(field)!r}; exact={exact_metadata.get(field)!r}",
        )

    for name in CORE_ARTIFACTS:
        primary_path = primary / name
        exact_path = exact_rerun / name
        primary_table = _read_table(primary_path)
        exact_table = _read_table(exact_path)
        try:
            pd.testing.assert_frame_equal(
                primary_table,
                exact_table,
                check_dtype=True,
                check_exact=True,
                check_like=False,
            )
            equal = True
            detail = f"rows={len(primary_table)}; exact dataframe equality"
        except AssertionError as error:
            equal = False
            detail = str(error).splitlines()[0]
        check(f"exact_rerun_{name}", equal, detail)

    primary_plots = sorted(path.name for path in primary.glob("episode_*.png"))
    exact_plots = sorted(path.name for path in exact_rerun.glob("episode_*.png"))
    check("exact_rerun_plot_set", primary_plots == exact_plots, f"plots={primary_plots}")
    for name in primary_plots:
        check(
            f"exact_rerun_{name}",
            sha256(primary / name) == sha256(exact_rerun / name),
            "PNG SHA-256 equality",
        )

    forecasts = pd.read_parquet(primary / "causal_edge_state_forecasts.parquet")
    decisions = pd.read_parquet(primary / "trade_decisions.parquet")
    panel = pd.read_parquet(primary / "session_payoff_panel.parquet")
    features = pd.read_parquet(primary / "causal_feature_panel.parquet")
    stress = pd.read_csv(primary / "stress_test_results.csv")

    forecast_keys = [
        "model_name",
        "period",
        "score_session",
        "loop_id",
        "orientation",
        "horizon",
    ]
    check(
        "one_frozen_forecast_per_model_cell_session",
        not forecasts.duplicated(forecast_keys).any(),
        f"rows={len(forecasts)}",
    )
    check(
        "registered_model_set",
        set(forecasts["model_name"].astype(str).unique()) == EXPECTED_MODELS,
        ",".join(sorted(forecasts["model_name"].astype(str).unique())),
    )
    check(
        "training_availability_precedes_decision",
        _timestamps_strictly_before(
            forecasts,
            "training_latest_availability_timestamp",
            "decision_timestamp",
        ),
        "null cold starts allowed; every populated timestamp is strict",
    )
    check(
        "feature_availability_precedes_decision",
        _timestamps_strictly_before(
            forecasts,
            "feature_max_availability_timestamp",
            "decision_timestamp",
        ),
        "null cold starts allowed; every populated timestamp is strict",
    )
    frozen_at = pd.to_datetime(forecasts["prediction_frozen_at"], utc=True, errors="raise")
    decision_at = pd.to_datetime(forecasts["decision_timestamp"], utc=True, errors="raise")
    check(
        "prediction_frozen_at_decision",
        bool(frozen_at.eq(decision_at).all()),
        "forecast ledger freeze timestamp equality",
    )
    check(
        "opportunity_forecast_frozen_before_payoff",
        bool(decisions["forecast_frozen_before_payoff"].fillna(False).all()),
        f"rows={len(decisions)}",
    )
    feature_availability = pd.to_datetime(
        features["feature_availability_timestamp"], utc=True, errors="coerce"
    )
    score_open = features["score_session"].map(
        lambda value: pd.Timestamp(f"{value} 09:30", tz="America/New_York").tz_convert("UTC")
    )
    check(
        "feature_panel_is_strictly_lagged",
        bool((feature_availability.isna() | feature_availability.lt(score_open)).all()),
        f"rows={len(features)}",
    )
    panel_keys = ["period", "session", "loop_id", "orientation", "horizon"]
    check(
        "one_session_level_statistical_unit",
        not panel.duplicated(panel_keys).any(),
        f"rows={len(panel)}",
    )
    check(
        "independent_stock_support_not_raw_fill_support",
        bool(
            panel["independent_stock_count"].gt(0).all()
            and panel["raw_fill_count"].ge(panel["independent_stock_count"]).all()
        ),
        "positive independent support and raw fills never below stocks",
    )
    check(
        "run_metadata_on_every_forecast",
        bool(
            forecasts["run_id"].eq(primary_metadata["run_id"]).all()
            and forecasts["configuration_hash"].eq(primary_metadata["configuration_hash"]).all()
            and forecasts["run_metadata_json"].notna().all()
        ),
        "run/config/row metadata",
    )
    loo = stress.loc[stress["stress_test"].eq("leave_one_stock_out")]
    expected_stock_count = len(primary_metadata["symbols"])
    check(
        "leave_one_stock_out_fully_retrained",
        bool(
            len(loo) == expected_stock_count
            and loo["detail"].astype(str).str.contains("full_model_retrained=true").all()
        ),
        f"rows={len(loo)}; expected={expected_stock_count}",
    )
    check(
        "historical_period_boundary",
        set(forecasts["period"].astype(int).unique()) == {2023, 2025},
        str(sorted(forecasts["period"].astype(int).unique())),
    )

    serialised_checks = [asdict(item) for item in checks]
    return {
        "audit_id": "dynamic_loop_edge_state_v2_independent_audit",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "primary": str(primary),
        "exact_rerun": str(exact_rerun),
        "check_count": len(checks),
        "passed_count": sum(item.passed for item in checks),
        "all_passed": all(item.passed for item in checks),
        "checks": serialised_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--exact-rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.primary.resolve(), args.exact_rerun.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
