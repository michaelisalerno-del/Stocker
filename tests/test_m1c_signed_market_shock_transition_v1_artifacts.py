from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-signed-market-shock-transition-v1"
)
PRIMARY = EXPERIMENT / "artifacts" / "primary"
TAIL_EPISODES = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-tail-phase-v1"
    / "artifacts"
    / "primary"
    / "fresh_episode_results_v1.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_required_signed_market_shock_artifacts_are_complete_and_consistent() -> None:
    required = {
        "frozen_configuration_v1.json",
        "canonical_market_proxy_audit_v1.json",
        "checkpoint_shock_threshold_manifest_v1.json",
        "predictor_calibration_table_v1.csv",
        "episode_market_state_response_v1.parquet",
        "unique_shock_events_v1.csv",
        "assessment_outcomes_v1.csv",
        "stress_outcomes_v1.csv",
        "continuation_arm_results_v1.csv",
        "resistance_arm_results_v1.csv",
        "checkpoint_stratified_mechanism_results_v1.csv",
        "continuous_ranking_results_v1.csv",
        "shock_sign_stratification_v1.csv",
        "normal_regime_comparison_v1.csv",
        "baseline_comparisons_v1.csv",
        "tail_phase_diagnostics_v1.csv",
        "session_cluster_bootstrap_v1.parquet",
        "shock_event_cluster_bootstrap_v1.parquet",
        "leave_one_month_out_v1.csv",
        "leave_one_stock_out_v1.csv",
        "leave_one_shock_event_out_v1.csv",
        "null_and_placebo_results_v1.json",
        "concentration_report_v1.csv",
        "summary_v1.json",
        "provenance_manifest_v1.json",
    }
    assert all((PRIMARY / name).is_file() for name in required)
    assert (
        EXPERIMENT / "reports" / "m1c_signed_market_shock_transition_v1.md"
    ).is_file()

    summary = json.loads((PRIMARY / "summary_v1.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (PRIMARY / "provenance_manifest_v1.json").read_text(encoding="utf-8")
    )
    assert summary["confirmations"] == {
        "a1_unchanged": True,
        "broker_accessed": False,
        "contaminated_fields_used": False,
        "m1c_unchanged": True,
        "option_profitability_tested": False,
        "order_placed": False,
        "order_routing_enabled": False,
        "protected_2026_outcomes_accessed": False,
        "tail_phase_v1_unchanged": True,
    }
    assert provenance["frozen_regression"]["passed"] is True
    assert provenance["target_partition_audit"][
        "directional_union_equals_frozen_strict_event"
    ]
    assert not any(provenance["execution_confirmations"].values())
    assert not any(
        provenance["protected_data_confirmation"][key]
        for key in (
            "protected_data_opened",
            "protected_outcomes_calculated",
            "protected_outcomes_displayed",
            "protected_outcomes_inspected",
        )
    )
    for relative, expected_hash in provenance["output_sha256"].items():
        path = EXPERIMENT / relative if relative.startswith("reports/") else PRIMARY / relative
        assert _sha256(path) == expected_hash


def test_episode_target_causality_and_frozen_system_regressions() -> None:
    episodes = pd.read_parquet(PRIMARY / "episode_market_state_response_v1.parquet")
    source = pd.read_parquet(TAIL_EPISODES)
    assert len(episodes) == len(source) == 1416
    assert episodes["session"].astype(str).max() < "2026-01-01"
    assert not episodes.duplicated(["stock", "session", "checkpoint"]).any()

    complete = episodes["primary_outcome_complete_v1"].astype(bool)
    directional = episodes["primary_outcome_state_v1"].isin(
        ["MATERIAL_UP", "MATERIAL_DOWN"]
    )
    strict = episodes["future_signed_return_15m"].abs().gt(episodes["threshold_15m"])
    assert directional.loc[complete].equals(strict.loc[complete])
    assert episodes.loc[complete, "primary_outcome_state_v1"].isin(
        ["MATERIAL_UP", "MATERIAL_DOWN", "NO_MATERIAL_MOVE"]
    ).all()

    maximum_market = pd.to_datetime(
        episodes["maximum_market_timestamp_v1"],
        utc=True,
        errors="coerce",
    )
    maximum_stock = pd.to_datetime(
        episodes["maximum_stock_timestamp_v1"],
        utc=True,
        errors="coerce",
    )
    signal = pd.to_datetime(episodes["signal_timestamp"], utc=True)
    assert maximum_market.dropna().le(signal.loc[maximum_market.notna()]).all()
    assert maximum_stock.dropna().le(signal.loc[maximum_stock.notna()]).all()
    assert episodes.loc[
        episodes["checkpoint"].ne(6) & episodes["market_window_complete_v1"],
        ["w0_bar_ordinals_v1", "w1_bar_ordinals_v1"],
    ].apply(
        lambda row: set(row["w0_bar_ordinals_v1"]).isdisjoint(
            row["w1_bar_ordinals_v1"]
        ),
        axis=1,
    ).all()

    identity = ["stock", "session", "checkpoint"]
    frozen_fields = [
        "M1C_probability",
        "m1c_high_tail_v1",
        "m1c_tail_phase_v1",
        "A1_probability_up_v1",
        "A1_action_v1",
        "episode_id",
        "existing_fresh_episode_identifier",
    ]
    comparison = source[[*identity, *frozen_fields]].merge(
        episodes[[*identity, *frozen_fields]],
        on=identity,
        suffixes=("_source", "_experiment"),
        validate="one_to_one",
    )
    for field in frozen_fields:
        left = comparison[f"{field}_source"]
        right = comparison[f"{field}_experiment"]
        if pd.api.types.is_numeric_dtype(left):
            assert np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
        else:
            assert left.equals(right)
    assert {
        "archived_signed_pressure",
        "archived_tension",
        "option_pnl",
        "option_return",
    }.isdisjoint(episodes.columns)


def test_calibration_events_replications_and_report_contract() -> None:
    thresholds = json.loads(
        (PRIMARY / "checkpoint_shock_threshold_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoints = thresholds["checkpoints"]
    assert [item["checkpoint"] for item in checkpoints] == list(range(6, 35, 2))
    assert checkpoints[0]["calibration_complete_v1"] is False
    assert checkpoints[0]["market_return_w0_support_v1"] == 0
    assert checkpoints[0]["market_return_w0_q10_v1"] is None
    assert all(item["calibration_complete_v1"] for item in checkpoints[1:])
    assert thresholds["pooling_fallback_used"] is False
    assert thresholds["calibration_period"]["predictors_only"] is True

    quintiles = json.loads(
        (PRIMARY / "response_quintile_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert quintiles["calibration"]["calibration_complete_v1"] is True
    assert quintiles["calibration"]["support_v1"] == 8
    assert quintiles["outcomes_used"] is False

    events = pd.read_csv(PRIMARY / "unique_shock_events_v1.csv")
    assert events["market_shock_event_id_v1"].is_unique
    assert events["session"].astype(str).max() < "2026-01-01"
    episodes = pd.read_parquet(PRIMARY / "episode_market_state_response_v1.parquet")
    onset = episodes.loc[
        episodes["market_shock_state_v1"].isin(
            ["NEGATIVE_SHOCK_ONSET", "POSITIVE_SHOCK_ONSET"]
        )
    ]
    shared = onset.groupby(
        ["session", "checkpoint", "market_proxy_v1", "shock_sign_v1"]
    )["market_shock_event_id_v1"].nunique()
    assert shared.eq(1).all()

    for filename in (
        "session_cluster_bootstrap_v1.parquet",
        "shock_event_cluster_bootstrap_v1.parquet",
    ):
        draws = pd.read_parquet(PRIMARY / filename)
        assert draws.groupby("period")["draw"].nunique().to_dict() == {
            "assessment": 5000,
            "stress": 5000,
        }
    null_draws = pd.read_parquet(PRIMARY / "primary_null_draws_v1.parquet")
    assert null_draws.groupby("period")["draw"].nunique().to_dict() == {
        "assessment": 1000,
        "stress": 1000,
    }
    accounting = pd.read_csv(PRIMARY / "event_accounting_v1.csv")
    episode_counts = episodes.groupby("partition").size().to_dict()
    assert all(
        row.incomplete_fresh_episodes <= episode_counts[row.period]
        for row in accounting.itertuples(index=False)
    )
    for filename in ("assessment_outcomes_v1.csv", "stress_outcomes_v1.csv"):
        outcomes = pd.read_csv(PRIMARY / filename)
        sign_class = outcomes.loc[
            outcomes["group_type"].eq("shock_sign_x_response_class")
        ]
        assert len(sign_class) == 10
        assert "UNKNOWN_INCOMPLETE" in "|".join(sign_class["group_value"])
        assert (
            outcomes["group_type"].eq("resisting_descriptive_subtype").sum()
            == 2
        )
    checkpoint_strata = pd.read_csv(
        PRIMARY / "checkpoint_stratified_mechanism_results_v1.csv"
    )
    assert len(checkpoint_strata) == 60
    assert set(checkpoint_strata["checkpoint"]) == set(range(6, 35, 2))

    source_signal = pd.to_datetime(episodes["signal_timestamp"], utc=True)
    market_signal = pd.to_datetime(
        episodes["market_signal_timestamp_v1"],
        utc=True,
    )
    assert source_signal.equals(market_signal)

    report = (
        EXPERIMENT / "reports" / "m1c_signed_market_shock_transition_v1.md"
    ).read_text(encoding="utf-8")
    assert "Option profitability: **Not tested**" in report
    assert "bid withdrawal" in report
    assert "ask withdrawal" in report
    assert "replenishment" in report
    assert "queue behaviour" in report
    assert "No broker was accessed" in report
