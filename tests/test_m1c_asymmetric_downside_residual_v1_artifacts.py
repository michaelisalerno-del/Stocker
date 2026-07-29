from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from stocker_research.m1c_asymmetric_downside_residual_v1 import (
    DOWNSIDE_FEATURES,
    M1C_HIGH_MOVEMENT_THRESHOLD,
    fit_downside_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-asymmetric-downside-residual-v1"
    / "artifacts"
    / "primary"
)
TAIL_EPISODES = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-tail-phase-v1"
    / "artifacts"
    / "primary"
    / "fresh_episode_results_v1.parquet"
)
IDENTITY = ["stock", "session", "checkpoint"]


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(PRIMARY / "assessment_episode_predictions_v1.parquet"),
            pd.read_parquet(PRIMARY / "stress_episode_predictions_v1.parquet"),
        ],
        ignore_index=True,
    )


def test_frozen_m1c_membership_and_fresh_identifiers_are_unchanged(
    predictions: pd.DataFrame,
) -> None:
    source = pd.read_parquet(
        TAIL_EPISODES,
        filters=[("session", ">=", "2025-01-01"), ("session", "<=", "2025-12-31")],
    )
    comparison = source[
        [
            *IDENTITY,
            "M1C_probability",
            "m1c_high_tail_v1",
            "episode_id",
            "existing_fresh_episode_identifier",
        ]
    ].merge(
        predictions[
            [
                *IDENTITY,
                "M1C_probability",
                "m1c_high_tail_v1",
                "episode_id",
                "existing_fresh_episode_identifier",
            ]
        ],
        on=IDENTITY,
        suffixes=("_source", "_experiment"),
        validate="one_to_one",
    )

    assert len(comparison) == len(predictions)
    assert np.array_equal(
        comparison["M1C_probability_source"],
        comparison["M1C_probability_experiment"],
    )
    assert predictions["M1C_probability"].ge(M1C_HIGH_MOVEMENT_THRESHOLD).all()
    assert predictions["m1c_high_tail_v1"].astype(bool).all()
    assert comparison["episode_id_source"].equals(comparison["episode_id_experiment"])
    assert comparison["existing_fresh_episode_identifier_source"].equals(
        comparison["existing_fresh_episode_identifier_experiment"]
    )


def test_frozen_a1_and_tail_phase_fields_are_unchanged(
    predictions: pd.DataFrame,
) -> None:
    source = pd.read_parquet(
        TAIL_EPISODES,
        filters=[("session", ">=", "2025-01-01"), ("session", "<=", "2025-12-31")],
    )
    fields = [
        "A1_probability_up_v1",
        "A1_action_v1",
        "m1c_tail_phase_v1",
        "movement_consumed_v1",
    ]
    comparison = source[[*IDENTITY, *fields]].merge(
        predictions[[*IDENTITY, *fields]],
        on=IDENTITY,
        suffixes=("_source", "_experiment"),
        validate="one_to_one",
    )

    assert np.allclose(
        comparison["A1_probability_up_v1_source"],
        comparison["A1_probability_up_v1_experiment"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert comparison["A1_action_v1_source"].equals(comparison["A1_action_v1_experiment"])
    assert comparison["m1c_tail_phase_v1_source"].equals(comparison["m1c_tail_phase_v1_experiment"])
    assert np.allclose(
        comparison["movement_consumed_v1_source"],
        comparison["movement_consumed_v1_experiment"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert predictions["phase_at_trigger_v1"].isin(["FIRST_ENTRY", "RE_ENTRY"]).all()


def test_artifacts_contain_no_contaminated_or_joint_probability_fields(
    predictions: pd.DataFrame,
) -> None:
    forbidden = ("signed_pressure", "tension", "peer_slate")
    assert not any(any(token in column.lower() for token in forbidden) for column in predictions)
    assert not any("joint" in column.lower() for column in predictions)
    manifest = cast(
        dict[str, Any],
        json.loads((PRIMARY / "feature_manifest_v1.json").read_text(encoding="utf-8")),
    )
    assert manifest["ordered_model_columns"] == list(DOWNSIDE_FEATURES)
    model = cast(
        dict[str, Any],
        json.loads((PRIMARY / "final_model_parameters_v1.json").read_text(encoding="utf-8")),
    )
    assert model["feature_names"] == list(DOWNSIDE_FEATURES)
    assert "upside_model" not in model


def test_frozen_decision_contract_blocks_the_29_action_assessment_put_cell() -> None:
    summary = cast(
        dict[str, Any],
        json.loads((PRIMARY / "summary_v1.json").read_text(encoding="utf-8")),
    )
    policy = pd.read_csv(PRIMARY / "combined_policy_metrics_v1.csv").set_index("period")

    assert policy.loc["assessment", "put_actions"] == 29
    assert not summary["decision_contract_checks"]["assessment_action_support"]
    assert summary["directional_diagnostic_decision"] == "blocked_insufficient_support"
    assert summary["descriptive_directional_finding"] == ("low_downside_does_not_imply_upside")
    phases = pd.read_csv(PRIMARY / "tail_phase_diagnostics_v1.csv")
    assessment_reentry = phases.loc[
        phases["period"].eq("assessment")
        & phases["phase"].eq("RE_ENTRY")
        & phases["diagnostic"].eq("summary")
    ].iloc[0]
    assert assessment_reentry["conditional_mover_rows"] == 22
    assert assessment_reentry["support_status"] == "blocked_insufficient_support"
    assert pd.isna(assessment_reentry["downside_auc"])

    baselines = pd.read_csv(PRIMARY / "baseline_comparisons_v1.csv")
    d2 = baselines.loc[
        baselines["period"].eq("assessment") & baselines["policy"].eq("existing_frozen_D2")
    ]
    assert set(d2["evaluation_scope"]) == {
        "all_complete_episodes",
        "asymmetric_policy_acted_episodes",
    }

    bootstrap = pd.read_csv(PRIMARY / "session_cluster_bootstrap_v1.csv")
    permutation = pd.read_csv(PRIMARY / "label_permutation_summary_v1.csv")
    assert bootstrap.groupby("period")["seed"].first().nunique() == 1
    assert permutation.groupby("period")["seed"].first().nunique() == 1


def test_2025_rows_cannot_enter_model_fitting_or_threshold_freezing(
    predictions: pd.DataFrame,
) -> None:
    frame = predictions.head(8).loc[:, ["session", *DOWNSIDE_FEATURES]].copy()
    frame["is_down_move_v1"] = [0, 1] * 4

    with pytest.raises(ValueError, match="restricted to 2024"):
        fit_downside_model(frame, target_column="is_down_move_v1")

    oof = pd.read_parquet(PRIMARY / "development_oof_predictions_v1.parquet")
    thresholds = cast(
        dict[str, Any],
        json.loads((PRIMARY / "frozen_action_thresholds_v1.json").read_text(encoding="utf-8")),
    )
    assert pd.to_datetime(oof["session"], utc=True).dt.year.eq(2024).all()
    assert thresholds["low"] == np.quantile(
        oof["q_down_oof"],
        0.2,
        method="linear",
    )
    assert thresholds["high"] == np.quantile(
        oof["q_down_oof"],
        0.8,
        method="linear",
    )


def test_protected_and_execution_guards_remain_fail_closed(
    predictions: pd.DataFrame,
) -> None:
    provenance = cast(
        dict[str, Any],
        json.loads((PRIMARY / "provenance_manifest_v1.json").read_text(encoding="utf-8")),
    )
    protected = provenance["protected_data_confirmation"]
    execution = provenance["execution_confirmation"]

    assert predictions["session"].astype(str).lt("2026-01-01").all()
    assert not protected["protected_2026_outcome_read"]
    assert not protected["protected_2026_outcome_calculated"]
    assert not protected["protected_2026_outcome_displayed"]
    assert not protected["protected_2026_outcome_inspected"]
    assert not execution["broker_accessed"]
    assert not execution["order_routing_path_imported"]
    assert not execution["order_routing_enabled"]
    assert not execution["orders_placed"]

    implementation = (
        REPO_ROOT
        / "packages"
        / "stocker_research"
        / "src"
        / "stocker_research"
        / "m1c_asymmetric_downside_residual_v1.py"
    ).read_text(encoding="utf-8")
    runner = (PRIMARY.parents[1] / "run_experiment.py").read_text(encoding="utf-8")
    assert "stocker_execution" not in implementation
    assert "stocker_execution" not in runner
    assert "placeOrder" not in implementation
    assert "placeOrder" not in runner


def test_probability_and_predictor_artifact_invariants(
    predictions: pd.DataFrame,
) -> None:
    assert predictions["q_down_v1"].between(0.0, 1.0).all()
    assert np.isfinite(predictions.loc[:, DOWNSIDE_FEATURES].to_numpy(float)).all()
    assert (
        predictions["maximum_predictor_bar_ordinal"] == predictions["checkpoint"].astype(int) - 1
    ).all()
    assert (
        pd.to_datetime(predictions["maximum_predictor_timestamp"], utc=True)
        <= pd.to_datetime(predictions["feature_available_timestamp_utc"], utc=True)
    ).all()
    assert not predictions["exact_probability_decomposition_supported_v1"].astype(bool).any()
