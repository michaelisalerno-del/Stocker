from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

WORK = Path(__file__).resolve().parents[1] / "research/slrno-v2/20260714-regime-loop-handoff/work"
RUNNER = WORK / "run_directed_economic_loop_regime_rotation_v1.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rotation_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_and_all_frozen_inputs_verify() -> None:
    module = _load()

    contract, hashes, snapshot = module.verify_contract_and_inputs()

    assert contract["activation_target"]["primary_window_sessions"] == 3
    assert hashes["family_mapping"] == contract["inputs"]["family_mapping"]["sha256"]
    assert len(snapshot) == 64


def test_forecasts_are_split_from_later_outcomes() -> None:
    module = _load()
    scored = pd.DataFrame(
        {
            "forecast_id": ["f-1"],
            "model_name": ["M3_directed_family_rotation"],
            "predicted_activation_probability": [0.4],
            "activation_target": [True],
            "target_available": [True],
            "label_availability_timestamp": pd.to_datetime(["2025-01-10T20:00Z"]),
            "target_status": ["activation_observed"],
            "first_activation_session": ["2025-01-08"],
            "target_episode_ids": ["episode-1"],
            "observed_activation_count": [1],
            "multiple_activation_flag": [False],
            "no_activation_flag": [False],
        }
    )

    forecasts, outcomes = module.split_forecast_and_outcome_ledgers(scored)

    assert "activation_target" not in forecasts.columns
    assert "target_status" not in forecasts.columns
    assert outcomes.loc[0, "forecast_id"] == "f-1"
    assert bool(outcomes.loc[0, "activation_target"])


def test_primary_scientific_decision_cannot_ignore_failed_increment() -> None:
    module = _load()
    comparisons = pd.DataFrame(
        {
            "comparison": ["M3_vs_M1", "M3_vs_M2"],
            "brier_improvement": [-0.001, 0.001],
            "log_loss_improvement": [-0.001, 0.001],
            "brier_interval_lower": [-0.002, -0.001],
        }
    )

    assert (
        module.scientific_decision(
            comparisons,
            economic_metrics=pd.DataFrame(),
            null_metrics=pd.DataFrame(),
            loo_results=pd.DataFrame(),
            concentration=pd.DataFrame(),
        )
        == "destination_own_history_sufficient"
    )


def test_exact_rerun_comparison_fails_on_changed_machine_readable_file(
    tmp_path: Path,
) -> None:
    module = _load()
    primary = tmp_path / "primary"
    rerun = tmp_path / "rerun"
    primary.mkdir()
    rerun.mkdir()
    (primary / "metrics.csv").write_bytes(b"a\n1\n")
    (rerun / "metrics.csv").write_bytes(b"a\n2\n")

    result = module.verify_exact_rerun(rerun, primary)

    assert result["byte_identical"] is False
    assert result["hash_mismatches"] == ["metrics.csv"]


def test_runner_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    module = _load()
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        module.ensure_new_output(output)
