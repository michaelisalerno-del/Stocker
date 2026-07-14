from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_loop_quality_failure_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("loop_quality_diagnostics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def _gate_detail(**overrides: bool) -> dict[str, object]:
    checks = {
        "support": True,
        "observed_rate": True,
        "mean_qcycle": True,
        "observed_over_qcontext": True,
        "lift_interval": True,
    }
    surface = {
        "relative_log_loss": True,
        "brier_difference": True,
        "daily_intervals": True,
        "quarter_and_stock_robustness": True,
        "ece_no_worse": True,
        "maximum_supported_bin_error": True,
    }
    for key, value in overrides.items():
        if key in checks:
            checks[key] = value
        elif key.startswith("conditional__"):
            surface[key.removeprefix("conditional__")] = value
        else:
            raise KeyError(key)
    return {
        "checks": checks,
        "conditional_gate": {"checks": surface},
        "joint_gate": {"checks": dict(surface)},
    }


def test_gate_family_parser_keeps_level_lift_loss_and_calibration_separate() -> None:
    detail = _gate_detail(
        observed_rate=False,
        conditional__daily_intervals=False,
        conditional__ece_no_worse=False,
    )
    flags = diagnostics.gate_family_flags(detail)
    assert flags["event_support"] is True
    assert flags["rate_level"] is False
    assert flags["rate_vs_context"] is True
    assert flags["residual_ci"] is True
    assert flags["conditional_core"] is True
    assert flags["conditional_daily_ci"] is False
    assert flags["conditional_calibration"] is False
    assert flags["joint_daily_ci"] is False


def test_weighted_rate_uses_frozen_inverse_overlap_weight() -> None:
    observed = np.asarray([1.0, 0.0, 1.0])
    weights = np.asarray([0.5, 1.0, 0.5])
    assert diagnostics.weighted_rate(observed, weights) == 0.5
    with pytest.raises(ValueError):
        diagnostics.weighted_rate([], [])


def test_two_axis_can_be_absolute_high_without_incremental_evidence() -> None:
    rows = []
    for tier in diagnostics.TIERS:
        for target in diagnostics.TARGETS:
            for horizon in diagnostics.HORIZONS:
                rows.append(
                    {
                        "tier": tier,
                        "target": target,
                        "horizon": horizon,
                        "pass": False,
                        "gate__rate_level": True,
                        "p75_high_rate_level": tier == "p75",
                        "incremental_ratio_and_residual": False,
                        "gate__conditional_core": True,
                        "gate__joint_core": True,
                        "gate__conditional_robustness": True,
                        "gate__joint_robustness": True,
                    }
                )
    row = diagnostics._period_axis_row(
        "2025",
        "cycle_test",
        pd.DataFrame(rows),
        support_pass=True,
        structural_pass=True,
        global_grade="unqualified",
    )
    assert row["absolute_movement_level_axis"] == "absolute_high_period"
    assert row["incremental_vs_context_axis"] == "incremental_unconfirmed"
    assert row["p75_high_level_cells_of_6"] == 6
    assert row["p75_full_pass_cells_of_6"] == 0


def test_cross_period_correlations_match_cells_and_preserve_rank() -> None:
    rows = []
    for tier in diagnostics.TIERS:
        for period, scale in (("2024_oof", 1.0), ("2025", 2.0), ("2023", 3.0)):
            for index, cycle_id in enumerate(("cycle_a", "cycle_b", "cycle_c"), 1):
                value = scale * index
                rows.append(
                    {
                        "period": period,
                        "cycle_id": cycle_id,
                        "target": "absolute_return_bps",
                        "horizon": 6,
                        "tier": tier,
                        "observed_rate": value,
                        "mean_qcycle_probability": value + 1.0,
                        "observed_rate_divided_by_mean_qcontext": value + 2.0,
                        "daily_residual_ci_low": value + 3.0,
                        "conditional_relative_log_loss_improvement": value + 4.0,
                        "joint_relative_log_loss_improvement": value + 5.0,
                    }
                )
    result = diagnostics.correlation_table(pd.DataFrame(rows))
    observed = result.loc[
        result["metric"].eq("observed_rate") & result["tier"].eq("p75")
    ]
    assert observed["matched_cells"].eq(3).all()
    assert np.allclose(observed["pearson"], 1.0)
    assert np.allclose(observed["spearman"], 1.0)


def test_root_guard_allows_only_sealed_input_and_dedicated_private_tmp_output() -> None:
    with pytest.raises(ValueError):
        diagnostics.validate_roots(
            Path("/private/tmp/not_the_frozen_inputs"),
            Path("/private/tmp/stocker_loop_quality_failure_diagnostics_test"),
        )
    with pytest.raises(ValueError):
        diagnostics.validate_roots(
            diagnostics.DEFAULT_ARTIFACT_ROOT,
            Path("/tmp/not_a_dedicated_diagnostic_name"),
        )


@pytest.mark.skipif(
    not diagnostics.DEFAULT_ARTIFACT_ROOT.is_dir(),
    reason="ephemeral frozen diagnostic inputs are unavailable",
)
def test_frozen_end_to_end_is_read_only_and_preserves_all_grades() -> None:
    output = Path("/private/tmp/stocker_loop_quality_failure_diagnostics_pytest")
    before = diagnostics.input_hashes(diagnostics.DEFAULT_ARTIFACT_ROOT)
    summary = diagnostics.run(diagnostics.DEFAULT_ARTIFACT_ROOT, output)
    after = diagnostics.input_hashes(diagnostics.DEFAULT_ARTIFACT_ROOT)

    assert before == after
    assert summary["research_only"] is True
    assert summary["live_ordering_enabled"] is False
    assert summary["order_placement"] == "disabled"
    assert summary["model_refit_performed"] is False
    assert summary["frozen_grade_changed"] is False
    assert summary["frozen_final_grade_counts"] == {"unqualified": 20}
    selected = summary["selective_diagnostics"]
    assert selected["exploratory_absolute_high_candidates"] == [
        "cycle_07",
        "cycle_13",
    ]
    assert selected[
        "exploratory_absolute_high_candidates_with_all_period_structural_reliability"
    ] == ["cycle_07"]
    assert selected["strongest_incremental_candidate"] == "cycle_09"
    assert json.loads((output / "summary.json").read_text()) == summary


def test_source_has_no_fitting_or_shadow_runtime_dependency() -> None:
    source = MODULE_PATH.read_text()
    assert "sklearn" not in source
    assert "LogisticRegression" not in source
    assert "shadow_validation" not in source
    assert ".fit(" not in source
