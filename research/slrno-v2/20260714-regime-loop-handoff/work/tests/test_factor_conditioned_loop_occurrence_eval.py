from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "factor_conditioned_loop_occurrence_eval.py"
)
SPEC = importlib.util.spec_from_file_location("factor_occurrence_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def _prediction_panel(
    *, sessions: int = 8, anchors_per_session: int = 2, cycles: int = 4
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    anchor_id = 0
    for session in range(sessions):
        date = f"2024-07-{session + 1:02d}"
        for local_anchor in range(anchors_per_session):
            positive_cycle = (session + local_anchor) % cycles
            for cycle in range(cycles):
                target = int(cycle == positive_cycle)
                row: dict[str, object] = {
                    "anchor_id": anchor_id,
                    "session_date": date,
                    "start_timestamp": f"{date}T14:{30 + local_anchor:02d}:00Z",
                    "symbol_norm": "AAA",
                    "quarter": "2024_q3",
                    "state": local_anchor,
                    "cycle_id": f"cycle_{cycle + 1:02d}",
                    "transition_length": 2 + cycle % 3,
                    "target": target,
                    "n_compatible": cycles,
                    "bar_ordinal": local_anchor,
                    "terminal": False,
                    "qhistory": 0.55 if target else 0.25,
                    "qold_limited_path": 0.58 if target else 0.23,
                    "qpattern": 0.62 if target else 0.20,
                    "qlimited4": 0.68 if target else 0.16,
                    "qfull9": 0.88 if target else 0.05,
                }
                for factor in evaluation.NEW_FACTOR_COLUMNS:
                    row[f"factor_quartile__{factor}"] = anchor_id % 4
                row["b0_entry_high_stress"] = anchor_id % 2
                row["b0_unknown"] = False
                row["entry_clock_quartile"] = anchor_id % 4
                rows.append(row)
            anchor_id += 1
    return pd.DataFrame(rows)


def _passing_gate_artifacts() -> dict[str, object]:
    comparison_rows: list[dict[str, object]] = []
    for surface in ("unweighted", "inverse_compatible"):
        for baseline in (*evaluation.PRIMARY_BASELINES, evaluation.LINEAGE_BASELINE):
            comparison_rows.append(
                {
                    "surface": surface,
                    "baseline": baseline,
                    "relative_log_loss_improvement": 0.02,
                    "log_loss_difference": -0.01,
                    "brier_difference": -0.01,
                }
            )
    ranking = pd.DataFrame(
        [
            {
                "model": "qhistory",
                "recall": 0.89,
                "precision": 0.19,
                "positive_anchor_hit_rate": 0.89,
            },
            {
                "model": "qpattern",
                "recall": 0.897,
                "precision": 0.19,
                "positive_anchor_hit_rate": 0.90,
            },
            {
                "model": "qlimited4",
                "recall": 0.897,
                "precision": 0.19,
                "positive_anchor_hit_rate": 0.90,
            },
            {
                "model": "qold_limited_path",
                "recall": 0.90,
                "precision": 0.19,
                "positive_anchor_hit_rate": 0.90,
            },
            {
                "model": "qfull9",
                "recall": 0.90,
                "precision": 0.20,
                "positive_anchor_hit_rate": 0.91,
            },
        ]
    )
    calibration = {
        "summary": pd.DataFrame(
            [
                {
                    "model": model,
                    "ece": 0.01 if model == "qfull9" else 0.02,
                    "maximum_supported_bin_error": 0.01,
                    "has_supported_bin": True,
                }
                for model in (*evaluation.PRIMARY_BASELINES, "qfull9")
            ]
        )
    }
    bootstrap = {
        "rows": pd.DataFrame(
            [
                {"baseline": baseline, "loss": loss, "pass": True}
                for baseline in evaluation.PRIMARY_BASELINES
                for loss in evaluation.LOSS_NAMES
            ]
        ),
        "pass": True,
    }
    slice_rows: list[dict[str, object]] = []

    def add(
        family: str,
        values: list[str],
        baselines: tuple[str, ...],
        losses: tuple[str, ...],
    ) -> None:
        for value in values:
            for baseline in baselines:
                for loss in losses:
                    slice_rows.append(
                        {
                            "family": family,
                            "value": value,
                            "baseline": baseline,
                            "loss": loss,
                            "supported": True,
                            "pass": True,
                        }
                    )

    add(
        "time",
        [f"2024-{month:02d}" for month in range(7, 13)],
        evaluation.PRIMARY_BASELINES,
        evaluation.LOSS_NAMES,
    )
    for family in (
        "leave_one_stock_out",
        "current_state",
        "transition_length",
        "nonterminal",
        "early_entry",
    ):
        add(family, ["x"], evaluation.PRIMARY_BASELINES, evaluation.LOSS_NAMES)
    add(
        "cycle",
        [f"cycle_{index:02d}" for index in range(1, 16)],
        evaluation.PRIMARY_BASELINES,
        ("log_loss",),
    )
    add(
        "cycle_current_state_orientation",
        ["cycle_01__s0"],
        ("qlimited4",),
        evaluation.LOSS_NAMES,
    )
    for column in evaluation.FACTOR_QUARTILE_COLUMNS:
        add(column, ["0"], ("qlimited4",), evaluation.LOSS_NAMES)
    return {
        "support": {"pass": True},
        "comparisons": pd.DataFrame(comparison_rows),
        "ranking": ranking,
        "calibration": calibration,
        "bootstrap": bootstrap,
        "slices": pd.DataFrame(slice_rows),
        "falsification": {"pass": True},
    }


def test_contract_identity_safety_and_self_tests() -> None:
    assert evaluation.CONTRACT_SHA256 == (
        "ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d"
    )
    assert evaluation.RESEARCH_ONLY is True
    assert evaluation.LIVE_ORDERING_ENABLED is False
    assert evaluation.ORDER_PLACEMENT == "disabled"
    evaluation.self_tests()


def test_binary_losses_and_inverse_compatible_weighting_are_exact() -> None:
    target = np.asarray([0, 1])
    probability = np.asarray([0.25, 0.75])
    losses = evaluation.binary_loss_arrays(target, probability)
    np.testing.assert_allclose(losses["log_loss"], -np.log(0.75), rtol=0, atol=1e-15)
    np.testing.assert_allclose(losses["brier"], 0.0625, rtol=0, atol=0)
    counts = np.asarray([1, 4])
    metrics = evaluation.loss_metrics(target, probability, compatible_counts=counts)
    expected = (1.0 * losses["log_loss"][0] + 0.25 * losses["log_loss"][1]) / 1.25
    assert metrics["log_loss"] == expected
    np.testing.assert_array_equal(
        evaluation.inverse_compatible_weights(counts), np.asarray([1.0, 0.25])
    )


def test_probability_and_weight_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        evaluation.binary_loss_arrays([0, 1], [0.2, 1.1])
    with pytest.raises(ValueError):
        evaluation.inverse_compatible_weights([1, 0])
    with pytest.raises(ValueError):
        evaluation.weighted_mean([1, 2], [0, 0])


def test_top_three_uses_cycle_id_tie_break_and_overlapping_labels() -> None:
    frame = pd.DataFrame(
        {
            "anchor_id": [1, 1, 1, 1, 2, 2],
            "cycle_id": ["c1", "c2", "c3", "c4", "c1", "c2"],
            "target": [1, 0, 0, 1, 0, 1],
            "probability": [0.5, 0.5, 0.5, 0.5, 0.2, 0.8],
        }
    )
    result = evaluation.top_three_metrics(frame, "probability")
    assert result["hits"] == 2
    assert result["positive_labels"] == 3
    assert result["selected_labels"] == 5
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["precision"] == pytest.approx(2 / 5)
    assert result["positive_anchor_hit_rate"] == 1.0
    assert result["top_one_recall"] == pytest.approx(2 / 3)
    assert result["mean_reciprocal_rank"] == 1.0


def test_calibration_places_probability_one_in_bin_nine_and_fails_without_support() -> None:
    target = np.asarray([0, 0, 1, 1, 1])
    probability = np.asarray([0.0, 0.09, 0.1, 0.99, 1.0])
    result = evaluation.fixed_bin_calibration(target, probability, minimum_rows=1)
    assert result["rows"].loc[0, "rows"] == 2
    assert result["rows"].loc[1, "rows"] == 1
    assert result["rows"].loc[9, "rows"] == 2
    assert result["has_supported_bin"] is True
    unsupported = evaluation.fixed_bin_calibration(target, probability, minimum_rows=10)
    assert unsupported["has_supported_bin"] is False
    assert math.isnan(unsupported["maximum_supported_bin_error"])


def test_lambda_month_losses_equal_weight_months_after_row_means() -> None:
    frame = pd.DataFrame(
        {
            "validation_month": ["2024-04", "2024-04", "2024-05"],
            "target": [0, 1, 1],
            "probability": [0.1, 0.8, 0.7],
        }
    )
    result = evaluation.lambda_month_log_losses(frame)
    assert list(result) == ["2024-04", "2024-05"]
    april = evaluation.binary_loss_arrays([0, 1], [0.1, 0.8])["log_loss"].mean()
    assert result["2024-04"] == april


def test_support_summary_and_population_validation() -> None:
    anchors = pd.DataFrame({"anchor_id": [1, 2, 3]})
    expanded = pd.DataFrame(
        {
            "anchor_id": [1, 1, 2, 2, 3, 3],
            "cycle_id": ["c1", "c2"] * 3,
            "target": [1, 0, 0, 1, 0, 0],
            "n_compatible": [2] * 6,
        }
    )
    contract = {
        "frozen_sources": {
            "runs_2024": {"rows": 3},
            "cycles": {"count": 2},
        },
        "population_and_target": {
            "compatible_anchor_cycle_rows_expected": {"2024": 6},
            "positive_rows_expected": {"2024": 2},
        },
    }
    observed = evaluation.validate_population(
        anchors, expanded, ["c1", "c2"], "2024", contract
    )
    assert observed["pass"] is True
    support_frame = expanded.assign(
        symbol_norm="AAA", quarter="2024_q3", state=[0, 0, 1, 1, 2, 2]
    )
    rule = {
        "minimum_compatible_rows": 6,
        "minimum_positive_rows": 2,
        "cycles": 2,
        "minimum_positive_rows_per_cycle": 1,
        "minimum_stocks": 1,
        "quarters": 1,
        "current_states": 3,
    }
    assert evaluation.support_summary(support_frame, rule)["pass"] is True


def test_terminal_labels_and_compatible_counts_are_validated() -> None:
    panel = _prediction_panel(sessions=2)
    evaluation.validate_prediction_panel(panel)
    broken = panel.copy()
    broken.loc[broken["anchor_id"].eq(0), "terminal"] = True
    with pytest.raises(ValueError, match="terminal"):
        evaluation.validate_prediction_panel(broken)
    broken = panel.copy()
    broken.loc[0, "n_compatible"] = 99
    with pytest.raises(ValueError, match="n_compatible"):
        evaluation.validate_prediction_panel(broken)


def test_slice_diagnostics_include_required_and_report_only_families() -> None:
    panel = _prediction_panel(sessions=2)
    terminal_anchor = panel["anchor_id"].max()
    mask = panel["anchor_id"].eq(terminal_anchor)
    panel.loc[mask, "terminal"] = True
    panel.loc[mask, "target"] = 0
    result = evaluation.slice_diagnostics(panel, "2024")
    families = set(result["family"])
    assert set(evaluation.FACTOR_QUARTILE_COLUMNS).issubset(families)
    assert set(evaluation.REPORT_SLICE_COLUMNS).issubset(families)
    assert {"terminal", "nonterminal", "early_entry"}.issubset(families)
    assert result.loc[result["family"].eq("terminal"), "gate_required"].eq(False).all()


def test_circular_five_session_bootstrap_is_common_and_deterministic() -> None:
    first = evaluation.common_block_positions(7, seed=20260711, draws=20)
    second = evaluation.common_block_positions(7, seed=20260711, draws=20)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 7)
    assert first.min() >= 0 and first.max() < 7
    panel = _prediction_panel(sessions=8)
    result = evaluation.familywise_bootstrap(panel, seed=20260711, draws=99)
    assert result["bootstrap_means"].shape == (99, 6)
    assert result["rows"]["pass"].all()
    assert result["pass"] is True


def test_holm_step_down_uses_declared_stable_family() -> None:
    p_values = {
        "unweighted_log_loss": 0.001,
        "inverse_compatible_weighted_log_loss": 0.002,
        "top_three_recall": 0.009,
    }
    result = evaluation.holm_step_down(p_values)
    assert result["pass"] is True
    failing = dict(p_values, unweighted_log_loss=0.006)
    assert evaluation.holm_step_down(failing)["pass"] is False


def test_left_rotation_preserves_unequal_whole_session_blocks() -> None:
    blocks = [np.asarray([0, 1]), np.asarray([2, 3, 4]), np.asarray([5])]
    np.testing.assert_array_equal(
        evaluation.left_rotate_session_blocks(blocks, 1),
        np.asarray([2, 3, 4, 5, 0, 1]),
    )
    np.testing.assert_array_equal(
        evaluation.left_rotate_session_blocks(blocks, 2),
        np.asarray([5, 0, 1, 2, 3, 4]),
    )


def test_falsification_is_coherent_and_deterministic() -> None:
    panel = _prediction_panel(sessions=8, anchors_per_session=2, cycles=4)
    first = evaluation.falsification_diagnostics(panel, draws=19, seed=20260711)
    second = evaluation.falsification_diagnostics(panel, draws=19, seed=20260711)
    np.testing.assert_array_equal(first["null_statistics"], second["null_statistics"])
    assert first["null_statistics"].shape == (19, 3)
    assert first["eligible_anchors"] == panel["anchor_id"].nunique()
    assert first["full_logit_replay_max_error"] <= 1e-12
    assert first["statistic_replay_max_error"] <= 1e-12
    assert list(first["statistics"]["name"]) == [
        "unweighted_log_loss",
        "inverse_compatible_weighted_log_loss",
        "top_three_recall",
    ]


def test_primary_gate_derivation_passes_complete_artifacts() -> None:
    payload = _passing_gate_artifacts()
    result = evaluation.evaluate_primary_gates(
        pd.DataFrame(),
        period="2024",
        support=payload["support"],
        comparisons=payload["comparisons"],
        ranking=payload["ranking"],
        calibration=payload["calibration"],
        bootstrap=payload["bootstrap"],
        slices=payload["slices"],
        falsification=payload["falsification"],
    )
    assert result["pooled_primary_pass"] is True
    assert result["primary_pass"] is True
    payload["calibration"]["summary"].loc[
        payload["calibration"]["summary"]["model"].eq("qfull9"), "ece"
    ] = 0.5
    failed = evaluation.evaluate_primary_gates(
        pd.DataFrame(),
        period="2024",
        support=payload["support"],
        comparisons=payload["comparisons"],
        ranking=payload["ranking"],
        calibration=payload["calibration"],
        bootstrap=payload["bootstrap"],
        slices=payload["slices"],
        falsification=payload["falsification"],
    )
    assert failed["primary_pass"] is False


def test_period_evaluation_returns_semantic_artifacts_without_fitting() -> None:
    panel = _prediction_panel(sessions=8)
    result = evaluation.evaluate_period(
        panel,
        "2024",
        include_falsification=False,
        bootstrap_draws=19,
    )
    assert result["period"] == "2024"
    assert result["primary_pass"] is False
    assert set(result["artifacts"]) == {
        "support",
        "overall",
        "ranking",
        "calibration",
        "comparisons",
        "bootstrap",
        "slices",
        "falsification",
        "gates",
    }


def test_decision_is_development_and_demotion_only() -> None:
    rejected = evaluation.derive_decision({"primary_pass": False})
    assert rejected["label"] == (
        "factor_conditioned_loop_occurrence_rejected_2024_"
        "and_do_not_score_later_periods"
    )
    candidate = evaluation.derive_decision({"primary_pass": True})
    assert candidate["label"] == "factor_conditioned_loop_occurrence_development_candidate"
    assert candidate["later_scoring_eligible_after_independent_audit"] is True
    assert candidate["later_scoring_authorized"] is False
    retained = evaluation.derive_decision(
        {"primary_pass": True}, {"primary_pass": True}, {"primary_pass": True}
    )
    assert retained["label"] == "development_candidate_retained_pending_prospective"
    demoted = evaluation.derive_decision(
        {"primary_pass": True}, {"primary_pass": True}, {"primary_pass": False}
    )
    assert demoted["label"] == "development_algorithm_unconfirmed"
    assert demoted["later_periods_can_promote"] is False


def _irregular_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    anchor = 0
    cycle_counts = [6] * 82 + [5] * 11
    for count in cycle_counts:
        symbol = evaluation.IRREGULAR_SYMBOLS[anchor % 5]
        for cycle in range(count):
            row: dict[str, object] = {
                    "anchor_id": anchor,
                    "session_date": evaluation.IRREGULAR_DATE,
                    "start_timestamp": f"2025-04-10T14:{anchor % 60:02d}:00Z",
                    "symbol_norm": symbol,
                    "quarter": "2025_q2",
                    "state": anchor % 8,
                    "cycle_id": f"cycle_{cycle + 1:02d}",
                    "transition_length": 2 + cycle % 3,
                    "target": 0,
                    "n_compatible": count,
                    "bar_ordinal": anchor % 78,
                    "terminal": False,
                    "qhistory": 0.12,
                    "qold_limited_path": 0.13,
                    "qpattern": 0.11,
                    "qlimited4": 0.10,
                    "qfull9": 0.09,
                    "b0_entry_high_stress": anchor % 2,
                    "b0_unknown": False,
                    "entry_clock_quartile": anchor % 4,
                }
            for factor in evaluation.NEW_FACTOR_COLUMNS:
                row[f"factor_quartile__{factor}"] = anchor % 4
            rows.append(row)
        anchor += 1
    # Retain a small unaffected panel so deletion evaluation remains defined.
    for unaffected in range(4):
        for cycle in range(4):
            target = int(cycle == unaffected)
            row = {
                    "anchor_id": anchor,
                    "session_date": f"2025-04-{11 + unaffected:02d}",
                    "start_timestamp": f"2025-04-{11 + unaffected:02d}T14:30:00Z",
                    "symbol_norm": "OTHER",
                    "quarter": "2025_q2",
                    "state": unaffected,
                    "cycle_id": f"cycle_{cycle + 1:02d}",
                    "transition_length": 2 + cycle % 3,
                    "target": target,
                    "n_compatible": 4,
                    "bar_ordinal": 0,
                    "terminal": False,
                    "qhistory": 0.55 if target else 0.25,
                    "qold_limited_path": 0.58 if target else 0.23,
                    "qpattern": 0.62 if target else 0.20,
                    "qlimited4": 0.68 if target else 0.16,
                    "qfull9": 0.88 if target else 0.05,
                    "b0_entry_high_stress": unaffected % 2,
                    "b0_unknown": False,
                    "entry_clock_quartile": unaffected % 4,
                }
            for factor in evaluation.NEW_FACTOR_COLUMNS:
                row[f"factor_quartile__{factor}"] = unaffected % 4
            rows.append(row)
        anchor += 1
    return pd.DataFrame(rows)


def test_irregular_deletion_is_exact_and_never_refits() -> None:
    panel = _irregular_panel()
    original = {"artifacts": {"gates": {"pooled_primary_pass": True}}}
    result = evaluation.evaluate_irregular_deletion(
        panel,
        original_evaluation=original,
        bootstrap_draws=19,
    )
    assert result["deleted_runs"] == 93
    assert result["deleted_compatible_rows"] == 547
    assert result["symbols"] == list(evaluation.IRREGULAR_SYMBOLS)
    assert result["no_refit"] is True
    assert result["count_pass"] is True
    assert result["pass"] is False  # tiny synthetic remainder cannot pass support


def test_module_has_no_source_or_outcome_io_surface() -> None:
    source = MODULE_PATH.read_text()
    assert "read_parquet" not in source
    assert "read_csv" not in source
    assert "Path(" not in source
