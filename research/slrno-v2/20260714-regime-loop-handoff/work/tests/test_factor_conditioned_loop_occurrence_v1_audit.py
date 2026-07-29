from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "audit_factor_conditioned_loop_occurrence_v1.py"
)
SPEC = importlib.util.spec_from_file_location("factor_occurrence_independent_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_contract_hash_safety_and_data_free_self_test() -> None:
    contract = audit.load_contract()
    assert audit.sha256(audit.CONTRACT_PATH) == audit.CONTRACT_SHA256
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    result = audit.data_free_self_test()
    assert result["passed"] == result["total"] == 7
    assert result["later_period_paths_resolved"] is False


def test_auditor_never_imports_production_modules_or_exposes_later_paths() -> None:
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)
    imports = [
        ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    forbidden_imports = (
        "factor_conditioned_loop_occurrence_core",
        "factor_conditioned_loop_occurrence_eval",
        "run_factor_conditioned_loop_occurrence_v1",
    )
    assert not any(any(name in text for name in forbidden_imports) for text in imports)
    assert "test_2025_filtered_runs" not in source
    assert "backward_2023_filtered_runs" not in source
    assert "stocker_eodhd_pre2024" not in source
    assert "pre_score_authorization.json\", result" not in source


def _cycles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cycle_index": 0,
                "cycle_id": "cycle_01",
                "cycle": "1->3->1",
                "transition_length": 2,
                "core": (1, 3),
            }
        ]
    )


def _anchors(states: list[int]) -> pd.DataFrame:
    count = len(states)
    frame = pd.DataFrame(
        {
            "anchor_id": np.arange(count),
            "period": "2024",
            "symbol_norm": "AAA",
            "session_date": "2024-07-01",
            "start_timestamp": pd.date_range("2024-07-01 13:30:00Z", periods=count, freq="5min"),
            "month": "2024-07",
            "quarter": "2024_q3",
            "state": states,
            "bar_ordinal": np.arange(count),
            "b0_unknown": False,
            "entry_minutes": np.arange(count) * 5.0,
            "b0_entry_numeric": 0.0,
            "b0_entry_high_stress": 0.0,
            "entry_time_sin": 0.0,
            "entry_time_cos": 1.0,
            "current_bar_log_return": 0.001,
            "return_sum_6": 0.002,
            "mean_abs_return_12": 0.003,
            "session_return": 0.004,
            "bar_range_pct": 0.005,
        }
    )
    series = pd.Series(states)
    frame["previous_state_1"] = series.shift(1).fillna(8).astype(int)
    frame["previous_state_2"] = series.shift(2).fillna(8).astype(int)
    frame["next_outcome"] = series.shift(-1).fillna(8).astype(int)
    frame["terminal"] = frame.index == count - 1
    for step in range(1, 5):
        frame[f"future_state_{step}"] = series.shift(-step).fillna(8).astype(int)
    frame["history_token"] = audit.history_token(
        frame["previous_state_2"], frame["previous_state_1"], frame["state"]
    )
    return frame


def test_overlapping_label_logic_keeps_terminal_zero_rows() -> None:
    anchors = _anchors([1, 3, 1, 3, 1])
    cycles = _cycles()
    routes = audit.build_routes
    original = audit.ROUTE_COUNT
    original_width = audit.ROUTE_CONTRAST_WIDTH
    try:
        audit.ROUTE_COUNT = 2
        audit.ROUTE_CONTRAST_WIDTH = 1
        expanded = audit.expand_labels(anchors, cycles)
    finally:
        audit.ROUTE_COUNT = original
        audit.ROUTE_CONTRAST_WIDTH = original_width
    assert expanded.loc[expanded["anchor_id"].eq(0), "target"].iloc[0] == 1
    terminal = expanded.loc[expanded["terminal"]]
    assert len(terminal) == 1 and terminal["target"].sum() == 0
    assert np.allclose(expanded.groupby("anchor_id")["inverse_compatible_weight"].sum(), 1.0)
    assert routes is audit.build_routes


def test_history_token_formula_and_boundaries() -> None:
    np.testing.assert_array_equal(audit.history_token([8, 0], [8, 1], [0, 7]), [640, 15])
    assert audit.history_token([8], [8], [7])[0] == 647
    with pytest.raises(AssertionError):
        audit.history_token([9], [0], [0])


def test_penalty_design_layout_and_embedding_are_exact() -> None:
    assert audit.factor_layout(0)["width"] == 44
    assert audit.factor_layout(4)["width"] == 2812
    assert audit.factor_layout(9)["width"] == 6272
    assert audit.penalties(0)[0] == 0.0
    assert set(np.unique(audit.penalties(9))) == {0.0, 1.0, 4.0, 8.0, 32.0}
    rng = np.random.default_rng(4)
    pattern = rng.normal(size=44)
    embedded = audit.embed(pattern, 0, 9)
    np.testing.assert_array_equal(embedded[:44], pattern)
    assert np.count_nonzero(embedded[44:]) == 0


def test_offset_ridge_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(8)
    rows = 35
    matrix = sparse.csr_matrix(rng.normal(size=(rows, 44)))
    target = rng.integers(0, 2, size=rows)
    offset = rng.normal(scale=0.1, size=rows)
    beta = rng.normal(scale=0.03, size=44)
    value, gradient = audit.objective_gradient(
        beta, matrix, target, offset, 0.001, audit.penalties(0)
    )
    assert np.isfinite(value) and np.isfinite(gradient).all()
    epsilon = 1e-6
    for column in (0, 1, 11, 43):
        step = np.zeros(44)
        step[column] = epsilon
        high = audit.objective_gradient(
            beta + step, matrix, target, offset, 0.001, audit.penalties(0)
        )[0]
        low = audit.objective_gradient(
            beta - step, matrix, target, offset, 0.001, audit.penalties(0)
        )[0]
        assert np.isclose(gradient[column], (high - low) / (2 * epsilon), atol=2e-8)


def test_lambda_selection_equal_weights_months_and_breaks_ties_largest() -> None:
    months = ("2024-04", "2024-05")
    rows = []
    for head in audit.HEADS:
        for value in audit.RIDGE_GRID:
            for month in months:
                rows.append(
                    {
                        "head": head,
                        "lambda": value,
                        "validation_month": month,
                        "log_loss": 1.0 if value in (0.001, 0.003) else 2.0,
                    }
                )
    selected, diagnostics = audit.select_lambdas(pd.DataFrame(rows), months)
    assert selected == {head: 0.003 for head in audit.HEADS}
    assert diagnostics.groupby("head")["selected"].sum().eq(1).all()


def test_calibration_probability_one_enters_last_bin() -> None:
    frame = pd.DataFrame({"target": np.r_[np.zeros(500), np.ones(500)], "p": np.r_[np.zeros(500), np.ones(500)]})
    result = audit.calibration(frame, "p")
    rows = result["rows"].set_index("bin")
    assert rows.loc[0, "rows"] == 500
    assert rows.loc[9, "rows"] == 500
    assert result["maximum_supported_bin_error"] == 0.0


def test_common_bootstrap_and_holm_are_deterministic() -> None:
    first = audit.common_block_positions(7, 20260711, draws=5)
    second = audit.common_block_positions(7, 20260711, draws=5)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 7)
    holm = audit.holm_payload(
        {
            "unweighted_log_loss": 0.001,
            "inverse_compatible_weighted_log_loss": 0.002,
            "top_three_recall": 0.003,
        }
    )
    assert holm["pass"] is True


def test_nested_comparison_distinguishes_boolean_and_number() -> None:
    assert audit.nested_differences({"x": True}, {"x": 1})
    assert not audit.nested_differences({"x": 1.0}, {"x": 1.0 + 1e-12})
    assert audit.nested_differences({"x": 1.0}, {"x": 1.1})


def test_parser_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([])
    assert audit.parse_args(["--self-test-only"]).self_test_only
    assert audit.parse_args(["--audit-rejection"]).audit_rejection


def test_rejection_audit_has_no_authorization_writer() -> None:
    source = MODULE_PATH.read_text()
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "audit_rejection"
    )
    body = ast.get_source_segment(source, function) or ""
    assert "pre_score_authorization.json" in body
    assert "write_json(output, result)" in body
    assert "write_json(root / \"pre_score_authorization.json\"" not in body
    assert '"scoring_authorized": False' in body

