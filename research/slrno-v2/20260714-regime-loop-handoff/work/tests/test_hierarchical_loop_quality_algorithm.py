from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_hierarchical_loop_quality_algorithm_v1.py"
)
SPEC = importlib.util.spec_from_file_location("hierarchical_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
algorithm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(algorithm)


def test_contract_hash_grid_and_safety_are_frozen() -> None:
    assert algorithm.sha256(algorithm.CONTRACT) == algorithm.CONTRACT_SHA256
    contract = json.loads(algorithm.CONTRACT.read_text())
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert tuple(map(tuple, contract["scale_grid"]["pairs"])) == algorithm.SCALE_GRID
    assert len(algorithm.SCALE_GRID) == 15


def test_json_safety_recurses_through_numpy_arrays() -> None:
    payload = {
        "daily": np.asarray([1.0, np.nan]),
        "matrix": np.asarray([[np.int64(2)], [np.int64(3)]]),
    }
    converted = algorithm.safe(payload)
    assert converted == {"daily": [1.0, None], "matrix": [[2], [3]]}
    assert json.loads(json.dumps(converted)) == converted


def test_nested_schedule_and_tie_break_are_exact() -> None:
    assert tuple(algorithm.INNER_SCHEDULE) == algorithm.OUTER_MONTHS
    assert algorithm.INNER_SCHEDULE["2024-07"] == (
        "2024-04",
        "2024-05",
        "2024-06",
    )
    objectives = {pair: 1.0 for pair in algorithm.SCALE_GRID}
    objectives[(0.25, 0.125)] = 0.5
    objectives[(0.125, 0.125)] = 0.5 + 5e-7
    objectives[(0.125, 0.0625)] = 0.5 + 9e-7
    assert algorithm.choose_scale_pair(objectives) == (0.125, 0.0625)


def test_weighted_cycle_and_within_cycle_route_centering() -> None:
    mapping = algorithm.route_mapping()
    # Every route appears, so all twenty cycles have positive weight.
    frame = mapping[["route_index", "cycle_index"]].copy()
    frame["conditional_weight"] = np.linspace(1.0, 2.0, len(frame))
    mu_cycle, mu_route = algorithm.weighted_centers(
        frame["cycle_index"],
        frame["route_index"],
        frame["conditional_weight"],
        mapping,
    )
    assert np.isclose(mu_cycle.sum(), 1.0)
    route_cycle = mapping.sort_values("route_index")["cycle_index"].to_numpy(int)
    for cycle in range(algorithm.CYCLE_WIDTH):
        assert np.isclose(mu_route[route_cycle == cycle].sum(), 1.0)
    cycle_block, route_block = algorithm.centered_blocks(
        frame,
        mapping,
        mu_cycle,
        mu_route,
        (0.5, 0.25),
    )
    weights = frame["conditional_weight"].to_numpy(float)
    assert np.allclose(weights @ cycle_block.toarray(), 0.0, atol=1e-12)
    route_values = route_block.toarray()
    for cycle in range(algorithm.CYCLE_WIDTH):
        rows = frame["cycle_index"].eq(cycle).to_numpy()
        inside = route_cycle == cycle
        outside = ~inside
        assert np.all(route_values[np.ix_(rows, outside)] == 0.0)
        assert np.allclose(weights[rows] @ route_values[np.ix_(rows, inside)], 0.0, atol=1e-12)


def test_nonzero_hierarchy_width_and_zero_endpoint_width() -> None:
    mapping = algorithm.route_mapping()
    frame = mapping.iloc[:4][["route_index", "cycle_index", "current_state"]].rename(
        columns={"current_state": "state"}
    )
    for column in algorithm.v3.TOPOLOGY_COLUMNS:
        frame[column] = 0.0
    context = sparse.csr_matrix(np.ones((len(frame), algorithm.CONTEXT_WIDTH)))
    # Use full-map centers so every cycle has a defined route distribution.
    full = mapping[["route_index", "cycle_index"]].copy()
    weights = np.ones(len(full))
    mu_cycle, mu_route = algorithm.weighted_centers(
        full["cycle_index"], full["route_index"], weights, mapping
    )
    assert algorithm.hierarchy_matrix(
        frame, context, mapping, mu_cycle, mu_route, (0.0, 0.0)
    ).shape == (len(frame), 80)
    assert algorithm.hierarchy_matrix(
        frame, context, mapping, mu_cycle, mu_route, (0.25, 0.125)
    ).shape == (len(frame), 144)


def test_common_block_matrix_is_exact_and_deterministic() -> None:
    first = algorithm.common_block_positions(13, draws=20)
    second = algorithm.common_block_positions(13, draws=20)
    assert np.array_equal(first, second)
    assert first.shape == (20, 13)
    assert first.min() >= 0 and first.max() < 13
    values = np.arange(13, dtype=float)
    means = algorithm.bootstrap_means(values, first)
    assert np.isclose(means[0, 0], values[first[0]].mean())


def test_holm_step_down_and_stable_tie_order() -> None:
    frame = pd.DataFrame(
        {
            "cycle_index": [1, 0, 2],
            "horizon": [6, 6, 6],
            "p": [0.001, 0.001, 0.03],
        }
    )
    rejected = algorithm.holm_rejections(frame, "p", alpha=0.025)
    assert rejected.tolist() == [True, True, False]


def test_scoring_lock_requires_fit_and_independent_authorization(tmp_path: Path) -> None:
    original = algorithm.OUT
    try:
        algorithm.OUT = tmp_path
        with pytest.raises(FileNotFoundError):
            algorithm.validate_fit_and_pre_score_lock()
    finally:
        algorithm.OUT = original


def test_fit_artifact_hashes_exclude_independent_audit() -> None:
    names = algorithm.fit_artifact_names()
    assert "pre_score_audit.json" not in names
    assert "fit_source_hashes.json" in names
    assert "inner_selection_2024.csv" in names
    assert "outer_fold_audit_2024.csv" in names


def test_fit_and_scoring_entrypoints_are_separate() -> None:
    fit_source = inspect.getsource(algorithm.run_fit_only)
    assert "scoring_sources_after_lock" not in fit_source
    assert "run_scoring" not in fit_source
    score_source = inspect.getsource(algorithm.run_scoring)
    assert "validate_fit_and_pre_score_lock()" in score_source
    source = MODULE_PATH.read_text()
    assert "shadow_validation" not in source


def test_validate_only_reconstructs_2024_without_fit() -> None:
    result = algorithm.validate_only()
    assert result["status"] == "validated_without_fit"
    assert result["contract_sha256"] == algorithm.CONTRACT_SHA256
    assert result["training_rows"] == 32677
    assert result["oof_rows"] == 216438
    assert result["route_mapping_rows"] == 44
    assert result["support"]["support_pass"] is True
    assert result["later_period_panels_read"] is False
