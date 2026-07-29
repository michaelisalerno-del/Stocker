from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


WORKSPACE = Path(__file__).resolve().parents[2]
AUDIT_PATH = WORKSPACE / "work/audit_hierarchical_loop_quality_algorithm_v1.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("hierarchical_v1_audit", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_frame(module, repeats: int = 3):
    mapping, vectors = module.build_rotation_mapping()
    rows = []
    for repetition in range(repeats):
        for route in mapping.itertuples(index=False):
            row = {
                "anchor_id": repetition * 1000 + int(route.route_index),
                "cycle_id": str(route.cycle_id),
                "cycle_index": int(route.cycle_index),
                "state": int(route.current_state),
                "route_index": int(route.route_index),
                "conditional_weight": 1.0 + 0.1 * repetition,
                "session_date": f"2024-0{repetition + 1}-02",
                "month_key": f"2024-0{repetition + 1}",
            }
            for name in module.NUMERIC_CONTROLS:
                row[name] = 0.0
            for name, value in zip(
                module.topology_column_names(),
                vectors[int(route.route_index)],
                strict=True,
            ):
                row[name] = float(value)
            for target in module.TARGETS:
                for horizon in module.HORIZONS:
                    row[f"quality_class__{target}__h{horizon}"] = repetition
            rows.append(row)
    return pd.DataFrame(rows), mapping


def test_contract_hash_and_semantics_are_exact():
    module = load_audit_module()
    audit = module.Audit()
    contract = module.verify_contract_semantics(audit)
    assert audit.all_passed
    assert module.sha256(module.CONTRACT_PATH) == module.CONTRACT_SHA256
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"


def test_auditor_has_no_production_runner_import():
    source = AUDIT_PATH.read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith("run_") for name in imported)
    assert "from work.run_hierarchical" not in source
    assert "import work.run_hierarchical" not in source


def test_literal_grid_count_constraints_and_tie_break():
    module = load_audit_module()
    assert len(module.SCALE_GRID) == 15
    assert module.SCALE_GRID[0] == (0.0, 0.0)
    assert module.SCALE_GRID[-1] == (1.0, 1.0)
    assert all(
        route > 0.0 and route <= cycle
        for cycle, route in module.SCALE_GRID[1:]
    )
    objectives = {index: 2.0 for index in range(15)}
    objectives[14] = 1.0
    objectives[1] = 1.0000005
    selected, tie = module.select_scale_pair(objectives)
    assert tie == [1, 14]
    assert selected == 1


def test_nested_schedule_is_strictly_causal():
    module = load_audit_module()
    module.validate_schedule()
    assert tuple(module.INNER_SCHEDULE) == module.OUTER_MONTHS
    assert all(
        inner < outer
        for outer, inner_months in module.INNER_SCHEDULE.items()
        for inner in inner_months
    )
    assert module.FULL_SELECTION_MONTHS == (
        "2024-10",
        "2024-11",
        "2024-12",
    )


def test_independent_route_mapping_and_topology_shape():
    module = load_audit_module()
    mapping, vectors = module.build_rotation_mapping()
    assert len(mapping) == 44
    assert vectors.shape == (44, 63)
    assert not mapping.duplicated(["cycle_id", "current_state"]).any()
    assert mapping["compatible_rotation_count"].sum() == 45
    assert mapping.loc[
        mapping["compatible_rotation_count"].gt(1), "route_id"
    ].tolist() == ["cycle_15@state_1"]
    assert len(module.topology_column_names()) == 63
    assert np.isfinite(vectors).all()


def test_pinned_training_topology_is_verified_then_preserved_for_replay():
    module = load_audit_module()
    mapping, independent = module.build_rotation_mapping()
    serialized = module.pinned_training_topology_vectors(mapping, independent)
    difference = np.abs(serialized - independent)
    assert serialized.shape == (44, 63)
    assert difference.max() <= 1e-12
    # The pinned CSV's default-parsed floats are the actual frozen numerical
    # boundary and contain sub-ulp differences from direct reconstruction.
    assert np.count_nonzero(difference) > 0
    assert difference.max() == np.float64(4.440892098500626e-16)


def test_within_cycle_centers_and_blocks_are_weighted_zero():
    module = load_audit_module()
    mapping, _ = module.build_rotation_mapping()
    frame = pd.DataFrame(
        {
            "cycle_index": mapping["cycle_index"].to_numpy(dtype=int),
            "route_index": mapping["route_index"].to_numpy(dtype=int),
        }
    )
    weights = np.linspace(1.0, 2.0, len(frame))
    mu_cycle, mu_route, route_cycle = module.weighted_hierarchy_centers(
        frame, weights, mapping
    )
    assert np.isclose(mu_cycle.sum(), 1.0)
    for cycle in range(20):
        assert np.isclose(mu_route[route_cycle == cycle].sum(), 1.0)

    cycle_block = module.centered_cycle_block(
        frame["cycle_index"].to_numpy(), mu_cycle, 0.5
    ).toarray()
    assert np.allclose(np.average(cycle_block, axis=0, weights=weights), 0.0)

    route_block = module.centered_route_block(
        frame["cycle_index"].to_numpy(),
        frame["route_index"].to_numpy(),
        mu_route,
        route_cycle,
        0.25,
    ).toarray()
    for cycle in range(20):
        rows = frame["cycle_index"].eq(cycle).to_numpy()
        columns = route_cycle == cycle
        outside = route_cycle != cycle
        assert np.allclose(route_block[np.ix_(rows, outside)], 0.0)
        assert np.allclose(
            np.average(route_block[np.ix_(rows, columns)], axis=0, weights=weights[rows]),
            0.0,
        )


def test_full_design_width_and_causal_feature_blocks():
    module = load_audit_module()
    frame, mapping = synthetic_frame(module)
    design = module.build_fold_design(frame, frame, (0.25, 0.125), mapping)
    assert design.train.shape == (len(frame), 144)
    assert design.validation.shape == (len(frame), 144)
    assert np.isfinite(design.train.data).all()
    assert np.isfinite(design.validation.data).all()
    route_cycle = mapping["cycle_index"].to_numpy(dtype=int)
    for cycle in range(20):
        assert np.isclose(design.mu_route[route_cycle == cycle].sum(), 1.0)


def test_ordered_model_and_tier_probabilities_are_nested():
    module = load_audit_module()
    frame, mapping = synthetic_frame(module, repeats=4)
    # Four repeats produce all three classes while keeping every frozen cycle present.
    for target in module.TARGETS:
        for horizon in module.HORIZONS:
            frame[f"quality_class__{target}__h{horizon}"] = np.arange(len(frame)) % 3
    predictions, metadata = module.fit_pair_predictions(
        frame, frame.iloc[:20].copy(), (0.125, 0.0625), mapping
    )
    assert metadata["feature_width"] == 144
    for target in module.TARGETS:
        for horizon in module.HORIZONS:
            p75 = predictions[(target, horizon, "p75")]
            p90 = predictions[(target, horizon, "p90")]
            assert np.isfinite(p75).all()
            assert np.isfinite(p90).all()
            assert np.all((0.0 <= p90) & (p90 <= p75) & (p75 <= 1.0))


def test_tier_probability_validity_allows_one_ulp_simplex_overflow_without_clipping():
    module = load_audit_module()
    upper_half = np.nextafter(np.float64(0.5), np.float64(1.0))
    classes = np.asarray([[0.0, upper_half, upper_half]], dtype=float)
    assert classes.sum(axis=1)[0] > 1.0
    tiers = module.tier_probabilities(classes)
    assert tiers["p75"][0] > 1.0
    assert tiers["p75"][0] == classes[0, 1] + classes[0, 2]
    assert tiers["p90"][0] == classes[0, 2]


def test_selection_objective_is_equal_mean_of_twelve_cells():
    module = load_audit_module()
    rows = 9
    validation = pd.DataFrame({"conditional_weight": np.ones(rows)})
    predictions = {}
    for target_index, target in enumerate(module.TARGETS):
        for horizon_index, horizon in enumerate(module.HORIZONS):
            ordered = np.arange(rows) % 3
            validation[f"quality_class__{target}__h{horizon}"] = ordered
            base = 0.2 + 0.02 * target_index + 0.01 * horizon_index
            predictions[(target, horizon, "p75")] = np.full(rows, base + 0.4)
            predictions[(target, horizon, "p90")] = np.full(rows, base)
    losses, weights, objective = module.selection_loss_cells(validation, predictions)
    assert len(losses) == 12
    assert len(weights) == 12
    assert all(value == rows for value in weights.values())
    assert np.isclose(objective, np.mean(list(losses.values())))


def test_scope_combination_and_tie_columns_are_deterministic():
    module = load_audit_module()
    months = ("2024-04", "2024-05", "2024-06")
    cache = {}
    for month_index, month in enumerate(months):
        for grid_index in range(15):
            losses = {}
            weights = {}
            for target in module.TARGETS:
                for horizon in module.HORIZONS:
                    for tier in module.TIERS:
                        key = f"{target}__h{horizon}__{tier}"
                        losses[key] = 1.0 + grid_index * 0.01 + month_index * 0.001
                        weights[key] = 10.0 + month_index
            cache[(month, grid_index)] = {"losses": losses, "weights": weights}
    frame = module.combine_selection_scope("outer:2024-07", months, cache)
    assert len(frame) == 15
    assert frame["selected"].sum() == 1
    assert int(frame.loc[frame["selected"], "grid_index"].iloc[0]) == 0
    assert json.loads(frame.iloc[0]["validation_months_json"]) == list(months)


def test_falsification_observed_statistics_have_exact_four_keys():
    module = load_audit_module()
    panel = pd.DataFrame(
        {
            "loop_occurs": [1, 1, 0, 0],
            "conditional_weight": [1.0, 1.0, 0.0, 0.0],
            "loop_probability": [0.8, 0.7, 0.2, 0.1],
        }
    )
    for target in module.TARGETS:
        for horizon in module.HORIZONS:
            panel[f"quality_class__{target}__h{horizon}"] = [2, 0, 1, 0]
            panel[f"joint_good_target__{target}__h{horizon}"] = [1, 0, 0, 0]
            panel[f"joint_high_target__{target}__h{horizon}"] = [1, 0, 0, 0]
            for tier, candidate in (("p75", [0.9, 0.1, 0.4, 0.2]), ("p90", [0.8, 0.1, 0.2, 0.1])):
                panel[f"qhier__{target}__h{horizon}__{tier}"] = candidate
                panel[f"qcontext__{target}__h{horizon}__{tier}"] = [0.6, 0.4, 0.5, 0.4]
                panel[f"joint__qcontext__{target}__h{horizon}__{tier}"] = (
                    panel["loop_probability"]
                    * panel[f"qcontext__{target}__h{horizon}__{tier}"]
                )
                panel[f"qroute_topology__{target}__h{horizon}__{tier}"] = [
                    0.7,
                    0.3,
                    0.4,
                    0.3,
                ]
    observed = module.falsification_observed_statistics(panel)
    assert set(observed) == {
        "qhier_vs_qcontext__conditional",
        "qhier_vs_qcontext__joint",
        "qhier_vs_qroute_topology__conditional",
        "qhier_vs_qroute_topology__joint",
    }
    assert np.isfinite(list(observed.values())).all()


def test_global_cycle_label_aggregation_requires_all_three_horizons():
    module = load_audit_module()
    rows = []
    for cycle in range(20):
        for horizon in module.HORIZONS:
            if cycle == 0:
                label = "development_high_candidate"
            elif cycle == 1 and horizon == 12:
                label = "development_good_candidate"
            elif cycle == 1:
                label = "development_high_candidate"
            else:
                label = "development_unqualified"
            rows.append(
                {
                    "cycle_index": cycle,
                    "cycle_id": f"cycle_{cycle:02d}",
                    "horizon": horizon,
                    "development_label": label,
                }
            )
    global_rows = module.aggregate_global_cycle_labels(pd.DataFrame(rows))
    assert len(global_rows) == 20
    indexed = global_rows.set_index("cycle_index")["global_development_label"]
    assert indexed.loc[0] == "development_high_candidate"
    assert indexed.loc[1] == "development_good_candidate"
    assert indexed.loc[2] == "development_unqualified"


def test_common_block_positions_and_calibration_are_deterministic():
    module = load_audit_module()
    first = module.common_block_positions(12, draws=25)
    second = module.common_block_positions(12, draws=25)
    assert first.shape == (25, 12)
    assert np.array_equal(first, second)
    assert first.min() >= 0 and first.max() < 12
    observed = np.array([0, 0, 1, 1], dtype=float)
    probability = np.array([0.05, 0.15, 0.85, 0.95], dtype=float)
    rows, ece, maximum = module.calibration_table(
        "synthetic",
        "conditional",
        "qhier",
        "absolute_return_bps",
        6,
        "p75",
        observed,
        probability,
        np.ones(4),
        1,
    )
    assert len(rows) == 10
    assert np.isclose(ece, 0.1)
    assert np.isclose(maximum, 0.15)


def test_holm_and_nested_comparison_fail_closed():
    module = load_audit_module()
    frame = pd.DataFrame(
        {
            "cycle_index": [0, 1, 2],
            "horizon": [6, 6, 6],
            "p": [0.001, 0.01, 0.02],
        }
    )
    passed = module.holm_passes(frame, "p", alpha=0.025)
    assert passed.tolist() == [True, True, True]
    assert module.nested_differences(
        {"x": [1.0, {"y": 2}]}, {"x": [1.0 + 1e-13, {"y": 2}]}
    ) == []
    assert module.nested_differences({"x": 1.0}, {"x": 1.1})
    assert (
        module.ordered_minimum_development_label(
            [
                "development_high_candidate",
                "development_good_candidate",
                "development_high_candidate",
            ]
        )
        == "development_good_candidate"
    )
    assert (
        module.ordered_minimum_development_label(
            ["development_high_candidate", "development_unqualified"]
        )
        == "development_unqualified"
    )


def test_cli_modes_and_postscore_fail_closed(tmp_path):
    module = load_audit_module()
    assert module.parse_args(["--preartifact-only"]).preartifact_only
    assert module.parse_args(["--pre-score-only"]).pre_score_only
    assert module.parse_args(["--post-score"]).post_score
    result = module.post_score_audit(tmp_path)
    assert result["all_passed"] is False
    assert result["later_period_outcomes_opened_by_audit"] is False
    assert result["research_only"] is True
    assert result["live_ordering_enabled"] is False
    assert result["order_placement"] == "disabled"


def test_stored_multinomial_probability_replay_is_manual_and_exact():
    module = load_audit_module()
    matrix = sparse.csr_matrix([[1.0, 0.5], [-0.25, 2.0]])
    key = "qhier__absolute_return_bps__h6"
    coefficients = np.asarray(
        [[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]], dtype=float
    )
    intercept = np.asarray([0.05, -0.02, 0.01], dtype=float)
    parameters = {
        f"{key}__classes": np.asarray([0, 1, 2], dtype=int),
        f"{key}__coef": coefficients,
        f"{key}__intercept": intercept,
        f"{key}__temperature": np.asarray([1.0]),
    }
    actual = module.independent_stored_class_probabilities(
        matrix, parameters, key
    )
    logits = matrix.toarray() @ coefficients.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    expected = np.exp(logits)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(actual, expected, atol=1e-15)
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_later_transfer_is_strictly_demotion_only():
    module = load_audit_module()
    base = pd.DataFrame(
        [
            {
                "cycle_index": 0,
                "cycle_id": "cycle_00",
                "horizon": 6,
                "development_label": "development_good_candidate",
                "global_development_label": "development_good_candidate",
            },
            {
                "cycle_index": 1,
                "cycle_id": "cycle_01",
                "horizon": 6,
                "development_label": "development_unqualified",
                "global_development_label": "development_unqualified",
            },
        ]
    )
    later_2025 = base.copy()
    later_2025.loc[0, ["development_label", "global_development_label"]] = (
        "development_high_candidate",
        "development_high_candidate",
    )
    later_2025.loc[1, ["development_label", "global_development_label"]] = (
        "development_high_candidate",
        "development_high_candidate",
    )
    later_2023 = base.copy()
    later_2023.loc[0, ["development_label", "global_development_label"]] = (
        "development_unqualified",
        "development_unqualified",
    )
    payload = module.independent_period_transfer(
        {
            "primary_algorithm_label": "development_algorithm_supported",
            "primary_algorithm_pass": True,
        },
        base,
        {"2025": later_2025, "2023": later_2023},
        {
            "2025": {
                "primary_algorithm_pass": True,
                "primary_algorithm_label": "development_algorithm_supported",
            },
            "2023": {
                "primary_algorithm_pass": False,
                "primary_algorithm_label": "development_algorithm_unconfirmed",
            },
        },
    )
    labels = [
        row["final_development_portability_label"]
        for row in payload["named_transfer"]
    ]
    assert labels == ["development_unqualified", "development_unqualified"]
    assert payload["algorithm_development_portable"] is False
    assert payload["later_promotion_performed"] is False
    assert not any(
        row["later_promotion_performed"] for row in payload["named_transfer"]
    )


def test_v1_scoring_support_uses_full_year_rules_not_oof_rules():
    module = load_audit_module()
    contract = json.loads(
        (
            module.WORKSPACE
            / "work/contracts/20260710-per-loop-movement-quality-v1.json"
        ).read_text()
    )
    rows = 5000
    realized = 800
    quarters = np.resize(
        np.asarray(["2025_q1", "2025_q2", "2025_q3", "2025_q4"]), rows
    )
    frame = pd.DataFrame(
        {
            "session_date": np.resize(
                np.asarray(
                    ["2025-02-03", "2025-05-05", "2025-08-04", "2025-11-03"]
                ),
                rows,
            ),
            "quarter": quarters,
            "loop_occurs": np.arange(rows) < realized,
            "symbol_norm": np.resize(
                np.asarray([f"stock_{index:02d}" for index in range(18)]), rows
            ),
        }
    )
    assert module.v1_base_support(frame, contract, "scoring") is True
    assert module.v1_base_support(frame, contract, "oof") is False


def test_exact_frame_comparison_rejects_infinity_and_finite_mismatch():
    module = load_audit_module()
    expected = pd.DataFrame({"key": [1, 2], "value": [0.5, np.nan]})
    stored = pd.DataFrame({"key": [1, 2], "value": [np.inf, np.nan]})
    passed, details = module._frame_maximum_error(stored, expected, ["key"])
    assert passed is False
    assert "value" in details["categorical_mismatches"]
    assert module.nested_differences({"pass": True}, {"pass": 1})
    assert module.nested_differences({"pass": False}, {"pass": 0})
