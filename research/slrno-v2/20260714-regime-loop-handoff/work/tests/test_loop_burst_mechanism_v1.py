from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER_PATH = WORKSPACE / "work/run_loop_burst_mechanism_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("loop_burst_v1", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_hash_sources_and_safety_are_exact() -> None:
    module = load_module()
    contract, hashes = module.verify_contract_and_sources()
    assert hashes == module.EXPECTED_HASHES
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["population_and_target"]["later_period_paths_permitted"] is False
    assert (
        contract["population_and_target"][
            "prospective_shadow_read_or_write_permitted"
        ]
        is False
    )


def test_parent_audit_schema_is_bound_fail_closed() -> None:
    module = load_module()
    audit = json.loads(module.OOF_AUDIT.read_text())
    assert audit["all_passed"] is True
    assert audit["check_count"] == 47
    assert audit["later_period_paths_resolved"] is False
    assert audit["shadow_tree_read"] is False
    assert audit["shadow_tree_written"] is False


def test_repeat_count_and_completed_dwell_features_are_causal() -> None:
    module = load_module()
    states = np.asarray([5, 7, 5, 7, 5, 6])
    durations = np.asarray([3, 4, 5, 6, 7, 8])
    values = module.sequence_feature(states, durations, 4, 5, 7)
    assert values[0] == 2
    assert values[1] == 5
    assert values[2] == 6
    assert values[3] == 7
    assert values[4] == 8
    assert np.isnan(values[5])
    assert values[6] == 11
    no_prior = module.sequence_feature(states, durations, 5, 6, 5)
    assert no_prior[:3] == (0, 0, 0)


def test_two_state_other_state_parser_rejects_non_two_state_cycle() -> None:
    module = load_module()
    assert module.other_state("5->7->5", 5) == 7
    assert module.other_state("5->7->5", 7) == 5
    try:
        module.other_state("1->2->3->1", 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("three-state cycle was accepted")


def test_design_widths_and_penalties_are_frozen() -> None:
    module = load_module()
    frame = pd.DataFrame(
        {
            **{feature: [0.0, 1.0] for feature in module.PHASE_FEATURES},
            "orientation_index": [0, 25],
        }
    )
    center = np.zeros(5)
    scale = np.ones(5)
    offset, offset_penalty = module.phase_matrix(
        frame, center, scale, "qoffset_calibration"
    )
    global_matrix, global_penalty = module.phase_matrix(
        frame, center, scale, "qburst_global"
    )
    orientation, orientation_penalty = module.phase_matrix(
        frame, center, scale, "qburst_orientation"
    )
    assert offset.shape == (2, 1)
    assert global_matrix.shape == (2, 6)
    assert orientation.shape == (2, 156)
    assert np.array_equal(offset_penalty, np.zeros(1))
    assert global_penalty.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert orientation_penalty[:6].tolist() == global_penalty.tolist()
    assert np.all(orientation_penalty[6:31] == 4.0)
    assert np.all(orientation_penalty[31:] == 8.0)


def test_offset_optimizer_and_binary_losses_are_exact() -> None:
    module = load_module()
    y = np.asarray([0, 0, 1, 1])
    matrix = np.ones((4, 1))
    beta, audit = module.fit_offset_model(
        matrix,
        np.zeros(4),
        y,
        np.ones(4),
        np.zeros(1),
        0.0,
    )
    assert audit["optimizer_success"] is True
    assert abs(beta[0]) < 1e-10
    log_loss, brier = module.binary_losses(
        np.asarray([0, 1]), np.asarray([0.25, 0.75])
    )
    assert np.allclose(log_loss, -np.log(0.75))
    assert np.allclose(brier, 0.0625)


def test_calibration_bootstrap_and_holm_are_deterministic() -> None:
    module = load_module()
    y = np.asarray([0, 0, 1, 1])
    p = np.asarray([0.05, 0.15, 0.85, 0.95])
    ece, maximum, bins = module.calibration(y, p, np.ones(4), minimum_rows=1)
    assert np.isclose(ece, 0.1)
    assert np.isclose(maximum, 0.15)
    assert bins == 4
    values = np.linspace(-0.02, 0.01, 20)
    assert module.bootstrap_interval(values, 7) == module.bootstrap_interval(values, 7)
    frame = pd.DataFrame(
        {
            "baseline": ["a", "b", "a", "b"],
            "endpoint": ["log_loss", "log_loss", "brier", "brier"],
            "p_value": [0.01, 0.04, 0.03, 0.02],
        }
    )
    adjusted = module.holm(frame)
    assert adjusted["holm_adjusted_p"].notna().all()
    assert len(adjusted) == 4

