from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/run_regime_loop_linkage_ideas_v1.py"
SPEC = importlib.util.spec_from_file_location("regime_loop_linkage", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contract_is_frozen_research_only_and_no_later_scoring() -> None:
    contract = MODULE.load_contract()
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["population_and_join"]["later_period_paths_permitted"] is False
    assert contract["population_and_join"]["prospective_shadow_read_or_write_permitted"] is False
    assert contract["decision"]["named_loop_good_or_high_promotion_permitted"] is False


def test_fixed_half_logit_blend_is_between_components() -> None:
    baseline = np.asarray([0.01, 0.2, 0.7])
    full = np.asarray([0.1, 0.5, 0.9])
    blended = MODULE.logit_blend(baseline, full, 0.5)
    assert np.all(blended > baseline)
    assert np.all(blended < full)
    assert np.allclose(MODULE.logit_blend(baseline, baseline, 0.5), baseline)


def test_fixed_compositions_use_declared_component_pairs() -> None:
    frame = pd.DataFrame(
        {
            "loop_occurs": [1],
            "qhistory": [0.2],
            "qlimited4": [0.3],
            "qfull9": [0.4],
        }
    )
    for target in MODULE.TARGETS:
        for horizon in MODULE.HORIZONS:
            frame[f"quality_class__{target}__h{horizon}"] = 2
            for tier in MODULE.TIERS:
                frame[f"qcontext__{target}__h{horizon}__{tier}"] = 0.25
                frame[f"qroute_topology__{target}__h{horizon}__{tier}"] = 0.5
                frame[f"qhier__{target}__h{horizon}__{tier}"] = 0.75
    result = MODULE.add_fixed_compositions(frame)
    assert result[MODULE.joint_target_column("absolute_return_bps", 6, "p75")].iloc[0] == 1
    assert np.isclose(
        result[MODULE.probability_column("baseline", "absolute_return_bps", 6, "p75")].iloc[0],
        0.2 * 0.25,
    )
    assert np.isclose(
        result[
            MODULE.probability_column("minimal_time_topology", "absolute_return_bps", 6, "p75")
        ].iloc[0],
        0.3 * 0.5,
    )
    assert np.isclose(
        result[MODULE.probability_column("raw_full_link", "absolute_return_bps", 6, "p75")].iloc[0],
        0.4 * 0.75,
    )


def test_dependency_features_have_separate_residuals_and_interaction() -> None:
    frame = pd.DataFrame(
        {
            "qhistory": [0.2, 0.3],
            "qfull9": [0.4, 0.35],
            "qcontext__absolute_return_bps__h6__p75": [0.25, 0.4],
            "qhier__absolute_return_bps__h6__p75": [0.5, 0.3],
        }
    )
    features = MODULE.meta_features(frame, "absolute_return_bps", 6, "p75")
    assert features.shape == (2, 5)
    assert np.allclose(features[:, 4], features[:, 1] * features[:, 3])


def test_binary_losses_and_calibration_are_exact_on_toy_data() -> None:
    observed = np.asarray([0, 1])
    probability = np.asarray([0.25, 0.75])
    log_loss, brier = MODULE.binary_losses(observed, probability)
    assert np.allclose(log_loss, -np.log(0.75))
    assert np.allclose(brier, 0.0625)
    ece, maximum, bins = MODULE.calibration_summary(
        observed, probability, np.ones(2), minimum_rows=1
    )
    assert np.isclose(ece, 0.25)
    assert np.isclose(maximum, 0.25)
    assert bins == 2


def test_holm_can_adjust_one_global_family_or_endpoint_families() -> None:
    frame = pd.DataFrame(
        {
            "endpoint": ["ll", "ll", "brier"],
            "p_value": [0.01, 0.04, 0.03],
        }
    )
    endpoint = MODULE.holm_adjust(frame, ["endpoint"])
    assert endpoint.loc[endpoint["endpoint"].eq("ll"), "family_size"].eq(2).all()
    global_family = MODULE.holm_adjust(frame, [])
    assert global_family["family_size"].eq(3).all()


def test_runner_has_no_later_or_shadow_input_constants() -> None:
    source = SOURCE.read_text()
    assert "OOF_2025" not in source
    assert "OOF_2023" not in source
    assert "ANCHOR_2026" not in source
    assert "prediction_ledger" not in source

