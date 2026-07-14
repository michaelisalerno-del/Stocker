from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/run_causal_state_pattern_discovery_v1.py"
SPEC = importlib.util.spec_from_file_location("causal_state_pattern_discovery", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contract_is_research_only_and_temporally_split() -> None:
    contract = MODULE.load_contract()
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["period_and_phase_lock"]["discovery_period"] == [
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
        "2024-06",
    ]
    assert contract["period_and_phase_lock"]["qualification_period"] == [
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
    ]
    assert contract["period_and_phase_lock"]["2023_or_2025_paths_permitted"] is False


def test_phase_one_whitelist_has_no_outcome_or_volume() -> None:
    contract = MODULE.load_contract()
    columns = contract["period_and_phase_lock"]["phase_1_allowed_anchor_columns"]
    assert tuple(columns) == MODULE.STATE_COLUMNS
    assert not any("return" in column or "range" in column for column in columns)
    assert not any("volume" in column for column in columns)


def test_occurrence_is_exact_directional_and_length_specific() -> None:
    frame = pd.DataFrame(
        {
            "state": [3, 3, 3],
            "future_state_1": [6, 5, 6],
            "future_state_2": [3, 3, 3],
            "future_state_3": [5, 6, 5],
            "future_state_4": [3, 3, 4],
        }
    )
    assert MODULE.occurrence(frame, (3, 6, 3)).tolist() == [True, False, True]
    assert MODULE.occurrence(frame, (3, 6, 3, 5, 3)).tolist() == [True, False, False]
    assert MODULE.occurrence(frame, (3, 5, 3)).tolist() == [False, True, False]


def test_candidate_ids_keep_entry_orientation() -> None:
    assert MODULE.candidate_id("closed_loop", (3, 6, 3)) == "closed_loop__L2__3_6_3"
    assert MODULE.candidate_id("closed_loop", (6, 3, 6)) == "closed_loop__L2__6_3_6"
    assert MODULE.candidate_id("upward_excursion", (0, 4)) == "upward_excursion__0_4"


def test_frozen_dictionary_rotation_marks_existing_paths() -> None:
    cycles = pd.DataFrame(
        {
            "cycle_id": [f"cycle_{index:02d}" for index in range(1, 21)],
            "cycle": ["0->1->0"] * 20,
            "transition_length": [2] * 20,
        }
    )
    paths = MODULE.exact_oriented_frozen_paths(cycles)
    assert (0, 1, 0) in paths
    assert (1, 0, 1) in paths
    assert (0, 2, 0) not in paths


def test_upward_excursion_definition_crosses_centroid_boundary() -> None:
    centroids = np.asarray([-1.2, -0.5, -0.3, -0.1, 0.27, 0.55, 0.62, 0.98])
    allowed = [
        (source, destination)
        for source in range(8)
        for destination in range(8)
        if centroids[source] < 0
        and centroids[destination] > 0
        and centroids[destination] - centroids[source] >= 0.25
    ]
    assert (0, 4) in allowed
    assert (3, 4) in allowed
    assert (4, 7) not in allowed
    assert (7, 0) not in allowed


def test_path_probability_product_updates_state_history() -> None:
    first = np.full((8, 9), 1 / 9)
    history = np.full((9, 9, 8, 9), 1 / 9)
    history[1, 2, 3, :] = 0
    history[1, 2, 3, 6] = 1
    history[2, 3, 6, :] = 0
    history[2, 3, 6, 3] = 1
    value = MODULE.path_probability_lookup(first, history, (3, 6, 3), [(1, 2)])[(1, 2)]
    assert np.allclose(value, (1 - MODULE.EPSILON, 1 / 81))


def test_candidate_feature_width_and_scale() -> None:
    context = sparse.csr_matrix(np.ones((3, 17)))
    result = MODULE.append_candidate_features(context, np.asarray([0, 2, 1]), 3, 0.5)
    assert result.shape == (3, 20)
    assert np.allclose(result[:, 17:].toarray().sum(axis=1), 0.5)


def test_realized_overlap_weights_sum_to_one_per_anchor() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": [1, 2],
            "state": [0, 0],
            "future_state_1": [1, 2],
            "future_state_2": [0, 0],
            "future_state_3": [1, 8],
            "future_state_4": [0, 8],
        }
    )
    manifest = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "candidate_index": [0, 1],
            "family": ["closed_loop", "closed_loop"],
            "start_state": [0, 0],
            "exact_path": ["0->1->0", "0->1->0->1->0"],
            "decision_eligible": [True, True],
        }
    )
    expanded = MODULE.expand_family(anchors, manifest)
    realized = expanded.loc[expanded["candidate_occurs"].eq(1)]
    totals = realized.groupby("anchor_id")["conditional_weight"].sum()
    assert np.allclose(totals, 1.0)
    assert sorted(realized.loc[realized["anchor_id"].eq(1), "conditional_weight"]) == [0.5, 0.5]


def test_destination_kernel_normalizes_and_backs_off() -> None:
    contract = MODULE.load_contract()
    training = pd.DataFrame(
        {
            "state": [0, 0, 1, 1],
            "future_state_1": [1, 1, 0, 8],
            "previous_state_1": [8, 8, 0, 0],
            "previous_state_2": [8, 8, 8, 8],
        }
    )
    first, history = MODULE.fit_destination_kernel(training, contract)
    assert first.shape == (8, 9)
    assert history.shape == (9, 9, 8, 9)
    assert np.allclose(first.sum(axis=1), 1)
    assert np.allclose(history.sum(axis=3), 1)
    assert history[7, 7, 0, 1] == first[0, 1]


def test_qualification_columns_use_exact_movement_but_no_direction_or_volume() -> None:
    columns = MODULE.qualification_columns(MODULE.load_contract())
    assert {"exact_6", "exact_12", "exact_24"}.issubset(columns)
    assert "absolute_return_bps_6" in columns
    assert "future_range_bps_24" in columns
    assert not any("signed_return" in column or "direction" in column for column in columns)
    assert not any("volume" in column for column in columns)


def test_holm_adjusts_separately_by_family_and_tier() -> None:
    frame = pd.DataFrame(
        {
            "family": ["a", "a", "a", "b"],
            "tier": ["p75", "p75", "p90", "p75"],
            "p_value": [0.01, 0.04, 0.04, 0.04],
        }
    )
    adjusted = MODULE.holm_adjust(frame)
    family_a_p75 = adjusted.loc[(adjusted["family"] == "a") & (adjusted["tier"] == "p75")]
    assert family_a_p75["family_size"].eq(2).all()
    assert adjusted.loc[(adjusted["family"] == "b"), "family_size"].eq(1).all()
    empty = MODULE.holm_adjust(pd.DataFrame(columns=["family", "tier", "p_value"]))
    assert empty.empty
    assert {"holm_adjusted_p", "holm_pass", "holm_rank", "family_size"}.issubset(empty)


def test_runner_has_no_later_or_shadow_input_constant() -> None:
    source = SOURCE.read_text()
    assert "ANCHOR_2025" not in source
    assert "ANCHOR_2023" not in source
    assert "partial_2026" not in source
    assert "prediction_ledger" not in source
