from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_per_loop_movement_quality.py"
SPEC = importlib.util.spec_from_file_location("independent_quality_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_canonical_cycles_and_repeated_state_routes_are_deduplicated() -> None:
    assert audit.canonical_cycle((3, 1, 3, 1)) == (1, 3, 1, 3)
    assert audit.compatible_routes((0, 1, 0, 1), 0) == [(0, 1, 0, 1, 0)]
    assert audit.compatible_routes((0, 1), 7) == []


def test_realized_label_requires_exact_rotated_path_and_end_is_negative() -> None:
    frame = pd.DataFrame(
        {
            "state": [0, 1, 0, 0],
            "future_state_1": [1, 0, 1, 1],
            "future_state_2": [0, 1, 0, audit.END],
            "future_state_3": [7, 7, 7, audit.END],
            "future_state_4": [7, 7, 7, audit.END],
        }
    )
    assert audit.realized_label(frame, (0, 1)).tolist() == [True, True, True, False]


def test_threshold_class_uses_strict_greater_than() -> None:
    values = np.asarray([0.0, 7.5, 7.50001, 9.0, 9.00001])
    assert audit.movement_class(values, 7.5, 9.0).tolist() == [0, 0, 1, 1, 2]


def test_oof_schedule_is_strictly_prior_month_only() -> None:
    months = [f"2024-{month:02d}" for month in range(1, 13)]
    frame = pd.DataFrame({"month_key": months})
    splits = audit.oof_splits(frame)
    assert [name for name, _, _ in splits] == list(audit.OOF_MONTHS)
    for name, train, validation in splits:
        assert all(frame.loc[train, "month_key"] < name)
        assert set(frame.loc[validation, "month_key"]) == {name}


def test_cycle_sparse_block_indices_and_frozen_scales() -> None:
    frame = pd.DataFrame(
        {
            "cycle_index": [2, 19],
            "state": [3, 7],
            "history_token": [17, 647],
        }
    )
    matrix = audit.cycle_blocks(frame).tocsr()
    assert matrix.shape == (2, 20 + 160 + 12960)
    row0 = matrix.getrow(0)
    expected0 = {
        2: 1.0,
        20 + 2 * 8 + 3: 0.5,
        20 + 160 + 2 * 648 + 17: 0.25,
    }
    assert dict(zip(row0.indices.tolist(), row0.data.tolist())) == expected0
    row1 = matrix.getrow(1)
    expected1 = {
        19: 1.0,
        20 + 19 * 8 + 7: 0.5,
        20 + 160 + 19 * 648 + 647: 0.25,
    }
    assert dict(zip(row1.indices.tolist(), row1.data.tolist())) == expected1


def test_temperature_tie_break_selects_one() -> None:
    probability = np.full((6, 3), 1.0 / 3.0)
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    weights = np.ones(6)
    selected, table = audit.select_temperature(labels, probability, weights)
    assert selected == 1.0
    assert len(table) == len(audit.TEMPERATURES)


def test_temperature_is_softmax_of_log_probability() -> None:
    probability = np.asarray([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])
    assert np.allclose(audit.apply_temperature(probability, 1.0), probability)
    sharpened = audit.apply_temperature(probability, 0.75)
    assert sharpened[0, 2] > probability[0, 2]
    assert sharpened[1, 0] > probability[1, 0]
    assert np.allclose(sharpened.sum(axis=1), 1.0)


def test_ordered_probabilities_are_nested_and_chain_rule_exact() -> None:
    structural = np.asarray([0.2, 0.8])
    classes = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.25, 0.65]])
    output = audit.ordered_outputs(structural, classes)
    assert np.allclose(output["q75"], [0.5, 0.9])
    assert np.allclose(output["q90"], [0.2, 0.65])
    assert np.allclose(output["j75"], structural * output["q75"])
    assert np.allclose(output["j90"], structural * output["q90"])
    assert np.all(output["q90"] <= output["q75"])


def _synthetic_anchor_panel() -> pd.DataFrame:
    rows = []
    for anchor_id, future in enumerate(((1, 0, 1, 0), (1, audit.END, audit.END, audit.END))):
        row = {
            "anchor_id": anchor_id,
            "symbol_norm": "AAA",
            "session_date": "2024-07-01",
            "month_key": "2024-07",
            "quarter": "2024_q3",
            "start_timestamp": pd.Timestamp("2024-07-01", tz="UTC")
            + pd.Timedelta(minutes=5 * anchor_id),
            "state": 0,
            "previous_state_1": 8,
            "previous_state_2": 8,
            "history_token": (8 * 9 + 8) * 8,
        }
        for name in audit.NUMERIC_CONTROLS:
            row[name] = float(anchor_id)
        for step, value in enumerate(future, start=1):
            row[f"future_state_{step}"] = value
        for target in audit.TARGETS:
            for horizon in audit.HORIZONS:
                row[f"{target}_{horizon}"] = 10.0 + anchor_id
        rows.append(row)
    return pd.DataFrame(rows)


def test_positive_overlap_weights_sum_to_one_per_anchor() -> None:
    anchors = _synthetic_anchor_panel()
    parameters = {
        "first_order": np.full((8, 9), 1.0 / 9.0),
        "history_intercept": np.zeros(9),
        "history_coef": np.zeros((9, 648)),
    }
    cycles = [
        {
            "cycle_id": "cycle_01",
            "cycle_index": 0,
            "cycle": "0->1->0",
            "transition_length": 2,
            "core": (0, 1),
        },
        {
            "cycle_id": "cycle_02",
            "cycle_index": 1,
            "cycle": "0->1->0->1->0",
            "transition_length": 4,
            "core": (0, 1, 0, 1),
        },
    ]
    long = audit.reconstruct_long_panel(anchors, cycles, parameters)
    first = long.loc[long["anchor_id"].eq(0)]
    assert first["loop_occurs"].tolist() == [1, 1]
    assert first["positive_cycle_count"].tolist() == [2, 2]
    assert first["conditional_weight"].tolist() == [0.5, 0.5]
    second = long.loc[long["anchor_id"].eq(1)]
    assert second["loop_occurs"].sum() == 0
    assert second["conditional_weight"].sum() == 0.0


def test_feature_widths_keep_penalty_scales_after_context_scaling() -> None:
    frame = _synthetic_anchor_panel()
    frame["cycle_index"] = [0, 1]
    train_x, predict_x, metadata = audit.feature_matrices(frame, frame)
    assert train_x["qcontext"].shape == (2, 17)
    assert train_x["qcycle"].shape == (2, 17 + 20 + 160 + 12960)
    assert predict_x["qcycle"].shape == train_x["qcycle"].shape
    blocks = train_x["qcycle"][:, 17:]
    assert set(np.unique(blocks.data)) == {0.25, 0.5, 1.0}
    assert metadata["widths"] == {"qcontext": 17, "qcycle": 13157}


def test_final_grade_is_minimum_across_all_three_periods() -> None:
    provisional = pd.DataFrame(
        {"cycle_id": ["cycle_01", "cycle_02"], "global_grade": ["high_movement_quality", "good_movement_quality"]}
    )
    development = pd.DataFrame(
        {"cycle_id": ["cycle_01", "cycle_02"], "global_grade": ["good_movement_quality", "high_movement_quality"]}
    )
    backward = pd.DataFrame(
        {"cycle_id": ["cycle_01", "cycle_02"], "global_grade": ["unqualified", "good_movement_quality"]}
    )
    tiers, gates = audit.reconstruct_final_tiers(
        provisional, development, backward
    )
    assert tiers["final_grade"].tolist() == ["unqualified", "good_movement_quality"]
    assert gates["qualified_good_or_high_cycles"] == 1
    assert gates["high_cycles"] == 0


def test_moving_block_interval_is_deterministic_by_seed() -> None:
    values = np.linspace(-0.02, 0.01, 30)
    first = audit.moving_block_interval(values, 123)
    second = audit.moving_block_interval(values, 123)
    third = audit.moving_block_interval(values, 124)
    assert first == second
    assert first != third
    assert first[1] <= first[0] <= first[2]


def test_structural_gate_rewards_informative_history_probability() -> None:
    observed = np.asarray([0, 1] * 100)
    frame = pd.DataFrame(
        {
            "loop_occurs": observed,
            "loop_probability": observed.astype(float),
            "first_order_probability": np.full(len(observed), 0.5),
        }
    )
    result = audit.structural_gate(frame, minimum_rows=10, tolerance=0.01)
    assert result["history_log_loss"] < result["first_order_log_loss"]
    assert result["history_brier"] < result["first_order_brier"]
    assert result["pass"] is True


def test_protected_snapshot_excludes_volatile_filesystem_metadata() -> None:
    snapshot = audit.content_snapshot()
    assert snapshot["runtime_outcomes_opened"] is False
    assert snapshot["ledger_lines"] == 0
    assert snapshot["ledger_size"] == 0
    for row in snapshot["files"]:
        assert "mode" in row
        assert "inode" not in row
        assert "mtime_ns" not in row
        assert "ctime_ns" not in row


def test_contract_safety_and_source_set_are_fit_only() -> None:
    contract = json.loads(audit.CONTRACT.read_text())
    audit._validate_contract(contract)
    sources = audit.expected_source_hashes()
    assert sources
    assert "runner.py" in sources
    assert "anchor_panel_train_2024.parquet" in sources
    assert not any("2025" in name or "2023" in name for name in sources)
