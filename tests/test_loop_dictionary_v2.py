from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.loop_dictionary_v2 import (
    ALLOWED_COMPOSITE_TRANSITION_LENGTHS,
    ALLOWED_PRIMITIVE_TRANSITION_LENGTHS,
    MAX_EVENT_TRANSITIONS,
    LegacyCycleRecord,
    LoopDictionary,
    MotifType,
    UnsupportedLoopError,
    decompose_closed_path,
)


def test_rotations_share_semantic_identity_but_keep_orientation_metadata() -> None:
    left = decompose_closed_path((2, 4, 2))
    right = decompose_closed_path((4, 2, 4))

    assert left.semantic_loop_id == right.semantic_loop_id == "loop_p_2-4-2"
    assert left.primitive_loop_id == right.primitive_loop_id == "loop_p_2-4-2"
    assert left.orientation_id_for((2, 4, 2)) != left.orientation_id_for((4, 2, 4))
    assert left.oriented_paths == ((2, 4, 2), (4, 2, 4))


def test_reverse_direction_is_not_silently_equated() -> None:
    clockwise = decompose_closed_path((1, 2, 3, 1))
    counter_clockwise = decompose_closed_path((1, 3, 2, 1))

    assert clockwise.semantic_loop_id == "loop_p_1-2-3-1"
    assert counter_clockwise.semantic_loop_id == "loop_p_1-3-2-1"
    assert clockwise.semantic_loop_id != counter_clockwise.semantic_loop_id


def test_repeated_traversal_reduces_to_primitive_root_and_repeat_depth() -> None:
    loop = decompose_closed_path((1, 3, 1, 3, 1))

    assert loop.motif_type is MotifType.REPEAT
    assert loop.primitive_core == (1, 3)
    assert loop.primitive_transition_length == 2
    assert loop.repeat_depth == 2
    assert loop.full_core == (1, 3, 1, 3)
    assert loop.full_transition_length == 4
    assert loop.primitive_loop_id == "loop_p_1-3-1"
    assert loop.semantic_loop_id == "loop_r2_1-3-1"


def test_nonperiodic_composite_retains_ordered_primitive_components() -> None:
    motif = decompose_closed_path((1, 2, 1, 3, 1))

    assert motif.motif_type is MotifType.COMPOSITE
    assert motif.repeat_depth == 1
    assert motif.component_primitive_ids == ("loop_p_1-2-1", "loop_p_1-3-1")
    assert motif.component_boundaries == ((0, 2), (2, 4))
    assert motif.semantic_loop_id.startswith("loop_c_")


def test_composite_decomposition_does_not_depend_on_minimum_state_anchor() -> None:
    anchored_at_two = decompose_closed_path((2, 1, 2, 3, 2))
    rotated = decompose_closed_path((1, 2, 3, 2, 1))

    assert anchored_at_two.motif_type is MotifType.COMPOSITE
    assert anchored_at_two.component_primitive_ids == (
        "loop_p_1-2-1",
        "loop_p_2-3-2",
    )
    assert anchored_at_two.component_boundaries == ((0, 2), (2, 4))
    assert anchored_at_two.canonical_orientation == (2, 1, 2, 3, 2)
    assert anchored_at_two.semantic_loop_id == rotated.semantic_loop_id


def test_simple_nonperiodic_cycle_is_primitive_not_composite() -> None:
    loop = decompose_closed_path((1, 2, 3, 1))

    assert loop.motif_type is MotifType.PRIMITIVE
    assert loop.component_primitive_ids == ()
    assert loop.repeat_depth == 1


def test_semantic_ids_do_not_depend_on_discovery_order_or_legacy_id() -> None:
    first = LoopDictionary.from_legacy(
        (
            LegacyCycleRecord("cycle_01", (2, 4, 2), discovery_rank=1),
            LegacyCycleRecord("cycle_02", (1, 3, 1, 3, 1), discovery_rank=2),
        ),
        version="semantic_loop_dictionary_v2",
    )
    second = LoopDictionary.from_legacy(
        (
            LegacyCycleRecord("renamed_b", (1, 3, 1, 3, 1), discovery_rank=99),
            LegacyCycleRecord("renamed_a", (4, 2, 4), discovery_rank=42),
        ),
        version="semantic_loop_dictionary_v2",
    )

    assert first.semantic_ids == second.semantic_ids
    assert first.dictionary_hash == second.dictionary_hash


def test_migration_keeps_legacy_identity_separate_and_is_deterministic() -> None:
    dictionary = LoopDictionary.from_legacy(
        (
            LegacyCycleRecord("cycle_14", (1, 3, 1, 3, 1), discovery_rank=14),
            LegacyCycleRecord("cycle_01", (1, 3, 1), discovery_rank=1),
        ),
        version="semantic_loop_dictionary_v2",
    )

    rows = dictionary.migration_rows()
    assert [row["legacy_cycle_id"] for row in rows] == ["cycle_01", "cycle_14"]
    assert rows[0]["semantic_loop_id"] == "loop_p_1-3-1"
    assert rows[1]["semantic_loop_id"] == "loop_r2_1-3-1"
    assert rows[1]["primitive_loop_id"] == "loop_p_1-3-1"
    assert rows[1]["repeat_depth"] == 2
    assert all(row["migration_status"] == "migrated" for row in rows)


def test_unsupported_legacy_cycle_fails_closed() -> None:
    path = tuple(range(MAX_EVENT_TRANSITIONS + 1)) + (0,)

    with pytest.raises(UnsupportedLoopError):
        LoopDictionary.from_legacy(
            (LegacyCycleRecord("cycle_bad", path, discovery_rank=1),),
            version="semantic_loop_dictionary_v2",
        )


def test_legacy_cycle_table_reader_preserves_ids_and_fails_closed() -> None:
    table = pd.DataFrame(
        {
            "legacy_cycle_id": ["cycle_07"],
            "cycle": ["2->4->2"],
            "discovery_rank": [7],
        }
    )

    dictionary = LoopDictionary.from_legacy_table(table, version="legacy_dictionary_v1")

    assert dictionary.migration_rows()[0]["legacy_cycle_id"] == "cycle_07"
    with pytest.raises(UnsupportedLoopError):
        LoopDictionary.from_legacy_table(
            table.drop(columns="cycle"), version="legacy_dictionary_v1"
        )


def test_allowed_length_constants_share_one_closed_world() -> None:
    assert frozenset({2, 3, 4, 5}) == ALLOWED_PRIMITIVE_TRANSITION_LENGTHS
    assert frozenset({4, 5, 6, 7, 8}) == ALLOWED_COMPOSITE_TRANSITION_LENGTHS
    assert MAX_EVENT_TRANSITIONS == 8
