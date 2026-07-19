from __future__ import annotations

import numpy as np
import pandas as pd

from stocker_research.regime_repair_comparison_v2 import (
    aligned_assignment_metrics,
    compare_loop_events,
    primitive_loop_events,
    run_boundary_ledger,
)


def _panel() -> pd.DataFrame:
    start = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "session": "2024-01-02",
            "segment_id": [
                "TEST::2024-01-02::segment_00",
                "TEST::2024-01-02::segment_00",
                "TEST::2024-01-02::segment_00",
                "TEST::2024-01-02::segment_01",
                "TEST::2024-01-02::segment_01",
            ],
            "bar_ordinal": [0, 1, 2, 4, 5],
            "bar_start_timestamp": [
                start + pd.Timedelta(minutes=5 * value) for value in [0, 1, 2, 4, 5]
            ],
            "bar_complete_timestamp": [
                start + pd.Timedelta(minutes=5 * (value + 1)) for value in [0, 1, 2, 4, 5]
            ],
        }
    )


def test_primitive_loop_never_crosses_a_source_gap() -> None:
    panel = _panel()
    labels = np.asarray([1, 2, 1, 2, 1])

    events = primitive_loop_events(panel, labels, minimum_transitions=2, maximum_transitions=2)

    assert len(events) == 1
    assert events.iloc[0]["segment_id"] == "TEST::2024-01-02::segment_00"
    assert events.iloc[0]["primitive_loop_id"] == "loop_p_1-2-1"


def test_run_boundaries_reset_at_segment() -> None:
    panel = _panel()
    labels = np.asarray([1, 1, 1, 1, 1])

    runs = run_boundary_ledger(panel, labels, lineage="TEST")

    assert len(runs) == 2
    assert runs["duration"].tolist() == [3, 2]


def test_aligned_metrics_apply_mapping_not_numeric_identity() -> None:
    reference = np.asarray([0, 0, 1, 1])
    candidate = np.asarray([1, 1, 0, 0])

    metrics = aligned_assignment_metrics(
        reference,
        candidate,
        candidate_to_reference={0: 1, 1: 0},
    )

    assert metrics["bar_level_aligned_agreement"] == 1.0
    assert metrics["normalized_mutual_information"] == 1.0
    assert metrics["adjusted_rand_index"] == 1.0


def test_loop_event_agreement_uses_identical_reference_population() -> None:
    reference = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "session": ["2024-01-02", "2024-01-02"],
            "primitive_loop_id": ["loop_p_1-2-1", "loop_p_2-3-2"],
            "event_bar_ordinal": [4, 8],
        }
    )
    candidate = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "session": ["2024-01-02", "2024-01-02"],
            "primitive_loop_id": ["loop_p_1-2-1", "loop_p_9-8-9"],
            "event_bar_ordinal": [5, 8],
        }
    )

    metrics = compare_loop_events(reference, candidate, allowed_shift_bars=1)

    assert metrics["reference_event_count"] == 2
    assert metrics["same_primitive_bounded_shift_fraction"] == 0.5
    assert metrics["exact_event_agreement"] == 0.0
