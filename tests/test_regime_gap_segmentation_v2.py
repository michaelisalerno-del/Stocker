from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.regime_gap_segmentation_v2 import (
    annotate_causal_segments,
    causal_segment_groups,
    reset_stateful_array_by_segment,
)


def _bars(ordinals: list[int], *, session: str = "2024-01-02") -> pd.DataFrame:
    start = pd.Timestamp(f"{session} 14:30:00", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "session": session,
            "bar_ordinal": ordinals,
            "bar_start_timestamp": [start + pd.Timedelta(minutes=5 * value) for value in ordinals],
        }
    )


def test_contiguous_session_is_one_complete_segment() -> None:
    annotated, gaps = annotate_causal_segments(
        _bars([0, 1, 2]), expected_bars={("TEST", "2024-01-02"): 3}
    )

    assert annotated["segment_id"].nunique() == 1
    assert annotated["session_source_complete"].all()
    assert annotated["segment_start_reason"].tolist() == ["session_open", "continued", "continued"]
    assert gaps.empty


def test_internal_gap_creates_two_segments_and_a_ledger_row() -> None:
    annotated, gaps = annotate_causal_segments(
        _bars([0, 1, 3, 4]), expected_bars={("TEST", "2024-01-02"): 5}
    )

    assert annotated["segment_index"].tolist() == [0, 0, 1, 1]
    assert annotated["session_source_complete"].eq(False).all()
    assert gaps[["previous_bar_ordinal", "next_bar_ordinal"]].to_dict("records") == [
        {"previous_bar_ordinal": 1, "next_bar_ordinal": 3}
    ]
    assert len(causal_segment_groups(annotated)) == 2


def test_duplicate_natural_key_fails_closed() -> None:
    frame = pd.concat([_bars([0, 1]), _bars([1])], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        annotate_causal_segments(frame, expected_bars={("TEST", "2024-01-02"): 2})


def test_timestamp_discontinuity_splits_even_when_ordinal_is_contiguous() -> None:
    frame = _bars([0, 1, 2])
    frame.loc[2, "bar_start_timestamp"] += pd.Timedelta(minutes=5)

    annotated, _ = annotate_causal_segments(frame, expected_bars={("TEST", "2024-01-02"): 3})

    assert annotated["segment_index"].tolist() == [0, 0, 1]


def test_stateful_array_reset_does_not_bridge_gap() -> None:
    annotated, _ = annotate_causal_segments(
        _bars([0, 1, 3]), expected_bars={("TEST", "2024-01-02"): 4}
    )
    values = np.asarray([2, 2, 2], dtype=np.int16)

    ages = reset_stateful_array_by_segment(
        values,
        causal_segment_groups(annotated),
        initial_value=1,
        update=lambda previous, current: previous + 1 if current == 2 else 1,
    )

    assert ages.tolist() == [1, 2, 1]
