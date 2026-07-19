from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stocker_research.observable_event_ranking_v1.contract import (
    REQUIRED_SAFETY_FLAGS,
    canonical_hash,
    frozen_contract,
)
from stocker_research.observable_event_ranking_v1.events import (
    assign_decision_time,
    leave_one_out_medians,
)
from stocker_research.observable_event_ranking_v1.targets import target_reference_times


def test_frozen_contract_carries_every_required_fail_closed_safety_flag() -> None:
    contract = frozen_contract()

    assert contract["safety"] == REQUIRED_SAFETY_FLAGS
    assert canonical_hash(contract) == canonical_hash(frozen_contract())
    assert contract["primary_hypothesis"]["event_family"] == ("E1_POSITIVE_RELATIVE_ACCELERATION")
    assert contract["primary_hypothesis"]["model"] == "M1_POOLED_LINEAR_RANKER"


def test_leave_one_out_median_uses_only_other_stocks() -> None:
    medians = leave_one_out_medians([1.0, 2.0, 100.0, 4.0])

    assert medians == pytest.approx([4.0, 4.0, 2.0, 2.0])


def test_event_at_exact_grid_is_deferred_without_proven_pre_score_availability() -> None:
    exact_grid = datetime(2025, 7, 7, 14, 0, tzinfo=UTC)  # 10:00 New York

    assigned = assign_decision_time(
        confirmation_time=exact_grid,
        availability_time=exact_grid,
        exact_grid_available_before_scoring=False,
    )

    assert assigned == datetime(2025, 7, 7, 14, 30, tzinfo=UTC)


def test_exact_grid_can_be_used_only_with_proven_earlier_availability() -> None:
    exact_grid = datetime(2025, 7, 7, 14, 0, tzinfo=UTC)
    available = datetime(2025, 7, 7, 13, 59, 59, tzinfo=UTC)

    assigned = assign_decision_time(
        confirmation_time=exact_grid,
        availability_time=available,
        exact_grid_available_before_scoring=True,
    )

    assert assigned == exact_grid


def test_target_reference_has_one_complete_bar_dispatch_delay() -> None:
    decision = datetime(2025, 7, 7, 14, 0, tzinfo=UTC)

    references = target_reference_times(decision)

    assert references.immediate_next_bar_open == datetime(2025, 7, 7, 14, 0, tzinfo=UTC)
    assert references.delayed_entry_reference == datetime(2025, 7, 7, 14, 5, tzinfo=UTC)
    assert references.exit_30m == datetime(2025, 7, 7, 14, 35, tzinfo=UTC)
    assert references.exit_60m == datetime(2025, 7, 7, 15, 5, tzinfo=UTC)
