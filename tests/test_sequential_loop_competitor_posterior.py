from __future__ import annotations

import math

from stocker_research.sequential_loop_competitor_veto import (
    PosteriorSnapshot,
    compatibility_status,
    initial_posterior,
    update_posterior,
)


def test_loop_is_eliminated_only_after_incompatible_transition_is_observable() -> None:
    cycles = {
        "good": "2->4->2",
        "bad_same_first_leg": "2->4->3->2",
        "bad_other_leg": "2->5->2",
    }
    anchor = initial_posterior({"good": 0.25, "bad_same_first_leg": 0.20, "bad_other_leg": 0.15})

    before_transition = update_posterior(anchor, cycles, 2, ())
    after_first_transition = update_posterior(anchor, cycles, 2, (4,))
    after_second_transition = update_posterior(after_first_transition, cycles, 2, (4, 2))

    assert before_transition.eliminated == ()
    assert after_first_transition.eliminated == ("bad_other_leg",)
    assert set(after_second_transition.eliminated) == {
        "bad_other_leg",
        "bad_same_first_leg",
    }
    assert after_second_transition.statuses["good"] == "completed"


def test_posterior_normalises_over_compatible_loops_and_explicit_unknown_mass() -> None:
    anchor = initial_posterior({"good": 0.20, "bad": 0.15})
    updated = update_posterior(
        anchor,
        {"good": "4->2->4", "bad": "4->6->4"},
        4,
        (2,),
        evidence_likelihoods={"good": 0.8, "bad": 0.4},
    )

    assert updated.known["bad"] == 0.0
    assert updated.unknown > 0.0
    assert math.isclose(sum(updated.known.values()) + updated.unknown, 1.0)


def test_unseen_but_possible_evidence_keeps_nonzero_smoothed_mass() -> None:
    anchor = PosteriorSnapshot(
        known={"good": 0.3, "other": 0.2},
        unknown=0.5,
        statuses={"good": "compatible", "other": "compatible"},
        eliminated=(),
        completed=(),
    )
    updated = update_posterior(
        anchor,
        {"good": "5->6->5", "other": "5->7->5"},
        5,
        (),
        evidence_likelihoods={"good": 0.0, "other": 0.0},
        minimum_possible_likelihood=0.05,
    )

    assert updated.known["good"] > 0.0
    assert updated.known["other"] > 0.0
    assert updated.unknown > anchor.unknown


def test_future_transitions_do_not_change_a_frozen_checkpoint() -> None:
    anchor = initial_posterior({"good": 0.25, "bad": 0.25})
    cycles = {"good": "4->2->4", "bad": "4->6->4"}

    frozen = update_posterior(anchor, cycles, 4, ())
    _future = update_posterior(frozen, cycles, 4, (2,))

    assert frozen == update_posterior(anchor, cycles, 4, ())


def test_compatible_loop_cannot_be_eliminated_by_later_payoff() -> None:
    assert compatibility_status("5->6->5", 5, (6,)) == "compatible"
    assert compatibility_status("5->6->5", 5, (6, 5)) == "completed"
