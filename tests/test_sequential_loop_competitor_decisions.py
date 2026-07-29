from __future__ import annotations

import pandas as pd

from stocker_research.sequential_loop_competitor_veto import (
    DecisionConfig,
    apply_irreversible_decisions,
    apply_veto_accounting,
    classify_decision,
    initial_posterior,
    summarise_posterior,
    update_posterior,
)


def test_synthetic_bad_competitor_elimination_changes_unresolved_to_admit() -> None:
    cycles = {
        "good": "4->2->4",
        "bad_a": "4->6->4",
        "bad_b": "4->3->4",
    }
    classes = {"good": "good", "bad_a": "bad", "bad_b": "bad"}
    means = {"good": 120.0, "bad_a": -80.0, "bad_b": -60.0}
    stds = {"good": 5.0, "bad_a": 10.0, "bad_b": 10.0}
    anchor = initial_posterior({"good": 0.25, "bad_a": 0.20, "bad_b": 0.15})
    anchor_summary = summarise_posterior(anchor, classes, means, stds, bars_remaining=24)
    resolved = update_posterior(
        anchor,
        cycles,
        4,
        (2,),
        evidence_likelihoods={"good": 4.0},
        unknown_likelihood=0.05,
    )
    resolved_summary = summarise_posterior(resolved, classes, means, stds, bars_remaining=18)

    assert classify_decision(anchor_summary, DecisionConfig()) == "unresolved"
    assert classify_decision(resolved_summary, DecisionConfig()) == "admit"


def test_bad_competitors_remaining_cause_abstention() -> None:
    snapshot = initial_posterior({"good": 0.2, "bad": 0.4})
    summary = summarise_posterior(
        snapshot,
        {"good": "good", "bad": "bad"},
        {"good": 50.0, "bad": -60.0},
        {"good": 20.0, "bad": 20.0},
        bars_remaining=18,
    )

    assert classify_decision(summary, DecisionConfig()) == "unresolved"


def test_target_elimination_rejects_and_rejection_is_irreversible() -> None:
    timeline = pd.DataFrame(
        {
            "opportunity_id": ["a", "a", "a"],
            "checkpoint_timestamp": pd.to_datetime(
                ["2025-01-01T14:00:00Z", "2025-01-01T14:05:00Z", "2025-01-01T14:10:00Z"],
                utc=True,
            ),
            "proposed_decision": ["unresolved", "reject", "admit"],
        }
    )

    result = apply_irreversible_decisions(timeline)

    assert result["decision_state"].tolist() == ["unresolved", "reject", "reject"]
    assert result["reason_codes"].iloc[-1] == "prior_rejection_irreversible"


def test_rejected_opportunity_cannot_be_replaced_in_veto_accounting() -> None:
    base = pd.DataFrame(
        {
            "opportunity_id": ["original_loss", "original_win"],
            "net_payoff_bps": [-40.0, 30.0],
        }
    )
    decisions = pd.DataFrame(
        {
            "opportunity_id": ["original_loss", "original_win"],
            "decision_state": ["reject", "admit"],
        }
    )

    result = apply_veto_accounting(base, decisions)

    assert result["opportunity_id"].tolist() == base["opportunity_id"].tolist()
    assert result["policy_net_payoff_bps"].tolist() == [0.0, 30.0]
    assert result["veto_value_bps"].tolist() == [40.0, 0.0]
    assert result["existing_position_action"].eq("unchanged").all()


def test_irreversible_decisions_are_isolated_by_registered_track() -> None:
    timeline = pd.DataFrame(
        {
            "opportunity_id": ["o1", "o1", "o1", "o1"],
            "track": ["named", "general", "named", "general"],
            "checkpoint_timestamp": pd.to_datetime(
                [
                    "2025-01-02 14:35Z",
                    "2025-01-02 14:35Z",
                    "2025-01-02 14:40Z",
                    "2025-01-02 14:40Z",
                ]
            ),
            "proposed_decision": ["reject", "unresolved", "admit", "admit"],
        }
    )

    result = apply_irreversible_decisions(
        timeline,
        identity_columns=("opportunity_id", "track"),
    )

    assert result["decision_state"].tolist() == [
        "reject",
        "unresolved",
        "reject",
        "admit",
    ]
