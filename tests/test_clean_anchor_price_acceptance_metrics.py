from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stocker_research.clean_anchor_price_acceptance.immutable_ledger import (
    ProspectiveAcceptanceLedger,
)
from stocker_research.clean_anchor_price_acceptance.metrics import (
    acceptance_diagnostics,
    four_cell_interaction,
    paired_variant_comparison,
    session_block_bootstrap,
    veto_accounting,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "opportunity_id": ["a", "b", "c", "d"],
            "period": [2025] * 4,
            "session_date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "loop_id": ["cycle_04"] * 4,
            "symbol": ["A", "B", "C", "D"],
            "static_anchor_veto_pass": [False, False, True, True],
            "price_acceptance_pass": [False, True, False, True],
            "acceptance_balance_bps": [-20.0, 5.0, -5.0, 20.0],
            "net_payoff_bps": [-30.0, -10.0, -5.0, 40.0],
            "gross_payoff_bps": [-20.0, 0.0, 5.0, 50.0],
            "total_cost_bps": [10.0] * 4,
        }
    )


def test_four_cell_interaction_preserves_all_registered_cells() -> None:
    result = four_cell_interaction(_source(), group_columns=["period", "loop_id"])

    assert len(result) == 4
    assert set(result["interaction_cell"]) == {
        "anchor_fail|acceptance_fail",
        "anchor_fail|acceptance_pass",
        "anchor_pass|acceptance_fail",
        "anchor_pass|acceptance_pass",
    }
    assert result["opportunities"].eq(1).all()


def test_veto_value_is_losses_avoided_minus_rejected_winners() -> None:
    source = _source()
    admitted = source["static_anchor_veto_pass"] & source["price_acceptance_pass"]
    result = veto_accounting(source, admitted=admitted)

    assert result["losses_avoided_bps"] == 45.0
    assert result["profits_mistakenly_rejected_bps"] == 0.0
    assert result["veto_value_bps"] == 45.0
    assert result["coverage"] == 0.25


def test_paired_comparison_rejects_population_mismatch() -> None:
    decisions = pd.DataFrame(
        {
            "variant": ["A", "A", "D"],
            "opportunity_id": ["one", "two", "one"],
            "policy_net_payoff_bps": [1.0, 2.0, 3.0],
            "session_date": ["2025-01-02"] * 3,
        }
    )

    with pytest.raises(ValueError, match="paired populations differ"):
        paired_variant_comparison(decisions, treatment="D", control="A")


def test_paired_comparison_uses_source_row_identity() -> None:
    decisions = pd.DataFrame(
        {
            "variant": ["A", "A", "D", "D"],
            "opportunity_id": ["one", "two", "one", "two"],
            "policy_net_payoff_bps": [-10.0, 20.0, 0.0, 20.0],
            "session_date": ["2025-01-02", "2025-01-03"] * 2,
        }
    )

    result = paired_variant_comparison(decisions, treatment="D", control="A")

    assert result["paired_opportunities"] == 2
    assert result["paired_total_difference_bps"] == 10.0
    assert result["paired_mean_difference_bps"] == 5.0


def test_session_block_bootstrap_is_reproducible() -> None:
    differences = pd.DataFrame(
        {
            "period": [2025] * 6,
            "session_date": pd.date_range("2025-01-02", periods=6, freq="D").astype(str),
            "difference_bps": [-1.0, 2.0, 3.0, -2.0, 4.0, 5.0],
        }
    )

    first = session_block_bootstrap(differences, resamples=100, block_length=2, seed=7)
    second = session_block_bootstrap(differences, resamples=100, block_length=2, seed=7)

    assert first == second


def test_acceptance_bins_are_fixed_by_cost_not_target_quantiles() -> None:
    result = acceptance_diagnostics(_source(), round_trip_cost_bps=10.0)

    assert result["acceptance_bin"].astype(str).tolist() == [
        "acceptance_balance<=-cost",
        "0<acceptance_balance<=cost",
        "-cost<acceptance_balance<=0",
        "acceptance_balance>cost",
    ]


def _forecast() -> dict[str, object]:
    freeze = "2026-07-17T14:40:00+00:00"
    return {
        "run_id": "run",
        "git_sha": "abc",
        "contract_hash": "contract",
        "data_snapshot_hash": "new-snapshot",
        "opportunity_id": "opp",
        "event_lineage_id": "lineage",
        "symbol": "ASTS",
        "session": "2026-07-17",
        "loop_id": "cycle_04",
        "orientation": "state_4",
        "frozen_direction": 1,
        "anchor_timestamp": "2026-07-17T14:30:00+00:00",
        "anchor_reference_price": 40.0,
        "static_anchor_veto_score": 2.0,
        "static_anchor_veto_pass": True,
        "static_anchor_veto_reason_codes": "pass",
        "checkpoint_timestamp": "2026-07-17T14:35:00+00:00",
        "checkpoint_open": 40.0,
        "checkpoint_high": 41.0,
        "checkpoint_low": 39.9,
        "checkpoint_close": 40.5,
        "signed_close_return_bps": 125.0,
        "favourable_excursion_bps": 250.0,
        "adverse_excursion_bps": 25.0,
        "acceptance_balance_bps": 225.0,
        "price_acceptance_pass": True,
        "predicted_remaining_range_bps": None,
        "range_permission_pass": None,
        "next_entry_timestamp": freeze,
        "original_terminal_timestamp": "2026-07-17T16:35:00+00:00",
        "variant_decisions": {"A": True, "D": True},
        "feature_availability_timestamps": {"checkpoint": freeze},
        "training_cutoff": "2026-07-17T14:29:59+00:00",
        "forecast_freeze_timestamp": freeze,
    }


def test_prospective_forecast_is_create_only_and_execution_free(tmp_path: Path) -> None:
    ledger = ProspectiveAcceptanceLedger(tmp_path, opened_periods={2023, 2025})
    path = ledger.append_forecast(_forecast(), holdout=True)
    payload = json.loads(path.read_text())

    assert payload["execution_enabled"] is False
    assert payload["broker_connection_enabled"] is False
    with pytest.raises(FileExistsError):
        ledger.append_forecast(_forecast(), holdout=True)


def test_prospective_holdout_rejects_opened_surface(tmp_path: Path) -> None:
    record = _forecast()
    record["session"] = "2025-07-17"
    ledger = ProspectiveAcceptanceLedger(tmp_path, opened_periods={2023, 2025})

    with pytest.raises(ValueError, match="opened period"):
        ledger.append_forecast(record, holdout=True)


def test_outcome_is_separate_create_only_record(tmp_path: Path) -> None:
    ledger = ProspectiveAcceptanceLedger(tmp_path, opened_periods={2023, 2025})
    ledger.append_forecast(_forecast(), holdout=True)
    outcome = {
        "outcome_id": "outcome",
        "opportunity_id": "opp",
        "settlement_timestamp": "2026-07-17T16:35:00+00:00",
        "net_payoff_bps": 12.0,
    }

    path = ledger.append_outcome(outcome)

    assert path.parent.name == "outcomes"
    with pytest.raises(FileExistsError):
        ledger.append_outcome(outcome)
