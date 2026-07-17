from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.directional_signature_atlas.evaluation import (
    baseline_predictions,
    evaluate_signature,
    null_permute_outcomes_within_period,
    survives_validation,
)
from stocker_research.directional_signature_atlas.models import apply_atlas_controller
from stocker_research.directional_signature_atlas.robustness import null_test_results
from stocker_research.directional_signature_atlas.signatures import (
    Condition,
    SearchCaps,
    Signature,
    SupportRules,
    apply_multiple_testing,
    apply_signature,
    controller_decision,
    generate_bounded_candidates,
    passes_support,
    retain_ranked_candidates,
    split_libraries,
)


def _panel(rows: int = 120) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "opportunity_id": [f"o{i}" for i in index],
            "period": np.where(index < rows // 2, 2024, 2025),
            "session": [f"2024-{1 + i // 20:02d}-{1 + i % 20:02d}" for i in index],
            "symbol": [f"S{i % 10:02d}" for i in index],
            "decision_clock": np.where(index % 2, "clock_12", "clock_36"),
            "state": np.where(index % 3, "supportive", "other"),
            "return_1": np.where(index % 2, 1.0, -1.0),
            "target": np.where(index % 3, "LONG", "NEUTRAL"),
            "long_net_bps": np.where(index % 3, 12.0, -10.0),
            "short_net_bps": np.where(index % 3, -32.0, -10.0),
        }
    )


def _long_rule() -> Signature:
    return Signature(
        "long_state_supportive",
        "LONG",
        (Condition("state", "==", "supportive", "state"),),
    )


def test_discovery_rules_use_discovery_data_only() -> None:
    frame = _panel()
    candidates, _ = generate_bounded_candidates(
        frame.loc[frame["period"] == 2024],
        {"state": "state"},
        SearchCaps(20, 20, 10, 10),
    )
    assert candidates
    assert all("2025" not in rule.signature_id for rule in candidates)


def test_validation_outcomes_cannot_regenerate_discovery_rules() -> None:
    discovery, _ = generate_bounded_candidates(
        _panel().query("period == 2024"), {"state": "state"}, SearchCaps(20, 20, 10, 10)
    )
    validation = _panel().query("period == 2025").assign(state="new_validation_level")
    frozen_ids = [rule.signature_id for rule in discovery]
    assert [rule.signature_id for rule in discovery] == frozen_ids
    assert all("new_validation_level" not in rule.signature_id for rule in discovery)
    assert not any(apply_signature(validation, rule).all() for rule in discovery)


def test_final_holdout_cannot_modify_validated_rules() -> None:
    rule = _long_rule()
    before = rule.to_dict()
    evaluate_signature(_panel().query("period == 2025"), rule)
    assert rule.to_dict() == before


def test_search_caps_are_enforced() -> None:
    frame = _panel().assign(extra=np.arange(len(_panel())) % 5)
    candidates, registry = generate_bounded_candidates(
        frame,
        {"state": "state", "extra": "price"},
        SearchCaps(univariate_and_pairwise=3, triples=1, tree=0, retained=2),
    )
    assert len([row for row in registry if row["stage"] in {"univariate", "pairwise"}]) <= 3
    retained = retain_ranked_candidates(
        candidates, {candidate.signature_id: 1.0 for candidate in candidates}, cap=2
    )
    assert len(retained) <= 2


def test_support_rules_are_enforced() -> None:
    rules = SupportRules(80, 30, 8, 0.25, 3, 15)
    assert not passes_support(_panel().iloc[:10], "LONG", rules)[0]


def test_rule_failing_stock_breadth_is_rejected() -> None:
    rules = SupportRules(20, 5, 8, 0.80, 1, 5)
    frame = _panel().iloc[:40].assign(symbol="ONLY")
    passed, reasons = passes_support(frame, "LONG", rules)
    assert not passed
    assert "insufficient_stocks" in reasons


def test_rule_dominated_by_one_stock_is_rejected() -> None:
    rules = SupportRules(20, 5, 2, 0.25, 1, 5)
    frame = _panel().iloc[:40].copy()
    frame["symbol"] = ["DOM"] * 30 + [f"S{i}" for i in range(10)]
    passed, reasons = passes_support(frame, "LONG", rules)
    assert not passed
    assert "stock_concentration" in reasons


def test_multiple_testing_correction_is_applied() -> None:
    corrected = apply_multiple_testing([0.001, 0.02, 0.20], method="fdr_bh")
    assert corrected[0] == pytest.approx(0.003)
    assert corrected[1] == pytest.approx(0.03)
    holm = apply_multiple_testing([0.001, 0.02, 0.20], method="holm")
    assert holm[0] == pytest.approx(0.003)


def test_failed_rules_remain_in_candidate_registry() -> None:
    _, registry = generate_bounded_candidates(
        _panel().iloc[:10], {"state": "state"}, SearchCaps(20, 20, 0, 20)
    )
    assert registry
    assert all("rejection_reasons" in row for row in registry)


def test_long_and_short_libraries_remain_separate() -> None:
    short = Signature("short", "SHORT", (Condition("state", "==", "other", "state"),))
    long_library, short_library = split_libraries([_long_rule(), short])
    assert [rule.direction for rule in long_library] == ["LONG"]
    assert [rule.direction for rule in short_library] == ["SHORT"]


def test_conflicting_long_and_short_rules_produce_neutral() -> None:
    decision = controller_decision(1, 1, movement_permitted=True, aggregate_value_positive=True)
    assert decision.state == "NEUTRAL"
    assert decision.reason == "conflicting_votes"


def test_no_rule_firing_produces_neutral() -> None:
    assert controller_decision(0, 0, True, True).state == "NEUTRAL"


def test_movement_permission_failure_produces_neutral() -> None:
    assert controller_decision(1, 0, False, True).state == "NEUTRAL"


def test_atlas_votes_are_not_weighted_by_holdout_performance() -> None:
    result = controller_decision(1, 0, True, True, final_holdout_weights={"winner": 999.0})
    assert result.long_votes == 1
    assert result.state == "LONG"


def test_atlas_requires_positive_frozen_conservative_value() -> None:
    frame = _panel().iloc[[1]].assign(movement_permission=True)
    entry = {
        "signature": _long_rule().to_dict(),
        "frozen_class_probabilities": {"LONG": 0.6, "SHORT": 0.2, "NEUTRAL": 0.2},
        "conservative_value_bps": 0.0,
    }
    neutral = apply_atlas_controller(
        frame, [entry], base_probabilities={"LONG": 0.4, "SHORT": 0.4, "NEUTRAL": 0.2}
    )
    assert neutral.iloc[0]["predicted_state"] == "NEUTRAL"
    assert neutral.iloc[0]["reason_code"] == "non_positive_conservative_value"
    entry["conservative_value_bps"] = 1.0
    directional = apply_atlas_controller(
        frame, [entry], base_probabilities={"LONG": 0.4, "SHORT": 0.4, "NEUTRAL": 0.2}
    )
    assert directional.iloc[0]["predicted_state"] == "LONG"


def test_baselines_use_identical_opportunity_population_and_clocks() -> None:
    frame = _panel()
    predictions = baseline_predictions(frame)
    assert set(predictions["opportunity_id"]) == set(frame["opportunity_id"])
    assert set(predictions["decision_clock"]) == set(frame["decision_clock"])


def test_one_bar_momentum_and_reversal_baselines_are_correct() -> None:
    frame = _panel().iloc[:2].copy()
    frame["return_1"] = [1.0, -1.0]
    predictions = baseline_predictions(frame).set_index("opportunity_id")
    assert list(predictions["one_bar_momentum"]) == ["LONG", "SHORT"]
    assert list(predictions["one_bar_reversal"]) == ["SHORT", "LONG"]


def test_null_permutation_breaks_synthetic_directional_relationship() -> None:
    frame = _panel().query("period == 2024").copy()
    original = evaluate_signature(frame, _long_rule())["mean_directional_net_bps"]
    permuted = null_permute_outcomes_within_period(frame, seed=20260717)
    null_effect = evaluate_signature(permuted, _long_rule())["mean_directional_net_bps"]
    assert original > null_effect


def test_all_frozen_null_families_execute_with_validation_correction() -> None:
    frame = _panel(200).assign(
        clock_phase="middle",
        state_motif_2="1>2",
        state_motif_3="1>2>1",
        state_motif_4="2>1>2>1",
        decision_timestamp=pd.date_range("2024-01-01", periods=200, freq="5min", tz="UTC"),
        round_trip_cost_bps=10.0,
    )
    library = [{"signature": _long_rule().to_dict(), "discovery_metrics": {}}]
    atlas = frame[["period"]].assign(predicted_state="NEUTRAL")
    nulls = null_test_results(frame, library, [_long_rule()], atlas, seed=20260717)
    assert len(nulls) == 7
    assert nulls["persistent_positive_count"].ge(0).all()


def test_synthetic_persistent_long_signature_survives_validation() -> None:
    frame = _panel()
    discovery = evaluate_signature(frame.query("period == 2024"), _long_rule())
    validation = evaluate_signature(frame.query("period == 2025"), _long_rule())
    assert survives_validation(discovery, validation, require_double_cost=False)


def test_synthetic_discovery_only_overfit_rule_fails_validation() -> None:
    frame = _panel()
    validation = frame.query("period == 2025").copy()
    validation.loc[validation["state"] == "supportive", "long_net_bps"] = -20.0
    discovery_metrics = evaluate_signature(frame.query("period == 2024"), _long_rule())
    validation_metrics = evaluate_signature(validation, _long_rule())
    assert not survives_validation(discovery_metrics, validation_metrics, False)


def test_synthetic_short_signature_is_not_inverse_long_automatically() -> None:
    short = Signature("short_other", "SHORT", (Condition("state", "==", "other", "state"),))
    assert short.conditions != _long_rule().conditions
    assert short.direction == "SHORT"


def test_synthetic_neutral_population_does_not_manufacture_directional_signatures() -> None:
    frame = _panel().assign(target="NEUTRAL", long_net_bps=-10.0, short_net_bps=-10.0)
    metrics = evaluate_signature(frame, _long_rule())
    assert metrics["mean_directional_net_bps"] < 0
    assert metrics["directional_lift"] <= 0


def test_synthetic_state_motif_with_value_is_recovered() -> None:
    frame = _panel().assign(state_motif_3=np.where(np.arange(120) % 3, "1>3>1", "0>1>0"))
    candidates, _ = generate_bounded_candidates(
        frame.query("period == 2024"),
        {"state_motif_3": "state_history"},
        SearchCaps(20, 20, 0, 20),
    )
    assert any(rule.conditions[0].value == "1>3>1" for rule in candidates)


def test_future_leaking_motif_is_rejected() -> None:
    bad = Signature(
        "bad_future_motif",
        "LONG",
        (Condition("future_state_motif_3", "==", "1>3>1", "state_history"),),
    )
    with pytest.raises(ValueError, match="future"):
        apply_signature(_panel(), bad)
