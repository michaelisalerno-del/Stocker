from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.directional_signature_atlas.analysis import (
    evaluate_candidate_census,
    freeze_discovery_library,
    validate_discovery_library,
)
from stocker_research.directional_signature_atlas.evaluation import (
    baseline_predictions,
    evaluate_signature,
    null_permute_outcomes_within_period,
    survives_validation,
)
from stocker_research.directional_signature_atlas.models import (
    _prior_price_context_prequential,
    apply_atlas_controller,
)
from stocker_research.directional_signature_atlas.robustness import (
    _motif_length_variants,
    null_test_results,
    stress_signature_library,
)
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
            "round_trip_cost_bps": 10.0,
        }
    )


def _long_rule() -> Signature:
    return Signature(
        "long_state_supportive",
        "LONG",
        (Condition("state", "==", "supportive", "state"),),
    )


def _production_stage(
    year: int,
    *,
    adverse_signal: bool = False,
    neutral_only: bool = False,
    feature: str = "signal",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_index in range(60):
        month = 1 + session_index // 20
        day = 1 + session_index % 20
        session = f"{year}-{month:02d}-{day:02d}"
        for stock_index in range(12):
            fires = (session_index + stock_index) % 2 == 0
            if neutral_only:
                target, long_payoff, short_payoff = "NEUTRAL", -10.0, -10.0
            elif fires and adverse_signal:
                target, long_payoff, short_payoff = "SHORT", -50.0, 30.0
            elif fires:
                target, long_payoff, short_payoff = "LONG", 30.0, -50.0
            else:
                target, long_payoff, short_payoff = "NEUTRAL", -10.0, -10.0
            rows.append(
                {
                    "opportunity_id": f"{year}-{session_index}-{stock_index}",
                    "period": year,
                    "chronology_stage": "discovery" if year == 2025 else "validation",
                    "session": session,
                    "symbol": f"S{stock_index:02d}",
                    "decision_clock": "clock_12",
                    "decision_timestamp": pd.Timestamp(f"{session} 15:30", tz="UTC"),
                    feature: "1>3>1"
                    if fires and feature.startswith("state_motif")
                    else (
                        "other" if feature.startswith("state_motif") else ("yes" if fires else "no")
                    ),
                    "target": target,
                    "long_net_bps": long_payoff,
                    "short_net_bps": short_payoff,
                    "gross_long_return_bps": long_payoff + 10.0,
                    "round_trip_cost_bps": 10.0,
                    "movement_permission": True,
                }
            )
    return pd.DataFrame(rows)


def _production_discovery_and_validation(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    feature: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    candidates, registry = generate_bounded_candidates(
        discovery,
        {feature: "state_history" if feature.startswith("state_motif") else "test"},
        SearchCaps(100, 20, 0, 20),
        minimum_parent_support=80,
    )
    support = SupportRules(80, 30, 8, 0.25, 3, 15)
    census = evaluate_candidate_census(
        discovery,
        candidates,
        registry,
        support_rules=support,
        ordered_bins={},
        fdr_q=0.10,
    )
    frozen = freeze_discovery_library(
        census,
        candidates,
        discovery,
        retained_stage_cap=20,
        per_direction_cap=10,
    )
    validation_metrics, survivors = validate_discovery_library(
        validation,
        frozen,
        support_rules=support,
        holm_alpha=0.10,
        per_direction_cap=5,
        bootstrap_draws=20,
        bootstrap_seed=20260717,
    )
    return census, frozen, validation_metrics, survivors


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


def test_broad_cap_balances_univariate_and_pairwise_candidates() -> None:
    frame = _panel().assign(
        motif=[f"m{i}" for i in np.arange(len(_panel()))],
        location=np.where(np.arange(len(_panel())) % 2, "high", "low"),
    )
    _, registry = generate_bounded_candidates(
        frame,
        {"motif": "state_history", "location": "price_location"},
        SearchCaps(40, 20, 0, 20),
        minimum_parent_support=80,
    )
    stages = pd.Series([row["stage"] for row in registry]).value_counts()
    assert stages["univariate"] == 20
    assert stages["pairwise"] == 20


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


def test_unavailable_opposite_library_feature_forces_neutral() -> None:
    frame = (
        _panel()
        .iloc[:1]
        .assign(
            movement_permission=True,
            available_signal="yes",
            unavailable_signal=np.nan,
        )
    )
    long = Signature(
        "long_available",
        "LONG",
        (Condition("available_signal", "==", "yes", "test"),),
    )
    short = Signature(
        "short_unavailable",
        "SHORT",
        (Condition("unavailable_signal", "==", "yes", "test"),),
    )
    library = [
        {
            "signature": long.to_dict(),
            "conservative_value_bps": 1.0,
            "frozen_class_probabilities": {"LONG": 0.6, "SHORT": 0.2, "NEUTRAL": 0.2},
        },
        {
            "signature": short.to_dict(),
            "conservative_value_bps": 1.0,
            "frozen_class_probabilities": {"LONG": 0.2, "SHORT": 0.6, "NEUTRAL": 0.2},
        },
    ]
    decisions = apply_atlas_controller(
        frame,
        library,
        base_probabilities={"LONG": 0.4, "SHORT": 0.4, "NEUTRAL": 0.2},
    )
    assert decisions.iloc[0]["long_vote_count"] == 1
    assert decisions.iloc[0]["predicted_state"] == "NEUTRAL"
    assert decisions.iloc[0]["reason_code"] == "required_causal_feature_unavailable"


def test_not_equal_condition_does_not_fire_on_missing_causal_value() -> None:
    frame = pd.DataFrame({"signal": ["other", "blocked", np.nan]})
    signature = Signature(
        "not_blocked",
        "LONG",
        (Condition("signal", "!=", "blocked", "test"),),
    )
    assert apply_signature(frame, signature).tolist() == [True, False, False]


def test_baselines_use_identical_opportunity_population_and_clocks() -> None:
    frame = _panel()
    predictions = baseline_predictions(frame)
    assert set(predictions["opportunity_id"]) == set(frame["opportunity_id"])
    assert set(predictions["decision_clock"]) == set(frame["decision_clock"])


def test_prior_static_baseline_keeps_full_population_score_status_unsuffixed() -> None:
    full = pd.DataFrame(
        {
            "opportunity_id": ["o1"],
            "score_status": ["scored"],
            "period": [2025],
            "session": ["2025-01-02"],
        }
    )
    first_touch = pd.DataFrame(
        {
            "opportunity_id": ["o1"],
            "score_status": ["scored"],
            "first_touch_target": ["NEITHER"],
        }
    )
    probabilities, states, eligible = _prior_price_context_prequential(full, first_touch)
    assert probabilities.shape == (1, 3)
    assert states.tolist() == ["NEUTRAL"]
    assert not eligible.any()


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


def test_null_families_score_only_rules_their_transform_changes() -> None:
    frame = _panel(200).assign(
        chronology_stage=np.where(np.arange(200) < 100, "discovery", "validation"),
        clock_phase="middle",
        state_motif_2="1>2",
        state_motif_3=np.where(np.arange(200) % 2, "1>2>1", "2>1>2"),
        state_motif_4="2>1>2>1",
        same_orientation_repeat_bin=np.where(np.arange(200) % 3, "one", "two_plus"),
        return_6_cross_sectional_rank_bin=np.where(np.arange(200) % 2, "upper_quintile", "middle"),
        universe_breadth_bin=np.where(np.arange(200) % 4, "positive", "negative"),
        decision_timestamp=pd.date_range("2024-01-01", periods=200, freq="5min", tz="UTC"),
        round_trip_cost_bps=10.0,
    )
    signatures = [
        Signature(
            "clock",
            "LONG",
            (Condition("decision_clock", "==", "clock_12", "clock"),),
        ),
        Signature(
            "motif",
            "LONG",
            (Condition("state_motif_3", "==", "1>2>1", "state_history"),),
        ),
        Signature(
            "repeat",
            "LONG",
            (
                Condition(
                    "same_orientation_repeat_bin",
                    "==",
                    "one",
                    "state_history",
                ),
            ),
        ),
        Signature(
            "stock_cross_section",
            "LONG",
            (
                Condition(
                    "return_6_cross_sectional_rank_bin",
                    "==",
                    "upper_quintile",
                    "cross_sectional",
                ),
            ),
        ),
        Signature(
            "breadth_environment",
            "LONG",
            (
                Condition(
                    "universe_breadth_bin",
                    "==",
                    "positive",
                    "cross_sectional_environment",
                ),
            ),
        ),
    ]
    library = [
        {"signature": signature.to_dict(), "discovery_metrics": {}} for signature in signatures
    ]
    atlas = frame[["period", "chronology_stage"]].assign(predicted_state="NEUTRAL")
    nulls = null_test_results(frame, library, signatures, atlas, seed=20260717).set_index("null")
    assert nulls.loc["feature_rows_wrong_session_lag", "tested_signatures"] == 4
    assert nulls.loc["stock_identity_permuted_within_timestamp", "tested_signatures"] == 1
    assert nulls.loc["state_history_permuted_within_clock_phase", "tested_signatures"] == 1


def test_empty_stress_library_retains_machine_readable_schema() -> None:
    frame = _production_stage(2025)
    delayed = frame[
        ["opportunity_id", "target", "gross_long_return_bps", "long_net_bps", "short_net_bps"]
    ].rename(
        columns={
            "long_net_bps": "net_long_return_bps",
            "short_net_bps": "net_short_return_bps",
        }
    )
    stress, leave_one_out = stress_signature_library(
        frame,
        [],
        delayed,
        ordered_bins={},
    )
    assert stress.empty and leave_one_out.empty
    assert "stress" in stress
    assert "direct_cross_sectional_features_recomputed" in leave_one_out


def test_synthetic_persistent_long_signature_survives_validation() -> None:
    frame = _panel()
    discovery = evaluate_signature(frame.query("period == 2024"), _long_rule())
    validation = evaluate_signature(frame.query("period == 2025"), _long_rule())
    assert survives_validation(discovery, validation, require_double_cost=False)


def test_production_census_persistent_long_survives_fdr_freeze_and_validation() -> None:
    census, frozen, validation_metrics, survivors = _production_discovery_and_validation(
        _production_stage(2025), _production_stage(2026), "signal"
    )
    assert census["discovery_eligible"].any()
    assert any(
        entry["signature"]["direction"] == "LONG"
        and entry["signature"]["conditions"][0]["value"] == "yes"
        for entry in frozen
    )
    assert validation_metrics["validation_survived"].any()
    assert any(entry["signature"]["direction"] == "LONG" for entry in survivors)


def test_synthetic_discovery_only_overfit_rule_fails_validation() -> None:
    frame = _panel()
    validation = frame.query("period == 2025").copy()
    validation.loc[validation["state"] == "supportive", "long_net_bps"] = -20.0
    discovery_metrics = evaluate_signature(frame.query("period == 2024"), _long_rule())
    validation_metrics = evaluate_signature(validation, _long_rule())
    assert not survives_validation(discovery_metrics, validation_metrics, False)


def test_production_census_discovery_only_rule_freezes_then_fails_validation() -> None:
    _, frozen, validation_metrics, survivors = _production_discovery_and_validation(
        _production_stage(2025),
        _production_stage(2026, adverse_signal=True),
        "signal",
    )
    assert any(entry["signature"]["direction"] == "LONG" for entry in frozen)
    assert not validation_metrics["validation_survived"].any()
    assert survivors == []


def test_synthetic_short_signature_is_not_inverse_long_automatically() -> None:
    short = Signature("short_other", "SHORT", (Condition("state", "==", "other", "state"),))
    assert short.conditions != _long_rule().conditions
    assert short.direction == "SHORT"


def test_synthetic_neutral_population_does_not_manufacture_directional_signatures() -> None:
    frame = _panel().assign(target="NEUTRAL", long_net_bps=-10.0, short_net_bps=-10.0)
    metrics = evaluate_signature(frame, _long_rule())
    assert metrics["mean_directional_net_bps"] < 0
    assert metrics["directional_lift"] <= 0


def test_production_census_neutral_population_freezes_no_directional_rule() -> None:
    census, frozen, _, survivors = _production_discovery_and_validation(
        _production_stage(2025, neutral_only=True),
        _production_stage(2026, neutral_only=True),
        "signal",
    )
    assert not census["discovery_eligible"].any()
    assert frozen == []
    assert survivors == []


def test_synthetic_state_motif_with_value_is_recovered() -> None:
    frame = _panel().assign(state_motif_3=np.where(np.arange(120) % 3, "1>3>1", "0>1>0"))
    candidates, _ = generate_bounded_candidates(
        frame.query("period == 2024"),
        {"state_motif_3": "state_history"},
        SearchCaps(20, 20, 0, 20),
    )
    assert any(rule.conditions[0].value == "1>3>1" for rule in candidates)


def test_production_census_recovers_causal_motif_through_validation() -> None:
    census, frozen, _, survivors = _production_discovery_and_validation(
        _production_stage(2025, feature="state_motif_3"),
        _production_stage(2026, feature="state_motif_3"),
        "state_motif_3",
    )
    assert census["discovery_eligible"].any()
    assert any(entry["signature"]["conditions"][0]["value"] == "1>3>1" for entry in frozen)
    assert any(entry["signature"]["conditions"][0]["value"] == "1>3>1" for entry in survivors)


def test_rare_levels_remain_in_pre_support_candidate_census() -> None:
    frame = _production_stage(2025)
    frame.loc[frame.index[0], "signal"] = "rare"
    candidates, registry = generate_bounded_candidates(
        frame,
        {"signal": "test"},
        SearchCaps(100, 20, 0, 20),
        minimum_parent_support=80,
    )
    support = SupportRules(80, 30, 8, 0.25, 3, 15)
    census = evaluate_candidate_census(
        frame,
        candidates,
        registry,
        support_rules=support,
        ordered_bins={},
        fdr_q=0.10,
    )
    rare = census.loc[census["conditions_json"].str.contains('"value":"rare"', regex=False)]
    assert len(rare) == 2
    assert rare["rejection_reasons_json"].str.contains("insufficient_rows").all()


def test_motif_length_stress_substitutes_causal_neighbouring_lengths() -> None:
    frame = pd.DataFrame(
        {
            "state_motif_2": ["2>3"],
            "state_motif_3": ["1>2>3"],
            "state_motif_4": ["0>1>2>3"],
        }
    )
    signature = Signature(
        "motif",
        "LONG",
        (Condition("state_motif_3", "==", "1>2>3", "state_history"),),
    )
    variants = _motif_length_variants(frame, signature, 0)
    assert {(length, token) for length, token, _ in variants} == {
        (2, "2>3"),
        (4, "0>1>2>3"),
    }


def test_future_leaking_motif_is_rejected() -> None:
    bad = Signature(
        "bad_future_motif",
        "LONG",
        (Condition("future_state_motif_3", "==", "1>3>1", "state_history"),),
    )
    with pytest.raises(ValueError, match="future"):
        apply_signature(_panel(), bad)
