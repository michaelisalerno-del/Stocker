from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from stocker_research.profitable_loop_episode_anatomy import (
    attach_episode_membership,
    block_circular_pair_shift,
    build_episode_ledgers,
    build_synchronized_panel,
    classify_episode,
    collapse_stock_contributions,
    common_factor_diagnostic,
    concentration_attribution,
    decode_history_token,
    decompose_payoff_components,
    early_leader_checkpoints,
    exact_rerun_identity,
    four_way_counterfactual,
    frozen_run_git_head,
    poisson_binomial_null,
    recompute_component_summary_after_stock_removal,
    reproduce_exploratory_census,
    validate_causal_indicators,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/frozen"
    / "20260717-profitable-loop-episode-anatomy-v1"
)
ARTIFACTS = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/artifacts"
    / "20260717-profitable-loop-episode-anatomy-v1/primary"
)
RUNNER_PATH = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work"
    / "run_profitable_loop_episode_anatomy_v1.py"
)


def _runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location("episode_anatomy_runner", RUNNER_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _calendar(*sessions: str, period: int = 2023) -> pd.DataFrame:
    return pd.DataFrame({"period": period, "score_session": list(sessions)})


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (2023, "2023-01-02", "a", "state_1", 10.0, True),
            (2023, "2023-01-02", "b", "state_1", 6.0, True),
            (2023, "2023-01-02", "c", "state_2", -2.0, False),
            (2023, "2023-01-03", "a", "state_1", 4.0, True),
            (2023, "2023-01-03", "b", "state_1", -4.0, False),
            (2023, "2023-01-03", "c", "state_2", 2.0, True),
        ],
        columns=[
            "period",
            "session",
            "loop",
            "regime",
            "robust_net_payoff_bps",
            "positive_pair_flag",
        ],
    ).assign(
        orientation=lambda frame: frame["regime"],
        pair=lambda frame: frame["loop"] + "|" + frame["regime"],
        eligible=True,
    )


def test_frozen_exploratory_census_reproduces_exactly() -> None:
    states = pd.read_parquet(FROZEN / "v2_hindsight_episode_states.parquet")
    pair_episodes = pd.read_parquet(FROZEN / "v2_hindsight_episode_diagnostics.parquet")
    feature_panel = pd.read_parquet(
        FROZEN / "v2_causal_feature_panel.parquet",
        columns=["period", "score_session"],
    )
    calendar = feature_panel.drop_duplicates().sort_values(["period", "score_session"])

    result = reproduce_exploratory_census(states, pair_episodes, calendar)

    assert result["positive_sessions"] == 322
    assert result["multi_pair_positive_sessions"] == 210
    assert result["multi_pair_positive_session_share"] == pytest.approx(0.6521739130434783)
    assert result["periods"] == {
        "2023": {
            "positive_sessions": 159,
            "multi_pair_positive_sessions": 93,
            "multi_pair_positive_session_share": pytest.approx(0.5849056603773585),
        },
        "2025": {
            "positive_sessions": 163,
            "multi_pair_positive_sessions": 117,
            "multi_pair_positive_session_share": pytest.approx(0.7177914110429447),
        },
    }
    assert result["same_regime_episodes"] == 107
    assert result["single_loop_same_regime_episodes"] == 65
    assert result["multi_loop_same_regime_episodes"] == 42
    assert result["multi_loop_same_regime_share_by_period"] == {
        "2023": pytest.approx(0.38095238095238093),
        "2025": pytest.approx(0.4090909090909091),
    }
    assert result["multi_loop_leader_share_available"] == 41
    assert result["multi_loop_leader_share_unavailable"] == 1
    assert result["multi_loop_leader_positive_payoff_share_median"] == pytest.approx(
        0.6498946257195766
    )
    assert result["multi_loop_majority_leader_episodes"] == 33
    assert result["multi_loop_over_80pct_leader_episodes"] == 11


def test_strict_positive_definition_does_not_use_raw_payoff() -> None:
    states = pd.DataFrame(
        {
            "period": [2023, 2023],
            "score_session": ["2023-01-02", "2023-01-03"],
            "loop_id": ["a", "a"],
            "orientation": ["state_1", "state_1"],
            "hindsight_payoff_state": ["positive", "decaying"],
            "robust_net_payoff_bps": [-999.0, 999.0],
        }
    )
    pair = pd.DataFrame(
        {
            "episode_id": ["p1"],
            "period": [2023],
            "loop_id": ["a"],
            "orientation": ["state_1"],
            "hindsight_estimated_onset": ["2023-01-02"],
            "hindsight_estimated_end": ["2023-01-03"],
            "total_episode_payoff_bps": [1.0],
        }
    )
    result = reproduce_exploratory_census(states, pair, _calendar("2023-01-02", "2023-01-03"))
    assert result["strict_positive_pair_rows"] == 1


def test_stock_collapse_prevents_repeated_fill_support() -> None:
    fills = pd.DataFrame(
        {
            "period": [2023, 2023, 2023],
            "session": ["2023-01-02"] * 3,
            "loop": ["a"] * 3,
            "orientation": ["state_1"] * 3,
            "stock": ["X", "X", "Y"],
            "gross_payoff_bps": [100.0, 0.0, 20.0],
            "cost_bps": [10.0, 10.0, 10.0],
            "net_payoff_bps": [90.0, -10.0, 10.0],
        }
    )
    result = collapse_stock_contributions(fills)
    row = result.iloc[0]
    assert row["independent_stock_count"] == 2
    assert row["raw_fill_count"] == 3
    assert row["occurrence_count"] == 2
    assert row["robust_net_payoff_bps"] == pytest.approx(25.0)


def test_stock_occurrence_identity_does_not_split_mixed_history_fills() -> None:
    runner = _runner()
    trades = pd.DataFrame(
        {
            "period": [2023, 2023],
            "model_name": ["no_payoff_state_filter"] * 2,
            "horizon": [24, 24],
            "status": ["filled", "filled"],
            "score_session": ["2023-01-02", "2023-01-02"],
            "loop_id": ["cycle_01", "cycle_01"],
            "orientation": ["state_1", "state_1"],
            "stock_id": ["X", "X"],
            "history_token": [1, 9],
            "gross_payoff_bps": [20.0, 40.0],
            "primary_net_payoff_bps": [10.0, 30.0],
            "fill_id": ["f1", "f2"],
            "month": ["2023-01", "2023-01"],
            "quarter": ["2023Q1", "2023Q1"],
            "volume_bucket": ["high", "high"],
            "state_change_phase": ["other", "other"],
        }
    )

    result = runner.stock_occurrence_ledger(trades)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["raw_fill_count"] == 2
    assert row["net_payoff_bps"] == pytest.approx(20.0)
    assert bool(row["history_mixed_within_occurrence"])
    assert row["regime_history_2"] == "unavailable"
    assert row["regime_history_3"] == "unavailable"


def test_stock_occurrence_preserves_unanimous_shorter_history() -> None:
    runner = _runner()
    trades = pd.DataFrame(
        {
            "period": [2023, 2023],
            "model_name": ["no_payoff_state_filter"] * 2,
            "horizon": [24, 24],
            "status": ["filled", "filled"],
            "score_session": ["2023-01-02", "2023-01-02"],
            "loop_id": ["cycle_01", "cycle_01"],
            "orientation": ["state_1", "state_1"],
            "stock_id": ["X", "X"],
            "history_token": [17, 89],
            "gross_payoff_bps": [20.0, 40.0],
            "primary_net_payoff_bps": [10.0, 30.0],
            "fill_id": ["f1", "f2"],
            "month": ["2023-01", "2023-01"],
            "quarter": ["2023Q1", "2023Q1"],
            "volume_bucket": ["high", "high"],
            "state_change_phase": ["other", "other"],
        }
    )

    row = runner.stock_occurrence_ledger(trades).iloc[0]

    assert bool(row["history_mixed_within_occurrence"])
    assert not bool(row["history_2_mixed_within_occurrence"])
    assert bool(row["history_3_mixed_within_occurrence"])
    assert row["regime_history_2"] == "state_2>state_1"
    assert row["regime_history_3"] == "unavailable"


def test_panel_history_requires_every_stock_context_to_agree_and_be_available() -> None:
    runner = _runner()
    panel = pd.DataFrame(
        {
            "period": [2025],
            "session": ["2025-05-06"],
            "loop": ["cycle_07"],
            "orientation": ["state_5"],
        }
    )
    occurrences = pd.DataFrame(
        {
            "period": [2025, 2025],
            "session": ["2025-05-06", "2025-05-06"],
            "loop": ["cycle_07", "cycle_07"],
            "regime": ["state_5", "state_5"],
            "stock": ["A", "B"],
            "previous_regime": ["state_4", "unavailable"],
            "regime_history_2": ["state_4>state_5", "unavailable"],
            "regime_history_3": ["unavailable", "unavailable"],
            "clock_phase": ["other", "other"],
            "history_mixed_within_occurrence": [False, False],
            "history_2_mixed_within_occurrence": [False, False],
            "history_3_mixed_within_occurrence": [False, False],
        }
    )

    row = runner.add_history_to_panel(panel, occurrences).iloc[0]

    assert bool(row["history_mixed"])
    assert not bool(row["history_available"])
    assert not bool(row["history_3_available"])
    assert row["regime_history_2"] == "unavailable"
    assert row["regime_history_3"] == "unavailable"


def test_frozen_occurrences_are_stock_capped_and_mixed_history_fails_closed() -> None:
    runner = _runner()
    result = runner.stock_occurrence_ledger(pd.read_parquet(FROZEN / "v2_trade_decisions.parquet"))

    assert len(result) == 9548
    assert result["raw_fill_count"].sum() == 9926
    assert not result.duplicated(["period", "session", "loop", "regime", "stock"]).any()
    mixed = result[result["history_mixed_within_occurrence"]]
    assert len(mixed) == 349
    assert mixed["regime_history_2"].ne("unavailable").sum() == 58
    assert (
        result.loc[result["history_2_mixed_within_occurrence"], "regime_history_2"]
        .eq("unavailable")
        .all()
    )
    assert mixed["regime_history_3"].eq("unavailable").all()


def test_sequence_session_block_bootstrap_is_deterministic_and_session_clustered() -> None:
    runner = _runner()
    left = pd.DataFrame(
        {
            "session": ["2023-01-02"] * 3 + ["2023-01-03"],
            "net_payoff_bps": [10.0, 10.0, 10.0, -2.0],
        }
    )
    right = pd.DataFrame(
        {
            "session": ["2023-01-02", "2023-01-03", "2023-01-03"],
            "net_payoff_bps": [0.0, 1.0, 1.0],
        }
    )

    first = runner.session_block_difference(left, right, seed=7, resamples=100, block_length=2)
    second = runner.session_block_difference(left, right, seed=7, resamples=100, block_length=2)

    assert first == second
    assert first["valid_resamples"] == 100
    assert 0 <= first["two_sided_p_value"] <= 1


def test_synchronized_panel_keeps_no_opportunity_missing() -> None:
    grid = pd.DataFrame(
        {
            "period": [2023, 2023],
            "score_session": ["2023-01-02"] * 2,
            "loop_id": ["a", "b"],
            "orientation": ["state_1", "state_1"],
            "horizon": [24, 24],
        }
    )
    payoff = pd.DataFrame(
        {
            "period": [2023],
            "score_session": ["2023-01-02"],
            "loop_id": ["a"],
            "orientation": ["state_1"],
            "horizon": [24],
            "robust_net_payoff_bps": [-1.0],
        }
    )
    result = build_synchronized_panel(grid, payoff, None)
    missing = result.loc[result["loop"].eq("b")].iloc[0]
    assert pd.isna(missing["robust_net_payoff_bps"])
    assert missing["eligible"]
    assert not missing["positive_pair_flag"]
    assert not bool(missing["positive_pair_available"])


def test_episode_levels_respect_adjacency_periods_and_missing_sessions() -> None:
    episodes = pd.DataFrame(
        {
            "episode_id": ["p1", "p2", "p3", "p4"],
            "period": [2023, 2023, 2023, 2025],
            "loop_id": ["a", "b", "c", "d"],
            "orientation": ["state_1", "state_1", "state_2", "state_1"],
            "hindsight_estimated_onset": [
                "2023-01-02",
                "2023-01-03",
                "2023-01-06",
                "2025-01-02",
            ],
            "hindsight_estimated_end": [
                "2023-01-02",
                "2023-01-03",
                "2023-01-06",
                "2025-01-02",
            ],
            "total_episode_payoff_bps": [1.0] * 4,
        }
    )
    calendar = pd.concat(
        [
            _calendar("2023-01-02", "2023-01-03", period=2023),
            _calendar("2025-01-02", period=2025),
        ],
        ignore_index=True,
    )
    pair, same, shared = build_episode_ledgers(episodes, calendar)
    assert len(pair) == 4
    assert len(same) == 3
    assert len(shared) == 3
    assert set(pair["episode_level"]) == {"pair"}
    assert set(same["episode_level"]) == {"same_regime"}
    assert set(shared["episode_level"]) == {"shared_market"}
    assert not same["source_pair_episode_ids"].str.contains("p3.*p1|p1.*p3").any()
    assert not shared["source_pair_episode_ids"].str.contains("p4.*p1|p1.*p4").any()


def test_panel_episode_memberships_are_exact_and_levels_remain_distinct() -> None:
    episodes = pd.DataFrame(
        {
            "episode_id": ["p1", "p2"],
            "period": [2023, 2023],
            "loop_id": ["a", "b"],
            "orientation": ["state_1", "state_1"],
            "hindsight_estimated_onset": ["2023-01-02", "2023-01-03"],
            "hindsight_estimated_end": ["2023-01-02", "2023-01-03"],
            "total_episode_payoff_bps": [1.0, 1.0],
        }
    )
    calendar = _calendar("2023-01-02", "2023-01-03", "2023-01-04")
    pair, same, shared = build_episode_ledgers(episodes, calendar)
    panel = pd.DataFrame(
        {
            "period": [2023] * 6,
            "session": ["2023-01-02", "2023-01-03", "2023-01-04"] * 2,
            "pair": ["a|state_1"] * 3 + ["b|state_1"] * 3,
            "regime": ["state_1"] * 6,
        }
    )

    result = attach_episode_membership(panel, pair, same, shared, calendar)

    a_rows = result[result["pair"].eq("a|state_1")].set_index("session")
    assert a_rows.loc["2023-01-02", "hindsight_pair_episode_id"] == "p1"
    assert pd.isna(a_rows.loc["2023-01-03", "hindsight_pair_episode_id"])
    assert a_rows.loc["2023-01-03", "same_regime_episode_id"] == "same_regime_0001"
    assert a_rows.loc["2023-01-03", "shared_session_episode_id"] == "shared_market_0001"
    assert pd.isna(a_rows.loc["2023-01-04", "same_regime_episode_id"])
    assert pd.isna(a_rows.loc["2023-01-04", "shared_session_episode_id"])


def test_component_identity_and_scope_and_no_fill_weighting() -> None:
    panel = _panel().assign(raw_fill_count=[1, 1000, 1, 1, 1, 1])
    result = decompose_payoff_components(panel)
    first = result[result["session"].eq("2023-01-02")]
    assert first["common_component"].nunique() == 1
    assert first["common_component"].iloc[0] == pytest.approx(6.0)
    state_1 = first[first["regime"].eq("state_1")]
    assert state_1["regime_component"].iloc[0] == pytest.approx(2.0)
    assert (
        result["common_component"] + result["regime_component"] + result["loop_excess_component"]
    ).equals(result["robust_net_payoff_bps"])


def test_component_excludes_ineligible_and_missing_rows() -> None:
    panel = _panel()
    panel.loc[0, "eligible"] = False
    panel.loc[1, "robust_net_payoff_bps"] = pd.NA
    result = decompose_payoff_components(panel)
    assert pd.isna(result.loc[0, "common_component"])
    assert pd.isna(result.loc[1, "common_component"])


@pytest.mark.parametrize(
    ("loop_count", "share", "expected"),
    [
        (1, 1.0, "SINGLE_LOOP_EPISODE"),
        (2, 0.81, "EXTREME_LOOP_DOMINANCE"),
        (2, 0.51, "MAJORITY_LOOP_DOMINANCE"),
        (2, 0.50, "DIFFUSE_MULTI_LOOP"),
    ],
)
def test_episode_anatomy_categories_are_deterministic(
    loop_count: int, share: float, expected: str
) -> None:
    assert classify_episode(loop_count, share) == expected


def test_leader_efficiency_and_ties_and_share_sums() -> None:
    rows = pd.DataFrame(
        {
            "episode_id": ["e"] * 4,
            "session": ["1", "1", "2", "2"],
            "loop": ["a", "b", "a", "b"],
            "positive_payoff": [5.0, 3.0, 5.0, 7.0],
            "occurrence_count": [1, 2, 1, 2],
        }
    )
    result = early_leader_checkpoints(rows)
    final = result.attrs["episode_summary"].iloc[0]
    assert final["leader_tie"]
    assert final["leader_loops"] == "a|b"
    assert final["positive_payoff_share_sum"] == pytest.approx(1.0)
    assert final["occurrence_share_sum"] == pytest.approx(1.0)
    zero = rows.assign(occurrence_count=0)
    zero_result = early_leader_checkpoints(zero)
    assert pd.isna(zero_result.attrs["episode_summary"].iloc[0]["leader_efficiency"])


def test_early_checkpoints_use_prefix_and_remaining_payoff_only() -> None:
    rows = pd.DataFrame(
        {
            "episode_id": ["early"] * 6 + ["late"] * 6,
            "session": ["1", "1", "2", "2", "3", "3"] * 2,
            "loop": ["a", "b"] * 6,
            "positive_payoff": [5, 1, 5, 1, 5, 1, 1, 5, 1, 5, 20, 1],
            "occurrence_count": [1] * 12,
        }
    )
    result = early_leader_checkpoints(rows)
    first = result[result["checkpoint"].eq("first_session")].set_index("episode_id")
    assert first.loc["early", "top_one_match"]
    assert not first.loc["late", "top_one_match"]
    assert first.loc["early", "payoff_remaining"] == pytest.approx(12.0)
    assert first.loc["late", "payoff_remaining"] == pytest.approx(27.0)


def test_frozen_final_ranking_does_not_change_provisional_prefix_ranking() -> None:
    rows = pd.DataFrame(
        {
            "episode_id": ["e"] * 4,
            "session": ["1", "1", "2", "2"],
            "loop": ["a", "b", "a", "b"],
            "positive_payoff": [10.0, 1.0, 0.0, 20.0],
            "occurrence_count": [1, 1, 1, 1],
            "final_positive_payoff": [10.0, 30.0, 10.0, 30.0],
        }
    )

    first = early_leader_checkpoints(rows).query("checkpoint == 'first_session'").iloc[0]

    assert first["provisional_leaders"] == "a"
    assert not first["top_one_match"]
    assert first["fraction_final_payoff_remaining"] == pytest.approx(20.0 / 31.0)


def test_no_positive_prefix_has_no_occurrence_share_or_efficiency() -> None:
    rows = pd.DataFrame(
        {
            "episode_id": ["e", "e"],
            "session": ["1", "2"],
            "loop": ["a", "a"],
            "positive_payoff": [0.0, 5.0],
            "occurrence_count": [1, 1],
        }
    )

    first = early_leader_checkpoints(rows).query("checkpoint == 'first_session'").iloc[0]

    assert first["provisional_leaders"] == ""
    assert pd.isna(first["provisional_leader_occurrence_share"])
    assert pd.isna(first["provisional_leader_efficiency"])


def test_frequency_and_payoff_efficiency_driven_leaders_are_separated() -> None:
    frequency = pd.DataFrame(
        {
            "episode_id": ["frequency"] * 2,
            "session": ["1", "1"],
            "loop": ["a", "b"],
            "positive_payoff": [8.0, 2.0],
            "occurrence_count": [8, 2],
        }
    )
    efficiency = frequency.assign(episode_id="efficiency", occurrence_count=[2, 8])
    frequency_summary = early_leader_checkpoints(frequency).attrs["episode_summary"].iloc[0]
    efficiency_summary = early_leader_checkpoints(efficiency).attrs["episode_summary"].iloc[0]
    assert frequency_summary["leader_efficiency"] == pytest.approx(1.0)
    assert efficiency_summary["leader_efficiency"] == pytest.approx(4.0)


def test_high_occurrence_low_efficiency_loop_does_not_become_leader() -> None:
    rows = pd.DataFrame(
        {
            "episode_id": ["e"] * 2,
            "session": ["1", "1"],
            "loop": ["common_weak", "rare_strong"],
            "positive_payoff": [1.0, 9.0],
            "occurrence_count": [9, 1],
        }
    )
    summary = early_leader_checkpoints(rows).attrs["episode_summary"].iloc[0]
    assert summary["leader_loops"] == "rare_strong"
    assert summary["leader_efficiency"] == pytest.approx(9.0)


def test_history_decode_uses_completed_prior_states() -> None:
    token = ((2 * 9 + 3) * 8) + 4
    decoded = decode_history_token(token)
    assert decoded == {
        "current_regime": "state_4",
        "previous_regime": "state_3",
        "regime_history_2": "state_3>state_4",
        "regime_history_3": "state_2>state_3>state_4",
    }
    assert "regime_history_4" not in decoded


def test_four_way_counterfactual_groups_are_exclusive_and_hold_identity() -> None:
    rows = pd.DataFrame(
        {
            "loop": ["target", "target", "other", "other"],
            "regime": ["state_4"] * 4,
            "regime_history_3": ["x", "y", "x", "y"],
            "robust_net_payoff_bps": [4.0, 1.0, 2.0, -1.0],
            "session": ["1", "2", "3", "4"],
            "stock": ["A", "B", "C", "D"],
        }
    )
    result = four_way_counterfactual(
        rows,
        target_loop="target",
        current_regime="state_4",
        target_sequence="x",
        sequence_column="regime_history_3",
    )
    assert set(result["counterfactual_group"]) == {"1", "2", "3", "4"}
    assert result["rows"].sum() == 4
    assert (result["regime"] == "state_4").all()


def test_nulls_preserve_eligibility_counts_period_boundaries_and_positive_counts() -> None:
    rows = pd.DataFrame(
        {
            "period": [2023] * 6 + [2025] * 6,
            "session": ["1", "2", "3"] * 4,
            "pair": ["a"] * 3 + ["b"] * 3 + ["a"] * 3 + ["b"] * 3,
            "eligible": [True, True, False, True, True, True] * 2,
            "positive_pair_flag": [True, False, False, False, True, False] * 2,
        }
    )
    poisson = poisson_binomial_null(rows)
    assert set(poisson["period"]) == {"2023", "2025"}
    shifted, summary = block_circular_pair_shift(rows, resamples=25, seed=7, block_length=2)
    assert summary["eligibility_mask_preserved"]
    assert summary["pair_positive_counts_preserved"]
    assert summary["period_boundaries_preserved"]
    expected_rows = rows.assign(period=rows["period"].astype(str))
    expected = expected_rows.groupby(["period", "pair"])["positive_pair_flag"].sum().sort_index()
    for _, replicate in shifted.groupby("replicate"):
        actual = replicate.groupby(["period", "pair"])["positive_pair_flag"].sum().sort_index()
        pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_null_distinguishes_no_excess_and_strong_coactivation() -> None:
    independent = pd.DataFrame(
        {
            "period": [2023] * 8,
            "session": ["1", "2", "3", "4"] * 2,
            "pair": ["a"] * 4 + ["b"] * 4,
            "eligible": True,
            "positive_pair_flag": [True, False, True, False, False, True, False, True],
        }
    )
    shared = independent.copy()
    shared["positive_pair_flag"] = [True, False, True, False] * 2
    _, null_independent = block_circular_pair_shift(
        independent, resamples=200, seed=11, block_length=1
    )
    _, null_shared = block_circular_pair_shift(shared, resamples=200, seed=11, block_length=1)
    assert null_shared["observed_minus_null_share"] > null_independent["observed_minus_null_share"]


def test_causal_indicator_guard_rejects_outcomes_future_and_late_timestamps() -> None:
    with pytest.raises(ValueError, match="forbidden causal indicators"):
        validate_causal_indicators(["mfe_bps", "mae_bps", "hindsight_episode_id"])
    with pytest.raises(ValueError, match="future"):
        validate_causal_indicators(["future_regime"])
    rows = pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(["2023-01-02 10:00Z"]),
            "feature_availability_timestamp": pd.to_datetime(["2023-01-02 10:01Z"]),
        }
    )
    with pytest.raises(ValueError, match="after decision"):
        validate_causal_indicators(["market_return"], rows)


def test_stock_concentration_and_removals_recalculate() -> None:
    rows = pd.DataFrame(
        {
            "episode_id": ["e"] * 4,
            "stock": ["A", "B", "C", "D"],
            "positive_payoff": [70.0, 10.0, 10.0, 10.0],
            "sector": [pd.NA] * 4,
        }
    )
    result = concentration_attribution(rows)
    row = result.iloc[0]
    assert row["top_one_share"] == pytest.approx(0.7)
    assert row["after_remove_best_stock"] == pytest.approx(30.0)
    assert row["sector_status"] == "unavailable"


def test_stock_removal_recomputes_common_regime_and_loop_components() -> None:
    occurrences = pd.DataFrame(
        {
            "period": [2023] * 4,
            "session": ["2023-01-02"] * 4,
            "pair": ["a|state_1", "a|state_1", "b|state_1", "b|state_1"],
            "loop": ["a", "a", "b", "b"],
            "regime": ["state_1"] * 4,
            "stock": ["A", "B", "A", "B"],
            "net_payoff_bps": [10.0, 0.0, 2.0, 2.0],
        }
    )

    result = recompute_component_summary_after_stock_removal(
        occurrences,
        member_pairs={"a|state_1", "b|state_1"},
        removed_stocks={"A"},
    )

    assert result["supported_pair_cells"] == 2
    assert result["robust_net_payoff_sum"] == pytest.approx(2.0)
    assert result["common_component_mean"] == pytest.approx(1.0)
    assert result["regime_component_mean"] == pytest.approx(0.0)
    assert result["loop_excess_component_mean"] == pytest.approx(0.0)
    assert result["component_identity_max_absolute_error"] == pytest.approx(0.0)


def test_recorded_occurrence_payoff_identity_and_missingness_are_fail_closed() -> None:
    occurrence = pd.read_parquet(ARTIFACTS / "occurrence_share_table.parquet")
    available = occurrence[occurrence["occurrence_population_complete"]]
    unavailable = occurrence[~occurrence["occurrence_population_complete"]]

    assert not available.empty
    assert available["payoff_occurrence_identity_error"].abs().max() <= 1e-10
    assert (
        available["total_loop_payoff"]
        - available["occurrence_count"] * available["mean_payoff_per_occurrence"]
    ).abs().max() <= 1e-10
    assert unavailable["mean_payoff_per_occurrence"].isna().all()
    assert unavailable["leader_efficiency"].isna().all()


def test_recorded_sequence_inference_is_session_blocked_and_clock_aware() -> None:
    census = pd.read_parquet(ARTIFACTS / "regime_sequence_census.parquet")
    four_way = pd.read_parquet(ARTIFACTS / "four_way_counterfactual_tables.parquet")
    increments = pd.read_parquet(ARTIFACTS / "sequence_increment_table.parquet")

    for frame in [census, four_way]:
        assert {
            "clock_phase_available_rows",
            "clock_phase_missing_rows",
            "clock_phase_availability_rate",
            "dominant_clock_phase",
            "clock_phase_counts_json",
        }.issubset(frame.columns)
        assert (
            frame["clock_phase_available_rows"] + frame["clock_phase_missing_rows"] == frame["rows"]
        ).all()
    for prefix in ["sequence_increment", "loop_increment"]:
        valid = increments[f"{prefix}_bootstrap_valid_resamples"].gt(0)
        assert increments.loc[valid, f"{prefix}_bootstrap_lower_95"].notna().all()
        assert increments.loc[valid, f"{prefix}_bootstrap_upper_95"].notna().all()
        assert increments.loc[~valid, f"{prefix}_bootstrap_lower_95"].isna().all()
        assert increments[f"{prefix}_bootstrap_valid_resamples"].between(0, 1000).all()
    assert increments["target_clock_phase_available_rows"].gt(0).all()
    supported = increments["increment_comparison_supported"]
    assert supported.sum() == 4
    for column in [
        "sequence_increment_bootstrap_lower_95",
        "sequence_increment_bootstrap_upper_95",
        "sequence_increment_p_value",
        "sequence_increment_fdr_q_value",
        "loop_increment_bootstrap_lower_95",
        "loop_increment_bootstrap_upper_95",
        "loop_increment_p_value",
        "loop_increment_fdr_q_value",
    ]:
        assert increments.loc[supported, column].notna().all()
        assert increments.loc[~supported, column].isna().all()
    assert not increments.loc[~supported, "interaction_fdr_pass"].any()


def test_recorded_panel_history_availability_fails_closed() -> None:
    panel = pd.read_parquet(ARTIFACTS / "session_regime_loop_orientation_panel.parquet")
    expected = panel["regime_history_2"].notna() & ~panel["regime_history_2"].astype(str).eq(
        "unavailable"
    )
    expected_3 = panel["regime_history_3"].notna() & ~panel["regime_history_3"].astype(str).eq(
        "unavailable"
    )

    assert panel["history_available"].astype(bool).equals(expected)
    assert panel["history_3_available"].astype(bool).equals(expected_3)
    assert panel.loc[~panel["history_available"], "regime_history_2"].eq("unavailable").all()
    assert panel.loc[~panel["history_3_available"], "regime_history_3"].eq("unavailable").all()


def test_recorded_route_path_persistence_and_network_contracts() -> None:
    route = pd.read_parquet(ARTIFACTS / "route_topology_anatomy.parquet")
    path = pd.read_parquet(ARTIFACTS / "sequential_path_deterioration.parquet")
    persistence = pd.read_parquet(ARTIFACTS / "leader_persistence_table.parquet")
    network = pd.read_csv(ARTIFACTS / "coactivation_network_edge_table.csv")

    assert route["outcome_only"].all()
    assert route["episode_level"].eq("pair").all()
    assert route["episode_id"].ne("all_episodes").all()
    assert {"onset", "early", "middle", "late", "decay"}.issubset(set(path["episode_phase"]))
    available_path = path[path["deterioration_before_decay_available"]]
    assert (
        available_path["deterioration_rises_before_leader_decay"]
        == (available_path["negative_tail_late_rate"] > available_path["negative_tail_middle_rate"])
    ).all()
    no_positive = persistence[
        persistence["status"].isin(
            ["no_positive_pair_current_session", "no_positive_pair_next_session"]
        )
    ]
    assert not no_positive.empty
    assert no_positive["top_one_persistence"].isna().all()
    for side in ["left", "right"]:
        assert network[f"node_total_positive_payoff_{side}"].notna().all()
        assert network[f"node_regime_{side}"].notna().all()
        assert network[f"node_leader_frequency_{side}"].dropna().between(0, 1).all()


def test_recorded_within_pair_indicators_and_stock_removal_components() -> None:
    manifestations = pd.read_parquet(ARTIFACTS / "indicator_manifestation_tables.parquet")
    within_pair = manifestations[
        manifestations["comparison"].eq("profitable_vs_unprofitable_same_pair")
    ]
    removal = pd.read_parquet(ARTIFACTS / "leave_one_stock_out_attribution.parquet")

    assert not within_pair.empty
    assert within_pair["pair"].str.contains(r"\|", regex=True).all()
    assert (within_pair["pair"].str.split("|").str[0] == within_pair["loop"]).all()
    assert (within_pair["pair"].str.split("|").str[1] == within_pair["regime"]).all()
    complete = removal[removal["component_summary_recalculated"]]
    assert not complete.empty
    assert complete["component_identity_max_absolute_error"].max() <= 1e-10
    assert {"leave_one_stock_out", "remove_best_stock", "remove_top_five_stocks"} == set(
        removal["removal_scope"]
    )
    assert not removal["model_retrained"].any()

    for name in [
        "same_regime_episode_ledger.parquet",
        "episode_session_timeline.parquet",
        "early_leader_table.parquet",
        "component_episode_attribution.parquet",
        "stock_and_cohort_concentration.parquet",
    ]:
        traced = pd.read_parquet(ARTIFACTS / name)
        for column in ["period", "session", "episode_id", "pair", "loop", "orientation", "regime"]:
            assert traced[column].notna().all(), (name, column)
            assert not traced[column].astype(str).str.startswith("all_").any(), (name, column)


def test_factor_diagnostic_is_secondary_period_specific_and_fixed() -> None:
    matrix = pd.DataFrame(
        {
            "period": [2023] * 6 + [2025] * 6,
            "session": ["1", "1", "2", "2", "3", "3"] * 2,
            "pair": ["a", "b"] * 6,
            "robust_net_payoff_bps": [1, 2, 2, 4, 3, 6, 2, 1, 4, 2, 6, 3],
        }
    )
    result = common_factor_diagnostic(matrix, factor_counts=(1, 2, 3))
    assert set(result["period"]) == {"2023", "2025"}
    assert set(result["factor_count"]) == {1, 2}
    assert (result["changes_primary_decomposition"] == False).all()  # noqa: E712


def test_exact_rerun_identity_detects_byte_drift(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    rerun = tmp_path / "rerun"
    primary.mkdir()
    rerun.mkdir()
    (primary / "a.json").write_bytes(b"{}\n")
    (rerun / "a.json").write_bytes(b"{}\n")
    assert exact_rerun_identity(primary, rerun)["byte_identical"]
    (rerun / "a.json").write_bytes(b"{ }\n")
    assert not exact_rerun_identity(primary, rerun)["byte_identical"]


def test_run_git_identity_remains_frozen_from_descendant_checkout() -> None:
    frozen = "d199bed1e1d66199ba63b3f5e12df03768728484"
    contract = {"lineage": {"starting_commit": frozen}}

    assert (
        frozen_run_git_head(
            contract,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            frozen_is_ancestor=True,
        )
        == frozen
    )
    with pytest.raises(ValueError, match="does not descend"):
        frozen_run_git_head(
            contract,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            frozen_is_ancestor=False,
        )


def test_research_module_has_no_execution_runtime_imports() -> None:
    package = (
        ROOT / "packages/stocker_research/src/stocker_research/profitable_loop_episode_anatomy"
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    forbidden = [
        "stocker_execution",
        "ig_integration",
        "place_order",
        "position_management",
        "paper_trading",
    ]
    assert not [token for token in forbidden if token in source]
