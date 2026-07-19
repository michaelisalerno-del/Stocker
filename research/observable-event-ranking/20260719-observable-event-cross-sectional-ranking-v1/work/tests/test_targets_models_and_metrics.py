from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.observable_event_ranking_v1.baselines import (
    TrainingOnlyStockClockPrior,
    deterministic_baseline_scores,
)
from stocker_research.observable_event_ranking_v1.contract import PRIMARY_FEATURES
from stocker_research.observable_event_ranking_v1.features import (
    add_causal_price_activity_features,
    build_feature_ledger,
    feature_manifest,
    primary_matrix,
)
from stocker_research.observable_event_ranking_v1.folds import build_expanding_folds
from stocker_research.observable_event_ranking_v1.linear_ranker import (
    equal_slate_sample_weights,
    fit_linear_ranker,
)
from stocker_research.observable_event_ranking_v1.metrics import (
    leave_one_stock_out_analysis,
    ndcg_score,
    pairwise_ranking_accuracy,
    per_slate_metrics,
    spearman_ic,
    top_decile_hit,
    top_two_minus_median,
)
from stocker_research.observable_event_ranking_v1.targets import (
    build_target_ledger,
    percentile_rank,
)
from stocker_research.observable_event_ranking_v1.uncertainty import (
    paired_session_block_bootstrap,
    session_block_sample_indices,
)


def _events(count: int = 10) -> pd.DataFrame:
    decision = pd.Timestamp("2025-07-07T14:00:00Z")
    rows: list[dict[str, object]] = []
    for number in range(count):
        row: dict[str, object] = {
            "event_id": f"event-{number}",
            "slate_id": "slate-1",
            "symbol": f"S{number:02d}",
            "sector": "Technology" if number < 5 else "Industrials",
            "assigned_decision_time": decision,
            "session": pd.Timestamp("2025-07-07", tz="UTC"),
            "session_close": pd.Timestamp("2025-07-07T20:00:00Z"),
        }
        for feature_number, feature in enumerate(PRIMARY_FEATURES):
            row[feature] = float(number + feature_number / 100)
        rows.append(row)
    return pd.DataFrame(rows)


def _settlement_bars(count: int = 10, *, missing_symbols: set[str] | None = None) -> pd.DataFrame:
    missing_symbols = missing_symbols or set()
    rows: list[dict[str, object]] = []
    for number in range(count):
        symbol = f"S{number:02d}"
        entry = 100.0
        times = [
            ("2025-07-07T14:00:00Z", "2025-07-07T14:05:00Z", 99.0, 99.5),
            ("2025-07-07T14:05:00Z", "2025-07-07T14:10:00Z", entry, 100.0),
            ("2025-07-07T14:15:00Z", "2025-07-07T14:20:00Z", 100.0, 100.5 + number),
            ("2025-07-07T14:30:00Z", "2025-07-07T14:35:00Z", 100.0, 101.0 + number),
            ("2025-07-07T15:00:00Z", "2025-07-07T15:05:00Z", 100.0, 102.0 + number),
            ("2025-07-07T19:55:00Z", "2025-07-07T20:00:00Z", 100.0, 103.0 + number),
        ]
        for bar_start, bar_end, open_price, close_price in times:
            if symbol in missing_symbols and bar_end == "2025-07-07T15:05:00Z":
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "bar_start": pd.Timestamp(bar_start),
                    "bar_end": pd.Timestamp(bar_end),
                    "open": open_price,
                    "close": close_price,
                    "fully_completed": True,
                    "gap_status": "complete",
                }
            )
    return pd.DataFrame(rows)


def test_feature_manifest_is_exact_and_primary_matrix_excludes_identifiers() -> None:
    events = _events()

    ledger = build_feature_ledger(events)
    matrix, columns = primary_matrix(ledger)
    manifest = feature_manifest()

    assert columns == PRIMARY_FEATURES
    assert tuple(item["name"] for item in manifest["features"]) == PRIMARY_FEATURES
    assert matrix.shape == (10, 12)
    assert "symbol" not in columns
    assert "sector" not in columns


def test_activity_shock_uses_prior_stock_clock_history_and_ignores_later_sessions() -> None:
    rows: list[dict[str, object]] = []
    for day in range(25):
        rows.append(
            {
                "symbol": "AAA",
                "session": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=day),
                "session_open": pd.Timestamp("2025-01-01T15:00:00Z") + pd.Timedelta(days=day),
                "bar_start": pd.Timestamp("2025-01-01T15:00:00Z") + pd.Timedelta(days=day),
                "bar_end": pd.Timestamp("2025-01-01T15:05:00Z") + pd.Timedelta(days=day),
                "feature_availability_time": pd.Timestamp("2025-01-01T15:05:00Z")
                + pd.Timedelta(days=day),
                "session_close": pd.Timestamp("2025-01-01T21:00:00Z") + pd.Timedelta(days=day),
                "open": 100.0 + day,
                "high": 101.0 + day,
                "low": 99.0 + day,
                "close": 100.0 + day,
                "volume": 1_000.0 + day,
                "fully_completed": True,
                "gap_status": "complete",
            }
        )
    frame = pd.DataFrame(rows)
    original = add_causal_price_activity_features(frame, min_activity_observations=20)
    target = original.index[20]
    mutated = frame.copy()
    mutated.loc[mutated.index > target, ["close", "volume"]] = 1_000_000.0
    rerun = add_causal_price_activity_features(mutated, min_activity_observations=20)

    assert rerun.loc[target, "activity_shock_z"] == pytest.approx(
        original.loc[target, "activity_shock_z"]
    )


def test_causal_features_fail_closed_for_missing_open_gap_and_incomplete_bar() -> None:
    session_open = pd.Timestamp("2025-07-07T13:30:00Z")
    rows: list[dict[str, object]] = []
    for number in range(8):
        bar_start = session_open + pd.Timedelta(minutes=5 * number)
        rows.append(
            {
                "symbol": "AAA",
                "session": pd.Timestamp("2025-07-07", tz="UTC"),
                "session_open": session_open,
                "bar_start": bar_start,
                "bar_end": bar_start + pd.Timedelta(minutes=5),
                "feature_availability_time": bar_start + pd.Timedelta(minutes=5),
                "session_close": pd.Timestamp("2025-07-07T20:00:00Z"),
                "open": 100.0 + number,
                "high": 101.0 + number,
                "low": 99.0 + number,
                "close": 100.5 + number,
                "volume": 1_000.0,
                "fully_completed": True,
                "gap_status": "complete",
            }
        )
    valid = add_causal_price_activity_features(pd.DataFrame(rows))
    assert bool(valid.iloc[-1]["feature_context_valid"])
    assert pd.notna(valid.iloc[-1]["realized_volatility_30m"])

    missing_open = add_causal_price_activity_features(pd.DataFrame(rows[1:]))
    assert not missing_open["feature_context_valid"].any()
    assert missing_open["distance_from_session_high"].isna().all()

    with_gap_rows = [dict(row) for row in rows]
    with_gap_rows[4]["gap_status"] = "source_gap"
    with_gap = add_causal_price_activity_features(pd.DataFrame(with_gap_rows))
    assert not bool(with_gap.iloc[4]["feature_context_valid"])
    assert not bool(with_gap.iloc[-1]["feature_context_valid"])
    assert pd.isna(with_gap.iloc[-1]["realized_volatility_30m"])

    incomplete_rows = [dict(row) for row in rows]
    incomplete_rows[-1]["fully_completed"] = False
    incomplete = add_causal_price_activity_features(pd.DataFrame(incomplete_rows))
    assert not bool(incomplete.iloc[-1]["feature_context_valid"])
    assert pd.isna(incomplete.iloc[-1]["session_fraction"])


def test_percentile_rank_uses_deterministic_average_ties_on_zero_to_one_range() -> None:
    ranked = percentile_rank(pd.Series([1.0, 2.0, 2.0, 4.0]))

    assert ranked.to_numpy() == pytest.approx([0.0, 0.5, 0.5, 1.0])


def test_target_ledger_uses_delayed_entry_and_exact_sixty_minute_close() -> None:
    target = build_target_ledger(_events(), _settlement_bars())
    first = target.iloc[0]

    assert first["delayed_entry_reference_time"] == pd.Timestamp("2025-07-07T14:05:00Z")
    assert first["entry_reference_open"] == 100.0
    assert first["exit_reference_15m_time"] == pd.Timestamp("2025-07-07T14:20:00Z")
    assert first["exit_reference_15m_close"] == 100.5
    assert first["exit_reference_30m_time"] == pd.Timestamp("2025-07-07T14:35:00Z")
    assert first["exit_reference_30m_close"] == 101.0
    assert first["exit_reference_60m_close"] == 102.0
    assert first["session_close_reference_time"] == pd.Timestamp("2025-07-07T20:00:00Z")
    assert first["exit_reference_session_close_close"] == 103.0
    assert first["future_return_60m"] == pytest.approx(0.02)
    assert first["target_rank_60m"] == pytest.approx(0.0)
    assert bool(first["slate_evaluable"])


def test_missing_target_preserves_membership_and_whole_slate_rule() -> None:
    one_missing = build_target_ledger(_events(), _settlement_bars(missing_symbols={"S09"}))
    two_missing = build_target_ledger(_events(), _settlement_bars(missing_symbols={"S08", "S09"}))

    assert len(one_missing) == 10
    assert one_missing["slate_original_size"].eq(10).all()
    assert one_missing["slate_evaluable"].all()
    assert one_missing["target_rank_60m"].notna().sum() == 9
    assert len(two_missing) == 10
    assert not two_missing["slate_evaluable"].any()
    assert two_missing["target_rank_60m"].isna().all()


def test_expanding_folds_are_non_overlapping_and_strictly_chronological() -> None:
    rows: list[dict[str, object]] = []
    for month in pd.date_range("2024-01-01", periods=12, freq="MS", tz="UTC"):
        rows.append(
            {
                "event_id": f"event-{month:%Y%m}",
                "slate_id": f"slate-{month:%Y%m}",
                "session": month,
                "target_rank_60m": 0.5,
                "slate_evaluable": True,
            }
        )
    frame = pd.DataFrame(rows)

    folds = build_expanding_folds(frame)

    assert len(folds) == 2
    assert folds[0].evaluation_months == ("2024-07", "2024-08", "2024-09")
    assert folds[1].evaluation_months == ("2024-10", "2024-11", "2024-12")
    assert max(frame.loc[list(folds[0].train_indices), "session"]) < min(
        frame.loc[list(folds[0].evaluation_indices), "session"]
    )
    assert set(folds[0].evaluation_indices).isdisjoint(folds[1].evaluation_indices)


def test_fold_calendar_blocks_retain_empty_months() -> None:
    sessions = [
        month
        for month in pd.date_range("2024-01-01", periods=12, freq="MS", tz="UTC")
        if month.month != 8
    ]
    ledger = pd.DataFrame(
        {
            "event_id": [f"event-{session:%Y%m}" for session in sessions],
            "slate_id": [f"slate-{session:%Y%m}" for session in sessions],
            "session": sessions,
            "target_rank_60m": 0.5,
            "slate_evaluable": True,
        }
    )

    folds = build_expanding_folds(ledger)

    assert folds[0].evaluation_months == ("2024-07", "2024-08", "2024-09")


def test_linear_ranker_uses_equal_total_slate_weights_and_training_only_preprocessing() -> None:
    features = pd.DataFrame({feature: [0.0, 1.0, 2.0, 3.0, 100.0] for feature in PRIMARY_FEATURES})
    targets = np.asarray([0.0, 0.5, 1.0, 0.25, 0.75])
    slates = pd.Series(["a", "a", "a", "a", "b"])

    weights = equal_slate_sample_weights(slates)
    model = fit_linear_ranker(features, targets, slates)
    changed_evaluation = features.copy()
    changed_evaluation.loc[4, :] = 1_000_000.0
    training_only_model = fit_linear_ranker(
        changed_evaluation.iloc[:4], targets[:4], slates.iloc[:4]
    )
    original_training_model = fit_linear_ranker(features.iloc[:4], targets[:4], slates.iloc[:4])

    assert weights[:4].sum() == pytest.approx(1.0)
    assert weights[4:].sum() == pytest.approx(1.0)
    assert model.alpha == 1.0
    assert training_only_model.preprocessor == original_training_model.preprocessor


def test_frozen_simple_baselines_and_training_only_prior_have_unseen_fallback() -> None:
    frame = _events()
    scores = deterministic_baseline_scores(frame, "B1_EVENT_STRENGTH")
    training = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "decision_clock": ["10:00", "10:00", "10:30"],
            "session": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"], utc=True),
            "target_rank_60m": [1.0, 0.5, 0.0],
        }
    )
    prior = TrainingOnlyStockClockPrior.fit(training, kind="mean_target")

    assert scores.to_numpy() == pytest.approx(frame["event_strength"].to_numpy())
    assert prior.score("NEVER_SEEN", "14:30") == pytest.approx(prior.global_prior)
    assert prior.score("AAA", "10:00") > prior.global_prior


def test_ranking_metrics_match_known_fixtures_and_undersized_top_decile_is_unavailable() -> None:
    outcomes = np.asarray([0.01, 0.02, 0.03, 0.04])
    perfect = np.asarray([1.0, 2.0, 3.0, 4.0])
    reverse = perfect[::-1]

    assert spearman_ic(outcomes, perfect) == pytest.approx(1.0)
    assert spearman_ic(outcomes, reverse) == pytest.approx(-1.0)
    assert ndcg_score(outcomes, perfect) == pytest.approx(1.0)
    assert pairwise_ranking_accuracy(outcomes, perfect) == pytest.approx(1.0)
    assert top_two_minus_median(outcomes, perfect) == pytest.approx(0.01)
    assert top_decile_hit(outcomes, perfect) is None


def test_per_slate_metrics_weight_slates_equally_and_report_candidate_baseline_pair() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["a"] * 4 + ["b"] * 4,
            "session": pd.to_datetime(["2025-01-01"] * 4 + ["2025-01-02"] * 4, utc=True),
            "symbol": [f"S{x}" for x in range(8)],
            "sector": ["Technology"] * 8,
            "future_return_60m": [0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01],
            "target_rank_60m": [0.0, 1 / 3, 2 / 3, 1.0, 1.0, 2 / 3, 1 / 3, 0.0],
            "candidate_score": [1.0, 2.0, 3.0, 4.0] * 2,
            "baseline_score": [4.0, 3.0, 2.0, 1.0] * 2,
            "slate_evaluable": True,
        }
    )

    metrics = per_slate_metrics(frame)

    assert len(metrics) == 2
    assert set(metrics["candidate_ic"]) == {1.0, -1.0}
    assert metrics["candidate_minus_baseline_ic"].mean() == pytest.approx(0.0)


def test_session_block_bootstrap_preserves_whole_slates_and_pairing() -> None:
    slates = pd.DataFrame(
        {
            "session": ["s1", "s1", "s2"],
            "slate_id": ["a", "b", "c"],
            "candidate_ic": [0.2, 0.4, 0.1],
            "baseline_ic": [0.1, 0.3, 0.0],
        }
    )

    indices = session_block_sample_indices(slates, draws=20, seed=7)
    result = paired_session_block_bootstrap(
        slates,
        candidate_column="candidate_ic",
        baseline_column="baseline_ic",
        draws=200,
        seed=7,
    )

    for draw in indices:
        assert (0 in draw) == (1 in draw)
    assert result.estimate == pytest.approx(0.1)
    assert result.lower == pytest.approx(0.1)
    assert result.upper == pytest.approx(0.1)


def test_leave_one_stock_out_removes_each_stock_from_every_slate() -> None:
    rows: list[dict[str, object]] = []
    for slate_number in range(2):
        for stock_number in range(9):
            rows.append(
                {
                    "slate_id": f"slate-{slate_number}",
                    "session": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=slate_number),
                    "symbol": f"S{stock_number}",
                    "sector": "Technology",
                    "future_return_60m": stock_number / 100,
                    "target_rank_60m": stock_number / 8,
                    "candidate_score": float(stock_number),
                    "baseline_score": float(8 - stock_number),
                    "slate_evaluable": True,
                }
            )
    analysis = leave_one_stock_out_analysis(pd.DataFrame(rows))

    assert set(analysis["removed_symbol"]) == {f"S{number}" for number in range(9)}
    assert analysis["remaining_rows"].eq(16).all()
