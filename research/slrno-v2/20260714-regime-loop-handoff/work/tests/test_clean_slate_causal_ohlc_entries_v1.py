from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER = WORKSPACE / "work/run_clean_slate_causal_ohlc_entries_v1.py"
ARTIFACT_ROOT = Path(
    "/private/tmp/stocker_clean_slate_causal_ohlc_entries_v1_20260712"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "clean_slate_causal_ohlc_entries_v1", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_tape(
    timestamps: list[pd.Timestamp],
    session_dates: list[str],
    segment_indices: list[int],
    opens: list[float],
    closes: list[float],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps),
            "open": np.asarray(opens, float),
            "close": np.asarray(closes, float),
            "symbol_norm": "X",
            "session_date": session_dates,
            "segment_index": np.asarray(segment_indices, np.int16),
        }
    )
    frame["high"] = np.maximum(frame["open"], frame["close"]) + 1.0
    frame["low"] = np.minimum(frame["open"], frame["close"]) - 1.0
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame["month_key"] = frame["session_date"].str[:7]
    frame["bar_ordinal"] = ((minute - 570) // 5).astype(np.int16)
    keys = ["symbol_norm", "session_date", "segment_index"]
    frame["segment_position"] = frame.groupby(keys, sort=False).cumcount().astype(
        np.int16
    )
    frame["segment_size"] = (
        frame.groupby(keys, sort=False)["timestamp"]
        .transform("size")
        .astype(np.int16)
    )
    frame["source_position"] = np.arange(len(frame), dtype=np.int64)
    return frame[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "symbol_norm",
            "session_date",
            "month_key",
            "bar_ordinal",
            "segment_index",
            "segment_position",
            "segment_size",
            "source_position",
        ]
    ]


def contiguous_tape(rows: int = 14) -> pd.DataFrame:
    start = pd.Timestamp("2024-07-02 13:30:00", tz="UTC")
    timestamps = [start + pd.Timedelta(minutes=5 * index) for index in range(rows)]
    opens = [100.0 + 0.2 * index for index in range(rows)]
    closes = [value + (-0.08 if index % 2 else 0.12) for index, value in enumerate(opens)]
    return make_tape(
        timestamps,
        ["2024-07-02"] * rows,
        [0] * rows,
        opens,
        closes,
    )


def synthetic_scored_predictions(module) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target = np.asarray([0.0, 1.0, 2.0])
    symbols = ("A", "A", "B")
    for algorithm, prediction in (
        ("clock_ridge", np.asarray([1.0, 2.0, 3.0])),
        ("full_ridge", np.asarray([2.0, 5.0, 12.0])),
    ):
        for index, (symbol, actual, predicted) in enumerate(
            zip(symbols, target, prediction, strict=True)
        ):
            row: dict[str, object] = {
                "anchor_id": f"anchor-{index}",
                "algorithm": algorithm,
                "horizon": 6,
                "fold_month": "2024-07",
                "session_date": "2024-07-02",
                "symbol_norm": symbol,
                "target_bps": actual,
                "prediction_bps": predicted,
            }
            for threshold in module.ACTION_THRESHOLDS:
                label = module._threshold_label(threshold)
                row[f"action_{label}"] = int(
                    module.action_from_prediction([predicted], threshold)[0]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_contract_safety_and_clean_feature_boundary_are_frozen() -> None:
    module = load_runner()
    contract, manifest = module.load_contract_and_verify(require_pre_score=False)
    assert manifest is None
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["broker_connection_enabled"] is False
    assert contract["paper_or_demo_execution_enabled"] is False
    assert contract["deployment_enabled"] is False
    assert contract["strategy_promotion_permitted"] is False
    assert contract["economic_edge_claim_permitted"] is False
    assert contract["sources"]["provider_volume_label"] == "historical_volume_not_used"
    assert tuple(contract["sources"]["columns_read"]) == module.SOURCE_COLUMNS
    assert tuple(contract["feature_policy"]["full_features_in_order"]) == module.FULL_FEATURES
    assert len(module.FULL_FEATURES) == 40
    forbidden = tuple(value.lower() for value in contract["scope"]["forbidden_inputs"])
    assert not any(
        token in feature.lower()
        for feature in module.FULL_FEATURES
        for token in forbidden
    )
    for year in (2023, 2025, 2026):
        assert contract["periods"][f"{year}_read_permitted"] is False


def test_source_is_whitelisted_ohlc_only_and_imports_no_prior_detector() -> None:
    module = load_runner()
    assert module.SOURCE_COLUMNS == ("timestamp", "open", "high", "low", "close")
    source = inspect.getsource(module.load_tape)
    assert source.count("pd.read_parquet(") == 1
    assert "columns=list(SOURCE_COLUMNS)" in source
    assert "filters=_year_filter()" in source

    tree = ast.parse(RUNNER.read_text(), filename=str(RUNNER))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append((node.module or "").lower())
    assert not any(
        token in name
        for name in imported
        for token in ("regime", "loop", "semimarkov", "detector", "template")
    )


def test_feature_surface_is_causal_prefix_invariant() -> None:
    module = load_runner()
    original = contiguous_tape()
    altered_future = original.copy()
    suffix = altered_future.index >= 8
    altered_future.loc[suffix, "open"] += 75.0
    altered_future.loc[suffix, "close"] += 75.0
    altered_future.loc[suffix, "high"] += 75.0
    altered_future.loc[suffix, "low"] += 75.0

    baseline = module.build_feature_surface(original)
    changed = module.build_feature_surface(altered_future)
    pd.testing.assert_frame_equal(
        baseline.loc[:7, list(module.FULL_FEATURES)],
        changed.loc[:7, list(module.FULL_FEATURES)],
        check_exact=True,
    )


def test_gap_resets_segment_rolls_and_new_session_resets_running_context() -> None:
    module = load_runner()
    timestamps = [
        pd.Timestamp("2024-07-02 13:30:00", tz="UTC"),
        pd.Timestamp("2024-07-02 13:35:00", tz="UTC"),
        pd.Timestamp("2024-07-02 13:40:00", tz="UTC"),
        pd.Timestamp("2024-07-02 13:50:00", tz="UTC"),
        pd.Timestamp("2024-07-02 13:55:00", tz="UTC"),
        pd.Timestamp("2024-07-03 13:30:00", tz="UTC"),
        pd.Timestamp("2024-07-03 13:35:00", tz="UTC"),
    ]
    tape = make_tape(
        timestamps,
        ["2024-07-02"] * 5 + ["2024-07-03"] * 2,
        [0, 0, 0, 1, 1, 0, 0],
        [100.0, 101.0, 102.0, 110.0, 111.0, 50.0, 51.0],
        [101.0, 102.0, 103.0, 111.0, 112.0, 51.0, 52.0],
    )
    features = module.build_feature_surface(tape)

    assert np.isnan(features.loc[3, "close_return_1"])
    assert np.isnan(features.loc[3, "mean_log_range_3"])
    assert np.isclose(features.loc[3, "availability_3"], 1.0 / 3.0)
    assert np.isclose(features.loc[3, "session_log_return"], np.log(111.0 / 100.0))

    assert np.isnan(features.loc[5, "close_return_1"])
    assert np.isclose(features.loc[5, "availability_3"], 1.0 / 3.0)
    assert np.isclose(features.loc[5, "session_log_return"], np.log(51.0 / 50.0))
    assert np.isclose(features.loc[5, "distance_to_session_high"], np.log(51.0 / 52.0))
    assert np.isclose(features.loc[5, "distance_from_session_low"], np.log(51.0 / 49.0))


def test_exact_horizon_support_uses_next_open_and_fixed_exit_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    tape = contiguous_tape(rows=8)
    features = module.build_feature_surface(tape)
    monkeypatch.setitem(module.EXPECTED_ROWS, 6, 2)
    monkeypatch.setitem(module.EXPECTED_VALIDATION_ROWS, 6, 2)
    anchors = module.build_horizon_surface(features, 6)
    assert anchors["source_position"].tolist() == [0, 1]
    assert (
        anchors["entry_timestamp"] - anchors["decision_timestamp"]
    ).eq(pd.Timedelta(minutes=5)).all()
    assert (
        anchors["exit_timestamp"] - anchors["decision_timestamp"]
    ).eq(pd.Timedelta(minutes=30)).all()

    outcomes = module.attach_outcomes(tape, anchors, 6)
    assert np.allclose(outcomes["next_bar_open"], tape.loc[[1, 2], "open"])
    assert np.allclose(outcomes["exit_close"], tape.loc[[6, 7], "close"])
    expected = 10000.0 * (
        tape.loc[[6, 7], "close"].to_numpy(float)
        / tape.loc[[1, 2], "open"].to_numpy(float)
        - 1.0
    )
    assert np.allclose(outcomes["target_bps"], expected)


def test_action_threshold_equality_is_inclusive() -> None:
    module = load_runner()
    predictions = np.asarray([-10.0001, -10.0, -9.9999, 0.0, 9.9999, 10.0, 10.0001])
    actions = module.action_from_prediction(predictions, 10.0)
    assert actions.tolist() == [-1, -1, 0, 0, 0, 1, 1]


def test_greedy_nonoverlap_rearms_exactly_at_exit_time() -> None:
    module = load_runner()
    start = pd.Timestamp("2024-07-02 13:30:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol_norm": ["X"] * 5,
            "session_date": ["2024-07-02"] * 5,
            "decision_timestamp": [
                start + pd.Timedelta(minutes=value) for value in (0, 25, 30, 55, 60)
            ],
        }
    )
    accepted = module.greedy_nonoverlap(frame, [1, 1, -1, -1, 1], horizon=6)
    assert accepted.tolist() == [1, 0, -1, 0, 1]


def test_short_gross_return_is_cash_consistent_simple_return() -> None:
    module = load_runner()
    observed = module.gross_returns([-1, 1], [100.0, 100.0], [90.0, 110.0])
    assert np.allclose(observed, [0.10, 0.10])
    assert not np.isclose(observed[0], 100.0 / 90.0 - 1.0)
    assert not np.isclose(observed[0], -np.log(90.0 / 100.0))


def test_training_weights_give_each_represented_symbol_equal_total_weight() -> None:
    module = load_runner()
    symbols = np.asarray(["A", "A", "B", "B", "B", "B"])
    weights = module.equal_symbol_weights(symbols)
    assert np.allclose(weights, [1.5, 1.5, 0.75, 0.75, 0.75, 0.75])
    assert np.isclose(weights.mean(), 1.0)
    totals = pd.Series(weights).groupby(symbols).sum()
    assert np.allclose(totals.to_numpy(float), [3.0, 3.0])


def test_daily_prediction_loss_averages_symbol_day_first_with_fixed_sleeves() -> None:
    module = load_runner()
    scored = synthetic_scored_predictions(module)
    _, _, _, daily = module.evaluate_predictions(scored)
    observed = daily.loc[
        daily["algorithm"].eq("full_ridge") & daily["horizon"].eq(6)
    ].iloc[0]

    # A has errors 2 and 4, so its symbol/day MSE is 10 and MAE is 3.
    # B has one error of 10, so its MSE is 100 and MAE is 10.  Missing
    # sleeves remain cash and the two represented sleeves are divided by 22.
    assert np.isclose(observed["mse_bps2"], (10.0 + 100.0) / 22.0)
    assert np.isclose(observed["mae_bps"], (3.0 + 10.0) / 22.0)
    assert not np.isclose(observed["mse_bps2"], np.mean([4.0, 16.0, 100.0]))
    assert not np.isclose(observed["mse_bps2"], np.mean([10.0, 100.0]))


def test_prediction_loso_schema_is_reconstructed_and_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    contract = json.loads(module.CONTRACT_PATH.read_text())
    assert "unchanged-prediction leave-one-stock-out deletion" in contract["metrics"][
        "slices"
    ]
    scored = synthetic_scored_predictions(module)
    monkeypatch.setattr(module, "SYMBOLS", ("A", "B"))
    monkeypatch.setattr(module, "ALGORITHMS", ("clock_ridge", "full_ridge"))
    monkeypatch.setattr(module, "HORIZONS", (6,))
    deletions = module.evaluate_prediction_stock_deletions(scored)
    assert len(deletions) == 4
    assert set(
        [
            "algorithm",
            "horizon",
            "deleted_symbol",
            "mse_bps2",
            "mae_bps",
            "clock_mse_bps2",
            "clock_mae_bps",
            "relative_mse_improvement_vs_clock",
            "relative_mae_improvement_vs_clock",
        ]
    ).issubset(deletions.columns)

    scoring_source = inspect.getsource(module.run_scoring)
    assert "evaluate_prediction_stock_deletions(scored)" in scoring_source
    assert 'prediction_deletions.to_csv(OUT / "prediction_stock_deletions.csv"' in scoring_source


def test_fold_chronology_and_preprocessing_are_training_only() -> None:
    module = load_runner()
    surface = pd.DataFrame(
        {
            "exit_timestamp": pd.to_datetime(
                [
                    "2024-06-30 23:55:00+00:00",
                    "2024-07-01 00:00:00+00:00",
                    "2024-07-02 14:00:00+00:00",
                    "2024-08-01 14:00:00+00:00",
                ],
                utc=True,
            ),
            "month_key": ["2024-06", "2024-06", "2024-07", "2024-08"],
        }
    )
    train, score = module.fold_masks(surface, "2024-07")
    assert train.tolist() == [True, False, False, False]
    assert score.tolist() == [False, False, True, False]
    assert not np.logical_and(train, score).any()

    train_values = np.asarray([[1.0, np.nan], [3.0, 5.0]])
    validation_values = np.asarray([[1000.0, np.nan]])
    medians = module.training_medians(train_values)
    transformed = module.apply_medians(validation_values, medians)
    assert np.allclose(medians, [2.0, 5.0])
    assert np.allclose(transformed, [[1000.0, 5.0]])

    fit_source = inspect.getsource(module.fit_monthly_oof_predictions)
    assert "training_medians(train_raw)" in fit_source
    assert "apply_medians(score_raw, medians)" in fit_source
    assert "scaler.fit(train_imputed, sample_weight=weights)" in fit_source
    assert "estimator.fit(train_values, target, sample_weight=weights)" in fit_source
    assert "attach_outcomes(tape, train, horizon)" in fit_source
    assert "attach_outcomes(tape, score" not in fit_source


def test_completed_artifacts_remain_research_only_and_audit_passes() -> None:
    if not ARTIFACT_ROOT.is_dir():
        pytest.skip("clean-slate score artifacts have not been created yet")
    decision_path = ARTIFACT_ROOT / "decision.json"
    audit_path = ARTIFACT_ROOT / "independent_audit.json"
    assert decision_path.is_file()
    assert audit_path.is_file()
    decision = json.loads(decision_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert decision["research_only"] is True
    assert decision["live_ordering_enabled"] is False
    assert decision["order_placement"] == "disabled"
    assert decision["strategy_promotion"] is False
    assert decision["economic_edge_claim"] is False
    assert set(decision["retained_candidates"]).issubset({"full_ridge", "full_hgb"})
    assert audit["all_passed"] is True
    assert audit["failed"] == 0
