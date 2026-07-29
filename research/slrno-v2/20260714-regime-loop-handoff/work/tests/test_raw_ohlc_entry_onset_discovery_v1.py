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
RUNNER = WORKSPACE / "work/run_raw_ohlc_entry_onset_discovery_v1.py"
CLEAN_RUNNER = WORKSPACE / "work/run_clean_slate_causal_ohlc_entries_v1.py"
CONTRACT = WORKSPACE / "work/contracts/20260712-raw-ohlc-entry-onset-discovery-v1.json"
ARTIFACT_ROOT = Path("/private/tmp/stocker_raw_ohlc_entry_onset_discovery_v1_20260712")


EXPECTED_FEATURES = (
    "clock_fraction",
    "clock_fraction_squared",
    "clock_sin_1",
    "clock_cos_1",
    "clock_sin_2",
    "clock_cos_2",
    "log_close_open",
    "log_high_low",
    "signed_body_fraction",
    "absolute_body_fraction",
    "upper_wick_fraction",
    "lower_wick_fraction",
    "close_location",
    "close_return_1",
    "close_return_3",
    "close_return_6",
    "close_return_12",
    "mean_abs_close_return_3",
    "mean_abs_close_return_6",
    "mean_abs_close_return_12",
    "std_close_return_3",
    "std_close_return_6",
    "std_close_return_12",
    "mean_log_range_3",
    "mean_log_range_6",
    "mean_log_range_12",
    "log_range_ratio_6",
    "log_range_ratio_12",
    "session_log_return",
    "distance_to_session_high",
    "distance_from_session_low",
    "session_range_location",
    "running_log_range",
    "distance_to_rolling_high_6",
    "distance_from_rolling_low_6",
    "distance_to_rolling_high_12",
    "distance_from_rolling_low_12",
    "availability_3",
    "availability_6",
    "availability_12",
)


FORBIDDEN_PRE_OUTCOME_COLUMNS = {
    "target",
    "target_class",
    "class_label",
    "status",
    "upside_mfe_bps",
    "downside_mfe_bps",
    "first_confirmation_step",
    "favourable_excursion_scale_units",
    "adverse_excursion_scale_units",
    "pre_confirmation_adverse_scale_units",
    "directional_dominance_scale_units",
    "rapid_correct_confirmation_within_3_bars",
    "exit_price",
    "gross_return",
    "net_return",
    "pnl",
    "position",
    "order",
}


def load_runner():
    work_path = str(RUNNER.parent)
    if work_path not in sys.path:
        sys.path.insert(0, work_path)
    spec = importlib.util.spec_from_file_location(
        "raw_ohlc_entry_onset_discovery_v1", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_clean_runner():
    spec = importlib.util.spec_from_file_location(
        "clean_slate_causal_ohlc_entries_v1_feature_reference", CLEAN_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_tape(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    start: str = "2024-07-02 13:30:00+00:00",
    symbol: str = "X",
    session_date: str = "2024-07-02",
    segment_index: int = 0,
) -> pd.DataFrame:
    rows = len(opens)
    timestamps = pd.date_range(start, periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.asarray(opens, float),
            "high": np.asarray(highs, float),
            "low": np.asarray(lows, float),
            "close": np.asarray(closes, float),
            "symbol_norm": symbol,
            "session_date": session_date,
            "month_key": session_date[:7],
            "bar_ordinal": np.arange(rows, dtype=np.int16),
            "segment_index": np.int16(segment_index),
            "segment_position": np.arange(rows, dtype=np.int16),
            "segment_size": np.int16(rows),
            "source_position": np.arange(rows, dtype=np.int64),
        }
    )
    return frame


def future_bars(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float] | None = None,
) -> pd.DataFrame:
    if closes is None:
        closes = opens
    return pd.DataFrame(
        {
            "open": np.asarray(opens, float),
            "high": np.asarray(highs, float),
            "low": np.asarray(lows, float),
            "close": np.asarray(closes, float),
        }
    )


def labelled_path(
    module,
    future: pd.DataFrame,
    *,
    reference: float = 100.0,
    scale_bps: float = 100.0,
    horizon: int | None = None,
) -> dict[str, object]:
    if horizon is None:
        horizon = len(future)
    observed = module.label_path(
        next_open=reference,
        future_bars=future.iloc[:horizon].copy(),
        scale_bps=scale_bps,
    )
    if isinstance(observed, pd.Series):
        return observed.to_dict()
    assert isinstance(observed, dict)
    return observed


def test_contract_safety_source_and_feature_surface_are_exactly_frozen() -> None:
    module = load_runner()
    contract, manifest = module.load_contract_and_verify(require_pre_outcome=False)
    assert manifest is None
    assert contract == json.loads(CONTRACT.read_text())
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["broker_connection_enabled"] is False
    assert contract["paper_or_demo_execution_enabled"] is False
    assert contract["deployment_enabled"] is False
    assert contract["strategy_promotion_permitted"] is False
    assert contract["economic_edge_claim_permitted"] is False
    assert contract["pnl_evaluation_permitted"] is False
    assert contract["sources"]["provider_volume_label"] == "historical_volume_not_used"
    assert tuple(contract["sources"]["columns_read"]) == module.SOURCE_COLUMNS
    assert module.SOURCE_COLUMNS == ("timestamp", "open", "high", "low", "close")
    assert tuple(contract["features"]["full_features_in_order"]) == EXPECTED_FEATURES
    assert module.FULL_FEATURES == EXPECTED_FEATURES
    assert len(module.FULL_FEATURES) == 40
    for year in (2023, 2025, 2026):
        assert contract["periods"][f"{year}_read_permitted"] is False


def test_runner_ast_has_no_detector_inputs_or_pnl_evaluation_surface() -> None:
    module = load_runner()
    source = RUNNER.read_text()
    tree = ast.parse(source, filename=str(RUNNER))

    imported: list[str] = []
    function_names: set[str] = set()
    assigned_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append((node.module or "").lower())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name.lower())
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned_names.add(node.id.lower())

    forbidden_import_tokens = ("regime", "semimarkov", "loop", "cycle", "template")
    assert not any(
        token in imported_name
        for imported_name in imported
        for token in forbidden_import_tokens
    )
    forbidden_runtime_names = {
        "gross_returns",
        "net_returns",
        "profit_factor",
        "sharpe",
        "drawdown",
        "build_positions",
        "place_order",
        "submit_order",
    }
    assert not (function_names | assigned_names) & forbidden_runtime_names

    clean_slate_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "clean_slate"
    }
    assert clean_slate_attributes <= {
        "SOURCE_COLUMNS",
        "SYMBOLS",
        "CLOCK_FEATURES",
        "FULL_FEATURES",
        "provider_path",
        "load_tape",
        "build_feature_surface",
        "__file__",
    }

    load_source = inspect.getsource(module.load_tape)
    assert "clean_slate.load_tape" in load_source
    frozen_load_source = inspect.getsource(module.clean_slate.load_tape)
    assert frozen_load_source.count("pd.read_parquet(") == 1
    assert "columns=list(SOURCE_COLUMNS)" in frozen_load_source
    assert "filters=_year_filter()" in frozen_load_source


def test_feature_values_exactly_match_the_frozen_clean_ohlc_surface() -> None:
    module = load_runner()
    reference = load_clean_runner()
    opens = [100.0 + 0.2 * index for index in range(16)]
    closes = [
        value + (0.15 if index % 2 == 0 else -0.09) for index, value in enumerate(opens)
    ]
    highs = [
        max(open_, close) + 0.25 for open_, close in zip(opens, closes, strict=True)
    ]
    lows = [
        min(open_, close) - 0.20 for open_, close in zip(opens, closes, strict=True)
    ]
    tape = make_tape(opens, highs, lows, closes)
    observed = module.build_feature_surface(tape)
    expected = reference.build_feature_surface(tape)
    pd.testing.assert_frame_equal(
        observed.loc[:, list(EXPECTED_FEATURES)],
        expected.loc[:, list(EXPECTED_FEATURES)],
        check_exact=True,
    )


def test_causal_scale_ends_at_t_minus_one_is_prefix_invariant_and_floored() -> None:
    module = load_runner()
    baseline = make_tape(
        opens=[100.0] * 7,
        highs=[100.0] * 7,
        lows=[100.0] * 7,
        closes=[100.0] * 7,
    )
    altered = baseline.copy()
    altered.loc[3:, ["open", "high", "low", "close"]] = [
        [100.0, 200.0, 50.0, 100.0],
        [100.0, 220.0, 40.0, 100.0],
        [100.0, 240.0, 30.0, 100.0],
        [100.0, 260.0, 20.0, 100.0],
    ]

    observed = np.asarray(module.causal_scale_bps(baseline), dtype=float)
    changed = np.asarray(module.causal_scale_bps(altered), dtype=float)
    assert np.isclose(observed[3], 1.0)
    assert np.isclose(changed[3], 1.0)
    assert observed[3] == changed[3]
    assert np.isclose(observed[6], 1.0)
    assert changed[6] > changed[3]
    assert np.isnan(observed[:3]).all()


def test_path_label_upper_first_lower_first_and_no_hit() -> None:
    module = load_runner()
    scale = 100.0
    upper = 100.0 * np.exp(scale / 10000.0)
    lower = 100.0 * np.exp(-scale / 10000.0)

    long_path = labelled_path(
        module,
        future_bars(
            [100.0, 100.0],
            [100.5, upper + 0.01],
            [99.5, lower + 0.01],
        ),
        scale_bps=scale,
    )
    short_path = labelled_path(
        module,
        future_bars(
            [100.0, 100.0],
            [100.5, upper - 0.01],
            [99.5, lower - 0.01],
        ),
        scale_bps=scale,
    )
    no_hit = labelled_path(
        module,
        future_bars([100.0, 100.0], [100.5, 100.6], [99.5, 99.4]),
        scale_bps=scale,
    )
    assert (long_path["status"], long_path["target_class"]) == ("long_first", 1)
    assert long_path["first_confirmation_step"] == 2
    assert (short_path["status"], short_path["target_class"]) == (
        "short_first",
        2,
    )
    assert short_path["first_confirmation_step"] == 2
    assert (no_hit["status"], no_hit["target_class"]) == (
        "no_hit_by_horizon",
        0,
    )
    assert pd.isna(no_hit["first_confirmation_step"])


def test_path_label_barrier_equality_last_bar_and_post_horizon() -> None:
    module = load_runner()
    scale = 100.0
    upper = 100.0 * np.exp(scale / 10000.0)
    lower = 100.0 * np.exp(-scale / 10000.0)

    equality = labelled_path(
        module,
        future_bars([100.0], [upper], [lower + 0.10]),
        scale_bps=scale,
    )
    assert equality["status"] == "long_first"
    assert equality["first_confirmation_step"] == 1

    bars = future_bars(
        [100.0] * 4,
        [100.2, 100.3, upper, upper + 1.0],
        [99.8, 99.7, lower + 0.10, lower + 0.10],
    )
    last_bar = labelled_path(module, bars, scale_bps=scale, horizon=3)
    assert last_bar["status"] == "long_first"
    assert last_bar["first_confirmation_step"] == 3

    post_horizon_only = bars.copy()
    post_horizon_only.loc[2, "high"] = upper - 0.01
    ignored = labelled_path(module, post_horizon_only, scale_bps=scale, horizon=3)
    assert ignored["status"] == "no_hit_by_horizon"
    assert pd.isna(ignored["first_confirmation_step"])


def test_dual_touch_is_ambiguous_but_open_gap_has_precedence() -> None:
    module = load_runner()
    scale = 100.0
    upper = 100.0 * np.exp(scale / 10000.0)
    lower = 100.0 * np.exp(-scale / 10000.0)

    ambiguous = labelled_path(
        module,
        future_bars([100.0], [upper], [lower]),
        scale_bps=scale,
    )
    assert (ambiguous["status"], ambiguous["target_class"]) == (
        "intrabar_ambiguous",
        0,
    )

    gap_up = labelled_path(
        module,
        future_bars(
            [100.0, upper],
            [100.5, upper + 1.0],
            [99.5, lower - 1.0],
        ),
        scale_bps=scale,
    )
    assert (gap_up["status"], gap_up["target_class"]) == ("long_first", 1)
    assert gap_up["first_confirmation_step"] == 2

    gap_down = labelled_path(
        module,
        future_bars(
            [100.0, lower],
            [100.5, upper + 1.0],
            [99.5, lower - 1.0],
        ),
        scale_bps=scale,
    )
    assert (gap_down["status"], gap_down["target_class"]) == (
        "short_first",
        2,
    )
    assert gap_down["first_confirmation_step"] == 2


def test_path_excursions_and_directional_quality_are_mirror_symmetric() -> None:
    module = load_runner()
    original = future_bars(
        [100.0, 100.2, 100.6],
        [100.4, 101.4, 102.2],
        [99.8, 99.4, 100.0],
        [100.2, 100.6, 101.7],
    )
    mirror = pd.DataFrame(
        {
            "open": 10000.0 / original["open"].to_numpy(float),
            "high": 10000.0 / original["low"].to_numpy(float),
            "low": 10000.0 / original["high"].to_numpy(float),
            "close": 10000.0 / original["close"].to_numpy(float),
        }
    )
    up = labelled_path(module, original)
    down = labelled_path(module, mirror)
    assert (up["status"], down["status"]) == ("long_first", "short_first")
    assert np.isclose(up["upside_mfe_bps"], down["downside_mfe_bps"])
    assert np.isclose(up["downside_mfe_bps"], down["upside_mfe_bps"])
    for name in (
        "favourable_excursion_scale_units",
        "adverse_excursion_scale_units",
        "pre_confirmation_adverse_scale_units",
        "directional_dominance_scale_units",
    ):
        assert np.isclose(up[name], down[name])


def test_training_and_metric_weights_are_nested_equal_symbol_and_session() -> None:
    module = load_runner()
    frame = pd.DataFrame(
        {
            "symbol_norm": ["A"] * 6 + ["B"] * 3,
            "session_date": ["d1"] * 2 + ["d2"] * 4 + ["d1"] * 3,
        }
    )
    observed = np.asarray(module.nested_symbol_session_weights(frame), dtype=float)
    assert np.isclose(observed.mean(), 1.0)

    totals = pd.Series(observed).groupby(frame["symbol_norm"].to_numpy()).sum()
    assert np.allclose(totals.loc[["A", "B"]], [4.5, 4.5])
    session_totals = (
        frame.assign(weight=observed)
        .groupby(["symbol_norm", "session_date"], sort=True)["weight"]
        .sum()
    )
    assert np.allclose(session_totals.loc[[("A", "d1"), ("A", "d2")]], [2.25, 2.25])
    assert np.isclose(session_totals.loc[("B", "d1")], 4.5)


def test_fold_training_requires_complete_target_path_strictly_before_month() -> None:
    module = load_runner()
    month_start = pd.Timestamp("2024-07-01 00:00:00", tz="UTC")
    surface = pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(
                [
                    "2024-06-28 14:00:00+00:00",
                    "2024-06-28 14:05:00+00:00",
                    "2024-07-02 14:00:00+00:00",
                    "2024-08-01 14:00:00+00:00",
                ],
                utc=True,
            ),
            "path_end_timestamp": pd.DatetimeIndex(
                [
                    month_start - pd.Timedelta(minutes=5),
                    month_start,
                    pd.Timestamp("2024-07-02 16:00:00", tz="UTC"),
                    pd.Timestamp("2024-08-01 16:00:00", tz="UTC"),
                ]
            ),
            "month_key": ["2024-06", "2024-06", "2024-07", "2024-08"],
        }
    )
    train, score = module.fold_masks(surface, "2024-07")
    assert np.asarray(train).tolist() == [True, False, False, False]
    assert np.asarray(score).tolist() == [False, False, True, False]
    assert not np.logical_and(train, score).any()


def test_weighted_quantile_is_left_inverse_cdf_with_stable_ties() -> None:
    module = load_runner()
    values = np.asarray([5.0, 1.0, 3.0, 3.0, 9.0])
    weights = np.asarray([1.0, 1.0, 2.0, 5.0, 1.0])
    assert module.weighted_quantile(values, weights, 0.0) == 1.0
    assert module.weighted_quantile(values, weights, 0.10) == 1.0
    assert module.weighted_quantile(values, weights, 0.11) == 3.0
    assert module.weighted_quantile(values, weights, 0.80) == 3.0
    assert module.weighted_quantile(values, weights, 0.81) == 5.0
    assert module.weighted_quantile(values, weights, 1.0) == 9.0


def probability_ledger_for_thresholds() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, base in (("2024-06", 0.00), ("2024-07", 0.40), ("2024-08", 0.70)):
        for index in range(1, 21):
            p_long = base + index / 100.0
            p_short = (21 - index) / 100.0
            decision_timestamp = pd.Timestamp(
                f"{month}-{1 + ((index - 1) // 5):02d} 14:00:00", tz="UTC"
            ) + pd.Timedelta(minutes=5 * ((index - 1) % 5))
            rows.append(
                {
                    "anchor_id": f"{month}-{index:02d}",
                    "algorithm": "full_logit",
                    "horizon": 6,
                    "fold_month": month,
                    "month_key": month,
                    "symbol_norm": "A" if index <= 10 else "B",
                    "session_date": f"{month}-{1 + ((index - 1) // 5):02d}",
                    "decision_timestamp": decision_timestamp,
                    "bar_ordinal": 6 + index,
                    "segment_index": 0,
                    "segment_position": 6 + index,
                    "causal_scale_bps": 20.0,
                    "availability_12": 1.0,
                    "clock_bin_15": (6 + index) // 3,
                    "clock_bin_30": (6 + index) // 6,
                    "availability_bucket": 2,
                    "p_no_entry": 1.0 - p_long - p_short,
                    "p_long_first": p_long,
                    "p_short_first": p_short,
                }
            )
    return pd.DataFrame(rows)


def test_alert_thresholds_use_immediately_prior_month_oof_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    ledger = probability_ledger_for_thresholds()
    ledger = ledger.loc[:, module.PRE_OUTCOME_LEDGER_COLUMNS]
    monkeypatch.setattr(module, "ALGORITHMS", ("full_logit",))
    monkeypatch.setattr(module, "HORIZONS", (6,))
    monkeypatch.setattr(module, "VALIDATION_MONTHS", ("2024-07", "2024-08"))
    thresholds = module.calibrate_prior_month_thresholds(ledger)
    july = thresholds.loc[
        thresholds["score_month"].eq("2024-07")
        & thresholds["algorithm"].eq("full_logit")
        & thresholds["horizon"].eq(6)
        & thresholds["side"].eq("long")
    ].iloc[0]
    august = thresholds.loc[
        thresholds["score_month"].eq("2024-08")
        & thresholds["algorithm"].eq("full_logit")
        & thresholds["horizon"].eq(6)
        & thresholds["side"].eq("long")
    ].iloc[0]
    assert july["threshold_source_month"] == "2024-06"
    assert np.isclose(july["fire_threshold"], 0.19)
    assert np.isclose(july["rearm_threshold"], 0.15)
    assert august["threshold_source_month"] == "2024-07"
    assert np.isclose(august["fire_threshold"], 0.59)
    assert np.isclose(august["rearm_threshold"], 0.55)

    altered = ledger.copy()
    changed_rows = altered["fold_month"].ne("2024-06")
    altered.loc[changed_rows, "p_long_first"] = 0.75
    altered.loc[changed_rows, "p_no_entry"] = (
        1.0
        - altered.loc[changed_rows, "p_long_first"]
        - altered.loc[changed_rows, "p_short_first"]
    )
    changed = module.calibrate_prior_month_thresholds(altered)
    july_changed = changed.loc[
        changed["score_month"].eq("2024-07")
        & changed["algorithm"].eq("full_logit")
        & changed["horizon"].eq(6)
        & changed["side"].eq("long")
    ].iloc[0]
    assert july_changed["fire_threshold"] == july["fire_threshold"]
    assert july_changed["rearm_threshold"] == july["rearm_threshold"]


def onset_ledger() -> pd.DataFrame:
    long_prob = [0.40, 0.50, 0.40, 0.39, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]
    short_prob = [0.10, 0.10, 0.10, 0.10, 0.10, 0.40, 0.50, 0.50, 0.39, 0.50, 0.10]
    rows = len(long_prob)
    decision_timestamps = pd.date_range(
        "2024-07-02 14:00:00", periods=rows, freq="5min", tz="UTC"
    )
    return pd.DataFrame(
        {
            "anchor_id": [f"a{index}" for index in range(rows)],
            "algorithm": "full_logit",
            "horizon": 6,
            "fold_month": "2024-07",
            "month_key": "2024-07",
            "symbol_norm": "X",
            "session_date": "2024-07-02",
            "decision_timestamp": decision_timestamps,
            "segment_index": [0] * 10 + [1],
            "segment_position": list(range(10)) + [0],
            "bar_ordinal": np.arange(rows),
            "causal_scale_bps": 20.0,
            "availability_12": 1.0,
            "availability_bucket": 2,
            "clock_bin_15": np.arange(rows) // 3,
            "clock_bin_30": np.arange(rows) // 6,
            "p_no_entry": 1.0 - np.asarray(long_prob) - np.asarray(short_prob),
            "p_long_first": long_prob,
            "p_short_first": short_prob,
        }
    )


def test_threshold_collision_hysteresis_deduplicates_and_resets_at_segment() -> None:
    module = load_runner()
    ledger = onset_ledger()
    thresholds = pd.DataFrame(
        [
            {
                "score_month": "2024-07",
                "threshold_source_month": "2024-06",
                "algorithm": "full_logit",
                "horizon": 6,
                "side": side,
                "fire_threshold": 0.40,
                "rearm_threshold": 0.40,
            }
            for side in ("long", "short")
        ]
    )
    onsets, states = module.extract_onsets(ledger, thresholds, return_state=True)
    ordered = onsets.sort_values("decision_timestamp", kind="stable")
    assert ordered["anchor_id"].tolist() == ["a0", "a4", "a6", "a9", "a10"]
    assert ordered["side"].tolist() == ["long", "long", "short", "short", "long"]
    assert "a1" not in set(onsets["anchor_id"])
    assert "a2" not in set(onsets["anchor_id"])
    assert "a5" not in set(onsets["anchor_id"])
    assert "a7" not in set(onsets["anchor_id"])
    assert onsets["horizon"].eq(6).all()
    state = states.set_index("anchor_id")
    assert not bool(state.loc["a2", "long_armed_after"])
    assert bool(state.loc["a3", "long_rearmed"])
    assert bool(state.loc["a10", "long_fire"])


def test_pre_outcome_ledger_rejects_every_outcome_or_pnl_column() -> None:
    module = load_runner()
    frozen_columns = {name.lower() for name in module.PRE_OUTCOME_LEDGER_COLUMNS}
    assert frozen_columns.isdisjoint(FORBIDDEN_PRE_OUTCOME_COLUMNS)
    safe = probability_ledger_for_thresholds().loc[:, module.PRE_OUTCOME_LEDGER_COLUMNS]
    module.validate_pre_outcome_ledger(safe)
    for forbidden in sorted(FORBIDDEN_PRE_OUTCOME_COLUMNS):
        contaminated = safe.copy()
        contaminated[forbidden] = 0.0
        with pytest.raises((AssertionError, ValueError)):
            module.validate_pre_outcome_ledger(contaminated)


def test_clock_controls_are_outcome_blind_deterministic_and_without_replacement() -> (
    None
):
    module = load_runner()
    onsets = pd.DataFrame(
        [
            {
                "onset_id": f"onset-{index}",
                "anchor_id": f"candidate-{index}",
                "candidate_algorithm": "full_logit",
                "horizon": 6,
                "side": "long",
                "fold_month": "2024-07",
                "symbol_norm": "X",
                "session_date": "2024-07-02",
                "bar_ordinal": 30,
                "clock_bin_15": 10,
                "clock_bin_30": 5,
                "availability_bucket": 2,
            }
            for index in (1, 2)
        ]
    )
    control_specs = (
        ("control-b", "2024-07-03", 0.90, 2.0),
        ("control-a", "2024-07-04", 0.90, -5.0),
        ("control-c", "2024-07-05", 0.80, 100.0),
    )
    ledger_rows: list[dict[str, object]] = []
    for control_id, session_date, probability, hidden_outcome in control_specs:
        ledger_rows.append(
            {
                "anchor_id": control_id,
                "algorithm": "clock_logit",
                "horizon": 6,
                "fold_month": "2024-07",
                "symbol_norm": "X",
                "session_date": session_date,
                "decision_timestamp": pd.Timestamp(
                    f"{session_date} 16:00:00", tz="UTC"
                ),
                "bar_ordinal": 30,
                "clock_bin_15": 10,
                "clock_bin_30": 5,
                "availability_bucket": 2,
                "p_no_entry": 0.05,
                "p_long_first": probability,
                "p_short_first": 0.95 - probability,
                "target_class": hidden_outcome,
                "upside_mfe_bps": -hidden_outcome,
            }
        )
    contaminated = pd.DataFrame(ledger_rows)
    baseline = module.match_clock_controls(
        onsets, contaminated.drop(columns=["target_class", "upside_mfe_bps"])
    )
    changed = contaminated.copy()
    changed["target_class"] = changed["target_class"].iloc[::-1].to_numpy()
    changed["upside_mfe_bps"] *= -1000.0
    outcome_perturbed = module.match_clock_controls(onsets, changed)

    assert baseline["control_anchor_id"].tolist() == ["control-a", "control-b"]
    assert baseline["control_anchor_id"].is_unique
    assert baseline["match_tier"].eq(0).all()
    pd.testing.assert_series_equal(
        baseline["control_anchor_id"],
        outcome_perturbed["control_anchor_id"],
        check_names=False,
    )


def test_logit_reasons_exactly_reconstruct_chosen_vs_opposite_margin() -> None:
    module = load_runner()

    class DummyMultinomialLogit:
        classes_ = np.asarray([0, 1, 2])
        coef_ = np.vstack(
            [
                np.linspace(-0.2, 0.3, len(EXPECTED_FEATURES)),
                np.linspace(0.4, -0.1, len(EXPECTED_FEATURES)),
                np.linspace(-0.3, 0.2, len(EXPECTED_FEATURES)),
            ]
        )
        intercept_ = np.asarray([0.15, -0.25, 0.40])

        def decision_function(self, values: np.ndarray) -> np.ndarray:
            return values @ self.coef_.T + self.intercept_

    estimator = DummyMultinomialLogit()
    standardized = np.vstack(
        [
            np.linspace(-2.0, 2.0, len(EXPECTED_FEATURES)),
            np.linspace(1.5, -0.5, len(EXPECTED_FEATURES)),
        ]
    )
    long_result = module.logit_reason_contributions(
        estimator, standardized, EXPECTED_FEATURES, "long"
    )
    short_result = module.logit_reason_contributions(
        estimator, standardized, EXPECTED_FEATURES, "short"
    )

    coefficient_delta = estimator.coef_[1] - estimator.coef_[2]
    expected_contributions = standardized * coefficient_delta
    expected_intercept = estimator.intercept_[1] - estimator.intercept_[2]
    expected_margin = expected_intercept + expected_contributions.sum(axis=1)
    assert np.isclose(long_result["intercept"], expected_intercept)
    assert np.allclose(
        long_result["feature_contributions"], expected_contributions, atol=0.0
    )
    assert np.allclose(long_result["directional_margin"], expected_margin)
    grouped = sum(long_result["group_contributions"].values())
    assert np.allclose(long_result["intercept"] + grouped, expected_margin)

    assert np.isclose(short_result["intercept"], -expected_intercept)
    assert np.allclose(
        short_result["feature_contributions"], -expected_contributions, atol=0.0
    )
    assert np.allclose(short_result["directional_margin"], -expected_margin)


def test_completed_artifacts_preserve_pre_outcome_boundary_and_pass_audit() -> None:
    if not ARTIFACT_ROOT.is_dir():
        pytest.skip("entry-onset artifacts have not been created yet")
    module = load_runner()
    decision_path = ARTIFACT_ROOT / "decision.json"
    audit_path = ARTIFACT_ROOT / "independent_audit.json"
    freeze_path = ARTIFACT_ROOT / "prediction_onset_control_reason_freeze.json"
    probability_path = ARTIFACT_ROOT / "probabilities_pre_outcome.parquet"
    for path in (decision_path, audit_path, freeze_path, probability_path):
        assert path.is_file(), path

    decision = json.loads(decision_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    audit = json.loads(audit_path.read_text())
    for document in (decision, freeze):
        assert document["research_only"] is True
        assert document["live_ordering_enabled"] is False
        assert document["order_placement"] == "disabled"
    assert freeze["validation_paths_present"] is False
    assert freeze["terminal_return_or_economic_outcome_used"] is False
    assert decision.get("strategy_promotion", False) is False
    assert decision.get("economic_edge_claim", False) is False

    probabilities = pd.read_parquet(probability_path)
    assert tuple(probabilities.columns) == module.PRE_OUTCOME_LEDGER_COLUMNS
    assert {column.lower() for column in probabilities.columns}.isdisjoint(
        FORBIDDEN_PRE_OUTCOME_COLUMNS
    )
    module.validate_pre_outcome_ledger(probabilities)
    assert audit["all_passed"] is True
    assert audit["failed"] == 0
