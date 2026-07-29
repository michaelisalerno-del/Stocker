from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "factor_conditioned_loop_occurrence_core.py"
)
SPEC = importlib.util.spec_from_file_location("factor_loop_core", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def _sequence_anchors(states: list[int]) -> pd.DataFrame:
    count = len(states)
    frame = pd.DataFrame(
        {
            "anchor_id": np.arange(count),
            "period": "synthetic",
            "symbol_norm": "AAA",
            "session_date": "2024-07-01",
            "start_timestamp": pd.date_range(
                "2024-07-01 13:30:00+00:00", periods=count, freq="5min"
            ),
            "month": "2024-07",
            "quarter": "2024_q3",
            "state": states,
            "bar_ordinal": np.arange(count),
            "b0_unknown": False,
            "entry_minutes": np.arange(count) * 5.0,
            "b0_entry_numeric": 0.0,
            "b0_entry_high_stress": 0.0,
            "entry_time_sin": 0.0,
            "entry_time_cos": 1.0,
            "current_bar_log_return": np.linspace(-0.01, 0.01, count),
            "return_sum_6": np.linspace(-0.02, 0.02, count),
            "mean_abs_return_12": np.linspace(0.001, 0.005, count),
            "session_return": np.linspace(-0.03, 0.03, count),
            "bar_range_pct": np.linspace(0.002, 0.006, count),
        }
    )
    state_series = pd.Series(states)
    frame["previous_state_1"] = state_series.shift(1).fillna(core.END_STATE).astype(int)
    frame["previous_state_2"] = state_series.shift(2).fillna(core.END_STATE).astype(int)
    frame["next_outcome"] = state_series.shift(-1).fillna(core.END_STATE).astype(int)
    frame["terminal"] = frame.index == count - 1
    for step in range(1, 5):
        frame[f"future_state_{step}"] = (
            state_series.shift(-step).fillna(core.END_STATE).astype(int)
        )
    frame["history_token"] = core.history_tokens(
        frame["previous_state_2"], frame["previous_state_1"], frame["state"]
    )
    return frame


def _uniform_kernels() -> tuple[core.TransitionKernel, core.TransitionKernel]:
    classes = np.arange(core.DESTINATIONS)
    history = core.TransitionKernel(
        classes,
        np.zeros((core.DESTINATIONS, core.TOKEN_WIDTH)),
        np.zeros(core.DESTINATIONS),
        0,
        1,
    )
    limited = core.TransitionKernel(
        classes,
        np.zeros((core.DESTINATIONS, core.TOKEN_WIDTH + 4)),
        np.zeros(core.DESTINATIONS),
        4,
        1,
    )
    return history, limited


def test_contract_hash_safety_widths_and_phase_guard() -> None:
    contract = core.validate_contract()
    assert core.sha256(core.CONTRACT_PATH) == core.CONTRACT_SHA256
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert core.self_tests()["passed"] == core.self_tests()["total"]
    fit_paths = core.phase_source_paths("fit")
    assert "runs_2025" not in fit_paths and "runs_2023" not in fit_paths
    with pytest.raises(PermissionError):
        core.phase_source_paths("score")
    authorization = {
        "auditor_all_passed": True,
        "development_2024_primary_pass": True,
        "scoring_authorized": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "shadow_tree_read": False,
        "shadow_tree_written": False,
    }
    with pytest.raises(PermissionError):
        core.phase_source_paths("score", guard_token=authorization)
    authorization["core_guard_token"] = True
    assert "runs_2025" in core.phase_source_paths("score", guard_token=authorization)


def test_history_token_formula_and_boundary() -> None:
    observed = core.history_tokens([8, 0], [8, 1], [0, 7])
    np.testing.assert_array_equal(observed, [640, 15])
    assert core.history_tokens([8], [8], [7])[0] == 647
    with pytest.raises(AssertionError):
        core.history_tokens([9], [0], [0])


def test_overlapping_loop_labels_and_terminal_rows_are_retained() -> None:
    anchors = _sequence_anchors([1, 3, 1, 3, 1])
    cycles = core.load_cycles()
    expanded = core.expand_compatible_labels(anchors, cycles)
    first = expanded.loc[expanded["anchor_id"].eq(0)].set_index("cycle_id")
    assert first.loc["cycle_01", "target"] == 1
    assert first.loc["cycle_14", "target"] == 1
    terminal = expanded.loc[expanded["anchor_id"].eq(len(anchors) - 1)]
    assert len(terminal) > 0
    assert terminal["terminal"].all()
    assert terminal["target"].sum() == 0
    assert np.allclose(
        expanded.groupby("anchor_id")["inverse_compatible_weight"].sum(), 1.0
    )


def test_year_predicate_provider_scan_and_no_physical_hash_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "provider"
    path = root / "symbol=AAA" / "timeframe=5m" / "data.parquet"
    path.parent.mkdir(parents=True)
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T14:30:00Z",
                    "2024-01-02T14:35:00Z",
                    "2024-01-02T14:40:00Z",
                    "2025-01-02T14:30:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0, 102.0, 999.0],
            "high": [102.0, 103.0, 104.0, 1000.0],
            "low": [99.0, 100.0, 101.0, 998.0],
            "close": [101.0, 102.0, 103.0, 999.5],
        }
    )
    bars.to_parquet(path, index=False)

    def forbidden_whole_file_hash(_: Path) -> str:
        raise AssertionError("physical shared provider file must not be hashed")

    monkeypatch.setattr(core, "sha256", forbidden_whole_file_hash)
    panel, digest, audit = core.scan_provider_factors(
        ["AAA"],
        root,
        2024,
        expected_file_hashes={"AAA": "pinned-manifest-value"},
        enforce_frozen_2024_placeholder_audit=False,
    )
    assert len(panel) == 3
    assert panel["timestamp"].dt.year.eq(2024).all()
    assert audit["provider_file_hashes"] == {"AAA": "pinned-manifest-value"}
    assert len(digest) == 64
    expected_first = np.log(101.0 / 100.0)
    expected_second = np.log(102.0 / 101.0)
    assert np.isclose(panel.loc[0, "current_bar_log_return"], expected_first)
    assert np.isclose(panel.loc[1, "return_sum_6"], expected_first + expected_second)

    later, _, later_audit = core.scan_provider_factors(
        ["AAA"],
        root,
        2025,
        expected_file_hashes={"AAA": "pinned-manifest-value"},
        enforce_frozen_2024_placeholder_audit=False,
        enforce_irregular_2025_audit=False,
    )
    assert len(later) == 1
    assert later["timestamp"].dt.year.eq(2025).all()
    assert later_audit["provider_file_hashes"] == {"AAA": "pinned-manifest-value"}


def test_contract_pins_canonical_2024_factor_table_hash() -> None:
    contract = core.validate_contract()
    expected = contract["feature_construction"]["provider_scan_and_factor_table"][
        "fit_2024_canonical_retained_factor_table_sha256"
    ]
    assert expected == "f07a1feae8aa4e61092131f659648eaaf39fdb8ce211d3e7f931b56abda1891a"
    assert len(expected) == 64


def test_placeholder_cleanup_discards_only_all_four_null_rows(tmp_path: Path) -> None:
    root = tmp_path / "provider"
    path = root / "symbol=AAA" / "timeframe=5m" / "data.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T14:30:00Z", "2024-01-02T14:35:00Z"], utc=True
            ),
            "open": [100.0, np.nan],
            "high": [101.0, np.nan],
            "low": [99.0, np.nan],
            "close": [100.5, np.nan],
        }
    )
    frame.to_parquet(path, index=False)
    panel, _, audit = core.scan_provider_factors(
        ["AAA"],
        root,
        2024,
        enforce_frozen_2024_placeholder_audit=False,
    )
    assert len(panel) == 1
    assert audit["placeholder_cleanup"]["discarded_rows"] == 1
    assert audit["placeholder_cleanup"]["partial_null_rows"] == 0

    frame.loc[1, "open"] = 100.0
    frame.to_parquet(path, index=False)
    with pytest.raises(AssertionError, match="partial-null"):
        core.scan_provider_factors(
            ["AAA"],
            root,
            2024,
            enforce_frozen_2024_placeholder_audit=False,
        )


def test_merge_recomputes_entry_clock_and_declared_b0_missing_policy() -> None:
    runs = _sequence_anchors([1]).drop(
        columns=["bar_ordinal", *core.NEW5, "entry_minutes", "entry_time_sin", "entry_time_cos", "b0_entry_numeric", "b0_entry_high_stress", "b0_unknown"]
    )
    runs["b0_state_numeric"] = np.nan
    runs["b0_high_stress"] = np.nan
    runs["start_timestamp"] = pd.to_datetime(["2024-07-01T13:33:19Z"], utc=True)
    factors = pd.DataFrame(
        {
            "symbol_norm": ["AAA"],
            "session_date": ["2024-07-01"],
            "timestamp": runs["start_timestamp"],
            "timestamp_ns_utc": runs["start_timestamp"].astype("int64"),
            "bar_ordinal": [1],
            **{column: [0.001] for column in core.NEW5},
        }
    )
    merged = core.merge_entry_factors(runs, factors)
    assert merged.loc[0, "b0_unknown"]
    assert merged.loc[0, "b0_entry_numeric"] == 0.0
    assert merged.loc[0, "b0_entry_high_stress"] == 0.0
    assert np.isclose(merged.loc[0, "entry_minutes"], 3.0 + 19.0 / 60.0)


def test_weighted_reference_contrasts_are_exactly_centered() -> None:
    cycles = core.load_cycles()
    mapping = core.route_mapping(cycles)
    expanded = mapping[["cycle_index", "route_index"]].copy()
    expanded["inverse_compatible_weight"] = np.linspace(0.2, 1.7, len(expanded))
    spec = core.fit_contrast_spec(expanded, mapping)
    cycle, route = core.contrast_blocks(expanded, spec, mapping)
    weights = expanded["inverse_compatible_weight"].to_numpy(float)
    np.testing.assert_allclose(weights @ cycle.toarray(), 0.0, atol=1e-12)
    np.testing.assert_allclose(weights @ route.toarray(), 0.0, atol=1e-12)
    assert cycle.shape[1] == 19
    assert route.shape[1] == 24


def test_design_widths_penalties_and_zero_block_embeddings() -> None:
    cycles = core.load_cycles()
    mapping = core.route_mapping(cycles)
    expanded = mapping[["cycle_index", "route_index", "current_state"]].copy()
    expanded["anchor_id"] = np.arange(len(expanded))
    expanded["history_token"] = np.arange(len(expanded))
    expanded["inverse_compatible_weight"] = 1.0
    spec = core.fit_contrast_spec(expanded, mapping)
    factors = np.linspace(-1.0, 1.0, len(expanded) * 9).reshape(len(expanded), 9)
    token_mask = np.ones(core.TOKEN_WIDTH, dtype=bool)
    designs = core.build_all_designs(expanded, factors, spec, mapping, token_mask)
    assert designs["qpattern"].shape == (44, 44)
    assert designs["qlimited4"].shape == (44, 2812)
    assert designs["qfull9"].shape == (44, 6272)
    assert all(matrix.dtype == np.float64 for matrix in designs.values())
    assert len(core.penalty_multipliers(0)) == 44
    assert len(core.penalty_multipliers(4)) == 2812
    assert len(core.penalty_multipliers(9)) == 6272
    rng = np.random.default_rng(20260711)
    errors = core.embedding_invariants(
        designs,
        np.linspace(-2.0, 2.0, len(expanded)),
        pattern_coefficients=rng.normal(size=44),
        limited_coefficients=rng.normal(size=2812),
    )
    assert max(errors.values()) <= 1e-12


def test_transform_quartiles_path_offsets_and_replay_payload() -> None:
    anchors = _sequence_anchors([1, 3, 1, 3, 1])
    anchors.loc[0, "b0_entry_numeric"] = np.nan
    transform = core.fit_numeric_transform(anchors)
    values = core.transform_numeric(anchors, transform)
    assert values.shape == (5, 9)
    assert np.isfinite(values).all()
    cutpoints = core.fit_quartile_cutpoints(anchors)
    quartiles = core.apply_quartile_cutpoints(anchors, cutpoints)
    assert set(quartiles) == set(core.QUARTILE_COLUMNS)
    assert all(value.min() >= 0 and value.max() <= 3 for value in quartiles.values())

    cycles = core.load_cycles()
    expanded = core.expand_compatible_labels(anchors, cycles)
    offsets = core.score_path_offsets(anchors, expanded, cycles, *_uniform_kernels())
    np.testing.assert_allclose(offsets["qhistory"], offsets["qold_limited_path"])
    assert offsets[["qhistory", "qold_limited_path"]].to_numpy().min() > 0.0

    mapping = core.route_mapping(cycles)
    design_rows = mapping[["cycle_index", "route_index", "current_state"]].copy()
    design_rows["anchor_id"] = np.arange(len(design_rows))
    design_rows["history_token"] = np.arange(len(design_rows))
    design_rows["inverse_compatible_weight"] = 1.0
    design_rows["target"] = 0
    spec = core.fit_contrast_spec(design_rows, mapping)
    design_factors = np.resize(values, (len(design_rows), 9))
    designs = core.build_all_designs(
        design_rows,
        design_factors,
        spec,
        mapping,
        np.ones(core.TOKEN_WIDTH, dtype=bool),
    )
    payload = core.design_replay_inputs(
        designs,
        design_rows["target"],
        np.zeros(len(design_rows)),
        design_rows["anchor_id"],
        design_rows["cycle_index"],
    )
    for name in designs:
        replay = core.replay_design_from_payload(payload, name)
        assert (replay != designs[name]).nnz == 0


def test_offset_ridge_optimizer_uses_fixed_offset_and_converges() -> None:
    rng = np.random.default_rng(7)
    count = 300
    x = rng.normal(size=count)
    design = sparse.csr_matrix(
        np.column_stack(
            [np.ones(count), x, np.zeros((count, core.PATTERN_WIDTH - 2))]
        )
    )
    probability = 1.0 / (1.0 + np.exp(-0.8 * x))
    target = (rng.random(count) < probability).astype(int)
    offset = np.linspace(-0.1, 0.1, count)
    fit = core.fit_offset_ridge(
        design,
        target,
        offset,
        0.001,
        factor_count=0,
    )
    predicted, eta = core.predict_offset_ridge(design, offset, fit)
    assert fit.iterations < 1000
    assert np.isfinite(fit.objective)
    assert predicted.shape == eta.shape == (count,)
    assert np.all((predicted > 0.0) & (predicted < 1.0))


def test_offset_ridge_gradient_matches_central_finite_difference() -> None:
    rng = np.random.default_rng(11)
    rows = 40
    design = sparse.csr_matrix(rng.normal(size=(rows, core.PATTERN_WIDTH)))
    target = rng.integers(0, 2, size=rows)
    offset = rng.normal(scale=0.2, size=rows)
    beta = rng.normal(scale=0.05, size=core.PATTERN_WIDTH)
    penalty = core.penalty_multipliers(0)
    value, gradient = core.offset_ridge_objective_gradient(
        beta, design, target, offset, 0.001, penalty
    )
    assert np.isfinite(value) and np.isfinite(gradient).all()
    epsilon = 1e-6
    for column in (0, 1, 7, core.PATTERN_WIDTH - 1):
        step = np.zeros_like(beta)
        step[column] = epsilon
        high = core.offset_ridge_objective_gradient(
            beta + step, design, target, offset, 0.001, penalty
        )[0]
        low = core.offset_ridge_objective_gradient(
            beta - step, design, target, offset, 0.001, penalty
        )[0]
        numeric = (high - low) / (2.0 * epsilon)
        assert np.isclose(gradient[column], numeric, atol=2e-8, rtol=2e-7)


def test_bundle_round_trip(tmp_path: Path) -> None:
    history, limited = _uniform_kernels()
    transform = core.NumericTransform(
        tuple(core.FULL9), np.arange(9, dtype=float), np.ones(9)
    )
    cycles = core.load_cycles()
    mapping = core.route_mapping(cycles)
    expanded = mapping[["cycle_index", "route_index"]].copy()
    expanded["inverse_compatible_weight"] = 1.0
    contrasts = core.fit_contrast_spec(expanded, mapping)

    def fit(width: int, factors: int) -> core.ResidualFit:
        return core.ResidualFit(
            np.linspace(-0.1, 0.1, width), 0.001, factors, 0.2, 1e-9, 3, "ok"
        )

    bundle = core.ModelBundle(
        history,
        limited,
        transform,
        contrasts,
        np.ones(core.TOKEN_WIDTH, dtype=bool),
        np.zeros((len(core.QUARTILE_COLUMNS), 3)),
        fit(core.PATTERN_WIDTH, 0),
        fit(core.LIMITED_WIDTH, 4),
        fit(core.FULL_WIDTH, 9),
    )
    path = tmp_path / "bundle.npz"
    core.save_bundle(path, bundle)
    loaded = core.load_bundle(path)
    np.testing.assert_array_equal(loaded.qfull9.coefficients, bundle.qfull9.coefficients)
    np.testing.assert_array_equal(loaded.token_mask, bundle.token_mask)
    assert loaded.numeric_transform.columns == tuple(core.FULL9)


def test_fold_subset_preserves_source_anchor_identity() -> None:
    anchors = _sequence_anchors([1, 3, 1, 3, 1])
    cycles = core.load_cycles()
    expanded = core.expand_compatible_labels(anchors, cycles)
    selected_anchors, selected = core.subset_anchors_expanded(
        anchors, expanded, np.asarray([False, True, True, False, True])
    )
    np.testing.assert_array_equal(selected_anchors["anchor_id"], [0, 1, 2])
    np.testing.assert_array_equal(selected_anchors["source_anchor_id"], [1, 2, 4])
    assert set(selected["source_anchor_id"]) == {1, 2, 4}
