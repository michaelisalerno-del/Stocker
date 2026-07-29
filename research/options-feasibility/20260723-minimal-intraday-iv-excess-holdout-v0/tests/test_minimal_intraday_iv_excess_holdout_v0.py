from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.broad_conflict_options_iv_screen_v0 import select_primary_atm_pair
from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
    GROUP_I,
    GROUP_O,
    ModelGateInputs,
    TailGateInputs,
    add_movement_outcomes,
    assert_2024_only_fitting,
    build_group_i,
    build_group_o,
    decide_experiment,
    exact_date_option_records,
    fixed_session_bootstrap_multiplicities,
    freeze_tail_thresholds,
    frozen_tail_membership,
    intraday_h0_bundle_null,
    minimal_feature_manifest,
    tail_overlap_metrics,
    validate_exact_previous_session_options,
    validate_holdout_dates,
    validate_no_excluded_features,
    weighted_quantile,
)


def _option_row(
    *,
    option_type: str,
    strike: float,
    expiry: str = "2025-09-19",
    dte: int = 16,
    contract_id: str | None = None,
    open_interest: int = 100,
) -> dict[str, object]:
    return {
        "option_type": option_type,
        "expiration_date": expiry,
        "dte": dte,
        "strike": strike,
        "contract_id": contract_id or f"{option_type}-{strike}",
        "bid": 1.0,
        "ask": 1.2,
        "midpoint": 1.1,
        "implied_volatility": 0.50,
        "open_interest": open_interest,
        "delta": 0.50 if option_type == "call" else -0.50,
        "gamma": 0.01,
    }


def test_holdout_date_isolation_and_2026_rejection() -> None:
    valid = pd.Series(pd.to_datetime(["2025-09-02", "2025-12-31"]))
    validate_holdout_dates(valid)
    with pytest.raises(ValueError, match="holdout"):
        validate_holdout_dates(pd.Series(pd.to_datetime(["2025-08-29"])))
    with pytest.raises(ValueError, match="protected"):
        validate_holdout_dates(pd.Series(pd.to_datetime(["2026-01-02"])))


def test_exact_previous_session_options_join_and_same_day_rejection() -> None:
    validate_exact_previous_session_options(
        signal_date=date(2025, 9, 2),
        required_options_date=date(2025, 8, 29),
        actual_options_date=date(2025, 8, 29),
    )
    with pytest.raises(ValueError, match="same-day"):
        validate_exact_previous_session_options(
            signal_date=date(2025, 9, 2),
            required_options_date=date(2025, 8, 29),
            actual_options_date=date(2025, 9, 2),
        )
    with pytest.raises(ValueError, match="stale"):
        validate_exact_previous_session_options(
            signal_date=date(2025, 9, 2),
            required_options_date=date(2025, 8, 29),
            actual_options_date=date(2025, 8, 28),
        )


def test_exact_date_filter_rejects_extra_and_protected_observations() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2025-08-29", "2025-08-28", "2026-01-02"],
            "contract_id": ["kept", "extra", "protected"],
        }
    )
    retained, audit = exact_date_option_records(frame, required_date=date(2025, 8, 29))
    assert retained["contract_id"].tolist() == ["kept"]
    assert audit == {
        "records_returned": 3,
        "exact_date_records_retained": 1,
        "extra_date_records_rejected": 2,
        "protected_date_records_rejected": 1,
    }


def test_blocked_download_audits_incomplete_page_and_all_plan_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from download_holdout_options import PlannedRequest, summarize_completed_cache

    rows = [
        PlannedRequest(
            symbol="AAPL",
            holdout_session="2025-09-02",
            required_options_date="2025-08-29",
            previous_close=200.0,
            strike_from=140.0,
            strike_to=260.0,
            expiration_from="2025-09-05",
            expiration_to="2025-10-13",
        ),
        PlannedRequest(
            symbol="AAPL",
            holdout_session="2025-10-01",
            required_options_date="2025-09-30",
            previous_close=200.0,
            strike_from=140.0,
            strike_to=260.0,
            expiration_from="2025-10-07",
            expiration_to="2025-11-14",
        ),
    ]
    payload = {
        "meta": {"offset": 0, "limit": 3, "total": 3},
        "data": [
            {
                "id": "contract-2025-08-29",
                "type": "options-eod",
                "attributes": {"tradetime": "2025-08-29"},
            },
            {
                "id": "contract-2025-08-28",
                "type": "options-eod",
                "attributes": {"tradetime": "2025-08-28"},
            },
            {
                "id": "contract-2026-01-02",
                "type": "options-eod",
                "attributes": {"tradetime": "2026-01-02"},
            },
        ],
        "links": {"next": None},
    }
    raw_path = tmp_path / "raw" / "partial.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, coverage = summarize_completed_cache(
        rows,
        cache_root=tmp_path,
        decision="blocked_quick_resource_limit",
        detail="test ceiling",
    )

    assert manifest["records_returned"] == 3
    assert manifest["exact_date_records_returned"] == 1
    assert manifest["exact_date_records_retained"] == 0
    assert manifest["exact_date_records_rejected_incomplete"] == 1
    assert manifest["extra_date_records_rejected"] == 2
    assert manifest["protected_date_records_rejected"] == 1
    assert manifest["bytes_downloaded"] == raw_path.stat().st_size
    assert len(coverage) == 2
    assert coverage["requests_planned"].sum() == 2
    assert coverage["requests_completed"].sum() == 0
    assert coverage["selected_pair_sessions"].isna().all()
    assert coverage["pair_coverage_status"].eq("blocked_before_pair_selection").all()


def test_front_pair_reconstruction_uses_nearest_expiry_and_common_atm_strike() -> None:
    chain = pd.DataFrame(
        [
            _option_row(option_type="call", strike=99.0),
            _option_row(option_type="put", strike=99.0),
            _option_row(option_type="call", strike=101.0),
            _option_row(option_type="put", strike=101.0),
            _option_row(option_type="call", strike=100.0, expiry="2025-10-17", dte=44),
            _option_row(option_type="put", strike=100.0, expiry="2025-10-17", dte=44),
        ]
    )
    selected = select_primary_atm_pair(chain, previous_close=100.2)
    assert selected.available
    assert selected.expiration_date == date(2025, 9, 19)
    assert selected.strike == 101.0


def test_front_pair_does_not_substitute_expiry_after_quality_failure() -> None:
    chain = pd.DataFrame(
        [
            _option_row(option_type="call", strike=100.0, open_interest=1),
            _option_row(option_type="put", strike=100.0, open_interest=1),
            _option_row(option_type="call", strike=100.0, expiry="2025-10-17", dte=44),
            _option_row(option_type="put", strike=100.0, expiry="2025-10-17", dte=44),
        ]
    )
    selected = select_primary_atm_pair(chain, previous_close=100.0)
    assert not selected.available
    assert selected.expiration_date == date(2025, 9, 19)
    assert selected.reason == "selected_pair_open_interest_below_10"


def test_group_o_and_group_i_construction_and_excluded_group_absence() -> None:
    row: dict[str, object] = {
        feature: float(index + 1) for index, feature in enumerate(GROUP_O + GROUP_I)
    }
    row["checkpoint"] = 6
    for checkpoint in range(6, 35, 2):
        row[f"checkpoint_{checkpoint}"] = float(checkpoint == 6)
    frame = pd.DataFrame([row])
    group_o = build_group_o(frame)
    group_i = build_group_i(frame)
    assert tuple(group_o.columns) == GROUP_O
    assert tuple(group_i.columns) == GROUP_I
    manifest = minimal_feature_manifest()
    assert manifest["models"]["M0"]["numeric_features"] == list(GROUP_O)
    assert manifest["models"]["M1"]["numeric_features"] == [*GROUP_O, *GROUP_I]
    validate_no_excluded_features([*GROUP_O, *GROUP_I])
    with pytest.raises(ValueError, match="excluded"):
        validate_no_excluded_features([*GROUP_O, "daily_compression"])


def test_2024_only_fitting() -> None:
    assert_2024_only_fitting(pd.Series(pd.to_datetime(["2024-01-02", "2024-12-31"])))
    with pytest.raises(ValueError, match="2024"):
        assert_2024_only_fitting(pd.Series(pd.to_datetime(["2025-01-02"])))


def test_weighted_top_5_percent_threshold_and_freezing() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.9], dtype=float)
    weights = np.asarray([1.0, 8.0, 1.0], dtype=float)
    assert weighted_quantile(probabilities, weights, 0.95) == pytest.approx(0.9)
    frozen = freeze_tail_thresholds(
        m0_probabilities=probabilities,
        m1_probabilities=probabilities + 0.01,
        weights=weights,
    )
    assert frozen["M0_top_5_percent_threshold"] == pytest.approx(0.9)
    assert frozen["M1_top_5_percent_threshold"] == pytest.approx(0.91)
    assert frozen_tail_membership(np.asarray([0.90, 0.899]), 0.90).tolist() == [True, False]


def test_movement_and_iv_scaling_at_all_frozen_horizons() -> None:
    frame = pd.DataFrame(
        {
            "entry_price": [100.0],
            "close_5m": [101.0],
            "close_10m": [102.0],
            "close_15m": [103.0],
            "close_30m": [106.0],
            "atm_iv": [0.50],
        }
    )
    result = add_movement_outcomes(frame)
    for horizon, close in ((5, 101.0), (10, 102.0), (15, 103.0), (30, 106.0)):
        movement = abs(math.log(close / 100.0))
        sigma = 0.50 * math.sqrt(horizon / (252 * 390))
        expected = sigma * math.sqrt(2 / math.pi)
        assert result.loc[0, f"absolute_log_return_{horizon}m"] == pytest.approx(movement)
        assert result.loc[0, f"iv_sigma_{horizon}m"] == pytest.approx(sigma)
        assert result.loc[0, f"iv_expected_absolute_{horizon}m"] == pytest.approx(expected)
        assert result.loc[0, f"iv_absolute_residual_{horizon}m"] == pytest.approx(
            movement - expected
        )
    assert result.loc[0, "movement_exceeds_prior_close_iv_15m"] == 1


def test_tail_overlap() -> None:
    metrics = tail_overlap_metrics(
        np.asarray([True, True, False, False]),
        np.asarray([False, True, True, False]),
    )
    assert metrics == {
        "intersection_rows": 1,
        "union_rows": 3,
        "jaccard_overlap": pytest.approx(1 / 3),
        "M1_only_rows": 1,
        "M0_only_rows": 1,
    }


def test_session_bootstrap_resamples_whole_sessions() -> None:
    sessions = pd.Series(["a", "a", "b", "b"])
    draws = fixed_session_bootstrap_multiplicities(sessions, draws=10, seed=17)
    assert len(draws) == 10
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)
    with pytest.raises(ValueError, match="exactly 10"):
        fixed_session_bootstrap_multiplicities(sessions, draws=9, seed=17)


def test_h0_bundle_null_preserves_complete_bundles_within_slate() -> None:
    frame = pd.DataFrame(
        {
            "period": ["development"] * 3,
            "session": ["2024-01-02"] * 3,
            "checkpoint": [6] * 3,
            "symbol": ["A", "B", "C"],
            "o": [10, 20, 30],
            "outcome": [0, 1, 0],
            GROUP_I[0]: [1.0, 2.0, 3.0],
            GROUP_I[1]: [101.0, 102.0, 103.0],
        }
    )
    permuted = intraday_h0_bundle_null(
        frame,
        group_i_columns=(GROUP_I[0], GROUP_I[1]),
        seed=7,
    )
    assert permuted[["o", "outcome"]].equals(frame[["o", "outcome"]])
    original_bundles = set(map(tuple, frame[[GROUP_I[0], GROUP_I[1]]].to_numpy()))
    permuted_bundles = set(map(tuple, permuted[[GROUP_I[0], GROUP_I[1]]].to_numpy()))
    assert permuted_bundles == original_bundles


def test_decision_logic_can_validate_both_binding_questions() -> None:
    model = ModelGateInputs(
        log_loss_improvement=0.01,
        brier_improvement=0.01,
        auc_improvement=0.0,
        average_precision_improvement=0.01,
        bootstrap_80_log_loss_lower=0.0,
        bootstrap_80_brier_lower=0.0,
        bootstrap_80_average_precision_lower=0.0,
        positive_log_loss_months=3,
        materially_adverse_checkpoint_groups=0,
        real_exceeds_all_nulls=True,
        support_passed=True,
    )
    tail = TailGateInputs(
        mean_iv_residual=0.001,
        median_iv_residual=0.0001,
        exceed_iv_rate=0.51,
        bootstrap_80_mean_lower=0.0,
        bootstrap_80_median_lower=0.0,
        positive_mean_months=3,
        positive_median_months=3,
        m1_minus_m0_mean=0.0001,
        bootstrap_80_difference_lower=0.0,
        concentration_passed=True,
        support_passed=True,
    )
    decision = decide_experiment(model=model, tail=tail)
    assert decision["overall_decision"] == "minimal_intraday_h0_iv_excess_tail_validated"
    assert decision["minimal_model_status"] == "supported"
    assert decision["frozen_top_5pct_status"] == "supported"


def test_decision_logic_keeps_positive_tail_separate_from_model_increment() -> None:
    model = ModelGateInputs(
        log_loss_improvement=-0.01,
        brier_improvement=-0.01,
        auc_improvement=-0.01,
        average_precision_improvement=-0.01,
        bootstrap_80_log_loss_lower=-0.01,
        bootstrap_80_brier_lower=-0.01,
        bootstrap_80_average_precision_lower=-0.01,
        positive_log_loss_months=0,
        materially_adverse_checkpoint_groups=3,
        real_exceeds_all_nulls=False,
        support_passed=True,
    )
    tail = TailGateInputs(
        mean_iv_residual=0.001,
        median_iv_residual=0.0001,
        exceed_iv_rate=0.51,
        bootstrap_80_mean_lower=0.0,
        bootstrap_80_median_lower=0.0,
        positive_mean_months=3,
        positive_median_months=3,
        m1_minus_m0_mean=0.0001,
        bootstrap_80_difference_lower=0.0,
        concentration_passed=True,
        support_passed=True,
    )
    assert (
        decide_experiment(model=model, tail=tail)["overall_decision"]
        == "positive_frozen_tail_without_model_increment"
    )
