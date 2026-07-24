from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest

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
    tail_overlap_metrics,
    validate_no_excluded_features,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v01 import (
    MAXIMUM_ADDITIONAL_BYTES,
    MAXIMUM_ADDITIONAL_RECORDS,
    MAXIMUM_CUMULATIVE_RECORDS,
    SAFETY_FLAGS,
    ResumeResourceLimitError,
    add_movement_outcomes_with_optional_30m,
    assert_v01_safety_flags,
    authorize_outcome_access,
    coverage_preflight,
    identify_interrupted_request,
    inventory_complete_receipts,
    movement_timing_metrics_with_optional_30m,
    remaining_resume_requests,
    request_identity,
    validate_additional_resource_usage,
)


def _plan_row(
    *,
    symbol: str = "AAPL",
    holdout_session: str = "2025-09-02",
    required_options_date: str = "2025-08-29",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "holdout_session": holdout_session,
        "required_options_date": required_options_date,
        "previous_close": 200.0,
        "strike_from": 140.0,
        "strike_to": 260.0,
        "expiration_from": "2025-09-05",
        "expiration_to": "2025-10-13",
    }


def _provider_record(
    *,
    observed: str,
    symbol: str = "AAPL",
    contract: str = "AAPL250919C00200000",
) -> dict[str, object]:
    return {
        "id": f"{contract}-{observed}",
        "type": "options-eod",
        "attributes": {
            "underlying_symbol": symbol,
            "contract": contract,
            "strike": 200.0,
            "exp_date": "2025-09-19",
            "type": "call",
            "bid_date": f"{observed}T15:59:00-04:00",
            "ask_date": f"{observed}T15:59:00-04:00",
            "tradetime": f"{observed}T15:59:00-04:00",
            "bid": 1.0,
            "ask": 1.2,
            "midpoint": 1.1,
            "open_interest": 100,
            "volatility": 0.5,
            "dte": (date(2025, 9, 19) - date.fromisoformat(observed)).days,
        },
    }


def _write_complete_receipt(
    cache_root: Path,
    planned: dict[str, object],
    records: list[dict[str, object]],
) -> tuple[Path, Path]:
    payload = {
        "meta": {"offset": 0, "limit": 1000, "total": len(records)},
        "data": records,
        "links": {"next": None},
    }
    content = json.dumps(payload, sort_keys=True).encode()
    response_hash = hashlib.sha256(content).hexdigest()
    raw_path = cache_root / "raw" / f"{response_hash}.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(content)
    receipt_id = request_identity(planned)
    receipt_path = cache_root / "manifests" / "completed" / f"{receipt_id}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "manifest_rows": [
            {
                "request_id": receipt_id,
                "underlying_symbol": planned["symbol"],
                "trade_date_from": planned["required_options_date"],
                "trade_date_to": planned["required_options_date"],
                "strike_from": planned["strike_from"],
                "strike_to": planned["strike_to"],
                "expiration_from": planned["expiration_from"],
                "expiration_to": planned["expiration_to"],
                "offset": 0,
                "limit": 1000,
                "response_status": 200,
                "record_count": len(records),
                "response_hash": response_hash,
                "attempts": 1,
                "started_at": "2026-07-24T00:00:00+00:00",
                "completed_at": "2026-07-24T00:00:01+00:00",
                "cache_path": str(raw_path),
                "superseded_by_split": False,
            }
        ]
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, raw_path


def test_v01_safety_contract_is_strict_resume() -> None:
    assert_v01_safety_flags(SAFETY_FLAGS)
    assert SAFETY_FLAGS["strict_resume"] is True
    assert SAFETY_FLAGS["completed_receipts_reused"] is True
    assert SAFETY_FLAGS["complete_requests_redownloaded"] is False
    assert SAFETY_FLAGS["partial_cohort_model_allowed"] is False
    assert SAFETY_FLAGS["holdout_outcomes_opened_only_after_coverage_preflight"] is True


def test_complete_receipt_reuse_and_complete_request_not_redownloaded(
    tmp_path: Path,
) -> None:
    first = _plan_row()
    second = _plan_row(
        symbol="MSFT",
        holdout_session="2025-09-03",
        required_options_date="2025-09-02",
    )
    receipt, _ = _write_complete_receipt(
        tmp_path,
        first,
        [_provider_record(observed="2025-08-29")],
    )
    inventory = inventory_complete_receipts(
        [first, second],
        cache_roots=[tmp_path],
        canonical_cache_path=tmp_path / "canonical/exact_holdout_options.parquet",
    )
    assert receipt.is_file()
    assert inventory.complete_receipts_found == 1
    assert inventory.complete_receipts_reused == 1
    assert inventory.corrupt_receipts == 0
    assert inventory.audit["pagination_complete"].tolist() == [True]
    assert inventory.audit["exact_date_record_count"].tolist() == [1]
    remaining = remaining_resume_requests(
        [first, second],
        verified_request_ids=inventory.verified_request_ids,
    )
    assert [row["symbol"] for row in remaining] == ["MSFT"]


def test_corrupt_receipt_is_audited_and_returns_to_missing_scope(tmp_path: Path) -> None:
    planned = _plan_row()
    receipt, raw = _write_complete_receipt(
        tmp_path,
        planned,
        [_provider_record(observed="2025-08-29")],
    )
    raw.write_bytes(b"corrupt")
    inventory = inventory_complete_receipts(
        [planned],
        cache_roots=[tmp_path],
        canonical_cache_path=tmp_path / "canonical/exact_holdout_options.parquet",
    )
    assert receipt.is_file()
    assert inventory.complete_receipts_found == 1
    assert inventory.complete_receipts_reused == 0
    assert inventory.corrupt_receipts == 1
    assert len(remaining_resume_requests([planned], inventory.verified_request_ids)) == 1


def test_interrupted_page_is_excluded_and_maps_to_one_missing_request(
    tmp_path: Path,
) -> None:
    interrupted = _plan_row(
        symbol="SMCI",
        holdout_session="2025-09-09",
        required_options_date="2025-09-08",
    )
    unrelated = _plan_row(
        symbol="SOFI",
        holdout_session="2025-09-09",
        required_options_date="2025-09-08",
    )
    query = urlencode(
        {
            "filter[underlying_symbol]": interrupted["symbol"],
            "filter[tradetime_from]": interrupted["required_options_date"],
            "filter[tradetime_to]": interrupted["required_options_date"],
            "filter[strike_from]": interrupted["strike_from"],
            "filter[strike_to]": interrupted["strike_to"],
            "filter[exp_date_from]": interrupted["expiration_from"],
            "filter[exp_date_to]": interrupted["expiration_to"],
            "page[offset]": 198,
            "page[limit]": 198,
        }
    )
    payload = {
        "meta": {"offset": 0, "limit": 198},
        "data": [
            _provider_record(
                observed="2025-09-08",
                symbol="SMCI",
                contract="SMCI250919C00200000",
            )
        ],
        "links": {"next": f"https://example.invalid/options?{query}"},
    }
    orphan = tmp_path / "raw" / "orphan.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text(json.dumps(payload), encoding="utf-8")
    identified = identify_interrupted_request(orphan, [unrelated, interrupted])
    assert identified.request_id == request_identity(interrupted)
    assert identified.incomplete_page_records == 1
    assert identified.incomplete_page_admitted is False
    assert identified.resume_method == "redownload_logical_request_from_beginning"


def test_resume_order_is_deterministic_stock_then_date() -> None:
    rows = [
        _plan_row(
            symbol="WULF",
            holdout_session="2025-09-03",
            required_options_date="2025-09-02",
        ),
        _plan_row(
            symbol="SMCI",
            holdout_session="2025-09-03",
            required_options_date="2025-09-02",
        ),
        _plan_row(
            symbol="SMCI",
            holdout_session="2025-09-02",
            required_options_date="2025-08-29",
        ),
    ]
    ordered = remaining_resume_requests(rows, verified_request_ids=frozenset())
    assert [(row["symbol"], row["required_options_date"]) for row in ordered] == [
        ("SMCI", "2025-08-29"),
        ("SMCI", "2025-09-02"),
        ("WULF", "2025-09-02"),
    ]


def test_exact_date_filter_rejects_extra_and_2026_dates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2025-09-08", "2025-09-05", "2026-01-02"],
            "contract_id": ["exact", "extra", "protected"],
        }
    )
    retained, audit = exact_date_option_records(frame, required_date=date(2025, 9, 8))
    assert retained["contract_id"].tolist() == ["exact"]
    assert audit["extra_date_records_rejected"] == 2
    assert audit["protected_date_records_rejected"] == 1


def test_resume_resource_limits_fail_closed() -> None:
    validate_additional_resource_usage(
        provider_records=MAXIMUM_ADDITIONAL_RECORDS,
        raw_bytes=MAXIMUM_ADDITIONAL_BYTES,
        cumulative_records=MAXIMUM_CUMULATIVE_RECORDS,
    )
    with pytest.raises(ResumeResourceLimitError, match="blocked_resume_resource_limit"):
        validate_additional_resource_usage(
            provider_records=MAXIMUM_ADDITIONAL_RECORDS + 1,
            raw_bytes=1,
            cumulative_records=MAXIMUM_CUMULATIVE_RECORDS,
        )


def _coverage_frame(*, stocks: int = 20, sessions: int = 85) -> pd.DataFrame:
    session_values = pd.bdate_range("2025-09-01", periods=sessions)
    rows: list[dict[str, Any]] = []
    for stock_index in range(stocks):
        for session in session_values:
            for checkpoint in (6, 8, 10):
                rows.append(
                    {
                        "symbol": f"S{stock_index:02d}",
                        "session": session.date().isoformat(),
                        "checkpoint": checkpoint,
                        "row_weight": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_coverage_preflight_precedes_outcomes_and_prohibits_partial_cohort() -> None:
    full = _coverage_frame()
    pairs = full[["symbol", "session"]].drop_duplicates()
    preflight = coverage_preflight(
        full,
        pairs,
        planned_stock_sessions=1_700,
        planned_session_count=85,
        planned_stock_month_cells=80,
    )
    assert preflight["passed"] is True
    assert preflight["total_holdout_stock_sessions"] == 1_700
    assert preflight["planned_session_count"] == 85
    assert preflight["pair_selected_stock_sessions"] == 1_700
    with pytest.raises(ValueError, match="freeze"):
        authorize_outcome_access(
            preflight,
            {"frozen": False, "holdout_outcomes_read_before_freeze": False},
        )
    authorize_outcome_access(
        preflight,
        {"frozen": True, "holdout_outcomes_read_before_freeze": False},
    )

    partial = _coverage_frame(stocks=14)
    partial_pairs = partial[["symbol", "session"]].drop_duplicates()
    failed = coverage_preflight(
        partial,
        partial_pairs,
        planned_stock_sessions=1_700,
        planned_session_count=85,
        planned_stock_month_cells=80,
    )
    assert failed["passed"] is False
    with pytest.raises(ValueError, match="coverage"):
        authorize_outcome_access(
            failed,
            {"frozen": True, "holdout_outcomes_read_before_freeze": False},
        )


def test_coverage_preflight_rejects_outcome_columns() -> None:
    frame = _coverage_frame()
    frame["absolute_log_return_15m"] = 0.01
    pairs = frame[["symbol", "session"]].drop_duplicates()
    with pytest.raises(ValueError, match="outcome"):
        coverage_preflight(
            frame,
            pairs,
            planned_stock_sessions=1_700,
            planned_session_count=85,
            planned_stock_month_cells=80,
        )


def test_m0_m1_features_and_2024_only_fitting_remain_frozen() -> None:
    row = {feature: float(index + 1) for index, feature in enumerate((*GROUP_O, *GROUP_I))}
    row["checkpoint"] = 6
    for checkpoint in range(6, 35, 2):
        row[f"checkpoint_{checkpoint}"] = float(checkpoint == 6)
    frame = pd.DataFrame([row])
    assert tuple(build_group_o(frame)) == GROUP_O
    assert tuple(build_group_i(frame)) == GROUP_I
    validate_no_excluded_features([*GROUP_O, *GROUP_I])
    with pytest.raises(ValueError, match="excluded"):
        validate_no_excluded_features([*GROUP_O, "daily_compression"])
    assert_2024_only_fitting(pd.Series(pd.to_datetime(["2024-01-02", "2024-12-31"])))
    with pytest.raises(ValueError, match="2024"):
        assert_2024_only_fitting(pd.Series(pd.to_datetime(["2025-01-02"])))


def test_weighted_threshold_freeze_and_tail_membership() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.9])
    weights = np.asarray([1.0, 8.0, 1.0])
    thresholds = freeze_tail_thresholds(
        m0_probabilities=probabilities,
        m1_probabilities=probabilities + 0.01,
        weights=weights,
    )
    assert thresholds["M0_top_5_percent_threshold"] == pytest.approx(0.9)
    assert thresholds["M1_top_5_percent_threshold"] == pytest.approx(0.91)
    assert frozen_tail_membership(np.asarray([0.91, 0.909]), 0.91).tolist() == [
        True,
        False,
    ]


def test_all_frozen_movement_horizons_and_iv_scaling() -> None:
    result = add_movement_outcomes(
        pd.DataFrame(
            {
                "entry_price": [100.0],
                "close_5m": [101.0],
                "close_10m": [102.0],
                "close_15m": [103.0],
                "close_30m": [106.0],
                "atm_iv": [0.5],
            }
        )
    )
    for horizon, close in ((5, 101.0), (10, 102.0), (15, 103.0), (30, 106.0)):
        movement = abs(math.log(close / 100.0))
        sigma = 0.5 * math.sqrt(horizon / (252 * 390))
        expected = sigma * math.sqrt(2 / math.pi)
        assert result.loc[0, f"absolute_log_return_{horizon}m"] == pytest.approx(movement)
        assert result.loc[0, f"iv_sigma_{horizon}m"] == pytest.approx(sigma)
        assert result.loc[0, f"iv_absolute_residual_{horizon}m"] == pytest.approx(
            movement - expected
        )


def test_thirty_minute_movement_is_optional_without_dropping_binding_row() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "session": ["2025-09-02", "2025-09-02"],
            "row_weight": [1.0, 1.0],
            "entry_price": [100.0, 100.0],
            "close_5m": [101.0, 101.0],
            "close_10m": [102.0, 102.0],
            "close_15m": [103.0, 103.0],
            "close_30m": [106.0, math.nan],
            "atm_iv": [0.5, 0.5],
        }
    )
    result = add_movement_outcomes_with_optional_30m(frame)
    assert len(result) == 2
    assert result["absolute_log_return_15m"].notna().all()
    assert result["absolute_log_return_30m"].notna().sum() == 1
    timing = movement_timing_metrics_with_optional_30m(result)
    assert timing.set_index("horizon_minutes").loc[15, "rows_available"] == 2
    assert timing.set_index("horizon_minutes").loc[30, "rows_available"] == 1


def test_tail_overlap_session_bootstrap_and_h0_bundle_null() -> None:
    overlap = tail_overlap_metrics(
        np.asarray([True, True, False]),
        np.asarray([False, True, True]),
    )
    assert overlap["intersection_rows"] == 1
    assert overlap["union_rows"] == 3
    sessions = pd.Series(["a", "a", "b", "b"])
    draws = fixed_session_bootstrap_multiplicities(sessions, draws=10, seed=17)
    assert len(draws) == 10
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)

    frame = pd.DataFrame(
        {
            "period": ["development"] * 3,
            "session": ["2024-01-02"] * 3,
            "checkpoint": [6] * 3,
            "symbol": ["A", "B", "C"],
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
    assert permuted["outcome"].equals(frame["outcome"])
    assert set(map(tuple, permuted[[GROUP_I[0], GROUP_I[1]]].to_numpy())) == set(
        map(tuple, frame[[GROUP_I[0], GROUP_I[1]]].to_numpy())
    )


def test_decision_logic_remains_exactly_frozen() -> None:
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
    assert (
        decide_experiment(model=model, tail=tail)["overall_decision"]
        == "minimal_intraday_h0_iv_excess_tail_validated"
    )
