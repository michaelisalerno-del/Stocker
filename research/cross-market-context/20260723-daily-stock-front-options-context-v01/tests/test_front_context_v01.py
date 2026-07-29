from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from stocker_research.daily_soft_regimes_v0 import (
    FrozenDimensionParameters,
    RobustValueScale,
)
from stocker_research.daily_stock_front_options_context_v01 import (
    FRONT_MISMATCH_FEATURES,
    MeanScale,
    add_front_mismatch_features,
    branch_availability,
)
from stocker_research.front_options_soft_regimes_v01 import (
    FRONT_OPTIONS_DIMENSIONS,
    apply_front_options_dimensions,
)


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "run_screen_v01.py"
    specification = importlib.util.spec_from_file_location("front_context_runner_test", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_preflight() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "back_expiry_preflight.py"
    specification = importlib.util.spec_from_file_location("front_context_preflight_test", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_seven_front_options_dimensions_follow_the_frozen_equations() -> None:
    scales = {
        name: RobustValueScale(center=0.0, scale=1.0)
        for name in (
            "atm_iv",
            "straddle_mid_pct",
            "call_put_iv_gap",
            "skew_25d",
            "combined_relative_spread",
            "iv_minus_realised_20d",
            "near_spot_oi_concentration",
            "call_put_oi_imbalance",
            "abs_call_put_iv_gap",
            "abs_skew_25d",
        )
    }
    parameters = FrozenDimensionParameters(
        kind="front_options",
        scales=scales,
        imputation_medians={
            "atm_iv": 0.0,
            "straddle_mid_pct": 0.0,
            "call_put_iv_gap": 0.0,
            "skew_25d": 0.0,
            "combined_relative_spread": 0.0,
            "iv_minus_realised_20d": 0.0,
            "near_spot_oi_concentration": 0.0,
            "call_put_oi_imbalance": 0.0,
        },
    )
    raw = pd.DataFrame(
        {
            "atm_iv": [3.0],
            "straddle_mid_pct": [6.0],
            "call_put_iv_gap": [-2.0],
            "skew_25d": [4.0],
            "combined_relative_spread": [5.0],
            "iv_minus_realised_20d": [9.0],
            "near_spot_oi_concentration": [7.0],
            "call_put_oi_imbalance": [-8.0],
        }
    )

    result = apply_front_options_dimensions(raw, parameters).iloc[0]

    assert tuple(FRONT_OPTIONS_DIMENSIONS) == (
        "front_options_implied_tension",
        "front_options_premium_richness",
        "front_options_downside_asymmetry",
        "front_options_liquidity_stress",
        "front_options_positioning_concentration",
        "front_options_directional_positioning",
        "front_options_surface_disagreement",
    )
    assert result["front_options_implied_tension"] == pytest.approx(6.0)
    assert result["front_options_premium_richness"] == pytest.approx(7.5)
    assert result["front_options_downside_asymmetry"] == pytest.approx(3.0)
    assert result["front_options_liquidity_stress"] == pytest.approx(5.0)
    assert result["front_options_positioning_concentration"] == pytest.approx(7.0)
    assert result["front_options_directional_positioning"] == pytest.approx(-8.0)
    assert result["front_options_surface_disagreement"] == pytest.approx(11.0 / 3.0)


def test_five_front_only_mismatch_features_follow_the_frozen_equations() -> None:
    frame = pd.DataFrame(
        {
            "daily_compression": [3.0],
            "daily_volatility_acceleration": [5.0],
            "front_options_implied_tension": [1.0],
            "prefix_family_entropy": [4.0],
            "front_options_premium_richness": [2.0],
            "signed_pressure": [-2.0],
            "front_options_directional_positioning": [3.0],
            "BROAD_CONFLICT": [1],
        }
    )
    scales = {
        column: MeanScale(mean=0.0, scale=1.0)
        for column in (
            "daily_compression",
            "daily_volatility_acceleration",
            "front_options_implied_tension",
            "prefix_family_entropy",
            "front_options_premium_richness",
            "signed_pressure",
            "front_options_directional_positioning",
        )
    }

    result = add_front_mismatch_features(frame, scales).iloc[0]

    assert tuple(FRONT_MISMATCH_FEATURES) == (
        "mismatch_compression_vs_front_iv",
        "mismatch_daily_volatility_vs_front_iv",
        "mismatch_route_vs_front_premium",
        "mismatch_direction_agreement",
        "mismatch_complacent_broad_conflict",
    )
    assert result["mismatch_compression_vs_front_iv"] == pytest.approx(2.0)
    assert result["mismatch_daily_volatility_vs_front_iv"] == pytest.approx(4.0)
    assert result["mismatch_route_vs_front_premium"] == pytest.approx(2.0)
    assert result["mismatch_direction_agreement"] == pytest.approx(-6.0)
    assert result["mismatch_complacent_broad_conflict"] == pytest.approx(-1.0)


def test_daily_stock_branch_remains_available_without_front_options() -> None:
    availability = branch_availability(
        structural_ready=True,
        daily_stock_ready=True,
        front_options_ready=False,
    )

    assert availability == {
        "branch_a": True,
        "branch_b": False,
        "branch_c": False,
    }


def test_a_b_and_c_feature_surfaces_are_strictly_nested() -> None:
    runner = load_runner()

    assert set(runner.A1_FEATURES) - set(runner.A0_FEATURES) == set(runner.STOCK_CONTEXT_FEATURES)
    assert runner.B0_FEATURES == runner.A1_FEATURES
    assert set(runner.B1_FEATURES) - set(runner.B0_FEATURES) == set(
        (*runner.FRONT_CONTEXT_FEATURES, *FRONT_MISMATCH_FEATURES)
    )
    assert set(runner.C1_FEATURES) - set(runner.C0_FEATURES) == set(
        (
            *runner.STOCK_CONTEXT_FEATURES,
            *runner.H0_NON_CLOCK_FEATURES,
            *runner.ROUTE_FEATURES,
            *FRONT_MISMATCH_FEATURES,
        )
    )
    assert set(runner.CHECKPOINT_FEATURES).issubset(runner.C0_FEATURES)


def test_daily_stock_pass_gate_requires_every_preregistered_condition() -> None:
    runner = load_runner()
    old = {
        "log_loss": 0.20,
        "brier_score": 0.04,
        "auc": 0.60,
        "average_precision": 0.10,
    }
    new = {
        "log_loss": 0.19,
        "brier_score": 0.03,
        "auc": 0.61,
        "average_precision": 0.11,
    }
    bootstrap = pd.DataFrame(
        [
            {
                "statistic": "A1_minus_A0_log_loss_improvement",
                "confidence": 0.80,
                "lower": 0.001,
            },
            {
                "statistic": "A1_minus_A0_brier_improvement",
                "confidence": 0.80,
                "lower": 0.001,
            },
        ]
    )

    status, gates = runner.evaluate_increment_status(
        prefix="A1_minus_A0",
        old=old,
        new=new,
        bootstrap=bootstrap,
        positive_months=4,
        adverse_checkpoint_groups=0,
        null_metrics=None,
    )

    assert status == "supported"
    assert gates["passed"] is True
    weakened = dict(new, average_precision=0.09)
    weakened_status, weakened_gates = runner.evaluate_increment_status(
        prefix="A1_minus_A0",
        old=old,
        new=weakened,
        bootstrap=bootstrap,
        positive_months=4,
        adverse_checkpoint_groups=0,
        null_metrics=None,
    )
    assert weakened_status == "not_supported"
    assert weakened_gates["average_precision_improved"] is False


def test_back_expiry_preflight_uses_exactly_one_noncompact_bounded_request() -> None:
    preflight = load_preflight()
    calls: list[tuple[str, dict[str, object], float]] = []

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"data":[],"meta":{"page":{"limit":100,"offset":0}}}'

        def json(self) -> object:
            return {
                "data": [
                    {
                        "id": "AAL250101C00012500-2025-08-21",
                        "type": "options-eod",
                        "attributes": {
                            "exp_date": "2025-10-17",
                            "strike": 12.5,
                        },
                    },
                    {
                        "id": "AAL250101C00012500-2025-08-25",
                        "type": "options-eod",
                        "attributes": {
                            "exp_date": "2025-10-17",
                            "strike": 12.5,
                        },
                    },
                ],
                "meta": {"page": {"limit": 100, "offset": 0}},
            }

    def fake_get(
        url: str,
        *,
        params: dict[str, object],
        timeout: float,
    ) -> Response:
        calls.append((url, params, timeout))
        return Response()

    result = preflight.perform_preflight(
        token="secret-for-test",
        get=fake_get,
    )

    assert result["status"] == "supported_noncompact_schema"
    assert len(calls) == 1
    _, parameters, _ = calls[0]
    assert parameters["compact"] == 0
    assert parameters["page[limit]"] == 100
    assert "api_token" in parameters
    assert result["parameters"].get("api_token") is None
    assert result["request_count"] == 1
    assert result["protected_records_returned"] == 1
    assert result["protected_records_persisted"] == 0
    assert result["raw_response_persisted"] is False
    assert result["raw_response_cache_path"] is None


def test_frozen_checkpoint_groups_use_the_preregistered_late_label() -> None:
    runner = load_runner()

    structural, reconstruction = runner.load_structural_panel()

    assert reconstruction["passed"] is True
    assert set(structural["checkpoint_group"]) == {
        "early_6_14",
        "middle_16_24",
        "late_26_34",
    }
