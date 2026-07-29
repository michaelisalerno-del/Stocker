"""Pure research helpers for the prior-close options IV movement screen V0."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge

from stocker_data.calendars import get_market_calendar
from stocker_research.broad_conflict_advance_hazard_v02 import (
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    ROUTE_FEATURES,
)

ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390
PRIMARY_HORIZON_MINUTES: Final[int] = 15
OPTIONS_PRIMARY_FEATURES: Final[tuple[str, ...]] = (
    "atm_iv",
    "call_put_iv_gap",
    "straddle_mid_pct",
    "combined_relative_spread",
    "log1p_combined_open_interest",
    "front_dte",
    "atm_log_moneyness",
    "skew_25d",
    "skew_25d_missing",
    "term_structure",
    "term_structure_missing",
)
OPTIONS_SCREEN_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "broad_conflict_predicts_iv_excess_movement",
        "route_competition_predicts_iv_excess_without_broad_conflict_specificity",
        "structural_hazard_does_not_translate_to_iv_excess_movement",
        "descriptive_options_movement_structure_only",
        "no_options_movement_increment",
        "blocked_missing_eodhd_api_token",
        "blocked_eodhd_options_schema_unverified",
        "blocked_historical_options_date_unavailable",
        "blocked_options_download_resource_limit",
        "blocked_options_download_incomplete",
        "blocked_options_data_integrity_failure",
        "blocked_underlying_symbol_mapping_failure",
        "blocked_structural_panel_reconstruction_failure",
        "blocked_insufficient_options_chain_coverage",
        "blocked_option_pair_selection_failure",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
)
FROZEN_COHORT: Final[tuple[str, ...]] = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
SAFETY_FLAGS: Final[dict[str, bool | str]] = {
    "research_only": True,
    "options_feasibility_screen": True,
    "options_data_granularity": "end_of_day",
    "options_information_time": "previous_trading_day_close",
    "intraday_option_fill_simulated": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "economic_strategy_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}


class ChronologyError(ValueError):
    """An option observation violates the exact prior-close information clock."""


@dataclass(frozen=True)
class OptionPairSelection:
    """Frozen front-expiry/common-strike call-put selection result."""

    available: bool
    reason: str
    expiration_date: date | None = None
    dte: int | None = None
    strike: float | None = None
    call_contract_id: str | None = None
    put_contract_id: str | None = None
    call: Mapping[str, Any] | None = None
    put: Mapping[str, Any] | None = None


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FrozenOptionsLinearModel:
    """Development-frozen preprocessing and fitted deterministic linear model."""

    model_id: str
    kind: str
    numeric_features: tuple[str, ...]
    numeric_medians: FloatArray
    numeric_means: FloatArray
    numeric_scales: FloatArray
    category_levels: Mapping[str, tuple[str, ...]]
    design_columns: tuple[str, ...]
    coefficients: FloatArray
    intercept: float
    iterations: int
    preprocessing_fitted_period: str = "development_2024_only"

    def _design(self, frame: pd.DataFrame) -> FloatArray:
        missing = sorted(set(self.numeric_features).difference(frame.columns))
        if missing:
            raise ValueError(f"model frame missing numeric features: {missing}")
        values = frame.loc[:, list(self.numeric_features)].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.numeric_medians)
        numeric = (values - self.numeric_means) / self.numeric_scales
        parts: list[FloatArray] = [np.asarray(numeric, dtype=np.float64)]
        categorical_values = _categorical_controls(frame)
        for control, levels in self.category_levels.items():
            observed = categorical_values[control].astype(str).to_numpy()
            for level in levels[1:]:
                parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
        design = np.concatenate(parts, axis=1)
        if design.shape[1] != len(self.design_columns):
            raise AssertionError("frozen options design width drifted")
        return np.asarray(design, dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        """Predict probabilities for logistic models or residuals for Ridge."""

        linear = self._design(frame) @ self.coefficients + self.intercept
        if self.kind == "ridge":
            return np.asarray(linear, dtype=np.float64)
        if self.kind != "logistic":
            raise ValueError(f"unknown model kind: {self.kind}")
        clipped = np.clip(linear, -709.0, 709.0)
        return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def previous_trading_session(signal_date: date, *, calendar_name: str = "NYSE") -> date:
    """Return the exact previous valid exchange session."""

    calendar = get_market_calendar(calendar_name)
    sessions = calendar.valid_days(
        start_date=signal_date - timedelta(days=30),
        end_date=signal_date - timedelta(days=1),
        tz="America/New_York",
    )
    if len(sessions) == 0:
        raise ChronologyError(f"no prior trading session available for {signal_date}")
    return cast(date, sessions[-1].date())


def validate_exact_previous_session_join(
    *, signal_date: date, required_options_date: date, actual_options_date: date
) -> None:
    """Reject same-day, future, or stale option observations."""

    if actual_options_date != required_options_date or actual_options_date >= signal_date:
        raise ChronologyError(
            "option chain must be from the exact previous trading session: "
            f"signal={signal_date}, required={required_options_date}, actual={actual_options_date}"
        )


def split_boundary_is_ambiguous(
    *, options_date: date, signal_date: date, split_dates: set[date]
) -> bool:
    """Return whether a corporate-action boundary separates chain and movement prices."""

    return any(options_date < split_date <= signal_date for split_date in split_dates)


def _number(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _relative_spread(row: Mapping[str, Any]) -> float:
    bid = _number(row, "bid")
    ask = _number(row, "ask")
    midpoint = _number(row, "midpoint")
    if bid is None or ask is None or midpoint is None or midpoint <= 0.0 or ask < bid:
        return math.inf
    return (ask - bid) / midpoint


def _combined_relative_spread(call: Mapping[str, Any], put: Mapping[str, Any]) -> float:
    call_bid = _number(call, "bid")
    call_ask = _number(call, "ask")
    put_bid = _number(put, "bid")
    put_ask = _number(put, "ask")
    call_midpoint = _number(call, "midpoint")
    put_midpoint = _number(put, "midpoint")
    values = (call_bid, call_ask, put_bid, put_ask, call_midpoint, put_midpoint)
    if any(value is None for value in values):
        return math.inf
    typed = cast(tuple[float, float, float, float, float, float], values)
    denominator = typed[4] + typed[5]
    if denominator <= 0.0 or typed[1] < typed[0] or typed[3] < typed[2]:
        return math.inf
    return (typed[1] - typed[0] + typed[3] - typed[2]) / denominator


def _pair_quality_reason(side: str, row: Mapping[str, Any]) -> str | None:
    iv = _number(row, "implied_volatility")
    if iv is None or not 0.005 <= iv <= 5.0:
        return f"selected_pair_{side}_iv_invalid"
    bid = _number(row, "bid")
    if bid is None or bid < 0.0:
        return f"selected_pair_{side}_bid_invalid"
    ask = _number(row, "ask")
    if ask is None or ask < bid:
        return f"selected_pair_{side}_ask_invalid"
    midpoint = _number(row, "midpoint")
    if midpoint is None or midpoint <= 0.0:
        return f"selected_pair_{side}_midpoint_not_positive"
    open_interest = _number(row, "open_interest")
    if open_interest is None or open_interest < 10.0:
        return "selected_pair_open_interest_below_10"
    if _relative_spread(row) > 1.0:
        return f"selected_pair_{side}_relative_spread_above_1"
    delta = _number(row, "delta")
    if delta is not None and abs(delta) > 1.05:
        return f"selected_pair_{side}_delta_implausible"
    gamma = _number(row, "gamma")
    if gamma is not None and gamma < 0.0:
        return f"selected_pair_{side}_gamma_negative"
    expiration = row.get("expiration_date")
    trade_day = row.get("trade_date")
    if isinstance(expiration, date) and isinstance(trade_day, date) and expiration < trade_day:
        return f"selected_pair_{side}_expiration_before_trade"
    return None


def _as_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _select_atm_pair(
    chain: pd.DataFrame, *, previous_close: float, minimum_dte: int, maximum_dte: int
) -> OptionPairSelection:
    """Apply a bounded nearest-expiry/common-strike rule without liquidity fallback."""

    required = {
        "option_type",
        "expiration_date",
        "dte",
        "strike",
        "contract_id",
        "bid",
        "ask",
        "midpoint",
        "implied_volatility",
        "open_interest",
    }
    missing = sorted(required.difference(chain.columns))
    if missing:
        raise ValueError(f"option chain missing columns: {missing}")
    if not math.isfinite(previous_close) or previous_close <= 0.0:
        raise ValueError("previous close must be finite and positive")
    working = chain.copy()
    working["expiration_date"] = pd.to_datetime(working["expiration_date"], errors="coerce").dt.date
    working["dte"] = pd.to_numeric(working["dte"], errors="coerce")
    working["strike"] = pd.to_numeric(working["strike"], errors="coerce")
    working["option_type"] = working["option_type"].astype(str).str.casefold()
    working = working.loc[
        working["dte"].between(minimum_dte, maximum_dte, inclusive="both")
        & working["strike"].gt(0.0)
        & working["option_type"].isin(["call", "put"])
        & working["expiration_date"].notna()
    ]
    eligible_expiries: list[tuple[int, date]] = []
    for expiry, expiry_frame in working.groupby("expiration_date", sort=True):
        calls = set(expiry_frame.loc[expiry_frame["option_type"].eq("call"), "strike"])
        puts = set(expiry_frame.loc[expiry_frame["option_type"].eq("put"), "strike"])
        if calls.intersection(puts):
            dte = int(pd.to_numeric(expiry_frame["dte"], errors="raise").min())
            eligible_expiries.append((dte, cast(date, expiry)))
    if not eligible_expiries:
        return OptionPairSelection(False, "no_eligible_common_strike_expiry")
    selected_dte, selected_expiry = min(eligible_expiries, key=lambda value: (value[0], value[1]))
    expiry_frame = working.loc[working["expiration_date"].eq(selected_expiry)]
    calls_by_strike: dict[float, list[dict[str, Any]]] = {}
    puts_by_strike: dict[float, list[dict[str, Any]]] = {}
    for record in _as_records(expiry_frame):
        destination = calls_by_strike if record["option_type"] == "call" else puts_by_strike
        destination.setdefault(float(record["strike"]), []).append(record)
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for strike in sorted(set(calls_by_strike).intersection(puts_by_strike)):
        for call in calls_by_strike[strike]:
            for put in puts_by_strike[strike]:
                call_oi = _number(call, "open_interest")
                put_oi = _number(put, "open_interest")
                minimum_oi = min(
                    -math.inf if call_oi is None else call_oi,
                    -math.inf if put_oi is None else put_oi,
                )
                call_iv = _number(call, "implied_volatility")
                put_iv = _number(put, "implied_volatility")
                iv_gap = math.inf if call_iv is None or put_iv is None else abs(call_iv - put_iv)
                rank = (
                    abs(math.log(strike / previous_close)),
                    -minimum_oi,
                    _combined_relative_spread(call, put),
                    iv_gap,
                    strike,
                    str(call["contract_id"]),
                    str(put["contract_id"]),
                )
                candidates.append((rank, call, put))
    _rank, call, put = min(candidates, key=lambda candidate: candidate[0])
    strike = float(call["strike"])
    for side, row in (("call", call), ("put", put)):
        reason = _pair_quality_reason(side, row)
        if reason is not None:
            return OptionPairSelection(
                available=False,
                reason=reason,
                expiration_date=selected_expiry,
                dte=selected_dte,
                strike=strike,
                call_contract_id=str(call["contract_id"]),
                put_contract_id=str(put["contract_id"]),
                call=call,
                put=put,
            )
    return OptionPairSelection(
        available=True,
        reason="selected",
        expiration_date=selected_expiry,
        dte=selected_dte,
        strike=strike,
        call_contract_id=str(call["contract_id"]),
        put_contract_id=str(put["contract_id"]),
        call=call,
        put=put,
    )


def select_primary_atm_pair(chain: pd.DataFrame, *, previous_close: float) -> OptionPairSelection:
    """Apply the frozen 7--45 DTE primary ATM-pair rule."""

    return _select_atm_pair(
        chain,
        previous_close=previous_close,
        minimum_dte=7,
        maximum_dte=45,
    )


def calculate_primary_option_features(
    selection: OptionPairSelection, *, previous_close: float
) -> dict[str, Any]:
    """Calculate only frozen prior-close ATM-pair features."""

    if not selection.available or selection.call is None or selection.put is None:
        raise ValueError("valid primary option pair required")
    if selection.strike is None or selection.dte is None or selection.expiration_date is None:
        raise ValueError("selected pair identity is incomplete")
    call = selection.call
    put = selection.put
    call_iv = cast(float, _number(call, "implied_volatility"))
    put_iv = cast(float, _number(put, "implied_volatility"))
    call_midpoint = cast(float, _number(call, "midpoint"))
    put_midpoint = cast(float, _number(put, "midpoint"))
    call_bid = cast(float, _number(call, "bid"))
    call_ask = cast(float, _number(call, "ask"))
    put_bid = cast(float, _number(put, "bid"))
    put_ask = cast(float, _number(put, "ask"))
    call_oi = int(cast(float, _number(call, "open_interest")))
    put_oi = int(cast(float, _number(put, "open_interest")))
    straddle_mid = call_midpoint + put_midpoint
    straddle_mid_pct = straddle_mid / previous_close
    combined_open_interest = call_oi + put_oi
    return {
        "atm_iv": (call_iv + put_iv) / 2.0,
        "call_put_iv_gap": call_iv - put_iv,
        "straddle_mid": straddle_mid,
        "straddle_mid_pct": straddle_mid_pct,
        "combined_relative_spread": (call_ask - call_bid + put_ask - put_bid) / straddle_mid,
        "combined_open_interest": combined_open_interest,
        "log1p_combined_open_interest": math.log1p(combined_open_interest),
        "front_dte": selection.dte,
        "atm_log_moneyness": math.log(selection.strike / previous_close),
        "scaled_straddle_move_15m": straddle_mid_pct
        * math.sqrt(15.0 / max(selection.dte * 390.0, 15.0)),
        "call_iv": call_iv,
        "put_iv": put_iv,
        "call_delta": _number(call, "delta"),
        "put_delta": _number(put, "delta"),
        "call_midpoint": call_midpoint,
        "put_midpoint": put_midpoint,
        "call_relative_spread": _relative_spread(call),
        "put_relative_spread": _relative_spread(put),
        "call_open_interest": call_oi,
        "put_open_interest": put_oi,
        "call_volume": _number(call, "volume"),
        "put_volume": _number(put, "volume"),
        "selected_strike": selection.strike,
        "selected_expiry": selection.expiration_date,
        "call_contract_id": selection.call_contract_id,
        "put_contract_id": selection.put_contract_id,
    }


def _nearest_delta_contract(
    frame: pd.DataFrame, *, option_type: str, target_delta: float
) -> Mapping[str, Any] | None:
    candidates: list[tuple[tuple[float, str], Mapping[str, Any]]] = []
    for row in _as_records(frame.loc[frame["option_type"].astype(str).eq(option_type)]):
        delta = _number(row, "delta")
        if delta is None:
            continue
        distance = abs(delta - target_delta)
        if distance > 0.10:
            continue
        if _pair_quality_reason(option_type, row) is not None:
            continue
        candidates.append(((distance, str(row.get("contract_id", ""))), row))
    return None if not candidates else min(candidates, key=lambda item: item[0])[1]


def calculate_optional_option_features(
    chain: pd.DataFrame,
    *,
    front_selection: OptionPairSelection,
    previous_close: float,
) -> dict[str, float | int]:
    """Calculate non-gating 25-delta skew and 46--90 DTE ATM term structure."""

    if not front_selection.available or front_selection.expiration_date is None:
        raise ValueError("valid front pair required for optional features")
    working = chain.copy()
    working["expiration_date"] = pd.to_datetime(working["expiration_date"], errors="coerce").dt.date
    working["option_type"] = working["option_type"].astype(str).str.casefold()
    front = working.loc[working["expiration_date"].eq(front_selection.expiration_date)]
    put_25d = _nearest_delta_contract(front, option_type="put", target_delta=-0.25)
    call_25d = _nearest_delta_contract(front, option_type="call", target_delta=0.25)
    if put_25d is None or call_25d is None:
        skew = math.nan
        skew_missing = 1
    else:
        skew = cast(float, _number(put_25d, "implied_volatility")) - cast(
            float, _number(call_25d, "implied_volatility")
        )
        skew_missing = 0
    back = _select_atm_pair(
        working,
        previous_close=previous_close,
        minimum_dte=46,
        maximum_dte=90,
    )
    if not back.available:
        term_structure = math.nan
        term_missing = 1
    else:
        back_features = calculate_primary_option_features(back, previous_close=previous_close)
        front_features = calculate_primary_option_features(
            front_selection, previous_close=previous_close
        )
        term_structure = float(back_features["atm_iv"]) - float(front_features["atm_iv"])
        term_missing = 0
    return {
        "skew_25d": skew,
        "skew_25d_missing": skew_missing,
        "term_structure": term_structure,
        "term_structure_missing": term_missing,
    }


def iv_movement_approximations(atm_iv: float, *, horizon_minutes: int = 15) -> dict[str, float]:
    """Scale previous-close annualised IV to a model-based intraday movement amount."""

    if not math.isfinite(atm_iv) or atm_iv <= 0.0:
        raise ValueError("ATM IV must be finite and positive")
    if horizon_minutes != PRIMARY_HORIZON_MINUTES:
        raise ValueError("primary IV approximation horizon is frozen at 15 minutes")
    sigma = atm_iv * math.sqrt(horizon_minutes / ANNUAL_TRADING_MINUTES)
    return {
        "iv_sigma_15m": sigma,
        "iv_expected_absolute_15m": sigma * math.sqrt(2.0 / math.pi),
    }


def add_iv_relative_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the frozen IV-relative underlying-movement outcomes."""

    required = {
        "absolute_log_return_15m",
        "iv_sigma_15m",
        "iv_expected_absolute_15m",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"movement panel missing IV-relative inputs: {missing}")
    output = frame.copy()
    movement = pd.to_numeric(output["absolute_log_return_15m"], errors="raise")
    sigma = pd.to_numeric(output["iv_sigma_15m"], errors="raise")
    expected = pd.to_numeric(output["iv_expected_absolute_15m"], errors="raise")
    if not np.isfinite(movement.to_numpy(float)).all() or bool(movement.lt(0.0).any()):
        raise ValueError("absolute movement must be finite and non-negative")
    if not np.isfinite(sigma.to_numpy(float)).all() or bool(sigma.le(0.0).any()):
        raise ValueError("IV sigma must be finite and positive")
    if not np.isfinite(expected.to_numpy(float)).all() or bool(expected.le(0.0).any()):
        raise ValueError("IV expected movement must be finite and positive")
    output["iv_absolute_residual_15m"] = movement - expected
    output["iv_sigma_ratio_15m"] = movement / sigma
    output["movement_exceeds_iv_expected_absolute"] = movement.gt(expected).astype(int)
    output["movement_exceeds_one_iv_sigma"] = movement.gt(sigma).astype(int)
    return output


def assign_development_frozen_iv_deciles(
    development_atm_iv: pd.Series, values: pd.Series
) -> tuple[pd.Series, tuple[float, ...]]:
    """Fit ATM-IV decile edges on development and apply them unchanged."""

    development = pd.to_numeric(development_atm_iv, errors="raise").to_numpy(float)
    if development.size < 10 or not np.isfinite(development).all():
        raise ValueError("ATM-IV deciles require at least ten finite development values")
    edges_array = np.quantile(development, np.arange(1, 10) / 10.0, method="linear")
    edges = tuple(float(value) for value in edges_array)
    observed = pd.to_numeric(values, errors="raise").to_numpy(float)
    if not np.isfinite(observed).all():
        raise ValueError("ATM-IV decile inputs must be finite")
    assigned = np.searchsorted(edges_array, observed, side="right").astype(int)
    return pd.Series(assigned, index=values.index, name="atm_iv_decile"), edges


def _dte_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    labels = pd.cut(
        numeric,
        bins=[6.999999, 14.0, 30.0, 45.0],
        labels=["7-14", "15-30", "31-45"],
        include_lowest=True,
        right=True,
    )
    if labels.isna().any():
        raise ValueError("front DTE falls outside frozen matching bins")
    return labels.astype(str)


def build_matched_control_relations(
    assessment: pd.DataFrame, *, minimum_controls: int = 5
) -> pd.DataFrame:
    """Construct deterministic same-stock/month/checkpoint/options-matched controls."""

    if minimum_controls != 5:
        raise ValueError("matched-control minimum is frozen at five")
    required = {
        "row_id",
        "symbol",
        "session",
        "year_month",
        "checkpoint",
        "route_resolution_state",
        "atm_iv_decile",
        "front_dte",
        "any_prefix_one_transition_from_completion",
        "advance_eligible",
    }
    if missing := sorted(required.difference(assessment.columns)):
        raise ValueError(f"matched-control panel missing columns: {missing}")
    working = assessment.copy()
    working["front_dte_bin"] = _dte_bin(working["front_dte"])
    treated = working.loc[working["route_resolution_state"].eq("BROAD_CONFLICT")]
    pool = working.loc[
        ~working["route_resolution_state"].eq("BROAD_CONFLICT")
        & working["any_prefix_one_transition_from_completion"].astype(int).eq(0)
        & working["advance_eligible"].astype(int).eq(1)
    ]
    match_columns = [
        "symbol",
        "year_month",
        "checkpoint",
        "atm_iv_decile",
        "front_dte_bin",
    ]
    pool_groups = {
        key: group.sort_values(["session", "row_id"], kind="mergesort")
        for key, group in pool.groupby(match_columns, sort=True, observed=True)
    }
    rows: list[dict[str, object]] = []
    for treated_row in _as_records(treated.sort_values("row_id", kind="mergesort")):
        key = tuple(treated_row[column] for column in match_columns)
        candidates = pool_groups.get(key)
        if candidates is None:
            continue
        different = candidates.loc[
            candidates["session"].astype(str).ne(str(treated_row["session"]))
        ]
        selected = different if len(different) >= minimum_controls else candidates
        if len(selected) < minimum_controls:
            continue
        weight = 1.0 / len(selected)
        for control in _as_records(selected):
            rows.append(
                {
                    "treated_row_id": str(treated_row["row_id"]),
                    "control_row_id": str(control["row_id"]),
                    "match_weight": weight,
                    "different_session": str(control["session"]) != str(treated_row["session"]),
                    "controls_for_treated": len(selected),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "treated_row_id",
            "control_row_id",
            "match_weight",
            "different_session",
            "controls_for_treated",
        ],
    )


def _categorical_controls(frame: pd.DataFrame) -> dict[str, pd.Series]:
    required = {"symbol", "checkpoint", "year_month"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"model frame missing fixed-effect controls: {missing}")
    month = frame["year_month"].astype(str).str[-2:]
    if not month.str.fullmatch(r"(?:0[1-9]|1[0-2])").all():
        raise ValueError("year_month cannot be mapped to month-of-year")
    return {
        "stock": frame["symbol"].astype(str),
        "checkpoint": frame["checkpoint"].astype(str),
        "month_of_year": month,
    }


def fit_options_linear_model(
    development: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    model_id: str,
    kind: str,
    target_column: str | None = None,
) -> FrozenOptionsLinearModel:
    """Fit one frozen O0/O1 logistic or R0/R1 Ridge model on 2024 development only."""

    features = tuple(numeric_features)
    if not features or len(set(features)) != len(features):
        raise ValueError("model numeric feature surface must be non-empty and unique")
    if "period" in development and not development["period"].astype(str).eq("development").all():
        raise ValueError("model fitting accepts development rows only")
    sessions = pd.to_datetime(development["session"], errors="raise")
    if not sessions.dt.year.eq(2024).all():
        raise ValueError("model preprocessing and fitting are frozen to 2024")
    if missing := sorted(set(features).difference(development.columns)):
        raise ValueError(f"model development frame missing features: {missing}")
    raw = development.loc[:, list(features)].to_numpy(float)
    finite = np.where(np.isfinite(raw), raw, np.nan)
    medians = np.nanmedian(finite, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("every numeric model feature needs finite development support")
    imputed = np.where(np.isfinite(raw), raw, medians)
    means = np.asarray(imputed.mean(axis=0), dtype=np.float64)
    scales = np.asarray(imputed.std(axis=0, ddof=0), dtype=np.float64)
    scales = np.where(scales >= 1e-12, scales, 1.0)
    controls = _categorical_controls(development)
    category_levels = {
        name: tuple(sorted(series.astype(str).unique())) for name, series in controls.items()
    }
    design_names = list(features)
    parts: list[FloatArray] = [np.asarray((imputed - means) / scales, dtype=np.float64)]
    for name, levels in category_levels.items():
        observed = controls[name].astype(str).to_numpy()
        for level in levels[1:]:
            parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
            design_names.append(f"control_{name}__{level}")
    design = np.concatenate(parts, axis=1)
    weights = pd.to_numeric(development["row_weight"], errors="raise").to_numpy(float)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("model row weights must be finite and positive")
    if kind == "logistic":
        target_name = target_column or "movement_exceeds_iv_expected_absolute"
        labels = pd.to_numeric(development[target_name], errors="raise").to_numpy(int)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{model_id} requires both binary target classes")
        estimator = LogisticRegression(
            penalty="l2",
            C=0.25,
            solver="liblinear",
            max_iter=300,
            class_weight=None,
            random_state=20260722,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(design, labels, sample_weight=weights)
        iterations = int(np.max(estimator.n_iter_))
        if iterations >= 300:
            raise RuntimeError(f"{model_id} failed to converge")
        coefficients = np.asarray(estimator.coef_[0], dtype=np.float64)
        intercept = float(estimator.intercept_[0])
    elif kind == "ridge":
        target_name = target_column or "iv_absolute_residual_15m"
        target = pd.to_numeric(development[target_name], errors="raise").to_numpy(float)
        if not np.isfinite(target).all():
            raise ValueError(f"{model_id} Ridge target must be finite")
        estimator = Ridge(alpha=10.0, fit_intercept=True, solver="cholesky")
        estimator.fit(design, target, sample_weight=weights)
        iterations = 1
        coefficients = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = float(estimator.intercept_)
    else:
        raise ValueError("model kind must be logistic or ridge")
    return FrozenOptionsLinearModel(
        model_id=model_id,
        kind=kind,
        numeric_features=features,
        numeric_medians=np.asarray(medians, dtype=np.float64),
        numeric_means=means,
        numeric_scales=scales,
        category_levels=category_levels,
        design_columns=tuple(design_names),
        coefficients=coefficients,
        intercept=intercept,
        iterations=iterations,
    )


def fixed_session_bootstrap_multiplicities(
    frame: pd.DataFrame, *, draws: int, seed: int
) -> npt.NDArray[np.int64]:
    """Return exactly 25 fixed-seed whole-session multiplicity vectors."""

    if draws != 25:
        raise ValueError("the V0 session bootstrap requires exactly 25 draws")
    if "session" not in frame or frame.empty:
        raise ValueError("session bootstrap requires non-empty session rows")
    sessions = frame["session"].astype(str)
    unique = np.asarray(sorted(sessions.unique()), dtype=object)
    rng = np.random.default_rng(seed)
    output = np.empty((draws, len(frame)), dtype=np.int64)
    for draw in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        counts = {session: int(np.sum(selected == session)) for session in unique}
        output[draw] = sessions.map(counts).to_numpy(dtype=np.int64)
    return output


def permute_intact_route_bundle(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Permute one intact frozen route bundle among stocks inside every causal slate."""

    strata = ("period", "session", "checkpoint")
    required = {*strata, "symbol", *ROUTE_FEATURES}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"route-null panel missing columns: {missing}")
    output = frame.copy()
    rng = np.random.default_rng(seed)
    for _key, slate in frame.groupby(list(strata), sort=True, observed=True):
        indices = slate.sort_values("symbol", kind="mergesort").index.to_numpy()
        sources = rng.permutation(indices)
        output.loc[indices, list(ROUTE_FEATURES)] = frame.loc[
            sources, list(ROUTE_FEATURES)
        ].to_numpy()
    return output


def assert_protected_boundary(
    *, signal_dates: pd.Series, options_dates: pd.Series, protected_start: date = date(2025, 8, 23)
) -> None:
    """Reject any signal or option observation at or beyond the protected boundary."""

    signals = pd.to_datetime(signal_dates, errors="raise").dt.date
    options = pd.to_datetime(options_dates, errors="raise").dt.date
    if signals.ge(protected_start).any() or options.ge(protected_start).any():
        raise ValueError("protected date boundary materialised")


def coverage_gates_pass(evidence: Mapping[str, object]) -> bool:
    """Apply every frozen options download, row-support, and concentration gate."""

    required = {
        "historical_symbols",
        "paired_symbols_development",
        "paired_symbols_assessment",
        "development_row_coverage",
        "assessment_row_coverage",
        "assessment_rows",
        "assessment_sessions",
        "assessment_months",
        "assessment_broad_conflict_rows",
        "assessment_low_route_support_rows",
        "maximum_stock_weight_share",
        "download_integrity_passed",
    }
    if missing := sorted(required.difference(evidence)):
        raise ValueError(f"options coverage gates missing: {missing}")
    return bool(
        bool(evidence["download_integrity_passed"])
        and int(cast(Any, evidence["historical_symbols"])) >= 15
        and int(cast(Any, evidence["paired_symbols_development"])) >= 12
        and int(cast(Any, evidence["paired_symbols_assessment"])) >= 12
        and float(cast(Any, evidence["development_row_coverage"])) >= 0.70
        and float(cast(Any, evidence["assessment_row_coverage"])) >= 0.70
        and int(cast(Any, evidence["assessment_rows"])) >= 20_000
        and int(cast(Any, evidence["assessment_sessions"])) >= 130
        and int(cast(Any, evidence["assessment_months"])) >= 7
        and int(cast(Any, evidence["assessment_broad_conflict_rows"])) >= 250
        and int(cast(Any, evidence["assessment_low_route_support_rows"])) >= 250
        and float(cast(Any, evidence["maximum_stock_weight_share"])) <= 0.12
    )


def o1_model_gate_passes(gates: Mapping[str, object]) -> bool:
    """Apply the exact O1-versus-O0 primary model gate."""

    required = {
        "log_loss_improvement",
        "brier_improvement",
        "auc_improvement",
        "average_precision_improvement",
        "bootstrap_80_log_loss_lower",
        "bootstrap_80_brier_lower",
        "bootstrap_80_average_precision_lower",
        "positive_months",
        "materially_adverse_checkpoint_groups",
        "real_exceeds_matching_nulls",
        "coverage_and_concentration_passed",
    }
    if missing := sorted(required.difference(gates)):
        raise ValueError(f"O1 gates missing: {missing}")
    return bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["average_precision_improvement"])) > 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_average_precision_lower"])) >= 0.0
        and int(cast(Any, gates["positive_months"])) >= 5
        and int(cast(Any, gates["materially_adverse_checkpoint_groups"])) == 0
        and int(cast(Any, gates["real_exceeds_matching_nulls"])) >= 4
        and bool(gates["coverage_and_concentration_passed"])
    )


def broad_conflict_iv_gate_passes(gates: Mapping[str, object]) -> bool:
    """Apply the exact BROAD_CONFLICT IV-excess state gate."""

    required = {
        "mean_residual",
        "minus_low_route_support_residual",
        "minus_matched_residual",
        "minus_matched_exceed_rate",
        "bootstrap_80_minus_low_residual_lower",
        "bootstrap_80_minus_matched_residual_lower",
        "bootstrap_80_minus_matched_exceed_lower",
        "positive_months",
        "materially_adverse_checkpoint_groups",
        "support_and_concentration_passed",
    }
    if missing := sorted(required.difference(gates)):
        raise ValueError(f"BROAD_CONFLICT IV gates missing: {missing}")
    return bool(
        float(cast(Any, gates["mean_residual"])) > 0.0
        and float(cast(Any, gates["minus_low_route_support_residual"])) > 0.0
        and float(cast(Any, gates["minus_matched_residual"])) > 0.0
        and float(cast(Any, gates["minus_matched_exceed_rate"])) > 0.0
        and float(cast(Any, gates["bootstrap_80_minus_low_residual_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_minus_matched_residual_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_minus_matched_exceed_lower"])) >= 0.0
        and int(cast(Any, gates["positive_months"])) >= 5
        and int(cast(Any, gates["materially_adverse_checkpoint_groups"])) == 0
        and bool(gates["support_and_concentration_passed"])
    )


def choose_options_movement_decision(
    *,
    blocker: str | None,
    o1_passed: bool,
    broad_conflict_passed: bool,
    descriptive_only: bool,
) -> str:
    """Choose exactly one frozen options movement-screen decision."""

    if blocker is not None:
        if blocker not in OPTIONS_SCREEN_DECISIONS or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown options-screen blocker: {blocker}")
        return blocker
    if o1_passed and broad_conflict_passed:
        return "broad_conflict_predicts_iv_excess_movement"
    if o1_passed:
        return "route_competition_predicts_iv_excess_without_broad_conflict_specificity"
    if broad_conflict_passed or descriptive_only:
        return "descriptive_options_movement_structure_only"
    return "structural_hazard_does_not_translate_to_iv_excess_movement"


def _positive_float(value: object, *, name: str) -> float:
    number = float(cast(Any, value))
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def compute_underlying_movement_outcomes(
    structural: pd.DataFrame, bars: pd.DataFrame
) -> pd.DataFrame:
    """Attach primary and descriptive outcomes using bars strictly after each checkpoint."""

    structural_required = {"row_id", "symbol", "session", "checkpoint_bar_ordinal_zero_based"}
    bar_required = {"symbol", "session", "bar_ordinal", "open", "high", "low", "close"}
    if missing := sorted(structural_required.difference(structural.columns)):
        raise ValueError(f"structural panel missing columns: {missing}")
    if missing := sorted(bar_required.difference(bars.columns)):
        raise ValueError(f"bar panel missing columns: {missing}")
    bar_groups = {
        (str(symbol), str(session)): group.sort_values("bar_ordinal", kind="mergesort")
        for (symbol, session), group in bars.groupby(["symbol", "session"], sort=False)
    }
    output_rows: list[dict[str, Any]] = []
    for source in _as_records(structural):
        key = (str(source["symbol"]), str(source["session"]))
        session_bars = bar_groups.get(key)
        if session_bars is None:
            raise ValueError(f"underlying bars unavailable for {key}")
        checkpoint = int(source["checkpoint_bar_ordinal_zero_based"])
        indexed = session_bars.set_index("bar_ordinal", drop=False)
        required_ordinals = [checkpoint + offset for offset in (1, 2, 3)]
        if any(ordinal not in indexed.index for ordinal in required_ordinals):
            raise ValueError(f"three future completed bars unavailable for {source['row_id']}")
        future = indexed.loc[required_ordinals]
        entry_price = _positive_float(future.iloc[0]["open"], name="entry price")
        closes = [_positive_float(value, name="future close") for value in future["close"]]
        highs = [_positive_float(value, name="future high") for value in future["high"]]
        lows = [_positive_float(value, name="future low") for value in future["low"]]
        five_minute_returns = [
            math.log(closes[0] / entry_price),
            math.log(closes[1] / closes[0]),
            math.log(closes[2] / closes[1]),
        ]
        maximum_high = max(highs)
        minimum_low = min(lows)
        record = dict(source)
        record.update(
            {
                "entry_price": entry_price,
                "absolute_log_return_10m": abs(math.log(closes[1] / entry_price)),
                "absolute_log_return_15m": abs(math.log(closes[2] / entry_price)),
                "realised_range_15m": math.log(maximum_high / minimum_low),
                "maximum_absolute_excursion_15m": max(
                    abs(math.log(maximum_high / entry_price)),
                    abs(math.log(minimum_low / entry_price)),
                ),
                "realised_variance_15m": sum(value * value for value in five_minute_returns),
                "primary_horizon_last_bar_ordinal": required_ordinals[-1],
            }
        )
        if {"bar_start_timestamp", "bar_complete_timestamp"}.issubset(future.columns):
            record["entry_bar_start_timestamp"] = future.iloc[0]["bar_start_timestamp"]
            record["primary_horizon_last_bar_complete_timestamp"] = future.iloc[-1][
                "bar_complete_timestamp"
            ]
        for bars_forward, column in (
            (6, "absolute_log_return_30m"),
            (12, "absolute_log_return_60m"),
        ):
            ordinal = checkpoint + bars_forward
            if ordinal in indexed.index:
                later_close = _positive_float(indexed.loc[ordinal, "close"], name="later close")
                record[column] = abs(math.log(later_close / entry_price))
            else:
                record[column] = math.nan
        lead_value = source.get("first_completion_lead")
        lead = None if lead_value is None or pd.isna(lead_value) else int(lead_value)
        record["registered_completion_in_bars_2_or_3"] = int(lead in {2, 3})
        if lead in {2, 3}:
            completion_close = closes[lead - 1]
            record["movement_before_completion"] = abs(math.log(completion_close / entry_price))
            record["movement_from_completion_to_horizon_end"] = abs(
                math.log(closes[2] / completion_close)
            )
        else:
            record["movement_before_completion"] = math.nan
            record["movement_from_completion_to_horizon_end"] = math.nan
        output_rows.append(record)
    return pd.DataFrame(output_rows)


def verify_structural_reconstruction(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> dict[str, int | float | bool]:
    """Compare frozen identities, route states, and numeric surfaces exactly."""

    required = {"row_id", "route_resolution_state", *feature_columns}
    for name, frame in (("reference", reference), ("reconstructed", reconstructed)):
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{name} structural panel missing columns: {missing}")
    left = reference.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    right = reconstructed.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    identities_left = left["row_id"].astype(str).tolist()
    identities_right = right["row_id"].astype(str).tolist()
    row_mismatches = abs(len(left) - len(right)) + sum(
        first != second for first, second in zip(identities_left, identities_right, strict=False)
    )
    if row_mismatches:
        return {
            "row_identity_mismatches": int(row_mismatches),
            "route_state_mismatches": int(row_mismatches),
            "maximum_difference": math.inf,
            "passed": False,
        }
    route_mismatches = int(
        left["route_resolution_state"]
        .astype(str)
        .ne(right["route_resolution_state"].astype(str))
        .sum()
    )
    if feature_columns:
        left_values = left.loc[:, list(feature_columns)].to_numpy(dtype=float)
        right_values = right.loc[:, list(feature_columns)].to_numpy(dtype=float)
        differences = np.abs(left_values - right_values)
        both_nan = np.isnan(left_values) & np.isnan(right_values)
        differences[both_nan] = 0.0
        maximum = float(np.max(differences)) if differences.size else 0.0
    else:
        maximum = 0.0
    return {
        "row_identity_mismatches": 0,
        "route_state_mismatches": route_mismatches,
        "maximum_difference": maximum,
        "passed": bool(route_mismatches == 0 and maximum <= 1e-12),
    }


__all__ = [
    "ANNUAL_TRADING_MINUTES",
    "ChronologyError",
    "DENSE_CHECKPOINTS",
    "DENSE_H0_FEATURES",
    "FROZEN_COHORT",
    "FrozenOptionsLinearModel",
    "OPTIONS_PRIMARY_FEATURES",
    "OPTIONS_SCREEN_DECISIONS",
    "OptionPairSelection",
    "PRIMARY_HORIZON_MINUTES",
    "ROUTE_FEATURES",
    "SAFETY_FLAGS",
    "add_iv_relative_outcomes",
    "assert_protected_boundary",
    "assign_development_frozen_iv_deciles",
    "broad_conflict_iv_gate_passes",
    "build_matched_control_relations",
    "calculate_optional_option_features",
    "calculate_primary_option_features",
    "choose_options_movement_decision",
    "compute_underlying_movement_outcomes",
    "coverage_gates_pass",
    "fit_options_linear_model",
    "fixed_session_bootstrap_multiplicities",
    "iv_movement_approximations",
    "o1_model_gate_passes",
    "permute_intact_route_bundle",
    "previous_trading_session",
    "select_primary_atm_pair",
    "split_boundary_is_ambiguous",
    "validate_exact_previous_session_join",
    "verify_structural_reconstruction",
]
