"""Pure helpers for the Stock/Options Cross-Market Quick Screen V0.

The module is retrospective research infrastructure.  It has no broker,
account, order, position, execution, portfolio, or deployment integration.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from stocker_research.broad_conflict_advance_hazard_v02 import (
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    ROUTE_FEATURES,
    candidate_normalized_weights,
)
from stocker_research.broad_conflict_options_iv_screen_v0 import (
    ANNUAL_TRADING_MINUTES,
    FROZEN_COHORT,
    calculate_optional_option_features,
    calculate_primary_option_features,
    compute_underlying_movement_outcomes,
    previous_trading_session,
    select_primary_atm_pair,
    validate_exact_previous_session_join,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DEVELOPMENT_START: Final[date] = date(2024, 1, 1)
DEVELOPMENT_END: Final[date] = date(2024, 12, 31)
ASSESSMENT_START: Final[date] = date(2025, 1, 1)
ASSESSMENT_END: Final[date] = date(2025, 8, 22)
PROTECTED_START: Final[date] = date(2025, 8, 23)
BOOTSTRAP_SEED: Final[int] = 20260722
OPTIONS_NULL_SEEDS: Final[tuple[int, int, int]] = (20260723, 20260724, 20260725)
ROUTE_NULL_SEEDS: Final[tuple[int, int, int]] = (20260726, 20260727, 20260728)

SAFETY_FLAGS: Final[dict[str, bool | str]] = {
    "research_only": True,
    "quick_cross_market_screen": True,
    "previous_close_options_only": True,
    "test_a_options_to_stock": True,
    "test_b_stock_to_options_movement": True,
    "intraday_option_quotes_used": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}

BASE_OPTIONS_FEATURES: Final[tuple[str, ...]] = (
    "atm_iv",
    "call_iv",
    "put_iv",
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
STOCK_RELATIVE_OPTIONS_FEATURES: Final[tuple[str, ...]] = (
    "atm_iv_stock_percentile",
    "atm_iv_stock_robust_z",
    "straddle_move_stock_robust_z",
    "skew_stock_robust_z",
    "term_structure_stock_robust_z",
)
REALIZED_VOLATILITY_FEATURES: Final[tuple[str, ...]] = (
    "realised_volatility_20d",
    "iv_minus_realised_20d",
)
OPTIONS_MODEL_FEATURES: Final[tuple[str, ...]] = (
    *BASE_OPTIONS_FEATURES,
    *STOCK_RELATIVE_OPTIONS_FEATURES,
    "iv_minus_realised_20d",
)
CROSS_MARKET_FEATURES: Final[tuple[str, ...]] = (
    "complacent_conflict",
    "structural_tension_gap",
    "route_vs_priced_move",
    "directional_agreement",
    "transition_vs_term_urgency",
)
ROUTE_STATE_LEVELS: Final[tuple[str, ...]] = (
    "BROAD_CONFLICT",
    "LOW_ROUTE_SUPPORT",
    "NARROWING",
    "OTHER",
)

TEST_A_S0_NUMERIC: Final[tuple[str, ...]] = (*DENSE_H0_FEATURES, *ROUTE_FEATURES)
TEST_A_S1_NUMERIC: Final[tuple[str, ...]] = (*TEST_A_S0_NUMERIC, *OPTIONS_MODEL_FEATURES)
TEST_A_S2_NUMERIC: Final[tuple[str, ...]] = (*TEST_A_S1_NUMERIC, *CROSS_MARKET_FEATURES)
TEST_B_O0_NUMERIC: Final[tuple[str, ...]] = OPTIONS_MODEL_FEATURES
TEST_B_O1_NUMERIC: Final[tuple[str, ...]] = (*TEST_B_O0_NUMERIC, *DENSE_H0_FEATURES)
TEST_B_O2_NUMERIC: Final[tuple[str, ...]] = (
    *TEST_B_O1_NUMERIC,
    *ROUTE_FEATURES,
    *CROSS_MARKET_FEATURES,
)
RIDGE_R0_NUMERIC: Final[tuple[str, ...]] = OPTIONS_MODEL_FEATURES
RIDGE_R1_NUMERIC: Final[tuple[str, ...]] = (
    *OPTIONS_MODEL_FEATURES,
    *DENSE_H0_FEATURES,
    *ROUTE_FEATURES,
    *CROSS_MARKET_FEATURES,
)

OVERALL_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "bidirectional_stock_options_information_supported",
        "options_improve_stock_method_only",
        "stock_structure_improves_options_forecast_only",
        "cross_market_disagreement_descriptive_only",
        "no_cross_market_increment",
        "blocked_insufficient_cached_options_coverage",
        "blocked_structural_panel_reconstruction_failure",
        "blocked_options_pair_reconstruction_failure",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
)
INDIVIDUAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"supported", "descriptive_only", "not_supported", "insufficient_support", "blocked"}
)


@dataclass(frozen=True)
class RobustStockScale:
    """One development-only empirical-CDF and robust-z fit."""

    sorted_values: tuple[float, ...]
    median: float
    scale: float


@dataclass(frozen=True)
class Standardization:
    """Development-only mean and population-standard-deviation fit."""

    mean: float
    scale: float


@dataclass(frozen=True)
class ExactHistoryExtraction:
    """Exact-date option observations decoded under the protected boundary."""

    records: tuple[dict[str, Any], ...]
    cached_records_scanned: int
    nonmatching_records_skipped: int
    protected_records_skipped_before_materialisation: int


@dataclass(frozen=True)
class FrozenCrossMarketModel:
    """Serializable deterministic development-fitted logistic or Ridge model."""

    model_id: str
    kind: str
    numeric_features: tuple[str, ...]
    category_controls: tuple[str, ...]
    numeric_medians: FloatArray
    numeric_means: FloatArray
    numeric_scales: FloatArray
    category_levels: Mapping[str, tuple[str, ...]]
    design_columns: tuple[str, ...]
    coefficients: FloatArray
    intercept: float
    iterations: int

    def design(self, frame: pd.DataFrame) -> FloatArray:
        """Apply the frozen numeric and categorical preprocessing."""

        missing = sorted(set(self.numeric_features).difference(frame.columns))
        if missing:
            raise ValueError(f"model frame missing numeric features: {missing}")
        raw = frame.loc[:, list(self.numeric_features)].to_numpy(dtype=float)
        values = np.where(np.isfinite(raw), raw, self.numeric_medians)
        parts: list[FloatArray] = [
            np.asarray((values - self.numeric_means) / self.numeric_scales, dtype=np.float64)
        ]
        controls = categorical_controls(frame)
        for control in self.category_controls:
            observed = controls[control].astype(str).to_numpy()
            levels = self.category_levels[control]
            for level in levels[1:]:
                parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
        design = np.concatenate(parts, axis=1)
        if design.shape[1] != len(self.design_columns):
            raise AssertionError("frozen model design width drifted")
        return np.asarray(design, dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        """Return probabilities for logistic models and values for Ridge."""

        linear = self.design(frame) @ self.coefficients + self.intercept
        if self.kind == "ridge":
            return np.asarray(linear, dtype=np.float64)
        if self.kind != "logistic":
            raise ValueError(f"unknown model kind: {self.kind}")
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))), dtype=np.float64)

    def as_dict(self) -> dict[str, object]:
        """Return the complete manual-reconstruction surface."""

        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "numeric_features": list(self.numeric_features),
            "category_controls": list(self.category_controls),
            "numeric_medians": self.numeric_medians.astype(float).tolist(),
            "numeric_means": self.numeric_means.astype(float).tolist(),
            "numeric_scales": self.numeric_scales.astype(float).tolist(),
            "category_levels": {key: list(values) for key, values in self.category_levels.items()},
            "design_columns": list(self.design_columns),
            "coefficients": self.coefficients.astype(float).tolist(),
            "intercept": float(self.intercept),
            "iterations": int(self.iterations),
            "preprocessing_fitted_period": "development_2024_only",
            "penalty": "l2" if self.kind == "logistic" else None,
            "C": 0.25 if self.kind == "logistic" else None,
            "solver": "liblinear" if self.kind == "logistic" else "cholesky",
            "max_iter": 300 if self.kind == "logistic" else None,
            "class_weight": None,
            "ridge_alpha": 10.0 if self.kind == "ridge" else None,
            "n_jobs": 1,
        }


def assert_safety_flags(value: Mapping[str, object]) -> None:
    """Fail closed unless every required research-only flag is exact."""

    mismatches = {
        key: (expected, value.get(key))
        for key, expected in SAFETY_FLAGS.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"cross-market safety flags differ: {mismatches}")


def assert_protected_dates(frame: pd.DataFrame, *, columns: Sequence[str]) -> None:
    """Reject any materialised market or option date at/after 2025-08-23."""

    for column in columns:
        if column not in frame:
            raise ValueError(f"protected-boundary column missing: {column}")
        values = pd.to_datetime(frame[column], errors="raise").dt.date
        if bool(values.ge(PROTECTED_START).any()):
            raise ValueError(f"protected date materialised in {column}")


def _json_data_object_slices(raw_json: str) -> Iterator[str]:
    """Yield raw object slices from a top-level data array without decoding them."""

    marker = re.search(r'"data"\s*:\s*\[', raw_json)
    if marker is None:
        raise ValueError("cached provider JSON has no data array")
    position = marker.end()
    length = len(raw_json)
    while position < length:
        while position < length and raw_json[position] in " \t\r\n,":
            position += 1
        if position >= length:
            raise ValueError("cached provider data array is unterminated")
        if raw_json[position] == "]":
            return
        if raw_json[position] != "{":
            raise ValueError("cached provider data array contains a non-object")
        start = position
        depth = 0
        in_string = False
        escaped = False
        while position < length:
            character = raw_json[position]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    position += 1
                    yield raw_json[start:position]
                    break
            position += 1
        else:
            raise ValueError("cached provider data object is unterminated")


def extract_exact_history_records(
    raw_json: str,
    *,
    required_date: date,
) -> ExactHistoryExtraction:
    """Decode only the required prior-close rows, skipping protected rows as text."""

    if required_date >= PROTECTED_START:
        raise ValueError("required option date crosses protected boundary")
    records: list[dict[str, Any]] = []
    cached_records = 0
    protected_skipped = 0
    nonmatching_skipped = 0
    for raw_object in _json_data_object_slices(raw_json):
        matches = re.findall(r'"id"\s*:\s*"([^"]+)"', raw_object)
        if len(matches) != 1:
            raise ValueError("cached option history row has ambiguous resource id")
        try:
            observation_date = date.fromisoformat(matches[0][-10:])
        except ValueError as error:
            raise ValueError(
                "cached option history row has invalid observation-date identity"
            ) from error
        cached_records += 1
        if observation_date >= PROTECTED_START:
            protected_skipped += 1
            continue
        if observation_date != required_date:
            nonmatching_skipped += 1
            continue
        decoded = json.loads(raw_object)
        if not isinstance(decoded, dict):
            raise ValueError("cached option history row is not an object")
        records.append(cast(dict[str, Any], decoded))
    return ExactHistoryExtraction(
        records=tuple(records),
        cached_records_scanned=cached_records,
        nonmatching_records_skipped=nonmatching_skipped,
        protected_records_skipped_before_materialisation=protected_skipped,
    )


def reconstruct_clean_structural_panel(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Re-derive eligibility, target, weights, and indicators from the frozen panel."""

    required = {
        "row_id",
        "symbol",
        "session",
        "period",
        "checkpoint",
        "registered_completion_next_1_bar",
        "any_prefix_one_transition_from_completion",
        "first_completion_lead",
        "completion_in_bars_2_or_3",
        "advance_eligible",
        "route_resolution_state",
        "row_weight",
        *DENSE_H0_FEATURES,
        *ROUTE_FEATURES,
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"frozen structural panel missing columns: {missing}")
    assert_protected_dates(source, columns=("session",))
    eligibility = source["registered_completion_next_1_bar"].fillna(0).astype(int).eq(0) & source[
        "any_prefix_one_transition_from_completion"
    ].fillna(0).astype(int).eq(0)
    target = pd.to_numeric(source["first_completion_lead"], errors="raise").isin([2, 3]).astype(int)
    eligibility_mismatches = int(eligibility.ne(source["advance_eligible"].astype(int).eq(1)).sum())
    target_mismatches = int(target.ne(source["completion_in_bars_2_or_3"].astype(int)).sum())
    reference = source.loc[source["advance_eligible"].astype(int).eq(1)].copy()
    reconstructed = source.loc[eligibility].copy()
    reconstructed["completion_in_bars_2_or_3"] = target.loc[eligibility].to_numpy(int)
    reconstructed = candidate_normalized_weights(reconstructed)
    reference = reference.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    reconstructed = reconstructed.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    reference_ids = reference["row_id"].astype(str).tolist()
    reconstructed_ids = reconstructed["row_id"].astype(str).tolist()
    row_identity_mismatches = abs(len(reference_ids) - len(reconstructed_ids)) + sum(
        left != right for left, right in zip(reference_ids, reconstructed_ids, strict=False)
    )
    route_state_mismatches = (
        row_identity_mismatches
        if row_identity_mismatches
        else int(
            reference["route_resolution_state"]
            .astype(str)
            .ne(reconstructed["route_resolution_state"].astype(str))
            .sum()
        )
    )
    shared_features = (*DENSE_H0_FEATURES, *ROUTE_FEATURES, "row_weight")
    if row_identity_mismatches:
        maximum_difference = math.inf
    else:
        maximum_difference = float(
            np.max(
                np.abs(
                    reference.loc[:, list(shared_features)].to_numpy(float)
                    - reconstructed.loc[:, list(shared_features)].to_numpy(float)
                )
            )
        )
    reconstruction: dict[str, object] = {
        **SAFETY_FLAGS,
        "source_rows": int(len(source)),
        "clean_advance_rows": int(len(reference)),
        "development_clean_rows": int(reference["period"].eq("development").sum()),
        "assessment_clean_rows": int(reference["period"].eq("assessment").sum()),
        "assessment_clean_positives": int(
            reference.loc[reference["period"].eq("assessment"), "completion_in_bars_2_or_3"].sum()
        ),
        "row_identity_mismatches": int(row_identity_mismatches),
        "route_state_mismatches": int(route_state_mismatches),
        "eligibility_mismatches": int(eligibility_mismatches),
        "target_mismatches": int(target_mismatches),
        "maximum_shared_feature_difference": maximum_difference,
        "checkpoints": list(DENSE_CHECKPOINTS),
        "frozen_h0_features": list(DENSE_H0_FEATURES),
        "frozen_route_features": list(ROUTE_FEATURES),
        "passed": bool(
            row_identity_mismatches == 0
            and route_state_mismatches == 0
            and eligibility_mismatches == 0
            and target_mismatches == 0
            and maximum_difference <= 1e-12
        ),
    }
    if not reconstruction["passed"]:
        raise RuntimeError("blocked_structural_panel_reconstruction_failure")
    reconstructed["registered_completion_clean_bars_2_or_3"] = reconstructed[
        "completion_in_bars_2_or_3"
    ].astype(int)
    reconstructed["BROAD_CONFLICT"] = (
        reconstructed["route_resolution_state"].eq("BROAD_CONFLICT").astype(int)
    )
    reconstructed["LOW_ROUTE_SUPPORT"] = (
        reconstructed["route_resolution_state"].eq("LOW_ROUTE_SUPPORT").astype(int)
    )
    return reconstructed, reconstruction


def trailing_realised_volatility_20d(bars: pd.DataFrame) -> pd.DataFrame:
    """Calculate close-to-close realised volatility known at each session close."""

    required = {"symbol", "session", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"realised-volatility bars missing columns: {missing}")
    working = bars.loc[:, ["symbol", "session", "close"]].copy()
    working["session"] = pd.to_datetime(working["session"], errors="raise").dt.date
    working["close"] = pd.to_numeric(working["close"], errors="raise")
    if not np.isfinite(working["close"].to_numpy(float)).all() or bool(
        working["close"].le(0.0).any()
    ):
        raise ValueError("daily closes must be finite and positive")
    daily = (
        working.sort_values(["symbol", "session"], kind="mergesort")
        .groupby(["symbol", "session"], sort=True, as_index=False)
        .agg(close=("close", "last"))
    )
    daily["log_return"] = daily.groupby("symbol", sort=False)["close"].transform(
        lambda values: np.log(values).diff()
    )
    daily["realised_volatility_20d"] = daily.groupby("symbol", sort=False)["log_return"].transform(
        lambda values: values.rolling(20, min_periods=15).std(ddof=1) * math.sqrt(252.0)
    )
    daily["valid_trailing_return_sessions"] = daily.groupby("symbol", sort=False)[
        "log_return"
    ].transform(lambda values: values.rolling(20, min_periods=1).count())
    return daily.rename(columns={"session": "required_options_date"})[
        [
            "symbol",
            "required_options_date",
            "realised_volatility_20d",
            "valid_trailing_return_sessions",
        ]
    ]


def _robust_scale(values: FloatArray) -> tuple[float, float]:
    median = float(np.median(values))
    mad_scale = float(np.median(np.abs(values - median)) * 1.4826)
    if mad_scale >= 1e-12:
        return median, mad_scale
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    return median, iqr_scale if iqr_scale >= 1e-12 else 1.0


def fit_stock_relative_options(
    development: pd.DataFrame,
) -> dict[str, dict[str, RobustStockScale]]:
    """Fit every stock-specific percentile/robust-z parameter on 2024 only."""

    required = {"symbol", "session", "atm_iv", "straddle_mid_pct", "skew_25d", "term_structure"}
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"stock-relative development surface missing: {missing}")
    sessions = pd.to_datetime(development["session"], errors="raise")
    if not sessions.dt.year.eq(2024).all():
        raise ValueError("stock-relative scaling must fit on 2024 only")
    unique = development.drop_duplicates(["symbol", "session"], keep="first")
    parameters: dict[str, dict[str, RobustStockScale]] = {}
    for symbol, group in unique.groupby("symbol", sort=True):
        symbol_parameters: dict[str, RobustStockScale] = {}
        for column in ("atm_iv", "straddle_mid_pct", "skew_25d", "term_structure"):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            finite = np.sort(values[np.isfinite(values)])
            if finite.size == 0:
                continue
            median, scale = _robust_scale(np.asarray(finite, dtype=np.float64))
            symbol_parameters[column] = RobustStockScale(
                sorted_values=tuple(float(value) for value in finite),
                median=median,
                scale=scale,
            )
        parameters[str(symbol)] = symbol_parameters
    return parameters


def apply_stock_relative_options(
    frame: pd.DataFrame,
    parameters: Mapping[str, Mapping[str, RobustStockScale]],
) -> pd.DataFrame:
    """Apply development-frozen stock-specific empirical and robust transforms."""

    output = frame.copy()
    percentile: list[float] = []
    robust_columns: dict[str, list[float]] = {
        "atm_iv_stock_robust_z": [],
        "straddle_move_stock_robust_z": [],
        "skew_stock_robust_z": [],
        "term_structure_stock_robust_z": [],
    }
    mapping = {
        "atm_iv_stock_robust_z": "atm_iv",
        "straddle_move_stock_robust_z": "straddle_mid_pct",
        "skew_stock_robust_z": "skew_25d",
        "term_structure_stock_robust_z": "term_structure",
    }
    for row in output.itertuples(index=False):
        symbol = str(cast(Any, row).symbol)
        stock = parameters.get(symbol, {})
        atm_value = float(cast(Any, row).atm_iv)
        atm_parameters = stock.get("atm_iv")
        if atm_parameters is None or not math.isfinite(atm_value):
            percentile.append(math.nan)
        else:
            sorted_values = np.asarray(atm_parameters.sorted_values, dtype=float)
            percentile.append(
                float(np.searchsorted(sorted_values, atm_value, side="right") / len(sorted_values))
            )
        for target, source in mapping.items():
            value = float(getattr(cast(Any, row), source))
            fitted = stock.get(source)
            robust_columns[target].append(
                0.0
                if fitted is None or not math.isfinite(value)
                else (value - fitted.median) / fitted.scale
            )
    output["atm_iv_stock_percentile"] = percentile
    for column, values in robust_columns.items():
        output[column] = values
    return output


_STANDARDIZED_SOURCE: Final[dict[str, str]] = {
    "standardised_tension": "tension",
    "standardised_prefix_family_entropy": "prefix_family_entropy",
    "standardised_signed_pressure": "signed_pressure",
    "standardised_call_put_iv_gap": "call_put_iv_gap",
    "standardised_transition_probability": "transition_probability",
}


def fit_cross_market_standardization(
    development: pd.DataFrame,
) -> dict[str, Standardization]:
    """Fit the five fixed disagreement-input standardizations on 2024 only."""

    sessions = pd.to_datetime(development["session"], errors="raise")
    if not sessions.dt.year.eq(2024).all():
        raise ValueError("cross-market standardization must fit on 2024 only")
    result: dict[str, Standardization] = {}
    for target, source in _STANDARDIZED_SOURCE.items():
        if source not in development:
            raise ValueError(f"cross-market standardization source missing: {source}")
        values = pd.to_numeric(development[source], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"cross-market standardization values not finite: {source}")
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        result[target] = Standardization(mean=mean, scale=scale if scale >= 1e-12 else 1.0)
    return result


def add_cross_market_disagreement(
    frame: pd.DataFrame,
    parameters: Mapping[str, Standardization],
) -> pd.DataFrame:
    """Create exactly the five fixed cross-market disagreement features."""

    output = frame.copy()
    for target, source in _STANDARDIZED_SOURCE.items():
        if target not in parameters:
            raise ValueError(f"cross-market standardization missing: {target}")
        fitted = parameters[target]
        output[target] = (
            pd.to_numeric(output[source], errors="raise") - fitted.mean
        ) / fitted.scale
    if "BROAD_CONFLICT" not in output:
        output["BROAD_CONFLICT"] = output["route_resolution_state"].eq("BROAD_CONFLICT").astype(int)
    output["complacent_conflict"] = output["BROAD_CONFLICT"].astype(float) * (
        -pd.to_numeric(output["atm_iv_stock_robust_z"], errors="coerce")
    )
    output["structural_tension_gap"] = output["standardised_tension"] - pd.to_numeric(
        output["atm_iv_stock_robust_z"], errors="coerce"
    )
    output["route_vs_priced_move"] = output["standardised_prefix_family_entropy"] - pd.to_numeric(
        output["straddle_move_stock_robust_z"], errors="coerce"
    )
    output["directional_agreement"] = (
        output["standardised_signed_pressure"] * output["standardised_call_put_iv_gap"]
    )
    output["transition_vs_term_urgency"] = output[
        "standardised_transition_probability"
    ] - pd.to_numeric(output["term_structure_stock_robust_z"], errors="coerce")
    return output


def add_test_a_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen clean two-to-three-bar completion eligibility and target."""

    required = {
        "first_completion_lead",
        "registered_completion_next_1_bar",
        "any_prefix_one_transition_from_completion",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Test A target inputs missing: {missing}")
    output = frame.copy()
    lead = pd.to_numeric(output["first_completion_lead"], errors="raise")
    eligible = output["registered_completion_next_1_bar"].fillna(0).astype(int).eq(0) & output[
        "any_prefix_one_transition_from_completion"
    ].fillna(0).astype(int).eq(0)
    output["clean_advance_eligible"] = eligible.astype(int)
    output["registered_completion_clean_bars_2_or_3"] = np.where(
        eligible,
        lead.isin([2, 3]).astype(float),
        np.nan,
    )
    return output


def add_test_b_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the prior-close-IV 15-minute binary and continuous movement targets."""

    required = {"absolute_log_return_15m", "atm_iv"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Test B target inputs missing: {missing}")
    output = frame.copy()
    movement = pd.to_numeric(output["absolute_log_return_15m"], errors="raise")
    iv = pd.to_numeric(output["atm_iv"], errors="raise")
    if (
        not np.isfinite(movement.to_numpy(float)).all()
        or bool(movement.lt(0.0).any())
        or not np.isfinite(iv.to_numpy(float)).all()
        or bool(iv.le(0.0).any())
    ):
        raise ValueError("Test B target inputs must be finite and in range")
    output["iv_sigma_15m"] = iv * math.sqrt(15.0 / ANNUAL_TRADING_MINUTES)
    output["iv_expected_absolute_15m"] = output["iv_sigma_15m"] * math.sqrt(2.0 / math.pi)
    output["movement_exceeds_prior_close_iv"] = movement.gt(
        output["iv_expected_absolute_15m"]
    ).astype(int)
    output["iv_absolute_residual_15m"] = movement - output["iv_expected_absolute_15m"]
    return output


def categorical_controls(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the frozen stock/checkpoint/month/route categorical controls."""

    required = {"symbol", "checkpoint", "session"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"model categorical controls missing: {missing}")
    session = pd.to_datetime(frame["session"], errors="raise")
    route = (
        frame["route_resolution_state"].astype(str)
        if "route_resolution_state" in frame
        else pd.Series("OTHER", index=frame.index, dtype="string")
    )
    return {
        "stock": frame["symbol"].astype(str),
        "checkpoint": frame["checkpoint"].astype(int).astype(str),
        "month_of_year": session.dt.strftime("%m"),
        "route_state": route,
    }


def fit_cross_market_model(
    development: pd.DataFrame,
    *,
    model_id: str,
    numeric_features: Sequence[str],
    category_control_names: Sequence[str],
    target_column: str,
    kind: str,
) -> FrozenCrossMarketModel:
    """Fit one fixed deterministic weighted logistic or Ridge model."""

    if "period" in development and not development["period"].astype(str).eq("development").all():
        raise ValueError("model fitting accepts development rows only")
    sessions = pd.to_datetime(development["session"], errors="raise")
    if not sessions.dt.year.eq(2024).all():
        raise ValueError("model preprocessing and fitting are frozen to 2024")
    features = tuple(numeric_features)
    if not features or len(set(features)) != len(features):
        raise ValueError("numeric feature surface must be non-empty and unique")
    missing = sorted({target_column, "row_weight", *features}.difference(development.columns))
    if missing:
        raise ValueError(f"model development frame missing columns: {missing}")
    raw = development.loc[:, list(features)].to_numpy(dtype=float)
    finite = np.where(np.isfinite(raw), raw, np.nan)
    medians = np.nanmedian(finite, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("every numeric feature needs finite development support")
    values = np.where(np.isfinite(raw), raw, medians)
    means = np.asarray(values.mean(axis=0), dtype=np.float64)
    scales = np.asarray(values.std(axis=0, ddof=0), dtype=np.float64)
    scales = np.where(scales >= 1e-12, scales, 1.0)
    parts: list[FloatArray] = [np.asarray((values - means) / scales, dtype=np.float64)]
    design_columns = list(features)
    all_controls = categorical_controls(development)
    controls = tuple(category_control_names)
    if unknown := sorted(set(controls).difference(all_controls)):
        raise ValueError(f"unknown categorical controls: {unknown}")
    levels: dict[str, tuple[str, ...]] = {}
    for control in controls:
        observed = all_controls[control].astype(str).to_numpy()
        control_levels = tuple(sorted(set(observed)))
        levels[control] = control_levels
        for level in control_levels[1:]:
            parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
            design_columns.append(f"control_{control}__{level}")
    design = np.concatenate(parts, axis=1)
    weights = pd.to_numeric(development["row_weight"], errors="raise").to_numpy(float)
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("model weights must be finite and positive")
    if kind == "logistic":
        target = pd.to_numeric(development[target_column], errors="raise").to_numpy(int)
        if set(target) != {0, 1}:
            raise ValueError(f"{model_id} requires both target classes")
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
            estimator.fit(design, target, sample_weight=weights)
        iterations = int(np.max(estimator.n_iter_))
        if iterations >= 300:
            raise RuntimeError("blocked_model_convergence_failure")
        coefficients = np.asarray(estimator.coef_[0], dtype=np.float64)
        intercept = float(estimator.intercept_[0])
    elif kind == "ridge":
        target_values = pd.to_numeric(development[target_column], errors="raise").to_numpy(float)
        if not np.isfinite(target_values).all():
            raise ValueError(f"{model_id} Ridge target must be finite")
        ridge = Ridge(alpha=10.0, fit_intercept=True, solver="cholesky")
        ridge.fit(design, target_values, sample_weight=weights)
        iterations = 1
        coefficients = np.asarray(ridge.coef_, dtype=np.float64)
        intercept = float(ridge.intercept_)
    else:
        raise ValueError("model kind must be logistic or ridge")
    return FrozenCrossMarketModel(
        model_id=model_id,
        kind=kind,
        numeric_features=features,
        category_controls=controls,
        numeric_medians=np.asarray(medians, dtype=np.float64),
        numeric_means=means,
        numeric_scales=scales,
        category_levels=levels,
        design_columns=tuple(design_columns),
        coefficients=coefficients,
        intercept=intercept,
        iterations=iterations,
    )


def manual_model_prediction(frame: pd.DataFrame, specification: Mapping[str, object]) -> FloatArray:
    """Independently reconstruct a serialized model's predictions."""

    features = tuple(
        str(value) for value in cast(Sequence[object], specification["numeric_features"])
    )
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    raw = frame.loc[:, list(features)].to_numpy(float)
    values = np.where(np.isfinite(raw), raw, medians)
    parts: list[FloatArray] = [np.asarray((values - means) / scales, dtype=np.float64)]
    controls = categorical_controls(frame)
    level_mapping = cast(Mapping[str, Sequence[object]], specification["category_levels"])
    for control_value in cast(Sequence[object], specification["category_controls"]):
        control = str(control_value)
        observed = controls[control].astype(str).to_numpy()
        levels = tuple(str(value) for value in level_mapping[control])
        for level in levels[1:]:
            parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
    design = np.concatenate(parts, axis=1)
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    intercept = float(cast(Any, specification["intercept"]))
    linear = design @ coefficients + intercept
    if specification["kind"] == "ridge":
        return np.asarray(linear, dtype=np.float64)
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))), dtype=np.float64)


def _calibration_fit(
    target: IntArray, probabilities: FloatArray, weights: FloatArray
) -> tuple[float, float]:
    if set(target) != {0, 1}:
        return math.nan, math.nan
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(len(target), dtype=float), logits])
    beta = np.asarray([0.0, 1.0], dtype=float)
    for _ in range(50):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35.0, 35.0)))
        gradient = design.T @ (weights * (target - fitted))
        curvature = weights * fitted * (1.0 - fitted)
        information = design.T @ (curvature[:, None] * design) + np.eye(2) * 1e-12
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        beta += step
        if float(np.max(np.abs(step))) <= 1e-10:
            break
    return float(beta[0]), float(beta[1])


def probability_quantile_boundaries(probabilities: Sequence[float]) -> dict[str, float]:
    """Fit unweighted development probability cutoffs once."""

    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("probability quantiles require finite development predictions")
    return {
        "top_decile": float(np.quantile(values, 0.90)),
        "top_quintile": float(np.quantile(values, 0.80)),
    }


def binary_metrics(
    frame: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
    boundaries: Mapping[str, float],
) -> dict[str, float | int]:
    """Calculate the complete weighted binary screen metric surface."""

    target = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=np.int64)
    probabilities = pd.to_numeric(frame[probability_column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if not (
        len(target) > 0
        and len(target) == len(probabilities) == len(weights)
        and np.isfinite(probabilities).all()
        and np.isfinite(weights).all()
        and bool((weights > 0.0).all())
        and bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
    ):
        raise ValueError("binary metrics require aligned finite inputs")
    total_weight = float(weights.sum())
    base_rate = float(np.sum(weights * target) / total_weight)
    brier = float(np.sum(weights * np.square(probabilities - target)) / total_weight)
    realised = np.where(target == 1, probabilities, 1.0 - probabilities)
    if set(target) == {0, 1}:
        auc = float(roc_auc_score(target, probabilities, sample_weight=weights))
        average_precision = float(
            average_precision_score(target, probabilities, sample_weight=weights)
        )
    else:
        auc = math.nan
        average_precision = math.nan
    intercept, slope = _calibration_fit(target, probabilities, weights)
    bin_edges = np.quantile(probabilities, np.linspace(0.0, 1.0, 11))
    bin_id = np.searchsorted(bin_edges[1:-1], probabilities, side="right")
    ece = 0.0
    for value in range(10):
        mask = bin_id == value
        if not bool(mask.any()):
            continue
        bin_weight = weights[mask]
        bin_total = float(bin_weight.sum())
        predicted = float(np.sum(bin_weight * probabilities[mask]) / bin_total)
        observed = float(np.sum(bin_weight * target[mask]) / bin_total)
        ece += bin_total / total_weight * abs(predicted - observed)

    def top_metrics(name: str) -> tuple[float, float]:
        threshold = float(boundaries[name])
        selected = probabilities >= threshold
        if not bool(selected.any()):
            return math.nan, math.nan
        selected_rate = float(
            np.sum(weights[selected] * target[selected]) / weights[selected].sum()
        )
        return selected_rate, selected_rate / base_rate if base_rate > 0.0 else math.nan

    decile_precision, decile_lift = top_metrics("top_decile")
    quintile_precision, quintile_lift = top_metrics("top_quintile")
    return {
        "log_loss": float(log_loss(target, probabilities, sample_weight=weights, labels=[0, 1])),
        "brier_score": brier,
        "auc": auc,
        "average_precision": average_precision,
        "expected_calibration_error": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "base_rate": base_rate,
        "mean_probability_realised_class": float(np.sum(weights * realised) / total_weight),
        "top_decile_precision": decile_precision,
        "top_decile_lift": decile_lift,
        "top_quintile_precision": quintile_precision,
        "top_quintile_lift": quintile_lift,
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "positive_outcomes": int(target.sum()),
    }


def continuous_residual_metrics(
    frame: pd.DataFrame, *, target_column: str, prediction_column: str
) -> dict[str, float | int]:
    """Calculate weighted MAE, RMSE, and R-squared for the Ridge diagnostic."""

    target = pd.to_numeric(frame[target_column], errors="raise").to_numpy(float)
    prediction = pd.to_numeric(frame[prediction_column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if not (
        len(target) > 0
        and np.isfinite(target).all()
        and np.isfinite(prediction).all()
        and np.isfinite(weights).all()
        and bool((weights > 0.0).all())
    ):
        raise ValueError("continuous metrics require finite aligned inputs")
    residual = target - prediction
    total = float(weights.sum())
    mae = float(np.sum(weights * np.abs(residual)) / total)
    rmse = float(math.sqrt(np.sum(weights * np.square(residual)) / total))
    mean = float(np.sum(weights * target) / total)
    denominator = float(np.sum(weights * np.square(target - mean)))
    r_squared = (
        math.nan
        if denominator <= 0.0
        else 1.0 - float(np.sum(weights * np.square(residual))) / denominator
    )
    return {"weighted_mae": mae, "weighted_rmse": rmse, "weighted_r_squared": r_squared}


def fixed_session_bootstrap_multiplicities(
    sessions: pd.Series, *, draws: int = 10, seed: int = BOOTSTRAP_SEED
) -> list[IntArray]:
    """Return exactly ten fixed-seed whole-session multiplicity vectors."""

    if draws != 10:
        raise ValueError("cross-market bootstrap requires exactly 10 draws")
    labels = sessions.astype(str).to_numpy()
    unique = np.asarray(sorted(set(labels)), dtype=object)
    if unique.size == 0:
        raise ValueError("bootstrap sessions are empty")
    rng = np.random.default_rng(seed)
    output: list[IntArray] = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        output.append(np.asarray([int(counts.get(value, 0)) for value in labels], dtype=np.int64))
    return output


def permute_options_bundle(
    frame: pd.DataFrame,
    *,
    seed: int,
    standardization: Mapping[str, Standardization],
) -> pd.DataFrame:
    """Permute the complete options bundle and rebuild disagreement features."""

    permuted = _permute_bundle(
        frame,
        columns=OPTIONS_MODEL_FEATURES,
        seed=seed,
        include_route_state=False,
    )
    return add_cross_market_disagreement(permuted, standardization)


def permute_route_bundle_and_state(
    frame: pd.DataFrame,
    *,
    seed: int,
    standardization: Mapping[str, Standardization],
) -> pd.DataFrame:
    """Permute route/state as one bundle and rebuild dependent disagreements."""

    permuted = _permute_bundle(
        frame,
        columns=(*ROUTE_FEATURES, "route_resolution_state"),
        seed=seed,
        include_route_state=True,
    )
    return add_cross_market_disagreement(permuted, standardization)


def _permute_bundle(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    seed: int,
    include_route_state: bool,
) -> pd.DataFrame:
    strata = ("period", "session", "checkpoint")
    required = {*strata, "symbol", *columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"null-permutation panel missing columns: {missing}")
    if frame.duplicated([*strata, "symbol"]).any():
        raise ValueError("null permutation requires one stock row per slate")
    output = frame.copy()
    rng = np.random.default_rng(seed)
    for _key, slate in frame.groupby(list(strata), sort=True, observed=True):
        indices = slate.sort_values("symbol", kind="mergesort").index.to_numpy()
        sources = rng.permutation(indices)
        output.loc[indices, list(columns)] = frame.loc[sources, list(columns)].to_numpy()
    if include_route_state:
        output["BROAD_CONFLICT"] = output["route_resolution_state"].eq("BROAD_CONFLICT").astype(int)
        output["LOW_ROUTE_SUPPORT"] = (
            output["route_resolution_state"].eq("LOW_ROUTE_SUPPORT").astype(int)
        )
    return output


def _weighted_quantile(values: FloatArray, weights: FloatArray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, quantile * weights.sum(), side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def route_state_movement_metrics(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Summarize the four binding Test B route-state outcome groups."""

    required = {
        "route_resolution_state",
        "absolute_log_return_15m",
        "iv_expected_absolute_15m",
        "iv_absolute_residual_15m",
        "movement_exceeds_prior_close_iv",
        "row_weight",
        "session",
        "symbol",
    }
    missing = sorted(required.difference(assessment.columns))
    if missing:
        raise ValueError(f"route-state metrics missing columns: {missing}")
    working = assessment.copy()
    working["route_state_group"] = np.where(
        working["route_resolution_state"].isin(
            ["BROAD_CONFLICT", "LOW_ROUTE_SUPPORT", "NARROWING"]
        ),
        working["route_resolution_state"],
        "OTHER",
    )
    rows: list[dict[str, float | int | str]] = []
    for state in ROUTE_STATE_LEVELS:
        group = working.loc[working["route_state_group"].eq(state)]
        if group.empty:
            rows.append({"route_state": state, "rows": 0, "sessions": 0, "stocks": 0})
            continue
        weights = pd.to_numeric(group["row_weight"], errors="raise").to_numpy(float)
        movement = pd.to_numeric(group["absolute_log_return_15m"], errors="raise").to_numpy(float)
        expected = pd.to_numeric(group["iv_expected_absolute_15m"], errors="raise").to_numpy(float)
        residual = pd.to_numeric(group["iv_absolute_residual_15m"], errors="raise").to_numpy(float)
        exceeds = pd.to_numeric(group["movement_exceeds_prior_close_iv"], errors="raise").to_numpy(
            float
        )
        total = float(weights.sum())
        positive = np.maximum(residual, 0.0)
        positive_total = float(np.sum(weights * positive))
        order = np.argsort(residual, kind="mergesort")[::-1]
        top_rows = max(1, math.ceil(0.05 * len(group)))
        top_indices = order[:top_rows]
        top_contribution = (
            0.0
            if positive_total <= 0.0
            else float(np.sum(weights[top_indices] * positive[top_indices]) / positive_total)
        )
        rows.append(
            {
                "route_state": state,
                "rows": int(len(group)),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "mean_absolute_movement": float(np.sum(weights * movement) / total),
                "median_absolute_movement": _weighted_quantile(movement, weights, 0.5),
                "mean_iv_expected_movement": float(np.sum(weights * expected) / total),
                "mean_iv_residual": float(np.sum(weights * residual) / total),
                "median_iv_residual": _weighted_quantile(residual, weights, 0.5),
                "iv_sigma_ratio": float(
                    np.sum(
                        weights
                        * movement
                        / (
                            pd.to_numeric(group["atm_iv"], errors="raise").to_numpy(float)
                            * math.sqrt(15.0 / ANNUAL_TRADING_MINUTES)
                        )
                    )
                    / total
                ),
                "exceed_iv_rate": float(np.sum(weights * exceeds) / total),
                "upper_decile_absolute_movement": _weighted_quantile(movement, weights, 0.9),
                "upper_decile_iv_residual": _weighted_quantile(residual, weights, 0.9),
                "top_5pct_positive_residual_contribution": top_contribution,
            }
        )
    table = pd.DataFrame(rows)
    broad = table.loc[table["route_state"].eq("BROAD_CONFLICT")]
    low = table.loc[table["route_state"].eq("LOW_ROUTE_SUPPORT")]
    contrast: dict[str, float] = {}
    if (
        not broad.empty
        and not low.empty
        and int(broad.iloc[0]["rows"])
        and int(low.iloc[0]["rows"])
    ):
        for column in (
            "mean_iv_residual",
            "median_iv_residual",
            "exceed_iv_rate",
            "upper_decile_iv_residual",
        ):
            contrast[f"{column}_difference"] = float(broad.iloc[0][column]) - float(
                low.iloc[0][column]
            )
    return table, contrast


def coverage_gates(evidence: Mapping[str, object]) -> tuple[bool, dict[str, bool]]:
    """Apply the quick-screen support gates without relaxation."""

    gates = {
        "assessment_rows": int(cast(Any, evidence["assessment_rows"])) >= 1_500,
        "assessment_sessions": int(cast(Any, evidence["assessment_sessions"])) >= 50,
        "assessment_stocks": int(cast(Any, evidence["assessment_stocks"])) >= 10,
        "assessment_months": int(cast(Any, evidence["assessment_months"])) >= 5,
        "test_a_positives": int(cast(Any, evidence["test_a_positives"])) >= 100,
        "test_b_positives": int(cast(Any, evidence["test_b_positives"])) >= 300,
        "broad_conflict_rows": int(cast(Any, evidence["broad_conflict_rows"])) >= 200,
        "low_route_support_rows": int(cast(Any, evidence["low_route_support_rows"])) >= 200,
        "maximum_stock_weight_share": float(cast(Any, evidence["maximum_stock_weight_share"]))
        <= 0.15,
        "exact_pair_row_coverage": float(cast(Any, evidence["exact_pair_row_coverage"])) >= 0.50,
    }
    return all(gates.values()), gates


def test_a_options_increment_passes(gates: Mapping[str, object]) -> bool:
    """Apply the exact S1-versus-S0 pass gates."""

    return bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and int(cast(Any, gates["positive_months"])) >= 4
        and bool(gates["real_log_loss_or_brier_exceeds_all_nulls"])
    )


def test_a_disagreement_increment_passes(gates: Mapping[str, object]) -> bool:
    """Apply the exact S2-versus-S1 pass gates."""

    return bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["average_precision_improvement"])) > 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and bool(gates["real_proper_score_exceeds_all_nulls"])
    )


def test_b_stock_increment_passes(gates: Mapping[str, object]) -> bool:
    """Apply the exact O1-versus-O0 pass gates."""

    return bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and int(cast(Any, gates["positive_months"])) >= 4
    )


def test_b_route_increment_passes(gates: Mapping[str, object]) -> bool:
    """Apply the exact O2-versus-O1 pass gates."""

    return bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["average_precision_improvement"])) > 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and bool(gates["real_proper_score_exceeds_all_nulls"])
    )


def choose_overall_decision(
    *,
    blocker: str | None,
    test_a_supported: bool,
    test_b_supported: bool,
    disagreement_descriptive: bool,
) -> str:
    """Choose exactly one preregistered overall cross-market decision."""

    if blocker is not None:
        if blocker not in OVERALL_DECISIONS or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown cross-market blocker: {blocker}")
        return blocker
    if test_a_supported and test_b_supported:
        return "bidirectional_stock_options_information_supported"
    if test_a_supported:
        return "options_improve_stock_method_only"
    if test_b_supported:
        return "stock_structure_improves_options_forecast_only"
    if disagreement_descriptive:
        return "cross_market_disagreement_descriptive_only"
    return "no_cross_market_increment"


def validate_individual_statuses(statuses: Mapping[str, str]) -> None:
    """Reject missing or non-preregistered individual status values."""

    required = {
        "test_a_options_to_stock_status",
        "test_a_disagreement_status",
        "test_b_stock_to_options_status",
        "test_b_route_increment_status",
    }
    missing = sorted(required.difference(statuses))
    if missing:
        raise ValueError(f"individual cross-market statuses missing: {missing}")
    invalid = {key: statuses[key] for key in required if statuses[key] not in INDIVIDUAL_STATUSES}
    if invalid:
        raise ValueError(f"individual cross-market statuses invalid: {invalid}")


__all__ = [
    "ASSESSMENT_END",
    "ASSESSMENT_START",
    "BASE_OPTIONS_FEATURES",
    "BOOTSTRAP_SEED",
    "CROSS_MARKET_FEATURES",
    "DEVELOPMENT_END",
    "DEVELOPMENT_START",
    "DENSE_CHECKPOINTS",
    "DENSE_H0_FEATURES",
    "ExactHistoryExtraction",
    "FROZEN_COHORT",
    "FrozenCrossMarketModel",
    "INDIVIDUAL_STATUSES",
    "OPTIONS_MODEL_FEATURES",
    "OPTIONS_NULL_SEEDS",
    "OVERALL_DECISIONS",
    "PROTECTED_START",
    "REALIZED_VOLATILITY_FEATURES",
    "RIDGE_R0_NUMERIC",
    "RIDGE_R1_NUMERIC",
    "ROUTE_FEATURES",
    "ROUTE_NULL_SEEDS",
    "ROUTE_STATE_LEVELS",
    "SAFETY_FLAGS",
    "STOCK_RELATIVE_OPTIONS_FEATURES",
    "Standardization",
    "TEST_A_S0_NUMERIC",
    "TEST_A_S1_NUMERIC",
    "TEST_A_S2_NUMERIC",
    "TEST_B_O0_NUMERIC",
    "TEST_B_O1_NUMERIC",
    "TEST_B_O2_NUMERIC",
    "add_cross_market_disagreement",
    "add_test_a_target",
    "add_test_b_target",
    "apply_stock_relative_options",
    "assert_protected_dates",
    "assert_safety_flags",
    "binary_metrics",
    "calculate_optional_option_features",
    "calculate_primary_option_features",
    "categorical_controls",
    "choose_overall_decision",
    "compute_underlying_movement_outcomes",
    "continuous_residual_metrics",
    "coverage_gates",
    "fit_cross_market_model",
    "fit_cross_market_standardization",
    "fit_stock_relative_options",
    "fixed_session_bootstrap_multiplicities",
    "extract_exact_history_records",
    "manual_model_prediction",
    "permute_options_bundle",
    "permute_route_bundle_and_state",
    "previous_trading_session",
    "probability_quantile_boundaries",
    "reconstruct_clean_structural_panel",
    "route_state_movement_metrics",
    "select_primary_atm_pair",
    "test_a_disagreement_increment_passes",
    "test_a_options_increment_passes",
    "test_b_route_increment_passes",
    "test_b_stock_increment_passes",
    "trailing_realised_volatility_20d",
    "validate_exact_previous_session_join",
    "validate_individual_statuses",
]
