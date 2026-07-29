"""Research-only raw-OHLC entry-onset discovery.

This runner asks whether information visible at a completed five-minute bar
marks the *onset* of a clean directional path from the exact next-bar open.
It deliberately excludes regimes, states, loops, cycles, B0, templates,
named pattern flags, earlier predictions, volume, terminal-return objectives,
costs, and P&L.

The scoring path is integrity-staged.  Each score-month probability is made
before that same month's outcomes are read.  Because this is expanding OOF
development, completed earlier validation-month path labels may and do train
later folds.  The global probability/threshold/onset/control/reason bundle is
written and hashed before the final all-fold evaluation join, not before every
validation label is read.  Artifact names containing ``pre_outcome`` retain a
legacy shorthand meaning ``pre_final_evaluation_join``.  ``--validate-only``
reads only 2024 timestamp/OHLC, constructs causal features and lagged scales,
and checks exact support; it never constructs a future-path label or fits a
model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_clean_slate_causal_ohlc_entries_v1 as clean_slate
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260712-raw-ohlc-entry-onset-discovery-v1.json"
PRE_SCORE_PATH = WORK / "contracts/20260712-raw-ohlc-entry-onset-discovery-v1-pre-score.json"
ENVIRONMENT_ROOT = Path("/Users/michaelsalerno/StockerLocal")
OUT = Path("/private/tmp/stocker_raw_ohlc_entry_onset_discovery_v1_20260712")

CONTRACT_ID = "raw_ohlc_entry_onset_discovery_v1"
SEED = 20260712
SOURCE_COLUMNS = tuple(clean_slate.SOURCE_COLUMNS)
SYMBOLS = tuple(clean_slate.SYMBOLS)
HORIZONS = (6, 12, 24)
PREDICTION_MONTHS = tuple(f"2024-{month:02d}" for month in range(6, 13))
VALIDATION_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
CLOCK_FEATURES = tuple(clean_slate.CLOCK_FEATURES)
FULL_FEATURES = tuple(clean_slate.FULL_FEATURES)
ALGORITHMS = ("clock_logit", "full_logit", "full_hgb")
CANDIDATES = ("full_logit", "full_hgb")
CLASS_VALUES = np.asarray([0, 1, 2], dtype=np.int8)
CLASS_COLUMNS = ("p_no_entry", "p_long_first", "p_short_first")

EXPECTED_REGULAR_ROWS = 424_583
EXPECTED_UNION_SESSIONS = 252
EXPECTED_SYMBOL_SESSIONS = 5_539
EXPECTED_GAPS = 2_612
EXPECTED_ANNUAL_ROWS = {6: 365_075, 12: 330_577, 24: 264_817}
EXPECTED_VALIDATION_ROWS = {6: 186_112, 12: 168_639, 24: 135_240}
EXPECTED_JUNE_ROWS = {6: 27_733, 12: 25_143, 24: 20_214}

MIN_SCALE_BARS = 3
MAX_SCALE_BARS = 12
SCALE_FLOOR_BPS = 1.0
FIRE_QUANTILE = 0.95
REARM_QUANTILE = 0.75
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_BLOCK = 5
LOWER_QUANTILE = 0.0125
UPPER_QUANTILE = 0.9875

LOGIT_PARAMETERS: dict[str, Any] = {
    "penalty": "l2",
    "C": 0.2,
    "fit_intercept": True,
    "solver": "lbfgs",
    "max_iter": 500,
    "tol": 1e-6,
    "random_state": SEED,
}
HGB_PARAMETERS: dict[str, Any] = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "max_depth": 3,
    "min_samples_leaf": 500,
    "l2_regularization": 10.0,
    "max_bins": 64,
    "early_stopping": False,
    "random_state": SEED,
}
ALGORITHM_FEATURES = {
    "clock_logit": CLOCK_FEATURES,
    "full_logit": FULL_FEATURES,
    "full_hgb": FULL_FEATURES,
}

REASON_GROUPS: dict[str, tuple[str, ...]] = {
    "clock": CLOCK_FEATURES,
    "current_bar_geometry": (
        "log_close_open",
        "log_high_low",
        "signed_body_fraction",
        "absolute_body_fraction",
        "upper_wick_fraction",
        "lower_wick_fraction",
        "close_location",
    ),
    "recent_directional_motion": (
        "close_return_1",
        "close_return_3",
        "close_return_6",
        "close_return_12",
    ),
    "volatility_and_range_level": (
        "mean_abs_close_return_3",
        "mean_abs_close_return_6",
        "mean_abs_close_return_12",
        "std_close_return_3",
        "std_close_return_6",
        "std_close_return_12",
        "mean_log_range_3",
        "mean_log_range_6",
        "mean_log_range_12",
        "running_log_range",
    ),
    "range_change": ("log_range_ratio_6", "log_range_ratio_12"),
    "session_drift": ("session_log_return",),
    "location_relative_to_extremes": (
        "distance_to_session_high",
        "distance_from_session_low",
        "session_range_location",
        "distance_to_rolling_high_6",
        "distance_from_rolling_low_6",
        "distance_to_rolling_high_12",
        "distance_from_rolling_low_12",
    ),
    "history_availability": (
        "availability_3",
        "availability_6",
        "availability_12",
    ),
}
REASON_TEXT = {
    "clock": "where the completed bar sits in the regular session",
    "current_bar_geometry": "the completed bar's body, range, wicks, and close location",
    "recent_directional_motion": "continuous recent close-to-close motion",
    "volatility_and_range_level": "the recent level of price variation and bar range",
    "range_change": "the current range relative to its recent range level",
    "session_drift": "continuous displacement from the session's first open",
    "location_relative_to_extremes": "continuous location relative to session and rolling extremes",
    "history_availability": "how much exact contiguous history was available",
}

PRE_OUTCOME_LEDGER_COLUMNS = (
    "anchor_id",
    "fold_month",
    "symbol_norm",
    "session_date",
    "decision_timestamp",
    "bar_ordinal",
    "segment_index",
    "segment_position",
    "algorithm",
    "horizon",
    "causal_scale_bps",
    "availability_12",
    "clock_bin_15",
    "clock_bin_30",
    "availability_bucket",
    *CLASS_COLUMNS,
)

PRE_SCORE_MANIFEST_CHRONOLOGY = {
    "validation_outcomes_read_before_manifest_freeze": False,
    "same_score_month_outcomes_read_before_that_month_probability": False,
    "prior_completed_validation_month_path_labels_permitted_for_later_folds": True,
    "global_bundle_expected_before_final_all_fold_evaluation_join": True,
    "global_bundle_expected_before_any_validation_outcome_is_read": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(
            f"contract drift for {label}: observed={observed!r}, expected={expected!r}"
        )


def provider_path(symbol: str) -> Path:
    return clean_slate.provider_path(symbol)


def source_paths() -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "frozen_clean_slate_runner": Path(clean_slate.__file__).resolve(),
        "environment_pyproject": ENVIRONMENT_ROOT / "pyproject.toml",
        "environment_uv_lock": ENVIRONMENT_ROOT / "uv.lock",
    }
    for symbol in SYMBOLS:
        paths[f"provider_full_file_{symbol}"] = provider_path(symbol)
    return paths


def current_source_hashes() -> dict[str, str]:
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen source files: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def load_contract_and_verify(
    require_pre_outcome: bool = True,
    *,
    require_pre_score: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate every frozen executable choice and, for scoring, source hashes."""

    if require_pre_score is not None:
        require_pre_outcome = bool(require_pre_score)
    contract = json.loads(CONTRACT_PATH.read_text())
    _require_equal(contract["contract_id"], CONTRACT_ID, "contract id")
    for key, expected in {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "broker_connection_enabled": False,
        "paper_or_demo_execution_enabled": False,
        "deployment_enabled": False,
        "strategy_promotion_permitted": False,
        "economic_edge_claim_permitted": False,
        "pnl_evaluation_permitted": False,
    }.items():
        _require_equal(contract[key], expected, key)
    _require_equal(contract["periods"]["data_year"], 2024, "data year")
    _require_equal(
        tuple(contract["periods"]["validation_months"]),
        VALIDATION_MONTHS,
        "validation months",
    )
    for year in (2023, 2025, 2026):
        _require_equal(contract["periods"][f"{year}_read_permitted"], False, f"{year} read")
    _require_equal(tuple(contract["sources"]["columns_read"]), SOURCE_COLUMNS, "columns")
    _require_equal(
        contract["sources"]["provider_volume_label"],
        "historical_volume_not_used",
        "volume label",
    )
    _require_equal(tuple(contract["universe"]["symbols"]), SYMBOLS, "symbols")
    _require_equal(tuple(contract["decision_and_path"]["horizons_bars"]), HORIZONS, "horizons")
    _require_equal(
        contract["decision_and_path"]["minimum_prior_contiguous_bars_for_scale"],
        MIN_SCALE_BARS,
        "minimum scale bars",
    )
    _require_equal(
        contract["decision_and_path"]["maximum_prior_contiguous_bars_for_scale"],
        MAX_SCALE_BARS,
        "maximum scale bars",
    )
    _require_equal(contract["decision_and_path"]["scale_floor_bps"], SCALE_FLOOR_BPS, "scale floor")
    _require_equal(
        contract["decision_and_path"]["pre_confirmation_adverse_endpoint"],
        "include the first-confirmation bar conservatively because within-bar "
        "ordering beyond the frozen first-touch rule is unknowable",
        "pre-confirmation adverse endpoint",
    )
    _require_equal(tuple(contract["features"]["clock_features"]), CLOCK_FEATURES, "clock features")
    _require_equal(tuple(contract["features"]["full_features_in_order"]), FULL_FEATURES, "features")
    observed_groups = {
        name: tuple(features)
        for name, features in contract["features"]["fixed_reason_groups"].items()
    }
    _require_equal(observed_groups, REASON_GROUPS, "reason groups")
    expected_algorithms = [
        ("clock_logit", "clock_features", "LogisticRegression", LOGIT_PARAMETERS),
        ("full_logit", "full_features_in_order", "LogisticRegression", LOGIT_PARAMETERS),
        ("full_hgb", "full_features_in_order", "HistGradientBoostingClassifier", HGB_PARAMETERS),
    ]
    observed_algorithms = [
        (item["name"], item["features"], item["estimator"], item["parameters"])
        for item in contract["algorithms"]
    ]
    _require_equal(observed_algorithms, expected_algorithms, "algorithms")
    thresholds = contract["thresholds_and_onsets"]
    _require_equal(thresholds["fire_probability_quantile"], FIRE_QUANTILE, "fire quantile")
    _require_equal(thresholds["rearm_probability_quantile"], REARM_QUANTILE, "rearm quantile")
    _require_equal(
        thresholds["weighted_quantile_method"],
        "left-continuous inverse weighted empirical CDF: the smallest stably "
        "sorted value whose cumulative weight is at least q times total weight",
        "weighted quantile method",
    )
    _require_equal(
        thresholds[
            "terminal_return_or_same_score_month_outcome_used_in_threshold_or_onset_mapping"
        ],
        False,
        "threshold/onset same-month outcome chronology",
    )
    _require_equal(contract["metrics"]["inference"]["draws"], BOOTSTRAP_DRAWS, "bootstrap draws")
    _require_equal(contract["metrics"]["inference"]["random_state"], SEED, "random state")
    controls = contract["matched_clock_controls"]
    _require_equal(
        controls["candidate_processing_order"],
        "onset_id ascending within candidate algorithm, horizon, and side",
        "control candidate processing order",
    )
    _require_equal(controls["own_anchor_excluded"], True, "control own-anchor exclusion")
    _require_equal(
        controls["same_session_policy"],
        "tier 0 requires a different session; relaxation tiers 1 through 3 may "
        "use the same session when necessary",
        "control same-session policy",
    )
    _require_equal(
        contract["metrics"]["clock_quartile_definition"],
        "min(bar_ordinal * 4 integer-divided by 78, 3)",
        "clock quartile definition",
    )
    gates = contract["retention_gate_per_candidate_side"]
    _require_equal(
        gates["probability_minimum_months_with_both_log_loss_and_brier_better_each_horizon"],
        4,
        "probability month stability gate",
    )
    _require_equal(
        gates[
            "probability_minimum_unchanged_prediction_stock_deletions_with_both_better_each_horizon"
        ],
        18,
        "probability stock-deletion stability gate",
    )
    chronology = contract["periods"]["fold_chronology"]
    _require_equal(
        chronology["same_score_month_outcomes_read_before_that_month_probability"],
        False,
        "same-month outcome chronology",
    )
    _require_equal(
        chronology["prior_completed_validation_month_path_labels_may_train_later_folds"],
        True,
        "prior-month expanding training chronology",
    )
    _require_equal(
        chronology[
            "global_bundle_written_after_some_prior_validation_month_training_labels_are_read"
        ],
        True,
        "global bundle prior-label chronology",
    )
    _require_equal(
        chronology["global_bundle_written_before_final_all_fold_evaluation_join"],
        True,
        "global bundle final evaluation chronology",
    )
    _require_equal(
        chronology["global_bundle_written_before_any_validation_outcome_is_read"],
        False,
        "global bundle globally unread chronology",
    )
    integrity = contract["integrity"]
    _require_equal(
        integrity["each_score_month_probability_generated_before_same_month_outcomes_are_read"],
        True,
        "integrity same-month outcome chronology",
    )
    _require_equal(
        integrity["prior_completed_validation_month_labels_used_in_later_expanding_folds"],
        True,
        "integrity expanding-fold chronology",
    )
    _require_equal(
        integrity["global_bundle_written_before_final_all_fold_evaluation_join"],
        True,
        "integrity final evaluation join chronology",
    )
    _require_equal(
        integrity["global_bundle_written_before_any_validation_outcome_is_read"],
        False,
        "integrity globally unread chronology",
    )
    _require_equal(
        integrity["pre_outcome_artifact_filename_semantics"],
        "legacy shorthand meaning before the final all-fold evaluation join; it "
        "does not mean that prior completed validation-month training labels were "
        "globally unread",
        "legacy artifact filename semantics",
    )

    support = contract["bar_validation"]
    for key, expected in {
        "observed_target_blind_regular_rows": EXPECTED_REGULAR_ROWS,
        "observed_target_blind_union_sessions": EXPECTED_UNION_SESSIONS,
        "observed_target_blind_symbol_sessions": EXPECTED_SYMBOL_SESSIONS,
        "observed_target_blind_within_session_nonfive_minute_gaps": EXPECTED_GAPS,
    }.items():
        _require_equal(support[key], expected, key)
    # These exact target-blind support fields were added in the final freeze.
    support_maps = {
        "expected_target_blind_eligible_rows_2024": EXPECTED_ANNUAL_ROWS,
        "expected_target_blind_validation_rows_july_december": EXPECTED_VALIDATION_ROWS,
        "expected_target_blind_calibration_rows_june": EXPECTED_JUNE_ROWS,
    }
    for key, expected in support_maps.items():
        observed = {int(k): int(v) for k, v in contract["decision_and_path"][key].items()}
        _require_equal(observed, expected, key)

    if not require_pre_outcome:
        return contract, None
    if not PRE_SCORE_PATH.is_file():
        raise FileNotFoundError(
            f"missing frozen pre-score manifest {PRE_SCORE_PATH}; validate and freeze "
            "contract/runner/environment/provider hashes before scoring"
        )
    manifest = json.loads(PRE_SCORE_PATH.read_text())
    expected_manifest_fields = {
        "contract_id",
        "frozen_at_utc",
        "frozen_before_scoring",
        "research_only",
        "live_ordering_enabled",
        "order_placement",
        "provider_volume_label",
        "scientific_status",
        "later_period_outcomes_read",
        *PRE_SCORE_MANIFEST_CHRONOLOGY,
        "environment_versions",
        "sha256",
    }
    _require_equal(set(manifest), expected_manifest_fields, "pre-score manifest schema")
    _require_equal(manifest["contract_id"], CONTRACT_ID, "pre-score contract id")
    for key, expected in {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "frozen_before_scoring": True,
        "provider_volume_label": "historical_volume_not_used",
        "scientific_status": "2024_internal_monthly_expanding_oof_entry_sign_discovery",
        "later_period_outcomes_read": False,
        **PRE_SCORE_MANIFEST_CHRONOLOGY,
    }.items():
        _require_equal(manifest[key], expected, f"pre-score {key}")
    frozen_at = pd.Timestamp(manifest["frozen_at_utc"])
    if frozen_at.tzinfo is None:
        raise AssertionError("pre-score frozen_at_utc must be timezone aware")
    _require_equal(manifest["sha256"], current_source_hashes(), "source hashes")
    _require_equal(manifest["environment_versions"], environment_versions(), "environment versions")
    return contract, manifest


def load_tape(
    symbols: Sequence[str] = SYMBOLS,
    *,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse the frozen predicate-filtered 2024 OHLC loader exactly."""

    result = clean_slate.load_tape(symbols, return_diagnostics=return_diagnostics)
    tape = result[0] if return_diagnostics else result
    if not isinstance(tape, pd.DataFrame):
        raise AssertionError("frozen tape loader returned an unexpected object")
    return result


def build_feature_surface(tape: pd.DataFrame) -> pd.DataFrame:
    """Reuse the hash-bound frozen forty-feature causal OHLC surface exactly."""

    frame = clean_slate.build_feature_surface(tape)
    _require_equal(
        tuple(name for name in FULL_FEATURES if name in frame), FULL_FEATURES, "feature order"
    )
    return frame


def causal_scale_bps(tape: pd.DataFrame) -> pd.Series:
    """Lagged t-1 median true range over 3--12 exact bars, floored at 1 bp.

    The segment's first bar uses log(high/low), as frozen, because an exact
    prior close does not exist.  A decision at segment position three is the
    first eligible decision: its scale uses positions zero through two and
    excludes the decision bar itself.
    """

    required = {
        "symbol_norm",
        "session_date",
        "segment_index",
        "segment_position",
        "high",
        "low",
        "close",
    }
    missing = required.difference(tape.columns)
    if missing:
        raise AssertionError(f"scale input missing columns {sorted(missing)}")
    if not tape.index.is_unique:
        raise AssertionError("scale input index must be unique")
    output = pd.Series(np.nan, index=tape.index, dtype=float, name="causal_scale_bps")
    keys = ["symbol_norm", "session_date", "segment_index"]
    for _, group in tape.groupby(keys, sort=False):
        ordered = group.sort_values("segment_position", kind="stable")
        positions = ordered["segment_position"].to_numpy(int)
        if not np.array_equal(positions, np.arange(len(ordered))):
            raise AssertionError("non-contiguous segment positions in scale input")
        high = ordered["high"].to_numpy(float)
        low = ordered["low"].to_numpy(float)
        close = ordered["close"].to_numpy(float)
        tr = 10_000.0 * np.log(high / low)
        if len(ordered) > 1:
            previous_close = close[:-1]
            tr[1:] = 10_000.0 * np.maximum.reduce(
                [
                    np.log(high[1:] / low[1:]),
                    np.abs(np.log(high[1:] / previous_close)),
                    np.abs(np.log(low[1:] / previous_close)),
                ]
            )
        lagged = (
            pd.Series(tr)
            .rolling(MAX_SCALE_BARS, min_periods=MIN_SCALE_BARS)
            .median()
            .shift(1)
            .to_numpy(float)
        )
        lagged = np.where(np.isfinite(lagged), np.maximum(lagged, SCALE_FLOOR_BPS), np.nan)
        output.loc[ordered.index] = lagged
    return output


compute_causal_scale_bps = causal_scale_bps


def _future_bar_arrays(
    future_bars: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(future_bars, pd.DataFrame):
        missing = {"open", "high", "low"}.difference(future_bars.columns)
        if missing:
            raise AssertionError(f"future path missing columns {sorted(missing)}")
        values = future_bars.loc[:, ["open", "high", "low"]].to_numpy(float)
    else:
        values = np.asarray(future_bars, float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise AssertionError("future path array must have open/high/low columns")
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise AssertionError("future path must contain finite positive bars")
    if ((values[:, 2] > values[:, 0]) | (values[:, 0] > values[:, 1])).any():
        raise AssertionError("future bar open lies outside low/high")
    return values[:, 0], values[:, 1], values[:, 2]


def label_path(
    next_open: float,
    future_bars: pd.DataFrame | np.ndarray,
    scale_bps: float,
) -> dict[str, Any]:
    """Apply the frozen symmetric first-passage ordering to one exact path."""

    if not math.isfinite(next_open) or next_open <= 0.0:
        raise AssertionError("next open must be finite and positive")
    if not math.isfinite(scale_bps) or scale_bps < SCALE_FLOOR_BPS:
        raise AssertionError("scale must be finite and at least the frozen floor")
    opens, highs, lows = _future_bar_arrays(future_bars)
    if not math.isclose(float(opens[0]), float(next_open), rel_tol=1e-12, abs_tol=0.0):
        raise AssertionError("next_open does not equal first future-bar open")
    upper = next_open * math.exp(scale_bps / 10_000.0)
    lower = next_open * math.exp(-scale_bps / 10_000.0)
    target_class = 0
    status = "no_hit_by_horizon"
    first_step: int | None = None
    for step, (open_value, high_value, low_value) in enumerate(
        zip(opens, highs, lows, strict=True), start=1
    ):
        gap_up = open_value >= upper
        gap_down = open_value <= lower
        if gap_up and gap_down:
            raise AssertionError("positive symmetric barriers cannot both be gap crossed")
        if gap_up:
            target_class, status, first_step = 1, "long_first", step
            break
        if gap_down:
            target_class, status, first_step = 2, "short_first", step
            break
        upper_touch = high_value >= upper
        lower_touch = low_value <= lower
        if upper_touch and lower_touch:
            target_class, status, first_step = 0, "intrabar_ambiguous", step
            break
        if upper_touch:
            target_class, status, first_step = 1, "long_first", step
            break
        if lower_touch:
            target_class, status, first_step = 2, "short_first", step
            break
    endpoint = first_step if first_step is not None else len(opens)
    pre_high = float(np.max(highs[:endpoint]))
    pre_low = float(np.min(lows[:endpoint]))
    upside_mfe_bps = 10_000.0 * math.log(float(np.max(highs)) / next_open)
    downside_mfe_bps = 10_000.0 * math.log(next_open / float(np.min(lows)))
    if target_class == 1:
        favourable = upside_mfe_bps / scale_bps
        adverse = downside_mfe_bps / scale_bps
        pre_adverse = 10_000.0 * math.log(next_open / pre_low) / scale_bps
    elif target_class == 2:
        favourable = downside_mfe_bps / scale_bps
        adverse = upside_mfe_bps / scale_bps
        pre_adverse = 10_000.0 * math.log(pre_high / next_open) / scale_bps
    else:
        favourable = adverse = pre_adverse = math.nan
    return {
        "target_class": target_class,
        "status": status,
        "first_confirmation_step": first_step,
        "upper_barrier": upper,
        "lower_barrier": lower,
        "upside_mfe_bps": upside_mfe_bps,
        "downside_mfe_bps": downside_mfe_bps,
        "upside_mfe_scale_units": upside_mfe_bps / scale_bps,
        "downside_mfe_scale_units": downside_mfe_bps / scale_bps,
        # These generic fields are oriented to the resolved true direction.
        # For no-hit/ambiguous paths there is no defensible orientation.
        "favourable_excursion_scale_units": favourable,
        "adverse_excursion_scale_units": adverse,
        "pre_confirmation_adverse_scale_units": pre_adverse,
        "directional_dominance_scale_units": favourable - adverse,
        # Including the confirmation bar is conservative because intrabar
        # ordering beyond the frozen first-touch rule is unknowable.
        "long_pre_confirmation_adverse_scale_units": (
            10_000.0 * math.log(next_open / pre_low) / scale_bps
        ),
        "short_pre_confirmation_adverse_scale_units": (
            10_000.0 * math.log(pre_high / next_open) / scale_bps
        ),
    }


def build_anchor_surface(feature_surface: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Construct target-blind exact scale/path support for a frozen horizon."""

    if horizon not in HORIZONS:
        raise AssertionError(f"unfrozen horizon {horizon}")
    scale = causal_scale_bps(feature_surface)
    exact_future = feature_surface["segment_position"].to_numpy(int) + horizon < feature_surface[
        "segment_size"
    ].to_numpy(int)
    supported = exact_future & np.isfinite(scale.to_numpy(float))
    columns = [
        "source_position",
        "symbol_norm",
        "session_date",
        "month_key",
        "timestamp",
        "bar_ordinal",
        "segment_index",
        "segment_position",
        *FULL_FEATURES,
    ]
    frame = feature_surface.loc[supported, columns].copy()
    frame = frame.rename(columns={"timestamp": "decision_timestamp"})
    frame["causal_scale_bps"] = scale.loc[frame.index].to_numpy(float)
    frame["horizon"] = np.int16(horizon)
    frame["path_end_timestamp"] = frame["decision_timestamp"] + pd.Timedelta(minutes=5 * horizon)
    frame["anchor_id"] = (
        frame["symbol_norm"].astype(str)
        + "|"
        + frame["decision_timestamp"].astype(str)
        + f"|h{horizon}"
    )
    frame["clock_bin_15"] = (frame["bar_ordinal"].to_numpy(int) // 3).astype(np.int16)
    frame["clock_bin_30"] = (frame["bar_ordinal"].to_numpy(int) // 6).astype(np.int16)
    availability = frame["availability_12"].to_numpy(float)
    frame["availability_bucket"] = np.where(
        availability < 0.5, 0, np.where(availability < 1.0, 1, 2)
    ).astype(np.int8)
    frame = frame.sort_values(
        ["symbol_norm", "session_date", "decision_timestamp"], kind="stable"
    ).reset_index(drop=True)
    expected = EXPECTED_ANNUAL_ROWS[horizon]
    if len(frame) != expected:
        raise AssertionError(f"target-blind h{horizon} support drift: {len(frame)} != {expected}")
    validation_count = int(frame["month_key"].isin(VALIDATION_MONTHS).sum())
    if validation_count != EXPECTED_VALIDATION_ROWS[horizon]:
        raise AssertionError(
            f"target-blind validation h{horizon} drift: {validation_count} != "
            f"{EXPECTED_VALIDATION_ROWS[horizon]}"
        )
    june_count = int(frame["month_key"].eq("2024-06").sum())
    if june_count != EXPECTED_JUNE_ROWS[horizon]:
        raise AssertionError(
            f"target-blind June h{horizon} drift: {june_count} != {EXPECTED_JUNE_ROWS[horizon]}"
        )
    if frame["anchor_id"].duplicated().any():
        raise AssertionError("duplicate anchor id")
    return frame


def fold_masks(surface: pd.DataFrame, month: str) -> tuple[np.ndarray, np.ndarray]:
    if month not in PREDICTION_MONTHS:
        raise AssertionError(f"unfrozen prediction month {month}")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    train = surface["path_end_timestamp"].lt(month_start).to_numpy(bool)
    score = surface["month_key"].eq(month).to_numpy(bool)
    if not train.any() or not score.any() or np.logical_and(train, score).any():
        raise AssertionError(f"invalid expanding fold masks for {month}")
    if not surface.loc[train, "path_end_timestamp"].lt(month_start).all():
        raise AssertionError("training future path reaches score month")
    return train, score


def nested_symbol_session_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal-symbol/equal-session weights, normalized to mean one."""

    required = {"symbol_norm", "session_date"}
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        raise AssertionError(f"weight input invalid; missing={sorted(missing)}")
    pairs = frame[["symbol_norm", "session_date"]].astype(str)
    sessions_per_symbol = (
        pairs.groupby("symbol_norm", sort=False)["session_date"]
        .transform("nunique")
        .to_numpy(float)
    )
    rows_per_session = (
        pairs.groupby(["symbol_norm", "session_date"], sort=False)["session_date"]
        .transform("size")
        .to_numpy(float)
    )
    raw = 1.0 / (sessions_per_symbol * rows_per_session)
    weights = raw / raw.mean()
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("nested weights do not have mean one")
    symbol_totals = (
        pd.Series(weights)
        .groupby(pairs["symbol_norm"].to_numpy(), sort=False)
        .sum()
        .to_numpy(float)
    )
    if not np.allclose(symbol_totals, symbol_totals[0], rtol=0.0, atol=1e-9):
        raise AssertionError("nested weights do not give symbols equal total weight")
    return weights


metric_weights = nested_symbol_session_weights


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    q: float,
) -> float:
    """Frozen left-continuous inverse weighted empirical CDF."""

    x = np.asarray(values, float)
    w = np.asarray(weights, float)
    if x.ndim != 1 or w.ndim != 1 or len(x) != len(w) or len(x) == 0:
        raise AssertionError("weighted quantile requires equal non-empty vectors")
    if (
        not (0.0 <= q <= 1.0)
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or (w < 0.0).any()
        or w.sum() <= 0.0
    ):
        raise AssertionError("invalid weighted quantile inputs")
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    sorted_w = w[order]
    target = q * float(sorted_w.sum())
    index = int(np.searchsorted(np.cumsum(sorted_w), target, side="left"))
    return float(sorted_x[min(index, len(sorted_x) - 1)])


def attach_path_labels(
    tape: pd.DataFrame,
    anchors: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Construct frozen first-passage outcomes for an already selected cohort.

    Callers are responsible for temporal eligibility.  The monthly fitting
    routine invokes this only for anchors whose complete path ended before the
    current score month.  Thus earlier completed validation-month paths may
    train later folds, but the score month's own paths cannot.  After the
    global bundle freeze, the final evaluation join attaches every validation
    fold's path to its already out-of-fold probability.
    """

    if horizon not in HORIZONS or anchors.empty:
        if anchors.empty:
            return pd.DataFrame(
                columns=[
                    "anchor_id",
                    "target_class",
                    "status",
                    "first_confirmation_step",
                    "upside_mfe_bps",
                    "downside_mfe_bps",
                    "upside_mfe_scale_units",
                    "downside_mfe_scale_units",
                    "long_pre_confirmation_adverse_scale_units",
                    "short_pre_confirmation_adverse_scale_units",
                ]
            )
        raise AssertionError(f"unfrozen horizon {horizon}")
    positions = anchors["source_position"].to_numpy(np.int64)
    n_rows = len(anchors)
    timestamp = tape["timestamp"].to_numpy()
    symbol = tape["symbol_norm"].to_numpy(str)
    session = tape["session_date"].to_numpy(str)
    expected_symbol = anchors["symbol_norm"].to_numpy(str)
    expected_session = anchors["session_date"].to_numpy(str)
    end_positions = positions + horizon
    if not (
        timestamp[end_positions] - anchors["decision_timestamp"].to_numpy()
        == np.timedelta64(5 * horizon, "m")
    ).all():
        raise AssertionError("inexact future-path timestamp support")
    if not (
        (symbol[positions + 1] == expected_symbol)
        & (symbol[end_positions] == expected_symbol)
        & (session[positions + 1] == expected_session)
        & (session[end_positions] == expected_session)
    ).all():
        raise AssertionError("future path crosses symbol/session boundary")

    raw_open = tape["open"].to_numpy(float)
    raw_high = tape["high"].to_numpy(float)
    raw_low = tape["low"].to_numpy(float)
    next_open = raw_open[positions + 1]
    scale = anchors["causal_scale_bps"].to_numpy(float)
    upper = next_open * np.exp(scale / 10_000.0)
    lower = next_open * np.exp(-scale / 10_000.0)
    target_class = np.zeros(n_rows, dtype=np.int8)
    status = np.full(n_rows, "no_hit_by_horizon", dtype=object)
    first_step = np.full(n_rows, np.nan, dtype=float)
    unresolved = np.ones(n_rows, dtype=bool)
    path_high = np.full(n_rows, -np.inf)
    path_low = np.full(n_rows, np.inf)
    pre_high = np.full(n_rows, -np.inf)
    pre_low = np.full(n_rows, np.inf)
    for step in range(1, horizon + 1):
        indices = positions + step
        open_value = raw_open[indices]
        high_value = raw_high[indices]
        low_value = raw_low[indices]
        path_high = np.maximum(path_high, high_value)
        path_low = np.minimum(path_low, low_value)
        # Update before resolving: the confirmation bar is included
        # conservatively in pre-confirmation adverse excursion.
        pre_high[unresolved] = np.maximum(pre_high[unresolved], high_value[unresolved])
        pre_low[unresolved] = np.minimum(pre_low[unresolved], low_value[unresolved])
        gap_up = unresolved & (open_value >= upper)
        gap_down = unresolved & (open_value <= lower)
        if np.logical_and(gap_up, gap_down).any():
            raise AssertionError("both symmetric barriers crossed by one open")
        target_class[gap_up] = 1
        status[gap_up] = "long_first"
        first_step[gap_up] = step
        target_class[gap_down] = 2
        status[gap_down] = "short_first"
        first_step[gap_down] = step
        unresolved &= ~(gap_up | gap_down)
        dual = unresolved & (high_value >= upper) & (low_value <= lower)
        status[dual] = "intrabar_ambiguous"
        first_step[dual] = step
        unresolved &= ~dual
        upper_only = unresolved & (high_value >= upper)
        lower_only = unresolved & (low_value <= lower)
        target_class[upper_only] = 1
        status[upper_only] = "long_first"
        first_step[upper_only] = step
        target_class[lower_only] = 2
        status[lower_only] = "short_first"
        first_step[lower_only] = step
        unresolved &= ~(upper_only | lower_only)

    upside = 10_000.0 * np.log(path_high / next_open)
    downside = 10_000.0 * np.log(next_open / path_low)
    output = anchors[
        [
            "anchor_id",
            "symbol_norm",
            "session_date",
            "month_key",
            "decision_timestamp",
            "bar_ordinal",
            "horizon",
            "causal_scale_bps",
        ]
    ].copy()
    output["target_class"] = target_class
    output["status"] = status
    output["first_confirmation_step"] = first_step
    output["upside_mfe_bps"] = upside
    output["downside_mfe_bps"] = downside
    output["upside_mfe_scale_units"] = upside / scale
    output["downside_mfe_scale_units"] = downside / scale
    output["long_pre_confirmation_adverse_scale_units"] = (
        10_000.0 * np.log(next_open / pre_low) / scale
    )
    output["short_pre_confirmation_adverse_scale_units"] = (
        10_000.0 * np.log(pre_high / next_open) / scale
    )
    if not np.isfinite(
        output[
            [
                "upside_mfe_bps",
                "downside_mfe_bps",
                "upside_mfe_scale_units",
                "downside_mfe_scale_units",
                "long_pre_confirmation_adverse_scale_units",
                "short_pre_confirmation_adverse_scale_units",
            ]
        ].to_numpy(float)
    ).all():
        raise AssertionError("non-finite path excursion")
    return output


def training_medians(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(np.asarray(values, float), axis=0)
    if np.isnan(medians).any():
        raise AssertionError(
            f"all-missing training features {np.flatnonzero(np.isnan(medians)).tolist()}"
        )
    return medians


def apply_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    output = np.asarray(values, float).copy()
    rows, columns = np.where(np.isnan(output))
    output[rows, columns] = medians[columns]
    if not np.isfinite(output).all():
        raise AssertionError("non-finite feature after fold-training median imputation")
    return output


def _class_probability_matrix(estimator: Any, values: np.ndarray) -> np.ndarray:
    classes = np.asarray(estimator.classes_, int)
    _require_equal(tuple(classes.tolist()), (0, 1, 2), "fitted classes")
    probabilities = np.asarray(estimator.predict_proba(values), float)
    if probabilities.shape != (len(values), 3):
        raise AssertionError("unexpected probability matrix shape")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise AssertionError("invalid fitted probabilities")
    return probabilities


def validate_pre_outcome_ledger(frame: pd.DataFrame) -> None:
    """Validate the legacy-named pre-final-evaluation probability ledger."""

    _require_equal(tuple(frame.columns), PRE_OUTCOME_LEDGER_COLUMNS, "pre-outcome ledger schema")
    forbidden_substrings = (
        "target_class",
        "status",
        "confirmation",
        "mfe",
        "excursion",
        "barrier",
        "future",
        "return",
        "cost",
        "pnl",
        "profit",
    )
    forbidden_exact = {
        "position",
        "order",
        "order_id",
        "position_size",
        "entry_order",
        "exit_order",
    }
    bad = [
        column
        for column in frame.columns
        if column.lower() in forbidden_exact
        or any(token in column.lower() for token in forbidden_substrings)
    ]
    if bad:
        raise AssertionError(f"future/outcome/economic fields in prediction ledger: {bad}")
    probability = frame.loc[:, CLASS_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise AssertionError("invalid pre-outcome probabilities")
    if frame.duplicated(["anchor_id", "algorithm"]).any():
        raise AssertionError("duplicate probability key")
    if not set(frame["algorithm"]).issubset(ALGORITHMS):
        raise AssertionError("unknown algorithm in probability ledger")
    if not set(frame["fold_month"]).issubset(PREDICTION_MONTHS):
        raise AssertionError("unknown prediction month")


def fit_monthly_oof_probabilities(
    tape: pd.DataFrame,
    feature_surface: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, int, str], dict[str, Any]],
]:
    """Fit June--December expanding OOF models without same-score-month labels.

    Prior completed validation months become eligible training data for later
    folds.  This is causal expanding OOF development, not a globally sealed
    July--December holdout.
    """

    prediction_parts: list[pd.DataFrame] = []
    preprocessing_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, int, str], dict[str, Any]] = {}
    label_cache: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        surface = build_anchor_surface(feature_surface, horizon)
        cached = pd.DataFrame()
        cached_ids: set[str] = set()
        for month in PREDICTION_MONTHS:
            train_mask, score_mask = fold_masks(surface, month)
            train = surface.loc[train_mask].copy()
            score = surface.loc[score_mask].copy()
            missing_train = train.loc[~train["anchor_id"].isin(cached_ids)]
            if not missing_train.empty:
                added = attach_path_labels(tape, missing_train, horizon)
                cached = pd.concat([cached, added], ignore_index=True)
                cached_ids.update(added["anchor_id"].astype(str))
            target_lookup = cached.set_index("anchor_id")["target_class"]
            target = train["anchor_id"].map(target_lookup)
            if target.isna().any():
                raise AssertionError("eligible training label absent from progressive cache")
            y = target.to_numpy(np.int8)
            if tuple(np.unique(y).tolist()) != (0, 1, 2):
                raise AssertionError("monthly fold lacks a frozen target class")
            weights = nested_symbol_session_weights(train)
            for algorithm in ALGORITHMS:
                features = ALGORITHM_FEATURES[algorithm]
                train_raw = train.loc[:, features].to_numpy(float)
                score_raw = score.loc[:, features].to_numpy(float)
                medians = training_medians(train_raw)
                train_imputed = apply_medians(train_raw, medians)
                score_imputed = apply_medians(score_raw, medians)
                scaler: StandardScaler | None = None
                scaler_mean = np.full(len(features), np.nan)
                scaler_scale = np.full(len(features), np.nan)
                if algorithm.endswith("logit"):
                    scaler = StandardScaler()
                    scaler.fit(train_imputed, sample_weight=weights)
                    train_values = scaler.transform(train_imputed)
                    score_values = scaler.transform(score_imputed)
                    scaler_mean = scaler.mean_.astype(float)
                    scaler_scale = scaler.scale_.astype(float)
                    estimator: LogisticRegression | HistGradientBoostingClassifier = (
                        LogisticRegression(**LOGIT_PARAMETERS)
                    )
                else:
                    train_values = train_imputed
                    score_values = score_imputed
                    estimator = HistGradientBoostingClassifier(**HGB_PARAMETERS)
                estimator.fit(train_values, y, sample_weight=weights)
                probabilities = _class_probability_matrix(estimator, score_values)
                output = score[
                    [
                        "anchor_id",
                        "symbol_norm",
                        "session_date",
                        "decision_timestamp",
                        "bar_ordinal",
                        "segment_index",
                        "segment_position",
                        "causal_scale_bps",
                        "availability_12",
                        "clock_bin_15",
                        "clock_bin_30",
                        "availability_bucket",
                    ]
                ].copy()
                output["fold_month"] = month
                output["algorithm"] = algorithm
                output["horizon"] = np.int16(horizon)
                for index, column in enumerate(CLASS_COLUMNS):
                    output[column] = probabilities[:, index]
                output = output.loc[:, PRE_OUTCOME_LEDGER_COLUMNS]
                prediction_parts.append(output)
                for index, feature in enumerate(features):
                    preprocessing_rows.append(
                        {
                            "fold_month": month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "feature_order": index,
                            "feature": feature,
                            "training_median": float(medians[index]),
                            "scaler_mean": float(scaler_mean[index]),
                            "scaler_scale": float(scaler_scale[index]),
                        }
                    )
                if isinstance(estimator, LogisticRegression):
                    for class_index, class_value in enumerate(estimator.classes_):
                        coefficient_rows.append(
                            {
                                "fold_month": month,
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "class_value": int(class_value),
                                "feature_order": -1,
                                "feature": "__intercept__",
                                "coefficient": float(estimator.intercept_[class_index]),
                            }
                        )
                        for feature_index, (feature, coefficient) in enumerate(
                            zip(features, estimator.coef_[class_index], strict=True)
                        ):
                            coefficient_rows.append(
                                {
                                    "fold_month": month,
                                    "algorithm": algorithm,
                                    "horizon": horizon,
                                    "class_value": int(class_value),
                                    "feature_order": feature_index,
                                    "feature": feature,
                                    "coefficient": float(coefficient),
                                }
                            )
                fold_rows.append(
                    {
                        "fold_month": month,
                        "algorithm": algorithm,
                        "horizon": horizon,
                        "train_rows": len(train),
                        "score_rows": len(score),
                        "train_symbols": int(train["symbol_norm"].nunique()),
                        "train_symbol_sessions": int(
                            train.groupby(["symbol_norm", "session_date"], sort=False).ngroups
                        ),
                        "maximum_training_path_end_timestamp": train["path_end_timestamp"].max(),
                        "minimum_scoring_timestamp": score["decision_timestamp"].min(),
                        "same_score_month_training_label_rows": int(
                            train["month_key"].eq(month).sum()
                        ),
                        "prior_completed_validation_month_training_label_rows": int(
                            train["month_key"].isin(VALIDATION_MONTHS).sum()
                        ),
                        "same_score_month_outcomes_read_before_probability": False,
                        "class_0_rows": int((y == 0).sum()),
                        "class_1_rows": int((y == 1).sum()),
                        "class_2_rows": int((y == 2).sum()),
                        "weight_min": float(weights.min()),
                        "weight_max": float(weights.max()),
                        "fitted_iterations": int(estimator.n_iter_)
                        if isinstance(estimator, HistGradientBoostingClassifier)
                        else int(estimator.n_iter_[0]),
                    }
                )
                bundles[(month, horizon, algorithm)] = {
                    "estimator": estimator,
                    "features": features,
                    "medians": medians,
                    "scaler": scaler,
                }
            del train, score
            gc.collect()
        label_cache[horizon] = cached
        del surface
        gc.collect()
    ledger = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(
            ["algorithm", "horizon", "symbol_norm", "session_date", "decision_timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    validate_pre_outcome_ledger(ledger)
    expected = sum(EXPECTED_JUNE_ROWS.values()) + sum(EXPECTED_VALIDATION_ROWS.values())
    if len(ledger) != expected * len(ALGORITHMS):
        raise AssertionError("probability ledger support drift")
    return (
        ledger,
        pd.DataFrame(preprocessing_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(fold_rows),
        bundles,
    )


def calibrate_prior_month_thresholds(oof: pd.DataFrame) -> pd.DataFrame:
    """Freeze q95 fire/q75 rearm thresholds from the immediately prior OOF month."""

    validate_pre_outcome_ledger(oof.loc[:, PRE_OUTCOME_LEDGER_COLUMNS])
    rows: list[dict[str, Any]] = []
    for score_month in VALIDATION_MONTHS:
        month_number = int(score_month[-2:])
        source_month = f"2024-{month_number - 1:02d}"
        for algorithm in ALGORITHMS:
            for horizon in HORIZONS:
                source = oof.loc[
                    oof["fold_month"].eq(source_month)
                    & oof["algorithm"].eq(algorithm)
                    & oof["horizon"].eq(horizon)
                ].copy()
                if source.empty:
                    raise AssertionError(
                        f"missing {source_month} probabilities for {algorithm} h{horizon}"
                    )
                weights = metric_weights(source)
                for side, probability_column in (
                    ("long", "p_long_first"),
                    ("short", "p_short_first"),
                ):
                    values = source[probability_column].to_numpy(float)
                    rows.append(
                        {
                            "score_month": score_month,
                            "threshold_source_month": source_month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "side": side,
                            "fire_threshold": weighted_quantile(values, weights, FIRE_QUANTILE),
                            "rearm_threshold": weighted_quantile(values, weights, REARM_QUANTILE),
                            "source_rows": len(source),
                            "source_weight_sum": float(weights.sum()),
                            "fire_quantile": FIRE_QUANTILE,
                            "rearm_quantile": REARM_QUANTILE,
                        }
                    )
    thresholds = (
        pd.DataFrame(rows)
        .sort_values(["algorithm", "horizon", "score_month", "side"], kind="stable")
        .reset_index(drop=True)
    )
    if (thresholds["fire_threshold"] < thresholds["rearm_threshold"]).any():
        raise AssertionError("fire threshold below rearm threshold")
    if thresholds.duplicated(["score_month", "algorithm", "horizon", "side"]).any():
        raise AssertionError("duplicate threshold key")
    return thresholds


build_thresholds = calibrate_prior_month_thresholds


def extract_onsets(
    ledger: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Extract first armed fires with independent side hysteresis and no cooldown."""

    candidate = ledger.loc[
        ledger["fold_month"].isin(VALIDATION_MONTHS) & ledger["algorithm"].isin(CANDIDATES)
    ].copy()
    onset_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    threshold_index = thresholds.set_index(["score_month", "algorithm", "horizon", "side"])
    group_keys = ["algorithm", "horizon", "symbol_norm", "session_date", "segment_index"]
    for group_key, group in candidate.groupby(group_keys, sort=False):
        algorithm, horizon, symbol, session_date, segment_index = group_key
        ordered = group.sort_values("decision_timestamp", kind="stable")
        long_armed = True
        short_armed = True
        for row in ordered.itertuples(index=False):
            score_month = str(row.fold_month)
            long_threshold = threshold_index.loc[(score_month, algorithm, horizon, "long")]
            short_threshold = threshold_index.loc[(score_month, algorithm, horizon, "short")]
            long_before, short_before = long_armed, short_armed
            if not long_armed and row.p_long_first < float(long_threshold["rearm_threshold"]):
                long_armed = True
            if not short_armed and row.p_short_first < float(short_threshold["rearm_threshold"]):
                short_armed = True
            long_rearmed = (not long_before) and long_armed
            short_rearmed = (not short_before) and short_armed
            long_fire = bool(
                long_armed
                and row.p_long_first >= float(long_threshold["fire_threshold"])
                and row.p_long_first > row.p_short_first
            )
            short_fire = bool(
                short_armed
                and row.p_short_first >= float(short_threshold["fire_threshold"])
                and row.p_short_first > row.p_long_first
            )
            conflict = long_fire and short_fire
            emitted_side: str | None = None
            onset_id: str | None = None
            if not conflict and long_fire:
                emitted_side = "long"
                long_armed = False
            elif not conflict and short_fire:
                emitted_side = "short"
                short_armed = False
            if emitted_side is not None:
                onset_id = f"{algorithm}|{row.anchor_id}|{emitted_side}"
                onset_rows.append(
                    {
                        "onset_id": onset_id,
                        "anchor_id": row.anchor_id,
                        "candidate_algorithm": algorithm,
                        "horizon": int(horizon),
                        "side": emitted_side,
                        "fold_month": score_month,
                        "symbol_norm": symbol,
                        "session_date": session_date,
                        "decision_timestamp": row.decision_timestamp,
                        "bar_ordinal": int(row.bar_ordinal),
                        "segment_index": int(segment_index),
                        "segment_position": int(row.segment_position),
                        "causal_scale_bps": float(row.causal_scale_bps),
                        "availability_12": float(row.availability_12),
                        "availability_bucket": int(row.availability_bucket),
                        "clock_bin_15": int(row.clock_bin_15),
                        "clock_bin_30": int(row.clock_bin_30),
                        "p_no_entry": float(row.p_no_entry),
                        "p_long_first": float(row.p_long_first),
                        "p_short_first": float(row.p_short_first),
                        "chosen_probability": float(
                            row.p_long_first if emitted_side == "long" else row.p_short_first
                        ),
                        "opposite_probability": float(
                            row.p_short_first if emitted_side == "long" else row.p_long_first
                        ),
                    }
                )
            state_rows.append(
                {
                    "anchor_id": row.anchor_id,
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "fold_month": score_month,
                    "symbol_norm": symbol,
                    "session_date": session_date,
                    "decision_timestamp": row.decision_timestamp,
                    "segment_index": int(segment_index),
                    "long_armed_before": long_before,
                    "short_armed_before": short_before,
                    "long_rearmed": long_rearmed,
                    "short_rearmed": short_rearmed,
                    "long_fire": long_fire,
                    "short_fire": short_fire,
                    "conflict": conflict,
                    "emitted_side": emitted_side,
                    "onset_id": onset_id,
                    "long_armed_after": long_armed,
                    "short_armed_after": short_armed,
                }
            )
    onsets = pd.DataFrame(onset_rows)
    if not onsets.empty:
        onsets = onsets.sort_values("onset_id", kind="stable").reset_index(drop=True)
        if onsets["onset_id"].duplicated().any():
            raise AssertionError("duplicate onset id")
    states = (
        pd.DataFrame(state_rows)
        .sort_values(
            ["algorithm", "horizon", "symbol_norm", "session_date", "decision_timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if return_state:
        return onsets, states
    return onsets


def _clock_probability_column(side: str) -> str:
    if side == "long":
        return "p_long_first"
    if side == "short":
        return "p_short_first"
    raise AssertionError(f"unknown side {side}")


def match_clock_controls(onsets: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Deterministically match one outcome-blind clock control per onset."""

    clock = ledger.loc[
        ledger["algorithm"].eq("clock_logit") & ledger["fold_month"].isin(VALIDATION_MONTHS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (algorithm, horizon, side), candidates in onsets.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        probability_column = _clock_probability_column(str(side))
        horizon_pool = clock.loc[clock["horizon"].eq(horizon)].copy()
        local_pools = {
            key: group.sort_values(
                [probability_column, "anchor_id"],
                ascending=[False, True],
                kind="stable",
            ).reset_index(drop=True)
            for key, group in horizon_pool.groupby(["symbol_norm", "fold_month"], sort=False)
        }
        used: set[str] = set()
        for candidate in candidates.sort_values("onset_id", kind="stable").itertuples(index=False):
            pool = local_pools[(candidate.symbol_norm, candidate.fold_month)]
            base = ~pool["anchor_id"].eq(candidate.anchor_id) & ~pool["anchor_id"].isin(used)
            tier_masks = [
                base
                & ~pool["session_date"].eq(candidate.session_date)
                & pool["clock_bin_15"].eq(candidate.clock_bin_15)
                & pool["availability_bucket"].eq(candidate.availability_bucket),
                base
                & pool["clock_bin_30"].eq(candidate.clock_bin_30)
                & pool["availability_bucket"].eq(candidate.availability_bucket),
                base & pool["clock_bin_30"].eq(candidate.clock_bin_30),
                base,
            ]
            chosen: pd.Series | None = None
            match_tier = -1
            for tier, mask in enumerate(tier_masks):
                eligible = pool.loc[mask]
                if not eligible.empty:
                    chosen = eligible.iloc[0]
                    match_tier = tier
                    break
            record: dict[str, Any] = {
                "onset_id": candidate.onset_id,
                "candidate_algorithm": algorithm,
                "horizon": int(horizon),
                "side": side,
                "candidate_anchor_id": candidate.anchor_id,
                "candidate_symbol_norm": candidate.symbol_norm,
                "candidate_session_date": candidate.session_date,
                "candidate_fold_month": candidate.fold_month,
                "candidate_bar_ordinal": int(candidate.bar_ordinal),
                "control_anchor_id": None,
                "control_symbol_norm": None,
                "control_session_date": None,
                "control_decision_timestamp": pd.NaT,
                "control_bar_ordinal": None,
                "control_clock_probability": math.nan,
                "match_tier": match_tier,
                "matched": False,
            }
            if chosen is not None:
                control_id = str(chosen["anchor_id"])
                used.add(control_id)
                record.update(
                    {
                        "control_anchor_id": control_id,
                        "control_symbol_norm": str(chosen["symbol_norm"]),
                        "control_session_date": str(chosen["session_date"]),
                        "control_decision_timestamp": chosen["decision_timestamp"],
                        "control_bar_ordinal": int(chosen["bar_ordinal"]),
                        "control_clock_probability": float(chosen[probability_column]),
                        "matched": True,
                    }
                )
            rows.append(record)
    controls = pd.DataFrame(rows).sort_values("onset_id", kind="stable").reset_index(drop=True)
    if len(controls) != len(onsets) or controls["onset_id"].duplicated().any():
        raise AssertionError("control matching did not preserve one row per onset")
    return controls


def logit_reason_contributions(
    estimator: LogisticRegression,
    standardized_values: np.ndarray,
    feature_names: Sequence[str],
    side: str,
) -> dict[str, Any]:
    """Exact chosen-versus-opposite standardized multinomial logit margin."""

    values = np.asarray(standardized_values, float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    names = tuple(feature_names)
    if values.shape[1] != len(names):
        raise AssertionError("logit reason feature width mismatch")
    classes = tuple(np.asarray(estimator.classes_, int).tolist())
    _require_equal(classes, (0, 1, 2), "logit reason classes")
    chosen_class, opposite_class = (1, 2) if side == "long" else (2, 1)
    if side not in {"long", "short"}:
        raise AssertionError(f"unknown reason side {side}")
    chosen_index = classes.index(chosen_class)
    opposite_index = classes.index(opposite_class)
    coefficient_delta = estimator.coef_[chosen_index] - estimator.coef_[opposite_index]
    intercept = float(estimator.intercept_[chosen_index] - estimator.intercept_[opposite_index])
    feature_contributions = values * coefficient_delta.reshape(1, -1)
    margin = intercept + feature_contributions.sum(axis=1)
    direct = (
        estimator.decision_function(values)[:, chosen_index]
        - estimator.decision_function(values)[:, opposite_index]
    )
    if not np.allclose(margin, direct, rtol=0.0, atol=1e-10):
        raise AssertionError("logit contribution reconstruction failed")
    group_contributions = {
        group: feature_contributions[:, [names.index(feature) for feature in features]].sum(axis=1)
        for group, features in REASON_GROUPS.items()
    }
    return {
        "intercept": intercept,
        "feature_names": names,
        "feature_contributions": feature_contributions,
        "group_contributions": group_contributions,
        "directional_margin": margin,
    }


def reason_dictionary() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "artifact_filename_semantics": (
            "pre_outcome is legacy shorthand for pre_final_evaluation_join; "
            "prior completed validation-month labels may have trained later folds"
        ),
        "interpretation": {
            "positive": "the causal OOF fitted-model sensitivity supports the emitted side",
            "negative": "the causal OOF fitted-model sensitivity opposes the emitted side",
            "zero": "no local directional contribution at machine precision",
            "full_logit": "exact additive standardized chosen-versus-opposite logit contribution",
            "full_hgb": (
                "local probability-margin sensitivity to replacing the group by "
                "fold-training medians; not additive and not causal"
            ),
        },
        "groups": {
            group: {"features": list(features), "plain_language": REASON_TEXT[group]}
            for group, features in REASON_GROUPS.items()
        },
        "top_groups_per_onset": 3,
        "recurring_month_requirement": 5,
        "recurring_stock_requirement": 15,
    }


def build_onset_reasons(
    onsets: pd.DataFrame,
    feature_surface: pd.DataFrame,
    bundles: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract frozen-group reasons before the final all-fold evaluation join."""

    if onsets.empty:
        return pd.DataFrame(), pd.DataFrame()
    surface_lookup = {
        horizon: build_anchor_surface(feature_surface, horizon).set_index("anchor_id")
        for horizon in HORIZONS
    }
    reason_rows: list[dict[str, Any]] = []
    for (month, horizon, algorithm, side), group in onsets.groupby(
        ["fold_month", "horizon", "candidate_algorithm", "side"], sort=False
    ):
        bundle = bundles[(str(month), int(horizon), str(algorithm))]
        features = tuple(bundle["features"])
        anchors = surface_lookup[int(horizon)].loc[group["anchor_id"].astype(str)]
        raw = anchors.loc[:, features].to_numpy(float)
        imputed = apply_medians(raw, np.asarray(bundle["medians"], float))
        estimator = bundle["estimator"]
        group_values: dict[str, np.ndarray]
        contribution_type: str
        intercept = np.full(len(group), np.nan)
        directional_margin: np.ndarray
        reconstruction_error = np.full(len(group), np.nan)
        if algorithm == "full_logit":
            scaler: StandardScaler = bundle["scaler"]
            standardized = scaler.transform(imputed)
            result = logit_reason_contributions(estimator, standardized, features, str(side))
            group_values = result["group_contributions"]
            intercept[:] = float(result["intercept"])
            directional_margin = np.asarray(result["directional_margin"], float)
            reconstruction = intercept + sum(group_values.values())
            reconstruction_error = reconstruction - directional_margin
            contribution_type = "exact_standardized_logit_margin_contribution"
        elif algorithm == "full_hgb":
            original_probability = _class_probability_matrix(estimator, imputed)
            chosen_index, opposite_index = (1, 2) if side == "long" else (2, 1)
            directional_margin = (
                original_probability[:, chosen_index] - original_probability[:, opposite_index]
            )
            group_values = {}
            for reason_group, reason_features in REASON_GROUPS.items():
                counterfactual = imputed.copy()
                indices = [features.index(feature) for feature in reason_features]
                counterfactual[:, indices] = np.asarray(bundle["medians"], float)[indices]
                counter_probability = _class_probability_matrix(estimator, counterfactual)
                counter_margin = (
                    counter_probability[:, chosen_index] - counter_probability[:, opposite_index]
                )
                group_values[reason_group] = directional_margin - counter_margin
            contribution_type = "local_median_replacement_probability_margin_sensitivity"
        else:
            raise AssertionError("reasons are only defined for frozen candidate algorithms")
        group_names = tuple(REASON_GROUPS)
        value_matrix = np.column_stack([group_values[name] for name in group_names])
        onset_records = group.reset_index(drop=True)
        for row_index, onset in onset_records.iterrows():
            ordering = sorted(
                range(len(group_names)),
                key=lambda index: (-abs(float(value_matrix[row_index, index])), group_names[index]),
            )
            ranks = {feature_index: rank + 1 for rank, feature_index in enumerate(ordering[:3])}
            for feature_index, reason_group in enumerate(group_names):
                value = float(value_matrix[row_index, feature_index])
                if value > 1e-15:
                    direction = "supports_chosen"
                elif value < -1e-15:
                    direction = "opposes_chosen"
                else:
                    direction = "neutral"
                reason_rows.append(
                    {
                        "onset_id": onset["onset_id"],
                        "anchor_id": onset["anchor_id"],
                        "candidate_algorithm": algorithm,
                        "horizon": int(horizon),
                        "side": side,
                        "fold_month": month,
                        "symbol_norm": onset["symbol_norm"],
                        "reason_group": reason_group,
                        "plain_language": REASON_TEXT[reason_group],
                        "contribution_type": contribution_type,
                        "contribution_value": value,
                        "contribution_direction": direction,
                        "top_rank": ranks.get(feature_index),
                        "is_top_group": feature_index in ranks,
                        "directional_intercept": float(intercept[row_index]),
                        "directional_margin": float(directional_margin[row_index]),
                        "reconstruction_error": float(reconstruction_error[row_index]),
                    }
                )
    reasons = (
        pd.DataFrame(reason_rows)
        .sort_values(["onset_id", "reason_group"], kind="stable")
        .reset_index(drop=True)
    )
    expected_rows = len(onsets) * len(REASON_GROUPS)
    if len(reasons) != expected_rows or reasons.duplicated(["onset_id", "reason_group"]).any():
        raise AssertionError("reason ledger support drift")
    logit = reasons["candidate_algorithm"].eq("full_logit")
    if logit.any() and float(reasons.loc[logit, "reconstruction_error"].abs().max()) > 1e-10:
        raise AssertionError("stored logit reasons do not reconstruct margin")
    top = reasons.loc[reasons["is_top_group"]].copy()
    recurring = (
        top.groupby(
            [
                "candidate_algorithm",
                "horizon",
                "side",
                "reason_group",
                "contribution_direction",
            ],
            sort=False,
        )
        .agg(
            onset_count=("onset_id", "size"),
            months=("fold_month", "nunique"),
            stocks=("symbol_norm", "nunique"),
        )
        .reset_index()
    )
    recurring["recurring_observable_sign"] = (
        recurring["months"].ge(5)
        & recurring["stocks"].ge(15)
        & recurring["contribution_direction"].ne("neutral")
    )
    return reasons, recurring


def write_pre_outcome_freeze(
    probabilities: pd.DataFrame,
    thresholds: pd.DataFrame,
    states: pd.DataFrame,
    onsets: pd.DataFrame,
    controls: pd.DataFrame,
    reasons: pd.DataFrame,
    recurring: pd.DataFrame,
    preprocessing: pd.DataFrame,
    coefficients: pd.DataFrame,
    folds: pd.DataFrame,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist prediction choices before the final all-fold evaluation join.

    ``pre_outcome`` artifact filenames are retained as a documented legacy
    shorthand.  They do not imply that earlier completed validation-month
    labels were unavailable to later expanding folds.
    """

    validate_pre_outcome_ledger(probabilities)
    if not folds["same_score_month_training_label_rows"].eq(0).all():
        raise AssertionError("a fold used its own score-month labels")
    if not folds["prior_completed_validation_month_training_label_rows"].gt(0).any():
        raise AssertionError("expanding folds did not record prior validation labels")
    files: dict[str, tuple[pd.DataFrame | dict[str, Any], str]] = {
        "probabilities_pre_outcome.parquet": (probabilities, "parquet"),
        "thresholds_pre_outcome.csv": (thresholds, "csv"),
        "onset_state_ledger_pre_outcome.parquet": (states, "parquet"),
        "candidate_onsets_pre_outcome.parquet": (onsets, "parquet"),
        "matched_clock_controls_pre_outcome.parquet": (controls, "parquet"),
        "onset_reasons_pre_outcome.parquet": (reasons, "parquet"),
        "recurring_reason_summary_pre_outcome.csv": (recurring, "csv"),
        "fold_preprocessing.csv": (preprocessing, "csv"),
        "logit_coefficients.csv": (coefficients, "csv"),
        "fold_metadata.csv": (folds, "csv"),
        "reason_dictionary_pre_outcome.json": (reason_dictionary(), "json"),
    }
    written: list[Path] = []
    for name, (value, file_type) in files.items():
        path = OUT / name
        if file_type == "parquet":
            assert isinstance(value, pd.DataFrame)
            value.to_parquet(path, index=False)
        elif file_type == "csv":
            assert isinstance(value, pd.DataFrame)
            value.to_csv(path, index=False)
        elif file_type == "json":
            assert isinstance(value, dict)
            write_json(path, value)
        else:
            raise AssertionError(f"unknown artifact format {file_type}")
        written.append(path)
    freeze = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "stage": "global_bundle_written_before_final_all_fold_evaluation_join",
        "artifact_filename_semantics": (
            "pre_outcome is legacy shorthand for pre_final_evaluation_join; "
            "prior completed validation-month labels trained later folds"
        ),
        "validation_paths_present": False,
        "final_all_fold_evaluation_path_table_present": False,
        "same_score_month_outcomes_read_before_that_month_probability": False,
        "prior_completed_validation_month_path_labels_used_in_later_folds": True,
        "global_bundle_written_before_any_validation_outcome_is_read": False,
        "global_bundle_written_before_final_all_fold_evaluation_join": True,
        "horizon_cooldown_used": False,
        "terminal_return_or_economic_outcome_used": False,
        "probability_rows": len(probabilities),
        "onset_rows": len(onsets),
        "matched_control_rows": int(controls["matched"].sum()),
        "reason_rows": len(reasons),
        "artifact_sha256": {path.name: sha256(path) for path in written},
        "frozen_source_manifest_sha256": sha256(PRE_SCORE_PATH),
        "frozen_source_sha256": source_manifest["sha256"],
    }
    freeze_path = OUT / "prediction_onset_control_reason_freeze.json"
    write_json(freeze_path, freeze)
    freeze["freeze_manifest_sha256"] = sha256(freeze_path)
    return freeze


def verify_pre_outcome_freeze(freeze: Mapping[str, Any]) -> None:
    freeze_path = OUT / "prediction_onset_control_reason_freeze.json"
    if sha256(freeze_path) != freeze["freeze_manifest_sha256"]:
        raise AssertionError("pre-outcome freeze manifest changed")
    observed = {name: sha256(OUT / name) for name in freeze["artifact_sha256"]}
    _require_equal(observed, freeze["artifact_sha256"], "pre-outcome artifact hashes")
    _require_equal(freeze["validation_paths_present"], False, "freeze validation path flag")
    _require_equal(
        freeze["final_all_fold_evaluation_path_table_present"],
        False,
        "freeze final evaluation table flag",
    )
    _require_equal(
        freeze["same_score_month_outcomes_read_before_that_month_probability"],
        False,
        "freeze same-month chronology",
    )
    _require_equal(
        freeze["prior_completed_validation_month_path_labels_used_in_later_folds"],
        True,
        "freeze prior-month chronology",
    )
    _require_equal(
        freeze["global_bundle_written_before_any_validation_outcome_is_read"],
        False,
        "freeze globally unread chronology",
    )
    _require_equal(
        freeze["global_bundle_written_before_final_all_fold_evaluation_join"],
        True,
        "freeze final evaluation chronology",
    )
    _require_equal(
        freeze["terminal_return_or_economic_outcome_used"],
        False,
        "freeze economic outcome flag",
    )


def build_validation_paths(tape: pd.DataFrame, feature_surface: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        surface = build_anchor_surface(feature_surface, horizon)
        validation = surface.loc[surface["month_key"].isin(VALIDATION_MONTHS)].copy()
        parts.append(attach_path_labels(tape, validation, horizon))
        del surface, validation
        gc.collect()
    paths = (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            ["horizon", "symbol_norm", "session_date", "decision_timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if paths.duplicated("anchor_id").any():
        raise AssertionError("duplicate validation path")
    _require_equal(
        {horizon: int((paths["horizon"] == horizon).sum()) for horizon in HORIZONS},
        EXPECTED_VALIDATION_ROWS,
        "validation path support",
    )
    return paths


def _weighted_mean(values: Sequence[float] | np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, float)
    valid = np.isfinite(x) & np.isfinite(weights)
    if not valid.any() or float(weights[valid].sum()) <= 0.0:
        return math.nan
    return float(np.average(x[valid], weights=weights[valid]))


def probability_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "rows": 0,
            "multiclass_log_loss": math.nan,
            "multiclass_brier": math.nan,
            "top_class_accuracy": math.nan,
            "macro_ovr_auc": math.nan,
        }
    weights = metric_weights(frame)
    probabilities = frame.loc[:, CLASS_COLUMNS].to_numpy(float)
    target = frame["target_class"].to_numpy(int)
    chosen = probabilities[np.arange(len(frame)), target]
    log_loss_value = _weighted_mean(-np.log(np.clip(chosen, 1e-15, 1.0)), weights)
    one_hot = np.eye(3, dtype=float)[target]
    brier = _weighted_mean(np.sum((probabilities - one_hot) ** 2, axis=1), weights)
    accuracy = _weighted_mean((probabilities.argmax(axis=1) == target).astype(float), weights)
    try:
        auc = float(
            roc_auc_score(
                target,
                probabilities,
                labels=[0, 1, 2],
                multi_class="ovr",
                average="macro",
                sample_weight=weights,
            )
        )
    except ValueError:
        auc = math.nan
    return {
        "rows": len(frame),
        "multiclass_log_loss": log_loss_value,
        "multiclass_brier": brier,
        "top_class_accuracy": accuracy,
        "macro_ovr_auc": auc,
    }


def evaluate_probability_metrics(
    probabilities: pd.DataFrame, paths: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = probabilities.loc[probabilities["fold_month"].isin(VALIDATION_MONTHS)].merge(
        paths[["anchor_id", "target_class"]],
        on="anchor_id",
        how="left",
        validate="many_to_one",
    )
    if validation["target_class"].isna().any():
        raise AssertionError("probability row lacks validation path class")
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for (algorithm, horizon), group in validation.groupby(["algorithm", "horizon"], sort=False):
        slices: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", group)]
        slices.extend(
            ("month", str(key), item) for key, item in group.groupby("fold_month", sort=False)
        )
        slices.extend(
            ("stock", str(key), item) for key, item in group.groupby("symbol_norm", sort=False)
        )
        slices.extend(
            (
                "leave_one_stock_out",
                str(symbol),
                group.loc[~group["symbol_norm"].eq(symbol)],
            )
            for symbol in SYMBOLS
        )
        for slice_type, slice_value, subset in slices:
            metric_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    **probability_statistics(subset),
                }
            )
        weights = metric_weights(group)
        target = group["target_class"].to_numpy(int)
        for class_value, probability_column in enumerate(CLASS_COLUMNS):
            class_probability = group[probability_column].to_numpy(float)
            bin_index = np.minimum((class_probability * 10.0).astype(int), 9)
            for calibration_bin in range(10):
                selected = bin_index == calibration_bin
                calibration_rows.append(
                    {
                        "algorithm": algorithm,
                        "horizon": int(horizon),
                        "class_value": class_value,
                        "probability_bin": calibration_bin,
                        "lower_bound": calibration_bin / 10.0,
                        "upper_bound": (calibration_bin + 1) / 10.0,
                        "rows": int(selected.sum()),
                        "weighted_mean_probability": _weighted_mean(
                            class_probability[selected], weights[selected]
                        ),
                        "weighted_observed_rate": _weighted_mean(
                            (target[selected] == class_value).astype(float),
                            weights[selected],
                        ),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(calibration_rows)


def orient_paths_for_side(frame: pd.DataFrame, side_column: str = "side") -> pd.DataFrame:
    """Orient path evidence to an imposed long/short entry-sign side."""

    output = frame.copy()
    long_side = output[side_column].eq("long").to_numpy(bool)
    target = output["target_class"].to_numpy(int)
    correct = np.where(long_side, target == 1, target == 2)
    wrong = np.where(long_side, target == 2, target == 1)
    ambiguous = output["status"].eq("intrabar_ambiguous").to_numpy(bool)
    no_hit = output["status"].eq("no_hit_by_horizon").to_numpy(bool)
    upside = output["upside_mfe_scale_units"].to_numpy(float)
    downside = output["downside_mfe_scale_units"].to_numpy(float)
    output["conservative_correct"] = correct.astype(np.int8)
    output["wrong_first"] = wrong.astype(np.int8)
    output["no_hit"] = no_hit.astype(np.int8)
    output["ambiguous"] = ambiguous.astype(np.int8)
    output["resolved"] = (correct | wrong).astype(np.int8)
    output["rapid_correct"] = (
        correct & output["first_confirmation_step"].le(3).fillna(False).to_numpy(bool)
    ).astype(np.int8)
    output["favourable_excursion_scale_units"] = np.where(long_side, upside, downside)
    output["adverse_excursion_scale_units"] = np.where(long_side, downside, upside)
    output["pre_confirmation_adverse_scale_units"] = np.where(
        long_side,
        output["long_pre_confirmation_adverse_scale_units"].to_numpy(float),
        output["short_pre_confirmation_adverse_scale_units"].to_numpy(float),
    )
    output["directional_dominance_scale_units"] = (
        output["favourable_excursion_scale_units"] - output["adverse_excursion_scale_units"]
    )
    output["clock_quartile"] = np.minimum(output["bar_ordinal"].to_numpy(int) * 4 // 78, 3).astype(
        np.int8
    )
    return output


def build_scored_events(
    onsets: pd.DataFrame,
    controls: pd.DataFrame,
    paths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path_columns = [
        "anchor_id",
        "target_class",
        "status",
        "first_confirmation_step",
        "upside_mfe_scale_units",
        "downside_mfe_scale_units",
        "long_pre_confirmation_adverse_scale_units",
        "short_pre_confirmation_adverse_scale_units",
    ]
    candidate = onsets.merge(
        paths[path_columns], on="anchor_id", how="left", validate="many_to_one"
    )
    if candidate["target_class"].isna().any():
        raise AssertionError("candidate onset lacks validation path")
    candidate = orient_paths_for_side(candidate)
    matched = controls.loc[controls["matched"]].copy()
    control_paths = paths[path_columns].rename(columns={"anchor_id": "control_anchor_id"})
    scored_controls = matched.merge(
        control_paths,
        on="control_anchor_id",
        how="left",
        validate="many_to_one",
    )
    if scored_controls["target_class"].isna().any():
        raise AssertionError("matched control lacks validation path")
    scored_controls = scored_controls.rename(
        columns={
            "candidate_algorithm": "candidate_algorithm",
            "candidate_symbol_norm": "symbol_norm",
            "candidate_session_date": "session_date",
            "candidate_fold_month": "fold_month",
            "candidate_bar_ordinal": "bar_ordinal",
        }
    )
    scored_controls = orient_paths_for_side(scored_controls)
    event_columns = [
        "onset_id",
        "conservative_correct",
        "wrong_first",
        "no_hit",
        "ambiguous",
        "rapid_correct",
        "favourable_excursion_scale_units",
        "adverse_excursion_scale_units",
        "pre_confirmation_adverse_scale_units",
        "directional_dominance_scale_units",
        "first_confirmation_step",
    ]
    pair = candidate[
        [
            "onset_id",
            "candidate_algorithm",
            "horizon",
            "side",
            "symbol_norm",
            "session_date",
            "fold_month",
            "clock_quartile",
            *event_columns[1:],
        ]
    ].merge(
        scored_controls[event_columns],
        on="onset_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_control"),
    )
    return candidate, scored_controls, pair


def event_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "onsets": 0,
            "stocks": 0,
            "conservative_correct_first_precision": math.nan,
            "wrong_first_rate": math.nan,
            "no_hit_rate": math.nan,
            "ambiguous_rate": math.nan,
            "resolved_only_precision": math.nan,
            "rapid_correct_within_3_rate": math.nan,
            "mean_favourable_excursion_scale_units": math.nan,
            "mean_adverse_excursion_scale_units": math.nan,
            "mean_pre_confirmation_adverse_scale_units": math.nan,
            "mean_directional_dominance_scale_units": math.nan,
            "mean_first_confirmation_step": math.nan,
        }
    weights = metric_weights(frame)
    correct = frame["conservative_correct"].to_numpy(float)
    resolved = frame["conservative_correct"].to_numpy(float) + frame["wrong_first"].to_numpy(float)
    resolved_weight = weights * resolved
    resolved_precision = (
        float(np.sum(weights * correct) / np.sum(resolved_weight))
        if float(np.sum(resolved_weight)) > 0.0
        else math.nan
    )
    return {
        "onsets": len(frame),
        "stocks": int(frame["symbol_norm"].nunique()),
        "conservative_correct_first_precision": _weighted_mean(correct, weights),
        "wrong_first_rate": _weighted_mean(frame["wrong_first"], weights),
        "no_hit_rate": _weighted_mean(frame["no_hit"], weights),
        "ambiguous_rate": _weighted_mean(frame["ambiguous"], weights),
        "resolved_only_precision": resolved_precision,
        "rapid_correct_within_3_rate": _weighted_mean(frame["rapid_correct"], weights),
        "mean_favourable_excursion_scale_units": _weighted_mean(
            frame["favourable_excursion_scale_units"], weights
        ),
        "mean_adverse_excursion_scale_units": _weighted_mean(
            frame["adverse_excursion_scale_units"], weights
        ),
        "mean_pre_confirmation_adverse_scale_units": _weighted_mean(
            frame["pre_confirmation_adverse_scale_units"], weights
        ),
        "mean_directional_dominance_scale_units": _weighted_mean(
            frame["directional_dominance_scale_units"], weights
        ),
        "mean_first_confirmation_step": _weighted_mean(frame["first_confirmation_step"], weights),
    }


def _event_slices(group: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "pooled", "all", group
    for key, item in group.groupby("fold_month", sort=False):
        yield "month", str(key), item
    for key, item in group.groupby("symbol_norm", sort=False):
        yield "stock", str(key), item
    for symbol in SYMBOLS:
        yield (
            "leave_one_stock_out",
            symbol,
            group.loc[~group["symbol_norm"].eq(symbol)],
        )
    for key, item in group.groupby("clock_quartile", sort=False):
        yield "clock_quartile", str(int(key)), item


def evaluate_event_metrics(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for sample_kind, frame in (("candidate", candidates), ("matched_clock_control", controls)):
        for (algorithm, horizon, side), group in frame.groupby(
            ["candidate_algorithm", "horizon", "side"], sort=False
        ):
            for slice_type, slice_value, subset in _event_slices(group):
                metric_rows.append(
                    {
                        "candidate_algorithm": algorithm,
                        "horizon": int(horizon),
                        "side": side,
                        "sample_kind": sample_kind,
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        **event_statistics(subset),
                    }
                )

    lift_frame = pairs.copy()
    lift_frame["precision_lift"] = (
        lift_frame["conservative_correct_candidate"] - lift_frame["conservative_correct_control"]
    )
    lift_frame["rapid_success_lift"] = (
        lift_frame["rapid_correct_candidate"] - lift_frame["rapid_correct_control"]
    )
    lift_frame["directional_dominance_lift"] = (
        lift_frame["directional_dominance_scale_units_candidate"]
        - lift_frame["directional_dominance_scale_units_control"]
    )
    lift_frame["pre_confirmation_adverse_improvement"] = (
        lift_frame["pre_confirmation_adverse_scale_units_control"]
        - lift_frame["pre_confirmation_adverse_scale_units_candidate"]
    )
    lift_frame["favourable_excursion_lift"] = (
        lift_frame["favourable_excursion_scale_units_candidate"]
        - lift_frame["favourable_excursion_scale_units_control"]
    )
    lift_rows: list[dict[str, Any]] = []
    for (algorithm, horizon, side), group in lift_frame.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        for slice_type, slice_value, subset in _event_slices(group):
            weights = metric_weights(subset) if not subset.empty else np.asarray([])
            candidate_precision = (
                _weighted_mean(subset["conservative_correct_candidate"], weights)
                if not subset.empty
                else math.nan
            )
            control_precision = (
                _weighted_mean(subset["conservative_correct_control"], weights)
                if not subset.empty
                else math.nan
            )
            lift_rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "matched_pairs": len(subset),
                    "candidate_precision": candidate_precision,
                    "control_precision": control_precision,
                    "precision_lift": candidate_precision - control_precision,
                    "relative_precision_lift": (
                        (candidate_precision - control_precision) / control_precision
                        if control_precision > 0.0
                        else math.nan
                    ),
                    "rapid_success_lift": _weighted_mean(subset["rapid_success_lift"], weights)
                    if not subset.empty
                    else math.nan,
                    "directional_dominance_lift": _weighted_mean(
                        subset["directional_dominance_lift"], weights
                    )
                    if not subset.empty
                    else math.nan,
                    "pre_confirmation_adverse_improvement": _weighted_mean(
                        subset["pre_confirmation_adverse_improvement"], weights
                    )
                    if not subset.empty
                    else math.nan,
                    "favourable_excursion_lift": _weighted_mean(
                        subset["favourable_excursion_lift"], weights
                    )
                    if not subset.empty
                    else math.nan,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(lift_rows)


def paired_moving_block_interval(
    frame: pd.DataFrame,
    value_column: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    random_state: int = SEED,
) -> dict[str, float | int]:
    """Paired non-circular overlapping five-session moving-block interval."""

    if frame.empty:
        return {"observed": math.nan, "lower": math.nan, "upper": math.nan, "draws": draws}
    session_means = (
        frame.groupby(["session_date", "symbol_norm"], sort=True)[value_column]
        .mean()
        .unstack("symbol_norm")
        .sort_index()
        .reindex(columns=sorted(frame["symbol_norm"].unique()))
    )
    dates = session_means.index.to_numpy()
    matrix = session_means.to_numpy(float)
    n_dates = len(dates)
    if n_dates < BOOTSTRAP_BLOCK:
        raise AssertionError("insufficient sessions for frozen moving-block bootstrap")
    observed_symbol = np.nanmean(matrix, axis=0)
    observed = float(np.nanmean(observed_symbol))
    rng = np.random.default_rng(random_state)
    blocks_needed = math.ceil(n_dates / BOOTSTRAP_BLOCK)
    start_max = n_dates - BOOTSTRAP_BLOCK
    samples = np.empty(draws, dtype=float)
    finite = np.isfinite(matrix)
    filled = np.where(finite, matrix, 0.0)
    for draw in range(draws):
        starts = rng.integers(0, start_max + 1, size=blocks_needed)
        indices = np.concatenate([np.arange(start, start + BOOTSTRAP_BLOCK) for start in starts])[
            :n_dates
        ]
        counts = np.bincount(indices, minlength=n_dates).astype(float)
        numerator = (filled * counts[:, None]).sum(axis=0)
        denominator = (finite * counts[:, None]).sum(axis=0)
        symbol_values = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0.0,
        )
        samples[draw] = np.nanmean(symbol_values)
    return {
        "observed": observed,
        "lower": float(np.quantile(samples, LOWER_QUANTILE)),
        "upper": float(np.quantile(samples, UPPER_QUANTILE)),
        "draws": draws,
    }


def build_event_bootstraps(pairs: pd.DataFrame) -> pd.DataFrame:
    work = pairs.copy()
    work["precision_lift"] = (
        work["conservative_correct_candidate"] - work["conservative_correct_control"]
    )
    work["rapid_success_lift"] = work["rapid_correct_candidate"] - work["rapid_correct_control"]
    work["directional_dominance_lift"] = (
        work["directional_dominance_scale_units_candidate"]
        - work["directional_dominance_scale_units_control"]
    )
    work["pre_confirmation_adverse_improvement"] = (
        work["pre_confirmation_adverse_scale_units_control"]
        - work["pre_confirmation_adverse_scale_units_candidate"]
    )
    rows: list[dict[str, Any]] = []
    for (algorithm, horizon, side), group in work.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        for metric_index, metric in enumerate(
            (
                "precision_lift",
                "rapid_success_lift",
                "directional_dominance_lift",
                "pre_confirmation_adverse_improvement",
            )
        ):
            interval = paired_moving_block_interval(
                group,
                metric,
                random_state=SEED
                + int(horizon) * 100
                + (0 if side == "long" else 10)
                + (0 if algorithm == "full_logit" else 1)
                + metric_index,
            )
            rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "metric": metric,
                    **interval,
                    "lower_quantile": LOWER_QUANTILE,
                    "upper_quantile": UPPER_QUANTILE,
                    "block_sessions": BOOTSTRAP_BLOCK,
                }
            )
    return pd.DataFrame(rows)


def probability_comparisons(probability_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = probability_metrics.loc[
        probability_metrics["algorithm"].isin(("clock_logit", *CANDIDATES))
    ].copy()
    keys = ["horizon", "slice_type", "slice_value"]
    baseline = selected.loc[selected["algorithm"].eq("clock_logit")].set_index(keys)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_rows = selected.loc[selected["algorithm"].eq(candidate)].set_index(keys)
        if not candidate_rows.index.equals(baseline.index):
            candidate_rows = candidate_rows.reindex(baseline.index)
        for key, candidate_row in candidate_rows.iterrows():
            baseline_row = baseline.loc[key]
            rows.append(
                {
                    "candidate_algorithm": candidate,
                    "horizon": int(key[0]),
                    "slice_type": key[1],
                    "slice_value": key[2],
                    "rows": int(candidate_row["rows"]),
                    "log_loss_improvement_vs_clock": float(
                        baseline_row["multiclass_log_loss"] - candidate_row["multiclass_log_loss"]
                    ),
                    "brier_improvement_vs_clock": float(
                        baseline_row["multiclass_brier"] - candidate_row["multiclass_brier"]
                    ),
                    "accuracy_lift_vs_clock": float(
                        candidate_row["top_class_accuracy"] - baseline_row["top_class_accuracy"]
                    ),
                    "auc_lift_vs_clock": float(
                        candidate_row["macro_ovr_auc"] - baseline_row["macro_ovr_auc"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def control_match_summary(onsets: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    onset_counts = (
        onsets.groupby(["candidate_algorithm", "horizon", "side"], sort=False)
        .size()
        .rename("onsets")
    )
    matched_counts = (
        controls.groupby(["candidate_algorithm", "horizon", "side"], sort=False)["matched"]
        .sum()
        .rename("matched_controls")
    )
    summary = pd.concat([onset_counts, matched_counts], axis=1).fillna(0).reset_index()
    summary["onsets"] = summary["onsets"].astype(int)
    summary["matched_controls"] = summary["matched_controls"].astype(int)
    summary["match_rate"] = summary["matched_controls"] / summary["onsets"]
    return summary


def final_decisions(
    probability_comparison: pd.DataFrame,
    onsets: pd.DataFrame,
    match_summary: pd.DataFrame,
    event_lifts: pd.DataFrame,
    bootstraps: pd.DataFrame,
    recurring_reasons: pd.DataFrame,
) -> dict[str, Any]:
    """Apply the frozen all-required research gates without economic claims."""

    probability: dict[str, Any] = {}
    for algorithm in CANDIDATES:
        detail: dict[str, Any] = {}
        all_horizons = True
        for horizon in HORIZONS:
            subset = probability_comparison.loc[
                probability_comparison["candidate_algorithm"].eq(algorithm)
                & probability_comparison["horizon"].eq(horizon)
            ]
            pooled = subset.loc[subset["slice_type"].eq("pooled")].iloc[0]
            months = subset.loc[subset["slice_type"].eq("month")]
            deletions = subset.loc[subset["slice_type"].eq("leave_one_stock_out")]
            gates = {
                "pooled_log_loss_better": bool(pooled["log_loss_improvement_vs_clock"] > 0.0),
                "pooled_brier_better": bool(pooled["brier_improvement_vs_clock"] > 0.0),
                "months_both_better_at_least_4": int(
                    (
                        months["log_loss_improvement_vs_clock"].gt(0.0)
                        & months["brier_improvement_vs_clock"].gt(0.0)
                    ).sum()
                )
                >= 4,
                "stock_deletions_both_better_at_least_18": int(
                    (
                        deletions["log_loss_improvement_vs_clock"].gt(0.0)
                        & deletions["brier_improvement_vs_clock"].gt(0.0)
                    ).sum()
                )
                >= 18,
            }
            passed = all(gates.values())
            detail[f"h{horizon}"] = {
                "passed": passed,
                "gates": gates,
                "log_loss_improvement_vs_clock": float(pooled["log_loss_improvement_vs_clock"]),
                "brier_improvement_vs_clock": float(pooled["brier_improvement_vs_clock"]),
            }
            all_horizons &= passed
        probability[algorithm] = {
            "retained": all_horizons,
            "interpretation": "2024 internal probability hypothesis only"
            if all_horizons
            else "rejected as a recurring probability hypothesis",
            "horizons": detail,
        }

    candidate_side: dict[str, Any] = {}
    for algorithm in CANDIDATES:
        for side in ("long", "short"):
            key = f"{algorithm}_{side}"
            horizon_detail: dict[str, Any] = {}
            all_horizons = bool(probability[algorithm]["retained"])
            for horizon in HORIZONS:
                all_algorithm_horizon = onsets.loc[
                    onsets["candidate_algorithm"].eq(algorithm) & onsets["horizon"].eq(horizon)
                ]
                side_onsets = all_algorithm_horizon.loc[all_algorithm_horizon["side"].eq(side)]
                pooled = event_lifts.loc[
                    event_lifts["candidate_algorithm"].eq(algorithm)
                    & event_lifts["horizon"].eq(horizon)
                    & event_lifts["side"].eq(side)
                    & event_lifts["slice_type"].eq("pooled")
                ].iloc[0]
                months = event_lifts.loc[
                    event_lifts["candidate_algorithm"].eq(algorithm)
                    & event_lifts["horizon"].eq(horizon)
                    & event_lifts["side"].eq(side)
                    & event_lifts["slice_type"].eq("month")
                ]
                deletions = event_lifts.loc[
                    event_lifts["candidate_algorithm"].eq(algorithm)
                    & event_lifts["horizon"].eq(horizon)
                    & event_lifts["side"].eq(side)
                    & event_lifts["slice_type"].eq("leave_one_stock_out")
                ]
                match = match_summary.loc[
                    match_summary["candidate_algorithm"].eq(algorithm)
                    & match_summary["horizon"].eq(horizon)
                    & match_summary["side"].eq(side)
                ].iloc[0]
                interval = bootstraps.loc[
                    bootstraps["candidate_algorithm"].eq(algorithm)
                    & bootstraps["horizon"].eq(horizon)
                    & bootstraps["side"].eq(side)
                ].set_index("metric")
                monthly_total = all_algorithm_horizon.groupby("fold_month").size()
                gates = {
                    "candidate_onsets_at_least_500": len(all_algorithm_horizon) >= 500,
                    "every_month_candidate_onsets_at_least_50": bool(
                        monthly_total.reindex(VALIDATION_MONTHS, fill_value=0).ge(50).all()
                    ),
                    "candidate_stocks_at_least_15": all_algorithm_horizon["symbol_norm"].nunique()
                    >= 15,
                    "side_onsets_at_least_100": len(side_onsets) >= 100,
                    "side_stocks_at_least_10": side_onsets["symbol_norm"].nunique() >= 10,
                    "matched_control_rate_at_least_0_95": float(match["match_rate"]) >= 0.95,
                    "absolute_precision_lift_at_least_0_05": float(pooled["precision_lift"])
                    >= 0.05,
                    "relative_precision_lift_at_least_0_10": float(
                        pooled["relative_precision_lift"]
                    )
                    >= 0.10,
                    "rapid_success_lift_at_least_0_02": float(pooled["rapid_success_lift"]) >= 0.02,
                    "directional_dominance_lift_positive": float(
                        pooled["directional_dominance_lift"]
                    )
                    > 0.0,
                    "pre_confirmation_adverse_not_worse": float(
                        pooled["pre_confirmation_adverse_improvement"]
                    )
                    >= 0.0,
                    "bootstrap_precision_positive": float(interval.loc["precision_lift", "lower"])
                    > 0.0,
                    "bootstrap_rapid_positive": float(interval.loc["rapid_success_lift", "lower"])
                    > 0.0,
                    "bootstrap_dominance_positive": float(
                        interval.loc["directional_dominance_lift", "lower"]
                    )
                    > 0.0,
                    "bootstrap_adverse_not_worse": float(
                        interval.loc["pre_confirmation_adverse_improvement", "lower"]
                    )
                    >= 0.0,
                    "positive_precision_lift_months_at_least_4": int(
                        months["precision_lift"].gt(0.0).sum()
                    )
                    >= 4,
                    "positive_precision_lift_stock_deletions_at_least_18": int(
                        deletions["precision_lift"].gt(0.0).sum()
                    )
                    >= 18,
                }
                passed = all(gates.values())
                horizon_detail[f"h{horizon}"] = {
                    "passed": passed,
                    "gates": gates,
                    "onsets": len(side_onsets),
                    "matched_control_rate": float(match["match_rate"]),
                    "precision_lift": float(pooled["precision_lift"]),
                    "rapid_success_lift": float(pooled["rapid_success_lift"]),
                    "directional_dominance_lift": float(pooled["directional_dominance_lift"]),
                    "pre_confirmation_adverse_improvement": float(
                        pooled["pre_confirmation_adverse_improvement"]
                    ),
                }
                all_horizons &= passed
            candidate_side[key] = {
                "retained": all_horizons,
                "interpretation": (
                    "recurring 2024 internal entry-sign hypothesis; prospective shadow required"
                    if all_horizons
                    else "rejected or descriptive only"
                ),
                "horizons": horizon_detail,
            }
    recurring = (
        recurring_reasons.loc[recurring_reasons["recurring_observable_sign"]]
        if not recurring_reasons.empty
        else recurring_reasons
    )
    return {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "economic_edge_claim_permitted": False,
        "validation_design": {
            "kind": "causal_monthly_expanding_oof_development",
            "same_score_month_outcomes_read_before_that_month_probability": False,
            "prior_completed_validation_month_path_labels_used_in_later_folds": True,
            "globally_sealed_validation_period": False,
        },
        "probability_hypotheses": probability,
        "candidate_side_hypotheses": candidate_side,
        "recurring_observable_signs": safe(recurring.to_dict("records")),
        "any_entry_sign_retained": any(item["retained"] for item in candidate_side.values()),
    }


def validate_only() -> dict[str, Any]:
    """Target-blind tape/feature/scale/support validation; no model or path label."""

    contract, _ = load_contract_and_verify(require_pre_outcome=False)
    tape, diagnostics = load_tape(return_diagnostics=True)
    features = build_feature_surface(tape)
    support: dict[str, Any] = {}
    for horizon in HORIZONS:
        surface = build_anchor_surface(features, horizon)
        support[f"h{horizon}"] = {
            "annual_rows": len(surface),
            "june_rows": int(surface["month_key"].eq("2024-06").sum()),
            "validation_rows": int(surface["month_key"].isin(VALIDATION_MONTHS).sum()),
            "minimum_scale_bps": float(surface["causal_scale_bps"].min()),
            "minimum_segment_position": int(surface["segment_position"].min()),
        }
        del surface
        gc.collect()
    return {
        "contract_id": contract["contract_id"],
        "mode": "validate_only_target_blind",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "validation_paths_or_labels_constructed": False,
        "models_fitted": False,
        "output_directory_written": False,
        "tape": diagnostics,
        "support": support,
        "environment_versions": environment_versions(),
        "source_hashes_for_future_freeze": current_source_hashes(),
    }


def run_scoring() -> dict[str, Any]:
    """Execute the frozen causal monthly expanding-OOF research experiment."""

    _, source_manifest = load_contract_and_verify(require_pre_outcome=True)
    assert source_manifest is not None
    if OUT.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen artifact directory {OUT}; use a new version"
        )
    OUT.mkdir(parents=True, exist_ok=False)
    source_binding = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "sha256": current_source_hashes(),
        "environment_versions": environment_versions(),
        "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
        "fold_chronology": {
            "same_score_month_outcomes_read_before_that_month_probability": False,
            "prior_completed_validation_month_path_labels_used_in_later_folds": True,
            "global_bundle_written_before_any_validation_outcome_is_read": False,
            "global_bundle_written_before_final_all_fold_evaluation_join": True,
        },
    }
    tape, tape_diagnostics = load_tape(return_diagnostics=True)
    feature_surface = build_feature_surface(tape)
    probabilities, preprocessing, coefficients, folds, bundles = fit_monthly_oof_probabilities(
        tape, feature_surface
    )
    thresholds = calibrate_prior_month_thresholds(probabilities)
    onsets, states = extract_onsets(probabilities, thresholds, return_state=True)
    controls = match_clock_controls(onsets, probabilities)
    reasons, recurring = build_onset_reasons(onsets, feature_surface, bundles)
    freeze = write_pre_outcome_freeze(
        probabilities,
        thresholds,
        states,
        onsets,
        controls,
        reasons,
        recurring,
        preprocessing,
        coefficients,
        folds,
        source_manifest,
    )
    verify_pre_outcome_freeze(freeze)
    source_binding["pre_outcome_freeze_manifest_sha256"] = freeze["freeze_manifest_sha256"]
    write_json(OUT / "source_hashes.json", source_binding)

    paths = build_validation_paths(tape, feature_surface)
    paths.to_parquet(OUT / "validation_paths.parquet", index=False)
    probability_metrics, probability_calibration = evaluate_probability_metrics(
        probabilities, paths
    )
    probability_comparison = probability_comparisons(probability_metrics)
    candidates, scored_controls, pairs = build_scored_events(onsets, controls, paths)
    event_metrics, event_lifts = evaluate_event_metrics(candidates, scored_controls, pairs)
    bootstraps = build_event_bootstraps(pairs)
    match_summary = control_match_summary(onsets, controls)
    decisions = final_decisions(
        probability_comparison,
        onsets,
        match_summary,
        event_lifts,
        bootstraps,
        recurring,
    )

    table_artifacts = {
        "probability_metrics.csv": probability_metrics,
        "probability_calibration.csv": probability_calibration,
        "probability_comparisons.csv": probability_comparison,
        "scored_candidate_onsets.parquet": candidates,
        "scored_matched_clock_controls.parquet": scored_controls,
        "paired_event_evidence.parquet": pairs,
        "event_metrics.csv": event_metrics,
        "event_lifts.csv": event_lifts,
        "event_bootstrap_intervals.csv": bootstraps,
        "control_match_summary.csv": match_summary,
    }
    for name, frame in table_artifacts.items():
        path = OUT / name
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
    write_json(OUT / "decision.json", decisions)
    summary = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "data": "regular-session five-minute provider OHLC; volume not read",
        "probability_rows": len(probabilities),
        "validation_path_rows": len(paths),
        "candidate_onsets": len(onsets),
        "matched_controls": int(controls["matched"].sum()),
        "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
        "pre_outcome_freeze_manifest_sha256": freeze["freeze_manifest_sha256"],
        "fold_chronology": source_binding["fold_chronology"],
        "artifact_filename_semantics": (
            "pre_outcome is legacy shorthand for pre_final_evaluation_join"
        ),
        "tape_diagnostics": tape_diagnostics,
        "decisions": decisions,
        "result_artifact_sha256": {
            name: sha256(OUT / name)
            for name in (
                "validation_paths.parquet",
                *table_artifacts.keys(),
                "decision.json",
            )
        },
    }
    write_json(OUT / "summary.json", summary)
    manifest_files = sorted(
        path for path in OUT.iterdir() if path.is_file() and path.name != "artifact_manifest.json"
    )
    write_json(
        OUT / "artifact_manifest.json",
        {
            "contract_id": CONTRACT_ID,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "stage": "pre_independent_audit_complete_artifact_manifest",
            "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
            "pre_outcome_freeze_manifest_sha256": freeze["freeze_manifest_sha256"],
            "fold_chronology": source_binding["fold_chronology"],
            "files_excluding_this_manifest": {path.name: sha256(path) for path in manifest_files},
        },
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate only target-blind 2024 tape/features/scale/support; do not score",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_only() if args.validate_only else run_scoring()
    print(json.dumps(safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
