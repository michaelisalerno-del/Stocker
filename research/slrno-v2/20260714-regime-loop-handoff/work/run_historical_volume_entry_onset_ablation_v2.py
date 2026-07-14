"""Research-only incremental historical-volume entry-onset ablation.

This is a separately versioned extension of the frozen raw-OHLC entry-onset
experiment.  It compares one fixed HGB model on the frozen 40 price/clock
features with the identical HGB model on those features plus eleven causal
features derived from provider ``historical_volume``.  Historical volume is
not order flow, exchange-wide consolidated volume, signed volume, quote
count, tick count, or order-book depth.

The 2024 entry-path outcomes were already known before this volume question
was frozen, so every result is post-outcome internal development.  No P&L,
execution, position, order, broker, paper, live, or deployment path exists.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_raw_ohlc_entry_onset_discovery_v1 as base
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260712-historical-volume-entry-onset-ablation-v2.json"
PRE_SCORE_PATH = (
    WORK / "contracts/20260712-historical-volume-entry-onset-ablation-v2-pre-score.json"
)
OUT = Path("/private/tmp/stocker_historical_volume_entry_onset_ablation_v2_20260712")
ENVIRONMENT_ROOT = Path("/Users/michaelsalerno/StockerLocal")

CONTRACT_ID = "historical_volume_entry_onset_ablation_v2"
SEED = 20260712
SYMBOLS = base.SYMBOLS
HORIZONS = base.HORIZONS
PREDICTION_MONTHS = base.PREDICTION_MONTHS
VALIDATION_MONTHS = base.VALIDATION_MONTHS
CLASS_COLUMNS = base.CLASS_COLUMNS
PRICE_FEATURES = base.FULL_FEATURES
ALGORITHMS = ("price_hgb", "price_historical_volume_hgb")
CANDIDATE = "price_historical_volume_hgb"
BASELINE = "price_hgb"
VOLUME_FEATURES = (
    "historical_volume_missing_current",
    "historical_volume_log_change_1",
    "historical_volume_log_ratio_prior_3",
    "historical_volume_log_ratio_prior_6",
    "historical_volume_log_ratio_prior_12",
    "historical_volume_log_ratio_segment_prior_mean",
    "historical_volume_recent_3_minus_older_12",
    "historical_volume_prior_12_log_std",
    "historical_volume_availability_12",
    "historical_volume_range_interaction_6",
    "historical_volume_body_interaction_6",
)
FEATURES_BY_ALGORITHM = {
    BASELINE: PRICE_FEATURES,
    CANDIDATE: (*PRICE_FEATURES, *VOLUME_FEATURES),
}
HGB_PARAMETERS = dict(base.HGB_PARAMETERS)
PREDICTION_COLUMNS = base.PRE_OUTCOME_LEDGER_COLUMNS

FIRE_QUANTILE = 0.95
REARM_QUANTILE = 0.75
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_BLOCK = 5
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975

EXPECTED_VOLUME_COVERAGE = {
    "accepted_price_valid_regular_rows": 424_583,
    "finite_positive_historical_volume_rows": 424_472,
    "missing_or_nonfinite_historical_volume_rows": 111,
    "zero_historical_volume_rows": 0,
    "negative_historical_volume_rows": 0,
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def current_source_hashes() -> dict[str, str]:
    result = {
        "contract": sha256(CONTRACT_PATH),
        "runner": sha256(Path(__file__).resolve()),
        "frozen_entry_onset_v1_runner": sha256(
            WORK / "run_raw_ohlc_entry_onset_discovery_v1.py"
        ),
        "frozen_clean_slate_runner": sha256(
            WORK / "run_clean_slate_causal_ohlc_entries_v1.py"
        ),
        "environment_pyproject": sha256(ENVIRONMENT_ROOT / "pyproject.toml"),
        "environment_uv_lock": sha256(ENVIRONMENT_ROOT / "uv.lock"),
    }
    for symbol in SYMBOLS:
        result[f"provider_full_file_{symbol}"] = sha256(base.clean_slate.provider_path(symbol))
    return result


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"contract drift for {label}: {observed!r} != {expected!r}")


def load_contract_and_verify(require_pre_score: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
    _require_equal(contract["sources"]["provider_volume_label"], "historical_volume", "volume")
    _require_equal(tuple(contract["universe"]["symbols"]), SYMBOLS, "symbols")
    _require_equal(tuple(contract["decision_and_path"]["horizons_bars"]), HORIZONS, "horizons")
    _require_equal(
        tuple(contract["historical_volume_features_in_order"]), VOLUME_FEATURES, "volume features"
    )
    _require_equal(contract["shared_estimator_parameters"], HGB_PARAMETERS, "HGB parameters")
    _require_equal(
        tuple(item["name"] for item in contract["algorithms"]), ALGORITHMS, "algorithms"
    )
    for year in (2023, 2025, 2026):
        _require_equal(contract["periods"][f"{year}_read_permitted"], False, f"{year} read")
    if not require_pre_score:
        return contract, None
    if not PRE_SCORE_PATH.is_file():
        raise FileNotFoundError(f"missing frozen pre-score manifest: {PRE_SCORE_PATH}")
    manifest = json.loads(PRE_SCORE_PATH.read_text())
    expected_keys = {
        "contract_id",
        "frozen_at_utc",
        "frozen_before_volume_scoring",
        "research_only",
        "live_ordering_enabled",
        "order_placement",
        "provider_volume_label",
        "scientific_status",
        "prior_price_only_outcomes_known_before_volume_contract",
        "same_score_month_outcomes_read_before_that_month_probability",
        "later_period_rows_read",
        "environment_versions",
        "sha256",
    }
    _require_equal(set(manifest), expected_keys, "pre-score manifest schema")
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "frozen_before_volume_scoring": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume",
        "prior_price_only_outcomes_known_before_volume_contract": True,
        "same_score_month_outcomes_read_before_that_month_probability": False,
        "later_period_rows_read": False,
    }.items():
        _require_equal(manifest[key], expected, f"pre-score {key}")
    _require_equal(manifest["environment_versions"], environment_versions(), "environment")
    _require_equal(manifest["sha256"], current_source_hashes(), "source hashes")
    return contract, manifest


def load_tape_with_historical_volume(
    *, return_diagnostics: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    price, tape_diagnostics = base.load_tape(return_diagnostics=True)
    parts: list[pd.DataFrame] = []
    for symbol in SYMBOLS:
        frame = pd.read_parquet(
            base.clean_slate.provider_path(symbol),
            columns=["timestamp", "volume"],
            filters=base.clean_slate._year_filter(),
        )
        if not frame["timestamp"].ge(pd.Timestamp("2024-01-01", tz="UTC")).all() or not frame[
            "timestamp"
        ].lt(pd.Timestamp("2025-01-01", tz="UTC")).all():
            raise AssertionError(f"non-2024 volume row materialized for {symbol}")
        if frame["timestamp"].duplicated().any():
            raise AssertionError(f"duplicate volume timestamp for {symbol}")
        frame = frame.rename(columns={"volume": "historical_volume"})
        frame["symbol_norm"] = symbol
        parts.append(frame)
    provider_volume = pd.concat(parts, ignore_index=True)
    merged = price.merge(
        provider_volume,
        on=["symbol_norm", "timestamp"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if not np.array_equal(merged["source_position"].to_numpy(), price["source_position"].to_numpy()):
        raise AssertionError("historical-volume merge changed tape order")
    raw = merged["historical_volume"].to_numpy(float)
    finite = np.isfinite(raw)
    negative = finite & (raw < 0.0)
    zero = finite & (raw == 0.0)
    positive = finite & (raw > 0.0)
    merged.loc[~finite | negative, "historical_volume"] = np.nan
    coverage = {
        "accepted_price_valid_regular_rows": len(merged),
        "finite_positive_historical_volume_rows": int(positive.sum()),
        "missing_or_nonfinite_historical_volume_rows": int((~finite).sum()),
        "zero_historical_volume_rows": int(zero.sum()),
        "negative_historical_volume_rows": int(negative.sum()),
        "minimum_positive_historical_volume": float(np.min(raw[positive])),
        "median_historical_volume": float(np.median(raw[positive])),
        "maximum_historical_volume": float(np.max(raw[positive])),
    }
    for key, expected in EXPECTED_VOLUME_COVERAGE.items():
        _require_equal(coverage[key], expected, f"volume coverage {key}")
    diagnostics = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume",
        "volume_is_order_flow": False,
        "volume_coverage": coverage,
        "price_tape": tape_diagnostics,
    }
    return (merged, diagnostics) if return_diagnostics else merged


def _volume_group_transform(
    values: pd.Series, tape: pd.DataFrame, function: Any
) -> pd.Series:
    keys = [tape["symbol_norm"], tape["session_date"], tape["segment_index"]]
    return values.groupby(keys, sort=False).transform(function)


def build_historical_volume_feature_surface(tape: pd.DataFrame) -> pd.DataFrame:
    surface = base.build_feature_surface(tape)
    raw = tape["historical_volume"].to_numpy(float)
    valid = np.isfinite(raw) & (raw >= 0.0)
    lv = pd.Series(np.where(valid, np.log1p(raw), np.nan), index=tape.index, dtype=float)

    prior_1 = _volume_group_transform(lv, tape, lambda s: s.shift(1))
    prior_3 = _volume_group_transform(
        lv, tape, lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    prior_6 = _volume_group_transform(
        lv, tape, lambda s: s.shift(1).rolling(6, min_periods=6).mean()
    )
    prior_12 = _volume_group_transform(
        lv, tape, lambda s: s.shift(1).rolling(12, min_periods=12).mean()
    )
    segment_prior = _volume_group_transform(
        lv, tape, lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    recent_3 = _volume_group_transform(lv, tape, lambda s: s.rolling(3, min_periods=3).mean())
    older_12 = _volume_group_transform(
        lv, tape, lambda s: s.shift(3).rolling(12, min_periods=12).mean()
    )
    prior_12_std = _volume_group_transform(
        lv, tape, lambda s: s.shift(1).rolling(12, min_periods=12).std(ddof=0)
    )
    availability_12 = _volume_group_transform(
        lv, tape, lambda s: s.rolling(12, min_periods=1).count() / 12.0
    )

    surface["historical_volume_missing_current"] = (~valid).astype(float)
    surface["historical_volume_log_change_1"] = lv - prior_1
    surface["historical_volume_log_ratio_prior_3"] = lv - prior_3
    surface["historical_volume_log_ratio_prior_6"] = lv - prior_6
    surface["historical_volume_log_ratio_prior_12"] = lv - prior_12
    surface["historical_volume_log_ratio_segment_prior_mean"] = lv - segment_prior
    surface["historical_volume_recent_3_minus_older_12"] = recent_3 - older_12
    surface["historical_volume_prior_12_log_std"] = prior_12_std
    surface["historical_volume_availability_12"] = availability_12
    surface["historical_volume_range_interaction_6"] = (
        surface["historical_volume_log_ratio_prior_6"] * surface["log_range_ratio_6"]
    )
    surface["historical_volume_body_interaction_6"] = (
        surface["historical_volume_log_ratio_prior_6"] * surface["signed_body_fraction"]
    )
    values = surface.loc[:, VOLUME_FEATURES].to_numpy(float)
    if np.isinf(values).any():
        raise AssertionError("infinite historical-volume feature")
    if not np.array_equal(surface["source_position"].to_numpy(), tape["source_position"].to_numpy()):
        raise AssertionError("volume feature surface order drift")
    return surface


def build_anchor_surface(surface: pd.DataFrame, horizon: int) -> pd.DataFrame:
    anchor = base.build_anchor_surface(surface, horizon)
    volume = surface[["source_position", *VOLUME_FEATURES]]
    anchor = anchor.merge(volume, on="source_position", how="left", sort=False, validate="one_to_one")
    if len(anchor) != base.EXPECTED_ANNUAL_ROWS[horizon]:
        raise AssertionError("volume anchor support drift")
    return anchor


def validate_prediction_ledger(frame: pd.DataFrame) -> None:
    _require_equal(tuple(frame.columns), PREDICTION_COLUMNS, "prediction schema")
    if frame.duplicated(["anchor_id", "algorithm"]).any():
        raise AssertionError("duplicate prediction key")
    if not set(frame["algorithm"]).issubset(ALGORITHMS):
        raise AssertionError("unknown algorithm")
    probabilities = frame.loc[:, CLASS_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
    ):
        raise AssertionError("invalid probabilities")


def fit_monthly_probabilities(
    tape: pd.DataFrame, feature_surface: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int, str], dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    preprocessing: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    bundles: dict[tuple[str, int, str], dict[str, Any]] = {}
    for horizon in HORIZONS:
        anchors = build_anchor_surface(feature_surface, horizon)
        cached = pd.DataFrame()
        cached_ids: set[str] = set()
        for month in PREDICTION_MONTHS:
            train_mask, score_mask = base.fold_masks(anchors, month)
            train = anchors.loc[train_mask].copy()
            score = anchors.loc[score_mask].copy()
            missing = train.loc[~train["anchor_id"].isin(cached_ids)]
            if not missing.empty:
                added = base.attach_path_labels(tape, missing, horizon)
                cached = pd.concat([cached, added], ignore_index=True)
                cached_ids.update(added["anchor_id"].astype(str))
            y = train["anchor_id"].map(cached.set_index("anchor_id")["target_class"])
            if y.isna().any() or tuple(np.unique(y).tolist()) != (0, 1, 2):
                raise AssertionError("invalid progressive training target")
            target = y.to_numpy(np.int8)
            weights = base.nested_symbol_session_weights(train)
            prior_validation_rows = int(
                train["month_key"].isin(VALIDATION_MONTHS).sum()
            )
            for algorithm in ALGORITHMS:
                features = FEATURES_BY_ALGORITHM[algorithm]
                train_raw = train.loc[:, features].to_numpy(float)
                score_raw = score.loc[:, features].to_numpy(float)
                medians = base.training_medians(train_raw)
                train_values = base.apply_medians(train_raw, medians)
                score_values = base.apply_medians(score_raw, medians)
                estimator = HistGradientBoostingClassifier(**HGB_PARAMETERS)
                estimator.fit(train_values, target, sample_weight=weights)
                probabilities = base._class_probability_matrix(estimator, score_values)
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
                predictions.append(output.loc[:, PREDICTION_COLUMNS])
                for feature_order, (feature, median) in enumerate(
                    zip(features, medians, strict=True)
                ):
                    preprocessing.append(
                        {
                            "fold_month": month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "feature_order": feature_order,
                            "feature": feature,
                            "training_median": float(median),
                        }
                    )
                folds.append(
                    {
                        "fold_month": month,
                        "algorithm": algorithm,
                        "horizon": horizon,
                        "train_rows": len(train),
                        "score_rows": len(score),
                        "same_score_month_training_label_rows": 0,
                        "prior_completed_validation_month_training_label_rows": prior_validation_rows,
                        "maximum_training_path_end_timestamp": train["path_end_timestamp"].max(),
                        "minimum_scoring_timestamp": score["decision_timestamp"].min(),
                        "class_0_rows": int((target == 0).sum()),
                        "class_1_rows": int((target == 1).sum()),
                        "class_2_rows": int((target == 2).sum()),
                        "fitted_iterations": int(estimator.n_iter_),
                    }
                )
                bundles[(month, horizon, algorithm)] = {
                    "estimator": estimator,
                    "features": features,
                    "medians": medians,
                }
            del train, score
            gc.collect()
        del anchors, cached
        gc.collect()
    ledger = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(
            ["algorithm", "horizon", "symbol_norm", "session_date", "decision_timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    validate_prediction_ledger(ledger)
    expected = (sum(base.EXPECTED_JUNE_ROWS.values()) + sum(base.EXPECTED_VALIDATION_ROWS.values())) * len(
        ALGORITHMS
    )
    if len(ledger) != expected:
        raise AssertionError(f"prediction support drift: {len(ledger)} != {expected}")
    return ledger, pd.DataFrame(preprocessing), pd.DataFrame(folds), bundles


def build_thresholds(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score_month in VALIDATION_MONTHS:
        source_month = f"2024-{int(score_month[-2:]) - 1:02d}"
        for algorithm in ALGORITHMS:
            for horizon in HORIZONS:
                source = ledger.loc[
                    ledger["fold_month"].eq(source_month)
                    & ledger["algorithm"].eq(algorithm)
                    & ledger["horizon"].eq(horizon)
                ]
                weights = base.metric_weights(source)
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
                            "fire_threshold": base.weighted_quantile(
                                values, weights, FIRE_QUANTILE
                            ),
                            "rearm_threshold": base.weighted_quantile(
                                values, weights, REARM_QUANTILE
                            ),
                            "source_rows": len(source),
                            "fire_quantile": FIRE_QUANTILE,
                            "rearm_quantile": REARM_QUANTILE,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "horizon", "score_month", "side"], kind="stable"
    ).reset_index(drop=True)


def extract_candidate_onsets(
    ledger: pd.DataFrame, thresholds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = ledger.loc[
        ledger["fold_month"].isin(VALIDATION_MONTHS) & ledger["algorithm"].eq(CANDIDATE)
    ].copy()
    threshold_index = thresholds.set_index(["score_month", "algorithm", "horizon", "side"])
    onsets: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    keys = ["algorithm", "horizon", "symbol_norm", "session_date", "segment_index"]
    for group_key, group in candidate.groupby(keys, sort=False):
        algorithm, horizon, symbol, session_date, segment_index = group_key
        long_armed = True
        short_armed = True
        for row in group.sort_values("decision_timestamp", kind="stable").itertuples(index=False):
            long_threshold = threshold_index.loc[(row.fold_month, algorithm, horizon, "long")]
            short_threshold = threshold_index.loc[(row.fold_month, algorithm, horizon, "short")]
            long_before, short_before = long_armed, short_armed
            if not long_armed and row.p_long_first < float(long_threshold["rearm_threshold"]):
                long_armed = True
            if not short_armed and row.p_short_first < float(short_threshold["rearm_threshold"]):
                short_armed = True
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
            side: str | None = None
            if long_fire and not short_fire:
                side = "long"
                long_armed = False
            elif short_fire and not long_fire:
                side = "short"
                short_armed = False
            onset_id = f"{algorithm}|{row.anchor_id}|{side}" if side else None
            if side:
                onsets.append(
                    {
                        "onset_id": onset_id,
                        "anchor_id": row.anchor_id,
                        "candidate_algorithm": algorithm,
                        "horizon": int(horizon),
                        "side": side,
                        "fold_month": row.fold_month,
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
                            row.p_long_first if side == "long" else row.p_short_first
                        ),
                        "opposite_probability": float(
                            row.p_short_first if side == "long" else row.p_long_first
                        ),
                    }
                )
            states.append(
                {
                    "anchor_id": row.anchor_id,
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "fold_month": row.fold_month,
                    "symbol_norm": symbol,
                    "session_date": session_date,
                    "decision_timestamp": row.decision_timestamp,
                    "segment_index": int(segment_index),
                    "long_armed_before": long_before,
                    "short_armed_before": short_before,
                    "long_fire": long_fire,
                    "short_fire": short_fire,
                    "emitted_side": side,
                    "onset_id": onset_id,
                    "long_armed_after": long_armed,
                    "short_armed_after": short_armed,
                }
            )
    onset_frame = pd.DataFrame(onsets).sort_values("onset_id", kind="stable").reset_index(drop=True)
    state_frame = pd.DataFrame(states).sort_values(
        ["horizon", "symbol_norm", "session_date", "decision_timestamp"], kind="stable"
    ).reset_index(drop=True)
    return onset_frame, state_frame


def match_price_controls(onsets: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    price = ledger.loc[
        ledger["algorithm"].eq(BASELINE) & ledger["fold_month"].isin(VALIDATION_MONTHS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (algorithm, horizon, side), candidates in onsets.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        probability_column = "p_long_first" if side == "long" else "p_short_first"
        pools = {
            key: group.sort_values(
                [probability_column, "anchor_id"], ascending=[False, True], kind="stable"
            ).reset_index(drop=True)
            for key, group in price.loc[price["horizon"].eq(horizon)].groupby(
                ["symbol_norm", "fold_month"], sort=False
            )
        }
        used: set[str] = set()
        for candidate in candidates.sort_values("onset_id", kind="stable").itertuples(index=False):
            pool = pools[(candidate.symbol_norm, candidate.fold_month)]
            base_mask = ~pool["anchor_id"].eq(candidate.anchor_id) & ~pool["anchor_id"].isin(used)
            tier_masks = [
                base_mask
                & ~pool["session_date"].eq(candidate.session_date)
                & pool["clock_bin_15"].eq(candidate.clock_bin_15)
                & pool["availability_bucket"].eq(candidate.availability_bucket),
                base_mask
                & pool["clock_bin_30"].eq(candidate.clock_bin_30)
                & pool["availability_bucket"].eq(candidate.availability_bucket),
                base_mask & pool["clock_bin_30"].eq(candidate.clock_bin_30),
                base_mask,
            ]
            chosen: pd.Series | None = None
            tier = -1
            for tier_index, mask in enumerate(tier_masks):
                eligible = pool.loc[mask]
                if not eligible.empty:
                    chosen = eligible.iloc[0]
                    tier = tier_index
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
                "control_price_probability": math.nan,
                "match_tier": tier,
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
                        "control_price_probability": float(chosen[probability_column]),
                        "matched": True,
                    }
                )
            rows.append(record)
    controls = pd.DataFrame(rows).sort_values("onset_id", kind="stable").reset_index(drop=True)
    if len(controls) != len(onsets) or controls["onset_id"].duplicated().any():
        raise AssertionError("price-control match cardinality failure")
    return controls


def build_validation_feature_lookup(feature_surface: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        anchors = build_anchor_surface(feature_surface, horizon)
        parts.append(
            anchors.loc[
                anchors["month_key"].isin(VALIDATION_MONTHS),
                ["anchor_id", "horizon", "month_key", *PRICE_FEATURES, *VOLUME_FEATURES],
            ]
        )
    return pd.concat(parts, ignore_index=True)


def build_volume_sensitivities(
    onsets: pd.DataFrame,
    feature_lookup: pd.DataFrame,
    bundles: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ``availability_12`` is carried by the onset ledger and is also one of
    # the frozen price features.  Keep the authoritative feature-lookup copy
    # so pandas cannot suffix the executable feature name during the merge.
    duplicate_feature_columns = [
        feature
        for feature in (*PRICE_FEATURES, *VOLUME_FEATURES)
        if feature in onsets.columns
    ]
    joined = onsets.drop(columns=duplicate_feature_columns).merge(
        feature_lookup,
        left_on=["anchor_id", "horizon", "fold_month"],
        right_on=["anchor_id", "horizon", "month_key"],
        how="left",
        validate="one_to_one",
    )
    if joined.loc[:, VOLUME_FEATURES].isna().all(axis=1).any():
        raise AssertionError("onset missing all historical-volume features")
    rows: list[pd.DataFrame] = []
    for (month, horizon), group in joined.groupby(["fold_month", "horizon"], sort=False):
        bundle = bundles[(month, int(horizon), CANDIDATE)]
        features = tuple(bundle["features"])
        medians = np.asarray(bundle["medians"], float)
        values = base.apply_medians(group.loc[:, features].to_numpy(float), medians)
        counterfactual = values.copy()
        volume_indices = [features.index(feature) for feature in VOLUME_FEATURES]
        counterfactual[:, volume_indices] = medians[volume_indices]
        estimator = bundle["estimator"]
        actual = base._class_probability_matrix(estimator, values)
        counter = base._class_probability_matrix(estimator, counterfactual)
        long_side = group["side"].eq("long").to_numpy(bool)
        chosen = np.where(long_side, 1, 2)
        opposite = np.where(long_side, 2, 1)
        index = np.arange(len(group))
        actual_margin = actual[index, chosen] - actual[index, opposite]
        counter_margin = counter[index, chosen] - counter[index, opposite]
        expected_chosen = group["chosen_probability"].to_numpy(float)
        if not np.allclose(actual[index, chosen], expected_chosen, atol=1e-12, rtol=0.0):
            raise AssertionError("volume-sensitivity actual probability mismatch")
        output = group[
            [
                "onset_id",
                "anchor_id",
                "fold_month",
                "candidate_algorithm",
                "horizon",
                "side",
                "symbol_norm",
                "session_date",
                "decision_timestamp",
                *VOLUME_FEATURES,
            ]
        ].copy()
        output["actual_directional_probability_margin"] = actual_margin
        output["price_median_volume_counterfactual_margin"] = counter_margin
        output["historical_volume_group_margin_sensitivity"] = actual_margin - counter_margin
        output["sensitivity_direction"] = np.where(
            output["historical_volume_group_margin_sensitivity"] > 0.0,
            "supports_chosen",
            np.where(
                output["historical_volume_group_margin_sensitivity"] < 0.0,
                "opposes_chosen",
                "neutral",
            ),
        )
        rows.append(output)
    sensitivity = pd.concat(rows, ignore_index=True).sort_values("onset_id", kind="stable")
    summary = (
        sensitivity.groupby(["horizon", "side", "sensitivity_direction"], sort=False)
        .agg(
            onsets=("onset_id", "size"),
            months=("fold_month", "nunique"),
            stocks=("symbol_norm", "nunique"),
            mean_margin_sensitivity=("historical_volume_group_margin_sensitivity", "mean"),
            median_margin_sensitivity=("historical_volume_group_margin_sensitivity", "median"),
        )
        .reset_index()
    )
    summary["recurring"] = (summary["months"] >= 5) & (summary["stocks"] >= 15)
    return sensitivity.reset_index(drop=True), summary


def write_pre_evaluation_freeze(
    artifacts: Mapping[str, tuple[Any, str]], source_binding: Mapping[str, Any]
) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for name, (payload, kind) in artifacts.items():
        path = OUT / name
        if kind == "parquet":
            payload.to_parquet(path, index=False)
        elif kind == "csv":
            payload.to_csv(path, index=False)
        elif kind == "json":
            write_json(path, payload)
        else:
            raise AssertionError(f"unknown artifact kind {kind}")
        hashes[name] = sha256(path)
    freeze = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume",
        "volume_is_order_flow": False,
        "stage": "predictions_onsets_price_controls_and_volume_sensitivities_frozen_before_final_evaluation_join",
        "prior_price_only_outcomes_known_before_volume_contract": True,
        "same_score_month_outcomes_read_before_that_month_probability": False,
        "source_binding": dict(source_binding),
        "artifact_sha256": hashes,
    }
    write_json(OUT / "pre_evaluation_freeze.json", freeze)
    return {
        **freeze,
        "freeze_manifest_sha256": sha256(OUT / "pre_evaluation_freeze.json"),
    }


def probability_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["horizon", "slice_type", "slice_value"]
    baseline = metrics.loc[metrics["algorithm"].eq(BASELINE)].set_index(keys)
    candidate = metrics.loc[metrics["algorithm"].eq(CANDIDATE)].set_index(keys).reindex(
        baseline.index
    )
    rows: list[dict[str, Any]] = []
    for key, row in candidate.iterrows():
        reference = baseline.loc[key]
        rows.append(
            {
                "horizon": int(key[0]),
                "slice_type": key[1],
                "slice_value": key[2],
                "rows": int(row["rows"]),
                "log_loss_improvement_vs_price": float(
                    reference["multiclass_log_loss"] - row["multiclass_log_loss"]
                ),
                "brier_improvement_vs_price": float(
                    reference["multiclass_brier"] - row["multiclass_brier"]
                ),
                "accuracy_lift_vs_price": float(
                    row["top_class_accuracy"] - reference["top_class_accuracy"]
                ),
                "auc_lift_vs_price": float(row["macro_ovr_auc"] - reference["macro_ovr_auc"]),
            }
        )
    return pd.DataFrame(rows)


def paired_moving_block_interval(
    frame: pd.DataFrame, value_column: str, *, random_state: int
) -> dict[str, float | int]:
    session = (
        frame.groupby(["session_date", "symbol_norm"], sort=True)[value_column]
        .mean()
        .unstack("symbol_norm")
        .sort_index()
        .reindex(columns=sorted(frame["symbol_norm"].unique()))
    )
    matrix = session.to_numpy(float)
    n_dates = len(matrix)
    if n_dates < BOOTSTRAP_BLOCK:
        raise AssertionError("insufficient bootstrap sessions")
    observed = float(np.nanmean(np.nanmean(matrix, axis=0)))
    finite = np.isfinite(matrix)
    filled = np.where(finite, matrix, 0.0)
    rng = np.random.default_rng(random_state)
    blocks_needed = math.ceil(n_dates / BOOTSTRAP_BLOCK)
    start_max = n_dates - BOOTSTRAP_BLOCK
    samples = np.empty(BOOTSTRAP_DRAWS, float)
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, start_max + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + BOOTSTRAP_BLOCK) for start in starts]
        )[:n_dates]
        counts = np.bincount(indices, minlength=n_dates).astype(float)
        numerator = (filled * counts[:, None]).sum(axis=0)
        denominator = (finite * counts[:, None]).sum(axis=0)
        symbol_values = np.divide(
            numerator,
            denominator,
            out=np.full(matrix.shape[1], np.nan),
            where=denominator > 0.0,
        )
        samples[draw] = np.nanmean(symbol_values)
    return {
        "observed": observed,
        "lower": float(np.quantile(samples, LOWER_QUANTILE)),
        "upper": float(np.quantile(samples, UPPER_QUANTILE)),
        "draws": BOOTSTRAP_DRAWS,
    }


def build_event_bootstraps(pairs: pd.DataFrame) -> pd.DataFrame:
    work = pairs.copy()
    work["precision_lift"] = (
        work["conservative_correct_candidate"] - work["conservative_correct_control"]
    )
    work["rapid_success_lift"] = (
        work["rapid_correct_candidate"] - work["rapid_correct_control"]
    )
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
            rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "metric": metric,
                    **paired_moving_block_interval(
                        group,
                        metric,
                        random_state=SEED
                        + int(horizon) * 100
                        + (0 if side == "long" else 10)
                        + metric_index,
                    ),
                    "lower_quantile": LOWER_QUANTILE,
                    "upper_quantile": UPPER_QUANTILE,
                    "block_sessions": BOOTSTRAP_BLOCK,
                }
            )
    return pd.DataFrame(rows)


def final_decision(
    comparisons: pd.DataFrame,
    onsets: pd.DataFrame,
    controls: pd.DataFrame,
    event_lifts: pd.DataFrame,
    bootstraps: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> dict[str, Any]:
    probability: dict[str, Any] = {}
    for horizon in HORIZONS:
        pooled = comparisons.loc[
            comparisons["horizon"].eq(horizon)
            & comparisons["slice_type"].eq("pooled")
        ].iloc[0]
        months = comparisons.loc[
            comparisons["horizon"].eq(horizon)
            & comparisons["slice_type"].eq("month")
        ]
        loso = comparisons.loc[
            comparisons["horizon"].eq(horizon)
            & comparisons["slice_type"].eq("leave_one_stock_out")
        ]
        month_both = int(
            ((months["log_loss_improvement_vs_price"] > 0.0) & (months["brier_improvement_vs_price"] > 0.0)).sum()
        )
        loso_both = int(
            ((loso["log_loss_improvement_vs_price"] > 0.0) & (loso["brier_improvement_vs_price"] > 0.0)).sum()
        )
        gates = {
            "pooled_log_loss_better": bool(pooled["log_loss_improvement_vs_price"] > 0.0),
            "pooled_brier_better": bool(pooled["brier_improvement_vs_price"] > 0.0),
            "months_both_better_at_least_4": month_both >= 4,
            "stock_deletions_both_better_at_least_18": loso_both >= 18,
        }
        probability[f"h{horizon}"] = {
            "log_loss_improvement_vs_price": float(pooled["log_loss_improvement_vs_price"]),
            "brier_improvement_vs_price": float(pooled["brier_improvement_vs_price"]),
            "months_both_better": month_both,
            "stock_deletions_both_better": loso_both,
            "gates": gates,
            "passed": all(gates.values()),
        }
    event: dict[str, Any] = {}
    for side in ("long", "short"):
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            subset = onsets.loc[onsets["side"].eq(side) & onsets["horizon"].eq(horizon)]
            matches = controls.loc[
                controls["side"].eq(side) & controls["horizon"].eq(horizon)
            ]
            pooled = event_lifts.loc[
                event_lifts["side"].eq(side)
                & event_lifts["horizon"].eq(horizon)
                & event_lifts["slice_type"].eq("pooled")
            ].iloc[0]
            months = event_lifts.loc[
                event_lifts["side"].eq(side)
                & event_lifts["horizon"].eq(horizon)
                & event_lifts["slice_type"].eq("month")
            ]
            loso = event_lifts.loc[
                event_lifts["side"].eq(side)
                & event_lifts["horizon"].eq(horizon)
                & event_lifts["slice_type"].eq("leave_one_stock_out")
            ]
            intervals = bootstraps.loc[
                bootstraps["side"].eq(side) & bootstraps["horizon"].eq(horizon)
            ].set_index("metric")
            month_min = int(subset.groupby("fold_month").size().min())
            gates = {
                "onsets_at_least_500": len(subset) >= 500,
                "each_month_at_least_50": month_min >= 50,
                "stocks_at_least_15": subset["symbol_norm"].nunique() >= 15,
                "side_onsets_at_least_100": len(subset) >= 100,
                "side_stocks_at_least_10": subset["symbol_norm"].nunique() >= 10,
                "match_rate_at_least_0_95": float(matches["matched"].mean()) >= 0.95,
                "precision_lift_at_least_0_02": float(pooled["precision_lift"]) >= 0.02,
                "relative_precision_lift_at_least_0_05": float(pooled["relative_precision_lift"]) >= 0.05,
                "rapid_lift_at_least_0_01": float(pooled["rapid_success_lift"]) >= 0.01,
                "dominance_positive": float(pooled["directional_dominance_lift"]) > 0.0,
                "adverse_not_worse": float(pooled["pre_confirmation_adverse_improvement"]) >= 0.0,
                "all_bootstrap_lower_bounds_positive": bool((intervals["lower"] > 0.0).all()),
                "positive_precision_months_at_least_4": int((months["precision_lift"] > 0.0).sum()) >= 4,
                "positive_precision_stock_deletions_at_least_18": int((loso["precision_lift"] > 0.0).sum()) >= 18,
            }
            horizons[f"h{horizon}"] = {
                "onsets": len(subset),
                "candidate_precision": float(pooled["candidate_precision"]),
                "matched_price_precision": float(pooled["control_precision"]),
                "precision_lift": float(pooled["precision_lift"]),
                "rapid_success_lift": float(pooled["rapid_success_lift"]),
                "directional_dominance_lift": float(pooled["directional_dominance_lift"]),
                "pre_confirmation_adverse_improvement": float(
                    pooled["pre_confirmation_adverse_improvement"]
                ),
                "gates": gates,
                "passed": all(gates.values()),
            }
        event[side] = {"horizons": horizons, "retained": all(x["passed"] for x in horizons.values())}
    probability_retained = all(item["passed"] for item in probability.values())
    any_side_retained = any(item["retained"] for item in event.values())
    return {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume",
        "volume_is_order_flow": False,
        "prior_price_only_outcomes_known_before_volume_contract": True,
        "probability_hypothesis": {
            "horizons": probability,
            "retained": probability_retained,
        },
        "entry_onset_hypotheses": event,
        "any_incremental_historical_volume_entry_sign_retained": bool(
            probability_retained and any_side_retained
        ),
        "recurring_volume_sensitivity_rows": safe(
            sensitivity_summary.loc[sensitivity_summary["recurring"]].to_dict("records")
        ),
        "interpretation": "2024 post-outcome development only; never order flow or a trading claim",
    }


def validate_only() -> dict[str, Any]:
    contract, _ = load_contract_and_verify(False)
    tape, diagnostics = load_tape_with_historical_volume(return_diagnostics=True)
    surface = build_historical_volume_feature_surface(tape)
    support: dict[str, Any] = {}
    for horizon in HORIZONS:
        anchors = build_anchor_surface(surface, horizon)
        support[f"h{horizon}"] = {
            "annual": len(anchors),
            "june": int(anchors["month_key"].eq("2024-06").sum()),
            "validation": int(anchors["month_key"].isin(VALIDATION_MONTHS).sum()),
        }
    return {
        "contract_id": contract["contract_id"],
        "mode": "target_blind_historical_volume_validation_only",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume",
        "volume_is_order_flow": False,
        "models_fitted": False,
        "path_labels_constructed": False,
        "output_directory_written": False,
        "environment_versions": environment_versions(),
        "source_hashes_for_future_freeze": current_source_hashes(),
        "diagnostics": diagnostics,
        "support": support,
        "finite_feature_counts": {
            feature: int(np.isfinite(surface[feature].to_numpy(float)).sum())
            for feature in VOLUME_FEATURES
        },
    }


def run_scoring() -> dict[str, Any]:
    contract, manifest = load_contract_and_verify(True)
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact root: {OUT}")
    OUT.mkdir(parents=True)
    source_binding = {
        "contract_id": CONTRACT_ID,
        "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
        "source_sha256": manifest["sha256"],
        "environment_versions": environment_versions(),
    }
    tape, diagnostics = load_tape_with_historical_volume(return_diagnostics=True)
    feature_surface = build_historical_volume_feature_surface(tape)
    predictions, preprocessing, folds, bundles = fit_monthly_probabilities(tape, feature_surface)
    thresholds = build_thresholds(predictions)
    onsets, states = extract_candidate_onsets(predictions, thresholds)
    controls = match_price_controls(onsets, predictions)
    feature_lookup = build_validation_feature_lookup(feature_surface)
    sensitivities, sensitivity_summary = build_volume_sensitivities(
        onsets, feature_lookup, bundles
    )
    freeze = write_pre_evaluation_freeze(
        {
            "probabilities_pre_evaluation.parquet": (predictions, "parquet"),
            "thresholds_pre_evaluation.csv": (thresholds, "csv"),
            "candidate_onsets_pre_evaluation.parquet": (onsets, "parquet"),
            "onset_states_pre_evaluation.parquet": (states, "parquet"),
            "matched_price_controls_pre_evaluation.parquet": (controls, "parquet"),
            "historical_volume_sensitivities_pre_evaluation.parquet": (
                sensitivities,
                "parquet",
            ),
            "historical_volume_sensitivity_summary_pre_evaluation.csv": (
                sensitivity_summary,
                "csv",
            ),
            "fold_preprocessing.csv": (preprocessing, "csv"),
            "fold_metadata.csv": (folds, "csv"),
            "source_hashes.json": (source_binding, "json"),
            "data_diagnostics.json": (diagnostics, "json"),
        },
        source_binding,
    )
    paths = base.build_validation_paths(tape, feature_surface)
    paths.to_parquet(OUT / "validation_paths.parquet", index=False)
    probability_metrics, calibration = base.evaluate_probability_metrics(predictions, paths)
    comparisons = probability_comparisons(probability_metrics)
    candidates, scored_controls, pairs = base.build_scored_events(onsets, controls, paths)
    event_metrics, event_lifts = base.evaluate_event_metrics(candidates, scored_controls, pairs)
    event_metrics["sample_kind"] = event_metrics["sample_kind"].replace(
        {"matched_clock_control": "matched_price_control"}
    )
    bootstraps = build_event_bootstraps(pairs)
    match_summary = base.control_match_summary(onsets, controls)
    decision = final_decision(
        comparisons,
        onsets,
        controls,
        event_lifts,
        bootstraps,
        sensitivity_summary,
    )
    tables = {
        "probability_metrics.csv": probability_metrics,
        "probability_calibration.csv": calibration,
        "probability_comparisons.csv": comparisons,
        "scored_candidate_onsets.parquet": candidates,
        "scored_matched_price_controls.parquet": scored_controls,
        "paired_event_evidence.parquet": pairs,
        "event_metrics.csv": event_metrics,
        "event_lifts.csv": event_lifts,
        "event_bootstrap_intervals.csv": bootstraps,
        "control_match_summary.csv": match_summary,
    }
    for name, frame in tables.items():
        path = OUT / name
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
    write_json(OUT / "decision.json", decision)
    summary = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume",
        "volume_is_order_flow": False,
        "prior_price_only_outcomes_known_before_volume_contract": True,
        "probability_rows": len(predictions),
        "candidate_onsets": len(onsets),
        "matched_price_controls": int(controls["matched"].sum()),
        "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
        "pre_evaluation_freeze_manifest_sha256": freeze["freeze_manifest_sha256"],
        "decision": decision,
    }
    write_json(OUT / "summary.json", summary)
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    write_json(
        OUT / "artifact_manifest.json",
        {
            "contract_id": CONTRACT_ID,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "provider_volume_label": "historical_volume",
            "stage": "pre_independent_audit_complete_artifact_manifest",
            "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
            "pre_evaluation_freeze_manifest_sha256": freeze["freeze_manifest_sha256"],
            "files_excluding_this_manifest": {path.name: sha256(path) for path in files},
        },
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_only() if args.validate_only else run_scoring()
    print(json.dumps(safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
