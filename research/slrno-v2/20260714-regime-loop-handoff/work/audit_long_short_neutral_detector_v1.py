"""Independent audit for the research-only long/short/neutral experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-long-short-neutral-detector-v1.json"
CONTRACT_ID = "20260714-long-short-neutral-detector-v1"
CLASSES = ("long", "neutral", "short")
MODELS = ("M0_clock_prior", "M1_price_context", "M2_price_plus_activity")
SEED = 20260714
FORBIDDEN = (
    "label",
    "target_class",
    "future",
    "upper_hit",
    "lower_hit",
    "payoff",
    "net_bps",
    "mfe",
    "mae",
)
PRICE = (
    "barrier_bps",
    "current_range_scale",
    "current_body_scale",
    "current_close_location",
    "current_upper_wick_fraction",
    "current_lower_wick_fraction",
    "return_1_scale",
    "return_3_scale",
    "return_6_scale",
    "return_12_scale",
    "mean_abs_return_6_scale",
    "compression_3_to_12",
    "session_return_scale",
    "session_mean_distance_scale",
    "opening_range_position",
    "opening_range_width_scale",
)
ACTIVITY = ("current_activity_ratio_12", "activity_trend_3_to_12")
FEATURES = {
    "M0_clock_prior": ((), ("decision_clock",)),
    "M1_price_context": (PRICE, ("decision_clock",)),
    "M2_price_plus_activity": ((*PRICE, *ACTIVITY), ("decision_clock",)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def load_sessions(contract: dict[str, Any]) -> dict[tuple[int, str, str], pd.DataFrame]:
    root = Path(contract["data"]["provider_root"])
    sessions: dict[tuple[int, str, str], pd.DataFrame] = {}
    for period in (2024, 2025, 2026):
        symbols = (
            contract["data"]["symbols_2026"]
            if period == 2026
            else contract["data"]["symbols_2024_2025"]
        )
        for symbol in symbols:
            frame = pd.read_parquet(
                provider_path(root, symbol),
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
            frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
            local = frame["timestamp"].dt.tz_convert("America/New_York")
            if period == 2026:
                in_period = local.dt.year.eq(2026) & local.dt.date.lt(
                    pd.Timestamp("2026-06-30").date()
                )
            else:
                in_period = local.dt.year.eq(period)
            selected = frame.loc[in_period].copy()
            selected_local = selected["timestamp"].dt.tz_convert("America/New_York")
            minute = selected_local.dt.hour * 60 + selected_local.dt.minute
            regular = minute.ge(570) & minute.lt(960)
            grid = (
                (minute - 570).mod(5).eq(0)
                & selected_local.dt.second.eq(0)
                & selected_local.dt.microsecond.eq(0)
            )
            prices = selected[["open", "high", "low", "close"]].apply(
                pd.to_numeric, errors="coerce"
            )
            volume = pd.to_numeric(selected["volume"], errors="coerce")
            valid = (
                prices.gt(0).all(axis=1)
                & prices["high"].ge(prices[["open", "close"]].max(axis=1))
                & prices["low"].le(prices[["open", "close"]].min(axis=1))
                & volume.ge(0)
                & np.isfinite(volume)
            )
            accepted = regular & grid & valid
            clean = selected.loc[accepted].copy()
            clean[["open", "high", "low", "close"]] = prices.loc[accepted]
            clean["volume"] = volume.loc[accepted]
            clean_local = clean["timestamp"].dt.tz_convert("America/New_York")
            clean_minute = clean_local.dt.hour * 60 + clean_local.dt.minute
            clean["session_date"] = clean_local.dt.strftime("%Y-%m-%d")
            clean["bar_ordinal"] = ((clean_minute - 570) // 5).astype(np.int16)
            clean = clean.sort_values("timestamp", kind="stable").reset_index(drop=True)
            previous = clean["close"].shift(1)
            denominator = previous.where(previous.gt(0), clean["open"])
            tr = np.maximum.reduce(
                [
                    (clean["high"] - clean["low"]).to_numpy(float),
                    (clean["high"] - denominator).abs().to_numpy(float),
                    (clean["low"] - denominator).abs().to_numpy(float),
                ]
            )
            clean["true_range_bps"] = 10000.0 * tr / denominator
            clean["range_bps"] = 10000.0 * (clean["high"] - clean["low"]) / clean["open"]
            for date, session in clean.groupby("session_date", sort=False):
                sessions[(period, symbol, str(date))] = session.reset_index(drop=True)
    return sessions


def divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def geometric(values: pd.Series) -> float:
    array = values.to_numpy(float)
    if not np.isfinite(array).all() or (array <= 0).any():
        return float("nan")
    return float(np.exp(np.log(array).mean()))


def reconstruct_features(event: Any, session: pd.DataFrame) -> dict[str, float]:
    decision = int(event.decision_ordinal)
    indexed = session.set_index("bar_ordinal", drop=False)
    prior = indexed.loc[list(range(decision - 12, decision))]
    current = indexed.loc[decision]
    scale = float(np.median(prior["true_range_bps"].to_numpy(float)))
    current_open = float(current["open"])
    current_close = float(current["close"])
    span = float(current["high"] - current["low"])
    returns = {}
    for lag in (1, 3, 6, 12):
        lag_close = float(indexed.loc[decision - lag, "close"])
        returns[f"return_{lag}_scale"] = divide(
            10000.0 * (current_close / lag_close - 1.0), scale
        )
    close_window = indexed.loc[list(range(decision - 6, decision + 1)), "close"].to_numpy(float)
    abs_return = np.abs(10000.0 * (close_window[1:] / close_window[:-1] - 1.0)).mean()
    prior_range_median = float(np.median(prior["range_bps"].to_numpy(float)))
    recent_range_median = float(
        np.median(indexed.loc[list(range(decision - 2, decision + 1)), "range_bps"].to_numpy(float))
    )
    so_far = indexed.loc[list(range(0, decision + 1))]
    typical = ((so_far["high"] + so_far["low"] + so_far["close"]) / 3.0).mean()
    opening = indexed.loc[list(range(0, 6))]
    opening_high = float(opening["high"].max())
    opening_low = float(opening["low"].min())
    opening_width = opening_high - opening_low
    prior_volume = geometric(prior["volume"])
    recent_volume = geometric(
        indexed.loc[list(range(decision - 2, decision + 1)), "volume"]
    )
    return {
        "prior_scale_bps": scale,
        "barrier_bps": float(np.clip(4.0 * scale, 40.0, 250.0)),
        "current_range_scale": float(current["range_bps"]) / scale,
        "current_body_scale": 10000.0 * (current_close / current_open - 1.0) / scale,
        "current_close_location": (current_close - float(current["low"])) / span,
        "current_upper_wick_fraction": (
            float(current["high"]) - max(current_open, current_close)
        ) / span,
        "current_lower_wick_fraction": (
            min(current_open, current_close) - float(current["low"])
        ) / span,
        **returns,
        "mean_abs_return_6_scale": abs_return / scale,
        "compression_3_to_12": divide(recent_range_median, prior_range_median),
        "session_return_scale": divide(
            10000.0 * (current_close / float(indexed.loc[0, "open"]) - 1.0), scale
        ),
        "session_mean_distance_scale": divide(
            10000.0 * (current_close / float(typical) - 1.0), scale
        ),
        "opening_range_position": divide(current_close - opening_low, opening_width),
        "opening_range_width_scale": divide(
            10000.0 * opening_width / float(indexed.loc[0, "open"]), scale
        ),
        "current_activity_ratio_12": divide(float(current["volume"]), prior_volume),
        "activity_trend_3_to_12": divide(recent_volume, prior_volume),
    }


def replay_outcome(event: Any, session: pd.DataFrame) -> dict[str, Any]:
    indexed = session.set_index("bar_ordinal", drop=False)
    path_ordinals = list(range(int(event.decision_ordinal) + 1, int(event.decision_ordinal) + 25))
    if any(ordinal not in indexed.index for ordinal in path_ordinals):
        return {"score_status": "missing_exact_24_bar_path"}
    path = indexed.loc[path_ordinals]
    entry = float(path.iloc[0]["open"])
    width = float(event.barrier_bps)
    upper = entry * (1.0 + width / 10000.0)
    lower = entry * (1.0 - width / 10000.0)
    actual = "neutral"
    reason = "no_touch"
    first = None
    for step, row in enumerate(path.itertuples(index=False), start=1):
        if float(row.open) >= upper:
            actual, reason, first = "long", "", step
            break
        if float(row.open) <= lower:
            actual, reason, first = "short", "", step
            break
        high = float(row.high) >= upper
        low = float(row.low) <= lower
        if high and low:
            actual, reason, first = "neutral", "same_bar_dual_touch", step
            break
        if high:
            actual, reason, first = "long", "", step
            break
        if low:
            actual, reason, first = "short", "", step
            break
    end_return = 10000.0 * (float(path.iloc[-1]["close"]) / entry - 1.0)
    if actual == "long":
        long_gross, short_gross = width, -width
    elif actual == "short":
        long_gross, short_gross = -width, width
    elif reason == "same_bar_dual_touch":
        long_gross = short_gross = -width
    else:
        long_gross, short_gross = end_return, -end_return
    return {
        "score_status": "scored",
        "entry_open": entry,
        "upper_barrier": upper,
        "lower_barrier": lower,
        "actual_class": actual,
        "neutral_reason": reason,
        "first_touch_step": first,
        "end_return_bps": end_return,
        "long_gross_bps": long_gross,
        "short_gross_bps": short_gross,
        "long_net_bps_5": long_gross - 10.0,
        "short_net_bps_5": short_gross - 10.0,
    }


def pipeline(numeric: tuple[str, ...], categorical: tuple[str, ...]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric),
            )
        )
    transformers.append(
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            list(categorical),
        )
    )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def compare_numeric(expected: float, observed: float) -> float:
    if pd.isna(expected) and pd.isna(observed):
        return 0.0
    return abs(float(expected) - float(observed))


def audit(artifact_root: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((artifact_root / "pre_score_manifest.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(artifact_root / "pre_outcome_events.parquet")
    outcomes = pd.read_parquet(artifact_root / "outcomes.parquet")
    predictions = pd.read_parquet(artifact_root / "predictions.parquet")
    checks: dict[str, bool] = {}
    errors: dict[str, Any] = {}

    source_errors = 0
    for value in manifest["source_hashes"].values():
        path = Path(value["path"])
        source_errors += int(
            not path.exists()
            or path.stat().st_size != value["bytes"]
            or sha256(path) != value["sha256"]
        )
    checks["all_source_hashes_match"] = source_errors == 0
    errors["source_hashes"] = source_errors

    frozen_errors = 0
    for name, expected in manifest["frozen_files"].items():
        path = artifact_root / name
        frozen_errors += int(
            not path.exists()
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        )
    checks["frozen_outcome_free_ledgers_match"] = frozen_errors == 0
    errors["frozen_ledgers"] = frozen_errors

    forbidden = [
        column for column in events.columns if any(token in column.lower() for token in FORBIDDEN)
    ]
    checks["pre_outcome_ledger_has_no_outcomes"] = not forbidden
    errors["forbidden_pre_outcome_columns"] = forbidden
    checks["event_identity_unique"] = events["event_id"].is_unique
    checks["outcome_identity_complete"] = set(events["event_id"]) == set(outcomes["event_id"])

    sessions = load_sessions(contract)
    maximum_feature_error = 0.0
    feature_errors = 0
    for event in events.itertuples(index=False):
        session = sessions[(int(event.period), str(event.symbol_norm), str(event.session_date))]
        reconstructed = reconstruct_features(event, session)
        for name, expected in reconstructed.items():
            error = compare_numeric(expected, getattr(event, name))
            maximum_feature_error = max(maximum_feature_error, error)
            feature_errors += int(error > 1e-10)
    checks["all_causal_features_independently_reconstructed"] = feature_errors == 0
    errors["causal_feature_errors"] = feature_errors

    outcome_lookup = outcomes.set_index("event_id")
    maximum_path_error = 0.0
    path_errors = 0
    categorical_errors = 0
    numeric_outcome_fields = (
        "entry_open",
        "upper_barrier",
        "lower_barrier",
        "end_return_bps",
        "long_gross_bps",
        "short_gross_bps",
        "long_net_bps_5",
        "short_net_bps_5",
    )
    for event in events.itertuples(index=False):
        observed = outcome_lookup.loc[event.event_id]
        expected = replay_outcome(
            event,
            sessions[(int(event.period), str(event.symbol_norm), str(event.session_date))],
        )
        categorical_errors += int(expected["score_status"] != observed["score_status"])
        if expected["score_status"] == "scored":
            for name in ("actual_class", "neutral_reason"):
                categorical_errors += int(str(expected[name]) != str(observed[name]))
            expected_touch = expected["first_touch_step"]
            observed_touch = observed["first_touch_step"]
            if expected_touch is None:
                categorical_errors += int(not pd.isna(observed_touch))
            else:
                categorical_errors += int(int(expected_touch) != int(observed_touch))
            for name in numeric_outcome_fields:
                error = compare_numeric(expected[name], observed[name])
                maximum_path_error = max(maximum_path_error, error)
                path_errors += int(error > 1e-10)
    checks["all_target_paths_and_payoffs_independently_replayed"] = (
        path_errors == 0 and categorical_errors == 0
    )
    errors["path_numeric_errors"] = path_errors
    errors["path_categorical_errors"] = categorical_errors

    probability_sum_error = float(
        np.abs(predictions[["p_long", "p_neutral", "p_short"]].sum(axis=1) - 1.0).max()
    )
    state_errors = 0
    payoff_errors = 0
    merged_outcomes = outcomes.set_index("event_id")
    for row in predictions.itertuples(index=False):
        width = float(row.barrier_bps)
        long_ev = (float(row.p_long) - float(row.p_short)) * width - 10.0
        short_ev = (float(row.p_short) - float(row.p_long)) * width - 10.0
        if long_ev > 0 and long_ev > short_ev:
            state = "long"
        elif short_ev > 0 and short_ev > long_ev:
            state = "short"
        else:
            state = "neutral"
        state_errors += int(state != row.economic_state)
        outcome = merged_outcomes.loc[row.event_id]
        expected_net = (
            float(outcome["long_net_bps_5"])
            if state == "long"
            else float(outcome["short_net_bps_5"])
            if state == "short"
            else 0.0
        )
        payoff_errors += int(abs(expected_net - float(row.realized_net_bps_5)) > 1e-10)
    checks["probabilities_normalised"] = probability_sum_error < 1e-12
    checks["economic_state_and_payoff_replayed"] = state_errors == 0 and payoff_errors == 0
    errors["economic_state_errors"] = state_errors
    errors["economic_payoff_errors"] = payoff_errors

    joined = events.merge(outcomes, on=["event_id", "period", "symbol_norm", "session_date", "decision_ordinal", "barrier_bps"], validate="one_to_one")
    usable = joined.loc[joined["score_status"].eq("scored")].replace([np.inf, -np.inf], np.nan)
    maximum_probability_error = 0.0
    refit_errors = 0
    sampled_dates: list[str] = []
    for period in (2025, 2026):
        dates = sorted(predictions.loc[predictions["period"].eq(period), "session_date"].unique())
        sampled_dates.extend([dates[0], dates[len(dates) // 2], dates[-1]])
    for date in sorted(set(sampled_dates)):
        score = usable.loc[usable["session_date"].eq(date)].copy()
        prior_dates = sorted(usable.loc[usable["session_date"].lt(date), "session_date"].unique())[-120:]
        train = usable.loc[usable["session_date"].isin(prior_dates)].copy()
        for model_id in MODELS:
            numeric, categorical = FEATURES[model_id]
            columns = [*numeric, *categorical]
            fitted = pipeline(numeric, categorical)
            fitted.fit(train[columns], train["actual_class"])
            raw = fitted.predict_proba(score[columns])
            labels = list(fitted.named_steps["model"].classes_)
            observed = predictions.loc[
                predictions["session_date"].eq(date) & predictions["model_id"].eq(model_id)
            ].sort_values("event_id")
            score = score.sort_values("event_id")
            if observed["event_id"].tolist() != score["event_id"].tolist():
                refit_errors += 1
                continue
            for label in CLASSES:
                error = np.max(
                    np.abs(raw[:, labels.index(label)] - observed[f"p_{label}"].to_numpy(float))
                )
                maximum_probability_error = max(maximum_probability_error, float(error))
                refit_errors += int(error > 1e-10)
    checks["sampled_prequential_models_independently_refit"] = refit_errors == 0
    errors["prequential_refit_errors"] = refit_errors

    metric_errors = 0
    maximum_metric_error = 0.0
    reported = pd.read_csv(artifact_root / "model_metrics.csv").set_index(["period", "model_id"])
    for (period, model_id), group in predictions.groupby(["period", "model_id"], sort=True):
        probability = group[["p_long", "p_neutral", "p_short"]].to_numpy(float)
        actual_matrix = np.column_stack(
            [group["actual_class"].eq(label).to_numpy(float) for label in CLASSES]
        )
        calculated = {
            "accuracy": accuracy_score(group["actual_class"], group["predicted_class"]),
            "balanced_accuracy": balanced_accuracy_score(
                group["actual_class"], group["predicted_class"]
            ),
            "macro_f1": f1_score(
                group["actual_class"],
                group["predicted_class"],
                labels=list(CLASSES),
                average="macro",
                zero_division=0,
            ),
            "log_loss": log_loss(group["actual_class"], probability, labels=list(CLASSES)),
            "multiclass_brier": float(np.mean(np.sum((probability - actual_matrix) ** 2, axis=1))),
        }
        for name, expected in calculated.items():
            error = abs(float(reported.loc[(period, model_id), name]) - float(expected))
            maximum_metric_error = max(maximum_metric_error, error)
            metric_errors += int(error > 1e-10)
    checks["aggregate_predictive_metrics_recomputed"] = metric_errors == 0
    errors["aggregate_metric_errors"] = metric_errors

    checks["research_only_opened_status_preserved"] = (
        contract["safety"]["research_only"]
        and not contract["safety"]["live_ordering_enabled"]
        and contract["safety"]["order_placement"] == "disabled"
    )
    passed = sum(checks.values())
    result = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "checks": checks,
        "errors": errors,
        "maximum_errors": {
            "causal_feature": maximum_feature_error,
            "path_numeric": maximum_path_error,
            "probability_sum": probability_sum_error,
            "sampled_prequential_probability": maximum_probability_error,
            "aggregate_metric": maximum_metric_error,
        },
        "passed": passed,
        "total": len(checks),
    }
    write_json(artifact_root / "independent_audit.json", result)
    files = {}
    for path in sorted(artifact_root.iterdir(), key=lambda item: item.name):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        artifact_root / "artifact_manifest.json",
        {
            "contract_id": CONTRACT_ID,
            "research_only": True,
            "files": files,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifact_root)
    print(json.dumps({"passed": result["passed"], "total": result["total"]}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
