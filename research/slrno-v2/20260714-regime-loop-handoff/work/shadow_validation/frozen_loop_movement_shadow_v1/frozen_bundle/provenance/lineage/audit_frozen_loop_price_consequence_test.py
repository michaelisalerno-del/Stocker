"""Independent audit for the frozen loop-score price-consequence test."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


STATE = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
BACKWARD = Path("/private/tmp/stocker_sealed_backward_2023_complete_detector_20260710")
PATH_ROOT = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710")
RAW_CURRENT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock"
)
RAW_BACKWARD = Path(
    "/private/tmp/stocker_eodhd_pre2024_intraday_20260710/"
    "source=eodhd/instrument_type=stock"
)
ARTIFACT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
OUT = ARTIFACT / "independent_artifact_audit.json"
EXPECTED_SOURCE_MANIFEST_HASH = (
    "b63b51dc41c7868fc54cb0a9b1114e9095b2fdd89a9510baaec7c375db1da619"
)
RUN_PATHS = {
    "train_2024": STATE / "train_2024_filtered_runs.csv",
    "2025": STATE / "test_2025_filtered_runs.csv",
    "2023": BACKWARD / "backward_2023_filtered_runs.parquet",
}
YEARS = {"train_2024": 2024, "2025": 2025, "2023": 2023}
RAW_ROOTS = {"train_2024": RAW_CURRENT, "2025": RAW_CURRENT, "2023": RAW_BACKWARD}
HORIZONS = (6, 12, 24)
REPRESENTATIONS = ("state_context", "raw_history", "loop_scores")
CONTINUOUS_TARGETS = (
    "signed_return_bps",
    "absolute_return_bps",
    "future_range_bps",
)
NUMERIC_CONTROLS = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
SEED = 20260710
K = 8
END = 8
TOKENS = 648
EPS = 1e-12


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def provider_path(root: Path, symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def source_path(name: str) -> Path:
    fixed = {
        "path_model_parameters.npz": PATH_ROOT / "model_parameters.npz",
        "path_gates.json": PATH_ROOT / "gates.json",
        "fixed_cycle_shuffled_nulls.csv": STATE / "fixed_cycle_shuffled_nulls.csv",
        "train_2024_filtered_runs.csv": RUN_PATHS["train_2024"],
        "test_2025_filtered_runs.csv": RUN_PATHS["2025"],
        "backward_2023_filtered_runs.parquet": RUN_PATHS["2023"],
    }
    if name in fixed:
        return fixed[name]
    if not name.startswith("provider_") or not name.endswith(".parquet"):
        raise KeyError(name)
    stem = name.removeprefix("provider_").removesuffix(".parquet")
    year_text, symbol = stem.split("_", 1)
    year = int(year_text)
    root = RAW_BACKWARD if year == 2023 else RAW_CURRENT
    return provider_path(root, symbol)


def canonical(core: tuple[int, ...]) -> tuple[int, ...]:
    return min(core[index:] + core[:index] for index in range(len(core)))


def routes(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (current,)
            for index, state in enumerate(core)
            if state == current
        }
    )


def cycles() -> list[dict[str, Any]]:
    source = pd.read_csv(STATE / "fixed_cycle_shuffled_nulls.csv")
    output = []
    for index, value in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in value.split("->"))
        core = canonical(closed[:-1])
        output.append(
            {
                "cycle_id": f"cycle_{index:02d}",
                "cycle": "->".join(str(state) for state in core + (core[0],)),
                "core": core,
            }
        )
    return output


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def token(prev2: np.ndarray, prev1: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (prev2 * 9 + prev1) * 8 + current


def token_matrix(values: np.ndarray) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (np.ones(len(values), dtype=np.float32), (np.arange(len(values)), values)),
        shape=(len(values), TOKENS),
    )


def route_probability(
    frame: pd.DataFrame,
    route: tuple[int, ...],
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    probability = np.ones(len(frame))
    prev2 = frame["previous_state_2"].to_numpy(int)
    prev1 = frame["previous_state_1"].to_numpy(int)
    current = np.full(len(frame), route[0], dtype=int)
    for destination in route[1:]:
        ids = token(prev2, prev1, current)
        logits = parameters["history_intercept"][None, :] + parameters[
            "history_coef"
        ][:, ids].T
        probability *= softmax(logits)[:, destination]
        prev2, prev1, current = prev1, current, np.full(len(frame), destination)
    return probability


def rebuild_loop_scores(
    frame: pd.DataFrame,
    definitions: list[dict[str, Any]],
    parameters: dict[str, np.ndarray],
) -> float:
    maximum_error = 0.0
    for index, cycle in enumerate(definitions, start=1):
        expected = np.zeros(len(frame))
        for current in sorted(set(cycle["core"])):
            mask = frame["state"].eq(current).to_numpy()
            subset = frame.loc[mask].reset_index(drop=True)
            values = np.zeros(len(subset))
            for route in routes(cycle["core"], current):
                values += route_probability(subset, route, parameters)
            expected[mask] = values
        maximum_error = max(
            maximum_error,
            float(
                np.max(
                    np.abs(
                        expected
                        - frame[f"loop_score_{index:02d}"].to_numpy(float)
                    )
                )
            ),
        )
    return maximum_error


def read_runs(period: str) -> pd.DataFrame:
    path = RUN_PATHS[period]
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True)
    frame = frame.sort_values(
        ["symbol_norm", "session_date", "start_pos"], kind="stable"
    ).reset_index(drop=True)
    grouped = frame.groupby(["symbol_norm", "session_date"], sort=False)["state"]
    prev1 = grouped.shift(1).fillna(END).astype(int)
    prev2 = grouped.shift(2).fillna(END).astype(int)
    assert np.array_equal(prev1, frame["previous_state_1"].astype(int))
    assert np.array_equal(prev2, frame["previous_state_2"].astype(int))
    return frame


def rebuild_prices(symbol: str, root: Path, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        provider_path(root, symbol),
        columns=["timestamp", "open", "high", "low", "close"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.loc[
        frame["timestamp"].ge(pd.Timestamp(f"{year}-01-01", tz="UTC"))
        & frame["timestamp"].lt(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
    ].dropna()
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["session_date"] = local.dt.strftime("%Y-%m-%d")
    frame["symbol_norm"] = symbol
    frame["bar_index_in_session"] = frame.groupby("session_date").cumcount()
    grouped = frame.groupby("session_date", sort=False)
    previous = grouped["close"].shift(1)
    first = frame["bar_index_in_session"].eq(0)
    frame["current_bar_log_return"] = np.log(
        frame["close"] / previous.where(~first, frame["open"])
    )
    frame["return_sum_6"] = grouped["current_bar_log_return"].transform(
        lambda values: values.rolling(6, min_periods=1).sum()
    )
    frame["mean_abs_return_12"] = grouped["current_bar_log_return"].transform(
        lambda values: values.abs().rolling(12, min_periods=1).mean()
    )
    frame["session_return"] = np.log(
        frame["close"] / grouped["open"].transform("first")
    )
    frame["bar_range_pct"] = (frame["high"] - frame["low"]) / frame["open"]
    for horizon in HORIZONS:
        signed = 10000 * np.log(
            grouped["close"].shift(-horizon).to_numpy(float)
            / frame["close"].to_numpy(float)
        )
        exact = np.ones(len(frame), dtype=bool)
        highs = []
        lows = []
        for step in range(1, horizon + 1):
            exact &= (
                (grouped["timestamp"].shift(-step) - frame["timestamp"])
                .eq(pd.Timedelta(minutes=5 * step))
                .fillna(False)
                .to_numpy(bool)
            )
            highs.append(grouped["high"].shift(-step).to_numpy(float))
            lows.append(grouped["low"].shift(-step).to_numpy(float))
        high_matrix = np.column_stack(highs)
        low_matrix = np.column_stack(lows)
        valid = np.isfinite(high_matrix).any(axis=1)
        high = np.full(len(frame), np.nan)
        low = np.full(len(frame), np.nan)
        high[valid] = np.nanmax(high_matrix[valid], axis=1)
        low[valid] = np.nanmin(low_matrix[valid], axis=1)
        future_range = 10000 * (high - low) / frame["close"].to_numpy(float)
        signed[~exact] = np.nan
        future_range[~exact] = np.nan
        frame[f"direction_{horizon}"] = np.where(
            exact, (signed > 0).astype(float), np.nan
        )
        frame[f"signed_return_bps_{horizon}"] = signed
        frame[f"absolute_return_bps_{horizon}"] = np.abs(signed)
        frame[f"future_range_bps_{horizon}"] = future_range
    return frame


def verify_anchor_panel(
    period: str,
    panel: pd.DataFrame,
    definitions: list[dict[str, Any]],
    parameters: dict[str, np.ndarray],
) -> dict[str, Any]:
    year = YEARS[period]
    runs = read_runs(period)
    keys = ["symbol_norm", "session_date", "start_timestamp"]
    run_columns = keys + [
        "state",
        "previous_state_1",
        "previous_state_2",
        "b0_state_numeric",
        "b0_high_stress",
    ]
    joined = panel.merge(
        runs[run_columns],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "__run"),
    )
    assert joined["state__run"].notna().all()
    assert np.array_equal(joined["state"].astype(int), joined["state__run"].astype(int))
    assert np.array_equal(
        joined["previous_state_1"].astype(int),
        joined["previous_state_1__run"].astype(int),
    )
    assert np.array_equal(
        joined["previous_state_2"].astype(int),
        joined["previous_state_2__run"].astype(int),
    )
    assert panel["bar_index_in_session"].le(53).all()
    assert set(pd.to_datetime(panel["session_date"]).dt.year) == {year}

    price_errors = []
    outcome_errors = []
    for symbol, selected in panel.groupby("symbol_norm", sort=True):
        rebuilt = rebuild_prices(str(symbol), RAW_ROOTS[period], year)
        observed = selected.sort_values("start_timestamp")
        expected = rebuilt.merge(
            observed[["start_timestamp"]],
            left_on="timestamp",
            right_on="start_timestamp",
            how="inner",
            validate="one_to_one",
        ).sort_values("start_timestamp")
        assert len(expected) == len(observed)
        for column in (
            "bar_index_in_session",
            "current_bar_log_return",
            "return_sum_6",
            "mean_abs_return_12",
            "session_return",
            "bar_range_pct",
        ):
            price_errors.append(
                float(
                    np.max(
                        np.abs(
                            expected[column].to_numpy(float)
                            - observed[column].to_numpy(float)
                        )
                    )
                )
            )
        for horizon in HORIZONS:
            for target in (
                "direction",
                "signed_return_bps",
                "absolute_return_bps",
                "future_range_bps",
            ):
                column = f"{target}_{horizon}"
                outcome_errors.append(
                    float(
                        np.max(
                            np.abs(
                                expected[column].to_numpy(float)
                                - observed[column].to_numpy(float)
                            )
                        )
                    )
                )
    expected_token = token(
        panel["previous_state_2"].to_numpy(int),
        panel["previous_state_1"].to_numpy(int),
        panel["state"].to_numpy(int),
    )
    assert np.array_equal(expected_token, panel["history_token"].to_numpy(int))
    loop_error = rebuild_loop_scores(panel, definitions, parameters)
    return {
        "rows": len(panel),
        "maximum_price_control_error": max(price_errors),
        "maximum_outcome_error": max(outcome_errors),
        "maximum_loop_score_error": loop_error,
    }


def build_features(
    panels: dict[str, pd.DataFrame], manifest: dict[str, Any]
) -> tuple[
    dict[str, sparse.csr_matrix],
    dict[str, dict[str, sparse.csr_matrix]],
    dict[str, StandardScaler],
]:
    medians = pd.Series(manifest["numeric_medians"])[list(NUMERIC_CONTROLS)]

    def raw(frame: pd.DataFrame) -> dict[str, sparse.csr_matrix]:
        numeric = frame[list(NUMERIC_CONTROLS)].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(medians)
        state = sparse.csr_matrix(
            np.eye(K, dtype=np.float32)[frame["state"].to_numpy(int)]
        )
        context = sparse.hstack(
            (state, sparse.csr_matrix(numeric.to_numpy(np.float32))), format="csr"
        )
        history = token_matrix(frame["history_token"].to_numpy(int))
        loop = sparse.csr_matrix(
            frame[[f"loop_score_{index:02d}" for index in range(1, 21)]].to_numpy(
                np.float32
            )
        )
        return {
            "state_context": context,
            "raw_history": sparse.hstack((context, history), format="csr"),
            "loop_scores": sparse.hstack((context, loop), format="csr"),
        }

    train_raw = raw(panels["train_2024"])
    test_raw = {period: raw(panels[period]) for period in ("2025", "2023")}
    train_x = {}
    test_x = {period: {} for period in ("2025", "2023")}
    scalers = {}
    for representation in REPRESENTATIONS:
        scaler = StandardScaler(with_mean=False).fit(train_raw[representation])
        train_x[representation] = scaler.transform(train_raw[representation]).tocsr()
        for period in test_x:
            test_x[period][representation] = scaler.transform(
                test_raw[period][representation]
            ).tocsr()
        scalers[representation] = scaler
    return train_x, test_x, scalers


def refit_and_verify_models(
    panels: dict[str, pd.DataFrame],
    train_x: dict[str, sparse.csr_matrix],
    test_x: dict[str, dict[str, sparse.csr_matrix]],
    scalers: dict[str, StandardScaler],
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    saved = dict(np.load(ARTIFACT / "outcome_model_parameters.npz"))
    predictions = {
        period: pd.read_parquet(ARTIFACT / f"price_predictions_{period}.parquet")
        for period in ("2025", "2023")
    }
    errors = {
        "scaler": 0.0,
        "model_parameter": 0.0,
        "prediction": 0.0,
    }
    for representation, scaler in scalers.items():
        for name, values in (
            ("scale", scaler.scale_),
            ("mean", scaler.mean_),
            ("var", scaler.var_),
        ):
            errors["scaler"] = max(
                errors["scaler"],
                float(
                    np.max(
                        np.abs(
                            values
                            - saved[f"{representation}__scaler_{name}"]
                        )
                    )
                ),
            )
    train = panels["train_2024"]
    for horizon in HORIZONS:
        y_direction = train[f"direction_{horizon}"].to_numpy(int)
        for representation in REPRESENTATIONS:
            logistic = LogisticRegression(
                C=0.2, solver="lbfgs", max_iter=500, random_state=SEED
            ).fit(train_x[representation], y_direction)
            prefix = f"{representation}__direction__h{horizon}"
            errors["model_parameter"] = max(
                errors["model_parameter"],
                float(np.max(np.abs(logistic.coef_ - saved[f"{prefix}__coef"]))),
                float(
                    np.max(
                        np.abs(
                            logistic.intercept_ - saved[f"{prefix}__intercept"]
                        )
                    )
                ),
            )
            positive = int(np.flatnonzero(logistic.classes_ == 1)[0])
            for period in predictions:
                expected = logistic.predict_proba(test_x[period][representation])[
                    :, positive
                ]
                observed = predictions[period][
                    f"{representation}__direction_probability_{horizon}"
                ].to_numpy(float)
                errors["prediction"] = max(
                    errors["prediction"], float(np.max(np.abs(expected - observed)))
                )
            for target in CONTINUOUS_TARGETS:
                ridge = Ridge(alpha=10.0, solver="lsqr").fit(
                    train_x[representation],
                    train[f"{target}_{horizon}"].to_numpy(float),
                )
                ridge_prefix = f"{representation}__{target}__h{horizon}"
                errors["model_parameter"] = max(
                    errors["model_parameter"],
                    float(
                        np.max(
                            np.abs(
                                ridge.coef_ - saved[f"{ridge_prefix}__coef"]
                            )
                        )
                    ),
                    abs(
                        float(ridge.intercept_)
                        - float(saved[f"{ridge_prefix}__intercept"][0])
                    ),
                )
                for period in predictions:
                    expected = ridge.predict(test_x[period][representation])
                    observed = predictions[period][
                        f"{representation}__{target}_prediction_{horizon}"
                    ].to_numpy(float)
                    errors["prediction"] = max(
                        errors["prediction"],
                        float(np.max(np.abs(expected - observed))),
                    )
    return predictions, errors


def direction_losses(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
    }


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    block = min(5, len(values))
    blocks = np.asarray(
        [values[start : start + block] for start in range(len(values) - block + 1)]
    )
    needed = math.ceil(len(values) / block)
    rng = np.random.default_rng(seed)
    draws = np.empty(5000)
    for index in range(5000):
        chosen = rng.integers(0, len(blocks), size=needed)
        draws[index] = blocks[chosen].reshape(-1)[: len(values)].mean()
    return values.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)


def close(left: Any, right: Any, tolerance: float = 1e-10) -> None:
    if pd.isna(left) and pd.isna(right):
        return
    if abs(float(left) - float(right)) > tolerance:
        raise AssertionError(f"{left} != {right}")


def verify_metrics(period: str, prediction: pd.DataFrame) -> dict[str, Any]:
    long = pd.read_parquet(ARTIFACT / f"price_scoring_long_{period}.parquet")
    direction_saved = pd.read_csv(ARTIFACT / "direction_metrics.csv")
    direction_saved = direction_saved.loc[
        direction_saved["period"].astype(str).eq(period)
    ]
    continuous_saved = pd.read_csv(ARTIFACT / "continuous_metrics.csv")
    continuous_saved = continuous_saved.loc[
        continuous_saved["period"].astype(str).eq(period)
    ]
    losses_direction = {}
    losses_continuous = {target: {} for target in CONTINUOUS_TARGETS}
    for representation in REPRESENTATIONS:
        y = long["direction_target"].to_numpy(int)
        p = long[f"{representation}__direction_probability"].to_numpy(float)
        losses_direction[representation] = direction_losses(y, p)
        for horizon in HORIZONS:
            mask = long["horizon"].eq(horizon).to_numpy()
            row = direction_saved.loc[
                direction_saved["representation"].eq(representation)
                & direction_saved["horizon"].eq(horizon)
            ].iloc[0]
            close(row["log_loss"], losses_direction[representation]["log_loss"][mask].mean())
            close(row["brier"], losses_direction[representation]["brier"][mask].mean())
            close(row["auc"], roc_auc_score(y[mask], p[mask]))
        for target in CONTINUOUS_TARGETS:
            outcome = long[f"{target}_target"].to_numpy(float)
            predicted = long[f"{representation}__{target}_prediction"].to_numpy(float)
            losses_continuous[target][representation] = {
                "mse": (predicted - outcome) ** 2,
                "mae": np.abs(predicted - outcome),
            }
            for horizon in HORIZONS:
                mask = long["horizon"].eq(horizon).to_numpy()
                row = continuous_saved.loc[
                    continuous_saved["representation"].eq(representation)
                    & continuous_saved["target"].eq(target)
                    & continuous_saved["horizon"].eq(horizon)
                ].iloc[0]
                close(row["mse"], losses_continuous[target][representation]["mse"][mask].mean())
                close(row["mae"], losses_continuous[target][representation]["mae"][mask].mean())
                close(row["pearson_correlation"], np.corrcoef(predicted[mask], outcome[mask])[0, 1])

    comparisons = pd.read_csv(ARTIFACT / "paired_comparisons.csv")
    comparisons = comparisons.loc[comparisons["period"].astype(str).eq(period)]
    seed_offset = 1000 if period == "2025" else 2000
    comparison_counter = 0
    specs = (
        ("loop_scores", "state_context"),
        ("raw_history", "state_context"),
        ("loop_scores", "raw_history"),
    )
    for candidate, baseline in specs:
        target_losses = [("direction", "log_loss", losses_direction), ("direction", "brier", losses_direction)]
        for target in CONTINUOUS_TARGETS:
            target_losses.extend(
                [
                    (target, "mse", losses_continuous[target]),
                    (target, "mae", losses_continuous[target]),
                ]
            )
        for target, loss_name, collection in target_losses:
            difference = collection[candidate][loss_name] - collection[baseline][loss_name]
            daily = (
                pd.DataFrame(
                    {"date": long["session_date"], "difference": difference}
                )
                .groupby("date", sort=True)["difference"]
                .mean()
                .to_numpy()
            )
            mean, low, high = bootstrap(
                daily, SEED + seed_offset + comparison_counter
            )
            comparison_counter += 1
            horizons = pd.Series(difference).groupby(long["horizon"]).mean()
            quarters = pd.Series(difference).groupby(long["quarter"]).mean()
            deletions = [
                difference[long["symbol_norm"].ne(symbol)].mean()
                for symbol in sorted(long["symbol_norm"].unique())
            ]
            row = comparisons.loc[
                comparisons["candidate"].eq(candidate)
                & comparisons["baseline"].eq(baseline)
                & comparisons["target"].eq(target)
                & comparisons["loss"].eq(loss_name)
            ].iloc[0]
            close(row["daily_mean_difference"], mean)
            close(row["daily_ci_low"], low)
            close(row["daily_ci_high"], high)
            assert int(row["negative_horizon_count"]) == int((horizons < 0).sum())
            assert int(row["negative_quarter_count"]) == int((quarters < 0).sum())
            close(row["leave_one_symbol_max_difference"], max(deletions))

    gates = json.loads((ARTIFACT / "gates.json").read_text())["periods"][period]
    direction_rows = comparisons.loc[
        comparisons["candidate"].eq("loop_scores")
        & comparisons["baseline"].eq("state_context")
        & comparisons["target"].eq("direction")
    ]
    signed_rows = comparisons.loc[
        comparisons["candidate"].eq("loop_scores")
        & comparisons["baseline"].eq("state_context")
        & comparisons["target"].eq("signed_return_bps")
    ]
    movement_passes = {}
    for target in ("absolute_return_bps", "future_range_bps"):
        rows = comparisons.loc[
            comparisons["candidate"].eq("loop_scores")
            & comparisons["baseline"].eq("state_context")
            & comparisons["target"].eq(target)
        ]
        robust = bool(
            rows["daily_ci_high"].lt(0).all()
            and rows["negative_horizon_count"].eq(3).all()
            and rows["negative_quarter_count"].eq(4).all()
            and rows["leave_one_symbol_all_negative"].all()
            and rows.loc[rows["loss"].eq("mse"), "relative_improvement"].iloc[0]
            >= 0.0025
        )
        movement_passes[target] = robust
        assert robust == gates["movement"][target]["pass"]
    return {
        "rows": len(long),
        "direction_gate_stored": bool(gates["direction"]["pass"]),
        "signed_gate_stored": bool(gates["signed_return"]["pass"]),
        "movement_gate_reconstructed": movement_passes,
        "direction_primary_relative": float(
            direction_rows.loc[
                direction_rows["loss"].eq("log_loss"), "relative_improvement"
            ].iloc[0]
        ),
        "signed_primary_relative": float(
            signed_rows.loc[
                signed_rows["loss"].eq("mse"), "relative_improvement"
            ].iloc[0]
        ),
    }


def main() -> None:
    checks = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "details": details})
        if not condition:
            raise AssertionError(name)

    check(
        "source_manifest_hash_frozen",
        digest(ARTIFACT / "source_hashes.json") == EXPECTED_SOURCE_MANIFEST_HASH,
    )
    source_hashes = json.loads((ARTIFACT / "source_hashes.json").read_text())
    current_hashes = {name: digest(source_path(name)) for name in source_hashes}
    check("all_source_hashes_current", current_hashes == source_hashes)
    check(
        "path_predecessor_retained",
        json.loads((PATH_ROOT / "gates.json").read_text())["history_retained"]
        is True,
    )
    definitions = cycles()
    check("twenty_fixed_cycles", len(definitions) == 20)
    parameters = dict(np.load(PATH_ROOT / "model_parameters.npz"))
    panels = {
        period: pd.read_parquet(ARTIFACT / f"anchor_panel_{period}.parquet")
        for period in ("train_2024", "2025", "2023")
    }
    panel_results = {}
    for period, panel in panels.items():
        result = verify_anchor_panel(period, panel, definitions, parameters)
        panel_results[period] = result
        check(
            f"{period}_price_controls_exact",
            result["maximum_price_control_error"] < 1e-11,
            result,
        )
        check(
            f"{period}_outcomes_exact",
            result["maximum_outcome_error"] < 1e-9,
            result,
        )
        check(
            f"{period}_loop_scores_exact",
            result["maximum_loop_score_error"] < 1e-11,
            result,
        )

    manifest = json.loads((ARTIFACT / "feature_manifest.json").read_text())
    check("volume_not_used", manifest["volume_label"] == "historical_volume_not_used")
    check("frozen_feature_widths", manifest["feature_widths"] == {"state_context": 17, "raw_history": 665, "loop_scores": 37})
    train_x, test_x, scalers = build_features(panels, manifest)
    predictions, model_errors = refit_and_verify_models(
        panels, train_x, test_x, scalers
    )
    check("scalers_refit_exact", model_errors["scaler"] < 1e-12, model_errors)
    check("outcome_models_refit_exact", model_errors["model_parameter"] < 1e-10, model_errors)
    check("predictions_reproduce", model_errors["prediction"] < 1e-10, model_errors)

    metric_results = {
        period: verify_metrics(period, predictions[period])
        for period in ("2025", "2023")
    }
    for period, result in metric_results.items():
        check(f"{period}_direction_reproduces_as_fail", result["direction_gate_stored"] is False, result)
        check(f"{period}_signed_return_reproduces_as_fail", result["signed_gate_stored"] is False, result)
        check(f"{period}_absolute_movement_reproduces", result["movement_gate_reconstructed"]["absolute_return_bps"], result)
        check(f"{period}_range_movement_reproduces", result["movement_gate_reconstructed"]["future_range_bps"], result)
    gates = json.loads((ARTIFACT / "gates.json").read_text())
    check("directional_consequence_rejected", gates["directional_consequence_retained"] is False)
    check("movement_consequence_retained", gates["movement_consequence_retained"] is True)
    check("economic_edge_not_claimed", gates["economic_edge_claim"] is False)
    check("research_only", gates["research_only"] is True)
    check("live_ordering_disabled", gates["live_ordering_enabled"] is False)
    check("order_placement_disabled", gates["order_placement"] == "disabled")
    scoring_columns = set(pd.read_parquet(ARTIFACT / "price_scoring_long_2025.parquet").columns)
    forbidden = {"pnl", "order", "position", "spread", "slippage", "entry", "exit"}
    check("no_strategy_or_cost_columns", not scoring_columns.intersection(forbidden))

    payload = {
        "all_passed": all(item["pass"] for item in checks),
        "check_count": len(checks),
        "checks": checks,
        "panel_results": panel_results,
        "model_errors": model_errors,
        "metric_results": metric_results,
    }
    encoder = lambda value: value.item() if isinstance(value, np.generic) else str(value)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True, default=encoder) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=encoder))


if __name__ == "__main__":
    main()
