"""Independent replay audit for orientation-calibration algorithms V1.

This file does not import the production runner.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-regime-loop-orientation-calibration-algorithms-v1.json"
RUNNER = HERE / "run_regime_loop_orientation_calibration_algorithms_v1.py"
SOURCE = Path(
    "/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711/linkage_predictions_2024_sep_dec.parquet"
)
SOURCE_AUDIT = Path(
    "/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711/independent_audit.json"
)
ROOT = Path(
    "/private/tmp/stocker_regime_loop_orientation_calibration_algorithms_v1_20260711"
)

EXPECTED = {
    "contract": "900f14c8c43456a28e3532be1cc499fe61d9b4b26b0f0904d0afafa1c7ad525d",
    "runner": "ecc4cc1d3e2fb574bb0ea0d792bbc5bae5083ec9264b9857a46b9d6ac4f5919a",
    "source": "99374428d372711b233cf6dfbe59a18f5667e032ef3039b2ae05df13400cd660",
    "source_audit": "92343c008f0cd585c3e02c1e3c60905aa4f9cde7e9c9b99111076a2cd8be300f",
}
MONTHS = ("2024-09", "2024-10", "2024-11", "2024-12")
VALIDATION_MONTHS = ("2024-11", "2024-12")
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
ALGORITHMS = (
    "weighted_isotonic",
    "beta_global",
    "orientation_residual",
    "orientation_clock_residual",
)
COMPARISONS = ("baseline", "raw_reference")
SEED = 20260711
EPSILON = 1e-6
LOSS_EPSILON = 1e-12
BOOTSTRAP_DRAWS = 4999
SIGN_FLIP_DRAWS = 9999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def ycol(target: str, horizon: int, tier: str) -> str:
    return f"joint_target__{target}__h{horizon}__{tier}"


def source_p(model: str, target: str, horizon: int, tier: str) -> str:
    return f"link__{model}__{target}__h{horizon}__{tier}"


def acol(algorithm: str, target: str, horizon: int, tier: str) -> str:
    return f"algorithm__{algorithm}__{target}__h{horizon}__{tier}"


def source_columns() -> list[str]:
    return [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "month",
        "cycle_index",
        "cycle_id",
        "state",
        "current_state",
        "entry_clock_quartile",
        "inverse_compatible_weight",
        *[ycol(target, horizon, tier) for target in TARGETS for horizon in HORIZONS for tier in TIERS],
        *[source_p(model, target, horizon, tier) for model in ("baseline", "raw_full_link") for target in TARGETS for horizon in HORIZONS for tier in TIERS],
    ]


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(SOURCE, columns=source_columns())
    frame["month"] = frame["month"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    orientation = (
        frame[["cycle_id", "current_state"]]
        .drop_duplicates()
        .sort_values(["cycle_id", "current_state"], kind="stable")
        .reset_index(drop=True)
    )
    orientation["orientation_index"] = np.arange(len(orientation))
    clock = (
        frame[["cycle_id", "current_state", "entry_clock_quartile"]]
        .drop_duplicates()
        .sort_values(["cycle_id", "current_state", "entry_clock_quartile"], kind="stable")
        .reset_index(drop=True)
    )
    clock["orientation_clock_index"] = np.arange(len(clock))
    frame = frame.merge(
        orientation,
        on=["cycle_id", "current_state"],
        validate="many_to_one",
    ).merge(
        clock,
        on=["cycle_id", "current_state", "entry_clock_quartile"],
        validate="many_to_one",
    )
    return frame, orientation, clock


def clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, float), EPSILON, 1 - EPSILON)


def logit(values: np.ndarray) -> np.ndarray:
    values = clip(values)
    return np.log(values / (1 - values))


def global_features(
    frame: pd.DataFrame, target: str, horizon: int, tier: str
) -> tuple[np.ndarray, np.ndarray]:
    baseline = logit(frame[source_p("baseline", target, horizon, tier)].to_numpy(float))
    raw = logit(frame[source_p("raw_full_link", target, horizon, tier)].to_numpy(float))
    residual = raw - baseline
    return np.column_stack((baseline, residual)), residual


def categorical(indices: np.ndarray, width: int, values: np.ndarray) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (values, (np.arange(len(indices)), indices)), shape=(len(indices), width)
    )


def orientation_features(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    tier: str,
    scaler: StandardScaler,
    orientation_width: int,
    clock_width: int | None,
) -> sparse.csr_matrix:
    values, residual = global_features(frame, target, horizon, tier)
    orientation = frame["orientation_index"].to_numpy(int)
    parts = [
        sparse.csr_matrix(scaler.transform(values)),
        categorical(orientation, orientation_width, np.full(len(frame), 0.25)),
        categorical(orientation, orientation_width, residual * 0.125),
    ]
    if clock_width is not None:
        clock = frame["orientation_clock_index"].to_numpy(int)
        parts.extend(
            [
                categorical(clock, clock_width, np.full(len(frame), 0.125)),
                categorical(clock, clock_width, residual * 0.0625),
            ]
        )
    return sparse.hstack(parts, format="csr")


def fit_logistic(x, y: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    return LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-10,
        random_state=SEED,
    ).fit(x, y, sample_weight=weights)


def refit(
    frame: pd.DataFrame,
    ledger: pd.DataFrame,
    orientation_width: int,
    clock_width: int,
) -> tuple[float, float, int]:
    parameters = np.load(ROOT / "model_parameters.npz")
    maximum_prediction = 0.0
    maximum_parameter = 0.0
    fits = 0
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                for month in VALIDATION_MONTHS:
                    training = frame.loc[frame["month"].lt(month)]
                    validation = frame.loc[frame["month"].eq(month)]
                    stored = ledger.loc[ledger["month"].eq(month)]
                    weights = training["inverse_compatible_weight"].to_numpy(float)
                    y = training[ycol(target, horizon, tier)].to_numpy(int)
                    raw_train = clip(training[source_p("raw_full_link", target, horizon, tier)])
                    raw_validation = clip(validation[source_p("raw_full_link", target, horizon, tier)])
                    prefix = f"{target}__h{horizon}__{tier}__{month}"
                    iso = IsotonicRegression(
                        y_min=EPSILON,
                        y_max=1 - EPSILON,
                        increasing=True,
                        out_of_bounds="clip",
                    ).fit(raw_train, y, sample_weight=weights)
                    probability = clip(iso.predict(raw_validation))
                    maximum_prediction = max(
                        maximum_prediction,
                        float(
                            np.max(
                                np.abs(
                                    probability
                                    - stored[acol("weighted_isotonic", target, horizon, tier)]
                                )
                            )
                        ),
                    )
                    maximum_parameter = max(
                        maximum_parameter,
                        float(np.max(np.abs(parameters[f"{prefix}__weighted_isotonic__x"] - iso.X_thresholds_))),
                        float(np.max(np.abs(parameters[f"{prefix}__weighted_isotonic__y"] - iso.y_thresholds_))),
                    )
                    fits += 1

                    beta_train = np.column_stack((np.log(raw_train), np.log1p(-raw_train)))
                    beta_validation = np.column_stack((np.log(raw_validation), np.log1p(-raw_validation)))
                    beta_scaler = StandardScaler().fit(beta_train, sample_weight=weights)
                    beta_model = fit_logistic(beta_scaler.transform(beta_train), y, weights)
                    probability = clip(beta_model.predict_proba(beta_scaler.transform(beta_validation))[:, 1])
                    maximum_prediction = max(
                        maximum_prediction,
                        float(np.max(np.abs(probability - stored[acol("beta_global", target, horizon, tier)]))),
                    )
                    maximum_parameter = max(
                        maximum_parameter,
                        compare_parameters(parameters, f"{prefix}__beta_global", beta_scaler, beta_model),
                    )
                    fits += 1

                    raw_global, _ = global_features(training, target, horizon, tier)
                    global_scaler = StandardScaler().fit(raw_global, sample_weight=weights)
                    for algorithm, included_clock in (
                        ("orientation_residual", None),
                        ("orientation_clock_residual", clock_width),
                    ):
                        train_x = orientation_features(
                            training, target, horizon, tier, global_scaler, orientation_width, included_clock
                        )
                        validation_x = orientation_features(
                            validation, target, horizon, tier, global_scaler, orientation_width, included_clock
                        )
                        model = fit_logistic(train_x, y, weights)
                        probability = clip(model.predict_proba(validation_x)[:, 1])
                        maximum_prediction = max(
                            maximum_prediction,
                            float(np.max(np.abs(probability - stored[acol(algorithm, target, horizon, tier)]))),
                        )
                        maximum_parameter = max(
                            maximum_parameter,
                            compare_parameters(parameters, f"{prefix}__{algorithm}", global_scaler, model),
                        )
                        fits += 1
    parameters.close()
    return maximum_prediction, maximum_parameter, fits


def compare_parameters(
    parameters, prefix: str, scaler: StandardScaler, model: LogisticRegression
) -> float:
    return max(
        float(np.max(np.abs(parameters[f"{prefix}__mean"] - scaler.mean_))),
        float(np.max(np.abs(parameters[f"{prefix}__scale"] - scaler.scale_))),
        float(np.max(np.abs(parameters[f"{prefix}__coef"] - model.coef_))),
        float(np.max(np.abs(parameters[f"{prefix}__intercept"] - model.intercept_))),
        float(np.max(np.abs(parameters[f"{prefix}__n_iter"] - model.n_iter_))),
    )


def losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), LOSS_EPSILON, 1 - LOSS_EPSILON)
    return (-(y * np.log(p) + (1 - y) * np.log(1 - p)), (y - p) ** 2)


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() else math.nan


def calibration(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray, minimum: int
) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    values: list[tuple[float, float]] = []
    for index in range(10):
        selected = bins == index
        if selected.sum() < minimum or weights[selected].sum() <= 0:
            continue
        error = abs(wmean(y[selected], weights[selected]) - wmean(p[selected], weights[selected]))
        values.append((weights[selected].sum(), error))
    if not values:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in values)
    return (
        float(sum(weight * error for weight, error in values) / total),
        float(max(error for _, error in values)),
        len(values),
    )


def probability(frame: pd.DataFrame, model: str, target: str, horizon: int, tier: str) -> np.ndarray:
    if model in ALGORITHMS:
        return frame[acol(model, target, horizon, tier)].to_numpy(float)
    return frame[
        source_p("baseline" if model == "baseline" else "raw_full_link", target, horizon, tier)
    ].to_numpy(float)


def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface, weights in (
        ("inverse_compatible", frame["inverse_compatible_weight"].to_numpy(float)),
        ("unweighted", np.ones(len(frame))),
    ):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    y = frame[ycol(target, horizon, tier)].to_numpy(int)
                    refs: dict[str, Any] = {}
                    for ref in ("baseline", "raw_reference"):
                        ll, brier = losses(y, probability(frame, ref, target, horizon, tier))
                        ece, maximum, bins = calibration(
                            y, probability(frame, ref, target, horizon, tier), weights, 250
                        )
                        refs[ref] = (wmean(ll, weights), wmean(brier, weights), ece, maximum, bins)
                    for algorithm in ALGORITHMS:
                        ll, brier = losses(y, probability(frame, algorithm, target, horizon, tier))
                        ece, maximum, bins = calibration(
                            y, probability(frame, algorithm, target, horizon, tier), weights, 250
                        )
                        ll_mean, brier_mean = wmean(ll, weights), wmean(brier, weights)
                        rows.append(
                            {
                                "weight_surface": surface,
                                "algorithm": algorithm,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "positives": int(y.sum()),
                                "weight_sum": float(weights.sum()),
                                "log_loss": ll_mean,
                                "baseline_log_loss": refs["baseline"][0],
                                "raw_log_loss": refs["raw_reference"][0],
                                "relative_log_loss_improvement_vs_baseline": (refs["baseline"][0] - ll_mean) / refs["baseline"][0],
                                "log_loss_difference_vs_baseline": ll_mean - refs["baseline"][0],
                                "log_loss_difference_vs_raw": ll_mean - refs["raw_reference"][0],
                                "brier": brier_mean,
                                "baseline_brier": refs["baseline"][1],
                                "raw_brier": refs["raw_reference"][1],
                                "brier_difference_vs_baseline": brier_mean - refs["baseline"][1],
                                "brier_difference_vs_raw": brier_mean - refs["raw_reference"][1],
                                "ece": ece,
                                "raw_ece": refs["raw_reference"][2],
                                "maximum_supported_bin_error": maximum,
                                "raw_maximum_supported_bin_error": refs["raw_reference"][3],
                                "supported_bins": bins,
                            }
                        )
    return pd.DataFrame(rows)


def pooled(frame: pd.DataFrame, algorithm: str, p75_only=False) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    tiers = ("p75",) if p75_only else TIERS
    storage = {key: [] for key in ("base_ll", "raw_ll", "cand_ll", "base_br", "raw_br", "cand_br")}
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in tiers:
                y = frame[ycol(target, horizon, tier)].to_numpy(int)
                for model, prefix in (("baseline", "base"), ("raw_reference", "raw"), (algorithm, "cand")):
                    ll, br = losses(y, probability(frame, model, target, horizon, tier))
                    storage[f"{prefix}_ll"].append(wmean(ll, weights))
                    storage[f"{prefix}_br"].append(wmean(br, weights))
    value = {key: float(np.mean(items)) for key, items in storage.items()}
    result = {
        "baseline_log_loss": value["base_ll"],
        "log_loss": value["cand_ll"],
        "relative_log_loss_improvement_vs_baseline": (value["base_ll"] - value["cand_ll"]) / value["base_ll"],
        "log_loss_difference_vs_baseline": value["cand_ll"] - value["base_ll"],
        "baseline_brier": value["base_br"],
        "brier": value["cand_br"],
        "brier_difference_vs_baseline": value["cand_br"] - value["base_br"],
    }
    if not p75_only:
        result.update(
            {
                "raw_log_loss": value["raw_ll"],
                "log_loss_difference_vs_raw": value["cand_ll"] - value["raw_ll"],
                "raw_brier": value["raw_br"],
                "brier_difference_vs_raw": value["cand_br"] - value["raw_br"],
            }
        )
    return result


def row_difference(
    frame: pd.DataFrame, algorithm: str, comparison: str, endpoint: str, p75_only=False
) -> np.ndarray:
    tiers = ("p75",) if p75_only else TIERS
    result = np.zeros(len(frame))
    denominator = 6 if p75_only else 12
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in tiers:
                y = frame[ycol(target, horizon, tier)].to_numpy(int)
                ref = "baseline" if comparison == "baseline" else "raw_reference"
                ref_loss = losses(y, probability(frame, ref, target, horizon, tier))[0 if endpoint == "log_loss" else 1]
                candidate = losses(y, probability(frame, algorithm, target, horizon, tier))[0 if endpoint == "log_loss" else 1]
                result += (candidate - ref_loss) / denominator
    return result


def daily(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    grouped = pd.DataFrame(
        {"date": frame["session_date"], "weighted": values * weights, "weight": weights}
    ).groupby("date", sort=True).sum()
    return (grouped["weighted"] / grouped["weight"]).to_numpy(float)


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = np.asarray(
        [values[index : index + 5].mean() for index in range(0, len(values), 5) if len(values[index : index + 5]) == 5]
    )
    sampled = np.random.default_rng(seed).choice(
        blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip(values: np.ndarray, seed: int) -> float:
    null = (
        np.random.default_rng(seed).choice(
            np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values))
        )
        @ values
    ) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def holm(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p"] = 1.0
    output["holm_pass"] = False
    families = {"all": output.index} if not groups else output.groupby(list(groups)).groups
    for _, positions in families.items():
        ordered = sorted(list(positions), key=lambda position: output.loc[position, "p_value"])
        running = 0.0
        for rank, position in enumerate(ordered, start=1):
            adjusted = min(1.0, max(running, (len(ordered) - rank + 1) * output.loc[position, "p_value"]))
            running = adjusted
            output.loc[position, "holm_adjusted_p"] = adjusted
            output.loc[position, "holm_pass"] = adjusted <= 0.05
    return output


def recall(frame: pd.DataFrame, model: str, target: str, horizon: int, tier: str) -> float:
    selected = frame[["anchor_id", ycol(target, horizon, tier)]].copy()
    selected["p"] = probability(frame, model, target, horizon, tier)
    selected = selected.sort_values(["anchor_id", "p"], ascending=[True, False], kind="stable")
    selected["rank"] = selected.groupby("anchor_id", sort=False).cumcount() + 1
    y = selected[ycol(target, horizon, tier)].to_numpy(int)
    return float(((selected["rank"].to_numpy(int) <= 3) & (y == 1)).sum() / y.sum())


def compare_table(
    calculated: pd.DataFrame,
    stored: pd.DataFrame,
    keys: Sequence[str],
    numeric: Sequence[str],
    tolerance=2e-11,
) -> tuple[bool, float, int]:
    joined = calculated.merge(stored, on=list(keys), suffixes=("__c", "__s"), validate="one_to_one")
    if len(joined) != len(calculated) or len(joined) != len(stored):
        return False, math.inf, 1
    maximum = 0.0
    errors = 0
    for column in numeric:
        left = joined[f"{column}__c"].to_numpy(float)
        right = joined[f"{column}__s"].to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        same_infinite = np.isinf(left) & np.isinf(right) & (np.sign(left) == np.sign(right))
        errors += int((~(finite | same_infinite)).sum())
        if finite.any():
            difference = np.abs(left[finite] - right[finite])
            maximum = max(maximum, float(difference.max()))
            errors += int((difference > tolerance).sum())
    return errors == 0, maximum, errors


def run_audit() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    hashes = {
        "contract": sha256(CONTRACT),
        "runner": sha256(RUNNER),
        "source": sha256(SOURCE),
        "source_audit": sha256(SOURCE_AUDIT),
    }
    record("frozen_hashes", hashes == EXPECTED, hashes)
    contract = json.loads(CONTRACT.read_text())
    record(
        "safety_labels",
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled",
        {key: contract[key] for key in ("research_only", "live_ordering_enabled", "order_placement")},
    )
    artifact = json.loads((ROOT / "artifact_manifest.json").read_text())["files"]
    mismatches = {
        name: sha256(ROOT / name)
        for name, descriptor in artifact.items()
        if sha256(ROOT / name) != descriptor["sha256"]
    }
    record("artifact_hashes", not mismatches, mismatches)
    frame, orientations, clocks = load_source()
    stored_orientations = pd.read_csv(ROOT / "orientation_dictionary.csv")
    stored_clocks = pd.read_csv(ROOT / "orientation_clock_dictionary.csv")
    orientations_equal = (
        orientations.columns.tolist() == stored_orientations.columns.tolist()
        and orientations.to_dict("records") == stored_orientations.to_dict("records")
    )
    clocks_equal = (
        clocks.columns.tolist() == stored_clocks.columns.tolist()
        and clocks.to_dict("records") == stored_clocks.to_dict("records")
    )
    record(
        "dictionary_replay",
        orientations_equal
        and clocks_equal
        and len(orientations) == 44
        and len(clocks) == 132,
        {"orientations": len(orientations), "clocks": len(clocks)},
    )
    ledger = pd.read_parquet(ROOT / "algorithm_predictions_2024_nov_dec.parquet")
    primary = frame.loc[frame["month"].isin(VALIDATION_MONTHS)].reset_index(drop=True)
    key_equal = primary[["anchor_id", "cycle_index"]].equals(
        ledger[["anchor_id", "cycle_index"]]
    )
    record(
        "primary_surface",
        key_equal and len(ledger) == 51235 and ledger["anchor_id"].nunique() == 9229,
        {"keys": key_equal, "rows": len(ledger), "anchors": ledger["anchor_id"].nunique()},
    )
    source_probability_columns = [
        *[ycol(target, horizon, tier) for target in TARGETS for horizon in HORIZONS for tier in TIERS],
        *[source_p(model, target, horizon, tier) for model in ("baseline", "raw_full_link") for target in TARGETS for horizon in HORIZONS for tier in TIERS],
    ]
    source_error = max(
        float(np.max(np.abs(primary[column].to_numpy(float) - ledger[column].to_numpy(float))))
        for column in source_probability_columns
    )
    record("source_column_replay", source_error == 0, source_error)
    prediction_error, parameter_error, fit_count = refit(
        frame, ledger, len(orientations), len(clocks)
    )
    record("algorithm_probability_refit", prediction_error < 2e-12, prediction_error)
    record("algorithm_parameter_refit", parameter_error < 2e-12 and fit_count == 96, {"error": parameter_error, "fits": fit_count})

    calculated_cells = cell_table(ledger)
    stored_cells = pd.read_csv(ROOT / "cell_metrics.csv")
    cell_numeric = [column for column in calculated_cells.columns if column not in {"weight_surface", "algorithm", "target", "horizon", "tier"}]
    passed, maximum, errors = compare_table(
        calculated_cells,
        stored_cells,
        ["weight_surface", "algorithm", "target", "horizon", "tier"],
        cell_numeric,
    )
    record("cell_metric_replay", passed, {"max": maximum, "errors": errors})

    pooled_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        row = {"algorithm": algorithm, "rows": len(ledger), "anchors": ledger["anchor_id"].nunique(), **pooled(ledger, algorithm)}
        for comparison_index, comparison in enumerate(COMPARISONS):
            for endpoint_index, endpoint in enumerate(("log_loss", "brier")):
                values = daily(ledger, row_difference(ledger, algorithm, comparison, endpoint))
                seed = SEED + algorithm_index * 1000 + comparison_index * 100 + endpoint_index
                lower, upper = bootstrap(values, seed)
                p_value = sign_flip(values, seed + 10)
                row[f"{comparison}__{endpoint}__daily_sessions"] = len(values)
                row[f"{comparison}__{endpoint}__bootstrap_lower"] = lower
                row[f"{comparison}__{endpoint}__bootstrap_upper"] = upper
                row[f"{comparison}__{endpoint}__p_value"] = p_value
                multiplicity_rows.append({"algorithm": algorithm, "comparison": comparison, "endpoint": endpoint, "p_value": p_value})
        pooled_rows.append(row)
    calculated_pooled = pd.DataFrame(pooled_rows)
    stored_pooled = pd.read_csv(ROOT / "pooled_metrics.csv")
    passed, maximum, errors = compare_table(
        calculated_pooled,
        stored_pooled,
        ["algorithm"],
        [column for column in calculated_pooled.columns if column != "algorithm"],
    )
    record("pooled_bootstrap_replay", passed, {"max": maximum, "errors": errors})
    calculated_holm = holm(pd.DataFrame(multiplicity_rows), ["comparison", "endpoint"])
    stored_holm = pd.read_csv(ROOT / "multiplicity.csv")
    passed, maximum, errors = compare_table(
        calculated_holm,
        stored_holm,
        ["algorithm", "comparison", "endpoint"],
        ["p_value", "holm_adjusted_p"],
    )
    bool_holm = calculated_holm.sort_values(["algorithm", "comparison", "endpoint"])["holm_pass"].tolist() == stored_holm.sort_values(["algorithm", "comparison", "endpoint"])["holm_pass"].tolist()
    record("Holm_replay", passed and bool_holm, {"max": maximum, "errors": errors, "bool": bool_holm})

    temporal_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for month in VALIDATION_MONTHS:
            temporal_rows.append({"algorithm": algorithm, "month": month, **pooled(ledger.loc[ledger["month"].eq(month)], algorithm)})
        for symbol in sorted(ledger["symbol_norm"].unique()):
            stock_rows.append({"algorithm": algorithm, "deleted_symbol": symbol, **pooled(ledger.loc[ledger["symbol_norm"].ne(symbol)], algorithm)})
        for (cycle, state), selected in ledger.groupby(["cycle_id", "current_state"], sort=True):
            positives = sum(int(selected[ycol(target, horizon, tier)].sum()) for target in TARGETS for horizon in HORIZONS for tier in TIERS)
            orientation_rows.append(
                {"algorithm": algorithm, "cycle_id": cycle, "current_state": int(state), "rows": len(selected), "joint_positives_across_cells": positives, "supported": len(selected) >= 250 and positives >= 15, **pooled(selected, algorithm)}
            )
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    raw = recall(ledger, "raw_reference", target, horizon, tier)
                    value = recall(ledger, algorithm, target, horizon, tier)
                    ranking_rows.append({"algorithm": algorithm, "target": target, "horizon": horizon, "tier": tier, "top_three_recall": value, "raw_top_three_recall": raw, "gain_vs_raw": value - raw})
    tables = [
        ("temporal_replay", pd.DataFrame(temporal_rows), pd.read_csv(ROOT / "temporal_slices.csv"), ["algorithm", "month"]),
        ("stock_replay", pd.DataFrame(stock_rows), pd.read_csv(ROOT / "stock_deletions.csv"), ["algorithm", "deleted_symbol"]),
        ("orientation_replay", pd.DataFrame(orientation_rows), pd.read_csv(ROOT / "orientation_slices.csv"), ["algorithm", "cycle_id", "current_state"]),
        ("ranking_replay", pd.DataFrame(ranking_rows), pd.read_csv(ROOT / "ranking.csv"), ["algorithm", "target", "horizon", "tier"]),
    ]
    calculated_tables: dict[str, pd.DataFrame] = {}
    for name, calculated, stored, keys in tables:
        calculated_tables[name] = calculated
        numeric = [column for column in calculated.columns if column not in set(keys) | {"supported"}]
        passed, maximum, errors = compare_table(calculated, stored, keys, numeric)
        bool_pass = True
        if "supported" in calculated:
            bool_pass = calculated.sort_values(keys)["supported"].tolist() == stored.sort_values(keys)["supported"].tolist()
        record(name, passed and bool_pass, {"max": maximum, "errors": errors, "bool": bool_pass})

    gates = json.loads((ROOT / "algorithm_gates.json").read_text())
    primary_cells = calculated_cells.loc[calculated_cells["weight_surface"].eq("inverse_compatible")]
    temporal = pd.DataFrame(temporal_rows)
    stocks = pd.DataFrame(stock_rows)
    orientations_frame = pd.DataFrame(orientation_rows)
    ranking = pd.DataFrame(ranking_rows)
    gate_errors: list[str] = []
    for algorithm in ALGORITHMS:
        pool = calculated_pooled.set_index("algorithm").loc[algorithm]
        cell = primary_cells.loc[primary_cells["algorithm"].eq(algorithm)]
        time = temporal.loc[temporal["algorithm"].eq(algorithm)]
        stock = stocks.loc[stocks["algorithm"].eq(algorithm)]
        orientation = orientations_frame.loc[
            orientations_frame["algorithm"].eq(algorithm) & orientations_frame["supported"]
        ]
        rank = ranking.loc[ranking["algorithm"].eq(algorithm)]
        multi = calculated_holm.loc[calculated_holm["algorithm"].eq(algorithm)]
        checks_here = {
            "pooled_gain_vs_baseline": pool["relative_log_loss_improvement_vs_baseline"] >= 0.02,
            "pooled_no_worse_than_raw": pool["log_loss_difference_vs_raw"] <= 0 and pool["brier_difference_vs_raw"] <= 0,
            "bootstrap_vs_baseline": all(pool[f"baseline__{endpoint}__bootstrap_upper"] <= 0 for endpoint in ("log_loss", "brier")),
            "bootstrap_vs_raw": all(pool[f"raw_reference__{endpoint}__bootstrap_upper"] <= 0 for endpoint in ("log_loss", "brier")),
            "Holm_all_four": bool(multi["holm_pass"].all()),
            "all_cell_losses_vs_baseline": bool((cell["log_loss_difference_vs_baseline"] <= 0).all() and (cell["brier_difference_vs_baseline"] <= 0).all()),
            "all_cell_calibration": bool((cell["ece"] <= cell["raw_ece"]).all() and (cell["maximum_supported_bin_error"] <= 0.02).all()),
            "both_months_vs_baseline_and_raw": bool((time["log_loss_difference_vs_baseline"] <= 0).all() and (time["brier_difference_vs_baseline"] <= 0).all() and (time["log_loss_difference_vs_raw"] <= 0).all() and (time["brier_difference_vs_raw"] <= 0).all()),
            "every_stock_vs_baseline": bool((stock["log_loss_difference_vs_baseline"] <= 0).all() and (stock["brier_difference_vs_baseline"] <= 0).all()),
            "zero_orientation_reversals": bool((orientation["log_loss_difference_vs_baseline"] <= 0).all() and (orientation["brier_difference_vs_baseline"] <= 0).all()),
            "ranking_vs_raw": bool((rank["gain_vs_raw"] >= 0).all()),
        }
        if checks_here != gates[algorithm]["checks"] or bool(all(checks_here.values())) != gates[algorithm]["pass"]:
            gate_errors.append(algorithm)
    record("algorithm_gate_replay", not gate_errors, gate_errors)

    stored_time = pd.read_csv(ROOT / "time_attraction_slices.csv")
    time_errors = 0
    time_maximum = 0.0
    slice_index = {
        (cycle, int(state), clock): index
        for index, ((cycle, state, clock), _) in enumerate(
            ledger.groupby(["cycle_id", "current_state", "entry_clock_quartile"], sort=True)
        )
    }
    recalculated_p: list[dict[str, Any]] = []
    for row in stored_time.itertuples(index=False):
        selected = ledger.loc[
            ledger["cycle_id"].eq(row.cycle_id)
            & ledger["current_state"].eq(row.current_state)
            & ledger["entry_clock_quartile"].eq(row.entry_clock_quartile)
        ]
        metrics = pooled(selected, "orientation_clock_residual", p75_only=True)
        for column, value in metrics.items():
            difference = abs(float(getattr(row, column)) - float(value))
            time_maximum = max(time_maximum, difference)
            time_errors += int(difference > 2e-11)
        p_value = sign_flip(
            daily(
                selected,
                row_difference(
                    selected,
                    "orientation_clock_residual",
                    "baseline",
                    "log_loss",
                    p75_only=True,
                ),
            ),
            SEED + slice_index[(row.cycle_id, int(row.current_state), row.entry_clock_quartile)],
        )
        difference = abs(float(row.p_value) - p_value)
        time_maximum = max(time_maximum, difference)
        time_errors += int(difference > 2e-11)
        recalculated_p.append({"cycle_id": row.cycle_id, "current_state": int(row.current_state), "entry_clock_quartile": row.entry_clock_quartile, "p_value": p_value})
    recalculated_holm = holm(pd.DataFrame(recalculated_p), [])
    passed, maximum, errors = compare_table(
        recalculated_holm,
        stored_time[["cycle_id", "current_state", "entry_clock_quartile", "p_value", "holm_adjusted_p"]],
        ["cycle_id", "current_state", "entry_clock_quartile"],
        ["p_value", "holm_adjusted_p"],
    )
    record("time_slice_replay", time_errors == 0 and passed, {"max": max(time_maximum, maximum), "errors": time_errors + errors})

    decision = json.loads((ROOT / "decision.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    record(
        "decision_replay",
        all(not gates[algorithm]["pass"] for algorithm in ALGORITHMS)
        and decision["passing_algorithms"] == []
        and decision["selected_algorithm"] is None
        and decision["qualified_time_slices"] == []
        and decision["named_loop_good_or_high_promoted"] is False,
        decision,
    )
    record(
        "summary_reconciliation",
        summary["decision"] == decision
        and summary["validation_rows"] == len(ledger)
        and summary["fit_count"] == 96
        and summary["direct_volume_fields_used"] == []
        and summary["direction_or_signed_return_used"] is False
        and summary["later_period_scoring_performed"] is False,
        {"rows": summary["validation_rows"], "fits": summary["fit_count"]},
    )
    all_passed = all(value["passed"] for value in checks.values())
    result = {
        "audit_id": "regime_loop_orientation_calibration_algorithms_v1_independent_audit",
        "all_passed": all_passed,
        "checks_passed": sum(value["passed"] for value in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "auditor_imported_runner": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    (ROOT / "independent_audit.json").write_text(
        json.dumps(safe(result), indent=2, sort_keys=True) + "\n"
    )
    if not all_passed:
        raise AssertionError(
            "orientation algorithm audit failed: "
            + str([name for name, value in checks.items() if not value["passed"]])
        )
    print(json.dumps(safe(result), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_audit()
