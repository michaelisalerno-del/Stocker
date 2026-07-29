"""Causal comparison of regime-loop calibration/shrinkage algorithms.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = (
    HERE
    / "contracts/20260711-regime-loop-orientation-calibration-algorithms-v1.json"
)
CONTRACT_SHA256 = "900f14c8c43456a28e3532be1cc499fe61d9b4b26b0f0904d0afafa1c7ad525d"
SOURCE_ROOT = Path("/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711")
SOURCE = SOURCE_ROOT / "linkage_predictions_2024_sep_dec.parquet"
SOURCE_AUDIT = SOURCE_ROOT / "independent_audit.json"
SOURCE_CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v3.json"
SOURCE_RUNNER = HERE / "run_regime_loop_linkage_ideas_v3.py"
OUT = Path(
    "/private/tmp/stocker_regime_loop_orientation_calibration_algorithms_v1_20260711"
)

EXPECTED_HASHES = {
    "source": "99374428d372711b233cf6dfbe59a18f5667e032ef3039b2ae05df13400cd660",
    "source_audit": "92343c008f0cd585c3e02c1e3c60905aa4f9cde7e9c9b99111076a2cd8be300f",
    "source_contract": "88a60956857e6ccb4fb5e74beb9085e46765e55b31763b26927dc496822ce947",
    "source_runner": "c0e8786670fd51e3d93290ecd56ba51322ebe6ace0fd7e521803f2fd8c1ce72e",
}

SOURCE_MONTHS = ("2024-09", "2024-10", "2024-11", "2024-12")
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def load_contract() -> dict[str, Any]:
    observed = sha256(CONTRACT)
    if observed != CONTRACT_SHA256:
        raise AssertionError(f"orientation algorithm contract changed: {observed}")
    contract = json.loads(CONTRACT.read_text())
    checks = {
        "id": contract.get("contract_id")
        == "regime_loop_orientation_calibration_algorithms_v1",
        "research": contract.get("research_only") is True,
        "live": contract.get("live_ordering_enabled") is False,
        "orders": contract.get("order_placement") == "disabled",
        "later": contract["population_and_causality"].get("later_period_paths_permitted")
        is False,
        "shadow": contract["population_and_causality"].get(
            "prospective_shadow_read_or_write_permitted"
        )
        is False,
        "promotion": contract["decision"].get(
            "named_loop_good_or_high_promotion_permitted"
        )
        is False,
        "trading": contract["decision"].get("trading_rule_or_PnL_model_permitted")
        is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"orientation algorithm safety failure: {checks}")
    return contract


def verify_sources() -> dict[str, str]:
    observed = {
        "source": sha256(SOURCE),
        "source_audit": sha256(SOURCE_AUDIT),
        "source_contract": sha256(SOURCE_CONTRACT),
        "source_runner": sha256(SOURCE_RUNNER),
    }
    if observed != EXPECTED_HASHES:
        raise AssertionError(f"frozen linkage source drift: {observed}")
    audit = json.loads(SOURCE_AUDIT.read_text())
    if not (
        audit.get("all_passed") is True
        and audit.get("checks_passed") == 19
        and audit.get("checks_total") == 19
        and audit.get("research_only") is True
        and audit.get("live_ordering_enabled") is False
        and audit.get("order_placement") == "disabled"
    ):
        raise AssertionError("linkage source audit is not passing")
    return observed


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
        *[
            f"joint_target__{target}__h{horizon}__{tier}"
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
        *[
            f"link__{model}__{target}__h{horizon}__{tier}"
            for model in ("baseline", "raw_full_link")
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
    ]


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(SOURCE, columns=source_columns())
    if len(frame) != 130672 or frame.duplicated(["anchor_id", "cycle_index"]).any():
        raise AssertionError("linkage source row identity changed")
    frame["month"] = frame["month"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    if set(frame["month"].unique()) != set(SOURCE_MONTHS):
        raise AssertionError("linkage source month surface changed")
    orientation = (
        frame.loc[:, ["cycle_id", "current_state"]]
        .drop_duplicates()
        .sort_values(["cycle_id", "current_state"], kind="stable")
        .reset_index(drop=True)
    )
    orientation["orientation_index"] = np.arange(len(orientation), dtype=int)
    if len(orientation) != 44:
        raise AssertionError(f"orientation dictionary changed: {len(orientation)}")
    orientation_clock = (
        frame.loc[:, ["cycle_id", "current_state", "entry_clock_quartile"]]
        .drop_duplicates()
        .sort_values(
            ["cycle_id", "current_state", "entry_clock_quartile"], kind="stable"
        )
        .reset_index(drop=True)
    )
    orientation_clock["orientation_clock_index"] = np.arange(
        len(orientation_clock), dtype=int
    )
    frame = frame.merge(
        orientation,
        on=["cycle_id", "current_state"],
        how="left",
        validate="many_to_one",
    ).merge(
        orientation_clock,
        on=["cycle_id", "current_state", "entry_clock_quartile"],
        how="left",
        validate="many_to_one",
    )
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise AssertionError("invalid source weights")
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                target_values = frame[target_column(target, horizon, tier)].to_numpy(int)
                if not set(np.unique(target_values)).issubset({0, 1}):
                    raise AssertionError("invalid joint target")
                for model in ("baseline", "raw_full_link"):
                    probability = frame[
                        source_probability_column(model, target, horizon, tier)
                    ].to_numpy(float)
                    if (
                        not np.isfinite(probability).all()
                        or probability.min() < 0
                        or probability.max() > 1
                    ):
                        raise AssertionError("invalid source probability")
    audit = {
        "rows": len(frame),
        "anchors": int(frame["anchor_id"].nunique()),
        "sessions": int(frame["session_date"].nunique()),
        "stocks": int(frame["symbol_norm"].nunique()),
        "cycles": int(frame["cycle_id"].nunique()),
        "states": sorted(int(value) for value in frame["current_state"].unique()),
        "orientation_count": len(orientation),
        "orientation_clock_count": len(orientation_clock),
        "validation_rows": int(frame["month"].isin(VALIDATION_MONTHS).sum()),
        "validation_anchors": int(
            frame.loc[frame["month"].isin(VALIDATION_MONTHS), "anchor_id"].nunique()
        ),
        "validation_sessions": int(
            frame.loc[frame["month"].isin(VALIDATION_MONTHS), "session_date"].nunique()
        ),
    }
    support_checks = {
        "validation_rows": audit["validation_rows"] >= 50000,
        "validation_anchors": audit["validation_anchors"] >= 8000,
        "sessions": audit["validation_sessions"] >= 40,
        "stocks": audit["stocks"] == 22,
        "cycles": audit["cycles"] == 20,
        "states": audit["states"] == list(range(8)),
    }
    audit["support_checks"] = support_checks
    if not all(support_checks.values()):
        raise AssertionError(f"algorithm support failed: {support_checks}")
    return frame, orientation, orientation_clock, audit


def target_column(target: str, horizon: int, tier: str) -> str:
    return f"joint_target__{target}__h{horizon}__{tier}"


def source_probability_column(
    model: str, target: str, horizon: int, tier: str
) -> str:
    return f"link__{model}__{target}__h{horizon}__{tier}"


def algorithm_probability_column(
    algorithm: str, target: str, horizon: int, tier: str
) -> str:
    return f"algorithm__{algorithm}__{target}__h{horizon}__{tier}"


def clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1 - EPSILON)


def logit(values: np.ndarray) -> np.ndarray:
    values = clip_probability(values)
    return np.log(values / (1 - values))


def global_residual_features(
    frame: pd.DataFrame, target: str, horizon: int, tier: str
) -> tuple[np.ndarray, np.ndarray]:
    baseline = logit(
        frame[
            source_probability_column("baseline", target, horizon, tier)
        ].to_numpy(float)
    )
    raw = logit(
        frame[
            source_probability_column("raw_full_link", target, horizon, tier)
        ].to_numpy(float)
    )
    residual = raw - baseline
    return np.column_stack((baseline, residual)), residual


def categorical_matrix(indices: np.ndarray, width: int, values: np.ndarray) -> sparse.csr_matrix:
    indices = np.asarray(indices, dtype=int)
    return sparse.csr_matrix(
        (
            np.asarray(values, dtype=float),
            (np.arange(len(indices)), indices),
        ),
        shape=(len(indices), width),
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
    global_values, residual = global_residual_features(frame, target, horizon, tier)
    scaled_global = sparse.csr_matrix(scaler.transform(global_values))
    orientation_index = frame["orientation_index"].to_numpy(int)
    parts: list[sparse.csr_matrix] = [
        scaled_global,
        categorical_matrix(
            orientation_index,
            orientation_width,
            np.full(len(frame), 0.25),
        ),
        categorical_matrix(
            orientation_index,
            orientation_width,
            residual * 0.125,
        ),
    ]
    if clock_width is not None:
        clock_index = frame["orientation_clock_index"].to_numpy(int)
        parts.extend(
            [
                categorical_matrix(
                    clock_index,
                    clock_width,
                    np.full(len(frame), 0.125),
                ),
                categorical_matrix(
                    clock_index,
                    clock_width,
                    residual * 0.0625,
                ),
            ]
        )
    return sparse.hstack(parts, format="csr")


def fit_logistic(features, y: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    if not np.array_equal(np.unique(y), np.asarray([0, 1])):
        raise AssertionError("algorithm training target lacks a class")
    model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-10,
        random_state=SEED,
    ).fit(features, y, sample_weight=weights)
    if int(model.n_iter_[0]) >= 2000:
        raise AssertionError("orientation algorithm did not converge")
    return model


def fit_algorithms(
    frame: pd.DataFrame, orientation_width: int, clock_width: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    output = frame.copy()
    for algorithm in ALGORITHMS:
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    output[
                        algorithm_probability_column(
                            algorithm, target, horizon, tier
                        )
                    ] = np.nan
    audits: list[dict[str, Any]] = []
    parameters: dict[str, np.ndarray] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                y_name = target_column(target, horizon, tier)
                raw_name = source_probability_column(
                    "raw_full_link", target, horizon, tier
                )
                for month in VALIDATION_MONTHS:
                    training = output.loc[output["month"].lt(month)].copy()
                    validation_positions = output.index[output["month"].eq(month)]
                    validation = output.loc[validation_positions]
                    weights = training["inverse_compatible_weight"].to_numpy(float)
                    y = training[y_name].to_numpy(int)
                    raw_training = clip_probability(training[raw_name].to_numpy(float))
                    raw_validation = clip_probability(validation[raw_name].to_numpy(float))
                    prefix = f"{target}__h{horizon}__{tier}__{month}"

                    isotonic = IsotonicRegression(
                        y_min=EPSILON,
                        y_max=1 - EPSILON,
                        increasing=True,
                        out_of_bounds="clip",
                    ).fit(raw_training, y, sample_weight=weights)
                    probability = clip_probability(isotonic.predict(raw_validation))
                    output.loc[
                        validation_positions,
                        algorithm_probability_column(
                            "weighted_isotonic", target, horizon, tier
                        ),
                    ] = probability
                    parameters[f"{prefix}__weighted_isotonic__x"] = np.asarray(
                        isotonic.X_thresholds_, dtype=float
                    )
                    parameters[f"{prefix}__weighted_isotonic__y"] = np.asarray(
                        isotonic.y_thresholds_, dtype=float
                    )
                    audits.append(
                        fold_audit_row(
                            "weighted_isotonic",
                            target,
                            horizon,
                            tier,
                            month,
                            training,
                            validation,
                            y,
                            weights,
                            1,
                            len(isotonic.X_thresholds_),
                        )
                    )

                    beta_train = np.column_stack(
                        (np.log(raw_training), np.log1p(-raw_training))
                    )
                    beta_validation = np.column_stack(
                        (np.log(raw_validation), np.log1p(-raw_validation))
                    )
                    beta_scaler = StandardScaler().fit(
                        beta_train, sample_weight=weights
                    )
                    beta_model = fit_logistic(
                        beta_scaler.transform(beta_train), y, weights
                    )
                    probability = clip_probability(
                        beta_model.predict_proba(
                            beta_scaler.transform(beta_validation)
                        )[:, 1]
                    )
                    output.loc[
                        validation_positions,
                        algorithm_probability_column(
                            "beta_global", target, horizon, tier
                        ),
                    ] = probability
                    store_logistic_parameters(
                        parameters,
                        f"{prefix}__beta_global",
                        beta_scaler,
                        beta_model,
                    )
                    audits.append(
                        fold_audit_row(
                            "beta_global",
                            target,
                            horizon,
                            tier,
                            month,
                            training,
                            validation,
                            y,
                            weights,
                            2,
                            int(beta_model.n_iter_[0]),
                        )
                    )

                    global_train, _ = global_residual_features(
                        training, target, horizon, tier
                    )
                    global_scaler = StandardScaler().fit(
                        global_train, sample_weight=weights
                    )
                    for algorithm, included_clock_width in (
                        ("orientation_residual", None),
                        ("orientation_clock_residual", clock_width),
                    ):
                        train_x = orientation_features(
                            training,
                            target,
                            horizon,
                            tier,
                            global_scaler,
                            orientation_width,
                            included_clock_width,
                        )
                        validation_x = orientation_features(
                            validation,
                            target,
                            horizon,
                            tier,
                            global_scaler,
                            orientation_width,
                            included_clock_width,
                        )
                        model = fit_logistic(train_x, y, weights)
                        probability = clip_probability(
                            model.predict_proba(validation_x)[:, 1]
                        )
                        output.loc[
                            validation_positions,
                            algorithm_probability_column(
                                algorithm, target, horizon, tier
                            ),
                        ] = probability
                        store_logistic_parameters(
                            parameters,
                            f"{prefix}__{algorithm}",
                            global_scaler,
                            model,
                        )
                        audits.append(
                            fold_audit_row(
                                algorithm,
                                target,
                                horizon,
                                tier,
                                month,
                                training,
                                validation,
                                y,
                                weights,
                                train_x.shape[1],
                                int(model.n_iter_[0]),
                            )
                        )
    primary = output["month"].isin(VALIDATION_MONTHS)
    columns = [
        algorithm_probability_column(algorithm, target, horizon, tier)
        for algorithm in ALGORITHMS
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    values = output.loc[primary, columns].to_numpy(float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise AssertionError("incomplete or invalid algorithm probabilities")
    return output, pd.DataFrame(audits), parameters


def fold_audit_row(
    algorithm: str,
    target: str,
    horizon: int,
    tier: str,
    month: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    feature_width: int,
    iterations_or_thresholds: int,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "target": target,
        "horizon": horizon,
        "tier": tier,
        "validation_month": month,
        "training_months": json.dumps(sorted(training["month"].unique().tolist())),
        "training_rows": len(training),
        "training_weight": float(weights.sum()),
        "training_positives": int(y.sum()),
        "validation_rows": len(validation),
        "validation_positives": int(
            validation[target_column(target, horizon, tier)].sum()
        ),
        "feature_width": feature_width,
        "iterations_or_thresholds": iterations_or_thresholds,
    }


def store_logistic_parameters(
    output: dict[str, np.ndarray],
    prefix: str,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> None:
    output[f"{prefix}__mean"] = np.asarray(scaler.mean_, dtype=float)
    output[f"{prefix}__scale"] = np.asarray(scaler.scale_, dtype=float)
    output[f"{prefix}__coef"] = np.asarray(model.coef_, dtype=float)
    output[f"{prefix}__intercept"] = np.asarray(model.intercept_, dtype=float)
    output[f"{prefix}__n_iter"] = np.asarray(model.n_iter_, dtype=int)


def binary_losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), LOSS_EPSILON, 1 - LOSS_EPSILON)
    return (-(y * np.log(p) + (1 - y) * np.log(1 - p)), (y - p) ** 2)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(values) or weights.sum() <= 0:
        return math.nan
    return float(np.average(values, weights=weights))


def calibration_summary(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray, minimum_rows: int
) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    supported: list[tuple[float, float]] = []
    for index in range(10):
        selected = bins == index
        if int(selected.sum()) < minimum_rows or weights[selected].sum() <= 0:
            continue
        error = abs(
            weighted_mean(y[selected], weights[selected])
            - weighted_mean(p[selected], weights[selected])
        )
        supported.append((float(weights[selected].sum()), float(error)))
    if not supported:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in supported)
    return (
        float(sum(weight * error for weight, error in supported) / total),
        float(max(error for _, error in supported)),
        len(supported),
    )


def probability_for(
    frame: pd.DataFrame,
    model: str,
    target: str,
    horizon: int,
    tier: str,
) -> np.ndarray:
    if model in ALGORITHMS:
        column = algorithm_probability_column(model, target, horizon, tier)
    else:
        column = source_probability_column(
            "baseline" if model == "baseline" else "raw_full_link",
            target,
            horizon,
            tier,
        )
    return frame[column].to_numpy(float)


def cell_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface, weights in (
        ("inverse_compatible", frame["inverse_compatible_weight"].to_numpy(float)),
        ("unweighted", np.ones(len(frame))),
    ):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    y = frame[target_column(target, horizon, tier)].to_numpy(int)
                    references: dict[str, dict[str, float]] = {}
                    for reference in ("baseline", "raw_reference"):
                        p = probability_for(frame, reference, target, horizon, tier)
                        ll, brier = binary_losses(y, p)
                        ece, maximum, bins = calibration_summary(y, p, weights, 250)
                        references[reference] = {
                            "log_loss": weighted_mean(ll, weights),
                            "brier": weighted_mean(brier, weights),
                            "ece": ece,
                            "maximum": maximum,
                            "bins": bins,
                        }
                    for algorithm in ALGORITHMS:
                        p = probability_for(frame, algorithm, target, horizon, tier)
                        ll, brier = binary_losses(y, p)
                        ece, maximum, bins = calibration_summary(y, p, weights, 250)
                        ll_mean = weighted_mean(ll, weights)
                        brier_mean = weighted_mean(brier, weights)
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
                                "baseline_log_loss": references["baseline"]["log_loss"],
                                "raw_log_loss": references["raw_reference"]["log_loss"],
                                "relative_log_loss_improvement_vs_baseline": (
                                    references["baseline"]["log_loss"] - ll_mean
                                )
                                / references["baseline"]["log_loss"],
                                "log_loss_difference_vs_baseline": ll_mean
                                - references["baseline"]["log_loss"],
                                "log_loss_difference_vs_raw": ll_mean
                                - references["raw_reference"]["log_loss"],
                                "brier": brier_mean,
                                "baseline_brier": references["baseline"]["brier"],
                                "raw_brier": references["raw_reference"]["brier"],
                                "brier_difference_vs_baseline": brier_mean
                                - references["baseline"]["brier"],
                                "brier_difference_vs_raw": brier_mean
                                - references["raw_reference"]["brier"],
                                "ece": ece,
                                "raw_ece": references["raw_reference"]["ece"],
                                "maximum_supported_bin_error": maximum,
                                "raw_maximum_supported_bin_error": references[
                                    "raw_reference"
                                ]["maximum"],
                                "supported_bins": bins,
                            }
                        )
    return pd.DataFrame(rows)


def pooled_metrics(frame: pd.DataFrame, algorithm: str) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    storage: dict[str, list[float]] = {
        "baseline_ll": [],
        "raw_ll": [],
        "candidate_ll": [],
        "baseline_brier": [],
        "raw_brier": [],
        "candidate_brier": [],
    }
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                y = frame[target_column(target, horizon, tier)].to_numpy(int)
                for model, key in (
                    ("baseline", "baseline"),
                    ("raw_reference", "raw"),
                    (algorithm, "candidate"),
                ):
                    ll, brier = binary_losses(
                        y, probability_for(frame, model, target, horizon, tier)
                    )
                    storage[f"{key}_ll"].append(weighted_mean(ll, weights))
                    storage[f"{key}_brier"].append(weighted_mean(brier, weights))
    values = {key: float(np.mean(item)) for key, item in storage.items()}
    return {
        "baseline_log_loss": values["baseline_ll"],
        "raw_log_loss": values["raw_ll"],
        "log_loss": values["candidate_ll"],
        "relative_log_loss_improvement_vs_baseline": (
            values["baseline_ll"] - values["candidate_ll"]
        )
        / values["baseline_ll"],
        "log_loss_difference_vs_baseline": values["candidate_ll"]
        - values["baseline_ll"],
        "log_loss_difference_vs_raw": values["candidate_ll"] - values["raw_ll"],
        "baseline_brier": values["baseline_brier"],
        "raw_brier": values["raw_brier"],
        "brier": values["candidate_brier"],
        "brier_difference_vs_baseline": values["candidate_brier"]
        - values["baseline_brier"],
        "brier_difference_vs_raw": values["candidate_brier"]
        - values["raw_brier"],
    }


def row_loss_difference(
    frame: pd.DataFrame, algorithm: str, comparison: str, endpoint: str, p75_only=False
) -> np.ndarray:
    tiers = ("p75",) if p75_only else TIERS
    denominator = 6.0 if p75_only else 12.0
    result = np.zeros(len(frame), dtype=float)
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in tiers:
                y = frame[target_column(target, horizon, tier)].to_numpy(int)
                reference = "baseline" if comparison == "baseline" else "raw_reference"
                reference_loss = binary_losses(
                    y, probability_for(frame, reference, target, horizon, tier)
                )[0 if endpoint == "log_loss" else 1]
                candidate_loss = binary_losses(
                    y, probability_for(frame, algorithm, target, horizon, tier)
                )[0 if endpoint == "log_loss" else 1]
                result += (candidate_loss - reference_loss) / denominator
    return result


def daily_values(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    daily = pd.DataFrame(
        {
            "date": frame["session_date"].astype(str).to_numpy(),
            "weighted": values * weights,
            "weight": weights,
        }
    ).groupby("date", sort=True).sum()
    return (daily["weighted"] / daily["weight"]).to_numpy(float)


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = np.asarray(
        [
            values[index : index + 5].mean()
            for index in range(0, len(values), 5)
            if len(values[index : index + 5]) == 5
        ]
    )
    if len(blocks) < 5:
        return math.nan, math.nan
    sampled = np.random.default_rng(seed).choice(
        blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip_p_value(values: np.ndarray, seed: int) -> float:
    if len(values) < 10:
        return math.nan
    null = (
        np.random.default_rng(seed)
        .choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values)))
        @ values
    ) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def holm_adjust(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p"] = 1.0
    output["holm_pass"] = False
    output["holm_rank"] = 0
    output["family_size"] = 0
    families = {"all": output.index} if not groups else output.groupby(list(groups)).groups
    for _, positions in families.items():
        ordered = sorted(list(positions), key=lambda position: output.loc[position, "p_value"])
        running = 0.0
        size = len(ordered)
        for rank, position in enumerate(ordered, start=1):
            adjusted = min(
                1.0,
                max(running, (size - rank + 1) * float(output.loc[position, "p_value"])),
            )
            running = adjusted
            output.loc[position, "holm_adjusted_p"] = adjusted
            output.loc[position, "holm_pass"] = adjusted <= 0.05
            output.loc[position, "holm_rank"] = rank
            output.loc[position, "family_size"] = size
    return output


def top_three_recall(
    frame: pd.DataFrame, model: str, target: str, horizon: int, tier: str
) -> float:
    selected = frame.loc[:, ["anchor_id", target_column(target, horizon, tier)]].copy()
    selected["probability"] = probability_for(
        frame, model, target, horizon, tier
    )
    selected = selected.sort_values(
        ["anchor_id", "probability"], ascending=[True, False], kind="stable"
    )
    selected["rank"] = selected.groupby("anchor_id", sort=False).cumcount() + 1
    y = selected[target_column(target, horizon, tier)].to_numpy(int)
    return float(((selected["rank"].to_numpy(int) <= 3) & (y == 1)).sum() / y.sum())


def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    primary = frame.loc[frame["month"].isin(VALIDATION_MONTHS)].reset_index(drop=True)
    cells = cell_metrics(primary)
    pooled_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        pooled = {
            "algorithm": algorithm,
            "rows": len(primary),
            "anchors": int(primary["anchor_id"].nunique()),
            **pooled_metrics(primary, algorithm),
        }
        for comparison_index, comparison in enumerate(COMPARISONS):
            for endpoint_index, endpoint in enumerate(("log_loss", "brier")):
                daily = daily_values(
                    primary,
                    row_loss_difference(
                        primary, algorithm, comparison, endpoint
                    ),
                )
                seed = (
                    SEED
                    + algorithm_index * 1000
                    + comparison_index * 100
                    + endpoint_index
                )
                lower, upper = bootstrap_interval(daily, seed)
                p_value = sign_flip_p_value(daily, seed + 10)
                pooled[f"{comparison}__{endpoint}__daily_sessions"] = len(daily)
                pooled[f"{comparison}__{endpoint}__bootstrap_lower"] = lower
                pooled[f"{comparison}__{endpoint}__bootstrap_upper"] = upper
                pooled[f"{comparison}__{endpoint}__p_value"] = p_value
                multiplicity_rows.append(
                    {
                        "algorithm": algorithm,
                        "comparison": comparison,
                        "endpoint": endpoint,
                        "p_value": p_value,
                    }
                )
        pooled_rows.append(pooled)
        for month in VALIDATION_MONTHS:
            selected = primary.loc[primary["month"].eq(month)]
            temporal_rows.append(
                {"algorithm": algorithm, "month": month, **pooled_metrics(selected, algorithm)}
            )
        for symbol in sorted(primary["symbol_norm"].unique()):
            selected = primary.loc[primary["symbol_norm"].ne(symbol)]
            stock_rows.append(
                {
                    "algorithm": algorithm,
                    "deleted_symbol": symbol,
                    **pooled_metrics(selected, algorithm),
                }
            )
        for (cycle_id, current_state), selected in primary.groupby(
            ["cycle_id", "current_state"], sort=True
        ):
            positives = sum(
                int(selected[target_column(target, horizon, tier)].sum())
                for target in TARGETS
                for horizon in HORIZONS
                for tier in TIERS
            )
            orientation_rows.append(
                {
                    "algorithm": algorithm,
                    "cycle_id": cycle_id,
                    "current_state": int(current_state),
                    "rows": len(selected),
                    "joint_positives_across_cells": positives,
                    "supported": len(selected) >= 250 and positives >= 15,
                    **pooled_metrics(selected, algorithm),
                }
            )
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    raw = top_three_recall(
                        primary, "raw_reference", target, horizon, tier
                    )
                    value = top_three_recall(
                        primary, algorithm, target, horizon, tier
                    )
                    ranking_rows.append(
                        {
                            "algorithm": algorithm,
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            "top_three_recall": value,
                            "raw_top_three_recall": raw,
                            "gain_vs_raw": value - raw,
                        }
                    )
    pooled = pd.DataFrame(pooled_rows)
    multiplicity = holm_adjust(
        pd.DataFrame(multiplicity_rows), ["comparison", "endpoint"]
    )
    temporal = pd.DataFrame(temporal_rows)
    stocks = pd.DataFrame(stock_rows)
    orientations = pd.DataFrame(orientation_rows)
    ranking = pd.DataFrame(ranking_rows)
    primary_cells = cells.loc[cells["weight_surface"].eq("inverse_compatible")]
    gates: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        pooled_row = pooled.set_index("algorithm").loc[algorithm]
        cell = primary_cells.loc[primary_cells["algorithm"].eq(algorithm)]
        time = temporal.loc[temporal["algorithm"].eq(algorithm)]
        stock = stocks.loc[stocks["algorithm"].eq(algorithm)]
        orientation = orientations.loc[
            orientations["algorithm"].eq(algorithm) & orientations["supported"]
        ]
        rank = ranking.loc[ranking["algorithm"].eq(algorithm)]
        holm = multiplicity.loc[multiplicity["algorithm"].eq(algorithm)]
        checks = {
            "pooled_gain_vs_baseline": pooled_row[
                "relative_log_loss_improvement_vs_baseline"
            ]
            >= 0.02,
            "pooled_no_worse_than_raw": pooled_row["log_loss_difference_vs_raw"]
            <= 0
            and pooled_row["brier_difference_vs_raw"] <= 0,
            "bootstrap_vs_baseline": all(
                pooled_row[f"baseline__{endpoint}__bootstrap_upper"] <= 0
                for endpoint in ("log_loss", "brier")
            ),
            "bootstrap_vs_raw": all(
                pooled_row[f"raw_reference__{endpoint}__bootstrap_upper"] <= 0
                for endpoint in ("log_loss", "brier")
            ),
            "Holm_all_four": bool(holm["holm_pass"].all()),
            "all_cell_losses_vs_baseline": bool(
                (cell["log_loss_difference_vs_baseline"] <= 0).all()
                and (cell["brier_difference_vs_baseline"] <= 0).all()
            ),
            "all_cell_calibration": bool(
                (cell["ece"] <= cell["raw_ece"]).all()
                and (cell["maximum_supported_bin_error"] <= 0.02).all()
            ),
            "both_months_vs_baseline_and_raw": bool(
                (time["log_loss_difference_vs_baseline"] <= 0).all()
                and (time["brier_difference_vs_baseline"] <= 0).all()
                and (time["log_loss_difference_vs_raw"] <= 0).all()
                and (time["brier_difference_vs_raw"] <= 0).all()
            ),
            "every_stock_vs_baseline": bool(
                (stock["log_loss_difference_vs_baseline"] <= 0).all()
                and (stock["brier_difference_vs_baseline"] <= 0).all()
            ),
            "zero_orientation_reversals": bool(
                len(orientation) > 0
                and (orientation["log_loss_difference_vs_baseline"] <= 0).all()
                and (orientation["brier_difference_vs_baseline"] <= 0).all()
            ),
            "ranking_vs_raw": bool((rank["gain_vs_raw"] >= 0).all()),
        }
        gates[algorithm] = {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "supported_orientations": len(orientation),
            "orientation_log_loss_reversals": int(
                (orientation["log_loss_difference_vs_baseline"] > 0).sum()
            ),
            "orientation_brier_reversals": int(
                (orientation["brier_difference_vs_baseline"] > 0).sum()
            ),
            "calibration_failures": int(
                (
                    (cell["ece"] > cell["raw_ece"])
                    | (cell["maximum_supported_bin_error"] > 0.02)
                ).sum()
            ),
            "cell_loss_reversals_vs_baseline": int(
                (
                    (cell["log_loss_difference_vs_baseline"] > 0)
                    | (cell["brier_difference_vs_baseline"] > 0)
                ).sum()
            ),
            "ranking_degradations_vs_raw": int((rank["gain_vs_raw"] < 0).sum()),
        }
    return {
        "primary": primary,
        "cells": cells,
        "pooled": pooled,
        "multiplicity": multiplicity,
        "temporal": temporal,
        "stocks": stocks,
        "orientations": orientations,
        "ranking": ranking,
        "gates": gates,
    }


def evaluate_time_slices(primary: pd.DataFrame, global_pass: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slice_index, ((cycle, state, clock), selected) in enumerate(
        primary.groupby(
            ["cycle_id", "current_state", "entry_clock_quartile"], sort=True
        )
    ):
        positives = sum(
            int(selected[target_column(target, horizon, "p75")].sum())
            for target in TARGETS
            for horizon in HORIZONS
        )
        if not (
            len(selected) >= 250
            and positives >= 25
            and selected["symbol_norm"].nunique() >= 10
        ):
            continue
        algorithm = "orientation_clock_residual"
        cell_passes: list[bool] = []
        maximum_error = 0.0
        for target in TARGETS:
            for horizon in HORIZONS:
                y = selected[target_column(target, horizon, "p75")].to_numpy(int)
                weights = selected["inverse_compatible_weight"].to_numpy(float)
                baseline_ll, baseline_brier = binary_losses(
                    y,
                    probability_for(
                        selected, "baseline", target, horizon, "p75"
                    ),
                )
                candidate_probability = probability_for(
                    selected, algorithm, target, horizon, "p75"
                )
                candidate_ll, candidate_brier = binary_losses(y, candidate_probability)
                cell_passes.append(
                    weighted_mean(candidate_ll - baseline_ll, weights) <= 0
                    and weighted_mean(candidate_brier - baseline_brier, weights) <= 0
                )
                _, error, _ = calibration_summary(
                    y, candidate_probability, weights, 100
                )
                maximum_error = max(maximum_error, error)
        month_passes: list[bool] = []
        for month in VALIDATION_MONTHS:
            month_frame = selected.loc[selected["month"].eq(month)]
            metrics = pooled_p75(month_frame, algorithm)
            month_passes.append(
                metrics["log_loss_difference_vs_baseline"] <= 0
                and metrics["brier_difference_vs_baseline"] <= 0
            )
        metrics = pooled_p75(selected, algorithm)
        daily = daily_values(
            selected,
            row_loss_difference(
                selected, algorithm, "baseline", "log_loss", p75_only=True
            ),
        )
        p_value = sign_flip_p_value(daily, SEED + slice_index)
        checks = {
            "global_algorithm": global_pass,
            "all_six_cells": all(cell_passes),
            "both_months": all(month_passes),
            "calibration": maximum_error <= 0.03,
        }
        rows.append(
            {
                "cycle_id": cycle,
                "current_state": int(state),
                "entry_clock_quartile": clock,
                "rows": len(selected),
                "joint_positives_across_p75_cells": positives,
                "stocks": int(selected["symbol_norm"].nunique()),
                **metrics,
                "maximum_supported_bin_error": maximum_error,
                "daily_sessions": len(daily),
                "p_value": p_value,
                "checks_before_Holm": json.dumps(checks, sort_keys=True),
                "pass_before_Holm": bool(all(checks.values())),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = holm_adjust(frame, [])
    frame["qualified_development_time_slice"] = (
        frame["pass_before_Holm"] & frame["holm_pass"]
    )
    return frame


def pooled_p75(frame: pd.DataFrame, algorithm: str) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    storage = {
        "baseline_ll": [],
        "candidate_ll": [],
        "baseline_brier": [],
        "candidate_brier": [],
    }
    for target in TARGETS:
        for horizon in HORIZONS:
            y = frame[target_column(target, horizon, "p75")].to_numpy(int)
            baseline_ll, baseline_brier = binary_losses(
                y, probability_for(frame, "baseline", target, horizon, "p75")
            )
            candidate_ll, candidate_brier = binary_losses(
                y, probability_for(frame, algorithm, target, horizon, "p75")
            )
            storage["baseline_ll"].append(weighted_mean(baseline_ll, weights))
            storage["candidate_ll"].append(weighted_mean(candidate_ll, weights))
            storage["baseline_brier"].append(weighted_mean(baseline_brier, weights))
            storage["candidate_brier"].append(weighted_mean(candidate_brier, weights))
    values = {key: float(np.mean(item)) for key, item in storage.items()}
    return {
        "baseline_log_loss": values["baseline_ll"],
        "log_loss": values["candidate_ll"],
        "relative_log_loss_improvement_vs_baseline": (
            values["baseline_ll"] - values["candidate_ll"]
        )
        / values["baseline_ll"],
        "log_loss_difference_vs_baseline": values["candidate_ll"]
        - values["baseline_ll"],
        "baseline_brier": values["baseline_brier"],
        "brier": values["candidate_brier"],
        "brier_difference_vs_baseline": values["candidate_brier"]
        - values["baseline_brier"],
    }


def artifact_manifest(root: Path, names: Iterable[str]) -> dict[str, Any]:
    return {
        "files": {
            name: {"size": (root / name).stat().st_size, "sha256": sha256(root / name)}
            for name in sorted(names)
        },
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def run() -> None:
    contract = load_contract()
    source_hashes = verify_sources()
    if OUT.exists():
        raise AssertionError(f"artifact root already exists: {OUT}")
    frame, orientations, clocks, source_audit = load_source()
    fitted, fold_audit, parameters = fit_algorithms(
        frame, len(orientations), len(clocks)
    )
    evaluated = evaluate(fitted)
    gates = evaluated["gates"]
    time_slices = evaluate_time_slices(
        evaluated["primary"],
        bool(gates["orientation_clock_residual"]["pass"]),
    )
    passing = [algorithm for algorithm in ALGORITHMS if gates[algorithm]["pass"]]
    selected = next((algorithm for algorithm in ALGORITHMS if algorithm in passing), None)
    qualified_slices = []
    if not time_slices.empty:
        qualified_slices = [
            f"{row.cycle_id}@state_{int(row.current_state)}@clock_{row.entry_clock_quartile}"
            for row in time_slices.loc[
                time_slices["qualified_development_time_slice"]
            ].itertuples(index=False)
        ]
    decision = {
        "label": (
            "development_orientation_calibration_candidate_pending_unseen_validation"
            if selected is not None
            else "orientation_calibration_algorithms_rejected_or_unconfirmed"
        ),
        "passing_algorithms": passing,
        "selected_algorithm": selected,
        "qualified_time_slices": qualified_slices,
        "raw_link_retained_as_diagnostic_only": selected is None,
        "named_loop_good_or_high_promoted": False,
        "same_experiment_refinement_performed": False,
        "later_period_scoring_performed": False,
        "prospective_validated": False,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "source_audit.json", source_audit)
    orientations.to_csv(OUT / "orientation_dictionary.csv", index=False)
    clocks.to_csv(OUT / "orientation_clock_dictionary.csv", index=False)
    fold_audit.to_csv(OUT / "fit_audit.csv", index=False)
    np.savez_compressed(OUT / "model_parameters.npz", **parameters)
    prediction_columns = [
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
        "orientation_index",
        "orientation_clock_index",
        "inverse_compatible_weight",
        *[
            target_column(target, horizon, tier)
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
        *[
            source_probability_column(model, target, horizon, tier)
            for model in ("baseline", "raw_full_link")
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
        *[
            algorithm_probability_column(algorithm, target, horizon, tier)
            for algorithm in ALGORITHMS
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
    ]
    evaluated["primary"].loc[:, prediction_columns].to_parquet(
        OUT / "algorithm_predictions_2024_nov_dec.parquet", index=False
    )
    evaluated["cells"].to_csv(OUT / "cell_metrics.csv", index=False)
    evaluated["pooled"].to_csv(OUT / "pooled_metrics.csv", index=False)
    evaluated["multiplicity"].to_csv(OUT / "multiplicity.csv", index=False)
    evaluated["temporal"].to_csv(OUT / "temporal_slices.csv", index=False)
    evaluated["stocks"].to_csv(OUT / "stock_deletions.csv", index=False)
    evaluated["orientations"].to_csv(OUT / "orientation_slices.csv", index=False)
    evaluated["ranking"].to_csv(OUT / "ranking.csv", index=False)
    time_slices.to_csv(OUT / "time_attraction_slices.csv", index=False)
    write_json(OUT / "algorithm_gates.json", gates)
    write_json(OUT / "decision.json", decision)
    summary = {
        "contract_id": contract["contract_id"],
        "scientific_status": contract["scientific_status"],
        "contract_sha256": sha256(CONTRACT),
        "runner_sha256": sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "source_audit": source_audit,
        "validation_months": list(VALIDATION_MONTHS),
        "validation_rows": len(evaluated["primary"]),
        "validation_anchors": int(evaluated["primary"]["anchor_id"].nunique()),
        "fit_count": len(fold_audit),
        "algorithm_pass": {
            algorithm: bool(gates[algorithm]["pass"])
            for algorithm in ALGORITHMS
        },
        "decision": decision,
        "direct_volume_fields_used": [],
        "volume_label": "historical_volume_not_used",
        "direction_or_signed_return_used": False,
        "later_period_scoring_performed": False,
        "prospective_shadow_read_or_write_performed": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "summary.json", summary)
    names = [
        "algorithm_gates.json",
        "algorithm_predictions_2024_nov_dec.parquet",
        "cell_metrics.csv",
        "decision.json",
        "fit_audit.csv",
        "model_parameters.npz",
        "multiplicity.csv",
        "orientation_clock_dictionary.csv",
        "orientation_dictionary.csv",
        "orientation_slices.csv",
        "pooled_metrics.csv",
        "ranking.csv",
        "source_audit.json",
        "stock_deletions.csv",
        "summary.json",
        "temporal_slices.csv",
        "time_attraction_slices.csv",
    ]
    write_json(OUT / "artifact_manifest.json", artifact_manifest(OUT, names))
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


def self_test() -> None:
    contract = load_contract()
    values = np.asarray([0.1, 0.5, 0.9])
    assert np.allclose(logit(values), -logit(1 - values))
    matrix = categorical_matrix(
        np.asarray([0, 2, 1]), 3, np.asarray([0.25, 0.5, 0.75])
    )
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix.sum(axis=1).A1, [0.25, 0.5, 0.75])
    assert contract["research_only"] is True
    print("self-test passed")


if __name__ == "__main__":
    run()
