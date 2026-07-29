"""Causal OOF test of regime-to-loop movement linkage ideas.

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v1.json"
CONTRACT_SHA256 = "3a8812b1c8a7980565329ab46c88f60ae1cb80bbfe5f738767cb15969589e950"
FACTOR_CONTRACT = HERE / "contracts/20260711-factor-conditioned-loop-occurrence-v1.json"
QUALITY_CONTRACT = HERE / "contracts/20260711-hierarchical-loop-quality-algorithm-v1.json"
FACTOR_ROOT = Path("/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711")
QUALITY_ROOT = Path("/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711")
FACTOR_OOF = FACTOR_ROOT / "oof_predictions_2024.parquet"
QUALITY_OOF = QUALITY_ROOT / "oof_predictions_2024.parquet"
FACTOR_AUDIT = FACTOR_ROOT / "pre_score_audit.json"
QUALITY_AUDIT = QUALITY_ROOT / "independent_artifact_audit.json"
OUT = Path("/private/tmp/stocker_regime_loop_linkage_ideas_v1_20260711")

EXPECTED_HASHES = {
    "factor_oof": "422a7cd24f7e797daef6e5a81756460308bb50a6bf9e2d179dd64abe0b07c6bc",
    "quality_oof": "d7fb8710bdcbda4687bd05a290a8081b89f7d8909697e5e2a2a5523f2caf0c74",
    "factor_audit": "18d4290c50f749ce6ec5434324afa82cd7bebafcd8be198ed8b1c6c7361eedb1",
    "quality_audit": "dba20b5692cca01eb70b60a1c0cb44230af0c0565ba54d035e8d61bb90b3c755",
    "factor_contract": "ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d",
    "quality_contract": "f6956b6ab0495a49669f714df834d1fd0fdaa13b0ecf4b123d6c54c0fc9b5936",
}

JOIN_KEYS = ("symbol_norm", "session_date", "start_timestamp", "cycle_index")
SOURCE_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
EVALUATION_MONTHS = tuple(f"2024-{month:02d}" for month in range(9, 13))
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
FIXED_VARIANTS = (
    "baseline",
    "occurrence_only",
    "topology_only",
    "minimal_time_topology",
    "raw_full_link",
    "partial_full_link",
)
META_VARIANTS = ("calibrated_raw_product", "dependency_stack")
CANDIDATE_VARIANTS = (
    "minimal_time_topology",
    "raw_full_link",
    "partial_full_link",
    "calibrated_raw_product",
    "dependency_stack",
)
ALL_VARIANTS = FIXED_VARIANTS + META_VARIANTS
SEED = 20260711
PROBABILITY_EPSILON = 1e-6
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
        raise AssertionError(f"linkage contract changed: {observed}")
    contract = json.loads(CONTRACT.read_text())
    checks = {
        "id": contract.get("contract_id") == "regime_loop_linkage_ideas_v1",
        "research": contract.get("research_only") is True,
        "live": contract.get("live_ordering_enabled") is False,
        "orders": contract.get("order_placement") == "disabled",
        "later": contract["population_and_join"].get("later_period_paths_permitted") is False,
        "shadow": contract["population_and_join"].get(
            "prospective_shadow_read_or_write_permitted"
        )
        is False,
        "promotion": contract["decision"].get("named_loop_good_or_high_promotion_permitted")
        is False,
        "trading": contract["decision"].get("trading_rule_or_PnL_model_permitted") is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"linkage safety/semantic contract failure: {checks}")
    return contract


def verify_sources() -> dict[str, str]:
    observed = {
        "factor_oof": sha256(FACTOR_OOF),
        "quality_oof": sha256(QUALITY_OOF),
        "factor_audit": sha256(FACTOR_AUDIT),
        "quality_audit": sha256(QUALITY_AUDIT),
        "factor_contract": sha256(FACTOR_CONTRACT),
        "quality_contract": sha256(QUALITY_CONTRACT),
    }
    if observed != EXPECTED_HASHES:
        raise AssertionError(f"frozen source drift: expected={EXPECTED_HASHES}, actual={observed}")
    factor_audit = json.loads(FACTOR_AUDIT.read_text())
    quality_audit = json.loads(QUALITY_AUDIT.read_text())
    audit_checks = {
        "factor_all_passed": factor_audit.get("all_passed") is True,
        "factor_rejection_verified": factor_audit.get("rejection_verified") is True,
        "factor_scoring_not_authorized": factor_audit.get("scoring_authorized") is False,
        "factor_later_unread": factor_audit.get("later_period_rows_read") == 0,
        "quality_all_passed": quality_audit.get("all_passed") is True,
        "quality_shadow_unread": quality_audit.get("shadow_tree_read") is False,
    }
    if not all(audit_checks.values()):
        raise AssertionError(f"parent audit boundary failed: {audit_checks}")
    return observed


def quality_probability_columns() -> list[str]:
    return [
        f"{model}__{target}__h{horizon}__{tier}"
        for model in ("qcontext", "qroute_topology", "qhier")
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]


def quality_class_columns() -> list[str]:
    return [
        f"quality_class__{target}__h{horizon}"
        for target in TARGETS
        for horizon in HORIZONS
    ]


def load_common_population(contract: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    factor_columns = [
        *JOIN_KEYS,
        "anchor_id",
        "cycle_id",
        "state",
        "current_state",
        "target",
        "inverse_compatible_weight",
        "entry_clock_quartile",
        "qhistory",
        "qlimited4",
        "qfull9",
        "month",
    ]
    quality_columns = [
        *JOIN_KEYS,
        "anchor_id",
        "cycle_id",
        "state",
        "current_state",
        "loop_occurs",
        "loop_probability",
        "month_key",
        "quarter",
        *quality_class_columns(),
        *quality_probability_columns(),
    ]
    factor = pd.read_parquet(FACTOR_OOF, columns=factor_columns)
    quality = pd.read_parquet(QUALITY_OOF, columns=quality_columns)
    if len(factor) != 361220 or len(quality) != 216438:
        raise AssertionError("parent OOF row count changed")
    if factor.duplicated(list(JOIN_KEYS)).any() or quality.duplicated(list(JOIN_KEYS)).any():
        raise AssertionError("parent OOF join keys are not one-to-one")
    factor = factor.rename(
        columns={
            "anchor_id": "factor_anchor_id",
            "cycle_id": "factor_cycle_id",
            "state": "factor_state",
            "current_state": "factor_current_state",
            "target": "factor_loop_occurs",
            "month": "factor_month",
        }
    )
    common = quality.merge(
        factor,
        on=list(JOIN_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    checks = {
        "all_quality_rows_joined": common["_merge"].eq("both").all(),
        "cycle_id": common["cycle_id"].astype(str).eq(common["factor_cycle_id"].astype(str)).all(),
        "state": common["state"].astype(int).eq(common["factor_state"].astype(int)).all(),
        "current_state": common["current_state"].astype(int).eq(
            common["factor_current_state"].astype(int)
        ).all(),
        "loop_label": common["loop_occurs"].astype(int).eq(
            common["factor_loop_occurs"].astype(int)
        ).all(),
        "month": common["month_key"].astype(str).eq(common["factor_month"].astype(str)).all(),
        "qhistory": np.allclose(
            common["qhistory"],
            common["loop_probability"],
            atol=float(contract["population_and_join"]["qhistory_must_equal_parent_loop_probability_tolerance"]),
            rtol=0,
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"parent population identity mismatch: {checks}")
    common = common.drop(columns=["_merge"])
    common["month"] = common["month_key"].astype(str)
    common["session_date"] = common["session_date"].astype(str)
    common["symbol_norm"] = common["symbol_norm"].astype(str)
    if set(common["month"].unique()) != set(SOURCE_MONTHS):
        raise AssertionError("source month surface changed")
    if common["anchor_id"].nunique() != 34169:
        raise AssertionError("quality anchor surface changed")
    weight_sums = common.groupby("anchor_id", sort=False)["inverse_compatible_weight"].sum()
    maximum_weight_error = float(np.max(np.abs(weight_sums.to_numpy(float) - 1.0)))
    if maximum_weight_error > 1e-12:
        raise AssertionError(f"inverse-compatible weights changed: {maximum_weight_error}")
    for column in ["qhistory", "qlimited4", "qfull9", *quality_probability_columns()]:
        values = common[column].to_numpy(float)
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise AssertionError(f"invalid source probability: {column}")
    audit = {
        "factor_rows": len(factor),
        "quality_rows": len(quality),
        "common_rows": len(common),
        "quality_anchors": int(common["anchor_id"].nunique()),
        "stocks": int(common["symbol_norm"].nunique()),
        "cycles": int(common["cycle_id"].nunique()),
        "states": sorted(int(value) for value in common["current_state"].unique()),
        "identity_checks": checks,
        "maximum_anchor_weight_error": maximum_weight_error,
        "factor_only_rows_excluded": len(factor) - len(common),
    }
    return common.reset_index(drop=True), audit


def probability_column(variant: str, target: str, horizon: int, tier: str) -> str:
    return f"link__{variant}__{target}__h{horizon}__{tier}"


def joint_target_column(target: str, horizon: int, tier: str) -> str:
    return f"joint_target__{target}__h{horizon}__{tier}"


def clip_component(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)


def logit(values: np.ndarray) -> np.ndarray:
    values = clip_component(values)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def logit_blend(base: np.ndarray, full: np.ndarray, weight: float = 0.5) -> np.ndarray:
    return sigmoid(logit(base) + float(weight) * (logit(full) - logit(base)))


def add_fixed_compositions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    history = output["qhistory"].to_numpy(float)
    limited = output["qlimited4"].to_numpy(float)
    full_occurrence = output["qfull9"].to_numpy(float)
    shrunk_occurrence = logit_blend(history, full_occurrence, 0.5)
    loop_occurs = output["loop_occurs"].to_numpy(bool)
    for target in TARGETS:
        for horizon in HORIZONS:
            quality_class = output[f"quality_class__{target}__h{horizon}"].to_numpy(int)
            for tier in TIERS:
                threshold = 1 if tier == "p75" else 2
                output[joint_target_column(target, horizon, tier)] = (
                    loop_occurs & (quality_class >= threshold)
                ).astype(np.int8)
                context = output[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float)
                topology = output[
                    f"qroute_topology__{target}__h{horizon}__{tier}"
                ].to_numpy(float)
                hierarchy = output[f"qhier__{target}__h{horizon}__{tier}"].to_numpy(float)
                shrunk_quality = logit_blend(context, hierarchy, 0.5)
                probabilities = {
                    "baseline": history * context,
                    "occurrence_only": full_occurrence * context,
                    "topology_only": history * topology,
                    "minimal_time_topology": limited * topology,
                    "raw_full_link": full_occurrence * hierarchy,
                    "partial_full_link": shrunk_occurrence * shrunk_quality,
                }
                for variant, probability in probabilities.items():
                    output[probability_column(variant, target, horizon, tier)] = np.clip(
                        probability, LOSS_EPSILON, 1 - LOSS_EPSILON
                    )
    return output


def meta_features(frame: pd.DataFrame, target: str, horizon: int, tier: str) -> np.ndarray:
    occurrence_base = logit(frame["qhistory"].to_numpy(float))
    occurrence_residual = logit(frame["qfull9"].to_numpy(float)) - occurrence_base
    quality_base = logit(frame[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float))
    quality_residual = (
        logit(frame[f"qhier__{target}__h{horizon}__{tier}"].to_numpy(float))
        - quality_base
    )
    return np.column_stack(
        (
            occurrence_base,
            occurrence_residual,
            quality_base,
            quality_residual,
            occurrence_residual * quality_residual,
        )
    )


def fit_meta_models(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    output = frame.copy()
    for variant in META_VARIANTS:
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    output[probability_column(variant, target, horizon, tier)] = np.nan
    folds: list[dict[str, Any]] = []
    parameters: dict[str, np.ndarray] = {}
    for target_index, target in enumerate(TARGETS):
        for horizon in HORIZONS:
            for tier_index, tier in enumerate(TIERS):
                y_column = joint_target_column(target, horizon, tier)
                raw_column = probability_column("raw_full_link", target, horizon, tier)
                for month_index, month in enumerate(EVALUATION_MONTHS):
                    training = output.loc[output["month"].lt(month)]
                    validation_positions = output.index[output["month"].eq(month)]
                    validation = output.loc[validation_positions]
                    if set(training["month"].unique()).issuperset({"2024-07", "2024-08"}) is False:
                        raise AssertionError(f"meta fold lacks minimum prefix: {month}")
                    weights = training["inverse_compatible_weight"].to_numpy(float)
                    y = training[y_column].to_numpy(int)
                    if not np.array_equal(np.unique(y), np.asarray([0, 1])):
                        raise AssertionError(f"meta target lacks a class: {target} h{horizon} {tier}")
                    raw_train = logit(training[raw_column].to_numpy(float)).reshape(-1, 1)
                    raw_validation = logit(validation[raw_column].to_numpy(float)).reshape(-1, 1)
                    stack_train = meta_features(training, target, horizon, tier)
                    stack_validation = meta_features(validation, target, horizon, tier)
                    prefix = f"{target}__h{horizon}__{tier}__{month}"
                    for variant, train_x, validation_x in (
                        ("calibrated_raw_product", raw_train, raw_validation),
                        ("dependency_stack", stack_train, stack_validation),
                    ):
                        scaler = StandardScaler().fit(train_x, sample_weight=weights)
                        train_scaled = scaler.transform(train_x)
                        validation_scaled = scaler.transform(validation_x)
                        model = LogisticRegression(
                            C=0.1,
                            solver="lbfgs",
                            max_iter=2000,
                            tol=1e-10,
                            random_state=SEED,
                        ).fit(train_scaled, y, sample_weight=weights)
                        if not np.array_equal(model.classes_, np.asarray([0, 1])):
                            raise AssertionError("meta class order changed")
                        if int(model.n_iter_[0]) >= 2000:
                            raise AssertionError("meta model did not converge")
                        probability = model.predict_proba(validation_scaled)[:, 1]
                        output.loc[
                            validation_positions,
                            probability_column(variant, target, horizon, tier),
                        ] = probability
                        key = f"{prefix}__{variant}"
                        parameters[f"{key}__mean"] = np.asarray(scaler.mean_, dtype=float)
                        parameters[f"{key}__scale"] = np.asarray(scaler.scale_, dtype=float)
                        parameters[f"{key}__coef"] = np.asarray(model.coef_, dtype=float)
                        parameters[f"{key}__intercept"] = np.asarray(model.intercept_, dtype=float)
                        parameters[f"{key}__n_iter"] = np.asarray(model.n_iter_, dtype=int)
                        folds.append(
                            {
                                "variant": variant,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "validation_month": month,
                                "training_months": json.dumps(
                                    sorted(training["month"].unique().tolist())
                                ),
                                "training_rows": len(training),
                                "training_weight": float(weights.sum()),
                                "training_positives": int(y.sum()),
                                "validation_rows": len(validation),
                                "validation_positives": int(validation[y_column].sum()),
                                "feature_width": train_x.shape[1],
                                "n_iter": int(model.n_iter_[0]),
                            }
                        )
    primary = output["month"].isin(EVALUATION_MONTHS)
    meta_columns = [
        probability_column(variant, target, horizon, tier)
        for variant in META_VARIANTS
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    values = output.loc[primary, meta_columns].to_numpy(float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise AssertionError("invalid or incomplete meta OOF probabilities")
    return output, pd.DataFrame(folds), parameters


def binary_losses(observed: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), LOSS_EPSILON, 1 - LOSS_EPSILON)
    return (
        -(observed * np.log(probability) + (1 - observed) * np.log(1 - probability)),
        (observed - probability) ** 2,
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(values) or weights.sum() <= 0:
        return math.nan
    return float(np.average(values, weights=weights))


def calibration_summary(
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    minimum_rows: int,
) -> tuple[float, float, int]:
    observed = np.asarray(observed, dtype=float)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    bins = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9)
    supported: list[tuple[float, float]] = []
    for index in range(10):
        selected = bins == index
        if int(selected.sum()) < int(minimum_rows) or weights[selected].sum() <= 0:
            continue
        error = abs(
            weighted_mean(observed[selected], weights[selected])
            - weighted_mean(probability[selected], weights[selected])
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


def cell_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for weight_label, weights in (
        ("inverse_compatible", frame["inverse_compatible_weight"].to_numpy(float)),
        ("unweighted", np.ones(len(frame), dtype=float)),
    ):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    observed = frame[joint_target_column(target, horizon, tier)].to_numpy(int)
                    baseline_probability = frame[
                        probability_column("baseline", target, horizon, tier)
                    ].to_numpy(float)
                    baseline_ll, baseline_brier = binary_losses(observed, baseline_probability)
                    baseline_ece, baseline_maximum, baseline_bins = calibration_summary(
                        observed, baseline_probability, weights, 500
                    )
                    for variant in ALL_VARIANTS:
                        probability = frame[
                            probability_column(variant, target, horizon, tier)
                        ].to_numpy(float)
                        log_loss, brier = binary_losses(observed, probability)
                        ece, maximum, bins = calibration_summary(observed, probability, weights, 500)
                        candidate_ll = weighted_mean(log_loss, weights)
                        base_ll = weighted_mean(baseline_ll, weights)
                        rows.append(
                            {
                                "weight_surface": weight_label,
                                "variant": variant,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "positives": int(observed.sum()),
                                "weight_sum": float(weights.sum()),
                                "log_loss": candidate_ll,
                                "baseline_log_loss": base_ll,
                                "relative_log_loss_improvement": (base_ll - candidate_ll) / base_ll,
                                "log_loss_difference": candidate_ll - base_ll,
                                "brier": weighted_mean(brier, weights),
                                "baseline_brier": weighted_mean(baseline_brier, weights),
                                "brier_difference": weighted_mean(brier - baseline_brier, weights),
                                "ece": ece,
                                "baseline_ece": baseline_ece,
                                "maximum_supported_bin_error": maximum,
                                "baseline_maximum_supported_bin_error": baseline_maximum,
                                "supported_bins": bins,
                                "baseline_supported_bins": baseline_bins,
                            }
                        )
    return pd.DataFrame(rows)


def pooled_difference(
    frame: pd.DataFrame, variant: str, weights: np.ndarray | None = None
) -> dict[str, float]:
    if weights is None:
        weights = frame["inverse_compatible_weight"].to_numpy(float)
    baseline_ll_values: list[float] = []
    candidate_ll_values: list[float] = []
    baseline_brier_values: list[float] = []
    candidate_brier_values: list[float] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                observed = frame[joint_target_column(target, horizon, tier)].to_numpy(int)
                baseline_ll, baseline_brier = binary_losses(
                    observed,
                    frame[probability_column("baseline", target, horizon, tier)].to_numpy(float),
                )
                candidate_ll, candidate_brier = binary_losses(
                    observed,
                    frame[probability_column(variant, target, horizon, tier)].to_numpy(float),
                )
                baseline_ll_values.append(weighted_mean(baseline_ll, weights))
                candidate_ll_values.append(weighted_mean(candidate_ll, weights))
                baseline_brier_values.append(weighted_mean(baseline_brier, weights))
                candidate_brier_values.append(weighted_mean(candidate_brier, weights))
    base_ll = float(np.mean(baseline_ll_values))
    candidate_ll = float(np.mean(candidate_ll_values))
    base_brier = float(np.mean(baseline_brier_values))
    candidate_brier = float(np.mean(candidate_brier_values))
    return {
        "baseline_log_loss": base_ll,
        "log_loss": candidate_ll,
        "relative_log_loss_improvement": (base_ll - candidate_ll) / base_ll,
        "log_loss_difference": candidate_ll - base_ll,
        "baseline_brier": base_brier,
        "brier": candidate_brier,
        "brier_difference": candidate_brier - base_brier,
    }


def per_row_equal_cell_loss_difference(
    frame: pd.DataFrame, variant: str, loss: str
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                observed = frame[joint_target_column(target, horizon, tier)].to_numpy(int)
                baseline = binary_losses(
                    observed,
                    frame[probability_column("baseline", target, horizon, tier)].to_numpy(float),
                )[0 if loss == "log_loss" else 1]
                candidate = binary_losses(
                    observed,
                    frame[probability_column(variant, target, horizon, tier)].to_numpy(float),
                )[0 if loss == "log_loss" else 1]
                result += (candidate - baseline) / 12.0
    return result


def daily_weighted_values(
    frame: pd.DataFrame, values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    daily = pd.DataFrame(
        {
            "session_date": frame["session_date"].astype(str).to_numpy(),
            "weighted": np.asarray(values, float) * np.asarray(weights, float),
            "weight": np.asarray(weights, float),
        }
    ).groupby("session_date", sort=True).sum()
    daily = daily.loc[daily["weight"] > 0]
    return (daily["weighted"] / daily["weight"]).to_numpy(float)


def five_session_blocks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray(
        [
            float(values[index : index + 5].mean())
            for index in range(0, len(values), 5)
            if len(values[index : index + 5]) == 5
        ],
        dtype=float,
    )


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = five_session_blocks(values)
    if len(blocks) < 5:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip_p_value(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return math.nan
    rng = np.random.default_rng(seed)
    null = (rng.choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values))) @ values) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def holm_adjust(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p"] = 1.0
    output["holm_pass"] = False
    output["holm_rank"] = 0
    output["family_size"] = 0
    if output.empty:
        return output
    groups = (
        {"all": output.index.tolist()}
        if not group_columns
        else output.groupby(list(group_columns), sort=True).groups
    )
    for _, positions in groups.items():
        ordered = sorted(positions, key=lambda position: float(output.loc[position, "p_value"]))
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


def ranking_recall(frame: pd.DataFrame, variant: str, target: str, horizon: int, tier: str) -> float:
    selected = frame.loc[:, ["anchor_id", joint_target_column(target, horizon, tier)]].copy()
    selected["probability"] = frame[
        probability_column(variant, target, horizon, tier)
    ].to_numpy(float)
    selected = selected.sort_values(
        ["anchor_id", "probability"], ascending=[True, False], kind="stable"
    )
    selected["rank"] = selected.groupby("anchor_id", sort=False).cumcount() + 1
    positives = selected[joint_target_column(target, horizon, tier)].to_numpy(int)
    denominator = int(positives.sum())
    if denominator == 0:
        return math.nan
    return float(((selected["rank"].to_numpy(int) <= 3) & (positives == 1)).sum() / denominator)


def evaluate_variants(
    frame: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    primary = frame.loc[frame["month"].isin(EVALUATION_MONTHS)].reset_index(drop=True)
    cells = cell_metric_rows(primary)
    pooled_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    weights = primary["inverse_compatible_weight"].to_numpy(float)
    for variant_index, variant in enumerate(ALL_VARIANTS):
        pooled = pooled_difference(primary, variant)
        detail: dict[str, Any] = {
            "variant": variant,
            "rows": len(primary),
            "anchors": int(primary["anchor_id"].nunique()),
            **pooled,
        }
        for loss_index, loss in enumerate(("log_loss", "brier")):
            values = per_row_equal_cell_loss_difference(primary, variant, loss)
            daily = daily_weighted_values(primary, values, weights)
            lower, upper = bootstrap_interval(daily, SEED + variant_index * 100 + loss_index)
            p_value = sign_flip_p_value(daily, SEED + variant_index * 100 + loss_index + 10)
            detail[f"{loss}_daily_sessions"] = len(daily)
            detail[f"{loss}_bootstrap_lower"] = lower
            detail[f"{loss}_bootstrap_upper"] = upper
            detail[f"{loss}_sign_flip_p_value"] = p_value
            if variant in CANDIDATE_VARIANTS:
                multiplicity_rows.append(
                    {"variant": variant, "endpoint": loss, "p_value": p_value}
                )
        pooled_rows.append(detail)

        for month in EVALUATION_MONTHS:
            selected = primary.loc[primary["month"].eq(month)]
            temporal_rows.append({"variant": variant, "slice": month, **pooled_difference(selected, variant)})
        for half, months in (
            ("2024-09_to_2024-10", ("2024-09", "2024-10")),
            ("2024-11_to_2024-12", ("2024-11", "2024-12")),
        ):
            selected = primary.loc[primary["month"].isin(months)]
            temporal_rows.append({"variant": variant, "slice": half, **pooled_difference(selected, variant)})
        for symbol in sorted(primary["symbol_norm"].unique()):
            selected = primary.loc[primary["symbol_norm"].ne(symbol)]
            stock_rows.append(
                {"variant": variant, "deleted_symbol": symbol, **pooled_difference(selected, variant)}
            )
        for (cycle_id, current_state), selected in primary.groupby(
            ["cycle_id", "current_state"], sort=True
        ):
            positives = sum(
                int(selected[joint_target_column(target, horizon, tier)].sum())
                for target in TARGETS
                for horizon in HORIZONS
                for tier in TIERS
            )
            supported = len(selected) >= 500 and positives >= 30
            orientation_rows.append(
                {
                    "variant": variant,
                    "cycle_id": cycle_id,
                    "current_state": int(current_state),
                    "rows": len(selected),
                    "joint_positives_across_cells": positives,
                    "supported": supported,
                    **pooled_difference(selected, variant),
                }
            )
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    baseline_recall = ranking_recall(primary, "baseline", target, horizon, tier)
                    recall = ranking_recall(primary, variant, target, horizon, tier)
                    ranking_rows.append(
                        {
                            "variant": variant,
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            "top_three_recall": recall,
                            "baseline_top_three_recall": baseline_recall,
                            "gain": recall - baseline_recall,
                        }
                    )
    pooled_frame = pd.DataFrame(pooled_rows)
    multiplicity = holm_adjust(pd.DataFrame(multiplicity_rows), ["endpoint"])
    temporal = pd.DataFrame(temporal_rows)
    stocks = pd.DataFrame(stock_rows)
    orientations = pd.DataFrame(orientation_rows)
    ranking = pd.DataFrame(ranking_rows)
    gates: dict[str, Any] = {}
    primary_cells = cells.loc[cells["weight_surface"].eq("inverse_compatible")]
    for variant in CANDIDATE_VARIANTS:
        pooled = pooled_frame.set_index("variant").loc[variant]
        variant_cells = primary_cells.loc[primary_cells["variant"].eq(variant)]
        variant_temporal = temporal.loc[
            temporal["variant"].eq(variant) & temporal["slice"].isin(EVALUATION_MONTHS)
        ]
        variant_stocks = stocks.loc[stocks["variant"].eq(variant)]
        variant_orientations = orientations.loc[
            orientations["variant"].eq(variant) & orientations["supported"]
        ]
        variant_ranking = ranking.loc[ranking["variant"].eq(variant)]
        holm = multiplicity.loc[multiplicity["variant"].eq(variant)].set_index("endpoint")
        checks = {
            "pooled_relative_log_loss": float(pooled["relative_log_loss_improvement"]) >= 0.005,
            "pooled_brier": float(pooled["brier_difference"]) < 0,
            "bootstrap_log_loss": float(pooled["log_loss_bootstrap_upper"]) < 0,
            "bootstrap_brier": float(pooled["brier_bootstrap_upper"]) < 0,
            "Holm_log_loss": bool(holm.loc["log_loss", "holm_pass"]),
            "Holm_brier": bool(holm.loc["brier", "holm_pass"]),
            "all_cell_losses": bool(
                (variant_cells["log_loss_difference"] <= 0).all()
                and (variant_cells["brier_difference"] <= 0).all()
            ),
            "all_cell_calibration": bool(
                (variant_cells["ece"] <= variant_cells["baseline_ece"]).all()
                and (variant_cells["maximum_supported_bin_error"] <= 0.02).all()
            ),
            "every_month": bool(
                (variant_temporal["log_loss_difference"] < 0).all()
                and (variant_temporal["brier_difference"] < 0).all()
            ),
            "every_stock_deletion": bool(
                (variant_stocks["log_loss_difference"] <= 0).all()
                and (variant_stocks["brier_difference"] <= 0).all()
            ),
            "zero_supported_orientation_reversals": bool(
                len(variant_orientations) > 0
                and (variant_orientations["log_loss_difference"] <= 0).all()
                and (variant_orientations["brier_difference"] <= 0).all()
            ),
            "ranking": bool((variant_ranking["gain"] >= 0).all()),
        }
        if variant == "dependency_stack":
            raw = pooled_frame.set_index("variant").loc["raw_full_link"]
            calibrated = pooled_frame.set_index("variant").loc["calibrated_raw_product"]
            checks["below_raw_full_link"] = bool(
                pooled["log_loss"] < raw["log_loss"] and pooled["brier"] < raw["brier"]
            )
            checks["below_calibrated_raw_product"] = bool(
                pooled["log_loss"] < calibrated["log_loss"]
                and pooled["brier"] < calibrated["brier"]
            )
        gates[variant] = {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "supported_orientation_rows": len(variant_orientations),
            "orientation_log_loss_reversals": int(
                (variant_orientations["log_loss_difference"] > 0).sum()
            ),
            "orientation_brier_reversals": int(
                (variant_orientations["brier_difference"] > 0).sum()
            ),
            "cell_log_loss_reversals": int((variant_cells["log_loss_difference"] > 0).sum()),
            "cell_brier_reversals": int((variant_cells["brier_difference"] > 0).sum()),
            "cell_calibration_failures": int(
                (
                    (variant_cells["ece"] > variant_cells["baseline_ece"])
                    | (variant_cells["maximum_supported_bin_error"] > 0.02)
                ).sum()
            ),
            "ranking_cell_degradations": int((variant_ranking["gain"] < 0).sum()),
        }
    return {
        "primary": primary,
        "cells": cells,
        "pooled": pooled_frame,
        "multiplicity": multiplicity,
        "temporal": temporal,
        "stocks": stocks,
        "orientations": orientations,
        "ranking": ranking,
        "gates": gates,
    }


def evaluate_attraction_slices(
    primary: pd.DataFrame, dependency_global_pass: bool
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slice_index, ((cycle_id, current_state, clock), selected) in enumerate(
        primary.groupby(["cycle_id", "current_state", "entry_clock_quartile"], sort=True)
    ):
        positive_count = sum(
            int(selected[joint_target_column(target, horizon, "p75")].sum())
            for target in TARGETS
            for horizon in HORIZONS
        )
        support = bool(
            len(selected) >= 500
            and positive_count >= 50
            and selected["symbol_norm"].nunique() >= 15
        )
        if not support:
            continue
        cell_checks: list[bool] = []
        maximum_error = 0.0
        for target in TARGETS:
            for horizon in HORIZONS:
                observed = selected[joint_target_column(target, horizon, "p75")].to_numpy(int)
                weights = selected["inverse_compatible_weight"].to_numpy(float)
                base_probability = selected[
                    probability_column("baseline", target, horizon, "p75")
                ].to_numpy(float)
                candidate_probability = selected[
                    probability_column("dependency_stack", target, horizon, "p75")
                ].to_numpy(float)
                base_ll, base_brier = binary_losses(observed, base_probability)
                candidate_ll, candidate_brier = binary_losses(observed, candidate_probability)
                cell_checks.append(
                    weighted_mean(candidate_ll - base_ll, weights) <= 0
                    and weighted_mean(candidate_brier - base_brier, weights) <= 0
                )
                _, error, _ = calibration_summary(
                    observed, candidate_probability, weights, 100
                )
                maximum_error = max(maximum_error, error)
        pooled = pooled_difference_p75(selected, "dependency_stack")
        half_checks: list[bool] = []
        for months in (("2024-09", "2024-10"), ("2024-11", "2024-12")):
            half = selected.loc[selected["month"].isin(months)]
            metrics = pooled_difference_p75(half, "dependency_stack")
            half_checks.append(
                metrics["log_loss_difference"] < 0 and metrics["brier_difference"] < 0
            )
        values = per_row_p75_loss_difference(selected, "dependency_stack")
        daily = daily_weighted_values(
            selected, values, selected["inverse_compatible_weight"].to_numpy(float)
        )
        p_value = sign_flip_p_value(daily, SEED + slice_index)
        checks = {
            "global_dependency_stack": dependency_global_pass,
            "relative_log_loss": pooled["relative_log_loss_improvement"] >= 0.01,
            "all_six_cells": all(cell_checks),
            "both_halves": all(half_checks),
            "calibration": maximum_error <= 0.03,
        }
        rows.append(
            {
                "cycle_id": cycle_id,
                "current_state": int(current_state),
                "entry_clock_quartile": clock,
                "rows": len(selected),
                "joint_positives_across_p75_cells": positive_count,
                "stocks": int(selected["symbol_norm"].nunique()),
                **pooled,
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
    frame = holm_adjust(frame, group_columns=[])
    frame["qualified_development_attraction_slice"] = (
        frame["pass_before_Holm"] & frame["holm_pass"]
    )
    return frame


def pooled_difference_p75(frame: pd.DataFrame, variant: str) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    base_lls: list[float] = []
    candidate_lls: list[float] = []
    base_briers: list[float] = []
    candidate_briers: list[float] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            observed = frame[joint_target_column(target, horizon, "p75")].to_numpy(int)
            base_ll, base_brier = binary_losses(
                observed,
                frame[probability_column("baseline", target, horizon, "p75")].to_numpy(float),
            )
            candidate_ll, candidate_brier = binary_losses(
                observed,
                frame[
                    probability_column(variant, target, horizon, "p75")
                ].to_numpy(float),
            )
            base_lls.append(weighted_mean(base_ll, weights))
            candidate_lls.append(weighted_mean(candidate_ll, weights))
            base_briers.append(weighted_mean(base_brier, weights))
            candidate_briers.append(weighted_mean(candidate_brier, weights))
    base_ll = float(np.mean(base_lls))
    candidate_ll = float(np.mean(candidate_lls))
    base_brier = float(np.mean(base_briers))
    candidate_brier = float(np.mean(candidate_briers))
    return {
        "baseline_log_loss": base_ll,
        "log_loss": candidate_ll,
        "relative_log_loss_improvement": (base_ll - candidate_ll) / base_ll,
        "log_loss_difference": candidate_ll - base_ll,
        "baseline_brier": base_brier,
        "brier": candidate_brier,
        "brier_difference": candidate_brier - base_brier,
    }


def per_row_p75_loss_difference(frame: pd.DataFrame, variant: str) -> np.ndarray:
    output = np.zeros(len(frame), dtype=float)
    for target in TARGETS:
        for horizon in HORIZONS:
            observed = frame[joint_target_column(target, horizon, "p75")].to_numpy(int)
            baseline = binary_losses(
                observed,
                frame[probability_column("baseline", target, horizon, "p75")].to_numpy(float),
            )[0]
            candidate = binary_losses(
                observed,
                frame[probability_column(variant, target, horizon, "p75")].to_numpy(float),
            )[0]
            output += (candidate - baseline) / 6.0
    return output


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
    common, join_audit = load_common_population(contract)
    composed = add_fixed_compositions(common)
    linked, fold_audit, parameters = fit_meta_models(composed)
    evaluated = evaluate_variants(linked, contract)
    gates = evaluated["gates"]
    attraction = evaluate_attraction_slices(
        evaluated["primary"], bool(gates["dependency_stack"]["pass"])
    )
    passing = [variant for variant in CANDIDATE_VARIANTS if gates[variant]["pass"]]
    priority = contract["decision"]["priority_if_multiple_global_variants_pass"]
    selected = next((variant for variant in priority if variant in passing), None)
    decision = {
        "label": (
            "development_linkage_candidate_pending_unseen_validation"
            if selected is not None
            else "linkage_idea_rejected_or_unconfirmed"
        ),
        "passing_variants": passing,
        "selected_variant": selected,
        "qualified_attraction_slices": (
            attraction.loc[attraction["qualified_development_attraction_slice"]]
            .apply(
                lambda row: f"{row['cycle_id']}@state_{int(row['current_state'])}@clock_{row['entry_clock_quartile']}",
                axis=1,
            )
            .tolist()
            if not attraction.empty
            else []
        ),
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
    write_json(OUT / "join_audit.json", join_audit)
    fold_audit.to_csv(OUT / "meta_fold_audit.csv", index=False)
    np.savez_compressed(OUT / "meta_model_parameters.npz", **parameters)
    prediction_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "month",
        "quarter",
        "cycle_index",
        "cycle_id",
        "state",
        "current_state",
        "entry_clock_quartile",
        "inverse_compatible_weight",
        "loop_occurs",
        "qhistory",
        "qlimited4",
        "qfull9",
        *quality_class_columns(),
        *quality_probability_columns(),
        *[
            joint_target_column(target, horizon, tier)
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
        *[
            probability_column(variant, target, horizon, tier)
            for variant in ALL_VARIANTS
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
    ]
    evaluated["primary"].loc[:, prediction_columns].to_parquet(
        OUT / "linkage_predictions_2024_sep_dec.parquet", index=False
    )
    evaluated["cells"].to_csv(OUT / "cell_metrics.csv", index=False)
    evaluated["pooled"].to_csv(OUT / "pooled_metrics.csv", index=False)
    evaluated["multiplicity"].to_csv(OUT / "multiplicity.csv", index=False)
    evaluated["temporal"].to_csv(OUT / "temporal_slices.csv", index=False)
    evaluated["stocks"].to_csv(OUT / "stock_deletions.csv", index=False)
    evaluated["orientations"].to_csv(OUT / "orientation_slices.csv", index=False)
    evaluated["ranking"].to_csv(OUT / "ranking.csv", index=False)
    attraction.to_csv(OUT / "attraction_slices.csv", index=False)
    write_json(OUT / "variant_gates.json", gates)
    write_json(OUT / "decision.json", decision)
    summary = {
        "contract_id": contract["contract_id"],
        "scientific_status": contract["scientific_status"],
        "contract_sha256": sha256(CONTRACT),
        "runner_sha256": sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "join_audit": join_audit,
        "source_months": list(SOURCE_MONTHS),
        "primary_evaluation_months": list(EVALUATION_MONTHS),
        "primary_rows": len(evaluated["primary"]),
        "primary_anchors": int(evaluated["primary"]["anchor_id"].nunique()),
        "meta_fits": len(fold_audit),
        "global_variant_pass": {
            variant: bool(gates[variant]["pass"]) for variant in CANDIDATE_VARIANTS
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
        "attraction_slices.csv",
        "cell_metrics.csv",
        "decision.json",
        "join_audit.json",
        "linkage_predictions_2024_sep_dec.parquet",
        "meta_fold_audit.csv",
        "meta_model_parameters.npz",
        "multiplicity.csv",
        "orientation_slices.csv",
        "pooled_metrics.csv",
        "ranking.csv",
        "stock_deletions.csv",
        "summary.json",
        "temporal_slices.csv",
        "variant_gates.json",
    ]
    write_json(OUT / "artifact_manifest.json", artifact_manifest(OUT, names))
    print(json.dumps(summary, indent=2, sort_keys=True))


def self_test() -> None:
    contract = load_contract()
    base = np.asarray([0.1, 0.4, 0.8])
    full = np.asarray([0.2, 0.6, 0.9])
    assert np.allclose(logit_blend(base, base), base)
    blended = logit_blend(base, full, 0.5)
    assert np.all(blended > np.minimum(base, full))
    assert np.all(blended < np.maximum(base, full))
    frame = pd.DataFrame(
        {"endpoint": ["a", "a", "b"], "p_value": [0.01, 0.04, 0.03]}
    )
    adjusted = holm_adjust(frame, ["endpoint"])
    assert adjusted.loc[adjusted["endpoint"].eq("a"), "family_size"].eq(2).all()
    assert contract["research_only"] is True
    print("self-test passed")


if __name__ == "__main__":
    run()
