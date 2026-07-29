"""Independent replay audit for regime-loop-linkage-ideas-v3.

This auditor does not import any linkage runner.

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-regime-loop-linkage-ideas-v3.json"
RUNNER = HERE / "run_regime_loop_linkage_ideas_v3.py"
V2_RUNNER = HERE / "run_regime_loop_linkage_ideas_v2.py"
V1_RUNNER = HERE / "run_regime_loop_linkage_ideas_v1.py"
FACTOR = Path(
    "/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711/oof_predictions_2024.parquet"
)
QUALITY = Path(
    "/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711/oof_predictions_2024.parquet"
)
ROOT = Path("/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711")

EXPECTED = {
    "contract": "88a60956857e6ccb4fb5e74beb9085e46765e55b31763b26927dc496822ce947",
    "runner": "c0e8786670fd51e3d93290ecd56ba51322ebe6ace0fd7e521803f2fd8c1ce72e",
    "v2_runner": "b38a17b5e5023951e992004fac51e4c264af2c65e7f19c4b35ecea14cbd5e6ba",
    "v1_runner": "e134f01f4d6da58581205fe8070f90a2f17d0fc0945dea0b42a2ca1c96bfa51a",
    "factor": "422a7cd24f7e797daef6e5a81756460308bb50a6bf9e2d179dd64abe0b07c6bc",
    "quality": "d7fb8710bdcbda4687bd05a290a8081b89f7d8909697e5e2a2a5523f2caf0c74",
}

JOIN_KEYS = ("symbol_norm", "session_date", "start_timestamp", "cycle_index")
MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
EVAL_MONTHS = tuple(f"2024-{month:02d}" for month in range(9, 13))
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
FIXED = (
    "baseline",
    "occurrence_only",
    "topology_only",
    "minimal_time_topology",
    "raw_full_link",
    "partial_full_link",
)
META = ("calibrated_raw_product", "dependency_stack")
ALL = FIXED + META
CANDIDATES = (
    "minimal_time_topology",
    "raw_full_link",
    "partial_full_link",
    "calibrated_raw_product",
    "dependency_stack",
)
SEED = 20260711
EPS = 1e-12
COMPONENT_EPS = 1e-6
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


def pcol(variant: str, target: str, horizon: int, tier: str) -> str:
    return f"link__{variant}__{target}__h{horizon}__{tier}"


def ycol(target: str, horizon: int, tier: str) -> str:
    return f"joint_target__{target}__h{horizon}__{tier}"


def quality_classes() -> list[str]:
    return [
        f"quality_class__{target}__h{horizon}"
        for target in TARGETS
        for horizon in HORIZONS
    ]


def quality_probabilities() -> list[str]:
    return [
        f"{model}__{target}__h{horizon}__{tier}"
        for model in ("qcontext", "qroute_topology", "qhier")
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, float), COMPONENT_EPS, 1 - COMPONENT_EPS)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    output[~positive] = exp_value / (1 + exp_value)
    return output


def blend(base: np.ndarray, full: np.ndarray) -> np.ndarray:
    return sigmoid(logit(base) + 0.5 * (logit(full) - logit(base)))


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
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
        *quality_classes(),
        *quality_probabilities(),
    ]
    factor = pd.read_parquet(FACTOR, columns=factor_columns).rename(
        columns={
            "anchor_id": "factor_anchor_id",
            "cycle_id": "factor_cycle_id",
            "state": "factor_state",
            "current_state": "factor_current_state",
            "target": "factor_target",
            "month": "factor_month",
        }
    )
    quality = pd.read_parquet(QUALITY, columns=quality_columns)
    common = quality.merge(factor, on=list(JOIN_KEYS), how="left", validate="one_to_one")
    identity = {
        "rows": len(common) == 216438,
        "cycle": common["cycle_id"].astype(str).eq(common["factor_cycle_id"].astype(str)).all(),
        "state": common["state"].astype(int).eq(common["factor_state"].astype(int)).all(),
        "current": common["current_state"].astype(int).eq(
            common["factor_current_state"].astype(int)
        ).all(),
        "target": common["loop_occurs"].astype(int).eq(common["factor_target"].astype(int)).all(),
        "month": common["month_key"].astype(str).eq(common["factor_month"].astype(str)).all(),
    }
    if not all(identity.values()):
        raise AssertionError(f"audit source identity failed: {identity}")
    common["month"] = common["month_key"].astype(str)
    common["session_date"] = common["session_date"].astype(str)
    common["symbol_norm"] = common["symbol_norm"].astype(str)
    difference = common["qhistory"].to_numpy(float) - common["loop_probability"].to_numpy(float)
    diagnostic = {
        "correlation": float(np.corrcoef(common["qhistory"], common["loop_probability"])[0, 1]),
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "maximum_absolute_difference": float(np.abs(difference).max()),
    }
    return common.reset_index(drop=True), {"identity": identity, "structural": diagnostic}


def add_fixed(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    history = output["qhistory"].to_numpy(float)
    limited = output["qlimited4"].to_numpy(float)
    full = output["qfull9"].to_numpy(float)
    shrunk_occurrence = blend(history, full)
    event = output["loop_occurs"].to_numpy(bool)
    for target in TARGETS:
        for horizon in HORIZONS:
            quality = output[f"quality_class__{target}__h{horizon}"].to_numpy(int)
            for tier in TIERS:
                threshold = 1 if tier == "p75" else 2
                output[ycol(target, horizon, tier)] = (
                    event & (quality >= threshold)
                ).astype(np.int8)
                context = output[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float)
                topology = output[
                    f"qroute_topology__{target}__h{horizon}__{tier}"
                ].to_numpy(float)
                hierarchy = output[f"qhier__{target}__h{horizon}__{tier}"].to_numpy(float)
                values = {
                    "baseline": history * context,
                    "occurrence_only": full * context,
                    "topology_only": history * topology,
                    "minimal_time_topology": limited * topology,
                    "raw_full_link": full * hierarchy,
                    "partial_full_link": shrunk_occurrence * blend(context, hierarchy),
                }
                for variant, probability in values.items():
                    output[pcol(variant, target, horizon, tier)] = np.clip(
                        probability, EPS, 1 - EPS
                    )
    return output


def features(frame: pd.DataFrame, target: str, horizon: int, tier: str) -> np.ndarray:
    occurrence = logit(frame["qhistory"].to_numpy(float))
    occurrence_residual = logit(frame["qfull9"].to_numpy(float)) - occurrence
    quality = logit(frame[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float))
    quality_residual = (
        logit(frame[f"qhier__{target}__h{horizon}__{tier}"].to_numpy(float)) - quality
    )
    return np.column_stack(
        (occurrence, occurrence_residual, quality, quality_residual, occurrence_residual * quality_residual)
    )


def refit_meta(
    frame: pd.DataFrame, ledger: pd.DataFrame
) -> tuple[float, float, int]:
    parameter_file = np.load(ROOT / "meta_model_parameters.npz")
    maximum_prediction_error = 0.0
    maximum_parameter_error = 0.0
    fits = 0
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                target_column = ycol(target, horizon, tier)
                raw_column = pcol("raw_full_link", target, horizon, tier)
                for month in EVAL_MONTHS:
                    training = frame.loc[frame["month"].lt(month)]
                    validation = frame.loc[frame["month"].eq(month)]
                    stored = ledger.loc[ledger["month"].eq(month)]
                    weights = training["inverse_compatible_weight"].to_numpy(float)
                    y = training[target_column].to_numpy(int)
                    pairs = (
                        (
                            "calibrated_raw_product",
                            logit(training[raw_column].to_numpy(float)).reshape(-1, 1),
                            logit(validation[raw_column].to_numpy(float)).reshape(-1, 1),
                        ),
                        (
                            "dependency_stack",
                            features(training, target, horizon, tier),
                            features(validation, target, horizon, tier),
                        ),
                    )
                    for variant, train_x, validation_x in pairs:
                        scaler = StandardScaler().fit(train_x, sample_weight=weights)
                        model = LogisticRegression(
                            C=0.1,
                            solver="lbfgs",
                            max_iter=2000,
                            tol=1e-10,
                            random_state=SEED,
                        ).fit(scaler.transform(train_x), y, sample_weight=weights)
                        probability = model.predict_proba(scaler.transform(validation_x))[:, 1]
                        expected = stored[pcol(variant, target, horizon, tier)].to_numpy(float)
                        maximum_prediction_error = max(
                            maximum_prediction_error, float(np.max(np.abs(probability - expected)))
                        )
                        key = f"{target}__h{horizon}__{tier}__{month}__{variant}"
                        for name, value in (
                            ("mean", scaler.mean_),
                            ("scale", scaler.scale_),
                            ("coef", model.coef_),
                            ("intercept", model.intercept_),
                            ("n_iter", model.n_iter_),
                        ):
                            maximum_parameter_error = max(
                                maximum_parameter_error,
                                float(np.max(np.abs(parameter_file[f"{key}__{name}"] - value))),
                            )
                        fits += 1
    parameter_file.close()
    return maximum_prediction_error, maximum_parameter_error, fits


def losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return (-(y * np.log(p) + (1 - y) * np.log(1 - p)), (y - p) ** 2)


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() else math.nan


def calibration(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray, minimum: int
) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    supported: list[tuple[float, float]] = []
    for index in range(10):
        selected = bins == index
        if selected.sum() < minimum or weights[selected].sum() <= 0:
            continue
        error = abs(wmean(y[selected], weights[selected]) - wmean(p[selected], weights[selected]))
        supported.append((weights[selected].sum(), error))
    if not supported:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in supported)
    return (
        float(sum(weight * error for weight, error in supported) / total),
        float(max(error for _, error in supported)),
        len(supported),
    )


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
                    base_p = frame[pcol("baseline", target, horizon, tier)].to_numpy(float)
                    base_ll, base_brier = losses(y, base_p)
                    base_ece, base_max, base_bins = calibration(y, base_p, weights, 500)
                    for variant in ALL:
                        p = frame[pcol(variant, target, horizon, tier)].to_numpy(float)
                        ll, brier = losses(y, p)
                        ece, maximum, bins = calibration(y, p, weights, 500)
                        ll_mean, base_ll_mean = wmean(ll, weights), wmean(base_ll, weights)
                        rows.append(
                            {
                                "weight_surface": surface,
                                "variant": variant,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "positives": int(y.sum()),
                                "weight_sum": float(weights.sum()),
                                "log_loss": ll_mean,
                                "baseline_log_loss": base_ll_mean,
                                "relative_log_loss_improvement": (base_ll_mean - ll_mean) / base_ll_mean,
                                "log_loss_difference": ll_mean - base_ll_mean,
                                "brier": wmean(brier, weights),
                                "baseline_brier": wmean(base_brier, weights),
                                "brier_difference": wmean(brier - base_brier, weights),
                                "ece": ece,
                                "baseline_ece": base_ece,
                                "maximum_supported_bin_error": maximum,
                                "baseline_maximum_supported_bin_error": base_max,
                                "supported_bins": bins,
                                "baseline_supported_bins": base_bins,
                            }
                        )
    return pd.DataFrame(rows)


def pooled(frame: pd.DataFrame, variant: str, p75_only: bool = False) -> dict[str, float]:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    base_lls: list[float] = []
    candidate_lls: list[float] = []
    base_briers: list[float] = []
    candidate_briers: list[float] = []
    tiers = ("p75",) if p75_only else TIERS
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in tiers:
                y = frame[ycol(target, horizon, tier)].to_numpy(int)
                base_ll, base_brier = losses(y, frame[pcol("baseline", target, horizon, tier)])
                ll, brier = losses(y, frame[pcol(variant, target, horizon, tier)])
                base_lls.append(wmean(base_ll, weights))
                candidate_lls.append(wmean(ll, weights))
                base_briers.append(wmean(base_brier, weights))
                candidate_briers.append(wmean(brier, weights))
    base_ll, candidate_ll = float(np.mean(base_lls)), float(np.mean(candidate_lls))
    base_brier, candidate_brier = float(np.mean(base_briers)), float(np.mean(candidate_briers))
    return {
        "baseline_log_loss": base_ll,
        "log_loss": candidate_ll,
        "relative_log_loss_improvement": (base_ll - candidate_ll) / base_ll,
        "log_loss_difference": candidate_ll - base_ll,
        "baseline_brier": base_brier,
        "brier": candidate_brier,
        "brier_difference": candidate_brier - base_brier,
    }


def row_difference(frame: pd.DataFrame, variant: str, endpoint: str, p75_only=False) -> np.ndarray:
    tiers = ("p75",) if p75_only else TIERS
    denominator = 6 if p75_only else 12
    result = np.zeros(len(frame))
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in tiers:
                y = frame[ycol(target, horizon, tier)].to_numpy(int)
                base = losses(y, frame[pcol("baseline", target, horizon, tier)])[0 if endpoint == "log_loss" else 1]
                candidate = losses(y, frame[pcol(variant, target, horizon, tier)])[0 if endpoint == "log_loss" else 1]
                result += (candidate - base) / denominator
    return result


def daily(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    grouped = pd.DataFrame(
        {
            "date": frame["session_date"].astype(str).to_numpy(),
            "weighted": values * weights,
            "weight": weights,
        }
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
        np.random.default_rng(seed)
        .choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values)))
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


def recall(frame: pd.DataFrame, variant: str, target: str, horizon: int, tier: str) -> float:
    selected = frame[["anchor_id", ycol(target, horizon, tier)]].copy()
    selected["p"] = frame[pcol(variant, target, horizon, tier)].to_numpy(float)
    selected = selected.sort_values(["anchor_id", "p"], ascending=[True, False], kind="stable")
    selected["rank"] = selected.groupby("anchor_id", sort=False).cumcount() + 1
    y = selected[ycol(target, horizon, tier)].to_numpy(int)
    return float(((selected["rank"].to_numpy(int) <= 3) & (y == 1)).sum() / y.sum())


def compare_table(
    calculated: pd.DataFrame,
    stored: pd.DataFrame,
    keys: Sequence[str],
    numeric: Sequence[str],
    tolerance: float = 2e-11,
) -> tuple[bool, float, int]:
    joined = calculated.merge(stored, on=list(keys), suffixes=("__calc", "__stored"), validate="one_to_one")
    if len(joined) != len(calculated) or len(joined) != len(stored):
        return False, math.inf, abs(len(calculated) - len(stored))
    maximum = 0.0
    errors = 0
    for column in numeric:
        left = joined[f"{column}__calc"].to_numpy(float)
        right = joined[f"{column}__stored"].to_numpy(float)
        difference = np.abs(left - right)
        finite = np.isfinite(left) & np.isfinite(right)
        both_infinite = np.isinf(left) & np.isinf(right) & (np.sign(left) == np.sign(right))
        valid = finite | both_infinite
        errors += int((~valid).sum())
        if finite.any():
            maximum = max(maximum, float(difference[finite].max()))
            errors += int((difference[finite] > tolerance).sum())
    return errors == 0, maximum, errors


def run_audit() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    hashes = {
        "contract": sha256(CONTRACT),
        "runner": sha256(RUNNER),
        "v2_runner": sha256(V2_RUNNER),
        "v1_runner": sha256(V1_RUNNER),
        "factor": sha256(FACTOR),
        "quality": sha256(QUALITY),
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
    artifact_mismatch = {
        name: sha256(ROOT / name)
        for name, descriptor in artifact.items()
        if sha256(ROOT / name) != descriptor["sha256"]
    }
    record("artifact_hashes", not artifact_mismatch, artifact_mismatch)

    source, source_audit = load_source()
    join_audit = json.loads((ROOT / "join_audit.json").read_text())
    record(
        "source_join_replay",
        source_audit["identity"]["rows"]
        and all(value for key, value in source_audit["identity"].items() if key != "rows")
        and abs(source_audit["structural"]["correlation"] - join_audit["structural_probability_diagnostic"]["correlation"]) < 1e-14
        and abs(source_audit["structural"]["mean_absolute_difference"] - join_audit["structural_probability_diagnostic"]["mean_absolute_difference"]) < 1e-14,
        source_audit,
    )
    full = add_fixed(source)
    ledger = pd.read_parquet(ROOT / "linkage_predictions_2024_sep_dec.parquet")
    primary = full.loc[full["month"].isin(EVAL_MONTHS)].reset_index(drop=True)
    key_equal = ledger[[*JOIN_KEYS]].reset_index(drop=True).equals(
        primary[[*JOIN_KEYS]].reset_index(drop=True)
    )
    record(
        "primary_key_surface",
        key_equal and len(ledger) == 130672 and ledger["anchor_id"].nunique() == 21341,
        {"rows": len(ledger), "anchors": ledger["anchor_id"].nunique(), "keys_equal": key_equal},
    )
    fixed_columns = [
        *quality_classes(),
        *quality_probabilities(),
        *[ycol(target, horizon, tier) for target in TARGETS for horizon in HORIZONS for tier in TIERS],
        *[pcol(variant, target, horizon, tier) for variant in FIXED for target in TARGETS for horizon in HORIZONS for tier in TIERS],
    ]
    fixed_error = max(
        float(np.max(np.abs(primary[column].to_numpy(float) - ledger[column].to_numpy(float))))
        for column in fixed_columns
    )
    record("fixed_composition_replay", fixed_error < 1e-14, fixed_error)
    prediction_error, parameter_error, fit_count = refit_meta(full, ledger)
    record("meta_probability_refit", prediction_error < 2e-12, prediction_error)
    record("meta_parameter_refit", parameter_error < 2e-12 and fit_count == 96, {"error": parameter_error, "fits": fit_count})

    calculated_cells = cell_table(ledger)
    stored_cells = pd.read_csv(ROOT / "cell_metrics.csv")
    cell_numeric = [
        "rows", "positives", "weight_sum", "log_loss", "baseline_log_loss",
        "relative_log_loss_improvement", "log_loss_difference", "brier",
        "baseline_brier", "brier_difference", "ece", "baseline_ece",
        "maximum_supported_bin_error", "baseline_maximum_supported_bin_error",
        "supported_bins", "baseline_supported_bins",
    ]
    passed, maximum, errors = compare_table(
        calculated_cells,
        stored_cells,
        ["weight_surface", "variant", "target", "horizon", "tier"],
        cell_numeric,
    )
    record("cell_metric_replay", passed, {"max_abs_error": maximum, "errors": errors})

    pooled_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(ALL):
        row = {"variant": variant, "rows": len(ledger), "anchors": ledger["anchor_id"].nunique(), **pooled(ledger, variant)}
        for endpoint_index, endpoint in enumerate(("log_loss", "brier")):
            values = daily(ledger, row_difference(ledger, variant, endpoint))
            lower, upper = bootstrap(values, SEED + variant_index * 100 + endpoint_index)
            p_value = sign_flip(values, SEED + variant_index * 100 + endpoint_index + 10)
            row[f"{endpoint}_daily_sessions"] = len(values)
            row[f"{endpoint}_bootstrap_lower"] = lower
            row[f"{endpoint}_bootstrap_upper"] = upper
            row[f"{endpoint}_sign_flip_p_value"] = p_value
            if variant in CANDIDATES:
                multiplicity_rows.append({"variant": variant, "endpoint": endpoint, "p_value": p_value})
        pooled_rows.append(row)
    calculated_pooled = pd.DataFrame(pooled_rows)
    stored_pooled = pd.read_csv(ROOT / "pooled_metrics.csv")
    pooled_numeric = [column for column in calculated_pooled.columns if column not in {"variant"}]
    passed, maximum, errors = compare_table(
        calculated_pooled, stored_pooled, ["variant"], pooled_numeric
    )
    record("pooled_bootstrap_test_replay", passed, {"max_abs_error": maximum, "errors": errors})
    calculated_multiplicity = holm(pd.DataFrame(multiplicity_rows), ["endpoint"])
    stored_multiplicity = pd.read_csv(ROOT / "multiplicity.csv")
    passed, maximum, errors = compare_table(
        calculated_multiplicity,
        stored_multiplicity,
        ["variant", "endpoint"],
        ["p_value", "holm_adjusted_p"],
    )
    holm_bool = calculated_multiplicity.sort_values(["variant", "endpoint"])["holm_pass"].tolist() == stored_multiplicity.sort_values(["variant", "endpoint"])["holm_pass"].tolist()
    record("Holm_replay", passed and holm_bool, {"max_abs_error": maximum, "errors": errors, "bool": holm_bool})

    temporal_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for variant in ALL:
        for month in EVAL_MONTHS:
            temporal_rows.append({"variant": variant, "slice": month, **pooled(ledger.loc[ledger["month"].eq(month)], variant)})
        for name, months in (
            ("2024-09_to_2024-10", ("2024-09", "2024-10")),
            ("2024-11_to_2024-12", ("2024-11", "2024-12")),
        ):
            temporal_rows.append({"variant": variant, "slice": name, **pooled(ledger.loc[ledger["month"].isin(months)], variant)})
        for symbol in sorted(ledger["symbol_norm"].unique()):
            stock_rows.append({"variant": variant, "deleted_symbol": symbol, **pooled(ledger.loc[ledger["symbol_norm"].ne(symbol)], variant)})
        for (cycle, state), selected in ledger.groupby(["cycle_id", "current_state"], sort=True):
            positives = sum(int(selected[ycol(target, horizon, tier)].sum()) for target in TARGETS for horizon in HORIZONS for tier in TIERS)
            orientation_rows.append(
                {
                    "variant": variant,
                    "cycle_id": cycle,
                    "current_state": int(state),
                    "rows": len(selected),
                    "joint_positives_across_cells": positives,
                    "supported": len(selected) >= 500 and positives >= 30,
                    **pooled(selected, variant),
                }
            )
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    base = recall(ledger, "baseline", target, horizon, tier)
                    value = recall(ledger, variant, target, horizon, tier)
                    ranking_rows.append(
                        {"variant": variant, "target": target, "horizon": horizon, "tier": tier, "top_three_recall": value, "baseline_top_three_recall": base, "gain": value - base}
                    )
    table_specs = [
        ("temporal_slice_replay", pd.DataFrame(temporal_rows), pd.read_csv(ROOT / "temporal_slices.csv"), ["variant", "slice"]),
        ("stock_deletion_replay", pd.DataFrame(stock_rows), pd.read_csv(ROOT / "stock_deletions.csv"), ["variant", "deleted_symbol"]),
        ("orientation_replay", pd.DataFrame(orientation_rows), pd.read_csv(ROOT / "orientation_slices.csv"), ["variant", "cycle_id", "current_state"]),
        ("ranking_replay", pd.DataFrame(ranking_rows), pd.read_csv(ROOT / "ranking.csv"), ["variant", "target", "horizon", "tier"]),
    ]
    for name, calculated, stored, keys in table_specs:
        numeric = [
            column
            for column in calculated.columns
            if column not in set(keys) | {"supported"}
        ]
        passed, maximum, errors = compare_table(calculated, stored, keys, numeric)
        bool_pass = True
        if "supported" in calculated:
            bool_pass = calculated.sort_values(keys)["supported"].tolist() == stored.sort_values(keys)["supported"].tolist()
        record(name, passed and bool_pass, {"max_abs_error": maximum, "errors": errors, "bool": bool_pass})

    gates_stored = json.loads((ROOT / "variant_gates.json").read_text())
    primary_cells = calculated_cells.loc[calculated_cells["weight_surface"].eq("inverse_compatible")]
    temporal = pd.DataFrame(temporal_rows)
    stocks = pd.DataFrame(stock_rows)
    orientations = pd.DataFrame(orientation_rows)
    ranking = pd.DataFrame(ranking_rows)
    gate_errors: list[str] = []
    for variant in CANDIDATES:
        pooled_row = calculated_pooled.set_index("variant").loc[variant]
        cell = primary_cells.loc[primary_cells["variant"].eq(variant)]
        time = temporal.loc[temporal["variant"].eq(variant) & temporal["slice"].isin(EVAL_MONTHS)]
        stock = stocks.loc[stocks["variant"].eq(variant)]
        orientation = orientations.loc[orientations["variant"].eq(variant) & orientations["supported"]]
        rank = ranking.loc[ranking["variant"].eq(variant)]
        multiplicity = calculated_multiplicity.loc[calculated_multiplicity["variant"].eq(variant)].set_index("endpoint")
        checks_here = {
            "pooled_relative_log_loss": pooled_row["relative_log_loss_improvement"] >= 0.005,
            "pooled_brier": pooled_row["brier_difference"] < 0,
            "bootstrap_log_loss": pooled_row["log_loss_bootstrap_upper"] < 0,
            "bootstrap_brier": pooled_row["brier_bootstrap_upper"] < 0,
            "Holm_log_loss": bool(multiplicity.loc["log_loss", "holm_pass"]),
            "Holm_brier": bool(multiplicity.loc["brier", "holm_pass"]),
            "all_cell_losses": bool((cell["log_loss_difference"] <= 0).all() and (cell["brier_difference"] <= 0).all()),
            "all_cell_calibration": bool((cell["ece"] <= cell["baseline_ece"]).all() and (cell["maximum_supported_bin_error"] <= 0.02).all()),
            "every_month": bool((time["log_loss_difference"] < 0).all() and (time["brier_difference"] < 0).all()),
            "every_stock_deletion": bool((stock["log_loss_difference"] <= 0).all() and (stock["brier_difference"] <= 0).all()),
            "zero_supported_orientation_reversals": bool((orientation["log_loss_difference"] <= 0).all() and (orientation["brier_difference"] <= 0).all()),
            "ranking": bool((rank["gain"] >= 0).all()),
        }
        if variant == "dependency_stack":
            raw = calculated_pooled.set_index("variant").loc["raw_full_link"]
            calibrated = calculated_pooled.set_index("variant").loc["calibrated_raw_product"]
            checks_here["below_raw_full_link"] = bool(pooled_row["log_loss"] < raw["log_loss"] and pooled_row["brier"] < raw["brier"])
            checks_here["below_calibrated_raw_product"] = bool(pooled_row["log_loss"] < calibrated["log_loss"] and pooled_row["brier"] < calibrated["brier"])
        if checks_here != gates_stored[variant]["checks"] or bool(all(checks_here.values())) != gates_stored[variant]["pass"]:
            gate_errors.append(variant)
    record("variant_gate_replay", not gate_errors, gate_errors)

    stored_attraction = pd.read_csv(ROOT / "attraction_slices.csv")
    attraction_errors = 0
    attraction_maximum = 0.0
    slice_indices = {
        (cycle, int(state), clock): index
        for index, ((cycle, state, clock), _) in enumerate(
            ledger.groupby(
                ["cycle_id", "current_state", "entry_clock_quartile"], sort=True
            )
        )
    }
    recalculated_attraction_p: list[dict[str, Any]] = []
    for row in stored_attraction.itertuples(index=False):
        selected = ledger.loc[
            ledger["cycle_id"].eq(row.cycle_id)
            & ledger["current_state"].eq(row.current_state)
            & ledger["entry_clock_quartile"].eq(row.entry_clock_quartile)
        ]
        metrics = pooled(selected, "dependency_stack", p75_only=True)
        for column, value in metrics.items():
            difference = abs(float(getattr(row, column)) - float(value))
            attraction_maximum = max(attraction_maximum, difference)
            attraction_errors += int(difference > 2e-11)
        expected_p = sign_flip(
            daily(selected, row_difference(selected, "dependency_stack", "log_loss", p75_only=True)),
            SEED
            + slice_indices[
                (row.cycle_id, int(row.current_state), row.entry_clock_quartile)
            ],
        )
        difference = abs(float(row.p_value) - expected_p)
        attraction_maximum = max(attraction_maximum, difference)
        attraction_errors += int(difference > 2e-11)
        recalculated_attraction_p.append(
            {
                "cycle_id": row.cycle_id,
                "current_state": int(row.current_state),
                "entry_clock_quartile": row.entry_clock_quartile,
                "p_value": expected_p,
            }
        )
    recalculated_attraction = holm(pd.DataFrame(recalculated_attraction_p), [])
    stored_attraction_holm = stored_attraction.loc[
        :, [
            "cycle_id",
            "current_state",
            "entry_clock_quartile",
            "p_value",
            "holm_adjusted_p",
            "holm_pass",
        ]
    ]
    passed_holm, maximum_holm, errors_holm = compare_table(
        recalculated_attraction,
        stored_attraction_holm,
        ["cycle_id", "current_state", "entry_clock_quartile"],
        ["p_value", "holm_adjusted_p"],
    )
    attraction_holm_bool = recalculated_attraction.sort_values(
        ["cycle_id", "current_state", "entry_clock_quartile"]
    )["holm_pass"].tolist() == stored_attraction_holm.sort_values(
        ["cycle_id", "current_state", "entry_clock_quartile"]
    )["holm_pass"].tolist()
    record(
        "attraction_slice_metric_replay",
        attraction_errors == 0
        and passed_holm
        and attraction_holm_bool,
        {
            "max_abs_error": max(attraction_maximum, maximum_holm),
            "errors": attraction_errors + errors_holm,
            "Holm_bool": attraction_holm_bool,
        },
    )

    decision = json.loads((ROOT / "decision.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    record(
        "decision_replay",
        all(not gates_stored[variant]["pass"] for variant in CANDIDATES)
        and decision["passing_variants"] == []
        and decision["selected_variant"] is None
        and decision["qualified_attraction_slices"] == []
        and decision["named_loop_good_or_high_promoted"] is False,
        decision,
    )
    record(
        "summary_and_safety_reconciliation",
        summary["decision"] == decision
        and summary["primary_rows"] == len(ledger)
        and summary["meta_fits"] == 96
        and summary["direct_volume_fields_used"] == []
        and summary["direction_or_signed_return_used"] is False
        and summary["later_period_scoring_performed"] is False
        and summary["prospective_shadow_read_or_write_performed"] is False,
        {
            "rows": summary["primary_rows"],
            "fits": summary["meta_fits"],
            "volume": summary["direct_volume_fields_used"],
            "later": summary["later_period_scoring_performed"],
        },
    )
    all_passed = all(value["passed"] for value in checks.values())
    result = {
        "audit_id": "regime_loop_linkage_ideas_v3_independent_audit",
        "all_passed": all_passed,
        "checks_passed": sum(value["passed"] for value in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "auditor_imported_linkage_runner": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    (ROOT / "independent_audit.json").write_text(
        json.dumps(safe(result), indent=2, sort_keys=True) + "\n"
    )
    if not all_passed:
        raise AssertionError(
            "independent linkage audit failed: "
            + str([name for name, value in checks.items() if not value["passed"]])
        )
    print(json.dumps(safe(result), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_audit()
