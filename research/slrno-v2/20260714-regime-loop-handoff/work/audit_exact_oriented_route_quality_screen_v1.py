"""Independent audit for the 2024 exact-oriented-route quality screen.

This auditor deliberately imports neither the screen runner nor the parent
quality runner.  It independently reconstructs the 45-route dictionary,
labels, overlap weights, causal qexact OOF predictions, decisive calibration
and non-inferiority gates, sign-flip/Holm family, and zero-candidate decision.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-exact-oriented-route-quality-screen-v1.json"
RUNNER = HERE / "run_exact_oriented_route_quality_screen_v1.py"
PARENT_CONTRACT = HERE / "contracts/20260710-per-loop-movement-quality-v1.json"
ANCHOR = Path(
    "/private/tmp/stocker_frozen_loop_price_consequence_20260710/anchor_panel_train_2024.parquet"
)
PARENT_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
PARENT_OOF = PARENT_ROOT / "oof_predictions_2024.parquet"
CYCLES = PARENT_ROOT / "fixed_cycles.csv"
ROOT = Path("/private/tmp/stocker_exact_oriented_route_quality_screen_v1_20260711")

CONTRACT_SHA256 = "858b02722ba1a4f6fe487977971510209edbcb3e9fc2f8eaf93034e1ef50bed2"
RUNNER_SHA256 = "062ceec9557b11893b8ded9d73277977a71897cd86e67d32f78cbb5eacdd7b6e"
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
NUMERIC = (
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
K = 8
ROUTE_COUNT = 45
ROUTE_SCALE = 0.5
CONTEXT_WIDTH = K + len(NUMERIC)
SEED = 20260711
OOF_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
EPSILON = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def paths(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current),)
            for index, state in enumerate(core)
            if int(state) == int(current)
        }
    )


def manifest(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cycle in cycles.sort_values("cycle_index", kind="stable").itertuples(
        index=False
    ):
        closed = tuple(int(value) for value in str(cycle.cycle).split("->"))
        core = closed[:-1]
        if closed[0] != closed[-1] or len(core) != int(cycle.transition_length):
            raise AssertionError("invalid frozen cycle")
        for current in sorted(set(core)):
            alternatives = paths(core, current)
            for path in alternatives:
                path_text = "->".join(map(str, path))
                split = len(alternatives) > 1
                route_id = f"{cycle.cycle_id}@state_{current}"
                if split:
                    route_id += "__path_" + "_".join(map(str, path))
                rows.append(
                    {
                        "route_id": route_id,
                        "cycle_index": int(cycle.cycle_index),
                        "cycle_id": str(cycle.cycle_id),
                        "cycle": str(cycle.cycle),
                        "current_state": int(current),
                        "transition_length": int(cycle.transition_length),
                        "exact_path": path_text,
                        "path_tuple": path,
                        "route_kind": "split_exact" if split else "exact_route",
                        "structural_probability_available": not split,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["cycle_index", "current_state", "exact_path"], kind="stable"
    )
    frame["route_index"] = np.arange(len(frame), dtype=int)
    if len(frame) != ROUTE_COUNT or frame["route_id"].duplicated().any():
        raise AssertionError("independent route dictionary mismatch")
    return frame.reset_index(drop=True)


def path_label(frame: pd.DataFrame, path: tuple[int, ...]) -> np.ndarray:
    label = np.ones(len(frame), dtype=bool)
    for step, destination in enumerate(path[1:], start=1):
        label &= frame[f"future_state_{step}"].to_numpy(int) == int(destination)
    return label


def parent_columns() -> list[str]:
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "cycle_index",
        "cycle_id",
        "state",
        "loop_probability",
        "first_order_probability",
        "loop_occurs",
        "conditional_weight",
    ]
    for target in TARGETS:
        for horizon in HORIZONS:
            columns.append(f"quality_class__{target}__h{horizon}")
            for model in ("qcontext", "qcycle"):
                for tier in TIERS:
                    columns.append(f"{model}__{target}__h{horizon}__{tier}")
    return columns


def anchor_columns() -> list[str]:
    return [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "state",
        "future_state_1",
        "future_state_2",
        "future_state_3",
        "future_state_4",
        *NUMERIC,
        *(f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS),
    ]


def reconstruct_oof(
    parent: pd.DataFrame, anchors: pd.DataFrame, routes: pd.DataFrame
) -> pd.DataFrame:
    features = anchors.loc[
        :,
        [
            "anchor_id",
            "future_state_1",
            "future_state_2",
            "future_state_3",
            "future_state_4",
            *NUMERIC,
        ],
    ]
    parent = parent.copy()
    parent["parent_row_id"] = np.arange(len(parent), dtype=int)
    parent["validation_month"] = pd.to_datetime(parent["session_date"]).dt.strftime(
        "%Y-%m"
    )
    merged = parent.merge(features, on="anchor_id", validate="many_to_one")
    rows: list[pd.DataFrame] = []
    for route in routes.itertuples(index=False):
        selected = merged.loc[
            merged["cycle_id"].eq(route.cycle_id)
            & merged["state"].eq(route.current_state)
        ].copy()
        occurrence = path_label(selected, tuple(route.path_tuple))
        selected["route_id"] = route.route_id
        selected["route_index"] = int(route.route_index)
        selected["exact_path"] = route.exact_path
        selected["route_kind"] = route.route_kind
        selected["structural_probability_available"] = bool(
            route.structural_probability_available
        )
        selected["exact_route_occurs"] = occurrence.astype(np.int8)
        selected["exact_conditional_weight"] = np.where(
            occurrence, selected["conditional_weight"], 0.0
        )
        rows.append(selected)
    exact = pd.concat(rows, ignore_index=True).sort_values(
        ["parent_row_id", "route_index"], kind="stable"
    )
    exact = exact.reset_index(drop=True)
    for column in list(exact.columns):
        if column.startswith("qcycle__"):
            exact = exact.rename(columns={column: "parent_" + column})
    return exact


def full_conditional(
    anchors: pd.DataFrame, routes: pd.DataFrame, parent_contract: dict[str, Any]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for route in routes.itertuples(index=False):
        selected = anchors.loc[anchors["state"].eq(route.current_state)].copy()
        selected = selected.loc[path_label(selected, tuple(route.path_tuple))].copy()
        selected["route_id"] = route.route_id
        selected["route_index"] = route.route_index
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["anchor_id", "route_index"], kind="stable"
    )
    result = result.reset_index(drop=True)
    count = result.groupby("anchor_id", sort=False)["route_id"].transform("size")
    result["exact_conditional_weight"] = 1.0 / count.to_numpy(float)
    result["month"] = pd.to_datetime(result["session_date"]).dt.strftime("%Y-%m")
    thresholds = parent_contract["outcomes"]["thresholds_bps"]
    for target in TARGETS:
        for horizon in HORIZONS:
            p75 = float(thresholds[target][str(horizon)]["p75"])
            p90 = float(thresholds[target][str(horizon)]["p90"])
            observed = result[f"{target}_{horizon}"].to_numpy(float)
            result[f"quality_class__{target}__h{horizon}"] = np.where(
                observed > p90, 2, np.where(observed > p75, 1, 0)
            ).astype(np.int8)
    return result


def medians(frame: pd.DataFrame) -> dict[str, float]:
    unique = frame.drop_duplicates("anchor_id")
    return {
        column: float(pd.to_numeric(unique[column], errors="coerce").median())
        for column in NUMERIC
    }


def context(frame: pd.DataFrame, fill: dict[str, float]) -> sparse.csr_matrix:
    numeric = frame.loc[:, list(NUMERIC)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(pd.Series(fill)).to_numpy(np.float64)
    state = frame["state"].to_numpy(int)
    return sparse.hstack(
        (
            sparse.csr_matrix(np.eye(K, dtype=float)[state]),
            sparse.csr_matrix(numeric),
        ),
        format="csr",
    )


def design(scaled: sparse.csr_matrix, route_index: np.ndarray) -> sparse.csr_matrix:
    route_index = np.asarray(route_index, int)
    route = sparse.csr_matrix(
        (
            np.full(len(route_index), ROUTE_SCALE),
            (np.arange(len(route_index)), route_index),
        ),
        shape=(len(route_index), ROUTE_COUNT),
    )
    return sparse.hstack((scaled, route), format="csr")


def replay_probabilities(
    saved: pd.DataFrame, training: pd.DataFrame
) -> tuple[float, list[dict[str, Any]]]:
    maximum = 0.0
    audit: list[dict[str, Any]] = []
    for month in OOF_MONTHS:
        train = training.loc[training["month"].lt(month)].copy()
        validation = saved.loc[saved["validation_month"].eq(month)].copy()
        fill = medians(train)
        train_raw = context(train, fill)
        validation_raw = context(validation, fill)
        weights = train["exact_conditional_weight"].to_numpy(float)
        scaler = StandardScaler(with_mean=False).fit(train_raw, sample_weight=weights)
        train_x = design(
            scaler.transform(train_raw).tocsr(), train["route_index"].to_numpy(int)
        )
        validation_x = design(
            scaler.transform(validation_raw).tocsr(),
            validation["route_index"].to_numpy(int),
        )
        for target in TARGETS:
            for horizon in HORIZONS:
                observed = train[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                model = LogisticRegression(
                    C=0.2,
                    solver="lbfgs",
                    max_iter=1000,
                    tol=0.0001,
                    random_state=SEED,
                ).fit(train_x, observed, sample_weight=weights)
                probability = model.predict_proba(validation_x)
                for tier, expected in (
                    ("p75", probability[:, 1] + probability[:, 2]),
                    ("p90", probability[:, 2]),
                ):
                    stored = validation[
                        f"qexact__{target}__h{horizon}__{tier}"
                    ].to_numpy(float)
                    error = float(np.max(np.abs(expected - stored)))
                    maximum = max(maximum, error)
                audit.append(
                    {
                        "month": month,
                        "target": target,
                        "horizon": horizon,
                        "n_iter": int(model.n_iter_[0]),
                    }
                )
    return maximum, audit


def binary_losses(y: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(y, float)
    q = np.clip(np.asarray(q, float), EPSILON, 1 - EPSILON)
    return {
        "log_loss": -(y * np.log(q) + (1 - y) * np.log1p(-q)),
        "brier": np.square(q - y),
    }


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, float), weights=np.asarray(weights, float)))


def calibration(
    observed: np.ndarray, probability: np.ndarray, weights: np.ndarray, minimum: int
) -> tuple[float, float]:
    observed = np.asarray(observed, float)
    probability = np.asarray(probability, float)
    weights = np.asarray(weights, float)
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    errors: list[float] = []
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        weight = float(weights[mask].sum())
        if weight <= 0:
            continue
        error = abs(wmean(observed[mask], weights[mask]) - wmean(probability[mask], weights[mask]))
        ece += weight / weights.sum() * error
        if count >= minimum:
            errors.append(error)
    return float(ece), float(max(errors)) if errors else math.nan


def support(frame: pd.DataFrame, contract: dict[str, Any]) -> bool:
    rule = contract["support_gates_each_route"]
    realized = frame.loc[frame["exact_route_occurs"].eq(1)]
    quarters = realized["quarter"].astype(str).value_counts()
    return bool(
        len(frame) >= rule["compatible_rows_minimum"]
        and len(realized) >= rule["realized_rows_minimum"]
        and realized["symbol_norm"].nunique() >= rule["realized_stocks_minimum"]
        and set(quarters.index) == set(rule["required_quarters"])
        and all(
            int(quarters.get(value, 0)) >= rule["realized_rows_each_quarter_minimum"]
            for value in rule["required_quarters"]
        )
    )


def structural(frame: pd.DataFrame) -> bool:
    y = frame["exact_route_occurs"].to_numpy(int)
    history = frame["loop_probability"].to_numpy(float)
    first = frame["first_order_probability"].to_numpy(float)
    history_loss = binary_losses(y, history)
    first_loss = binary_losses(y, first)
    history_ece, history_max = calibration(y, history, np.ones(len(frame)), 250)
    first_ece, first_max = calibration(y, first, np.ones(len(frame)), 250)
    return bool(
        history_loss["log_loss"].mean() < first_loss["log_loss"].mean()
        and history_loss["brier"].mean() < first_loss["brier"].mean()
        and history_ece <= first_ece
        and np.isfinite(history_max)
        and np.isfinite(first_max)
        and history_max <= first_max + 0.01
    )


def arrays(
    frame: pd.DataFrame, target: str, horizon: int, tier: str, surface: str
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    threshold = 1 if tier == "p75" else 2
    if surface == "conditional":
        selected = frame.loc[frame["exact_route_occurs"].eq(1)]
        y = (
            selected[f"quality_class__{target}__h{horizon}"].to_numpy(int)
            >= threshold
        ).astype(int)
        weights = selected["exact_conditional_weight"].to_numpy(float)
        probability = {
            "qcontext": selected[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float),
            "parent": selected[f"parent_qcycle__{target}__h{horizon}__{tier}"].to_numpy(float),
            "qexact": selected[f"qexact__{target}__h{horizon}__{tier}"].to_numpy(float),
        }
    else:
        yclass = frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
        y = (frame["exact_route_occurs"].to_numpy(bool) & (yclass >= threshold)).astype(int)
        weights = np.ones(len(frame))
        structural_probability = frame["loop_probability"].to_numpy(float)
        probability = {
            "qcontext": structural_probability
            * frame[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float),
            "parent": structural_probability
            * frame[f"parent_qcycle__{target}__h{horizon}__{tier}"].to_numpy(float),
            "qexact": structural_probability
            * frame[f"qexact__{target}__h{horizon}__{tier}"].to_numpy(float),
        }
    return y, weights, probability


def strict_pass(
    frame: pd.DataFrame, target: str, horizon: int, tier: str, contract: dict[str, Any]
) -> bool:
    rule = contract["quality_gates_each_route_target_horizon_tier"]
    cal = rule["calibration"]
    noninf = rule["qexact_noninferiority_to_parent_qcycle"]
    passed = True
    for surface in ("conditional", "joint"):
        y, weights, probabilities = arrays(frame, target, horizon, tier, surface)
        minimum = cal[
            "conditional_supported_bin_rows_minimum"
            if surface == "conditional"
            else "joint_supported_bin_rows_minimum"
        ]
        metrics = {
            model: calibration(y, probability, weights, minimum)
            for model, probability in probabilities.items()
        }
        candidate_ece, candidate_max = metrics["qexact"]
        absolute = cal[
            "conditional_absolute_maximum_supported_bin_error"
            if surface == "conditional"
            else "joint_absolute_maximum_supported_bin_error"
        ]
        cal_pass = bool(
            candidate_ece <= metrics["qcontext"][0]
            and candidate_ece <= metrics["parent"][0]
            and np.isfinite(candidate_max)
            and candidate_max <= absolute
        )
        losses = {
            model: binary_losses(y, probability)
            for model, probability in probabilities.items()
        }
        candidate_ll = wmean(losses["qexact"]["log_loss"], weights)
        parent_ll = wmean(losses["parent"]["log_loss"], weights)
        candidate_brier = wmean(losses["qexact"]["brier"], weights)
        parent_brier = wmean(losses["parent"]["brier"], weights)
        ll_limit = noninf[
            "conditional_relative_log_loss_degradation_maximum"
            if surface == "conditional"
            else "joint_relative_log_loss_degradation_maximum"
        ]
        brier_limit = noninf[
            "conditional_brier_difference_maximum"
            if surface == "conditional"
            else "joint_brier_difference_maximum"
        ]
        passed &= bool(
            cal_pass
            and (candidate_ll - parent_ll) / parent_ll <= ll_limit
            and candidate_brier - parent_brier <= brier_limit
        )
    return bool(passed)


def daily_difference(
    frame: pd.DataFrame, target: str, horizon: int, tier: str
) -> pd.Series:
    selected = frame.loc[frame["exact_route_occurs"].eq(1)]
    threshold = 1 if tier == "p75" else 2
    y = (
        selected[f"quality_class__{target}__h{horizon}"].to_numpy(int) >= threshold
    ).astype(int)
    weights = selected["exact_conditional_weight"].to_numpy(float)
    candidate = binary_losses(
        y, selected[f"qexact__{target}__h{horizon}__{tier}"]
    )["log_loss"]
    baseline = binary_losses(
        y, selected[f"qcontext__{target}__h{horizon}__{tier}"]
    )["log_loss"]
    daily = pd.DataFrame(
        {
            "date": selected["session_date"].astype(str),
            "weighted": (candidate - baseline) * weights,
            "weight": weights,
        }
    ).groupby("date", sort=True).sum()
    return daily["weighted"] / daily["weight"]


def pvalue(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(9999, len(values)))
    null = signs @ values / len(values)
    return float((1 + np.sum(null <= values.mean())) / 10000)


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["holm_adjusted_p"] = np.nan
    result["holm_pass"] = False
    for _, positions in result.groupby("tier", sort=True).groups.items():
        ordered = sorted(positions, key=lambda position: result.loc[position, "p_value"])
        running = 0.0
        for rank, position in enumerate(ordered, start=1):
            adjusted = min(
                1.0,
                max(running, (len(ordered) - rank + 1) * result.loc[position, "p_value"]),
            )
            running = adjusted
            result.loc[position, "holm_adjusted_p"] = adjusted
            result.loc[position, "holm_pass"] = adjusted <= 0.05
            result.loc[position, "holm_rank"] = rank
            result.loc[position, "family_size"] = len(ordered)
    return result


def compare_frame(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    tolerance: float = 1e-12,
) -> float:
    left = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    right = observed.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise AssertionError(f"frame keys differ: {keys}")
    maximum = 0.0
    for column in columns:
        if pd.api.types.is_bool_dtype(left[column]) or pd.api.types.is_object_dtype(left[column]):
            if not left[column].astype(str).equals(right[column].astype(str)):
                raise AssertionError(f"frame column differs: {column}")
        else:
            a = left[column].to_numpy(float)
            b = right[column].to_numpy(float)
            error = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
            maximum = max(maximum, error)
            if not np.allclose(a, b, atol=tolerance, rtol=0, equal_nan=True):
                raise AssertionError(f"numeric frame column differs: {column}, {error}")
    return maximum


def audit(root: Path = ROOT) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": clean(details)})
        if not passed:
            raise AssertionError(name)

    contract = json.loads(CONTRACT.read_text())
    parent_contract = json.loads(PARENT_CONTRACT.read_text())
    record("contract_hash_exact", sha256(CONTRACT) == CONTRACT_SHA256)
    record("runner_hash_exact", sha256(RUNNER) == RUNNER_SHA256)
    tree = ast.parse(Path(__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    record(
        "auditor_has_no_production_import",
        not any(name.startswith("work.") for name in imported),
        imported,
    )
    record(
        "safety_and_phase_exact",
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["period_and_phase_lock"]["fit_and_evaluate"] == "2024_only"
        and contract["decision_and_stop_rules"]["later_period_scoring"] is False,
    )

    artifact_manifest = json.loads((root / "artifact_manifest.json").read_text())
    mismatches = []
    for name, specification in artifact_manifest["artifacts"].items():
        path = root / name
        if path.stat().st_size != specification["size"] or sha256(path) != specification["sha256"]:
            mismatches.append(name)
    record("all_declared_artifact_hashes_exact", not mismatches, mismatches)

    cycles = pd.read_csv(CYCLES)
    routes = manifest(cycles)
    stored_manifest = pd.read_csv(root / "route_manifest.csv")
    manifest_error = compare_frame(
        routes.drop(columns="path_tuple"),
        stored_manifest,
        ["route_id"],
        [
            "cycle_index",
            "cycle_id",
            "current_state",
            "transition_length",
            "exact_path",
            "route_kind",
            "structural_probability_available",
            "route_index",
        ],
    )
    record("exact_45_route_manifest_independently_exact", manifest_error == 0)

    parent = pd.read_parquet(PARENT_OOF, columns=parent_columns())
    anchors = pd.read_parquet(ANCHOR, columns=anchor_columns())
    expected = reconstruct_oof(parent, anchors, routes)
    saved = pd.read_parquet(root / "exact_route_oof_predictions.parquet")
    key_columns = ["parent_row_id", "route_id"]
    reconstruction_error = compare_frame(
        expected,
        saved,
        key_columns,
        [
            "route_index",
            "exact_path",
            "route_kind",
            "exact_route_occurs",
            "exact_conditional_weight",
        ],
    )
    record(
        "exact_labels_and_weights_independently_exact",
        reconstruction_error <= 1e-12
        and len(saved) == 221894
        and int(saved["exact_route_occurs"].sum()) == 15584,
        {"rows": len(saved), "realized": int(saved["exact_route_occurs"].sum())},
    )
    split = saved.loc[saved["route_kind"].eq("split_exact")]
    record(
        "cycle_15_split_exhaustive_exclusive",
        len(split) == 10912
        and int(split["exact_route_occurs"].sum()) == 266
        and int(
            split.groupby("parent_row_id")["exact_route_occurs"].sum().max()
        )
        <= 1,
    )

    training = full_conditional(anchors, routes, parent_contract)
    record(
        "full_2024_exact_training_population_exact",
        len(training) == 32677
        and np.allclose(
            training.groupby("anchor_id")["exact_conditional_weight"].sum(), 1.0
        ),
    )
    probability_error, fit_audit = replay_probabilities(saved, training)
    record(
        "all_36_qexact_oof_fits_independently_replayed",
        probability_error <= 1e-10 and len(fit_audit) == 36,
        {"maximum_probability_error": probability_error, "fits": len(fit_audit)},
    )

    stored_support = pd.read_csv(root / "route_support.csv").set_index("route_id")
    support_map: dict[str, bool] = {}
    structural_map: dict[str, bool] = {}
    stored_structural = pd.read_csv(root / "route_structural.csv").set_index("route_id")
    for route_id, frame in saved.groupby("route_id", sort=True):
        support_map[route_id] = support(frame, contract)
        route = routes.set_index("route_id").loc[route_id]
        structural_map[route_id] = bool(
            route["structural_probability_available"] and structural(frame)
        )
    record(
        "all_route_support_gates_independently_exact",
        all(bool(stored_support.loc[key, "pass"]) == value for key, value in support_map.items()),
        {"passing": sum(support_map.values())},
    )
    record(
        "all_route_structural_gates_independently_exact",
        all(bool(stored_structural.loc[key, "pass"]) == value for key, value in structural_map.items()),
        {"passing": sum(structural_map.values())},
    )

    stored_cells = pd.read_csv(root / "route_quality_cells.csv").set_index(
        ["route_id", "target", "horizon", "tier"]
    )
    strict_map: dict[tuple[str, str, int, str], bool] = {}
    multiplicity_rows: list[dict[str, Any]] = []
    for route_position, (route_id, frame) in enumerate(saved.groupby("route_id", sort=True)):
        if not support_map[route_id]:
            continue
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier_index, tier in enumerate(TIERS):
                    daily = daily_difference(frame, target, horizon, tier)
                    multiplicity_rows.append(
                        {
                            "route_id": route_id,
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            "daily_sessions": len(daily),
                            "observed_daily_mean": float(daily.mean()),
                            "p_value": pvalue(
                                daily.to_numpy(float),
                                SEED
                                + route_position * 1000
                                + TARGETS.index(target) * 100
                                + horizon * 3
                                + tier_index,
                            ),
                        }
                    )
                    route_available = bool(
                        routes.set_index("route_id").loc[
                            route_id, "structural_probability_available"
                        ]
                    )
                    if route_available:
                        key = (route_id, target, horizon, tier)
                        strict_map[key] = strict_pass(
                            frame, target, horizon, tier, contract
                        )
    record(
        "all_decisive_strict_gates_independently_exact",
        all(
            bool(stored_cells.loc[key, "strict_calibration_and_qcycle_noninferiority_pass"])
            == value
            for key, value in strict_map.items()
        ),
        {"passing": sum(strict_map.values()), "evaluated": len(strict_map)},
    )
    independently_holm = holm(pd.DataFrame(multiplicity_rows))
    stored_holm = pd.read_csv(root / "multiplicity.csv")
    holm_error = compare_frame(
        independently_holm,
        stored_holm,
        ["route_id", "target", "horizon", "tier"],
        [
            "daily_sessions",
            "observed_daily_mean",
            "p_value",
            "holm_adjusted_p",
            "holm_pass",
            "holm_rank",
            "family_size",
        ],
        tolerance=1e-12,
    )
    record(
        "all_sign_flip_and_Holm_results_independently_exact",
        holm_error <= 1e-12,
        {"maximum_error": holm_error},
    )

    holm_lookup = independently_holm.set_index(
        ["route_id", "target", "horizon", "tier"]
    )["holm_pass"].to_dict()
    decisive_failure = []
    for route in routes.itertuples(index=False):
        for horizon in HORIZONS:
            decisive = all(
                strict_map.get((route.route_id, target, horizon, "p75"), False)
                and holm_lookup.get((route.route_id, target, horizon, "p75"), False)
                for target in TARGETS
            )
            decisive_failure.append(not decisive)
    stored_grades = pd.read_csv(root / "route_horizon_grades.csv")
    record(
        "zero_candidate_rejection_independently_sufficient",
        all(decisive_failure)
        and len(stored_grades) == 135
        and stored_grades["grade"].eq("development_unqualified").all(),
        {"route_horizons": len(decisive_failure)},
    )
    decision = json.loads((root / "decision.json").read_text())
    record(
        "decision_and_safety_exact",
        decision["candidate_count"] == 0
        and decision["candidate_ids"] == []
        and decision["label"] == "no_exact_route_quality_screen_candidate"
        and decision["later_period_scoring_performed"] is False
        and decision["further_refinement_performed"] is False
        and decision["research_only"] is True
        and decision["live_ordering_enabled"] is False
        and decision["order_placement"] == "disabled",
    )
    completion = json.loads((root / "screen_complete.json").read_text())
    record(
        "completion_has_no_later_or_shadow_access",
        completion["later_period_paths_resolved"] is False
        and completion["later_period_rows_read"] is False
        and completion["shadow_tree_read"] is False
        and completion["shadow_tree_written"] is False
        and completion["historical_volume_used"] is False,
    )

    result = {
        "phase": "exact_oriented_route_quality_screen_v1_independent_2024_audit",
        "all_passed": all(item["passed"] for item in checks),
        "check_count": len(checks),
        "checks": checks,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "auditor_source_sha256": sha256(Path(__file__)),
        "maximum_qexact_probability_replay_error": probability_error,
        "candidate_count": 0,
        "rejection_verified": True,
        "further_refinement_authorized": False,
        "later_period_paths_resolved": False,
        "later_period_rows_read": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "historical_volume_used": False,
        "volume_label": "historical_volume_not_used",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(root / "independent_audit.json", result)
    return result


def self_test() -> dict[str, Any]:
    values = np.asarray([-0.2, -0.1, -0.3] * 4)
    checks = {
        "pvalue_deterministic": pvalue(values, 5) == pvalue(values, 5),
        "holm_monotone": bool(
            np.all(
                np.diff(
                    holm(
                        pd.DataFrame(
                            {
                                "tier": ["p75"] * 3,
                                "p_value": [0.001, 0.02, 0.04],
                            }
                        )
                    )
                    .sort_values("p_value")["holm_adjusted_p"]
                    .to_numpy(float)
                )
                >= -1e-15
            )
        ),
        "no_production_import": ("from " + "work")
        not in Path(__file__).read_text(),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"all_passed": True, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not args.audit:
        raise SystemExit("use --audit or --self-test")
    print(json.dumps(clean(audit(args.root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
