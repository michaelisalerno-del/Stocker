"""Causal 2024-only exact-oriented-route movement-quality screen.

This is a post-inspection development screen.  It may freeze a smaller set of
exact route/horizon hypotheses, but it cannot certify quality, enter a shadow,
or support a direction, P&L, economic-edge, tradability, or deployment claim.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from work.run_per_loop_movement_quality import (
    HORIZONS,
    K,
    NUMERIC_CONTROLS,
    TARGETS,
    TIERS,
    _calibration_summary,
    _quality_cell_gate,
    _structural_gate,
    binary_losses,
    safe,
    weighted_mean,
)


CONTRACT = HERE / "contracts/20260711-exact-oriented-route-quality-screen-v1.json"
CONTRACT_SHA256 = "858b02722ba1a4f6fe487977971510209edbcb3e9fc2f8eaf93034e1ef50bed2"
PARENT_CONTRACT = HERE / "contracts/20260710-per-loop-movement-quality-v1.json"
ANCHOR_2024 = Path(
    "/private/tmp/stocker_frozen_loop_price_consequence_20260710/anchor_panel_train_2024.parquet"
)
PARENT_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
PARENT_OOF = PARENT_ROOT / "oof_predictions_2024.parquet"
CYCLES = PARENT_ROOT / "fixed_cycles.csv"
PARENT_AUDIT = PARENT_ROOT / "pre_score_audit.json"
OUT = Path("/private/tmp/stocker_exact_oriented_route_quality_screen_v1_20260711")

SEED = 20260711
OOF_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
ROUTE_SCALE = 0.5
ROUTE_COUNT = 45
CONTEXT_WIDTH = K + len(NUMERIC_CONTROLS)
QEXACT_WIDTH = CONTEXT_WIDTH + ROUTE_COUNT
EPSILON = 1e-12

EXPECTED_INPUT_HASHES = {
    "anchor_panel_2024": "788fd81909d1c5d3e6ee20e3e36e3ebb74199188e41052ea1b04f61c96fa9932",
    "parent_oof_2024": "689b8853ec482c07a46faea48f49665df8c92612ef28bc9934fe2df2e97e7d30",
    "fixed_cycles": "bf9292fa51de1e545e5a319fa2e2faf2088926acd5315b9106597b1da318b253",
    "parent_contract": "67d64c463df52f01f360561ef0a69d5772b7eec0409468c93d6eb5a630dee02e",
    "parent_runner": "7da5e88e603583d3dba7422569bc8e27837171c7165e69bcaafade472738e2ea",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def load_contract() -> dict[str, Any]:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise AssertionError("exact-route screen contract changed")
    contract = json.loads(CONTRACT.read_text())
    checks = {
        "id": contract.get("contract_id")
        == "exact_oriented_route_quality_screen_v1",
        "research": contract.get("research_only") is True,
        "live": contract.get("live_ordering_enabled") is False,
        "orders": contract.get("order_placement") == "disabled",
        "period": contract["period_and_phase_lock"].get("fit_and_evaluate")
        == "2024_only",
        "later": contract["period_and_phase_lock"].get(
            "later_period_paths_permitted"
        )
        is False,
        "shadow": contract["period_and_phase_lock"].get(
            "prospective_shadow_read_or_write_permitted"
        )
        is False,
        "route_count": contract["exact_route_dictionary"].get("exact_units")
        == ROUTE_COUNT,
    }
    if not all(checks.values()):
        raise AssertionError(f"screen safety/semantic contract failure: {checks}")
    return contract


def deduplicated_oriented_paths(
    core: tuple[int, ...], current_state: int
) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current_state),)
            for index, state in enumerate(core)
            if int(state) == int(current_state)
        }
    )


def build_exact_route_manifest(cycles: pd.DataFrame) -> pd.DataFrame:
    required = {"cycle_index", "cycle_id", "cycle", "transition_length"}
    if not required.issubset(cycles.columns) or len(cycles) != 20:
        raise AssertionError("frozen cycle dictionary changed")
    rows: list[dict[str, Any]] = []
    for cycle in cycles.sort_values("cycle_index", kind="stable").itertuples(
        index=False
    ):
        closed = tuple(int(value) for value in str(cycle.cycle).split("->"))
        if closed[0] != closed[-1] or len(closed) - 1 != int(
            cycle.transition_length
        ):
            raise AssertionError(f"invalid frozen cycle: {cycle.cycle}")
        core = closed[:-1]
        for current_state in sorted(set(core)):
            paths = deduplicated_oriented_paths(core, current_state)
            for path in paths:
                path_string = "->".join(str(value) for value in path)
                split = len(paths) > 1
                route_id = f"{cycle.cycle_id}@state_{current_state}"
                if split:
                    route_id += "__path_" + "_".join(str(value) for value in path)
                rows.append(
                    {
                        "route_id": route_id,
                        "cycle_index": int(cycle.cycle_index),
                        "cycle_id": str(cycle.cycle_id),
                        "cycle": str(cycle.cycle),
                        "current_state": int(current_state),
                        "transition_length": int(cycle.transition_length),
                        "exact_path": path_string,
                        "path_tuple": tuple(int(value) for value in path),
                        "route_kind": "split_exact" if split else "exact_route",
                        "structural_probability_available": not split,
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["cycle_index", "current_state", "exact_path"], kind="stable"
    )
    result["route_index"] = np.arange(len(result), dtype=int)
    if len(result) != ROUTE_COUNT or result["route_id"].duplicated().any():
        raise AssertionError("exact route manifest did not produce 45 unique routes")
    split = result.loc[result["route_kind"].eq("split_exact")]
    if split["route_id"].tolist() != [
        "cycle_15@state_1__path_1_2_1_3_1",
        "cycle_15@state_1__path_1_3_1_2_1",
    ]:
        raise AssertionError("unexpected exact-route split")
    return result.reset_index(drop=True)


def parent_probability_columns() -> list[str]:
    columns: list[str] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            columns.append(f"quality_class__{target}__h{horizon}")
            for model in ("qcontext", "qcycle"):
                for tier in TIERS:
                    columns.append(f"{model}__{target}__h{horizon}__{tier}")
    return columns


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    observed_hashes = {
        "anchor_panel_2024": sha256(ANCHOR_2024),
        "parent_oof_2024": sha256(PARENT_OOF),
        "fixed_cycles": sha256(CYCLES),
        "parent_contract": sha256(PARENT_CONTRACT),
        "parent_runner": sha256(HERE / "run_per_loop_movement_quality.py"),
    }
    if observed_hashes != EXPECTED_INPUT_HASHES:
        raise AssertionError(
            f"frozen parent source changed: expected={EXPECTED_INPUT_HASHES}, actual={observed_hashes}"
        )
    parent_audit = json.loads(PARENT_AUDIT.read_text())
    if parent_audit.get("all_passed") is not True:
        raise AssertionError("parent independent audit is not passing")

    cycles = pd.read_csv(CYCLES)
    parent_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "cycle_index",
        "cycle_id",
        "state",
        "history_token",
        "loop_probability",
        "first_order_probability",
        "loop_occurs",
        "positive_cycle_count",
        "conditional_weight",
        *parent_probability_columns(),
    ]
    parent = pd.read_parquet(PARENT_OOF, columns=parent_columns)
    parent["parent_row_id"] = np.arange(len(parent), dtype=int)
    dates = pd.to_datetime(parent["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {2024}:
        raise AssertionError("parent OOF crossed the 2024 boundary")
    parent["validation_month"] = dates.dt.strftime("%Y-%m")
    if set(parent["validation_month"].unique()) != set(OOF_MONTHS):
        raise AssertionError("parent OOF month schedule changed")
    if len(parent) != 216438 or parent.duplicated(["anchor_id", "cycle_id"]).any():
        raise AssertionError("parent OOF population changed")
    probability_columns = [
        column
        for column in parent.columns
        if column.startswith(("qcontext__", "qcycle__"))
    ]
    values = parent[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise AssertionError("invalid frozen parent probability")

    anchor_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "state",
        "future_state_1",
        "future_state_2",
        "future_state_3",
        "future_state_4",
        *NUMERIC_CONTROLS,
        *(f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS),
    ]
    anchors = pd.read_parquet(ANCHOR_2024, columns=anchor_columns)
    anchor_dates = pd.to_datetime(anchors["session_date"], errors="raise")
    if set(anchor_dates.dt.year.unique()) != {2024} or anchors["anchor_id"].duplicated().any():
        raise AssertionError("2024 anchor source changed")
    if len(anchors) != 70374:
        raise AssertionError("2024 anchor count changed")
    return cycles, parent, anchors, observed_hashes


def path_occurrence(frame: pd.DataFrame, path: tuple[int, ...]) -> np.ndarray:
    occurrence = np.ones(len(frame), dtype=bool)
    for step, destination in enumerate(path[1:], start=1):
        occurrence &= frame[f"future_state_{step}"].to_numpy(dtype=int) == int(
            destination
        )
    return occurrence


def expand_exact_routes(
    parent: pd.DataFrame, anchors: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor_features = anchors.loc[
        :,
        [
            "anchor_id",
            "future_state_1",
            "future_state_2",
            "future_state_3",
            "future_state_4",
            *NUMERIC_CONTROLS,
        ],
    ]
    merged = parent.merge(
        anchor_features,
        on="anchor_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError("parent OOF failed the one-to-one anchor feature merge")
    merged = merged.drop(columns="_merge")
    rows: list[pd.DataFrame] = []
    manifest_lookup = manifest.set_index("route_id")
    for route in manifest.itertuples(index=False):
        selected = merged.loc[
            merged["cycle_id"].eq(route.cycle_id)
            & merged["state"].eq(int(route.current_state))
        ].copy()
        if selected.empty:
            raise AssertionError(f"empty exact route: {route.route_id}")
        occurrence = path_occurrence(selected, tuple(route.path_tuple))
        selected["route_id"] = str(route.route_id)
        selected["route_index"] = int(route.route_index)
        selected["exact_path"] = str(route.exact_path)
        selected["route_kind"] = str(route.route_kind)
        selected["structural_probability_available"] = bool(
            route.structural_probability_available
        )
        selected["exact_route_occurs"] = occurrence.astype(np.int8)
        selected["exact_conditional_weight"] = np.where(
            occurrence,
            selected["conditional_weight"].to_numpy(dtype=float),
            0.0,
        )
        if bool(route.structural_probability_available) and not np.array_equal(
            occurrence.astype(np.int8), selected["loop_occurs"].to_numpy(dtype=np.int8)
        ):
            raise AssertionError(f"unique exact route label drift: {route.route_id}")
        rows.append(selected)
    exact = pd.concat(rows, ignore_index=True).sort_values(
        ["parent_row_id", "route_index"], kind="stable"
    )
    exact = exact.reset_index(drop=True)

    split_ids = manifest.loc[
        manifest["route_kind"].eq("split_exact"), "route_id"
    ].tolist()
    split = exact.loc[exact["route_id"].isin(split_ids)]
    split_check = split.groupby("parent_row_id", sort=False)["exact_route_occurs"].agg(
        ["sum", "max"]
    )
    parent_split = parent.loc[
        parent["cycle_id"].eq("cycle_15") & parent["state"].eq(1)
    ].set_index("parent_row_id")["loop_occurs"]
    split_check = split_check.join(parent_split.rename("parent_occurs"), how="left")
    if split_check["parent_occurs"].isna().any():
        raise AssertionError("split route lost a parent row")
    if (split_check["sum"] > 1).any() or not np.array_equal(
        split_check["max"].to_numpy(dtype=int),
        split_check["parent_occurs"].to_numpy(dtype=int),
    ):
        raise AssertionError("cycle_15 state-1 split is not exhaustive and exclusive")

    parent_weights = parent.groupby("anchor_id", sort=False)["conditional_weight"].sum()
    exact_weights = exact.groupby("anchor_id", sort=False)["exact_conditional_weight"].sum()
    aligned = parent_weights.to_frame("parent").join(
        exact_weights.rename("exact"), how="outer"
    ).fillna(0.0)
    if not np.allclose(aligned["parent"], aligned["exact"], atol=1e-12):
        raise AssertionError("exact-route split changed positive overlap weight")

    for column in list(exact.columns):
        if column.startswith("qcycle__"):
            exact = exact.rename(columns={column: "parent_" + column})
    audit = {
        "parent_rows": len(parent),
        "exact_compatible_rows": len(exact),
        "parent_realized_rows": int(parent["loop_occurs"].sum()),
        "exact_realized_rows": int(exact["exact_route_occurs"].sum()),
        "split_parent_rows": len(parent_split),
        "split_child_rows": len(split),
        "split_parent_positives": int(parent_split.sum()),
        "split_child_positives": int(split["exact_route_occurs"].sum()),
        "weight_maximum_error": float(
            np.max(np.abs(aligned["parent"] - aligned["exact"]))
        ),
        "exact_routes": len(manifest_lookup),
    }
    if audit["parent_realized_rows"] != audit["exact_realized_rows"]:
        raise AssertionError("exact route expansion changed realized-row count")
    return exact, audit


def build_full_exact_conditional(
    anchors: pd.DataFrame,
    manifest: pd.DataFrame,
    parent_contract: dict[str, Any],
) -> pd.DataFrame:
    """Reconstruct every realised exact-route row across full 2024.

    The full-year positive cohort is used only as the expanding-prefix training
    pool.  July-December evaluation remains the frozen parent OOF population.
    """

    rows: list[pd.DataFrame] = []
    for route in manifest.itertuples(index=False):
        selected = anchors.loc[
            anchors["state"].eq(int(route.current_state))
        ].copy()
        occurrence = path_occurrence(selected, tuple(route.path_tuple))
        selected = selected.loc[occurrence].copy()
        if selected.empty:
            raise AssertionError(f"full 2024 exact route has no occurrence: {route.route_id}")
        selected["route_id"] = str(route.route_id)
        selected["route_index"] = int(route.route_index)
        selected["exact_path"] = str(route.exact_path)
        rows.append(selected)
    conditional = pd.concat(rows, ignore_index=True).sort_values(
        ["anchor_id", "route_index"], kind="stable"
    )
    conditional = conditional.reset_index(drop=True)
    positive_count = conditional.groupby("anchor_id", sort=False)[
        "route_id"
    ].transform("size")
    conditional["exact_positive_route_count"] = positive_count.astype(np.int16)
    conditional["exact_conditional_weight"] = 1.0 / positive_count.to_numpy(float)
    weight = conditional.groupby("anchor_id", sort=False)[
        "exact_conditional_weight"
    ].sum()
    if not np.allclose(weight.to_numpy(float), 1.0, atol=1e-12):
        raise AssertionError("full exact-route conditional weights do not sum to one")
    dates = pd.to_datetime(conditional["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {2024}:
        raise AssertionError("full exact conditional cohort crossed 2024")
    conditional["month"] = dates.dt.strftime("%Y-%m")
    thresholds = parent_contract["outcomes"]["thresholds_bps"]
    for target in TARGETS:
        for horizon in HORIZONS:
            p75 = float(thresholds[target][str(horizon)]["p75"])
            p90 = float(thresholds[target][str(horizon)]["p90"])
            outcome = conditional[f"{target}_{horizon}"].to_numpy(float)
            conditional[f"quality_class__{target}__h{horizon}"] = np.where(
                outcome > p90, 2, np.where(outcome > p75, 1, 0)
            ).astype(np.int8)
    if len(conditional) != 32677:
        raise AssertionError(
            f"full exact-route realized population changed: {len(conditional)}"
        )
    return conditional


def _training_medians(frame: pd.DataFrame) -> dict[str, float]:
    anchors = frame.drop_duplicates("anchor_id", keep="first")
    medians: dict[str, float] = {}
    for column in NUMERIC_CONTROLS:
        values = pd.to_numeric(anchors[column], errors="coerce")
        median = float(values.median())
        if not np.isfinite(median):
            raise AssertionError(f"non-finite training median: {column}")
        medians[column] = median
    return medians


def raw_context(frame: pd.DataFrame, medians: dict[str, float]) -> sparse.csr_matrix:
    numeric = frame.loc[:, list(NUMERIC_CONTROLS)].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.fillna(pd.Series(medians))
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("non-finite exact-route causal context")
    state = frame["state"].to_numpy(dtype=int)
    if state.min(initial=0) < 0 or state.max(initial=0) >= K:
        raise AssertionError("state outside frozen range")
    state_matrix = sparse.csr_matrix(np.eye(K, dtype=np.float64)[state])
    result = sparse.hstack(
        (state_matrix, sparse.csr_matrix(values)), format="csr"
    )
    if result.shape[1] != CONTEXT_WIDTH:
        raise AssertionError("exact-route context width drift")
    return result


def qexact_features(
    scaled_context: sparse.csr_matrix, route_index: np.ndarray
) -> sparse.csr_matrix:
    route_index = np.asarray(route_index, dtype=int)
    if route_index.min(initial=0) < 0 or route_index.max(initial=0) >= ROUTE_COUNT:
        raise AssertionError("route index outside frozen range")
    route = sparse.csr_matrix(
        (
            np.full(len(route_index), ROUTE_SCALE, dtype=np.float64),
            (np.arange(len(route_index)), route_index),
        ),
        shape=(len(route_index), ROUTE_COUNT),
    )
    result = sparse.hstack((scaled_context, route), format="csr")
    if result.shape[1] != QEXACT_WIDTH:
        raise AssertionError("qexact feature width drift")
    return result


def fit_qexact_oof(
    exact: pd.DataFrame, full_conditional: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = exact.copy()
    for target in TARGETS:
        for horizon in HORIZONS:
            output[f"qexact__{target}__h{horizon}__p75"] = np.nan
            output[f"qexact__{target}__h{horizon}__p90"] = np.nan

    oof_conditional = output.loc[output["exact_route_occurs"].eq(1)].copy()
    expected_keys = set(
        map(
            tuple,
            oof_conditional.loc[:, ["anchor_id", "route_id"]].itertuples(
                index=False, name=None
            ),
        )
    )
    full_oof = full_conditional.loc[
        full_conditional["month"].isin(OOF_MONTHS)
    ]
    observed_keys = set(
        map(
            tuple,
            full_oof.loc[:, ["anchor_id", "route_id"]].itertuples(
                index=False, name=None
            ),
        )
    )
    if expected_keys != observed_keys or len(expected_keys) != len(oof_conditional):
        raise AssertionError("full conditional reconstruction disagrees with parent OOF")
    fold_rows: list[dict[str, Any]] = []
    for validation_month in OOF_MONTHS:
        train = full_conditional.loc[
            full_conditional["month"].astype(str).lt(validation_month)
        ].copy()
        validation = output.loc[
            output["validation_month"].astype(str).eq(validation_month)
        ].copy()
        if train.empty or validation.empty:
            raise AssertionError(f"empty qexact fold: {validation_month}")
        medians = _training_medians(train)
        train_raw = raw_context(train, medians)
        validation_raw = raw_context(validation, medians)
        train_weights = train["exact_conditional_weight"].to_numpy(dtype=float)
        if train_weights.sum() <= 0:
            raise AssertionError("qexact training fold has zero weight")
        scaler = StandardScaler(with_mean=False)
        scaler.fit(train_raw, sample_weight=train_weights)
        train_x = qexact_features(
            scaler.transform(train_raw).tocsr(), train["route_index"].to_numpy(int)
        )
        validation_x = qexact_features(
            scaler.transform(validation_raw).tocsr(),
            validation["route_index"].to_numpy(int),
        )
        validation_indices = validation.index.to_numpy(dtype=int)
        for target in TARGETS:
            for horizon in HORIZONS:
                target_column = f"quality_class__{target}__h{horizon}"
                train_target = train[target_column].to_numpy(dtype=int)
                if not np.array_equal(np.unique(train_target), np.asarray([0, 1, 2])):
                    raise AssertionError(
                        f"qexact training target lacks a class: {validation_month} {target} h{horizon}"
                    )
                model = LogisticRegression(
                    C=0.2,
                    solver="lbfgs",
                    max_iter=1000,
                    tol=0.0001,
                    random_state=SEED,
                )
                model.fit(train_x, train_target, sample_weight=train_weights)
                if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
                    raise AssertionError("qexact class order changed")
                if int(model.n_iter_[0]) >= 1000:
                    raise AssertionError("qexact did not converge")
                probability = model.predict_proba(validation_x)
                if not np.allclose(probability.sum(axis=1), 1.0):
                    raise AssertionError("qexact class probability did not normalize")
                p75 = probability[:, 1] + probability[:, 2]
                p90 = probability[:, 2]
                output.loc[
                    validation_indices, f"qexact__{target}__h{horizon}__p75"
                ] = p75
                output.loc[
                    validation_indices, f"qexact__{target}__h{horizon}__p90"
                ] = p90
                fold_rows.append(
                    {
                        "validation_month": validation_month,
                        "target": target,
                        "horizon": horizon,
                        "training_rows": len(train),
                        "training_weight": float(train_weights.sum()),
                        "validation_compatible_rows": len(validation),
                        "validation_realized_rows": int(
                            validation["exact_route_occurs"].sum()
                        ),
                        "feature_width": train_x.shape[1],
                        "n_iter": int(model.n_iter_[0]),
                        "numeric_medians": json.dumps(medians, sort_keys=True),
                        "scaler_scale": json.dumps(
                            [float(value) for value in scaler.scale_]
                        ),
                    }
                )
    qexact_columns = [column for column in output if column.startswith("qexact__")]
    values = output[qexact_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise AssertionError("incomplete or invalid qexact OOF probability")
    for target in TARGETS:
        for horizon in HORIZONS:
            if (
                output[f"qexact__{target}__h{horizon}__p90"]
                > output[f"qexact__{target}__h{horizon}__p75"] + 1e-12
            ).any():
                raise AssertionError("qexact ordered probability nesting failed")
    return output, pd.DataFrame(fold_rows)


def support_gate(frame: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    rule = contract["support_gates_each_route"]
    realized = frame.loc[frame["exact_route_occurs"].eq(1)]
    quarter_counts = realized["quarter"].astype(str).value_counts()
    checks = {
        "compatible_rows": len(frame) >= int(rule["compatible_rows_minimum"]),
        "realized_rows": len(realized) >= int(rule["realized_rows_minimum"]),
        "stocks": realized["symbol_norm"].nunique()
        >= int(rule["realized_stocks_minimum"]),
        "quarters": set(quarter_counts.index)
        == set(rule["required_quarters"]),
        "quarter_rows": all(
            int(quarter_counts.get(quarter, 0))
            >= int(rule["realized_rows_each_quarter_minimum"])
            for quarter in rule["required_quarters"]
        ),
    }
    return {
        "compatible_rows": len(frame),
        "realized_rows": len(realized),
        "realized_stocks": int(realized["symbol_norm"].nunique()),
        "minimum_quarter_rows": min(
            (int(quarter_counts.get(value, 0)) for value in rule["required_quarters"]),
            default=0,
        ),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _probability_column(model: str, target: str, horizon: int, tier: str) -> str:
    if model == "parent_qcycle":
        return f"parent_qcycle__{target}__h{horizon}__{tier}"
    return f"{model}__{target}__h{horizon}__{tier}"


def _surface_arrays(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    tier: str,
    surface: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    threshold = 1 if tier == "p75" else 2
    if surface == "conditional":
        selected = frame.loc[frame["exact_route_occurs"].eq(1)].reset_index(drop=True)
        observed = (
            selected[f"quality_class__{target}__h{horizon}"].to_numpy(int)
            >= threshold
        ).astype(int)
        weights = selected["exact_conditional_weight"].to_numpy(float)
        probabilities = {
            model: selected[_probability_column(model, target, horizon, tier)].to_numpy(
                float
            )
            for model in ("qcontext", "parent_qcycle", "qexact")
        }
    else:
        selected = frame.reset_index(drop=True)
        quality = selected[f"quality_class__{target}__h{horizon}"].to_numpy(int)
        observed = (
            selected["exact_route_occurs"].to_numpy(bool) & (quality >= threshold)
        ).astype(int)
        weights = np.ones(len(selected), dtype=float)
        structural = selected["loop_probability"].to_numpy(float)
        probabilities = {
            model: structural
            * selected[_probability_column(model, target, horizon, tier)].to_numpy(float)
            for model in ("qcontext", "parent_qcycle", "qexact")
        }
    return selected, observed, weights, probabilities


def strict_calibration_and_noninferiority(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    tier: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    rules = contract["quality_gates_each_route_target_horizon_tier"]
    calibration_rule = rules["calibration"]
    noninferiority_rule = rules["qexact_noninferiority_to_parent_qcycle"]
    result: dict[str, Any] = {}
    all_pass = True
    for surface in ("conditional", "joint"):
        selected, observed, weights, probabilities = _surface_arrays(
            frame, target, horizon, tier, surface
        )
        minimum = int(
            calibration_rule[
                "conditional_supported_bin_rows_minimum"
                if surface == "conditional"
                else "joint_supported_bin_rows_minimum"
            ]
        )
        metrics: dict[str, Any] = {}
        losses: dict[str, dict[str, np.ndarray]] = {}
        for model, probability in probabilities.items():
            ece, maximum = _calibration_summary(
                observed, probability, weights, minimum
            )
            losses[model] = binary_losses(observed, probability)
            metrics[model] = {"ece": ece, "maximum_supported_bin_error": maximum}
        candidate = metrics["qexact"]
        calibration_checks = {
            "ece_no_worse_than_qcontext": candidate["ece"]
            <= metrics["qcontext"]["ece"],
            "ece_no_worse_than_parent_qcycle": candidate["ece"]
            <= metrics["parent_qcycle"]["ece"],
            "absolute_maximum_supported_bin_error": bool(
                np.isfinite(candidate["maximum_supported_bin_error"])
                and candidate["maximum_supported_bin_error"]
                <= float(
                    calibration_rule[
                        "conditional_absolute_maximum_supported_bin_error"
                        if surface == "conditional"
                        else "joint_absolute_maximum_supported_bin_error"
                    ]
                )
            ),
        }
        candidate_ll = weighted_mean(losses["qexact"]["log_loss"], weights)
        parent_ll = weighted_mean(losses["parent_qcycle"]["log_loss"], weights)
        candidate_brier = weighted_mean(losses["qexact"]["brier"], weights)
        parent_brier = weighted_mean(losses["parent_qcycle"]["brier"], weights)
        relative_degradation = (candidate_ll - parent_ll) / parent_ll
        brier_difference = candidate_brier - parent_brier
        noninferiority_checks = {
            "relative_log_loss": relative_degradation
            <= float(
                noninferiority_rule[
                    "conditional_relative_log_loss_degradation_maximum"
                    if surface == "conditional"
                    else "joint_relative_log_loss_degradation_maximum"
                ]
            ),
            "brier": brier_difference
            <= float(
                noninferiority_rule[
                    "conditional_brier_difference_maximum"
                    if surface == "conditional"
                    else "joint_brier_difference_maximum"
                ]
            ),
        }
        surface_pass = bool(
            all(calibration_checks.values()) and all(noninferiority_checks.values())
        )
        all_pass &= surface_pass
        result[surface] = {
            "rows": len(selected),
            "calibration": metrics,
            "calibration_checks": calibration_checks,
            "relative_log_loss_degradation_vs_parent_qcycle": relative_degradation,
            "brier_difference_vs_parent_qcycle": brier_difference,
            "noninferiority_checks": noninferiority_checks,
            "pass": surface_pass,
        }
    result["pass"] = bool(all_pass)
    return result


def daily_conditional_log_loss_difference(
    frame: pd.DataFrame, target: str, horizon: int, tier: str
) -> pd.Series:
    selected, observed, weights, probabilities = _surface_arrays(
        frame, target, horizon, tier, "conditional"
    )
    candidate = binary_losses(observed, probabilities["qexact"])["log_loss"]
    baseline = binary_losses(observed, probabilities["qcontext"])["log_loss"]
    daily = pd.DataFrame(
        {
            "session_date": selected["session_date"].astype(str).to_numpy(),
            "weighted": (candidate - baseline) * weights,
            "weight": weights,
        }
    ).groupby("session_date", sort=True).sum()
    return daily["weighted"] / daily["weight"]


def sign_flip_p_value(values: np.ndarray, seed: int, draws: int = 9999) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return math.nan
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = (signs @ values) / len(values)
    return float((1 + np.sum(null <= observed)) / (draws + 1))


def holm_table(rows: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    result = rows.copy()
    result["holm_adjusted_p"] = np.nan
    result["holm_pass"] = False
    for tier, positions in result.groupby("tier", sort=True).groups.items():
        positions = list(positions)
        ordered = sorted(positions, key=lambda position: result.loc[position, "p_value"])
        family_size = len(ordered)
        running = 0.0
        for rank, position in enumerate(ordered, start=1):
            adjusted = min(
                1.0,
                max(
                    running,
                    (family_size - rank + 1)
                    * float(result.loc[position, "p_value"]),
                ),
            )
            running = adjusted
            result.loc[position, "holm_adjusted_p"] = adjusted
            result.loc[position, "holm_pass"] = bool(adjusted <= alpha)
            result.loc[position, "holm_rank"] = rank
            result.loc[position, "family_size"] = family_size
    return result


def prepare_route_frame(frame: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    output = frame.copy()
    output["loop_occurs"] = output["exact_route_occurs"].astype(np.int8)
    output["conditional_weight"] = output["exact_conditional_weight"].astype(float)
    quality = output[f"quality_class__{target}__h{horizon}"].to_numpy(int)
    output[f"joint_good_target__{target}__h{horizon}"] = (
        output["exact_route_occurs"].to_numpy(bool) & (quality >= 1)
    ).astype(np.int8)
    output[f"joint_high_target__{target}__h{horizon}"] = (
        output["exact_route_occurs"].to_numpy(bool) & (quality >= 2)
    ).astype(np.int8)
    structural = output["loop_probability"].to_numpy(float)
    for tier in TIERS:
        exact_column = f"qexact__{target}__h{horizon}__{tier}"
        output[f"qcycle__{target}__h{horizon}__{tier}"] = output[exact_column]
        output[f"joint__qcontext__{target}__h{horizon}__{tier}"] = (
            structural * output[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float)
        )
        output[f"joint__qcycle__{target}__h{horizon}__{tier}"] = (
            structural * output[exact_column].to_numpy(float)
        )
    return output


def descriptive_split_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    realized = frame.loc[frame["exact_route_occurs"].eq(1)]
    weights = realized["exact_conditional_weight"].to_numpy(float)
    for target in TARGETS:
        for horizon in HORIZONS:
            quality = realized[f"quality_class__{target}__h{horizon}"].to_numpy(int)
            for tier in TIERS:
                observed = (quality >= (1 if tier == "p75" else 2)).astype(int)
                rows.append(
                    {
                        "route_id": str(frame["route_id"].iloc[0]),
                        "target": target,
                        "horizon": horizon,
                        "tier": tier,
                        "compatible_rows": len(frame),
                        "realized_rows": len(realized),
                        "observed_rate": weighted_mean(observed, weights),
                        "mean_qcontext": weighted_mean(
                            realized[f"qcontext__{target}__h{horizon}__{tier}"],
                            weights,
                        ),
                        "mean_parent_qcycle": weighted_mean(
                            realized[
                                f"parent_qcycle__{target}__h{horizon}__{tier}"
                            ],
                            weights,
                        ),
                        "mean_qexact": weighted_mean(
                            realized[f"qexact__{target}__h{horizon}__{tier}"],
                            weights,
                        ),
                        "structural_probability_available": False,
                        "qualification_permitted": False,
                    }
                )
    return rows


def evaluate_screen(
    exact: pd.DataFrame,
    manifest: pd.DataFrame,
    contract: dict[str, Any],
    parent_contract: dict[str, Any],
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    support_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    multiplicity_inputs: list[dict[str, Any]] = []
    cell_details: dict[tuple[str, str, int, str], dict[str, Any]] = {}

    for route_position, (route_id, route_frame) in enumerate(
        exact.groupby("route_id", sort=True)
    ):
        route_frame = route_frame.reset_index(drop=True)
        descriptor = manifest.set_index("route_id").loc[route_id]
        support = support_gate(route_frame, contract)
        support_rows.append(
            {
                "route_id": route_id,
                "cycle_id": descriptor["cycle_id"],
                "current_state": int(descriptor["current_state"]),
                "exact_path": descriptor["exact_path"],
                "route_kind": descriptor["route_kind"],
                **{key: value for key, value in support.items() if key != "checks"},
                "checks": json.dumps(support["checks"], sort_keys=True),
            }
        )
        structural_available = bool(descriptor["structural_probability_available"])
        if structural_available:
            structural = _structural_gate(route_frame, 250, 0.01)
        else:
            structural = {
                "pass": False,
                "checks": {"exact_structural_probability_available": False},
            }
            split_rows.extend(descriptive_split_rows(route_frame))
        structural_rows.append(
            {
                "route_id": route_id,
                "cycle_id": descriptor["cycle_id"],
                "current_state": int(descriptor["current_state"]),
                "exact_path": descriptor["exact_path"],
                "structural_probability_available": structural_available,
                **{key: value for key, value in structural.items() if key != "checks"},
                "checks": json.dumps(structural["checks"], sort_keys=True),
            }
        )

        if support["pass"]:
            for target in TARGETS:
                for horizon in HORIZONS:
                    for tier_index, tier in enumerate(TIERS):
                        daily = daily_conditional_log_loss_difference(
                            route_frame, target, horizon, tier
                        )
                        multiplicity_inputs.append(
                            {
                                "route_id": route_id,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "daily_sessions": len(daily),
                                "observed_daily_mean": float(daily.mean()),
                                "p_value": sign_flip_p_value(
                                    daily.to_numpy(float),
                                    SEED
                                    + route_position * 1000
                                    + TARGETS.index(target) * 100
                                    + horizon * 3
                                    + tier_index,
                                ),
                            }
                        )
                        if not structural_available:
                            continue
                        prepared = prepare_route_frame(route_frame, target, horizon)
                        parent_result = _quality_cell_gate(
                            prepared,
                            target,
                            horizon,
                            tier,
                            "oof",
                            ["2024_q3", "2024_q4"],
                            parent_contract,
                            SEED
                            + route_position * 1000
                            + TARGETS.index(target) * 200
                            + horizon * 5
                            + tier_index,
                        )
                        strict = strict_calibration_and_noninferiority(
                            route_frame, target, horizon, tier, contract
                        )
                        key = (route_id, target, horizon, tier)
                        cell_details[key] = {
                            "parent": parent_result,
                            "strict": strict,
                        }
                        cell_rows.append(
                            {
                                "route_id": route_id,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "parent_primary_pass": bool(parent_result["pass"]),
                                "strict_calibration_and_qcycle_noninferiority_pass": bool(
                                    strict["pass"]
                                ),
                                "positive_rows": parent_result["positive_rows"],
                                "negative_rows": parent_result["negative_rows"],
                                "observed_rate": parent_result["observed_rate"],
                                "mean_qcontext": parent_result[
                                    "mean_qcontext_probability"
                                ],
                                "mean_qexact": parent_result[
                                    "mean_qcycle_probability"
                                ],
                                "observed_rate_over_qcontext": parent_result[
                                    "observed_rate_divided_by_mean_qcontext"
                                ],
                                "conditional_relative_log_loss_improvement": parent_result[
                                    "conditional_gate"
                                ]["relative_log_loss_improvement"],
                                "joint_relative_log_loss_improvement": parent_result[
                                    "joint_gate"
                                ]["relative_log_loss_improvement"],
                                "parent_gate_detail": json.dumps(
                                    safe(parent_result), sort_keys=True
                                ),
                                "strict_gate_detail": json.dumps(
                                    safe(strict), sort_keys=True
                                ),
                            }
                        )

    multiplicity = holm_table(pd.DataFrame(multiplicity_inputs))
    holm_lookup = multiplicity.set_index(
        ["route_id", "target", "horizon", "tier"]
    )["holm_pass"].to_dict()
    support_lookup = pd.DataFrame(support_rows).set_index("route_id")["pass"].to_dict()
    structural_lookup = pd.DataFrame(structural_rows).set_index("route_id")[
        "pass"
    ].to_dict()
    grades: list[dict[str, Any]] = []
    for route in manifest.itertuples(index=False):
        route_id = str(route.route_id)
        for horizon in HORIZONS:
            p75_cells = [
                cell_details.get((route_id, target, horizon, "p75"))
                for target in TARGETS
            ]
            p90_cells = [
                cell_details.get((route_id, target, horizon, "p90"))
                for target in TARGETS
            ]
            p75_complete = all(value is not None for value in p75_cells)
            p90_complete = all(value is not None for value in p90_cells)
            p75_pass = bool(
                p75_complete
                and all(value["parent"]["pass"] and value["strict"]["pass"] for value in p75_cells)
                and all(
                    holm_lookup.get((route_id, target, horizon, "p75"), False)
                    for target in TARGETS
                )
            )
            good = bool(
                support_lookup.get(route_id, False)
                and structural_lookup.get(route_id, False)
                and p75_pass
            )
            high_p75 = bool(
                good
                and all(
                    value["parent"]["observed_rate"] >= 0.35
                    and value["parent"]["mean_qcycle_probability"] >= 0.35
                    for value in p75_cells
                )
            )
            p90_pass = bool(
                p90_complete
                and all(value["parent"]["pass"] and value["strict"]["pass"] for value in p90_cells)
                and all(
                    holm_lookup.get((route_id, target, horizon, "p90"), False)
                    for target in TARGETS
                )
            )
            high = bool(high_p75 and p90_pass)
            grade = (
                "development_high_screen_candidate"
                if high
                else "development_good_screen_candidate"
                if good
                else "development_unqualified"
            )
            grades.append(
                {
                    "route_id": route_id,
                    "cycle_id": route.cycle_id,
                    "current_state": int(route.current_state),
                    "exact_path": route.exact_path,
                    "horizon": horizon,
                    "support_pass": bool(support_lookup.get(route_id, False)),
                    "structural_pass": bool(
                        structural_lookup.get(route_id, False)
                    ),
                    "both_targets_p75_pass": p75_pass,
                    "high_p75_rate_pass": high_p75,
                    "both_targets_p90_pass": p90_pass,
                    "grade": grade,
                    "certified_good_or_high": False,
                    "prospective_validated": False,
                }
            )
    grade_frame = pd.DataFrame(grades)
    candidates = grade_frame.loc[
        grade_frame["grade"].isin(
            [
                "development_good_screen_candidate",
                "development_high_screen_candidate",
            ]
        )
    ].copy()
    cap = int(contract["decision_and_stop_rules"]["candidate_cap"])
    if len(candidates) == 0:
        label = "no_exact_route_quality_screen_candidate"
    elif len(candidates) <= cap:
        label = "exact_route_quality_screen_candidates_frozen_pending_separate_contract"
    else:
        label = "candidate_cap_exceeded_fail_closed"
        candidates = candidates.iloc[0:0].copy()
    decision = {
        "label": label,
        "candidate_count": len(candidates),
        "candidate_cap": cap,
        "candidate_ids": [
            f"{row.route_id}@h{int(row.horizon)}"
            for row in candidates.itertuples(index=False)
        ],
        "later_period_scoring_performed": False,
        "further_refinement_performed": False,
        "parent_grade_changed": False,
        "certified_good_or_high": False,
        "prospective_validated": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    return {
        "support": pd.DataFrame(support_rows),
        "structural": pd.DataFrame(structural_rows),
        "cells": pd.DataFrame(cell_rows),
        "split": pd.DataFrame(split_rows),
        "multiplicity": multiplicity,
        "grades": grade_frame,
        "candidates": candidates,
        "decision": decision,
    }


def artifact_manifest(root: Path, names: Iterable[str]) -> dict[str, Any]:
    rows = {
        name: {"size": (root / name).stat().st_size, "sha256": sha256(root / name)}
        for name in sorted(names)
    }
    return {
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "later_period_paths_resolved": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "artifacts": rows,
    }


def run(output: Path = OUT) -> dict[str, Any]:
    contract = load_contract()
    parent_contract = json.loads(PARENT_CONTRACT.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"screen output root is not pristine: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cycles, parent, anchors, source_hashes = load_inputs()
    manifest = build_exact_route_manifest(cycles)
    exact, expansion_audit = expand_exact_routes(parent, anchors, manifest)
    full_conditional = build_full_exact_conditional(
        anchors, manifest, parent_contract
    )
    exact, fold_audit = fit_qexact_oof(exact, full_conditional)
    evaluated = evaluate_screen(exact, manifest, contract, parent_contract)

    manifest_output = manifest.drop(columns="path_tuple").copy()
    artifacts: dict[str, Any] = {
        "route_manifest.csv": manifest_output,
        "expansion_audit.json": expansion_audit,
        "fold_audit.csv": fold_audit,
        "exact_route_oof_predictions.parquet": exact,
        "route_support.csv": evaluated["support"],
        "route_structural.csv": evaluated["structural"],
        "route_quality_cells.csv": evaluated["cells"],
        "split_route_diagnostics.csv": evaluated["split"],
        "multiplicity.csv": evaluated["multiplicity"],
        "route_horizon_grades.csv": evaluated["grades"],
        "candidate_set.csv": evaluated["candidates"],
        "decision.json": evaluated["decision"],
        "source_manifest.json": {
            "source_hashes": source_hashes,
            "contract_sha256": CONTRACT_SHA256,
            "runner_sha256": sha256(Path(__file__)),
            "fit_year": 2024,
            "parent_oof_rows": len(parent),
            "anchor_rows": len(anchors),
            "full_exact_realized_rows": len(full_conditional),
            "later_period_paths_resolved": False,
            "later_period_rows_read": False,
            "shadow_tree_read": False,
            "shadow_tree_written": False,
            "volume_label": "historical_volume_not_used",
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
        },
    }
    written: list[str] = []
    for name, value in artifacts.items():
        path = output / name
        if isinstance(value, pd.DataFrame):
            if path.suffix == ".parquet":
                value.to_parquet(path, index=False)
            else:
                value.to_csv(path, index=False)
        else:
            write_json(path, value)
        written.append(name)
    complete_manifest = artifact_manifest(output, written)
    write_json(output / "artifact_manifest.json", complete_manifest)
    completion = {
        "status": "complete_2024_development_screen_no_further_refinement",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "artifact_manifest_sha256": sha256(output / "artifact_manifest.json"),
        "decision": evaluated["decision"],
        "independent_audit_pending": True,
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
    write_json(output / "screen_complete.json", completion)
    return completion


def validate_only() -> dict[str, Any]:
    contract = load_contract()
    parent_contract = json.loads(PARENT_CONTRACT.read_text())
    cycles, parent, anchors, source_hashes = load_inputs()
    manifest = build_exact_route_manifest(cycles)
    exact, expansion_audit = expand_exact_routes(parent, anchors, manifest)
    conditional = build_full_exact_conditional(anchors, manifest, parent_contract)
    return {
        "contract_sha256": CONTRACT_SHA256,
        "route_units": len(manifest),
        "split_units": int(manifest["route_kind"].eq("split_exact").sum()),
        "exact_oof_rows": len(exact),
        "exact_oof_realized_rows": int(exact["exact_route_occurs"].sum()),
        "full_2024_realized_rows": len(conditional),
        "expansion_audit": expansion_audit,
        "source_hashes": source_hashes,
        "later_period_paths_resolved": False,
        "research_only": contract["research_only"],
        "live_ordering_enabled": contract["live_ordering_enabled"],
        "order_placement": contract["order_placement"],
    }


def self_test() -> dict[str, Any]:
    cycles = pd.DataFrame(
        [
            {
                "cycle_index": index,
                "cycle_id": f"cycle_{index + 1:02d}",
                "cycle": cycle,
                "transition_length": len(cycle.split("->")) - 1,
            }
            for index, cycle in enumerate(
                [
                    "1->3->1",
                    "1->2->1",
                    "0->1->0",
                    "2->4->2",
                    "0->3->0",
                    "4->6->4",
                    "5->6->5",
                    "3->4->3",
                    "3->6->3",
                    "1->4->1",
                    "2->5->2",
                    "0->1->0->1->0",
                    "5->7->5",
                    "1->3->1->3->1",
                    "1->2->1->3->1",
                    "1->2->3->1",
                    "1->2->1->2->1",
                    "0->3->1->0",
                    "0->1->3->0",
                    "2->3->2",
                ]
            )
        ]
    )
    manifest = build_exact_route_manifest(cycles)
    path = (1, 2, 1)
    frame = pd.DataFrame({"future_state_1": [2, 3], "future_state_2": [1, 1]})
    checks = {
        "manifest_45": len(manifest) == 45,
        "split_2": int(manifest["route_kind"].eq("split_exact").sum()) == 2,
        "path_label": path_occurrence(frame, path).tolist() == [True, False],
        "sign_flip": 0.0 <= sign_flip_p_value(np.asarray([-1.0] * 12), 7, 99) <= 1.0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"checks": checks, "all_passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if args.validate_only:
        print(json.dumps(safe(validate_only()), indent=2, sort_keys=True))
        return
    if not args.run:
        raise SystemExit("use --run, --validate-only, or --self-test")
    print(json.dumps(run(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
