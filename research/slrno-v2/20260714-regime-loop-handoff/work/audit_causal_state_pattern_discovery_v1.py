"""Independent replay audit for causal-state-pattern-discovery-v1.

This file deliberately does not import the production runner.

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-causal-state-pattern-discovery-v1.json"
RUNNER = HERE / "run_causal_state_pattern_discovery_v1.py"
ANCHOR = Path(
    "/private/tmp/stocker_frozen_loop_price_consequence_20260710/anchor_panel_train_2024.parquet"
)
PARAMETERS = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/frozen_semimarkov_parameters.npz"
)
CYCLES = Path("/private/tmp/stocker_per_loop_movement_quality_20260710/fixed_cycles.csv")
ROOT = Path("/private/tmp/stocker_causal_state_pattern_discovery_v1_20260711")

EXPECTED = {
    "contract": "cb3c217da9bcbac1606ca0ef69b13bad16ae54307084c839b092edba4f7d5759",
    "runner": "69bd696c13a8ae52c49d371f4849902e6a1c3ef285ffe5de42e5203f5a3ce3a1",
    "anchor": "788fd81909d1c5d3e6ee20e3e36e3ebb74199188e41052ea1b04f61c96fa9932",
    "manifest": "e82b354cad060e272465661e657c300714be8fbad85b8ba9944a63153cd13b3e",
}
DISCOVERY_MONTHS = tuple(f"2024-{month:02d}" for month in range(1, 7))
QUALIFICATION_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
SEED = 20260711
EPSILON = 1e-12
K = 8
BOOTSTRAP_DRAWS = 1999
SIGN_FLIP_DRAWS = 4999

STATE_COLUMNS = [
    "anchor_id",
    "symbol_norm",
    "session_date",
    "month",
    "state",
    "previous_state_1",
    "previous_state_2",
    "future_state_1",
    "future_state_2",
    "future_state_3",
    "future_state_4",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_path(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in str(value).split("->"))


def path_text(path: Sequence[int]) -> str:
    return "->".join(str(int(value)) for value in path)


def path_id(family: str, path: Sequence[int]) -> str:
    token = "_".join(str(int(value)) for value in path)
    return (
        f"closed_loop__L{len(path) - 1}__{token}"
        if family == "closed_loop"
        else f"upward_excursion__{token}"
    )


def occurs(frame: pd.DataFrame, path: Sequence[int]) -> np.ndarray:
    output = frame["state"].to_numpy(int) == int(path[0])
    for step, destination in enumerate(path[1:], start=1):
        output &= frame[f"future_state_{step}"].to_numpy(int) == int(destination)
    return output


def normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("novel", "existing_control", "decision_eligible"):
        if output[column].dtype != bool:
            output[column] = output[column].astype(str).str.lower().map(
                {"true": True, "false": False}
            )
    for column in (
        "candidate_index",
        "start_state",
        "destination_state",
        "transition_length",
        "discovery_occurrences",
        "discovery_stocks",
        "minimum_discovery_month_occurrences",
    ):
        output[column] = output[column].astype(int)
    return output


def frozen_paths() -> set[tuple[int, ...]]:
    result: set[tuple[int, ...]] = set()
    cycles = pd.read_csv(CYCLES)
    assert len(cycles) == 20
    for cycle in cycles.itertuples(index=False):
        closed = parse_path(cycle.cycle)
        core = closed[:-1]
        for index in range(len(core)):
            result.add(core[index:] + core[:index] + (core[index],))
    return result


def support_values(selected: pd.DataFrame) -> tuple[int, int, int]:
    month = selected["month"].astype(str).value_counts()
    return (
        len(selected),
        selected["symbol_norm"].nunique(),
        min(int(month.get(value, 0)) for value in DISCOVERY_MONTHS),
    )


def rebuild_manifest(state: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    discovery = state.loc[state["month"].astype(str).isin(DISCOVERY_MONTHS)]
    known = frozen_paths()
    loop_rule = contract["candidate_discovery"]["family_exact_closed_loop"]
    loop_rows: list[dict[str, Any]] = []
    for length in (2, 3, 4):
        columns = ["state", *[f"future_state_{step}" for step in range(1, length + 1)]]
        closed = discovery.loc[discovery[f"future_state_{length}"].eq(discovery["state"])]
        for values, indices in closed.groupby(columns, sort=True).groups.items():
            path = tuple(int(value) for value in values)
            count, stocks, minimum = support_values(discovery.loc[list(indices)])
            eligible = (
                count >= int(loop_rule["minimum_discovery_occurrences"])
                and stocks >= int(loop_rule["minimum_discovery_stocks"])
                and minimum >= int(loop_rule["minimum_occurrences_each_discovery_month"])
            )
            loop_rows.append(
                {
                    "candidate_id": path_id("closed_loop", path),
                    "family": "closed_loop",
                    "start_state": path[0],
                    "destination_state": path[-1],
                    "transition_length": length,
                    "exact_path": path_text(path),
                    "novel": path not in known,
                    "existing_control": path in known,
                    "decision_eligible": path not in known,
                    "discovery_occurrences": count,
                    "discovery_stocks": stocks,
                    "minimum_discovery_month_occurrences": minimum,
                    "eligible": eligible,
                }
            )
    loops = pd.DataFrame(loop_rows)
    selected_loops: list[pd.DataFrame] = []
    for novel, cap in ((True, 40), (False, 8)):
        selected_loops.append(
            loops.loc[loops["eligible"] & loops["novel"].eq(novel)]
            .sort_values(
                [
                    "minimum_discovery_month_occurrences",
                    "discovery_occurrences",
                    "discovery_stocks",
                    "exact_path",
                ],
                ascending=[False, False, False, True],
                kind="stable",
            )
            .head(cap)
        )
    with np.load(PARAMETERS) as payload:
        centroids = np.asarray(payload["means"], dtype=float)[:, 5]
    edge_rule = contract["candidate_discovery"]["family_directed_excursion"]
    edge_rows: list[dict[str, Any]] = []
    for source in range(8):
        for destination in range(8):
            delta = float(centroids[destination] - centroids[source])
            if not (centroids[source] < 0 < centroids[destination] and delta >= 0.25):
                continue
            selected = discovery.loc[
                discovery["state"].eq(source) & discovery["future_state_1"].eq(destination)
            ]
            count, stocks, minimum = support_values(selected)
            eligible = (
                count >= int(edge_rule["minimum_discovery_occurrences"])
                and stocks >= int(edge_rule["minimum_discovery_stocks"])
                and minimum >= int(edge_rule["minimum_occurrences_each_discovery_month"])
            )
            path = (source, destination)
            edge_rows.append(
                {
                    "candidate_id": path_id("upward_excursion", path),
                    "family": "upward_excursion",
                    "start_state": source,
                    "destination_state": destination,
                    "transition_length": 1,
                    "exact_path": path_text(path),
                    "novel": True,
                    "existing_control": False,
                    "decision_eligible": True,
                    "discovery_occurrences": count,
                    "discovery_stocks": stocks,
                    "minimum_discovery_month_occurrences": minimum,
                    "eligible": eligible,
                }
            )
    edges = (
        pd.DataFrame(edge_rows)
        .loc[lambda value: value["eligible"]]
        .sort_values(
            [
                "minimum_discovery_month_occurrences",
                "discovery_occurrences",
                "discovery_stocks",
                "start_state",
                "destination_state",
            ],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        .head(16)
    )
    selected = pd.concat([*selected_loops, edges], ignore_index=True)
    selected = selected.sort_values(["family", "candidate_id"], kind="stable")
    selected["candidate_index"] = selected.groupby("family", sort=True).cumcount()
    columns = [
        "candidate_id",
        "family",
        "candidate_index",
        "start_state",
        "destination_state",
        "transition_length",
        "exact_path",
        "novel",
        "existing_control",
        "decision_eligible",
        "discovery_occurrences",
        "discovery_stocks",
        "minimum_discovery_month_occurrences",
    ]
    return normalize_manifest(selected[columns].reset_index(drop=True))


def quality_columns(contract: dict[str, Any]) -> list[str]:
    numeric = contract["movement_models"]["qcontext_features"]["causal_numeric_controls"]
    return list(
        dict.fromkeys(
            [
                *STATE_COLUMNS,
                "quarter",
                "bar_index_in_session",
                *numeric,
                *[f"exact_{horizon}" for horizon in HORIZONS],
                *[f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS],
            ]
        )
    )


def load_anchors(contract: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(ANCHOR, columns=quality_columns(contract))
    frame["month"] = frame["month"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["quarter"] = frame["quarter"].astype(str)
    bar = frame["bar_index_in_session"].astype(int).to_numpy()
    frame["session_bucket"] = np.where(bar <= 25, "open", np.where(bar <= 50, "middle", "late"))
    for target in TARGETS:
        for horizon in HORIZONS:
            values = frame[f"{target}_{horizon}"].to_numpy(float)
            thresholds = contract["outcomes"]["thresholds"][target][str(horizon)]
            frame[f"quality_class__{target}__h{horizon}"] = np.where(
                values > float(thresholds["p90"]),
                2,
                np.where(values > float(thresholds["p75"]), 1, 0),
            ).astype(np.int8)
    return frame


def expand_realized(anchors: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for candidate in manifest.itertuples(index=False):
        compatible = anchors.loc[anchors["state"].eq(candidate.start_state)].copy()
        selected = compatible.loc[occurs(compatible, parse_path(candidate.exact_path))].copy()
        selected["candidate_id"] = candidate.candidate_id
        selected["candidate_index"] = candidate.candidate_index
        selected["family"] = candidate.family
        rows.append(selected)
    output = pd.concat(rows, ignore_index=True)
    count = output.groupby(["family", "anchor_id"], sort=False)["candidate_id"].transform("count")
    output["conditional_weight"] = 1.0 / count.to_numpy(float)
    return output


def destination_probabilities(training: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first_count = np.zeros((8, 9), dtype=float)
    history_count = np.zeros((9, 9, 8, 9), dtype=float)
    np.add.at(
        first_count,
        (training["state"].to_numpy(int), training["future_state_1"].to_numpy(int)),
        1,
    )
    np.add.at(
        history_count,
        (
            training["previous_state_2"].to_numpy(int),
            training["previous_state_1"].to_numpy(int),
            training["state"].to_numpy(int),
            training["future_state_1"].to_numpy(int),
        ),
        1,
    )
    first = (first_count + 1) / (first_count.sum(axis=1, keepdims=True) + 9)
    prior = first[np.newaxis, np.newaxis, :, :]
    history = (history_count + 20 * prior) / (history_count.sum(axis=3, keepdims=True) + 20)
    return first, history


def one_path_probability(
    first: np.ndarray, history: np.ndarray, path: Sequence[int], previous_2: int, previous_1: int
) -> tuple[float, float]:
    p2, p1, current = int(previous_2), int(previous_1), int(path[0])
    hp = fp = 1.0
    for destination in path[1:]:
        hp *= float(history[p2, p1, current, destination])
        fp *= float(first[current, destination])
        p2, p1, current = p1, current, int(destination)
    return (
        float(np.clip(hp, EPSILON, 1 - EPSILON)),
        float(np.clip(fp, EPSILON, 1 - EPSILON)),
    )


def raw_context(
    frame: pd.DataFrame, numeric: Sequence[str], medians: dict[str, float]
) -> sparse.csr_matrix:
    values = (
        frame.loc[:, list(numeric)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(pd.Series(medians))
        .to_numpy(float)
    )
    states = sparse.csr_matrix(np.eye(8)[frame["state"].to_numpy(int)])
    return sparse.hstack((states, sparse.csr_matrix(values)), format="csr")


def candidate_features(
    context: sparse.csr_matrix, indices: np.ndarray, width: int
) -> sparse.csr_matrix:
    identity = sparse.csr_matrix(
        (np.full(len(indices), 0.5), (np.arange(len(indices)), indices)),
        shape=(len(indices), width),
    )
    return sparse.hstack((context, identity), format="csr")


def refit_movement(
    anchors: pd.DataFrame,
    manifest: pd.DataFrame,
    ledger: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[float, float]:
    numeric = tuple(contract["movement_models"]["qcontext_features"]["causal_numeric_controls"])
    realized = expand_realized(anchors, manifest)
    source = anchors.set_index("anchor_id")
    parameter_file = np.load(ROOT / "movement_model_parameters.npz")
    max_prediction_error = 0.0
    max_parameter_error = 0.0
    for family in sorted(manifest["family"].unique()):
        family_manifest = manifest.loc[manifest["family"].eq(family)]
        width = len(family_manifest)
        family_realized = realized.loc[realized["family"].eq(family)]
        for month in QUALIFICATION_MONTHS:
            training = family_realized.loc[family_realized["month"].lt(month)].copy()
            validation = ledger.loc[(ledger["family"] == family) & (ledger["month"] == month)].copy()
            source_rows = source.loc[validation["anchor_id"].to_numpy()].reset_index()
            for column in numeric:
                validation[column] = source_rows[column].to_numpy()
            weights = training["conditional_weight"].to_numpy(float)
            unique = training.drop_duplicates("anchor_id", keep="first")
            medians = {
                column: float(pd.to_numeric(unique[column], errors="coerce").median())
                for column in numeric
            }
            train_raw = raw_context(training, numeric, medians)
            validation_raw = raw_context(validation, numeric, medians)
            scaler = StandardScaler(with_mean=False).fit(train_raw, sample_weight=weights)
            train_context = scaler.transform(train_raw).tocsr()
            validation_context = scaler.transform(validation_raw).tocsr()
            train_candidate = candidate_features(
                train_context, training["candidate_index"].to_numpy(int), width
            )
            validation_candidate = candidate_features(
                validation_context, validation["candidate_index"].to_numpy(int), width
            )
            prefix = f"{family}__{month}"
            max_parameter_error = max(
                max_parameter_error,
                float(
                    np.max(
                        np.abs(
                            parameter_file[f"{prefix}__medians"]
                            - np.asarray([medians[column] for column in numeric])
                        )
                    )
                ),
                float(np.max(np.abs(parameter_file[f"{prefix}__scaler_scale"] - scaler.scale_))),
            )
            for target in TARGETS:
                for horizon in HORIZONS:
                    y = training[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                    for label, train_x, validation_x in (
                        ("qcontext", train_context, validation_context),
                        ("qcandidate", train_candidate, validation_candidate),
                    ):
                        model = LogisticRegression(
                            C=0.2,
                            solver="lbfgs",
                            max_iter=1000,
                            tol=0.0001,
                            random_state=SEED,
                        ).fit(train_x, y, sample_weight=weights)
                        probability = model.predict_proba(validation_x)
                        for tier, values in (
                            ("p75", probability[:, 1] + probability[:, 2]),
                            ("p90", probability[:, 2]),
                        ):
                            expected = validation[f"{label}__{target}__h{horizon}__{tier}"].to_numpy(float)
                            max_prediction_error = max(
                                max_prediction_error, float(np.max(np.abs(values - expected)))
                            )
                        key = f"{prefix}__{target}__h{horizon}__{label}"
                        max_parameter_error = max(
                            max_parameter_error,
                            float(np.max(np.abs(parameter_file[f"{key}__coef"] - model.coef_))),
                            float(
                                np.max(
                                    np.abs(
                                        parameter_file[f"{key}__intercept"] - model.intercept_
                                    )
                                )
                            ),
                            float(
                                np.max(
                                    np.abs(parameter_file[f"{key}__n_iter"] - model.n_iter_)
                                )
                            ),
                        )
    parameter_file.close()
    return max_prediction_error, max_parameter_error


def losses(y: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(np.asarray(probability, float), EPSILON, 1 - EPSILON)
    y = np.asarray(y, float)
    return (
        -(y * np.log(probability) + (1 - y) * np.log(1 - probability)),
        (y - probability) ** 2,
    )


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() else math.nan


def calibration(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray, minimum: int
) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9)
    values: list[tuple[float, float]] = []
    for index in range(10):
        selected = bins == index
        if selected.sum() < minimum or weights[selected].sum() <= 0:
            continue
        error = abs(wmean(y[selected], weights[selected]) - wmean(probability[selected], weights[selected]))
        values.append((weights[selected].sum(), error))
    if not values:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in values)
    return (
        float(sum(weight * error for weight, error in values) / total),
        float(max(error for _, error in values)),
        len(values),
    )


def daily(frame: pd.DataFrame, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    grouped = pd.DataFrame(
        {
            "date": frame["session_date"].astype(str).to_numpy(),
            "value": values * weights,
            "weight": weights,
        }
    ).groupby("date", sort=True).sum()
    grouped = grouped.loc[grouped["weight"] > 0]
    return (grouped["value"] / grouped["weight"]).to_numpy(float)


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = np.asarray(
        [values[index : index + 5].mean() for index in range(0, len(values), 5) if len(values[index : index + 5]) == 5]
    )
    if len(blocks) < 5:
        return math.nan, math.nan
    sampled = np.random.default_rng(seed).choice(
        blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip(values: np.ndarray, seed: int) -> float:
    if len(values) < 10:
        return math.nan
    null = (
        np.random.default_rng(seed)
        .choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values)))
        @ values
    ) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def audit_cell(
    frame: pd.DataFrame, target: str, horizon: int, tier: str, seed: int
) -> dict[str, Any]:
    cutoff = 1 if tier == "p75" else 2
    realized = frame.loc[frame["candidate_occurs"].eq(1)].reset_index(drop=True)
    y = (realized[f"quality_class__{target}__h{horizon}"].to_numpy(int) >= cutoff).astype(int)
    weights = realized["conditional_weight"].to_numpy(float)
    pc = realized[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float)
    pq = realized[f"qcandidate__{target}__h{horizon}__{tier}"].to_numpy(float)
    llc, bc = losses(y, pc)
    llq, bq = losses(y, pq)
    difference = llq - llc
    daily_difference = daily(realized, difference, weights)
    boot = bootstrap(daily_difference, seed)
    qdiff = {
        quarter: wmean(
            difference[realized["quarter"].eq(quarter).to_numpy()],
            weights[realized["quarter"].eq(quarter).to_numpy()],
        )
        for quarter in ("2024_q3", "2024_q4")
    }
    loso = {
        symbol: wmean(
            difference[realized["symbol_norm"].astype(str).ne(symbol).to_numpy()],
            weights[realized["symbol_norm"].astype(str).ne(symbol).to_numpy()],
        )
        for symbol in sorted(realized["symbol_norm"].astype(str).unique())
    }
    cce, ccm, ccbins = calibration(y, pc, weights, 25)
    qce, qcm, qcbins = calibration(y, pq, weights, 25)
    lift = bootstrap(daily(realized, y.astype(float) - pc, weights), seed + 1)
    all_y = (
        frame["candidate_occurs"].to_numpy(bool)
        & (frame[f"quality_class__{target}__h{horizon}"].to_numpy(int) >= cutoff)
    ).astype(int)
    structural = frame["history_probability"].to_numpy(float)
    all_pc = structural * frame[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float)
    all_pq = structural * frame[f"qcandidate__{target}__h{horizon}__{tier}"].to_numpy(float)
    jlc, jbc = losses(all_y, all_pc)
    jlq, jbq = losses(all_y, all_pq)
    ones = np.ones(len(frame))
    jce, jcm, jcbins = calibration(all_y, all_pc, ones, 100)
    jqe, jqm, jqbins = calibration(all_y, all_pq, ones, 100)
    cllc, qll = wmean(llc, weights), wmean(llq, weights)
    jcll, jqll = float(jlc.mean()), float(jlq.mean())
    return {
        "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()),
        "observed_rate": wmean(y, weights),
        "mean_qcontext": wmean(pc, weights),
        "mean_qcandidate": wmean(pq, weights),
        "observed_rate_over_qcontext": wmean(y, weights) / max(wmean(pc, weights), EPSILON),
        "conditional_context_log_loss": cllc,
        "conditional_candidate_log_loss": qll,
        "conditional_relative_log_loss_improvement": (cllc - qll) / cllc,
        "conditional_brier_difference": wmean(bq - bc, weights),
        "conditional_bootstrap_lower": boot[0],
        "conditional_bootstrap_upper": boot[1],
        "maximum_leave_one_stock_out_log_loss_difference": max(loso.values(), default=math.inf),
        "conditional_context_ece": cce,
        "conditional_candidate_ece": qce,
        "conditional_context_maximum_supported_bin_error": ccm,
        "conditional_candidate_maximum_supported_bin_error": qcm,
        "conditional_context_supported_bins": ccbins,
        "conditional_candidate_supported_bins": qcbins,
        "joint_context_log_loss": jcll,
        "joint_candidate_log_loss": jqll,
        "joint_relative_log_loss_improvement": (jcll - jqll) / jcll,
        "joint_brier_difference": float((jbq - jbc).mean()),
        "joint_context_ece": jce,
        "joint_candidate_ece": jqe,
        "joint_context_maximum_supported_bin_error": jcm,
        "joint_candidate_maximum_supported_bin_error": jqm,
        "joint_context_supported_bins": jcbins,
        "joint_candidate_supported_bins": jqbins,
        "lift_bootstrap_lower": lift[0],
        "lift_bootstrap_upper": lift[1],
        "daily_sessions": len(daily_difference),
        "sign_flip_p_value": sign_flip(daily_difference, seed + 2),
        "quarter": qdiff,
        "loso": loso,
    }


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p"] = 1.0
    output["holm_pass"] = False
    for _, positions in output.groupby(["family", "tier"], sort=True).groups.items():
        ordered = sorted(positions, key=lambda position: output.loc[position, "p_value"])
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
    return output


def close(left: Any, right: Any, tolerance: float = 2e-11) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(abs(float(left) - float(right)) <= tolerance)


def run_audit() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text())
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    record("contract_hash", sha256(CONTRACT) == EXPECTED["contract"], sha256(CONTRACT))
    record("runner_hash", sha256(RUNNER) == EXPECTED["runner"], sha256(RUNNER))
    record("anchor_hash", sha256(ANCHOR) == EXPECTED["anchor"], sha256(ANCHOR))
    record(
        "safety_labels",
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled",
        {
            "research_only": contract["research_only"],
            "live_ordering_enabled": contract["live_ordering_enabled"],
            "order_placement": contract["order_placement"],
        },
    )
    lock = json.loads((ROOT / "discovery_lock.json").read_text())
    record(
        "discovery_lock",
        lock["contract_sha256"] == EXPECTED["contract"]
        and lock["runner_sha256"] == EXPECTED["runner"]
        and lock["manifest_sha256"] == EXPECTED["manifest"]
        and lock["movement_outcomes_opened"] is False,
        lock,
    )
    artifact = json.loads((ROOT / "artifact_manifest.json").read_text())["files"]
    mismatches = {
        name: sha256(ROOT / name)
        for name, descriptor in artifact.items()
        if sha256(ROOT / name) != descriptor["sha256"]
    }
    record("artifact_hashes", not mismatches, mismatches)

    state = pd.read_parquet(ANCHOR, columns=STATE_COLUMNS)
    manifest = normalize_manifest(pd.read_csv(ROOT / "candidate_manifest.csv"))
    rebuilt = rebuild_manifest(state, contract)
    manifest_equal = manifest.equals(rebuilt)
    record(
        "outcome_blind_manifest_replay",
        manifest_equal and sha256(ROOT / "candidate_manifest.csv") == EXPECTED["manifest"],
        {"rows": len(rebuilt), "equal": manifest_equal},
    )
    discovery = json.loads((ROOT / "discovery_summary.json").read_text())
    record(
        "discovery_column_guard",
        discovery["phase_1_columns_read"] == STATE_COLUMNS
        and discovery["forbidden_outcome_columns_read"] == []
        and discovery["movement_outcomes_opened"] is False,
        discovery["phase_1_columns_read"],
    )

    anchors = load_anchors(contract)
    ledger = pd.read_parquet(ROOT / "oof_predictions_2024_h2.parquet")
    expected_rows = sum(
        int(
            (
                anchors["month"].isin(QUALIFICATION_MONTHS)
                & anchors["state"].eq(candidate.start_state)
            ).sum()
        )
        for candidate in manifest.itertuples(index=False)
    )
    record(
        "oof_key_surface",
        len(ledger) == expected_rows
        and not ledger.duplicated(["anchor_id", "candidate_id"]).any()
        and set(ledger["month"]) == set(QUALIFICATION_MONTHS),
        {"rows": len(ledger), "expected": expected_rows},
    )

    source = anchors.set_index("anchor_id")
    label_errors = 0
    class_errors = 0
    for candidate in manifest.itertuples(index=False):
        selected = ledger.loc[ledger["candidate_id"].eq(candidate.candidate_id)]
        anchor_rows = source.loc[selected["anchor_id"].to_numpy()].reset_index()
        expected_occurs = occurs(anchor_rows, parse_path(candidate.exact_path)).astype(int)
        label_errors += int(np.sum(expected_occurs != selected["candidate_occurs"].to_numpy(int)))
        for target in TARGETS:
            for horizon in HORIZONS:
                class_errors += int(
                    np.sum(
                        anchor_rows[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                        != selected[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                    )
                )
    record("candidate_occurrence_labels", label_errors == 0, label_errors)
    record("movement_quality_labels", class_errors == 0, class_errors)
    realized = ledger.loc[ledger["candidate_occurs"].eq(1)].copy()
    expected_weight = 1.0 / realized.groupby(["family", "anchor_id"])["candidate_id"].transform("count")
    weight_error = float(np.max(np.abs(expected_weight - realized["conditional_weight"])))
    record("conditional_overlap_weights", weight_error == 0, weight_error)

    structural_history_error = 0.0
    structural_first_error = 0.0
    for month in QUALIFICATION_MONTHS:
        first, history = destination_probabilities(anchors.loc[anchors["month"].lt(month)])
        month_rows = ledger.loc[ledger["month"].eq(month)]
        for candidate in manifest.itertuples(index=False):
            selected = month_rows.loc[month_rows["candidate_id"].eq(candidate.candidate_id)]
            path = parse_path(candidate.exact_path)
            calculated = np.asarray(
                [
                    one_path_probability(first, history, path, p2, p1)
                    for p2, p1 in zip(selected["previous_state_2"], selected["previous_state_1"])
                ]
            )
            structural_history_error = max(
                structural_history_error,
                float(np.max(np.abs(calculated[:, 0] - selected["history_probability"])))
                if len(calculated)
                else 0,
            )
            structural_first_error = max(
                structural_first_error,
                float(np.max(np.abs(calculated[:, 1] - selected["first_order_probability"])))
                if len(calculated)
                else 0,
            )
    record(
        "structural_probability_replay",
        structural_history_error < 1e-14 and structural_first_error < 1e-14,
        {"history_max_abs_error": structural_history_error, "first_max_abs_error": structural_first_error},
    )

    prediction_error, parameter_error = refit_movement(anchors, manifest, ledger, contract)
    record("movement_probability_refit", prediction_error < 2e-12, prediction_error)
    record("movement_parameter_refit", parameter_error < 2e-12, parameter_error)

    support_artifact = pd.read_csv(ROOT / "qualification_support.csv").set_index("candidate_id")
    support_mismatches: list[str] = []
    for candidate in manifest.itertuples(index=False):
        frame = ledger.loc[ledger["candidate_id"].eq(candidate.candidate_id)]
        event = frame.loc[frame["candidate_occurs"].eq(1)]
        values = {
            "compatible_rows": len(frame),
            "realized_rows": len(event),
            "realized_stocks": event["symbol_norm"].nunique(),
            "q3_realized_rows": int(event["quarter"].eq("2024_q3").sum()),
            "q4_realized_rows": int(event["quarter"].eq("2024_q4").sum()),
        }
        observed = support_artifact.loc[candidate.candidate_id]
        if any(int(observed[key]) != int(value) for key, value in values.items()):
            support_mismatches.append(candidate.candidate_id)
    record("support_replay", not support_mismatches, support_mismatches)

    structural_artifact = pd.read_csv(ROOT / "structural_metrics.csv").set_index("candidate_id")
    structural_metric_error = 0.0
    structural_pass_errors = 0
    for candidate in manifest.itertuples(index=False):
        frame = ledger.loc[ledger["candidate_id"].eq(candidate.candidate_id)]
        y = frame["candidate_occurs"].to_numpy(int)
        hp, fp = frame["history_probability"].to_numpy(float), frame["first_order_probability"].to_numpy(float)
        hl, hb = losses(y, hp)
        fl, fb = losses(y, fp)
        he, hm, _ = calibration(y, hp, np.ones(len(y)), 100)
        fe, fm, _ = calibration(y, fp, np.ones(len(y)), 100)
        values = {
            "history_log_loss": hl.mean(),
            "first_order_log_loss": fl.mean(),
            "relative_log_loss_improvement": (fl.mean() - hl.mean()) / fl.mean(),
            "history_brier": hb.mean(),
            "first_order_brier": fb.mean(),
            "history_ece": he,
            "first_order_ece": fe,
            "history_maximum_supported_bin_error": hm,
            "first_order_maximum_supported_bin_error": fm,
        }
        observed = structural_artifact.loc[candidate.candidate_id]
        structural_metric_error = max(
            structural_metric_error,
            max(abs(float(observed[key]) - float(value)) for key, value in values.items()),
        )
        passed = hl.mean() < fl.mean() and hb.mean() < fb.mean() and he <= fe + 0.01 and hm <= fm + 0.02
        structural_pass_errors += int(bool(observed["pass"]) != bool(passed))
    record(
        "structural_metric_replay",
        structural_metric_error < 2e-12 and structural_pass_errors == 0,
        {"max_abs_error": structural_metric_error, "pass_errors": structural_pass_errors},
    )

    cells_artifact = pd.read_csv(ROOT / "quality_cells.csv").set_index(
        ["candidate_id", "target", "horizon", "tier"]
    )
    numeric_columns = [
        "positive_rows",
        "negative_rows",
        "observed_rate",
        "mean_qcontext",
        "mean_qcandidate",
        "observed_rate_over_qcontext",
        "conditional_context_log_loss",
        "conditional_candidate_log_loss",
        "conditional_relative_log_loss_improvement",
        "conditional_brier_difference",
        "conditional_bootstrap_lower",
        "conditional_bootstrap_upper",
        "maximum_leave_one_stock_out_log_loss_difference",
        "conditional_context_ece",
        "conditional_candidate_ece",
        "conditional_context_maximum_supported_bin_error",
        "conditional_candidate_maximum_supported_bin_error",
        "conditional_context_supported_bins",
        "conditional_candidate_supported_bins",
        "joint_context_log_loss",
        "joint_candidate_log_loss",
        "joint_relative_log_loss_improvement",
        "joint_brier_difference",
        "joint_context_ece",
        "joint_candidate_ece",
        "joint_context_maximum_supported_bin_error",
        "joint_candidate_maximum_supported_bin_error",
        "joint_context_supported_bins",
        "joint_candidate_supported_bins",
        "lift_bootstrap_lower",
        "lift_bootstrap_upper",
        "daily_sessions",
        "sign_flip_p_value",
    ]
    cell_max_error = 0.0
    cell_error_count = 0
    recomputed_p: list[dict[str, Any]] = []
    support_lookup = support_artifact["pass"].to_dict()
    for family in sorted(manifest["family"].unique()):
        family_manifest = manifest.loc[manifest["family"].eq(family)]
        for position, candidate in enumerate(family_manifest.itertuples(index=False)):
            frame = ledger.loc[ledger["candidate_id"].eq(candidate.candidate_id)].reset_index(drop=True)
            for target_index, target in enumerate(TARGETS):
                for horizon in HORIZONS:
                    for tier_index, tier in enumerate(TIERS):
                        result = audit_cell(
                            frame,
                            target,
                            horizon,
                            tier,
                            SEED + position * 10000 + target_index * 1000 + horizon * 10 + tier_index,
                        )
                        observed = cells_artifact.loc[(candidate.candidate_id, target, horizon, tier)]
                        for column in numeric_columns:
                            error = abs(float(observed[column]) - float(result[column]))
                            cell_max_error = max(cell_max_error, error)
                            cell_error_count += int(error > 2e-11)
                        if candidate.decision_eligible and support_lookup[candidate.candidate_id]:
                            recomputed_p.append(
                                {
                                    "candidate_id": candidate.candidate_id,
                                    "family": family,
                                    "target": target,
                                    "horizon": horizon,
                                    "tier": tier,
                                    "p_value": result["sign_flip_p_value"],
                                }
                            )
    record(
        "quality_metric_replay",
        cell_error_count == 0,
        {"max_abs_error": cell_max_error, "field_errors": cell_error_count},
    )

    recalculated_holm = holm(pd.DataFrame(recomputed_p)).set_index(
        ["candidate_id", "target", "horizon", "tier"]
    )
    holm_artifact = pd.read_csv(ROOT / "multiplicity.csv").set_index(
        ["candidate_id", "target", "horizon", "tier"]
    )
    same_holm_keys = set(recalculated_holm.index) == set(holm_artifact.index)
    holm_error = max(
        (
            abs(
                float(recalculated_holm.loc[key, "holm_adjusted_p"])
                - float(holm_artifact.loc[key, "holm_adjusted_p"])
            )
            for key in recalculated_holm.index
        ),
        default=0.0,
    )
    holm_pass_error = sum(
        bool(recalculated_holm.loc[key, "holm_pass"])
        != bool(holm_artifact.loc[key, "holm_pass"])
        for key in recalculated_holm.index
    )
    record(
        "holm_replay",
        same_holm_keys and holm_error < 1e-14 and holm_pass_error == 0,
        {"keys_equal": same_holm_keys, "max_abs_error": holm_error, "pass_errors": holm_pass_error},
    )

    grades = pd.read_csv(ROOT / "horizon_grades.csv")
    candidates = pd.read_csv(ROOT / "development_candidates.csv")
    decision = json.loads((ROOT / "decision.json").read_text())
    expected_unqualified = len(manifest) * len(HORIZONS)
    record(
        "grade_and_stop_replay",
        len(grades) == expected_unqualified
        and grades["grade"].eq("development_unqualified").all()
        and candidates.empty
        and decision["total_frozen_development_candidates"] == 0,
        {
            "grade_rows": len(grades),
            "unqualified": int(grades["grade"].eq("development_unqualified").sum()),
            "candidate_rows": len(candidates),
        },
    )
    summary = json.loads((ROOT / "summary.json").read_text())
    record(
        "summary_reconciliation",
        summary["qualification_oof_rows"] == len(ledger)
        and summary["support_pass_candidates"] == int(support_artifact["pass"].sum())
        and summary["structural_pass_candidates"] == int(structural_artifact["pass"].sum())
        and summary["holm_pass_tests"] == int(holm_artifact["holm_pass"].sum())
        and summary["decision"] == decision,
        {
            "oof_rows": len(ledger),
            "support_pass": int(support_artifact["pass"].sum()),
            "structural_pass": int(structural_artifact["pass"].sum()),
            "holm_pass": int(holm_artifact["holm_pass"].sum()),
        },
    )
    record(
        "no_direction_volume_or_later_period",
        summary["direct_volume_fields_used"] == []
        and summary["direction_or_signed_return_used"] is False
        and summary["later_period_scoring_performed"] is False
        and summary["prospective_shadow_read_or_write_performed"] is False,
        {
            "volume": summary["direct_volume_fields_used"],
            "direction": summary["direction_or_signed_return_used"],
            "later": summary["later_period_scoring_performed"],
            "shadow": summary["prospective_shadow_read_or_write_performed"],
        },
    )

    all_passed = all(value["passed"] for value in checks.values())
    result = {
        "audit_id": "causal_state_pattern_discovery_v1_independent_audit",
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
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if not all_passed:
        failed = [name for name, value in checks.items() if not value["passed"]]
        raise AssertionError(f"independent audit failed: {failed}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_audit()
