#!/usr/bin/env python3
"""Independently audit the Route-Competition Fixed-Lead Audit V0.1 artifacts."""

from __future__ import annotations

# ruff: noqa: E402 -- local package roots are resolved before research imports.
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.route_competition_hazard_v0 import (
    BASELINE_FEATURES,
    CHECKPOINTS,
    H1_FEATURES,
    ROUTE_FEATURES,
)

PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-hazard-quick-v0"
    / "artifacts"
    / "primary"
)
DICTIONARY_PATH = (
    REPO_ROOT
    / "research"
    / "slrno-v2"
    / "20260714-regime-loop-handoff"
    / "work"
    / "artifacts"
    / "20260718-loop-event-semantics-v2"
    / "primary"
    / "semantic_loop_dictionary_v2.csv"
)
EXPECTED_DICTIONARY_SHA256 = "9550810616f9249f3a8adf32b08fe17c0e6fdc1cf582466d9d10ee6df639cb7a"
EXPECTED_DICTIONARY_HASH = "497142c8d0ab880e59385da123d9eb2189469e9e3a4a631e0f63eb6fc77030d3"
EXPECTED_PREDECESSOR_COMMIT = "1001693c70e99f92ae77777b0d6b3633777bf7af"
RUNNER = EXPERIMENT_DIR / "run_screen_v01.py"
SAFETY_FLAGS: dict[str, bool] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "fixed_lead_audit": True,
    "next_bar_completion_test": True,
    "two_to_three_bar_advance_test": True,
    "near_complete_prefix_excluded_from_advance_test": True,
    "exact_route_identity_modelled": False,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
    "options_outcomes_opened": False,
    "execution_enabled": False,
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
FROZEN_COHORT = {
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
}


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def independent_fixed_leads(
    source_panel: pd.DataFrame, registered: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in registered.groupby(["symbol", "session"], sort=False)
    }
    empty = registered.iloc[:0]
    leads: list[int] = []
    identities: list[str] = []
    for row in source_panel.itertuples(index=False):
        events = groups.get((str(row.symbol), str(row.session)), empty)
        future = sorted(
            set(
                events.loc[
                    events["bar_ordinal"].astype(int).gt(int(row.checkpoint))
                    & events["bar_ordinal"].astype(int).le(int(row.checkpoint) + 3),
                    "bar_ordinal",
                ].astype(int)
            )
        )
        if not future:
            leads.append(0)
            identities.append("[]")
            continue
        earliest = future[0]
        leads.append(earliest - int(row.checkpoint))
        at_lead = events.loc[events["bar_ordinal"].astype(int).eq(earliest), "semantic_loop_id"]
        identities.append(json.dumps(sorted(set(at_lead.astype(str)))))
    return np.asarray(leads, dtype=int), identities


def independent_dictionary_routes() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct valid oriented route lengths directly from the frozen dictionary."""

    source_hash = sha256_file(DICTIONARY_PATH)
    table = pd.read_csv(DICTIONARY_PATH)
    dictionary_hashes = set(table["dictionary_hash"].astype(str))
    rows: list[dict[str, Any]] = []
    canonical_paths_valid = True
    for row in table.itertuples(index=False):
        semantic_loop_id = str(row.semantic_loop_id)
        canonical = tuple(int(value) for value in json.loads(str(row.canonical_orientation)))
        valid_paths = [
            tuple(int(value) for value in path)
            for path in json.loads(str(row.all_valid_oriented_paths))
        ]
        canonical_paths_valid &= canonical in valid_paths
        canonical_paths_valid &= all(
            len(path) >= 3 and path[0] == path[-1] and len(path) == len(canonical)
            for path in valid_paths
        )
        for path in valid_paths:
            rows.append(
                {
                    "semantic_loop_id": semantic_loop_id,
                    "orientation_id": (
                        f"{semantic_loop_id}__o_{'-'.join(str(value) for value in path)}"
                    ),
                    "dictionary_motif_type": str(row.motif_type),
                    "canonical_total_transitions": len(path) - 1,
                }
            )
    routes = pd.DataFrame(rows)
    manifest = {
        "source_hash_matches": source_hash == EXPECTED_DICTIONARY_SHA256,
        "dictionary_hash_matches": dictionary_hashes == {EXPECTED_DICTIONARY_HASH},
        "canonical_paths_valid": canonical_paths_valid,
        "orientations_unique": not routes.duplicated(["semantic_loop_id", "orientation_id"]).any(),
        "source_sha256": source_hash,
        "dictionary_hashes": sorted(dictionary_hashes),
    }
    return routes, manifest


def independent_prefix_proximity(
    prefixes: pd.DataFrame, canonical_routes: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    current = prefixes.loc[prefixes["bar_ordinal"].astype(int).isin(CHECKPOINTS)].drop_duplicates(
        ["symbol", "session", "bar_ordinal", "semantic_loop_id", "orientation_id"]
    )
    current = current.merge(
        canonical_routes,
        on=["semantic_loop_id", "orientation_id"],
        how="left",
        validate="many_to_one",
    )
    progress = current["progress_states"].astype(int)
    declared_remaining = current["transitions_remaining"].astype(int)
    current = current.assign(
        independently_calculated_remaining=(current["canonical_total_transitions"] - (progress - 1))
    )
    diagnostics = {
        "missing_canonical_orientations": int(current["canonical_total_transitions"].isna().sum()),
        "motif_mismatches": int(
            (
                ~current["motif_type"].astype(str).eq(current["dictionary_motif_type"].astype(str))
            ).sum()
        ),
        "declared_remaining_mismatches": int(
            (~current["independently_calculated_remaining"].eq(declared_remaining)).sum()
        ),
    }
    current["one_away"] = current["independently_calculated_remaining"].eq(1)
    proximity = (
        current.groupby(["symbol", "session", "bar_ordinal"], sort=True)
        .agg(
            minimum_remaining_transitions=("independently_calculated_remaining", "min"),
            number_of_one_transition_away_prefixes=("one_away", "sum"),
        )
        .reset_index()
        .rename(columns={"bar_ordinal": "checkpoint"})
    )
    return proximity, diagnostics


def _max_difference(left: pd.Series, right: pd.Series) -> float:
    return float(np.max(np.abs(left.to_numpy(float) - right.to_numpy(float))))


def manual_probability(frame: pd.DataFrame, specification: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct a stored logistic probability without the fitted-model wrapper."""

    features = [str(value) for value in specification["feature_names"]]
    matrix = frame.loc[:, features].to_numpy(float)
    mean = np.asarray(specification["scaler_mean"], dtype=float)
    scale = np.asarray(specification["scaler_scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    linear = ((matrix - mean) / scale) @ coefficient + float(specification["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))


def independent_core_metrics(
    frame: pd.DataFrame, *, target: str, probability_column: str
) -> dict[str, float]:
    y_true = frame[target].to_numpy(int)
    probability = frame[probability_column].to_numpy(float)
    weights = frame["row_weight"].to_numpy(float)
    return {
        "log_loss": float(log_loss(y_true, probability, sample_weight=weights, labels=[0, 1])),
        "brier_score": float(np.average((probability - y_true) ** 2, weights=weights)),
        "auc": float(roc_auc_score(y_true, probability, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(y_true, probability, sample_weight=weights)
        ),
    }


def independent_pair_increments(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    route: str,
    top_decile_boundaries: Mapping[str, float] | None = None,
) -> dict[str, float]:
    baseline_metrics = independent_core_metrics(
        frame, target=target, probability_column=f"{baseline}_probability"
    )
    route_metrics = independent_core_metrics(
        frame, target=target, probability_column=f"{route}_probability"
    )
    result = {
        "log_loss_improvement": baseline_metrics["log_loss"] - route_metrics["log_loss"],
        "brier_improvement": baseline_metrics["brier_score"] - route_metrics["brier_score"],
        "auc_improvement": route_metrics["auc"] - baseline_metrics["auc"],
        "average_precision_improvement": (
            route_metrics["average_precision"] - baseline_metrics["average_precision"]
        ),
    }
    if top_decile_boundaries is not None:
        precisions: dict[str, float] = {}
        for model in (baseline, route):
            selected = frame.loc[
                frame[f"{model}_probability"].ge(float(top_decile_boundaries[model]))
            ]
            precisions[model] = float(np.average(selected[target], weights=selected["row_weight"]))
        result["top_decile_precision_improvement"] = precisions[route] - precisions[baseline]
    return result


def independent_session_multiplicities(
    sessions: pd.Series, *, draws: int, seed: int
) -> list[np.ndarray]:
    labels = sessions.astype(str).to_numpy()
    unique = np.asarray(sorted(set(labels)), dtype=object)
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        result.append(np.asarray([int(counts.get(value, 0)) for value in labels]))
    return result


def independent_route_bundle_permutation(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    result = frame.copy()
    rng = np.random.default_rng(seed)
    columns = list(ROUTE_FEATURES)
    for _, group in frame.groupby(["period", "session", "checkpoint"], sort=True, dropna=False):
        indices = group.index.to_numpy()
        donors = rng.permutation(indices)
        result.loc[indices, columns] = frame.loc[donors, columns].to_numpy()
    return result


def _maximum_mapping_difference(
    left: Mapping[str, float], right: Mapping[str, Any], keys: Sequence[str]
) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in keys)


def audit_supported_models(
    fixed: pd.DataFrame,
    configurations: Mapping[str, Any],
    coefficients: Mapping[str, Any],
    decision: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Independently verify the complete supported model/resampling path."""

    assessment = fixed.loc[fixed["period"].eq("assessment")]
    definitions = {
        "N0": ("completion_next_1_bar", False),
        "N1": ("completion_next_1_bar", False),
        "A0": ("completion_in_bars_2_or_3", True),
        "A1": ("completion_in_bars_2_or_3", True),
    }
    primary = cast(Mapping[str, Mapping[str, Any]], coefficients["primary_models"])
    maximum_probability_difference = 0.0
    maximum_scaler_difference = 0.0
    primary_metadata_matches = True
    manual_rows: dict[str, int] = {}
    for model, (target, advance_only) in definitions.items():
        population = fixed.loc[fixed["advance_eligible"].eq(1)] if advance_only else fixed
        development = population.loc[population["period"].eq("development")]
        expected_features = H1_FEATURES if model in {"N1", "A1"} else BASELINE_FEATURES
        matrix = development.loc[:, list(expected_features)].to_numpy(float)
        expected_mean = matrix.mean(axis=0)
        expected_scale = matrix.std(axis=0, ddof=0)
        expected_scale = np.where(expected_scale == 0.0, 1.0, expected_scale)
        specification = primary[model]
        maximum_scaler_difference = max(
            maximum_scaler_difference,
            float(
                np.max(
                    np.abs(expected_mean - np.asarray(specification["scaler_mean"], dtype=float))
                )
            ),
            float(
                np.max(
                    np.abs(expected_scale - np.asarray(specification["scaler_scale"], dtype=float))
                )
            ),
        )
        primary_metadata_matches &= bool(
            tuple(str(value) for value in specification["feature_names"]) == expected_features
            and specification["target"] == target
            and specification["population"]
            == ("advance_eligible" if advance_only else "all_resolvable_rows")
            and specification["penalty"] == "l2"
            and float(specification["C"]) == 0.25
            and specification["solver"] == "liblinear"
            and int(specification["max_iter"]) == 300
            and specification["class_weight"] is None
            and int(specification["random_state"]) == 20260722
        )
        sample = population.sort_values("row_id", kind="mergesort").head(100)
        manual = manual_probability(sample, specification)
        maximum_probability_difference = max(
            maximum_probability_difference,
            float(np.max(np.abs(manual - sample[f"{model}_probability"].to_numpy(float)))),
        )
        manual_rows[model] = len(sample)

    immediate_metrics = pd.read_csv(PRIMARY / "immediate_metrics.csv").set_index("model")
    advance_metrics = pd.read_csv(PRIMARY / "advance_metrics.csv").set_index("model")
    maximum_metric_difference = 0.0
    for model, (target, advance_only) in definitions.items():
        population = (
            assessment.loc[assessment["advance_eligible"].eq(1)] if advance_only else assessment
        )
        regenerated = independent_core_metrics(
            population, target=target, probability_column=f"{model}_probability"
        )
        archived = advance_metrics.loc[model] if advance_only else immediate_metrics.loc[model]
        maximum_metric_difference = max(
            maximum_metric_difference,
            _maximum_mapping_difference(
                regenerated,
                cast(Mapping[str, Any], archived.to_dict()),
                ("log_loss", "brier_score", "auc", "average_precision"),
            ),
        )

    boundaries = cast(
        Mapping[str, Mapping[str, float]], configurations["probability_quantile_boundaries"]
    )
    maximum_boundary_difference = 0.0
    for model, (_, advance_only) in definitions.items():
        population = fixed.loc[fixed["advance_eligible"].eq(1)] if advance_only else fixed
        development_probability = population.loc[
            population["period"].eq("development"), f"{model}_probability"
        ]
        maximum_boundary_difference = max(
            maximum_boundary_difference,
            abs(
                float(development_probability.quantile(0.90))
                - float(boundaries[model]["top_decile"])
            ),
            abs(
                float(development_probability.quantile(0.80))
                - float(boundaries[model]["top_quintile"])
            ),
        )
    advance_assessment = assessment.loc[assessment["advance_eligible"].eq(1)]
    real_increments = {
        "immediate": independent_pair_increments(
            assessment,
            target="completion_next_1_bar",
            baseline="N0",
            route="N1",
            top_decile_boundaries={
                "N0": float(boundaries["N0"]["top_decile"]),
                "N1": float(boundaries["N1"]["top_decile"]),
            },
        ),
        "advance": independent_pair_increments(
            advance_assessment,
            target="completion_in_bars_2_or_3",
            baseline="A0",
            route="A1",
            top_decile_boundaries={
                "A0": float(boundaries["A0"]["top_decile"]),
                "A1": float(boundaries["A1"]["top_decile"]),
            },
        ),
    }
    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    draw_differences: list[float] = []
    regenerated_draws: dict[tuple[str, str], list[float]] = {}
    multiplicities = independent_session_multiplicities(
        assessment["session"], draws=15, seed=20260722
    )
    for draw, multiplicity in enumerate(multiplicities):
        sampled = assessment.copy()
        sampled["row_weight"] *= multiplicity
        sampled = sampled.loc[sampled["row_weight"].gt(0)]
        for comparison, target, baseline, route, advance_only in (
            ("immediate", "completion_next_1_bar", "N0", "N1", False),
            ("advance", "completion_in_bars_2_or_3", "A0", "A1", True),
        ):
            population = sampled.loc[sampled["advance_eligible"].eq(1)] if advance_only else sampled
            increments = independent_pair_increments(
                population,
                target=target,
                baseline=baseline,
                route=route,
                top_decile_boundaries={
                    baseline: float(boundaries[baseline]["top_decile"]),
                    route: float(boundaries[route]["top_decile"]),
                },
            )
            for statistic, value in increments.items():
                if comparison == "immediate" and statistic == "top_decile_precision_improvement":
                    continue
                archived = bootstrap.loc[
                    bootstrap["record_type"].eq("draw")
                    & bootstrap["comparison"].eq(comparison)
                    & bootstrap["draw"].eq(draw)
                    & bootstrap["statistic"].eq(statistic),
                    "value",
                ]
                draw_differences.append(abs(value - float(archived.iloc[0])))
                regenerated_draws.setdefault((comparison, statistic), []).append(value)
    interval_differences: list[float] = []
    for (comparison, statistic), values in regenerated_draws.items():
        for level in (0.80, 0.90, 0.95):
            alpha = 1.0 - level
            expected = (
                float(np.quantile(values, alpha / 2.0)),
                float(np.quantile(values, 1.0 - alpha / 2.0)),
            )
            archived = bootstrap.loc[
                bootstrap["record_type"].eq("interval")
                & bootstrap["comparison"].eq(comparison)
                & bootstrap["statistic"].eq(statistic)
                & bootstrap["interval_level"].eq(level)
            ].iloc[0]
            interval_differences.extend(
                [
                    abs(expected[0] - float(archived["lower"])),
                    abs(expected[1] - float(archived["upper"])),
                ]
            )

    route_null = pd.read_csv(PRIMARY / "route_null_metrics.csv")
    null_models = cast(Mapping[str, Mapping[str, Any]], coefficients["route_null_models"])
    maximum_null_difference = 0.0
    null_hash_mismatches = 0
    regenerated_nulls: dict[tuple[str, str], list[float]] = {}
    for comparison, target, baseline, route, advance_only, seed_start in (
        ("immediate", "completion_next_1_bar", "N0", "N1", False, 20260731),
        ("advance", "completion_in_bars_2_or_3", "A0", "A1", True, 20260803),
    ):
        population = fixed.loc[fixed["advance_eligible"].eq(1)] if advance_only else fixed
        real_assessment = (
            assessment.loc[assessment["advance_eligible"].eq(1)] if advance_only else assessment
        )
        for draw in range(3):
            permuted = independent_route_bundle_permutation(population, seed=seed_start + draw)
            permuted_assessment = permuted.loc[permuted["period"].eq("assessment")]
            probability = manual_probability(
                permuted_assessment, null_models[f"{comparison}_{draw}"]
            )
            null_frame = real_assessment.copy()
            null_frame[f"{route}_probability"] = probability
            increments = independent_pair_increments(
                null_frame, target=target, baseline=baseline, route=route
            )
            archived = route_null.loc[
                route_null["record_type"].eq("draw")
                & route_null["comparison"].eq(comparison)
                & route_null["draw"].eq(draw)
            ].iloc[0]
            expected_hash = stable_frame_hash(
                permuted,
                ["period", "session", "checkpoint", "symbol", *ROUTE_FEATURES],
            )
            null_hash_mismatches += int(expected_hash != str(archived["route_bundle_hash"]))
            maximum_null_difference = max(
                maximum_null_difference,
                _maximum_mapping_difference(
                    increments,
                    cast(Mapping[str, Any], archived.to_dict()),
                    ("log_loss_improvement", "brier_improvement", "auc_improvement"),
                ),
            )
            for statistic in (
                "log_loss_improvement",
                "brier_improvement",
                "auc_improvement",
            ):
                regenerated_nulls.setdefault((comparison, statistic), []).append(
                    increments[statistic]
                )

    def stability(
        population: pd.DataFrame,
        *,
        target: str,
        baseline: str,
        route: str,
        group_column: str,
    ) -> tuple[int, int]:
        positive = 0
        adverse = 0
        for _, group in population.groupby(group_column, sort=True):
            increment = independent_pair_increments(
                group, target=target, baseline=baseline, route=route
            )
            positive += int(increment["log_loss_improvement"] > 0.0)
            adverse += int(
                increment["log_loss_improvement"] < -0.005
                or increment["brier_improvement"] < -0.002
            )
        return positive, adverse

    immediate_positive_months, _ = stability(
        assessment,
        target="completion_next_1_bar",
        baseline="N0",
        route="N1",
        group_column="year_month",
    )
    advance_positive_months, _ = stability(
        advance_assessment,
        target="completion_in_bars_2_or_3",
        baseline="A0",
        route="A1",
        group_column="year_month",
    )
    _, immediate_adverse_checkpoints = stability(
        assessment,
        target="completion_next_1_bar",
        baseline="N0",
        route="N1",
        group_column="checkpoint",
    )
    _, advance_adverse_checkpoints = stability(
        advance_assessment,
        target="completion_in_bars_2_or_3",
        baseline="A0",
        route="A1",
        group_column="checkpoint",
    )

    def bootstrap_lower(comparison: str, statistic: str) -> float:
        return float(np.quantile(regenerated_draws[(comparison, statistic)], 0.10))

    def maximum_stock_share(population: pd.DataFrame) -> float:
        return float(
            population.groupby("symbol")["row_weight"].sum().max() / population["row_weight"].sum()
        )

    support_passed = bool(
        len(assessment) >= 24_000
        and assessment["session"].nunique() >= 140
        and assessment["symbol"].nunique() >= 15
        and assessment["year_month"].nunique() >= 8
        and int(assessment["completion_next_1_bar"].sum()) >= 300
        and maximum_stock_share(assessment) <= 0.10
        and len(advance_assessment) >= 18_000
        and advance_assessment["session"].nunique() >= 140
        and advance_assessment["symbol"].nunique() >= 15
        and advance_assessment["year_month"].nunique() >= 8
        and int(advance_assessment["completion_in_bars_2_or_3"].sum()) >= 250
        and np.isfinite(advance_assessment.loc[:, list(H1_FEATURES)].to_numpy(float))
        .all(axis=1)
        .mean()
        >= 0.90
        and maximum_stock_share(advance_assessment) <= 0.10
    )
    immediate_null_passed = any(
        real_increments["immediate"][statistic] > max(regenerated_nulls[("immediate", statistic)])
        for statistic in ("log_loss_improvement", "brier_improvement")
    )
    advance_null_passed = any(
        real_increments["advance"][statistic] > max(regenerated_nulls[("advance", statistic)])
        for statistic in ("log_loss_improvement", "brier_improvement")
    )
    immediate_passed = bool(
        real_increments["immediate"]["log_loss_improvement"] > 0.0
        and real_increments["immediate"]["brier_improvement"] > 0.0
        and real_increments["immediate"]["auc_improvement"] >= 0.0
        and bootstrap_lower("immediate", "log_loss_improvement") >= 0.0
        and bootstrap_lower("immediate", "brier_improvement") >= 0.0
        and immediate_positive_months >= 5
        and immediate_adverse_checkpoints == 0
        and immediate_null_passed
        and support_passed
    )
    advance_passed = bool(
        real_increments["advance"]["log_loss_improvement"] > 0.0
        and real_increments["advance"]["brier_improvement"] > 0.0
        and real_increments["advance"]["auc_improvement"] >= 0.0
        and real_increments["advance"]["average_precision_improvement"] > 0.0
        and bootstrap_lower("advance", "log_loss_improvement") >= 0.0
        and bootstrap_lower("advance", "brier_improvement") >= 0.0
        and bootstrap_lower("advance", "average_precision_improvement") >= 0.0
        and advance_positive_months >= 5
        and advance_adverse_checkpoints == 0
        and advance_null_passed
        and support_passed
    )
    supported_state_rates: list[float] = []
    for _, group in advance_assessment.groupby("route_resolution_state", sort=True):
        positives = int(group["completion_in_bars_2_or_3"].sum())
        if len(group) >= 100 and positives >= 25:
            supported_state_rates.append(
                float(np.average(group["completion_in_bars_2_or_3"], weights=group["row_weight"]))
            )
    descriptive = bool(
        len(supported_state_rates) >= 2 and max(supported_state_rates) > min(supported_state_rates)
    )
    a0_metrics = independent_core_metrics(
        advance_assessment,
        target="completion_in_bars_2_or_3",
        probability_column="A0_probability",
    )
    base_rate = float(
        np.average(
            advance_assessment["completion_in_bars_2_or_3"],
            weights=advance_assessment["row_weight"],
        )
    )
    constant_log_loss = float(
        -np.average(
            advance_assessment["completion_in_bars_2_or_3"] * np.log(max(base_rate, 1e-12))
            + (1 - advance_assessment["completion_in_bars_2_or_3"])
            * np.log(max(1.0 - base_rate, 1e-12)),
            weights=advance_assessment["row_weight"],
        )
    )
    baseline_meaningful = bool(
        a0_metrics["auc"] > 0.5 and a0_metrics["log_loss"] < constant_log_loss
    )
    if immediate_passed and advance_passed:
        expected_decision = "route_competition_adds_immediate_and_advance_warning"
    elif advance_passed:
        expected_decision = "route_competition_adds_advance_warning"
    elif immediate_passed:
        expected_decision = "route_competition_is_imminent_confirmation_only"
    elif descriptive:
        expected_decision = "descriptive_fixed_lead_structure_only"
    elif baseline_meaningful:
        expected_decision = "compressed_transition_baseline_only"
    else:
        expected_decision = "no_fixed_lead_route_increment"
    expected_statuses = {
        "immediate_completion_status": "supported" if immediate_passed else "not_supported",
        "advance_completion_status": (
            "supported"
            if advance_passed
            else ("descriptive_only" if descriptive else "not_supported")
        ),
        "non_imminent_route_status": (
            "supported"
            if advance_passed
            else ("descriptive_only" if descriptive else "not_supported")
        ),
    }
    maximum_decision_increment_difference = max(
        _maximum_mapping_difference(
            real_increments[comparison],
            cast(Mapping[str, Any], decision[f"{comparison}_increments"]),
            (
                "log_loss_improvement",
                "brier_improvement",
                "auc_improvement",
                "average_precision_improvement",
                "top_decile_precision_improvement",
            ),
        )
        for comparison in ("immediate", "advance")
    )

    checks = {
        "development_only_preprocessing": bool(
            int(configurations["primary_models_fitted"]) == 4
            and set(primary) == {"N0", "N1", "A0", "A1"}
            and all(int(specification["n_jobs"]) == 1 for specification in primary.values())
            and primary_metadata_matches
            and maximum_scaler_difference <= 1e-12
            and maximum_boundary_difference <= 1e-12
        ),
        "model_coefficients_and_manual_probabilities": bool(
            all(rows >= 100 for rows in manual_rows.values())
            and maximum_probability_difference <= 1e-12
        ),
        "independent_proper_scores": maximum_metric_difference <= 1e-12,
        "session_bootstrap": bool(
            len(multiplicities) == 15
            and max(draw_differences, default=float("inf")) <= 1e-12
            and max(interval_differences, default=float("inf")) <= 1e-12
        ),
        "route_bundle_null": bool(
            len(null_models) == 6 and null_hash_mismatches == 0 and maximum_null_difference <= 1e-12
        ),
        "supported_determinism_artifact": bool(
            determinism["passed"]
            and determinism["models_refit"] == ["N0", "N1", "A0", "A1"]
            and int(determinism["row_identity_mismatches"]) == 0
            and int(determinism["lead_label_mismatches"]) == 0
            and int(determinism["advance_eligibility_mismatches"]) == 0
            and float(determinism["maximum_probability_difference"]) <= 1e-12
            and float(determinism["maximum_metric_difference"]) <= 1e-12
            and bool(determinism["final_decision_match"])
        ),
        "supported_decision_artifact": bool(
            decision["blocker"] is None
            and decision["primary_decision"] == expected_decision
            and all(decision[key] == value for key, value in expected_statuses.items())
            and maximum_decision_increment_difference <= 1e-12
            and int(decision["models_fitted"]) == 4
            and int(decision["bootstrap_draws_executed"]) == 15
            and decision["route_null_refits_executed"] == {"immediate": 3, "advance": 3}
        ),
    }
    diagnostics = {
        "manual_probability_rows_per_model": manual_rows,
        "maximum_manual_probability_difference": maximum_probability_difference,
        "maximum_development_scaler_difference": maximum_scaler_difference,
        "maximum_development_boundary_difference": maximum_boundary_difference,
        "maximum_independent_metric_difference": maximum_metric_difference,
        "maximum_bootstrap_draw_difference": max(draw_differences, default=None),
        "maximum_bootstrap_interval_difference": max(interval_differences, default=None),
        "maximum_route_null_difference": maximum_null_difference,
        "route_null_hash_mismatches": null_hash_mismatches,
        "independently_regenerated_primary_decision": expected_decision,
        "maximum_decision_increment_difference": maximum_decision_increment_difference,
    }
    return checks, diagnostics


def run_audit() -> dict[str, Any]:
    contract = read_json(PRIMARY / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    source_manifest = read_json(PRIMARY / "source_manifest.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    predecessor_reconstruction = read_json(PRIMARY / "predecessor_reconstruction.json")
    lead_manifest = read_json(PRIMARY / "lead_target_manifest.json")
    proximity_manifest = read_json(PRIMARY / "prefix_proximity_manifest.json")
    configurations = read_json(PRIMARY / "model_configurations.json")
    coefficients = read_json(PRIMARY / "model_coefficients.json")
    determinism = read_json(PRIMARY / "determinism_check.json")
    fixed = pd.read_parquet(PRIMARY / "fixed_lead_panel.parquet")
    assessment = fixed.loc[fixed["period"].eq("assessment")]
    source_panel = pd.read_parquet(PREDECESSOR / "decision_panel.parquet")
    source_assessment = pd.read_parquet(PREDECESSOR / "assessment_predictions.parquet")
    ledger = pd.read_parquet(PREDECESSOR / "route_competition_ledger.parquet")
    registered = ledger.loc[ledger["ledger_kind"].eq("registered_completion")]
    prefixes = ledger.loc[ledger["ledger_kind"].eq("active_prefix")]
    canonical_routes, dictionary_audit = independent_dictionary_routes()

    checks: dict[str, bool] = {}
    checks["safety_flags"] = all(
        contract.get(key) == expected
        and decision.get(key) == expected
        and source_manifest.get(key) == expected
        for key, expected in SAFETY_FLAGS.items()
    )
    protected_start = pd.Timestamp("2025-08-23T00:00:00Z")
    decision_protected_rows = int(
        pd.to_datetime(fixed["checkpoint_timestamp_utc"], utc=True, errors="raise")
        .ge(protected_start)
        .sum()
    )
    structural_protected_rows = int(
        pd.to_datetime(ledger["available_timestamp_utc"], utc=True, errors="raise")
        .ge(protected_start)
        .sum()
    )
    checks["dates_and_protected_boundary"] = bool(
        fixed["session"].astype(str).between("2024-01-01", "2025-08-22").all()
        and decision_protected_rows == 0
        and structural_protected_rows == 0
        and int(protected["protected_decision_rows_materialised"]) == decision_protected_rows
        and int(protected["protected_structural_rows_materialised"]) == structural_protected_rows
        and int(protected["protected_rows_materialised"]) == 0
        and bool(protected["passed"])
    )
    checks["frozen_cohort"] = set(fixed["symbol"].astype(str)) == FROZEN_COHORT
    checks["eight_checkpoints"] = tuple(sorted(fixed["checkpoint"].unique())) == CHECKPOINTS

    left = source_panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    right = fixed.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    row_mismatches = abs(len(left) - len(right)) + int(
        (left["row_id"].astype(str) != right["row_id"].astype(str)).sum()
    )
    maximum_feature_difference = float(
        np.max(
            np.abs(
                left.loc[:, [*BASELINE_FEATURES, *ROUTE_FEATURES]].to_numpy(float)
                - right.loc[:, [*BASELINE_FEATURES, *ROUTE_FEATURES]].to_numpy(float)
            )
        )
    )
    maximum_predecessor_probability_difference = float(
        np.max(
            np.abs(
                left.loc[:, ["H0_probability", "H1_probability"]].to_numpy(float)
                - right.loc[:, ["H0_probability", "H1_probability"]].to_numpy(float)
            )
        )
    )
    assessment_ids = assessment["row_id"].astype(str).tolist()
    source_assessment_ids = source_assessment["row_id"].astype(str).tolist()
    assessment_row_mismatches = abs(len(assessment_ids) - len(source_assessment_ids)) + sum(
        first != second
        for first, second in zip(assessment_ids, source_assessment_ids, strict=False)
    )
    checks["predecessor_panel_reconstruction"] = bool(
        row_mismatches == 0
        and assessment_row_mismatches == 0
        and maximum_feature_difference <= 1e-12
        and maximum_predecessor_probability_difference <= 1e-12
        and bool(predecessor_reconstruction["passed"])
    )

    independent_lead, independent_identities = independent_fixed_leads(left, registered)
    lead_mismatches = int((independent_lead != right["first_completion_lead"].to_numpy(int)).sum())
    identity_mismatches = sum(
        first != second
        for first, second in zip(
            independent_identities,
            right["first_completion_semantic_loop_ids"].astype(str),
            strict=True,
        )
    )
    checks["earliest_completion_leads_0_1_2_3"] = bool(
        lead_mismatches == 0
        and identity_mismatches == 0
        and set(independent_lead) == {0, 1, 2, 3}
        and int(lead_manifest["unresolved_rows_excluded"]) == 0
    )
    checks["lead_targets"] = bool(
        right["completion_next_1_bar"].to_numpy(int).tolist()
        == (independent_lead == 1).astype(int).tolist()
        and right["completion_in_bars_2_or_3"].to_numpy(int).tolist()
        == np.isin(independent_lead, [2, 3]).astype(int).tolist()
    )

    independent_proximity, prefix_diagnostics = independent_prefix_proximity(
        prefixes, canonical_routes
    )
    reconstructed = left.loc[:, ["row_id", "symbol", "session", "checkpoint"]].merge(
        independent_proximity,
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    reconstructed["number_of_one_transition_away_prefixes"] = (
        reconstructed["number_of_one_transition_away_prefixes"].fillna(0).astype(int)
    )
    reconstructed["one_away"] = (
        reconstructed["number_of_one_transition_away_prefixes"].gt(0).astype(int)
    )
    proximity_count_mismatches = int(
        (
            reconstructed["number_of_one_transition_away_prefixes"].to_numpy(int)
            != right["number_of_one_transition_away_prefixes"].to_numpy(int)
        ).sum()
    )
    one_away_mismatches = int(
        (
            reconstructed["one_away"].to_numpy(int)
            != right["any_prefix_one_transition_from_completion"].to_numpy(int)
        ).sum()
    )
    prefix_checkpoint_rows = prefixes.loc[prefixes["bar_ordinal"].astype(int).isin(CHECKPOINTS)]
    remaining_mismatches = prefix_diagnostics["declared_remaining_mismatches"]
    checks["every_prefix_remaining_transition_count"] = bool(
        remaining_mismatches == 0
        and prefix_diagnostics["missing_canonical_orientations"] == 0
        and prefix_diagnostics["motif_mismatches"] == 0
        and all(bool(value) for key, value in dictionary_audit.items() if key.endswith("matches"))
        and bool(dictionary_audit["canonical_paths_valid"])
        and bool(dictionary_audit["orientations_unique"])
        and set(prefix_checkpoint_rows["motif_type"].astype(str))
        == {"primitive", "repeat", "composite"}
        and int(proximity_manifest["active_prefix_rows_checked"]) == len(prefix_checkpoint_rows)
    )
    checks["one_transition_away_prefix_flag"] = bool(
        proximity_count_mismatches == 0 and one_away_mismatches == 0
    )
    expected_eligible = (independent_lead != 1) & reconstructed["one_away"].to_numpy(int).astype(
        bool
    ).__invert__()
    eligibility_mismatches = int(
        (expected_eligible.astype(int) != right["advance_eligible"].to_numpy(int)).sum()
    )
    advance = right.loc[right["period"].eq("assessment") & right["advance_eligible"].eq(1)]
    checks["advance_eligible_population"] = bool(
        eligibility_mismatches == 0
        and not advance["first_completion_lead"].eq(1).any()
        and not advance["any_prefix_one_transition_from_completion"].eq(1).any()
    )
    checks["frozen_feature_surfaces"] = bool(
        tuple(configurations["H0_features"]) == BASELINE_FEATURES
        and tuple(configurations["H1_features"]) == H1_FEATURES
        and tuple(H1_FEATURES) == (*BASELINE_FEATURES, *ROUTE_FEATURES)
        and np.isfinite(right.loc[:, list(H1_FEATURES)].to_numpy(float)).all()
    )
    immediate_metrics = pd.read_csv(PRIMARY / "immediate_metrics.csv")
    advance_metrics = pd.read_csv(PRIMARY / "advance_metrics.csv")
    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    route_null = pd.read_csv(PRIMARY / "route_null_metrics.csv")
    support = cast(Mapping[str, Any], decision["support"])
    theoretical_rows = int(assessment["session"].nunique()) * len(FROZEN_COHORT) * len(CHECKPOINTS)
    calculated_support = {
        "theoretical_eligible_rows": theoretical_rows,
        "retained_rows": len(assessment),
        "retention": len(assessment) / theoretical_rows,
        "next_bar_positive_outcomes": int(assessment["completion_next_1_bar"].sum()),
        "advance_rows": len(advance),
        "advance_positive_outcomes": int(advance["completion_in_bars_2_or_3"].sum()),
    }
    expected_support_blocker = (
        "blocked_insufficient_advance_positive_support"
        if calculated_support["advance_positive_outcomes"] < 250
        else None
    )
    checks["corrected_support_and_decision_logic"] = bool(
        all(
            (
                abs(float(support[key]) - float(value)) <= 1e-15
                if key == "retention"
                else int(support[key]) == int(value)
            )
            for key, value in calculated_support.items()
        )
        and decision["blocker"] == expected_support_blocker
        and (
            expected_support_blocker is None
            or (
                decision["primary_decision"] == expected_support_blocker
                and decision["immediate_completion_status"] == "descriptive_only"
                and decision["advance_completion_status"] == "insufficient_support"
                and decision["non_imminent_route_status"] == "insufficient_support"
            )
        )
    )
    model_diagnostics: dict[str, Any]
    if expected_support_blocker is not None:
        checks["development_only_preprocessing_not_opened_after_stop"] = bool(
            int(configurations["primary_models_fitted"]) == 0
            and configurations["planned_primary_models"] == ["N0", "N1", "A0", "A1"]
        )
        checks["model_coefficients_and_manual_probabilities_not_applicable"] = bool(
            coefficients["primary_models"] == {}
            and int(coefficients["models_fitted"]) == 0
            and decision["primary_decision"] == expected_support_blocker
        )
        checks["proper_scores_not_opened_after_stop"] = bool(
            immediate_metrics.empty and advance_metrics.empty
        )
        checks["bootstrap_not_opened_after_stop"] = bool(
            bootstrap.empty
            and int(configurations["bootstrap_draws_executed"]) == 0
            and int(configurations["planned_bootstrap_draws"]) == 15
        )
        checks["route_bundle_null_not_opened_after_stop"] = bool(
            route_null.empty
            and configurations["route_null_refits_executed"] == {"immediate": 0, "advance": 0}
            and configurations["planned_route_null_refits"] == {"immediate": 3, "advance": 3}
        )
        checks["determinism_artifact"] = bool(
            determinism["passed"]
            and int(determinism["lead_label_mismatches"]) == 0
            and int(determinism["advance_eligibility_mismatches"]) == 0
            and determinism["maximum_probability_difference"] is None
            and determinism["probability_check_status"] == "not_applicable_pre_model_support_stop"
            and int(determinism["probability_comparison_rows"]) == 0
            and bool(determinism["final_decision_match"])
        )
        model_diagnostics = {
            "manual_probability_rows_per_model": 0,
            "model_checks_status": "not_applicable_pre_model_support_stop",
        }
    else:
        supported_checks, model_diagnostics = audit_supported_models(
            right, configurations, coefficients, decision, determinism
        )
        checks.update(supported_checks)
    checks["artifact_identity"] = bool(
        sha256_file(PREDECESSOR / "decision_panel.parquet")
        == source_manifest["predecessor_artifact_hashes"]["decision_panel.parquet"]
        and source_manifest["predecessor_commit"] == EXPECTED_PREDECESSOR_COMMIT
        and source_manifest["semantic_loop_dictionary"]["source_sha256"]
        == EXPECTED_DICTIONARY_SHA256
        and source_manifest["semantic_loop_dictionary"]["dictionary_hash"]
        == EXPECTED_DICTIONARY_HASH
        and stable_frame_hash(
            left,
            ["row_id", *BASELINE_FEATURES, *ROUTE_FEATURES, "H0_probability", "H1_probability"],
        )
        == stable_frame_hash(
            right,
            ["row_id", *BASELINE_FEATURES, *ROUTE_FEATURES, "H0_probability", "H1_probability"],
        )
    )

    passed = all(checks.values())
    return {
        **SAFETY_FLAGS,
        "auditor": "audit_screen_v01.py",
        "independent_artifact_reload": True,
        "pre_model_support_blocker": expected_support_blocker is not None,
        "model_refits_performed": 0,
        **model_diagnostics,
        "bootstrap_refits_performed": 0,
        "route_null_refits_performed": 0,
        "row_identity_mismatches": row_mismatches,
        "assessment_row_identity_mismatches": assessment_row_mismatches,
        "lead_label_mismatches": lead_mismatches,
        "lead_identity_mismatches": identity_mismatches,
        "prefix_remaining_transition_mismatches": remaining_mismatches,
        "missing_canonical_orientations": prefix_diagnostics["missing_canonical_orientations"],
        "prefix_motif_mismatches": prefix_diagnostics["motif_mismatches"],
        "protected_decision_rows_materialised": decision_protected_rows,
        "protected_structural_rows_materialised": structural_protected_rows,
        "prefix_proximity_count_mismatches": proximity_count_mismatches,
        "one_transition_away_flag_mismatches": one_away_mismatches,
        "advance_eligibility_mismatches": eligibility_mismatches,
        "maximum_predecessor_feature_difference": maximum_feature_difference,
        "maximum_predecessor_probability_difference": (maximum_predecessor_probability_difference),
        "checks": checks,
        "passed": passed,
    }


def main() -> int:
    result = run_audit()
    if result["passed"]:
        runner = load_module(RUNNER, "route_fixed_lead_report_finalizer")
        report = cast(dict[str, Any], runner.finalize_report(PRIMARY, audit=result))
        result["report_sha256"] = report["sha256"]
        result["report_copies_match"] = bool(report["copies_match"])
        cast(dict[str, bool], result["checks"])["report_copies_match"] = bool(
            report["copies_match"]
        )
        result["passed"] = all(cast(Mapping[str, bool], result["checks"]).values())
    (PRIMARY / "lightweight_audit.json").write_text(canonical_json(result), encoding="utf-8")
    print(canonical_json(result), end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
