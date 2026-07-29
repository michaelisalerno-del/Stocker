"""Independent audit for the dense broad-conflict advance screen V0.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stocker_research.broad_conflict_advance_hazard_v02 import (  # noqa: E402
    BASELINE_NON_CLOCK_FEATURES,
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    DENSE_H1_FEATURES,
    ROUTE_FEATURES,
)

DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
V0_PRIMARY = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-hazard-quick-v0"
    / "artifacts"
    / "primary"
)
V01_PRIMARY = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-fixed-lead-audit-v01"
    / "artifacts"
    / "primary"
)
DICTIONARY = (
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
PROTECTED_START = pd.Timestamp("2025-08-23", tz="UTC")
DEVELOPMENT_START = pd.Timestamp("2024-01-01", tz="UTC")
ASSESSMENT_START = pd.Timestamp("2025-01-01", tz="UTC")
READ_END = pd.Timestamp("2025-08-22 23:59:59.999999999", tz="UTC")
BOOTSTRAP_DRAWS = 15
BOOTSTRAP_SEED = 20260722
NULL_REFITS = 3
NULL_SEED = 20260811
ROUTE_STATES = (
    "BROAD_CONFLICT",
    "NARROWING",
    "DOMINANT_ROUTE",
    "LOW_ROUTE_SUPPORT",
    "OTHER",
)
SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "dense_even_checkpoints": True,
    "clean_two_to_three_bar_advance_target": True,
    "next_bar_completions_excluded": True,
    "one_transition_away_prefixes_excluded": True,
    "broad_route_conflict_test": True,
    "exact_route_identity_modelled": False,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
    "options_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "predecessor_reconstruction.json",
    "dense_checkpoint_manifest.json",
    "lead_target_manifest.json",
    "prefix_proximity_manifest.json",
    "dense_advance_panel.parquet",
    "weight_audit.csv",
    "lead_support.csv",
    "lead_route_diagnostics.csv",
    "route_resolution_state_metrics.csv",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "pooled_metrics.csv",
    "checkpoint_metrics.csv",
    "checkpoint_group_metrics.csv",
    "monthly_metrics.csv",
    "subgroup_metrics.csv",
    "bootstrap_metrics.csv",
    "route_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "determinism_check.json",
    "report.md",
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def weighted_rate(frame: pd.DataFrame, target: str) -> float:
    return float(np.average(frame[target], weights=frame["row_weight"]))


def manual_probabilities(frame: pd.DataFrame, specification: Mapping[str, Any]) -> np.ndarray:
    features = [str(value) for value in specification["feature_names"]]
    matrix = frame.loc[:, features].to_numpy(float)
    mean = np.asarray(specification["scaler_mean"], dtype=float)
    scale = np.asarray(specification["scaler_scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    intercept = float(specification["intercept"])
    logits = ((matrix - mean) / scale) @ coefficient + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0)))


def independent_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    labels = frame["completion_in_bars_2_or_3"].to_numpy(int)
    probabilities = frame[probability_column].to_numpy(float)
    weights = frame["row_weight"].to_numpy(float)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return {
        "log_loss": float(
            -np.average(
                labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped),
                weights=weights,
            )
        ),
        "brier_score": float(np.average((probabilities - labels) ** 2, weights=weights)),
        "auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=weights)
        ),
    }


def independent_route_states(
    frame: pd.DataFrame, thresholds: Mapping[str, Sequence[float]]
) -> pd.Series:
    labels = pd.Series("OTHER", index=frame.index, dtype="string")
    broad = frame["prefix_family_entropy"].ge(
        float(thresholds["prefix_family_entropy"][2])
    ) & frame["top_minus_second_prefix_depth"].le(
        float(thresholds["top_minus_second_prefix_depth"][0])
    )
    narrowing = frame["active_prefix_count_change_last_3_bars"].lt(0) & frame[
        "depth_margin_change_last_3_bars"
    ].gt(0)
    dominant = frame["top_prefix_depth_fraction"].ge(
        float(thresholds["top_prefix_depth_fraction"][2])
    ) & frame["top_minus_second_prefix_depth"].ge(
        float(thresholds["top_minus_second_prefix_depth"][2])
    )
    low = frame["active_prefix_count"].le(2)
    labels.loc[low] = "LOW_ROUTE_SUPPORT"
    labels.loc[dominant] = "DOMINANT_ROUTE"
    labels.loc[narrowing] = "NARROWING"
    labels.loc[broad] = "BROAD_CONFLICT"
    return labels


def independent_permutation(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    output = frame.copy()
    rng = np.random.default_rng(seed)
    columns = list(ROUTE_FEATURES)
    for _, group in frame.groupby(["period", "session", "checkpoint"], sort=True, dropna=False):
        indices = group.index.to_numpy()
        donors = rng.permutation(indices)
        output.loc[indices, columns] = frame.loc[donors, columns].to_numpy()
    return output


def completion_leads(panel: pd.DataFrame, ledger: pd.DataFrame) -> np.ndarray:
    keys = {
        (str(row.symbol), str(row.session), int(row.bar_ordinal))
        for row in ledger.loc[
            ledger["ledger_kind"].eq("registered_completion"),
            ["symbol", "session", "bar_ordinal"],
        ].itertuples(index=False)
    }
    result = np.zeros(len(panel), dtype=np.int8)
    for lead in (1, 2, 3):
        present = np.fromiter(
            (
                (str(row.symbol), str(row.session), int(row.checkpoint) + lead) in keys
                for row in panel[["symbol", "session", "checkpoint"]].itertuples(index=False)
            ),
            dtype=bool,
            count=len(panel),
        )
        result[(result == 0) & present] = lead
    return result


def prefix_audit(panel: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, int | float | bool]:
    dictionary = pd.read_csv(DICTIONARY)
    orientation_rows: list[dict[str, Any]] = []
    for row in dictionary.itertuples(index=False):
        for path in json.loads(str(row.all_valid_oriented_paths)):
            orientation_rows.append(
                {
                    "semantic_loop_id": str(row.semantic_loop_id),
                    "orientation_id": f"{row.semantic_loop_id}__o_"
                    + "-".join(str(int(value)) for value in path),
                    "dictionary_motif_type": str(row.motif_type),
                    "canonical_total_transitions": len(path) - 1,
                }
            )
    orientations = pd.DataFrame(orientation_rows)
    raw_prefixes = ledger.loc[
        ledger["ledger_kind"].eq("active_prefix")
        & ledger["bar_ordinal"].astype(int).isin(DENSE_CHECKPOINTS)
    ]
    checked = raw_prefixes.merge(
        orientations,
        on=["semantic_loop_id", "orientation_id"],
        how="left",
        validate="many_to_one",
    )
    calculated = checked["canonical_total_transitions"].astype(float) - (
        checked["progress_states"].astype(int) - 1
    )
    remainder_mismatches = int(
        (~calculated.eq(checked["transitions_remaining"].astype(float))).sum()
    )
    motif_mismatches = int(
        (checked["motif_type"].astype(str) != checked["dictionary_motif_type"].astype(str)).sum()
    )
    checked = checked.assign(
        calculated_remaining=calculated,
        one_away=calculated.eq(1),
    )
    unique_checked = checked.drop_duplicates(
        ["symbol", "session", "bar_ordinal", "semantic_loop_id", "orientation_id"]
    )
    proximity = (
        unique_checked.groupby(["symbol", "session", "bar_ordinal"], sort=True)
        .agg(
            minimum=("calculated_remaining", "min"),
            one_away_count=("one_away", "sum"),
        )
        .reset_index()
        .rename(columns={"bar_ordinal": "checkpoint"})
    )
    comparison = panel[["row_id", "symbol", "session", "checkpoint"]].merge(
        proximity,
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    comparison["one_away_count"] = comparison["one_away_count"].fillna(0).astype(int)
    count_mismatches = int(
        (
            comparison["one_away_count"].to_numpy(int)
            != panel["number_of_one_transition_away_prefixes"].to_numpy(int)
        ).sum()
    )
    minimum_match = np.isclose(
        comparison["minimum"].to_numpy(float),
        panel["minimum_remaining_transitions"].to_numpy(float),
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    minimum_mismatches = int((~minimum_match).sum())
    return {
        "raw_active_prefix_rows_checked": len(checked),
        "unique_prefix_candidates_after_frozen_deduplication": len(unique_checked),
        "duplicate_prefix_rows": len(checked) - len(unique_checked),
        "dictionary_orientation_rows": len(orientations),
        "missing_dictionary_orientations": int(checked["canonical_total_transitions"].isna().sum()),
        "motif_mismatches": motif_mismatches,
        "remaining_transition_mismatches": remainder_mismatches,
        "one_away_count_mismatches": count_mismatches,
        "minimum_remaining_transition_mismatches": minimum_mismatches,
        "passed": all(
            value == 0
            for value in (
                motif_mismatches,
                remainder_mismatches,
                count_mismatches,
                minimum_mismatches,
            )
        ),
    }


def bootstrap_multiplicities(sessions: pd.Series) -> list[np.ndarray]:
    labels = sessions.astype(str).to_numpy()
    unique = np.asarray(sorted(set(labels)), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        draws.append(np.asarray([int(counts.get(value, 0)) for value in labels]))
    return draws


def increments(
    frame: pd.DataFrame, boundaries: Mapping[str, Mapping[str, float]]
) -> dict[str, float]:
    left = independent_metrics(frame, "A0_probability")
    right = independent_metrics(frame, "A1_probability")

    def top_precision(model: str) -> float:
        selected = frame.loc[frame[f"{model}_probability"].ge(boundaries[model]["top_decile"])]
        return weighted_rate(selected, "completion_in_bars_2_or_3")

    return {
        "log_loss_improvement": left["log_loss"] - right["log_loss"],
        "brier_improvement": left["brier_score"] - right["brier_score"],
        "auc_improvement": right["auc"] - left["auc"],
        "average_precision_improvement": right["average_precision"] - left["average_precision"],
        "top_decile_precision_improvement": top_precision("A1") - top_precision("A0"),
    }


def audit_bootstrap(
    assessment: pd.DataFrame,
    archived: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    expected: dict[tuple[int, str], float] = {}
    for draw, multiplicity in enumerate(bootstrap_multiplicities(assessment["session"])):
        sample = assessment.copy()
        sample["row_weight"] *= multiplicity
        sample = sample.loc[sample["row_weight"].gt(0)]
        values = increments(sample, boundaries)
        broad = sample.loc[sample["route_resolution_state"].eq("BROAD_CONFLICT")]
        low = sample.loc[sample["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
        values.update(
            {
                "broad_conflict_completion_rate": weighted_rate(broad, "completion_in_bars_2_or_3"),
                "broad_conflict_minus_pooled_rate": weighted_rate(
                    broad, "completion_in_bars_2_or_3"
                )
                - weighted_rate(sample, "completion_in_bars_2_or_3"),
                "broad_conflict_minus_low_route_support_rate": weighted_rate(
                    broad, "completion_in_bars_2_or_3"
                )
                - weighted_rate(low, "completion_in_bars_2_or_3"),
            }
        )
        expected.update({(draw, key): value for key, value in values.items()})
    draw_rows = archived.loc[archived["record_type"].eq("draw")]
    differences = [
        abs(float(row.value) - expected[(int(row.draw), str(row.statistic))])
        for row in draw_rows.itertuples(index=False)
    ]
    interval_differences: list[float] = []
    for row in archived.loc[archived["record_type"].eq("interval")].itertuples(index=False):
        values = np.asarray(
            [expected[(draw, str(row.statistic))] for draw in range(BOOTSTRAP_DRAWS)]
        )
        alpha = 1.0 - float(row.interval_level)
        interval_differences.extend(
            [
                abs(float(row.lower) - float(np.quantile(values, alpha / 2.0))),
                abs(float(row.upper) - float(np.quantile(values, 1.0 - alpha / 2.0))),
            ]
        )
    maximum = max([0.0, *differences, *interval_differences])
    return {
        "draws": BOOTSTRAP_DRAWS,
        "draw_rows": len(draw_rows),
        "interval_rows": int(archived["record_type"].eq("interval").sum()),
        "maximum_difference": maximum,
        "passed": maximum <= 1e-12,
    }


def audit_nulls(
    panel: pd.DataFrame,
    archived: pd.DataFrame,
    specifications: Mapping[str, Any],
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    population = panel.loc[panel["advance_eligible"].eq(1)]
    assessment = population.loc[population["period"].eq("assessment")]
    draw_rows = archived.loc[archived["record_type"].eq("draw")]
    hash_mismatches = 0
    maximum_probability_difference = 0.0
    maximum_increment_difference = 0.0
    for draw in range(NULL_REFITS):
        permuted = independent_permutation(population, NULL_SEED + draw)
        row = draw_rows.loc[draw_rows["draw"].eq(draw)].iloc[0]
        actual_hash = stable_frame_hash(
            permuted,
            ["period", "session", "checkpoint", "symbol", *ROUTE_FEATURES],
        )
        hash_mismatches += int(actual_hash != str(row["route_bundle_hash"]))
        permuted_assessment = permuted.loc[permuted["period"].eq("assessment")]
        probability = manual_probabilities(
            permuted_assessment, cast(Mapping[str, Any], specifications[str(draw)])
        )
        maximum_probability_difference = max(
            maximum_probability_difference,
            float(
                np.max(
                    np.abs(
                        probability - assessment[f"route_null_{draw}_probability"].to_numpy(float)
                    )
                )
            ),
        )
        null_frame = assessment.copy()
        null_frame["A1_probability"] = probability
        expected = increments(null_frame, boundaries)
        maximum_increment_difference = max(
            maximum_increment_difference,
            max(
                abs(float(row[key]) - expected[key])
                for key in (
                    "log_loss_improvement",
                    "brier_improvement",
                    "auc_improvement",
                    "average_precision_improvement",
                )
            ),
        )
    passed = bool(
        hash_mismatches == 0
        and maximum_probability_difference <= 1e-12
        and maximum_increment_difference <= 1e-12
    )
    return {
        "null_refits": NULL_REFITS,
        "route_bundle_hash_mismatches": hash_mismatches,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_increment_difference": maximum_increment_difference,
        "passed": passed,
    }


def compare_mappings(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, int | float | bool]:
    """Compare a serialized mapping with an independently reconstructed mapping."""

    mismatches = 0
    maximum_difference = 0.0
    if set(actual) != set(expected):
        mismatches += len(set(actual).symmetric_difference(expected))
    for key in set(actual).intersection(expected):
        left = actual[key]
        right = expected[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            nested = compare_mappings(left, right)
            mismatches += int(nested["mismatches"])
            maximum_difference = max(maximum_difference, float(nested["maximum_difference"]))
        elif isinstance(left, bool) or isinstance(right, bool):
            mismatches += int(left is not right)
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            left_value = float(left)
            right_value = float(right)
            if math.isnan(left_value) and math.isnan(right_value):
                continue
            difference = abs(left_value - right_value)
            if not math.isfinite(difference):
                mismatches += 1
            else:
                maximum_difference = max(maximum_difference, difference)
                mismatches += int(difference > 1e-12)
        else:
            mismatches += int(left != right)
    return {
        "mismatches": mismatches,
        "maximum_difference": maximum_difference,
        "passed": mismatches == 0,
    }


def reconstruct_support(
    panel: pd.DataFrame, states: pd.DataFrame
) -> tuple[dict[str, Any], str | None]:
    """Reconstruct all preregistered support and concentration gates from rows."""

    raw = panel.loc[panel["period"].eq("assessment")]
    advance = raw.loc[raw["advance_eligible"].eq(1)]
    state_sessions = pd.to_datetime(states["session"], utc=True, errors="raise")
    assessment_sessions = int(
        states.loc[state_sessions.between(ASSESSMENT_START, READ_END), "session"].nunique()
    )
    theoretical = assessment_sessions * 20 * len(DENSE_CHECKPOINTS)
    raw_retention = len(raw) / theoretical if theoretical else 0.0
    route_retention = (
        float(np.isfinite(advance.loc[:, list(ROUTE_FEATURES)].to_numpy(float)).all(axis=1).mean())
        if not advance.empty
        else 0.0
    )
    weighted_stock_share = (
        advance.groupby("symbol")["row_weight"].sum() / advance["row_weight"].sum()
        if not advance.empty
        else pd.Series(dtype=float)
    )
    support: dict[str, Any] = {
        "eligible_assessment_sessions": assessment_sessions,
        "theoretical_raw_assessment_rows": theoretical,
        "reconstructed_raw_assessment_rows": len(raw),
        "raw_retention": raw_retention,
        "raw_sessions": int(raw["session"].nunique()),
        "raw_stocks": int(raw["symbol"].nunique()),
        "raw_months": int(raw["year_month"].nunique()),
        "advance_rows": len(advance),
        "advance_sessions": int(advance["session"].nunique()),
        "advance_stocks": int(advance["symbol"].nunique()),
        "advance_months": int(advance["year_month"].nunique()),
        "advance_positive_outcomes": int(advance["completion_in_bars_2_or_3"].sum()),
        "route_feature_retention": route_retention,
        "maximum_weighted_stock_share": float(weighted_stock_share.max())
        if not weighted_stock_share.empty
        else float("nan"),
        "base_rate": weighted_rate(advance, "completion_in_bars_2_or_3")
        if not advance.empty
        else float("nan"),
    }
    gates = {
        "raw_retention": raw_retention >= 0.95,
        "raw_sessions": support["raw_sessions"] >= 140,
        "raw_stocks": support["raw_stocks"] == 20,
        "raw_months": support["raw_months"] == 8,
        "advance_rows": support["advance_rows"] >= 30_000,
        "advance_sessions": support["advance_sessions"] >= 140,
        "advance_stocks": support["advance_stocks"] >= 15,
        "advance_months": support["advance_months"] == 8,
        "advance_positives": support["advance_positive_outcomes"] >= 400,
        "route_feature_retention": route_retention >= 0.95,
        "concentration": support["maximum_weighted_stock_share"] <= 0.10,
    }
    blocker: str | None = None
    if not all(gates[key] for key in ("raw_retention", "raw_sessions", "raw_stocks", "raw_months")):
        blocker = "blocked_insufficient_raw_checkpoint_support"
    elif not all(
        gates[key]
        for key in (
            "advance_rows",
            "advance_sessions",
            "advance_stocks",
            "advance_months",
            "route_feature_retention",
            "concentration",
        )
    ):
        blocker = "blocked_insufficient_dense_advance_support"
    elif not gates["advance_positives"]:
        blocker = "blocked_insufficient_dense_advance_positive_support"
    support["gates"] = gates
    support["passed"] = blocker is None
    return support, blocker


def concentration_artifact_audit(
    archived: pd.DataFrame, support: Mapping[str, Any]
) -> dict[str, int | float | bool]:
    expected = {
        ("raw", "retention"): (float(support["raw_retention"]), 0.95),
        ("advance", "rows"): (float(support["advance_rows"]), 30_000.0),
        ("advance", "positive_outcomes"): (
            float(support["advance_positive_outcomes"]),
            400.0,
        ),
        ("advance", "route_feature_retention"): (
            float(support["route_feature_retention"]),
            0.95,
        ),
        ("advance", "maximum_weighted_stock_share"): (
            float(support["maximum_weighted_stock_share"]),
            0.10,
        ),
    }
    maximum_difference = 0.0
    mismatches = 0
    actual_keys = {
        (str(row.population), str(row.metric)) for row in archived.itertuples(index=False)
    }
    mismatches += len(actual_keys.symmetric_difference(expected))
    for key, (expected_value, expected_threshold) in expected.items():
        row = archived.loc[archived["population"].eq(key[0]) & archived["metric"].eq(key[1])]
        if len(row) != 1:
            mismatches += 1
            continue
        value_difference = abs(float(row.iloc[0]["value"]) - expected_value)
        threshold_difference = abs(float(row.iloc[0]["threshold"]) - expected_threshold)
        maximum_difference = max(maximum_difference, value_difference, threshold_difference)
        mismatches += int(value_difference > 1e-12) + int(threshold_difference > 1e-12)
        expected_pass = (
            expected_value <= expected_threshold
            if key[1] == "maximum_weighted_stock_share"
            else expected_value >= expected_threshold
        )
        mismatches += int(bool(row.iloc[0]["passed"]) != expected_pass)
    return {
        "mismatches": mismatches,
        "maximum_difference": maximum_difference,
        "passed": mismatches == 0,
    }


def independent_boundary_audit(
    panel: pd.DataFrame,
    states: pd.DataFrame,
    ledger: pd.DataFrame,
    protected: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect every market-time surface rather than trusting boundary counters."""

    timestamps = {
        "causal_state_trace": pd.to_datetime(
            states["bar_complete_timestamp"], utc=True, errors="raise"
        ),
        "causal_state_session": pd.to_datetime(states["session"], utc=True, errors="raise"),
        "structural_ledger": pd.to_datetime(
            ledger["available_timestamp_utc"], utc=True, errors="raise"
        ),
        "structural_ledger_session": pd.to_datetime(ledger["session"], utc=True, errors="raise"),
        "checkpoint_timestamp": pd.to_datetime(
            panel["checkpoint_timestamp_utc"], utc=True, errors="raise"
        ),
        "feature_available_timestamp": pd.to_datetime(
            panel["feature_available_timestamp_utc"], utc=True, errors="raise"
        ),
        "decision_panel_session": pd.to_datetime(panel["session"], utc=True, errors="raise"),
    }
    counts = {key: int(values.ge(PROTECTED_START).sum()) for key, values in timestamps.items()}
    reported = cast(Mapping[str, Any], protected["protected_rows_by_source"])
    reported_counts_match = bool(
        int(reported["causal_state_trace"]) == counts["causal_state_trace"]
        and int(reported["structural_ledger"]) == counts["structural_ledger"]
        and int(reported["dense_checkpoint_panel"]) == counts["checkpoint_timestamp"]
    )
    reported_timestamp_match = bool(
        pd.Timestamp(str(protected["minimum_timestamp_read"]))
        == timestamps["causal_state_trace"].min()
        and pd.Timestamp(str(protected["maximum_timestamp_read"]))
        == timestamps["causal_state_trace"].max()
        and pd.Timestamp(str(protected["maximum_structural_timestamp_read"]))
        == timestamps["structural_ledger"].max()
        and pd.Timestamp(str(protected["maximum_checkpoint_timestamp"]))
        == timestamps["checkpoint_timestamp"].max()
        and pd.Timestamp(str(source_manifest["minimum_timestamp_read"]))
        == timestamps["causal_state_trace"].min()
        and pd.Timestamp(str(source_manifest["maximum_timestamp_read"]))
        == timestamps["causal_state_trace"].max()
    )
    passed = bool(
        all(values.min() >= DEVELOPMENT_START for values in timestamps.values())
        and all(values.max() <= READ_END for values in timestamps.values())
        and all(value == 0 for value in counts.values())
        and int(protected["protected_rows_materialised"]) == 0
        and int(source_manifest["protected_rows_materialised"]) == 0
        and bool(protected["passed"])
        and reported_counts_match
        and reported_timestamp_match
    )
    return {
        "protected_rows_by_independently_read_surface": counts,
        "reported_counts_match": reported_counts_match,
        "reported_timestamps_match": reported_timestamp_match,
        "minimum_timestamp_by_surface": {
            key: str(values.min()) for key, values in timestamps.items()
        },
        "maximum_timestamp_by_surface": {
            key: str(values.max()) for key, values in timestamps.items()
        },
        "passed": passed,
    }


def independently_reconstruct_chronology_labels(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Derive split, month, and frozen checkpoint groups from immutable row chronology."""

    sessions = pd.to_datetime(panel["session"], utc=True, errors="raise")
    checkpoints = panel["checkpoint"].astype(int)
    expected_period = pd.Series(
        np.where(sessions.lt(ASSESSMENT_START), "development", "assessment"),
        index=panel.index,
        dtype="string",
    )
    expected_month = sessions.dt.strftime("%Y-%m").astype("string")
    expected_group = pd.Series("INVALID", index=panel.index, dtype="string")
    expected_group.loc[checkpoints.between(6, 14)] = "early_6_14"
    expected_group.loc[checkpoints.between(16, 24)] = "middle_16_24"
    expected_group.loc[checkpoints.between(26, 34)] = "later_26_34"
    period_mismatches = int(
        (panel["period"].astype(str).to_numpy() != expected_period.astype(str).to_numpy()).sum()
    )
    month_mismatches = int(
        (panel["year_month"].astype(str).to_numpy() != expected_month.astype(str).to_numpy()).sum()
    )
    checkpoint_group_mismatches = int(
        (
            panel["checkpoint_group"].astype(str).to_numpy()
            != expected_group.astype(str).to_numpy()
        ).sum()
    )
    invalid_session_rows = int((~sessions.between(DEVELOPMENT_START, READ_END)).sum())
    invalid_checkpoint_rows = int((~checkpoints.isin(DENSE_CHECKPOINTS)).sum())
    corrected = panel.copy()
    corrected["period"] = expected_period
    corrected["year_month"] = expected_month
    corrected["checkpoint_group"] = expected_group
    result = {
        "period_mismatches": period_mismatches,
        "year_month_mismatches": month_mismatches,
        "checkpoint_group_mismatches": checkpoint_group_mismatches,
        "invalid_session_rows": invalid_session_rows,
        "invalid_checkpoint_rows": invalid_checkpoint_rows,
        "passed": all(
            value == 0
            for value in (
                period_mismatches,
                month_mismatches,
                checkpoint_group_mismatches,
                invalid_session_rows,
                invalid_checkpoint_rows,
            )
        ),
    }
    return corrected, result


def grouped_proper_score_audit(
    frame: pd.DataFrame, archived: pd.DataFrame, key: str
) -> dict[str, int | float | bool]:
    """Recalculate monthly or checkpoint-group proper scores from predictions."""

    maximum_difference = 0.0
    mismatches = 0
    expected_keys = {
        (str(group_key), model) for group_key in frame[key].unique() for model in ("A0", "A1")
    }
    archived_keys = {
        (str(row[0]), str(row[1]))
        for row in archived.loc[:, [key, "model"]].itertuples(index=False, name=None)
    }
    mismatches += len(expected_keys.symmetric_difference(archived_keys))
    for group_key, group in frame.groupby(key, sort=True):
        for model in ("A0", "A1"):
            row = archived.loc[
                archived[key].astype(str).eq(str(group_key)) & archived["model"].eq(model)
            ]
            if len(row) != 1:
                mismatches += 1
                continue
            expected = independent_metrics(group, f"{model}_probability")
            for metric in ("log_loss", "brier_score"):
                difference = abs(float(row.iloc[0][metric]) - expected[metric])
                maximum_difference = max(maximum_difference, difference)
                mismatches += int(difference > 1e-12)
    return {
        "mismatches": mismatches,
        "maximum_difference": maximum_difference,
        "passed": mismatches == 0,
    }


def route_state_artifact_audit(
    panel: pd.DataFrame,
    archived: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, int | float | bool]:
    """Recalculate the frozen state table used by the mechanism gate."""

    maximum_difference = 0.0
    mismatches = 0
    for period in ("development", "assessment"):
        population = panel.loc[panel["period"].eq(period) & panel["advance_eligible"].eq(1)]
        for state in ROUTE_STATES:
            group = population.loc[population["route_resolution_state"].eq(state)]
            row = archived.loc[
                archived["period"].eq(period) & archived["route_resolution_state"].eq(state)
            ]
            if len(row) != 1:
                mismatches += 1
                continue
            observed = row.iloc[0]
            expected_counts = {
                "advance_eligible_rows": len(group),
                "positive_outcomes": int(group["completion_in_bars_2_or_3"].sum()),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "months": int(group["year_month"].nunique()),
            }
            mismatches += sum(
                int(int(observed[key]) != value) for key, value in expected_counts.items()
            )
            if group.empty:
                expected_values = {
                    "completion_rate": float("nan"),
                    "A1_minus_A0_log_loss_improvement": float("nan"),
                    "A1_minus_A0_brier_improvement": float("nan"),
                }
            else:
                model_increment = increments(group, boundaries)
                expected_values = {
                    "completion_rate": weighted_rate(group, "completion_in_bars_2_or_3"),
                    "A1_minus_A0_log_loss_improvement": model_increment["log_loss_improvement"],
                    "A1_minus_A0_brier_improvement": model_increment["brier_improvement"],
                }
            for key, expected in expected_values.items():
                actual = float(observed[key])
                if math.isnan(actual) and math.isnan(expected):
                    continue
                difference = abs(actual - expected)
                if not math.isfinite(difference):
                    mismatches += 1
                else:
                    maximum_difference = max(maximum_difference, difference)
                    mismatches += int(difference > 1e-12)
    return {
        "mismatches": mismatches,
        "maximum_difference": maximum_difference,
        "passed": mismatches == 0,
    }


def bootstrap_lower_from_draws(archived: pd.DataFrame, comparison: str, statistic: str) -> float:
    draws = archived.loc[
        archived["record_type"].eq("draw")
        & archived["comparison"].eq(comparison)
        & archived["statistic"].eq(statistic),
        "value",
    ].to_numpy(float)
    if len(draws) != BOOTSTRAP_DRAWS:
        return float("nan")
    return float(np.quantile(draws, 0.10))


def model_stability_from_panel(frame: pd.DataFrame, key: str) -> tuple[int, int]:
    positive = 0
    adverse = 0
    for _, group in frame.groupby(key, sort=True):
        values = increments(group, {"A0": {"top_decile": 0.0}, "A1": {"top_decile": 0.0}})
        positive += int(values["log_loss_improvement"] > 0.0)
        adverse += int(
            values["log_loss_improvement"] < -0.005 or values["brier_improvement"] < -0.002
        )
    return positive, adverse


def state_rate_stability_from_panel(frame: pd.DataFrame, key: str) -> tuple[int, int]:
    positive = 0
    adverse = 0
    for _, group in frame.groupby(key, sort=True):
        broad = group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT")]
        difference = weighted_rate(broad, "completion_in_bars_2_or_3") - weighted_rate(
            group, "completion_in_bars_2_or_3"
        )
        positive += int(difference > 0.0)
        adverse += int(difference < -0.005)
    return positive, adverse


def run_audit(output: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        return {**SAFETY_FLAGS, "passed": False, "missing_artifacts": missing}
    contract = read_json(output / "contract.json")
    decision = read_json(output / "decision.json")
    source_manifest = read_json(output / "source_manifest.json")
    protected = read_json(output / "protected_boundary_audit.json")
    predecessor = read_json(output / "predecessor_reconstruction.json")
    checkpoint_manifest = read_json(output / "dense_checkpoint_manifest.json")
    prefix_manifest = read_json(output / "prefix_proximity_manifest.json")
    configuration = read_json(output / "model_configurations.json")
    coefficients = read_json(output / "model_coefficients.json")
    determinism = read_json(output / "determinism_check.json")
    serialized_panel = pd.read_parquet(output / "dense_advance_panel.parquet")
    panel, chronology_labels = independently_reconstruct_chronology_labels(serialized_panel)
    ledger = pd.read_parquet(V0_PRIMARY / "route_competition_ledger.parquet")
    states = pd.read_parquet(
        V0_PRIMARY / "causal_state_trace.parquet",
        columns=["session", "bar_complete_timestamp"],
    )
    assessment = panel.loc[panel["period"].eq("assessment") & panel["advance_eligible"].eq(1)]
    development = panel.loc[panel["period"].eq("development") & panel["advance_eligible"].eq(1)]
    safety = all(
        artifact.get(key) == value
        for artifact in (contract, decision, protected)
        for key, value in SAFETY_FLAGS.items()
    )
    dates = pd.to_datetime(panel["session"])
    boundary = independent_boundary_audit(panel, states, ledger, protected, source_manifest)
    boundary_passed = bool(boundary["passed"])
    reconstructed_support, support_blocker = reconstruct_support(panel, states)
    support_artifact = compare_mappings(
        cast(Mapping[str, Any], decision.get("support", {})), reconstructed_support
    )
    concentration_artifact = concentration_artifact_audit(
        pd.read_csv(output / "concentration_metrics.csv"), reconstructed_support
    )
    cohort = sorted(panel["symbol"].unique().tolist())
    source_cohort = sorted(source_manifest["frozen_audited_cohort"])
    checkpoints = tuple(sorted(panel["checkpoint"].unique().astype(int).tolist()))
    raw_assessment = panel.loc[panel["period"].eq("assessment")]
    theoretical = int(reconstructed_support["theoretical_raw_assessment_rows"])
    population_passed = bool(
        checkpoints == DENSE_CHECKPOINTS
        and cohort == source_cohort
        and theoretical == int(checkpoint_manifest["nominal_theoretical_assessment_rows"]) == 48_000
        and len(raw_assessment) == int(checkpoint_manifest["assessment_rows"])
    )
    row_identity_hash = stable_frame_hash(panel, ["row_id"])
    feature_surface_hash = stable_frame_hash(panel, ["row_id", *DENSE_H1_FEATURES])
    manifest_surface_passed = bool(
        not panel["row_id"].duplicated().any()
        and len(panel) == int(checkpoint_manifest["raw_rows_all_periods"])
        and row_identity_hash == checkpoint_manifest["row_identity_sha256"]
        and feature_surface_hash == checkpoint_manifest["causal_feature_surface_sha256"]
    )
    v0_panel = pd.read_parquet(V0_PRIMARY / "decision_panel.parquet")
    shared_checkpoints = (6, 10, 14, 18, 22, 26, 30, 34)
    v0_shared = v0_panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    dense_shared = (
        panel.loc[panel["checkpoint"].isin(shared_checkpoints)]
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
    )
    shared_columns = (
        *BASELINE_NON_CLOCK_FEATURES,
        *(f"checkpoint_{checkpoint}" for checkpoint in shared_checkpoints),
        *ROUTE_FEATURES,
    )
    shared_row_mismatches = abs(len(v0_shared) - len(dense_shared)) + sum(
        left != right
        for left, right in zip(
            v0_shared["row_id"].astype(str),
            dense_shared["row_id"].astype(str),
            strict=False,
        )
    )
    if shared_row_mismatches:
        shared_feature_difference = float("inf")
        shared_timestamp_mismatches = shared_row_mismatches
        shared_target_mismatches = shared_row_mismatches
        shared_state_mismatches = shared_row_mismatches
    else:
        shared_feature_difference = float(
            np.max(
                np.abs(
                    v0_shared.loc[:, list(shared_columns)].to_numpy(float)
                    - dense_shared.loc[:, list(shared_columns)].to_numpy(float)
                )
            )
        )
        shared_timestamp_mismatches = int(
            (
                pd.to_datetime(v0_shared["checkpoint_timestamp_utc"], utc=True)
                != pd.to_datetime(dense_shared["checkpoint_timestamp_utc"], utc=True)
            ).sum()
        )
        shared_target_mismatches = int(
            (
                v0_shared["registered_completion_next_3_bars"].to_numpy(int)
                != dense_shared["registered_completion_next_3_bars"].to_numpy(int)
            ).sum()
        )
        shared_state_mismatches = int(
            (
                v0_shared["route_resolution_state"].astype(str).to_numpy()
                != dense_shared["route_resolution_state"].astype(str).to_numpy()
            ).sum()
        )
    shared_surface_passed = bool(
        shared_row_mismatches == 0
        and shared_timestamp_mismatches == 0
        and shared_target_mismatches == 0
        and shared_state_mismatches == 0
        and shared_feature_difference <= 1e-12
    )
    leads = completion_leads(panel, ledger)
    lead_mismatches = int((leads != panel["first_completion_lead"].to_numpy(int)).sum())
    eligibility = (
        (leads != 1) & (panel["any_prefix_one_transition_from_completion"].to_numpy(int) == 0)
    ).astype(int)
    eligibility_mismatches = int((eligibility != panel["advance_eligible"].to_numpy(int)).sum())
    chronology_passed = bool(
        lead_mismatches == 0
        and eligibility_mismatches == 0
        and not assessment["first_completion_lead"].eq(1).any()
        and not assessment["any_prefix_one_transition_from_completion"].eq(1).any()
    )
    prefix = prefix_audit(panel, ledger)
    prefix_manifest_counts_passed = bool(
        int(prefix["raw_active_prefix_rows_checked"])
        == int(prefix_manifest["raw_active_prefix_rows_checked"])
        and int(prefix["unique_prefix_candidates_after_frozen_deduplication"])
        == int(prefix_manifest["unique_prefix_candidates_after_frozen_deduplication"])
    )
    expected_weights = 1.0 / (
        assessment.groupby(["period", "session"])["symbol"].transform("nunique")
        * assessment.groupby(["period", "session", "symbol"])["symbol"].transform("size")
    )
    weight_difference = float(
        np.max(np.abs(expected_weights.to_numpy(float) - assessment["row_weight"].to_numpy(float)))
    )
    weights_passed = bool(weight_difference <= 1e-12)
    v0_route = read_json(V0_PRIMARY / "route_competition_feature_manifest.json")
    thresholds = cast(
        Mapping[str, Sequence[float]],
        v0_route["development_frozen_bins"]["route_quartiles"],
    )
    state_labels = independent_route_states(panel, thresholds)
    state_mismatches = int(
        (
            state_labels.astype(str).to_numpy()
            != panel["route_resolution_state"].astype(str).to_numpy()
        ).sum()
    )
    blocker = cast(str | None, decision.get("blocker"))
    if support_blocker is not None:
        checks = {
            "required_artifacts": not missing,
            "safety_flags": safety,
            "dates_and_protected_boundary": boundary_passed,
            "chronology_and_group_labels": bool(chronology_labels["passed"]),
            "frozen_cohort_and_dense_checkpoints": population_passed,
            "independently_reconstructed_support": bool(support_artifact["passed"]),
            "concentration_artifact": bool(concentration_artifact["passed"]),
            "shared_predecessor_reconstruction": bool(
                predecessor["passed"] and shared_surface_passed
            ),
            "pinned_dense_identity_and_feature_surface": manifest_surface_passed,
            "earliest_completion_lead": lead_mismatches == 0,
            "prefix_remaining_transitions": bool(
                prefix["passed"] and prefix_manifest_counts_passed
            ),
            "advance_eligibility": chronology_passed,
            "candidate_normalized_weights": weights_passed,
            "frozen_route_resolution_states": state_mismatches == 0,
            "blocked_decision_logic": blocker
            in {support_blocker, "blocked_reproducibility_or_audit_failure"},
            "fast_determinism": bool(determinism["passed"]),
        }
        passed = all(checks.values())
        if passed:
            write_json(
                output / "decision.json",
                {
                    **decision,
                    "primary_decision": support_blocker,
                    "blocker": support_blocker,
                    "support": reconstructed_support,
                },
            )
        return {
            **SAFETY_FLAGS,
            "checks": checks,
            "decision_reconstructed": support_blocker,
            "protected_rows_materialised": int(protected["protected_rows_materialised"]),
            "independent_protected_boundary_audit": boundary,
            "chronology_label_audit": chronology_labels,
            "support_artifact_audit": support_artifact,
            "concentration_artifact_audit": concentration_artifact,
            "theoretical_raw_assessment_rows": theoretical,
            "reconstructed_raw_assessment_rows": len(raw_assessment),
            "advance_rows": len(assessment),
            "advance_positive_outcomes": int(assessment["completion_in_bars_2_or_3"].sum()),
            "prefix_audit": prefix,
            "shared_surface_audit": {
                "row_identity_mismatches": shared_row_mismatches,
                "maximum_feature_difference": shared_feature_difference,
                "passed": shared_surface_passed,
            },
            "passed": passed,
        }
    feature_surface_passed = bool(
        tuple(configuration["A0_features"]) == DENSE_H0_FEATURES
        and tuple(configuration["A1_features"]) == DENSE_H1_FEATURES
        and tuple(configuration["route_bundle"]) == ROUTE_FEATURES
        and tuple(BASELINE_NON_CLOCK_FEATURES)
        == tuple(DENSE_H0_FEATURES[: len(BASELINE_NON_CLOCK_FEATURES)])
        and np.isfinite(assessment.loc[:, list(DENSE_H1_FEATURES)].to_numpy(float)).all()
    )
    scaler_difference = 0.0
    manual_probability_difference = 0.0
    audited_probability_rows: dict[str, int] = {}
    primary_specs = cast(Mapping[str, Any], coefficients["primary_models"])
    for model, features in (("A0", DENSE_H0_FEATURES), ("A1", DENSE_H1_FEATURES)):
        specification = cast(Mapping[str, Any], primary_specs[model])
        matrix = development.loc[:, list(features)].to_numpy(float)
        expected_scale = matrix.std(axis=0, ddof=0)
        expected_scale[expected_scale == 0.0] = 1.0
        scaler_difference = max(
            scaler_difference,
            float(
                np.max(
                    np.abs(
                        matrix.mean(axis=0) - np.asarray(specification["scaler_mean"], dtype=float)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(expected_scale - np.asarray(specification["scaler_scale"], dtype=float))
                )
            ),
        )
        sample = assessment.iloc[:100]
        manual = manual_probabilities(sample, specification)
        manual_probability_difference = max(
            manual_probability_difference,
            float(np.max(np.abs(manual - sample[f"{model}_probability"].to_numpy(float)))),
        )
        audited_probability_rows[model] = len(sample)
    pooled = pd.read_csv(output / "pooled_metrics.csv").set_index("model")
    metric_difference = 0.0
    for model in ("A0", "A1"):
        expected = independent_metrics(assessment, f"{model}_probability")
        metric_difference = max(
            metric_difference,
            max(abs(expected[key] - float(pooled.loc[model, key])) for key in expected),
        )
    boundaries = cast(
        Mapping[str, Mapping[str, float]], configuration["probability_quantile_boundaries"]
    )
    bootstrap_frame = pd.read_csv(output / "bootstrap_metrics.csv")
    null_frame = pd.read_csv(output / "route_null_metrics.csv")
    monthly_frame = pd.read_csv(output / "monthly_metrics.csv")
    checkpoint_group_frame = pd.read_csv(output / "checkpoint_group_metrics.csv")
    state_frame = pd.read_csv(output / "route_resolution_state_metrics.csv")
    bootstrap = audit_bootstrap(assessment, bootstrap_frame, boundaries)
    nulls = audit_nulls(
        panel,
        null_frame,
        cast(Mapping[str, Any], coefficients["route_null_models"]),
        boundaries,
    )
    monthly_artifact = grouped_proper_score_audit(assessment, monthly_frame, "year_month")
    checkpoint_group_artifact = grouped_proper_score_audit(
        assessment, checkpoint_group_frame, "checkpoint_group"
    )
    state_artifact = route_state_artifact_audit(panel, state_frame, boundaries)

    real_increments = increments(assessment, boundaries)
    positive_months, _ = model_stability_from_panel(assessment, "year_month")
    _, adverse_checkpoint_groups = model_stability_from_panel(assessment, "checkpoint_group")
    null_draws = null_frame.loc[null_frame["record_type"].eq("draw")]
    real_exceeds_all_nulls = any(
        bool((real_increments[statistic] > null_draws[statistic].to_numpy(float)).all())
        for statistic in ("log_loss_improvement", "brier_improvement")
    )
    advance_gates: dict[str, Any] = {
        **real_increments,
        "bootstrap_80_log_loss_lower": bootstrap_lower_from_draws(
            bootstrap_frame, "advance", "log_loss_improvement"
        ),
        "bootstrap_80_brier_lower": bootstrap_lower_from_draws(
            bootstrap_frame, "advance", "brier_improvement"
        ),
        "bootstrap_80_average_precision_lower": bootstrap_lower_from_draws(
            bootstrap_frame, "advance", "average_precision_improvement"
        ),
        "positive_months": positive_months,
        "materially_adverse_checkpoint_groups": adverse_checkpoint_groups,
        "real_exceeds_all_nulls": real_exceeds_all_nulls,
        "support_and_concentration_passed": bool(reconstructed_support["passed"]),
    }
    assessment_broad = assessment.loc[assessment["route_resolution_state"].eq("BROAD_CONFLICT")]
    assessment_low = assessment.loc[assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
    development_broad = development.loc[development["route_resolution_state"].eq("BROAD_CONFLICT")]
    broad_positive_months, _ = state_rate_stability_from_panel(assessment, "year_month")
    _, broad_adverse_groups = state_rate_stability_from_panel(assessment, "checkpoint_group")
    broad_gates: dict[str, Any] = {
        "assessment_minus_pooled": weighted_rate(assessment_broad, "completion_in_bars_2_or_3")
        - weighted_rate(assessment, "completion_in_bars_2_or_3"),
        "assessment_minus_low_route_support": weighted_rate(
            assessment_broad, "completion_in_bars_2_or_3"
        )
        - weighted_rate(assessment_low, "completion_in_bars_2_or_3"),
        "bootstrap_80_pooled_difference_lower": bootstrap_lower_from_draws(
            bootstrap_frame, "broad_conflict", "broad_conflict_minus_pooled_rate"
        ),
        "bootstrap_80_low_route_difference_lower": bootstrap_lower_from_draws(
            bootstrap_frame,
            "broad_conflict",
            "broad_conflict_minus_low_route_support_rate",
        ),
        "development_minus_pooled": weighted_rate(development_broad, "completion_in_bars_2_or_3")
        - weighted_rate(development, "completion_in_bars_2_or_3"),
        "positive_assessment_months": broad_positive_months,
        "materially_adverse_checkpoint_groups": broad_adverse_groups,
        "assessment_rows": len(assessment_broad),
        "assessment_positives": int(assessment_broad["completion_in_bars_2_or_3"].sum()),
        "assessment_sessions": int(assessment_broad["session"].nunique()),
        "assessment_stocks": int(assessment_broad["symbol"].nunique()),
    }
    advance_gate_artifact = compare_mappings(
        cast(Mapping[str, Any], decision.get("advance_model_gates", {})), advance_gates
    )
    broad_gate_artifact = compare_mappings(
        cast(Mapping[str, Any], decision.get("broad_conflict_gates", {})), broad_gates
    )
    increment_artifact = compare_mappings(
        cast(Mapping[str, Any], decision.get("A1_minus_A0_increments", {})),
        real_increments,
    )
    advance_passed = bool(
        float(advance_gates["log_loss_improvement"]) > 0
        and float(advance_gates["brier_improvement"]) > 0
        and float(advance_gates["auc_improvement"]) >= 0
        and float(advance_gates["average_precision_improvement"]) > 0
        and float(advance_gates["bootstrap_80_log_loss_lower"]) >= 0
        and float(advance_gates["bootstrap_80_brier_lower"]) >= 0
        and float(advance_gates["bootstrap_80_average_precision_lower"]) >= 0
        and int(advance_gates["positive_months"]) >= 5
        and int(advance_gates["materially_adverse_checkpoint_groups"]) == 0
        and bool(advance_gates["real_exceeds_all_nulls"])
        and bool(advance_gates["support_and_concentration_passed"])
    )
    broad_passed = bool(
        float(broad_gates["assessment_minus_pooled"]) > 0
        and float(broad_gates["assessment_minus_low_route_support"]) > 0
        and float(broad_gates["bootstrap_80_pooled_difference_lower"]) >= 0
        and float(broad_gates["bootstrap_80_low_route_difference_lower"]) >= 0
        and float(broad_gates["development_minus_pooled"]) > 0
        and int(broad_gates["positive_assessment_months"]) >= 5
        and int(broad_gates["materially_adverse_checkpoint_groups"]) == 0
        and int(broad_gates["assessment_rows"]) >= 3000
        and int(broad_gates["assessment_positives"]) >= 100
        and int(broad_gates["assessment_sessions"]) >= 100
        and int(broad_gates["assessment_stocks"]) >= 15
    )
    broad_descriptive = bool(
        float(broad_gates["assessment_minus_pooled"]) > 0
        and float(broad_gates["assessment_minus_low_route_support"]) > 0
        and float(broad_gates["development_minus_pooled"]) > 0
    )
    a0 = pooled.loc["A0"]
    base_rate = weighted_rate(assessment, "completion_in_bars_2_or_3")
    constant_log_loss = float(
        -np.average(
            assessment["completion_in_bars_2_or_3"] * math.log(max(base_rate, 1e-12))
            + (1 - assessment["completion_in_bars_2_or_3"]) * math.log(max(1.0 - base_rate, 1e-12)),
            weights=assessment["row_weight"],
        )
    )
    baseline_meaningful = bool(float(a0["auc"]) > 0.5 and float(a0["log_loss"]) < constant_log_loss)
    if advance_passed and broad_passed:
        reconstructed_decision = "broad_route_conflict_adds_clean_advance_warning"
    elif advance_passed:
        reconstructed_decision = (
            "route_competition_adds_clean_advance_warning_without_state_specificity"
        )
    elif broad_descriptive:
        reconstructed_decision = "descriptive_broad_conflict_structure_only"
    elif baseline_meaningful:
        reconstructed_decision = "compressed_transition_baseline_only"
    else:
        reconstructed_decision = "no_clean_advance_route_increment"
    decision_passed = bool(
        decision["primary_decision"]
        in {reconstructed_decision, "blocked_reproducibility_or_audit_failure"}
    )
    expected_lead_statuses: dict[str, str] = {}
    for lead, key in ((2, "lead_two_status"), (3, "lead_three_status")):
        group = assessment.loc[assessment["first_completion_lead"].eq(lead)]
        enough = bool(
            len(group) >= 100
            and group["session"].nunique() >= 100
            and group["symbol"].nunique() >= 15
        )
        if not enough:
            expected_lead_statuses[key] = "insufficient_support"
        elif advance_passed and float(
            np.average(group["A1_probability"], weights=group["row_weight"])
        ) > float(np.average(group["A0_probability"], weights=group["row_weight"])):
            expected_lead_statuses[key] = "supported"
        else:
            expected_lead_statuses[key] = "descriptive_only"
    statuses_passed = bool(
        decision["advance_model_status"] == ("supported" if advance_passed else "not_supported")
        and decision["broad_conflict_status"]
        == (
            "supported"
            if broad_passed
            else ("descriptive_only" if broad_descriptive else "not_supported")
        )
        and all(decision[key] == value for key, value in expected_lead_statuses.items())
    )
    checks = {
        "required_artifacts": not missing,
        "safety_flags": safety,
        "dates_and_protected_boundary": boundary_passed,
        "chronology_and_group_labels": bool(chronology_labels["passed"]),
        "frozen_cohort_and_dense_checkpoints": population_passed,
        "independently_reconstructed_support": bool(support_artifact["passed"]),
        "concentration_artifact": bool(concentration_artifact["passed"]),
        "shared_predecessor_reconstruction": bool(predecessor["passed"] and shared_surface_passed),
        "pinned_dense_identity_and_feature_surface": manifest_surface_passed,
        "earliest_completion_lead": lead_mismatches == 0,
        "prefix_remaining_transitions": bool(prefix["passed"] and prefix_manifest_counts_passed),
        "advance_eligibility": chronology_passed,
        "candidate_normalized_weights": weights_passed,
        "frozen_feature_surfaces": feature_surface_passed,
        "frozen_route_resolution_states": state_mismatches == 0,
        "development_only_preprocessing": scaler_difference <= 1e-12,
        "manual_probability_reconstruction": manual_probability_difference <= 1e-12,
        "proper_scores_and_discrimination": metric_difference <= 1e-12,
        "monthly_metric_artifact": bool(monthly_artifact["passed"]),
        "checkpoint_group_metric_artifact": bool(checkpoint_group_artifact["passed"]),
        "route_state_metric_artifact": bool(state_artifact["passed"]),
        "session_bootstrap": bool(bootstrap["passed"]),
        "route_bundle_null": bool(nulls["passed"]),
        "independently_reconstructed_decision_gates": bool(
            advance_gate_artifact["passed"]
            and broad_gate_artifact["passed"]
            and increment_artifact["passed"]
        ),
        "decision_logic": decision_passed,
        "decision_statuses": statuses_passed,
        "fast_determinism": bool(determinism["passed"]),
    }
    passed = all(checks.values())
    result = {
        **SAFETY_FLAGS,
        "checks": checks,
        "dates": {"minimum": str(dates.min()), "maximum": str(dates.max())},
        "protected_rows_materialised": int(protected["protected_rows_materialised"]),
        "independent_protected_boundary_audit": boundary,
        "chronology_label_audit": chronology_labels,
        "support_artifact_audit": support_artifact,
        "concentration_artifact_audit": concentration_artifact,
        "theoretical_raw_assessment_rows": theoretical,
        "reconstructed_raw_assessment_rows": len(raw_assessment),
        "advance_rows": len(assessment),
        "advance_positive_outcomes": int(assessment["completion_in_bars_2_or_3"].sum()),
        "lead_label_mismatches": lead_mismatches,
        "advance_eligibility_mismatches": eligibility_mismatches,
        "prefix_audit": prefix,
        "shared_surface_audit": {
            "row_identity_mismatches": shared_row_mismatches,
            "checkpoint_timestamp_mismatches": shared_timestamp_mismatches,
            "target_mismatches": shared_target_mismatches,
            "route_resolution_label_mismatches": shared_state_mismatches,
            "maximum_feature_difference": shared_feature_difference,
            "passed": shared_surface_passed,
        },
        "row_identity_sha256": row_identity_hash,
        "causal_feature_surface_sha256": feature_surface_hash,
        "manifest_surface_passed": manifest_surface_passed,
        "maximum_weight_difference": weight_difference,
        "route_resolution_label_mismatches": state_mismatches,
        "maximum_scaler_difference": scaler_difference,
        "manual_probability_rows_by_model": audited_probability_rows,
        "maximum_manual_probability_difference": manual_probability_difference,
        "maximum_primary_metric_difference": metric_difference,
        "monthly_metric_artifact_audit": monthly_artifact,
        "checkpoint_group_metric_artifact_audit": checkpoint_group_artifact,
        "route_state_metric_artifact_audit": state_artifact,
        "bootstrap_audit": bootstrap,
        "route_null_audit": nulls,
        "advance_gate_artifact_audit": advance_gate_artifact,
        "broad_conflict_gate_artifact_audit": broad_gate_artifact,
        "increment_artifact_audit": increment_artifact,
        "advance_model_gates_reconstructed": advance_gates,
        "broad_conflict_gates_reconstructed": broad_gates,
        "decision_reconstructed": reconstructed_decision,
        "status_values_reconstructed": {
            "advance_model_status": "supported" if advance_passed else "not_supported",
            "broad_conflict_status": "supported"
            if broad_passed
            else ("descriptive_only" if broad_descriptive else "not_supported"),
            **expected_lead_statuses,
        },
        "passed": passed,
    }
    if passed:
        write_json(
            output / "decision.json",
            {
                **decision,
                "primary_decision": reconstructed_decision,
                "blocker": None,
                "A1_minus_A0_increments": real_increments,
                "advance_model_gates": advance_gates,
                "broad_conflict_gates": broad_gates,
                "support": reconstructed_support,
            },
        )
    else:
        updated = {
            **decision,
            "primary_decision": "blocked_reproducibility_or_audit_failure",
            "blocker": "blocked_reproducibility_or_audit_failure",
        }
        write_json(output / "decision.json", updated)
    return result


def finalize_audited_report(output: Path, audit: Mapping[str, Any]) -> None:
    """Synchronize the human report with the post-audit fail-closed decision."""

    output.mkdir(parents=True, exist_ok=True)
    decision_path = output / "decision.json"
    try:
        loaded_decision = read_json(decision_path) if decision_path.exists() else {}
    except (json.JSONDecodeError, OSError, TypeError, UnicodeError):
        loaded_decision = {}
    decision = dict(loaded_decision) if isinstance(loaded_decision, Mapping) else {}
    decision = {**decision, **SAFETY_FLAGS}
    if not bool(audit.get("passed", False)) or "primary_decision" not in decision:
        decision.update(
            {
                "primary_decision": "blocked_reproducibility_or_audit_failure",
                "blocker": "blocked_reproducibility_or_audit_failure",
            }
        )
    write_json(decision_path, decision)

    report_path = output / "report.md"
    if report_path.exists():
        source = report_path.read_text(encoding="utf-8").splitlines()
    else:
        source = [
            "# Broad-Conflict Advance-Hazard Dense-Checkpoint Quick Screen V0.2",
            "",
            f"Primary decision: `{decision['primary_decision']}`.",
            "",
            "This blocked screen provides no evidence of economic value, directional edge, "
            "options edge, trading utility, prospective validation, or a deployable strategy.",
        ]
    cleaned: list[str] = []
    primary_decision_written = False
    for line in source:
        if line.startswith("Primary decision:"):
            cleaned.append(f"Primary decision: `{decision['primary_decision']}`.")
            primary_decision_written = True
        elif not line.startswith("Independent lightweight audit passed:"):
            cleaned.append(line)
    if not primary_decision_written:
        insertion_index = 2 if cleaned and cleaned[0].startswith("# ") else 0
        cleaned[insertion_index:insertion_index] = [
            f"Primary decision: `{decision['primary_decision']}`.",
            "",
        ]
    marker = next(
        (
            index
            for index, line in enumerate(cleaned)
            if line.startswith("These findings are not prospective")
            or line.startswith("This blocked screen provides no evidence")
        ),
        len(cleaned),
    )
    while marker > 1 and cleaned[marker - 1] == cleaned[marker - 2] == "":
        del cleaned[marker - 1]
        marker -= 1
    insertion = [] if marker > 0 and cleaned[marker - 1] == "" else [""]
    insertion.append(f"Independent lightweight audit passed: {bool(audit['passed'])}.")
    cleaned[marker:marker] = insertion
    report = "\n".join(cleaned).rstrip() + "\n"
    primary = report_path
    copy = output.parents[1] / "reports" / "report.md"
    copy.parent.mkdir(parents=True, exist_ok=True)
    primary.write_text(report, encoding="utf-8")
    copy.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_emergency_fail_closed_artifacts(output: Path, audit: Mapping[str, Any]) -> None:
    """Write minimal durable blockers without reading any potentially corrupt artifact."""

    output.mkdir(parents=True, exist_ok=True)
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": "blocked_reproducibility_or_audit_failure",
        "blocker": "blocked_reproducibility_or_audit_failure",
    }
    write_json(output / "lightweight_audit.json", audit)
    write_json(output / "decision.json", decision)
    report = (
        "# Broad-Conflict Advance-Hazard Dense-Checkpoint Quick Screen V0.2\n\n"
        "Primary decision: `blocked_reproducibility_or_audit_failure`.\n\n"
        "Independent lightweight audit passed: False.\n"
        "This blocked screen provides no evidence of economic value, directional edge, "
        "options edge, trading utility, prospective validation, or a deployable strategy.\n"
    )
    report_copy = output.parents[1] / "reports" / "report.md"
    report_copy.parent.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")


def audit_entrypoint(output: Path) -> tuple[dict[str, Any], int]:
    """Run the audit and fail closed even when malformed artifacts raise."""

    output.mkdir(parents=True, exist_ok=True)
    try:
        audit = run_audit(output)
        write_json(output / "lightweight_audit.json", audit)
        finalize_audited_report(output, audit)
    except Exception as error:  # noqa: BLE001 - the independent audit must fail closed
        audit = {
            **SAFETY_FLAGS,
            "checks": {"audit_execution": False},
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "passed": False,
        }
        write_emergency_fail_closed_artifacts(output, audit)
    return audit, 0 if audit["passed"] else 1


def main() -> int:
    arguments = parse_args()
    audit, exit_code = audit_entrypoint(arguments.output)
    print(json.dumps(audit, sort_keys=True, indent=2) + "\n", end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
