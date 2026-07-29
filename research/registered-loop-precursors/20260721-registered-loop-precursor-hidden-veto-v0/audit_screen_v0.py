#!/usr/bin/env python3
"""Independently audit the registered-precursor/hidden-veto quick screen V0."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-registered-precursor-veto-audit-mpl")

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.hidden_loop_economics_registered_bridge_v0 import (
    expanding_logistic_crossfit,
    fit_weighted_logistic,
)

BRIDGE_DIR = (
    REPO_ROOT
    / "research"
    / "hidden-loop-economics"
    / "20260721-hidden-loop-economics-registered-bridge-v0"
)
BRIDGE_PRIMARY = BRIDGE_DIR / "artifacts" / "primary"
OPENING_DIR = (
    REPO_ROOT
    / "research"
    / "unregistered-loop-families"
    / "20260721-opening-trajectory-unregistered-families-v0"
)
OPENING_PRIMARY = OPENING_DIR / "artifacts" / "primary"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORT_COPY = EXPERIMENT_DIR / "reports" / "report.md"

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "registered_loop_precursor_census": True,
    "hidden_route_diversion_veto_test": True,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
KEYS = ["symbol", "session", "decision_ordinal"]
LOOKBACKS = (3, 6, 12)
FROZEN_HIDDEN = (
    "unregistered_primitive_like__5-6-5",
    "unregistered_primitive_like__2-3-2",
    "unregistered_primitive_like__2-5-2",
    "unregistered_primitive_like__4-7-4",
)
OTHER_HIDDEN = "OTHER_UNREGISTERED_FAMILY"
BOOLEAN_PRECURSORS = (
    "same_registered_identity",
    "same_registered_broad_family_different_identity",
    "different_registered_broad_family",
    "any_prior_registered_completion",
    "hidden_5_6_5",
    "hidden_2_3_2",
    "hidden_2_5_2",
    "hidden_4_7_4",
    "hidden_other_unregistered_family",
    "any_hidden_unregistered_completion",
    "active_prefix_immediately_before_completion",
    "active_prefix_any",
    "matching_prefix_any",
    "other_prefix_any",
    "any_regime_transition",
    "no_identified_structural_precursor",
)
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "opening_panel_reconstruction.json",
    "registered_completion_event_ledger.parquet",
    "precursor_feature_ledger.parquet",
    "precursor_census.csv",
    "nearest_precursor_census.csv",
    "precursor_monthly_metrics.csv",
    "precursor_checkpoint_metrics.csv",
    "precursor_null_metrics.csv",
    "exact_precursor_transition_counts.csv",
    "exact_precursor_multiplicity.csv",
    "candidate_threshold_manifest.json",
    "candidate_population.parquet",
    "hidden_risk_thresholds.json",
    "candidate_group_metrics.csv",
    "veto_model_configurations.json",
    "veto_model_coefficients.json",
    "veto_assessment_predictions.parquet",
    "veto_metrics.csv",
    "veto_monthly_metrics.csv",
    "veto_checkpoint_metrics.csv",
    "realised_diversion_metrics.csv",
    "bootstrap_metrics.csv",
    "veto_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "determinism_check.json",
    "report.md",
    "b0_development_oof_predictions.parquet",
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load predecessor module {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def serialised_probability(frame: pd.DataFrame, specification: dict[str, Any]) -> np.ndarray:
    features = list(specification["feature_names"])
    matrix = frame.loc[:, features].to_numpy(float)
    mean = np.asarray(specification["scaler_mean"], dtype=float)
    scale = np.asarray(specification["scaler_scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    linear = (matrix - mean) / scale @ coefficient + float(specification["intercept"])
    return 1.0 / (1.0 + np.exp(-linear))


def weighted_rate(frame: pd.DataFrame, column: str) -> float:
    weights = frame["row_weight"].to_numpy(float)
    return float(np.sum(weights * frame[column].to_numpy(float)) / np.sum(weights))


def core_metrics(frame: pd.DataFrame, probability: str) -> dict[str, float]:
    target = frame["registered_completion_within_12_bars"].to_numpy(int)
    prediction = np.clip(frame[probability].to_numpy(float), 1e-15, 1.0 - 1e-15)
    weight = frame["row_weight"].to_numpy(float)
    return {
        "log_loss": float(
            -np.sum(weight * (target * np.log(prediction) + (1 - target) * np.log1p(-prediction)))
            / np.sum(weight)
        ),
        "brier_score": float(np.sum(weight * (target - prediction) ** 2) / np.sum(weight)),
        "auc": float(roc_auc_score(target, prediction, sample_weight=weight)),
        "average_precision": float(
            average_precision_score(target, prediction, sample_weight=weight)
        ),
    }


def increments(frame: pd.DataFrame) -> dict[str, float]:
    v0 = core_metrics(frame, "V0_probability")
    v1 = core_metrics(frame, "V1_probability")
    return {
        "log_loss_improvement": v0["log_loss"] - v1["log_loss"],
        "brier_improvement": v0["brier_score"] - v1["brier_score"],
        "auc_improvement": v1["auc"] - v0["auc"],
    }


def bh_adjust(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


class Audit:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.checks: dict[str, dict[str, Any]] = {}
        self.maximum_difference = 0.0

    def record(self, name: str, passed: bool, **details: Any) -> None:
        self.checks[name] = {"passed": bool(passed), **details}

    def close(self, left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
        difference = float(
            np.max(
                np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)),
                initial=0.0,
            )
        )
        self.maximum_difference = max(self.maximum_difference, difference)
        return difference <= tolerance

    def audit_required_files_and_safety(self) -> tuple[dict[str, Any], dict[str, Any]]:
        missing = [name for name in REQUIRED_ARTIFACTS if not (self.output / name).exists()]
        self.record("required_artifacts", not missing, missing=missing)
        contract = read_json(self.output / "contract.json")
        decision = read_json(self.output / "decision.json")
        safety_files = [
            "contract.json",
            "source_manifest.json",
            "protected_boundary_audit.json",
            "opening_panel_reconstruction.json",
            "candidate_threshold_manifest.json",
            "hidden_risk_thresholds.json",
            "veto_model_configurations.json",
            "veto_model_coefficients.json",
            "decision.json",
            "determinism_check.json",
        ]
        differences: list[str] = []
        for filename in safety_files:
            payload = read_json(self.output / filename)
            for key, expected in SAFETY_FLAGS.items():
                if payload.get(key) != expected:
                    differences.append(f"{filename}:{key}")
        self.record("safety_flags", not differences, differences=differences)
        hard = contract["hard_limits"]
        limits_ok = (
            hard["one_process"] is True
            and hard["n_jobs"] == 1
            and hard["maximum_expanding_development_folds"] == 4
            and hard["session_bootstrap_draws"] == 25
            and hard["precursor_null_draws"] == 25
            and hard["hidden_probability_null_refits"] == 5
            and hard["maximum_plots"] == 2
        )
        self.record("hard_speed_limits", limits_ok)
        return contract, decision

    def reconstruct_opening(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[str, ...],
    ]:
        opening_runner = load_module(OPENING_DIR / "run_screen_v0.py", "audit_opening_source")
        bridge_runner = load_module(BRIDGE_DIR / "run_screen_v0.py", "audit_bridge_source")
        opening, _, reconstruction, t0_features, _ = bridge_runner.reconstruct_frozen_population(
            opening_runner
        )
        completions = pd.read_parquet(BRIDGE_PRIMARY / "registered_completion_ledger.parquet")
        groups = {
            (str(symbol), str(session)): group
            for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
        }
        targets = []
        for row in opening.itertuples(index=False):
            group = groups.get((str(row.symbol), str(row.session)), pd.DataFrame())
            ordinal = (
                group["completion_bar_ordinal"].to_numpy(int)
                if not group.empty
                else np.asarray([], dtype=int)
            )
            targets.append(
                int(
                    np.any(
                        (ordinal > int(row.repo_bar_start_ordinal))
                        & (ordinal <= int(row.repo_bar_start_ordinal) + 12)
                    )
                )
            )
        opening["registered_completion_within_12_bars"] = targets
        opening["actual_hidden_event_within_6_bars"] = opening["unregistered_event"].eq(1.0)
        slate_counts = opening.groupby("slate_id", sort=True).size()
        opening["row_weight"] = opening["slate_id"].map((1.0 / slate_counts).to_dict())
        opening["p_unregistered_within_6_bars"] = opening["U1_probability"]
        coefficient = read_json(BRIDGE_PRIMARY / "bridge_model_coefficients.json")
        opening["B0_probability"] = serialised_probability(opening, coefficient["B0"])
        opening["B1_probability"] = serialised_probability(opening, coefficient["B1"])
        archived_assessment = pd.read_parquet(
            BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet"
        )
        archived_development = pd.read_parquet(BRIDGE_PRIMARY / "bridge_development_panel.parquet")
        reconstructed = opening.loc[opening["year"].eq(2025)].copy()
        fields = [
            *t0_features,
            "registered_completion_within_12_bars",
            "actual_hidden_event_within_6_bars",
            "row_weight",
            "B0_probability",
            "B1_probability",
            "U1_probability",
        ]
        comparison = archived_assessment.loc[:, [*KEYS, *fields]].merge(
            reconstructed.loc[:, [*KEYS, *fields]],
            on=KEYS,
            how="outer",
            suffixes=("_archived", "_reconstructed"),
            indicator=True,
            validate="one_to_one",
        )
        field_differences = [
            np.max(
                np.abs(
                    comparison[f"{field}_archived"].to_numpy(float)
                    - comparison[f"{field}_reconstructed"].to_numpy(float)
                ),
                initial=0.0,
            )
            for field in fields
        ]
        maximum = float(max(field_differences, default=0.0))
        self.maximum_difference = max(self.maximum_difference, maximum)
        panel_ok = (
            comparison["_merge"].eq("both").all()
            and maximum <= 1e-12
            and reconstruction["passed"] is True
        )
        self.record(
            "opening_panel_reconstruction",
            bool(panel_ok),
            rows=len(comparison),
            maximum_shared_or_probability_difference=maximum,
        )

        timestamp_fields = [
            field
            for field in (
                "decision_timestamp_utc",
                "feature_available_timestamp_utc",
                "decision_bar_start_timestamp_utc",
                "bar_start_timestamp",
                "bar_complete_timestamp",
            )
            if field in opening.columns
            and field in archived_assessment.columns
            and field in archived_development.columns
        ]

        def timestamp_mismatches(archived: pd.DataFrame, reconstructed_panel: pd.DataFrame) -> int:
            comparison = archived.loc[:, [*KEYS, *timestamp_fields]].merge(
                reconstructed_panel.loc[:, [*KEYS, *timestamp_fields]],
                on=KEYS,
                how="outer",
                suffixes=("_archived", "_reconstructed"),
                indicator=True,
                validate="one_to_one",
            )
            failures = int((~comparison["_merge"].eq("both")).sum())
            for field in timestamp_fields:
                left = pd.to_datetime(comparison[f"{field}_archived"], utc=True)
                right = pd.to_datetime(comparison[f"{field}_reconstructed"], utc=True)
                failures += int(left.ne(right).sum())
            return failures

        development_reconstructed = opening.loc[opening["year"].eq(2024)].merge(
            archived_development.loc[:, KEYS],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        timestamp_failures = timestamp_mismatches(
            archived_assessment, reconstructed
        ) + timestamp_mismatches(archived_development, development_reconstructed)
        cross_fitted_u1_difference = float(
            np.max(
                np.abs(
                    archived_development["p_unregistered_within_6_bars"].to_numpy(float)
                    - archived_development["oof_p_unregistered_within_6_bars"].to_numpy(float)
                ),
                initial=0.0,
            )
        )
        self.maximum_difference = max(self.maximum_difference, cross_fitted_u1_difference)
        self.record(
            "opening_timestamps_and_development_cross_fitted_U1",
            timestamp_failures == 0 and cross_fitted_u1_difference <= 1e-12,
            timestamp_fields=timestamp_fields,
            timestamp_mismatches=timestamp_failures,
            cross_fitted_U1_alias_maximum_difference=cross_fitted_u1_difference,
        )

        path = pd.read_parquet(OPENING_PRIMARY / "unregistered_path_ledger.parquet")
        path_keys = set(map(tuple, path[KEYS].astype({"decision_ordinal": int}).to_numpy()))
        hidden_target = opening.apply(
            lambda row: (
                (str(row["symbol"]), str(row["session"]), int(row["decision_ordinal"])) in path_keys
            ),
            axis=1,
        )
        path_origin = path.merge(
            opening.loc[:, [*KEYS, "repo_bar_start_ordinal"]],
            on=KEYS,
            validate="many_to_one",
        )
        hidden_window_ok = bool(
            path_origin["completion_bar_ordinal"].gt(path_origin["repo_bar_start_ordinal"]).all()
            and path_origin["completion_bar_ordinal"]
            .le(path_origin["repo_bar_start_ordinal"] + 6)
            .all()
        )
        self.record(
            "hidden_event_target",
            bool(
                hidden_target.to_numpy().tolist()
                == opening["actual_hidden_event_within_6_bars"].to_numpy().tolist()
                and hidden_window_ok
            ),
            positive_rows=int(hidden_target.sum()),
        )
        archived_target = archived_assessment["registered_completion_within_12_bars"].to_numpy(int)
        self.record(
            "registered_completion_target",
            np.array_equal(
                reconstructed["registered_completion_within_12_bars"].to_numpy(int),
                archived_target,
            ),
            assessment_positive_rows=int(archived_target.sum()),
        )
        return (
            opening,
            completions,
            archived_assessment,
            archived_development,
            tuple(str(value) for value in t0_features),
        )

    def audit_dates(self, opening: pd.DataFrame) -> None:
        protected = read_json(self.output / "protected_boundary_audit.json")
        source = read_json(self.output / "source_manifest.json")
        date_ok = (
            pd.to_datetime(opening["session"]).min() >= pd.Timestamp("2024-01-01")
            and pd.to_datetime(opening["session"]).max() <= pd.Timestamp("2025-08-22")
            and pd.Timestamp(source["maximum_timestamp_read"])
            < pd.Timestamp("2025-08-23T00:00:00Z")
            and source["protected_rows_materialised"] == 0
            and protected["protected_rows_materialised"] == 0
            and protected["passed"] is True
        )
        self.record(
            "dates_and_protected_boundary",
            bool(date_ok),
            minimum_session=str(opening["session"].min()),
            maximum_session=str(opening["session"].max()),
            protected_rows_materialised=source["protected_rows_materialised"],
        )

    def audit_event_dedup(self, opening: pd.DataFrame, completions: pd.DataFrame) -> None:
        archived = pd.read_parquet(self.output / "registered_completion_event_ledger.parquet")
        assessment_decisions = opening.loc[opening["year"].eq(2025)].copy()
        rows: list[tuple[str, str, pd.Timestamp, str, int]] = []
        identity = ["symbol", "session", "completion_timestamp_utc", "semantic_loop_id"]
        for key, group in completions.groupby(identity, sort=True):
            symbol, session, timestamp, semantic = key
            if str(session) < "2025-01-01":
                continue
            first = group.sort_values("orientation_id", kind="mergesort").iloc[0]
            completion_ordinal = int(first["completion_bar_ordinal"])
            available = pd.to_datetime(group["completion_available_timestamp_utc"], utc=True).min()
            candidates = assessment_decisions.loc[
                assessment_decisions["symbol"].astype(str).eq(str(symbol))
                & assessment_decisions["session"].astype(str).eq(str(session))
                & assessment_decisions["repo_bar_start_ordinal"].lt(completion_ordinal)
                & assessment_decisions["repo_bar_start_ordinal"].add(12).ge(completion_ordinal)
                & pd.to_datetime(
                    assessment_decisions["feature_available_timestamp_utc"], utc=True
                ).lt(available)
            ]
            if candidates.empty:
                continue
            latest = candidates.sort_values(
                ["feature_available_timestamp_utc", "decision_ordinal"], kind="mergesort"
            ).iloc[-1]
            rows.append(
                (
                    str(symbol),
                    str(session),
                    pd.Timestamp(timestamp),
                    str(semantic),
                    int(latest["decision_ordinal"]),
                )
            )
        reconstructed = pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "session",
                "completion_timestamp_utc",
                "semantic_loop_id",
                "decision_ordinal",
            ],
        ).sort_values(
            ["symbol", "session", "completion_timestamp_utc", "semantic_loop_id"],
            kind="mergesort",
        )
        archived_keys = archived.loc[:, reconstructed.columns].sort_values(
            ["symbol", "session", "completion_timestamp_utc", "semantic_loop_id"],
            kind="mergesort",
        )
        self.record(
            "registered_event_deduplication",
            reconstructed.reset_index(drop=True).equals(archived_keys.reset_index(drop=True)),
            rows=len(archived),
            duplicate_identities=int(archived.duplicated(identity).sum()),
        )

    def audit_precursors(self, completions: pd.DataFrame) -> None:
        ledger = pd.read_parquet(self.output / "precursor_feature_ledger.parquet")
        relationship_rows: list[dict[str, Any]] = []
        mismatch: dict[str, int] = {field: 0 for field in BOOLEAN_PRECURSORS}
        mismatch.update(
            {
                "windows": 0,
                "prefix_candidate_count": 0,
                "maximum_prefix_depth": 0,
                "regime_transition_count": 0,
                "posterior_transition_probability": 0,
                "posterior_entropy_change": 0,
                "top_state_probability_change": 0,
                "expected_state_age_change": 0,
                "nearest_precursor_priority": 0,
            }
        )
        for row in ledger.itertuples(index=False):
            start = int(row.completion_bar_ordinal) - int(row.lookback_bars)
            end = int(row.completion_bar_ordinal) - 1
            completed = json.loads(str(row.registered_precursors_json))
            prefixes = json.loads(str(row.prefix_precursors_json))
            states = json.loads(str(row.state_metrics_json))
            hard = json.loads(str(row.hard_state_path_json))
            transition_states = json.loads(str(row.transition_state_path_json))
            for kind, identity in sorted(
                {(str(value["kind"]), str(value["identity"])) for value in completed}
            ):
                relationship_rows.append(
                    {
                        "record_type": str(row.record_type),
                        "period": str(row.period),
                        "draw": int(row.draw),
                        "event_id": str(row.event_id),
                        "source_event_id": str(row.source_event_id),
                        "lookback_bars": int(row.lookback_bars),
                        "precursor_kind": kind,
                        "precursor_identity": identity,
                        "target_semantic_loop_id": str(row.target_semantic_loop_id),
                    }
                )
            ordinals = [int(value["bar_ordinal"]) for value in states]
            transition_ordinals = [int(value["bar_ordinal"]) for value in transition_states]
            transition_hard = [int(value["causal_hard_state"]) for value in transition_states]
            windows_ok = (
                int(row.window_start_bar_ordinal) == start
                and int(row.window_end_bar_ordinal) == end
                and all(start <= value <= end for value in ordinals)
                and all(start <= int(value["bar_ordinal"]) <= end for value in completed)
                and all(start <= int(value["bar_ordinal"]) <= end for value in prefixes)
                and bool(row.complete_prior_history) == (ordinals == list(range(start, end + 1)))
                and int(row.history_bars_available) == len(set(ordinals))
                and all(start - 1 <= value <= end for value in transition_ordinals)
                and [value for value in transition_ordinals if value >= start] == ordinals
                and [
                    state
                    for ordinal, state in zip(transition_ordinals, transition_hard, strict=True)
                    if ordinal >= start
                ]
                == hard
            )
            mismatch["windows"] += int(not windows_ok)
            registered = [value for value in completed if value["kind"] == "registered"]
            hidden = [value for value in completed if value["kind"] == "hidden"]
            target = str(row.target_semantic_loop_id)
            target_family = str(row.target_motif_type)
            registered_ids = [str(value["identity"]) for value in registered]
            expected_bool = {
                "same_registered_identity": target in registered_ids,
                "same_registered_broad_family_different_identity": any(
                    str(value.get("motif_type")) == target_family
                    and str(value["identity"]) != target
                    for value in registered
                ),
                "different_registered_broad_family": any(
                    str(value.get("motif_type")) != target_family for value in registered
                ),
                "any_prior_registered_completion": bool(registered),
                "hidden_5_6_5": any(value["identity"] == FROZEN_HIDDEN[0] for value in hidden),
                "hidden_2_3_2": any(value["identity"] == FROZEN_HIDDEN[1] for value in hidden),
                "hidden_2_5_2": any(value["identity"] == FROZEN_HIDDEN[2] for value in hidden),
                "hidden_4_7_4": any(value["identity"] == FROZEN_HIDDEN[3] for value in hidden),
                "hidden_other_unregistered_family": any(
                    value["identity"] == OTHER_HIDDEN for value in hidden
                ),
                "any_hidden_unregistered_completion": bool(hidden),
                "active_prefix_immediately_before_completion": any(
                    int(value["bar_ordinal"]) == end for value in prefixes
                ),
                "active_prefix_any": bool(prefixes),
                "matching_prefix_any": any(
                    str(value["semantic_loop_id"]) == target for value in prefixes
                ),
                "other_prefix_any": any(
                    str(value["semantic_loop_id"]) != target for value in prefixes
                ),
            }
            transition_count = int(
                sum(
                    right_ordinal == left_ordinal + 1
                    and right_ordinal >= start
                    and left_state != right_state
                    for left_ordinal, right_ordinal, left_state, right_state in zip(
                        transition_ordinals[:-1],
                        transition_ordinals[1:],
                        transition_hard[:-1],
                        transition_hard[1:],
                        strict=True,
                    )
                )
            )
            expected_bool["any_regime_transition"] = transition_count > 0
            expected_bool["no_identified_structural_precursor"] = not (
                bool(completed) or bool(prefixes) or transition_count > 0
            )
            for field, expected in expected_bool.items():
                mismatch[field] += int(bool(getattr(row, field)) != bool(expected))
            distinct_prefixes = {
                (
                    int(value["bar_ordinal"]),
                    str(value["semantic_loop_id"]),
                    str(value["orientation_id"]),
                    int(value["progress_states"]),
                )
                for value in prefixes
            }
            mismatch["prefix_candidate_count"] += int(
                int(row.prefix_candidate_count) != len(distinct_prefixes)
            )
            mismatch["maximum_prefix_depth"] += int(
                int(row.maximum_prefix_depth)
                != max((int(value["progress_states"]) for value in prefixes), default=0)
            )
            mismatch["regime_transition_count"] += int(
                int(row.regime_transition_count) != transition_count
            )
            if states:
                expected_numeric = {
                    "posterior_transition_probability": float(states[-1]["transition_probability"]),
                    "posterior_entropy_change": float(states[-1]["posterior_entropy"])
                    - float(states[0]["posterior_entropy"]),
                    "top_state_probability_change": float(states[-1]["top_state_probability"])
                    - float(states[0]["top_state_probability"]),
                    "expected_state_age_change": float(states[-1]["expected_state_age"])
                    - float(states[0]["expected_state_age"]),
                }
                for field, expected in expected_numeric.items():
                    mismatch[field] += int(abs(float(getattr(row, field)) - expected) > 1e-12)
            if completed:
                nearest = sorted(
                    completed, key=lambda value: (int(value["bar_ordinal"]), str(value["identity"]))
                )[-1]
                expected_label = "NEAREST_COMPLETED_LOOP_EVENT"
                nearest_ok = (
                    str(row.nearest_completed_kind) == str(nearest["kind"])
                    and str(row.nearest_completed_identity) == str(nearest["identity"])
                    and int(row.nearest_completed_bars_before)
                    == int(row.completion_bar_ordinal) - int(nearest["bar_ordinal"])
                )
            elif expected_bool["matching_prefix_any"]:
                expected_label = "ACTIVE_MATCHING_PREFIX"
                nearest_ok = True
            elif expected_bool["other_prefix_any"]:
                expected_label = "OTHER_ACTIVE_PREFIX"
                nearest_ok = True
            elif transition_count > 0:
                expected_label = "REGIME_TRANSITION"
                nearest_ok = True
            else:
                expected_label = "NO_IDENTIFIED_STRUCTURAL_PRECURSOR"
                nearest_ok = True
            mismatch["nearest_precursor_priority"] += int(
                str(row.nearest_precursor_label) != expected_label or not nearest_ok
            )
        self.record(
            "precursor_windows_taxonomy_and_nearest_priority",
            not any(mismatch.values()),
            rows=len(ledger),
            mismatches=mismatch,
        )

        observed = ledger.loc[
            ledger["record_type"].eq("observed") & ledger["period"].eq("assessment")
        ]
        census = pd.read_csv(self.output / "precursor_census.csv")
        census_differences: list[float] = []
        for row in census.itertuples(index=False):
            group = observed.loc[observed["lookback_bars"].eq(int(row.lookback_bars))]
            census_differences.append(
                abs(
                    float(group[str(row.precursor_type)].astype(bool).mean())
                    - row.observed_prevalence
                )
            )
        self.record(
            "precursor_census_recalculation",
            max(census_differences, default=0.0) <= 1e-12,
            maximum_difference=max(census_differences, default=0.0),
        )

        null = ledger.loc[ledger["record_type"].eq("null")]
        null_manifest = read_json(self.output / "source_manifest.json")["precursor_null"]
        null_ok = True
        null_details: dict[str, Any] = {}
        total_match_failures = 0
        for period in ("development", "assessment"):
            period_observed = ledger.loc[
                ledger["record_type"].eq("observed") & ledger["period"].eq(period)
            ]
            period_null = null.loc[null["period"].eq(period)]
            observed_lookup = period_observed.drop_duplicates("event_id").set_index("event_id")
            null_once = period_null.drop_duplicates(["draw", "event_id"])
            match_failures = 0
            for row in null_once.itertuples(index=False):
                source = observed_lookup.loc[str(row.source_event_id)]
                completion_collision = bool(
                    (
                        completions["symbol"].astype(str).eq(str(row.symbol))
                        & completions["session"].astype(str).eq(str(row.session))
                        & completions["completion_bar_ordinal"].eq(int(row.completion_bar_ordinal))
                    ).any()
                )
                match_failures += int(
                    str(row.symbol) != str(source["symbol"])
                    or str(row.year_month) != str(source["year_month"])
                    or str(row.clock_bin) != str(source["clock_bin"])
                    or completion_collision
                )
            draw_counts = period_null.groupby("draw")["source_event_id"].nunique()
            eligible_ids = set(
                period_observed.loc[
                    period_observed["lookback_bars"].eq(12)
                    & period_observed["complete_prior_history"].astype(bool),
                    "event_id",
                ].astype(str)
            )
            sampled_ids = set(period_null["source_event_id"].astype(str))
            manifest = null_manifest["periods"][period]
            period_ok = (
                period_null["draw"].nunique() == 25
                and len(draw_counts) == 25
                and draw_counts.nunique() == 1
                and int(draw_counts.iloc[0]) == len(eligible_ids)
                and sampled_ids == eligible_ids
                and int(manifest["matched_full_history_events"]) == len(eligible_ids)
                and match_failures == 0
                and bool(
                    period_null.loc[
                        period_null["lookback_bars"].eq(12), "complete_prior_history"
                    ].all()
                )
            )
            null_ok = null_ok and period_ok
            total_match_failures += match_failures
            null_details[period] = {
                "events_per_draw": int(draw_counts.iloc[0]),
                "eligible_observed_events": len(eligible_ids),
                "match_failures": match_failures,
                "passed": period_ok,
            }
        self.record(
            "stock_month_clock_matched_precursor_null",
            bool(null_ok),
            draws=int(null["draw"].nunique()),
            periods=null_details,
            match_failures=total_match_failures,
        )

        exact = pd.read_csv(self.output / "exact_precursor_multiplicity.csv")
        exact_identity_ok = bool(
            (
                exact["precursor_kind"].eq("registered")
                | (
                    exact["precursor_kind"].eq("hidden")
                    & exact["precursor_identity"].isin(FROZEN_HIDDEN)
                )
            ).all()
        )
        support = (
            exact["development_occurrences"].ge(30)
            & exact["occurrences"].ge(20)
            & exact["sessions"].ge(10)
            & exact["stocks"].ge(8)
        )
        support_match = support.equals(exact["support_passed"].astype(bool))
        eligible = exact.loc[support]
        q_ok = True
        if not eligible.empty:
            q_ok = self.close(bh_adjust(eligible["p_value"].tolist()), eligible["q_value"])
        relationships = pd.DataFrame(relationship_rows)
        direction_differences: list[float] = []
        direction_failures = 0
        maximum_direction_detail: dict[str, Any] = {}
        for row in exact.itertuples(index=False):
            relationship_mask = (
                relationships["lookback_bars"].eq(int(row.lookback_bars))
                & relationships["precursor_kind"].eq(str(row.precursor_kind))
                & relationships["precursor_identity"].eq(str(row.precursor_identity))
                & relationships["target_semantic_loop_id"].eq(str(row.target_semantic_loop_id))
            )
            period_values: dict[str, tuple[int, float, float, float, float]] = {}
            for period in ("development", "assessment"):
                period_null = ledger.loc[
                    ledger["record_type"].eq("null") & ledger["period"].eq(period)
                ]
                matched_ids = set(period_null["source_event_id"].astype(str))
                denominator = len(matched_ids)
                observed_relationships = relationships.loc[
                    relationship_mask
                    & relationships["record_type"].eq("observed")
                    & relationships["period"].eq(period)
                    & relationships["event_id"].astype(str).isin(matched_ids)
                ]
                occurrence = int(observed_relationships["event_id"].nunique())
                prevalence = occurrence / denominator
                null_draw_counts = [
                    int(
                        relationships.loc[
                            relationship_mask
                            & relationships["record_type"].eq("null")
                            & relationships["period"].eq(period)
                            & relationships["draw"].eq(draw),
                            "event_id",
                        ].nunique()
                    )
                    for draw in range(25)
                ]
                mean_null = float(np.mean(null_draw_counts) / denominator)
                p_value = float((1 + sum(value >= occurrence for value in null_draw_counts)) / 26.0)
                period_values[period] = (
                    occurrence,
                    prevalence,
                    mean_null,
                    prevalence - mean_null,
                    p_value,
                )
            development_values = period_values["development"]
            assessment_values = period_values["assessment"]
            archived_values = (
                float(row.development_matched_occurrences),
                float(row.development_matched_prevalence),
                float(row.development_mean_null_prevalence),
                float(row.development_enrichment),
                float(row.matched_assessment_occurrences),
                float(row.matched_assessment_prevalence),
                float(row.mean_null_prevalence),
                float(row.assessment_enrichment),
                float(row.p_value),
            )
            recalculated_values = (
                *map(float, development_values[:4]),
                *map(float, assessment_values),
            )
            comparison_fields = (
                "development_matched_occurrences",
                "development_matched_prevalence",
                "development_mean_null_prevalence",
                "development_enrichment",
                "matched_assessment_occurrences",
                "matched_assessment_prevalence",
                "mean_null_prevalence",
                "assessment_enrichment",
                "p_value",
            )
            for field, archived_value, recalculated_value in zip(
                comparison_fields, archived_values, recalculated_values, strict=True
            ):
                difference = abs(archived_value - recalculated_value)
                direction_differences.append(difference)
                if difference > float(maximum_direction_detail.get("difference", -1.0)):
                    maximum_direction_detail = {
                        "field": field,
                        "difference": difference,
                        "archived": archived_value,
                        "recalculated": recalculated_value,
                        "lookback_bars": int(row.lookback_bars),
                        "precursor_kind": str(row.precursor_kind),
                        "precursor_identity": str(row.precursor_identity),
                        "target_semantic_loop_id": str(row.target_semantic_loop_id),
                    }
            expected_direction = development_values[3] > 0.0 and assessment_values[3] > 0.0
            direction_failures += int(
                bool(row.same_direction_development_assessment) != expected_direction
            )
        self.record(
            "exact_transition_support_and_bh",
            bool(
                support_match
                and exact_identity_ok
                and q_ok
                and direction_failures == 0
                and max(direction_differences, default=0.0) <= 1e-12
            ),
            rows=len(exact),
            supported_rows=int(support.sum()),
            exact_identity_rows_only=exact_identity_ok,
            direction_failures=direction_failures,
            maximum_direction_null_difference=max(direction_differences, default=0.0),
            maximum_direction_detail=maximum_direction_detail,
        )

    def audit_candidate_and_models(
        self,
        archived_assessment: pd.DataFrame,
        archived_development: pd.DataFrame,
        opening: pd.DataFrame,
        t0_features: tuple[str, ...],
    ) -> pd.DataFrame:
        threshold_manifest = read_json(self.output / "candidate_threshold_manifest.json")
        hidden_manifest = read_json(self.output / "hidden_risk_thresholds.json")
        oof = pd.read_parquet(self.output / "b0_development_oof_predictions.parquet")
        development_source = opening.loc[
            opening["year"].eq(2024)
            & opening.loc[:, list(t0_features)].notna().all(axis=1)
            & opening["registered_completion_within_12_bars"].notna()
        ].copy()
        fresh_probability, fresh_folds = expanding_logistic_crossfit(
            development_source,
            features=t0_features,
            target="registered_completion_within_12_bars",
            folds=4,
            warmup_fraction=0.2,
            weight_column="row_weight",
        )
        development_source["B0_oof_probability"] = fresh_probability
        fresh_oof = development_source.loc[development_source["B0_oof_probability"].notna()].copy()
        oof_comparison = oof.loc[:, [*KEYS, "B0_oof_probability"]].merge(
            fresh_oof.loc[:, [*KEYS, "B0_oof_probability"]],
            on=KEYS,
            how="outer",
            suffixes=("_archived", "_fresh"),
            indicator=True,
            validate="one_to_one",
        )
        oof_difference = float(
            np.max(
                np.abs(
                    oof_comparison["B0_oof_probability_archived"].to_numpy(float)
                    - oof_comparison["B0_oof_probability_fresh"].to_numpy(float)
                ),
                initial=0.0,
            )
        )
        self.maximum_difference = max(self.maximum_difference, oof_difference)
        threshold = float(np.quantile(fresh_oof["B0_oof_probability"].to_numpy(float), 0.80))
        folds = pd.DataFrame(threshold_manifest["fold_manifest"])
        chronology = bool(
            folds["train_session_end"]
            .astype(str)
            .lt(folds["prediction_session_start"].astype(str))
            .all()
            and len(folds) == 4
        )
        fresh_fold_records = json.loads(
            json.dumps(fresh_folds.to_dict(orient="records"), default=str)
        )
        self.record(
            "expanding_B0_predictions_and_candidate_threshold",
            chronology
            and oof_comparison["_merge"].eq("both").all()
            and oof_difference <= 1e-12
            and fresh_fold_records == threshold_manifest["fold_manifest"]
            and abs(threshold - float(threshold_manifest["threshold"])) <= 1e-12,
            folds=len(folds),
            source_rows=len(oof),
            threshold=threshold,
            independently_refitted_folds=4,
            maximum_probability_difference=oof_difference,
        )

        population = pd.read_parquet(self.output / "candidate_population.parquet")
        dev = population.loc[population["period"].eq("development")].copy()
        assess = population.loc[population["period"].eq("assessment")].copy()
        expected_assessment = archived_assessment.loc[
            archived_assessment["B0_probability"].ge(threshold), KEYS
        ].sort_values(KEYS, kind="mergesort")
        actual_assessment = assess.loc[:, KEYS].sort_values(KEYS, kind="mergesort")
        membership_ok = expected_assessment.reset_index(drop=True).equals(
            actual_assessment.reset_index(drop=True)
        )
        expected_development = fresh_oof.loc[fresh_oof["B0_oof_probability"].ge(threshold)].merge(
            archived_development.loc[:, [*KEYS, "p_unregistered_within_6_bars"]].rename(
                columns={"p_unregistered_within_6_bars": "development_cross_fitted_U1_probability"}
            ),
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        expected_development_keys = expected_development.loc[:, KEYS].sort_values(
            KEYS, kind="mergesort"
        )
        actual_development_keys = dev.loc[:, KEYS].sort_values(KEYS, kind="mergesort")
        development_membership_ok = expected_development_keys.reset_index(drop=True).equals(
            actual_development_keys.reset_index(drop=True)
        )
        cross_fitted_u1 = dev.loc[:, [*KEYS, "U1_for_veto"]].merge(
            archived_development.loc[:, [*KEYS, "p_unregistered_within_6_bars"]],
            on=KEYS,
            validate="one_to_one",
        )
        cross_fitted_u1_ok = self.close(
            cross_fitted_u1["U1_for_veto"],
            cross_fitted_u1["p_unregistered_within_6_bars"],
        )
        self.record(
            "assessment_candidate_membership",
            membership_ok and development_membership_ok and cross_fitted_u1_ok,
            assessment_rows=len(assess),
            development_rows=len(dev),
            development_cross_fitted_U1_verified=True,
        )

        low = float(
            np.quantile(expected_development["development_cross_fitted_U1_probability"], 0.25)
        )
        high = float(
            np.quantile(expected_development["development_cross_fitted_U1_probability"], 0.75)
        )
        quintiles = np.quantile(
            expected_development["development_cross_fitted_U1_probability"],
            [0.2, 0.4, 0.6, 0.8],
        )
        threshold_ok = (
            abs(low - float(hidden_manifest["low_maximum"])) <= 1e-12
            and abs(high - float(hidden_manifest["high_minimum"])) <= 1e-12
            and self.close(quintiles, hidden_manifest["quintile_boundaries"])
        )
        assignment_failures = 0
        for row in population.itertuples(index=False):
            value = float(row.U1_for_veto)
            group = "low" if value <= low else "high" if value >= high else "middle"
            quintile = int(np.searchsorted(quintiles, value, side="right") + 1)
            assignment_failures += int(
                str(row.hidden_risk_group) != group or int(row.hidden_risk_quintile) != quintile
            )
        self.record(
            "frozen_U1_quartile_and_quintile_thresholds",
            threshold_ok and assignment_failures == 0,
            assignment_failures=assignment_failures,
        )

        predictions = pd.read_parquet(self.output / "veto_assessment_predictions.parquet")
        configuration = read_json(self.output / "veto_model_configurations.json")
        coefficients = read_json(self.output / "veto_model_coefficients.json")
        expected_v0 = ["logit_B0_probability", "checkpoint_12"]
        expected_v1 = [*expected_v0, "logit_U1_probability"]
        model_config_ok = (
            configuration["V0"]["features"] == expected_v0
            and configuration["V1"]["features"] == expected_v1
            and configuration["actual_hidden_event_excluded"] is True
            and coefficients["V0"]["feature_names"] == expected_v0
            and coefficients["V1"]["feature_names"] == expected_v1
        )
        feature_ok = self.close(
            predictions["logit_B0_probability"],
            np.log(
                np.clip(predictions["B0_probability"], 1e-6, 1 - 1e-6)
                / (1 - np.clip(predictions["B0_probability"], 1e-6, 1 - 1e-6))
            ),
        ) and self.close(
            predictions["logit_U1_probability"],
            np.log(
                np.clip(predictions["U1_for_veto"], 1e-6, 1 - 1e-6)
                / (1 - np.clip(predictions["U1_for_veto"], 1e-6, 1 - 1e-6))
            ),
        )
        self.record(
            "V0_V1_features_and_actual_hidden_exclusion",
            model_config_ok and feature_ok,
        )
        refit_development = dev.copy().reset_index(drop=True)
        clipped_development_b0 = np.clip(
            refit_development["candidate_B0_probability"], 1e-6, 1 - 1e-6
        )
        clipped_development_u1 = np.clip(refit_development["U1_for_veto"], 1e-6, 1 - 1e-6)
        refit_development["logit_B0_probability"] = np.log(
            clipped_development_b0 / (1.0 - clipped_development_b0)
        )
        refit_development["checkpoint_12"] = (
            refit_development["decision_ordinal"].eq(12).astype(float)
        )
        refit_development["logit_U1_probability"] = np.log(
            clipped_development_u1 / (1.0 - clipped_development_u1)
        )
        refit_v0 = fit_weighted_logistic(
            refit_development,
            features=tuple(expected_v0),
            target="registered_completion_within_12_bars",
        )
        refit_v1 = fit_weighted_logistic(
            refit_development,
            features=tuple(expected_v1),
            target="registered_completion_within_12_bars",
        )
        refit_checks: list[bool] = []
        for model_name, fitted in (("V0", refit_v0), ("V1", refit_v1)):
            refit_spec = fitted.as_dict()
            archived_spec = coefficients[model_name]
            refit_checks.extend(
                [
                    self.close(refit_spec["scaler_mean"], archived_spec["scaler_mean"]),
                    self.close(refit_spec["scaler_scale"], archived_spec["scaler_scale"]),
                    self.close(refit_spec["coefficient"], archived_spec["coefficient"]),
                    self.close([refit_spec["intercept"]], [archived_spec["intercept"]]),
                    self.close(
                        fitted.predict_probability(predictions),
                        predictions[f"{model_name}_probability"],
                    ),
                ]
            )
        self.record(
            "model_coefficients_independent_refit",
            all(refit_checks),
            refitted_models=2,
            settings={
                "penalty": "l2",
                "C": 0.25,
                "solver": "liblinear",
                "max_iter": 300,
                "class_weight": None,
                "n_jobs": 1,
            },
        )
        manual_rows = predictions.iloc[: max(100, min(len(predictions), 100))]
        v0_manual = serialised_probability(manual_rows, coefficients["V0"])
        v1_manual = serialised_probability(manual_rows, coefficients["V1"])
        manual_ok = self.close(v0_manual, manual_rows["V0_probability"]) and self.close(
            v1_manual, manual_rows["V1_probability"]
        )
        self.record(
            "manual_probability_reconstruction",
            manual_ok and len(manual_rows) >= 100,
            rows=len(manual_rows),
        )

        assessment_weight = archived_assessment.loc[:, [*KEYS, "row_weight", "slate_id"]]
        weight_comparison = predictions.loc[:, [*KEYS, "row_weight"]].merge(
            assessment_weight,
            on=KEYS,
            suffixes=("_screen", "_predecessor"),
            validate="one_to_one",
        )
        slate_sums = archived_assessment.groupby("slate_id")["row_weight"].sum()
        weight_ok = self.close(
            weight_comparison["row_weight_screen"], weight_comparison["row_weight_predecessor"]
        ) and self.close(slate_sums, np.ones(len(slate_sums)))
        self.record("exact_slate_weighting", weight_ok)

        archived_metrics = pd.read_csv(self.output / "veto_metrics.csv").set_index("model")
        metric_differences: dict[str, float] = {}
        for model in ("V0", "V1"):
            recalculated = core_metrics(predictions, f"{model}_probability")
            for metric, value in recalculated.items():
                difference = abs(value - float(archived_metrics.loc[model, metric]))
                metric_differences[f"{model}_{metric}"] = difference
        self.record(
            "log_loss_brier_auc_average_precision",
            max(metric_differences.values(), default=0.0) <= 1e-12,
            differences=metric_differences,
        )
        return predictions

    def audit_realised_diversion(self, predictions: pd.DataFrame) -> None:
        archived = pd.read_csv(self.output / "realised_diversion_metrics.csv")
        successful = predictions.loc[predictions["registered_completion_within_12_bars"].eq(1)]
        failed = predictions.loc[predictions["registered_completion_within_12_bars"].eq(0)]
        expected = {
            "hidden_event_rate_successful": weighted_rate(
                successful, "actual_hidden_event_within_6_bars"
            ),
            "hidden_event_rate_failed": weighted_rate(failed, "actual_hidden_event_within_6_bars"),
        }
        expected["failed_minus_successful_hidden_event_rate"] = (
            expected["hidden_event_rate_failed"] - expected["hidden_event_rate_successful"]
        )
        differences = []
        for metric, value in expected.items():
            archived_value = float(
                archived.loc[
                    archived["scope"].eq("pooled") & archived["metric"].eq(metric), "value"
                ].iloc[0]
            )
            differences.append(abs(value - archived_value))
        for family in (*FROZEN_HIDDEN, OTHER_HIDDEN):
            family_event = predictions["hidden_family_class"].astype(str).eq(family).astype(float)
            temporary = predictions.assign(_family_event=family_event)
            family_success = weighted_rate(
                temporary.loc[temporary["registered_completion_within_12_bars"].eq(1)],
                "_family_event",
            )
            family_failed = weighted_rate(
                temporary.loc[temporary["registered_completion_within_12_bars"].eq(0)],
                "_family_event",
            )
            family_expected = {
                "hidden_family_rate_successful": family_success,
                "hidden_family_rate_failed": family_failed,
                "hidden_family_rate_failed_minus_successful": family_failed - family_success,
            }
            for metric, value in family_expected.items():
                archived_value = float(
                    archived.loc[
                        archived["scope"].eq(family) & archived["metric"].eq(metric),
                        "value",
                    ].iloc[0]
                )
                differences.append(abs(value - archived_value))
        mechanism_columns = (
            "hidden_event_before_registered_completion",
            "hidden_event_with_no_later_registered_completion",
            "registered_completion_with_no_hidden_event",
        )
        for column in mechanism_columns:
            archived_value = float(
                archived.loc[
                    archived["scope"].eq("pooled") & archived["metric"].eq(column), "value"
                ].iloc[0]
            )
            differences.append(abs(weighted_rate(predictions, column) - archived_value))
        self.record(
            "realised_diversion_diagnostic",
            max(differences, default=0.0) <= 1e-12,
            maximum_difference=max(differences, default=0.0),
        )

    def audit_bootstrap(self, predictions: pd.DataFrame) -> None:
        archived = pd.read_csv(self.output / "bootstrap_metrics.csv")
        sessions = sorted(predictions["session"].astype(str).unique())
        positions = {
            session: np.flatnonzero(predictions["session"].astype(str).eq(session).to_numpy())
            for session in sessions
        }
        generator = np.random.default_rng(20260722)
        differences: list[float] = []
        draw_values: dict[str, list[float]] = {}
        for draw in range(25):
            sampled_sessions = generator.choice(
                np.asarray(sessions, dtype=object), size=len(sessions), replace=True
            )
            indices = np.concatenate([positions[str(session)] for session in sampled_sessions])
            sample = predictions.iloc[indices]
            values = increments(sample)
            high = sample.loc[sample["hidden_risk_group"].eq("high")]
            low = sample.loc[sample["hidden_risk_group"].eq("low")]
            failed = sample.loc[sample["registered_completion_within_12_bars"].eq(0)]
            success = sample.loc[sample["registered_completion_within_12_bars"].eq(1)]
            values["high_minus_low_completion_rate"] = weighted_rate(
                high, "registered_completion_within_12_bars"
            ) - weighted_rate(low, "registered_completion_within_12_bars")
            values["failed_minus_successful_hidden_event_rate"] = weighted_rate(
                failed, "actual_hidden_event_within_6_bars"
            ) - weighted_rate(success, "actual_hidden_event_within_6_bars")
            for metric, value in values.items():
                draw_values.setdefault(metric, []).append(value)
                archived_value = float(
                    archived.loc[
                        archived["record_type"].eq("draw")
                        & archived["draw"].eq(draw)
                        & archived["metric"].eq(metric),
                        "value",
                    ].iloc[0]
                )
                differences.append(abs(value - archived_value))
        for metric, values in draw_values.items():
            for level in (0.8, 0.9, 0.95):
                alpha = (1.0 - level) / 2.0
                row = archived.loc[
                    archived["record_type"].eq("interval")
                    & archived["metric"].eq(metric)
                    & archived["interval_level"].eq(level)
                ].iloc[0]
                differences.extend(
                    [
                        abs(float(np.quantile(values, alpha)) - float(row["lower"])),
                        abs(float(np.quantile(values, 1.0 - alpha)) - float(row["upper"])),
                    ]
                )
        self.record(
            "fixed_prediction_whole_session_bootstrap",
            max(differences, default=0.0) <= 1e-12
            and archived.loc[archived["record_type"].eq("draw"), "draw"].nunique() == 25,
            draws=25,
            maximum_difference=max(differences, default=0.0),
            models_refit=False,
        )

    @staticmethod
    def permute(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
        result = frame.copy()
        generator = np.random.default_rng(seed)
        for _, positions in result.groupby("slate_id", sort=True).indices.items():
            index = np.asarray(positions, dtype=int)
            values = result.iloc[index]["U1_for_veto"].to_numpy(copy=True)
            result.iloc[index, result.columns.get_loc("U1_for_veto")] = generator.permutation(
                values
            )
        return result

    def audit_hidden_null(self, predictions: pd.DataFrame) -> None:
        archived = pd.read_csv(self.output / "veto_null_metrics.csv")
        oof = pd.read_parquet(self.output / "b0_development_oof_predictions.parquet")
        development_u1 = pd.read_parquet(BRIDGE_PRIMARY / "bridge_development_panel.parquet").loc[
            :, [*KEYS, "p_unregistered_within_6_bars"]
        ]
        development = oof.merge(development_u1, on=KEYS, validate="one_to_one")
        development["U1_for_veto"] = development["p_unregistered_within_6_bars"]
        assessment = pd.read_parquet(BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet")
        assessment["U1_for_veto"] = assessment["U1_probability"]
        threshold = float(read_json(self.output / "candidate_threshold_manifest.json")["threshold"])
        real = increments(predictions)
        differences: list[float] = []
        fingerprint_failures = 0
        for draw in range(5):
            dev = self.permute(development, 20260723 + draw)
            assess = self.permute(assessment, 20260723 + draw)
            dev_candidate = dev.loc[dev["B0_oof_probability"].ge(threshold)].reset_index(drop=True)
            assess_candidate = assess.loc[assess["B0_probability"].ge(threshold)].reset_index(
                drop=True
            )
            row = archived.loc[archived["draw"].eq(draw)].iloc[0]
            dev_hash = hashlib.sha256(
                dev_candidate["U1_for_veto"].to_numpy(np.float64).tobytes()
            ).hexdigest()
            assess_hash = hashlib.sha256(
                assess_candidate["U1_for_veto"].to_numpy(np.float64).tobytes()
            ).hexdigest()
            fingerprint_failures += int(
                dev_hash != str(row["development_permutation_sha256"])
                or assess_hash != str(row["assessment_permutation_sha256"])
            )
            clipped_b0 = np.clip(assess_candidate["B0_probability"], 1e-6, 1 - 1e-6)
            clipped_u1 = np.clip(assess_candidate["U1_for_veto"], 1e-6, 1 - 1e-6)
            assess_candidate["logit_B0_probability"] = np.log(clipped_b0 / (1 - clipped_b0))
            assess_candidate["checkpoint_12"] = (
                assess_candidate["decision_ordinal"].eq(12).astype(float)
            )
            assess_candidate["logit_U1_probability"] = np.log(clipped_u1 / (1 - clipped_u1))
            specification = json.loads(str(row["V1_model_json"]))
            assess_candidate["V0_probability"] = predictions["V0_probability"].to_numpy()
            assess_candidate["V1_probability"] = serialised_probability(
                assess_candidate, specification
            )
            value = increments(assess_candidate)
            for metric, increment in value.items():
                archived_value = float(
                    archived.loc[
                        archived["draw"].eq(draw) & archived["metric"].eq(metric),
                        "null_increment",
                    ].iloc[0]
                )
                differences.append(abs(increment - archived_value))
                expected_exceeds = real[metric] > increment
                actual_exceeds = bool(
                    archived.loc[
                        archived["draw"].eq(draw) & archived["metric"].eq(metric),
                        "real_exceeds_null",
                    ].iloc[0]
                )
                fingerprint_failures += int(expected_exceeds != actual_exceeds)
        self.record(
            "within_slate_hidden_probability_null",
            archived["draw"].nunique() == 5
            and fingerprint_failures == 0
            and max(differences, default=0.0) <= 1e-12,
            draws=5,
            fingerprint_or_comparison_failures=fingerprint_failures,
            maximum_difference=max(differences, default=0.0),
        )

    def audit_decision(self, decision: dict[str, Any], predictions: pd.DataFrame) -> None:
        event = pd.read_parquet(self.output / "registered_completion_event_ledger.parquet")
        feature = pd.read_parquet(self.output / "precursor_feature_ledger.parquet")
        complete_rate = float(
            feature.loc[
                feature["record_type"].eq("observed")
                & feature["period"].eq("assessment")
                & feature["lookback_bars"].eq(12),
                "complete_prior_history",
            ].mean()
        )
        precursor_support = (
            len(event) >= 500
            and event["session"].nunique() >= 120
            and event["symbol"].nunique() >= 15
            and event["year_month"].nunique() >= 6
            and complete_rate >= 0.90
        )
        high = predictions.loc[predictions["hidden_risk_group"].eq("high")]
        low = predictions.loc[predictions["hidden_risk_group"].eq("low")]
        max_stock = float(predictions["symbol"].value_counts(normalize=True).max())
        max_class = float(
            predictions["registered_completion_within_12_bars"].value_counts(normalize=True).max()
        )
        candidate_support = (
            len(predictions) >= 1000
            and predictions["session"].nunique() >= 120
            and predictions["symbol"].nunique() >= 15
            and predictions["registered_completion_within_12_bars"].sum() >= 150
            and predictions["actual_hidden_event_within_6_bars"].sum() >= 150
            and len(high) >= 200
            and len(low) >= 200
            and max_stock <= 0.10
            and max_class <= 0.85
        )
        precursor_status = (
            "insufficient_support"
            if not precursor_support
            else "supported"
            if decision["precursor_gate"]["registered_precursor_structure_passed"]
            else "descriptive_only"
            if any(
                value["checks"]["positive_enrichment"]
                for value in decision["precursor_gate"]["broad_precursor_results"]
            )
            else "not_supported"
        )
        predictive_status = (
            "insufficient_support"
            if not candidate_support
            else "supported"
            if all(decision["predictive_veto_gate"]["gate_checks"].values())
            else "descriptive_only"
            if float(decision["V1_hidden_risk_coefficient"]) < 0
            or weighted_rate(high, "registered_completion_within_12_bars")
            - weighted_rate(low, "registered_completion_within_12_bars")
            < 0
            else "not_supported"
        )
        failed = predictions.loc[predictions["registered_completion_within_12_bars"].eq(0)]
        success = predictions.loc[predictions["registered_completion_within_12_bars"].eq(1)]
        realised_difference = weighted_rate(
            failed, "actual_hidden_event_within_6_bars"
        ) - weighted_rate(success, "actual_hidden_event_within_6_bars")
        realised_status = (
            "insufficient_support"
            if not candidate_support
            else "supported"
            if all(decision["realised_diversion_gate"]["gate_checks"].values())
            else "descriptive_only"
            if realised_difference > 0
            else "not_supported"
        )
        if precursor_status == "supported" and predictive_status == "supported":
            primary = "hidden_diversion_veto_and_registered_precursor_supported"
        elif predictive_status == "supported":
            primary = "hidden_diversion_veto_supported_only"
        elif precursor_status == "supported":
            primary = "registered_precursor_structure_supported_only"
        elif "descriptive_only" in (precursor_status, predictive_status, realised_status) or (
            realised_status == "supported"
        ):
            primary = "descriptive_precursor_or_veto_structure_only"
        elif precursor_status == "insufficient_support":
            primary = "blocked_precursor_support_failure"
        elif predictive_status == "insufficient_support":
            primary = "blocked_candidate_veto_support_failure"
        else:
            primary = "no_hidden_veto_or_precursor_enrichment"
        self.record(
            "decision_logic",
            precursor_status == decision["precursor_status"]
            and predictive_status == decision["predictive_veto_status"]
            and realised_status == decision["realised_diversion_status"]
            and primary == decision["primary_decision"],
            reconstructed_primary_decision=primary,
            precursor_support=precursor_support,
            candidate_support=candidate_support,
        )

    def finish(self, decision: dict[str, Any]) -> int:
        passed = bool(self.checks) and all(value["passed"] for value in self.checks.values())
        result = {
            **SAFETY_FLAGS,
            "passed": passed,
            "independent": True,
            "checks": self.checks,
            "maximum_numeric_difference": self.maximum_difference,
            "manual_probability_rows": self.checks.get("manual_probability_reconstruction", {}).get(
                "rows", 0
            ),
            "bootstrap_repeated": False,
            "precursor_null_repeated": False,
            "hidden_probability_null_models_refit": False,
        }
        write_json(self.output / "lightweight_audit.json", result)
        decision["lightweight_audit_passed"] = passed
        if not passed:
            decision["pre_audit_primary_decision"] = decision["primary_decision"]
            decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
        write_json(self.output / "decision.json", decision)
        report_path = self.output / "report.md"
        report = report_path.read_text(encoding="utf-8")
        marker = "\n## Lightweight independent audit\n"
        report = report.split(marker, maxsplit=1)[0]
        report += (
            f"{marker}\nPassed: `{str(passed).lower()}`. "
            f"Checks passed: {sum(value['passed'] for value in self.checks.values())}/"
            f"{len(self.checks)}. Maximum numeric difference: "
            f"`{self.maximum_difference:.3g}`.\n"
        )
        report_path.write_text(report, encoding="utf-8")
        REPORT_COPY.parent.mkdir(parents=True, exist_ok=True)
        REPORT_COPY.write_text(report, encoding="utf-8")
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    output = parse_args().output.expanduser().resolve()
    audit = Audit(output)
    contract, decision = audit.audit_required_files_and_safety()
    del contract
    opening, completions, assessment, development, t0_features = audit.reconstruct_opening()
    audit.audit_dates(opening)
    audit.audit_event_dedup(opening, completions)
    audit.audit_precursors(completions)
    predictions = audit.audit_candidate_and_models(assessment, development, opening, t0_features)
    audit.audit_realised_diversion(predictions)
    audit.audit_bootstrap(predictions)
    audit.audit_hidden_null(predictions)
    audit.audit_decision(decision, predictions)
    return audit.finish(decision)


if __name__ == "__main__":
    raise SystemExit(main())
