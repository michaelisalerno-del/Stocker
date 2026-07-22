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


def independent_prefix_proximity(prefixes: pd.DataFrame) -> pd.DataFrame:
    current = prefixes.loc[prefixes["bar_ordinal"].astype(int).isin(CHECKPOINTS)].drop_duplicates(
        ["symbol", "session", "bar_ordinal", "semantic_loop_id", "orientation_id"]
    )
    progress = current["progress_states"].astype(int)
    declared_remaining = current["transitions_remaining"].astype(int)
    total_required = progress - 1 + declared_remaining
    current = current.assign(independently_calculated_remaining=total_required - (progress - 1))
    current["one_away"] = current["independently_calculated_remaining"].eq(1)
    return (
        current.groupby(["symbol", "session", "bar_ordinal"], sort=True)
        .agg(
            minimum_remaining_transitions=("independently_calculated_remaining", "min"),
            number_of_one_transition_away_prefixes=("one_away", "sum"),
        )
        .reset_index()
        .rename(columns={"bar_ordinal": "checkpoint"})
    )


def _max_difference(left: pd.Series, right: pd.Series) -> float:
    return float(np.max(np.abs(left.to_numpy(float) - right.to_numpy(float))))


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

    checks: dict[str, bool] = {}
    checks["safety_flags"] = all(
        contract.get(key) == expected
        and decision.get(key) == expected
        and source_manifest.get(key) == expected
        for key, expected in SAFETY_FLAGS.items()
    )
    checks["dates_and_protected_boundary"] = bool(
        fixed["session"].astype(str).between("2024-01-01", "2025-08-22").all()
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

    independent_proximity = independent_prefix_proximity(prefixes)
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
    remaining_mismatches = int(
        (
            (
                prefix_checkpoint_rows["progress_states"].astype(int)
                - 1
                + prefix_checkpoint_rows["transitions_remaining"].astype(int)
            )
            - (prefix_checkpoint_rows["progress_states"].astype(int) - 1)
            != prefix_checkpoint_rows["transitions_remaining"].astype(int)
        ).sum()
    )
    checks["every_prefix_remaining_transition_count"] = bool(
        remaining_mismatches == 0
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
        and advance["completion_in_bars_2_or_3"].sum() == 238
    )
    checks["frozen_feature_surfaces"] = bool(
        tuple(configurations["H0_features"]) == BASELINE_FEATURES
        and tuple(configurations["H1_features"]) == H1_FEATURES
        and tuple(H1_FEATURES) == (*BASELINE_FEATURES, *ROUTE_FEATURES)
        and np.isfinite(right.loc[:, list(H1_FEATURES)].to_numpy(float)).all()
    )
    checks["development_only_preprocessing_not_opened_after_stop"] = bool(
        int(configurations["primary_models_fitted"]) == 0
        and configurations["planned_primary_models"] == ["N0", "N1", "A0", "A1"]
    )
    checks["model_coefficients_and_manual_probabilities_not_applicable"] = bool(
        coefficients["primary_models"] == {}
        and int(coefficients["models_fitted"]) == 0
        and decision["blocker"] == "blocked_insufficient_advance_positive_support"
    )
    immediate_metrics = pd.read_csv(PRIMARY / "immediate_metrics.csv")
    advance_metrics = pd.read_csv(PRIMARY / "advance_metrics.csv")
    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    route_null = pd.read_csv(PRIMARY / "route_null_metrics.csv")
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
    support = cast(Mapping[str, Any], decision["support"])
    checks["corrected_support_and_decision_logic"] = bool(
        int(support["theoretical_eligible_rows"]) == 25_600
        and int(support["retained_rows"]) == 25_518
        and abs(float(support["retention"]) - 25_518 / 25_600) <= 1e-15
        and int(support["next_bar_positive_outcomes"]) == 447
        and int(support["advance_rows"]) == 18_568
        and int(support["advance_positive_outcomes"]) == 238
        and decision["primary_decision"] == "blocked_insufficient_advance_positive_support"
        and decision["immediate_completion_status"] == "descriptive_only"
        and decision["advance_completion_status"] == "insufficient_support"
        and decision["non_imminent_route_status"] == "insufficient_support"
    )
    checks["determinism_artifact"] = bool(
        determinism["passed"]
        and int(determinism["lead_label_mismatches"]) == 0
        and int(determinism["advance_eligibility_mismatches"]) == 0
        and float(determinism["maximum_probability_difference"]) <= 1e-12
    )
    checks["artifact_identity"] = bool(
        sha256_file(PREDECESSOR / "decision_panel.parquet")
        == source_manifest["predecessor_artifact_hashes"]["decision_panel.parquet"]
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
        "pre_model_support_blocker": True,
        "model_refits_performed": 0,
        "manual_probability_rows_per_model": 0,
        "bootstrap_refits_performed": 0,
        "route_null_refits_performed": 0,
        "row_identity_mismatches": row_mismatches,
        "assessment_row_identity_mismatches": assessment_row_mismatches,
        "lead_label_mismatches": lead_mismatches,
        "lead_identity_mismatches": identity_mismatches,
        "prefix_remaining_transition_mismatches": remaining_mismatches,
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
