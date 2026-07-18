from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORK_DIR.parents[3]
PRIMARY = WORK_DIR / "artifacts" / "20260718-loop-event-semantics-v2" / "primary"
EXACT = WORK_DIR / "artifacts" / "20260718-loop-event-semantics-v2" / "exact_rerun"
CONTRACT = WORK_DIR / "contracts" / "20260718-loop-event-semantics-v2.json"
REPORT = WORK_DIR / "reports" / "20260718-loop-event-semantics-v2.md"
SAFETY = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "strategy_promotion": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_contract_is_research_only_and_forbids_prediction_and_execution() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert all(contract[key] == value for key, value in SAFETY.items())
    assert contract["scope"]["predictive_model_training"] is False
    assert contract["scope"]["economic_edge_claim"] is False
    assert contract["source"]["protected_prospective_data_enabled"] is False


def test_implementation_census_preceded_rewrite_and_covers_required_defects() -> None:
    census = pd.read_csv(PRIMARY / "implementation_census.csv")
    defect_columns = {
        "affected_b0_provenance",
        "affected_overlapping_target",
        "affected_primitive_repeat_mix",
        "affected_rank_id",
        "affected_length_mismatch",
        "affected_duration_tail",
        "affected_hard_state_only",
        "affected_run_entry_only",
        "affected_static_future_context",
        "affected_weak_null",
        "affected_raw_frequency",
    }

    scope = json.loads(
        (WORK_DIR / "contracts" / "20260718-loop-implementation-census-scope.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(census) >= 100
    assert census["file"].nunique() >= 100
    assert scope["classification_policy"].startswith("deterministic_ast")
    assert defect_columns <= set(census)
    assert census["file"].str.contains("run_causal_semimarkov_regime_loops.py").any()
    assert census["file"].str.contains("factor_conditioned_loop_occurrence_core.py").any()


def test_b0_source_position_is_confirmed_but_2024_b0_values_are_invariant() -> None:
    summary = pd.read_csv(PRIMARY / "b0_start_end_difference_summary.csv").set_index("field")

    assert summary.loc["b0_state_numeric", "runs_start_end_differ"] == 0
    assert summary.loc["b0_high_stress", "runs_start_end_differ"] == 0
    assert summary.loc["clock_sin", "runs_start_end_differ"] > 70_000
    assert summary.loc["clock_cos", "runs_start_end_differ"] > 70_000


def test_feature_availability_manifest_uses_bar_completion_not_bar_start() -> None:
    manifest = json.loads(
        (PRIMARY / "feature_availability_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["provider_timestamp_semantics"] == "bar_start"
    assert manifest["bar_duration_minutes"] == 5
    assert manifest["fields"]["current_bar_log_return"]["available_timestamp_column"] == (
        "bar_complete_timestamp"
    )
    assert manifest["fields"]["b0_state_numeric"]["source_timestamp_column"] == (
        "b0_source_timestamp"
    )
    required_provenance = {
        "source_timestamp_column",
        "source_bar_ordinal_column",
        "available_timestamp_column",
        "decision_timestamp_column",
        "causal_valid_column",
        "missing_reason_column",
        "source_artifact_hash_column",
    }
    assert all(
        required_provenance <= set(specification) for specification in manifest["fields"].values()
    )


def test_legacy_migration_is_total_deterministic_and_explicit_about_repeats() -> None:
    migration = pd.read_csv(PRIMARY / "legacy_to_v2_loop_mapping.csv")

    assert len(migration) == 20
    assert migration["legacy_cycle_id"].is_unique
    assert migration["migration_status"].eq("migrated").all()
    assert migration["ambiguity_reason"].isna().all()
    assert migration["motif_type"].eq("repeat").sum() == 3
    assert migration["motif_type"].eq("composite").sum() == 1


def test_semantic_dictionary_is_structural_and_not_raw_frequency_ranked() -> None:
    dictionary = pd.read_csv(PRIMARY / "semantic_loop_dictionary_v2.csv")

    assert len(dictionary) == 20
    assert dictionary["semantic_loop_id"].is_unique
    assert dictionary["fdr_q_value"].le(0.1).all()
    assert dictionary["rate_ratio"].ge(1.05).all()
    raw_order = dictionary.sort_values("observed_completions", ascending=False)[
        "semantic_loop_id"
    ].tolist()
    selected_order = dictionary.sort_values("selection_rank")["semantic_loop_id"].tolist()
    assert raw_order != selected_order


def test_first_event_population_is_mutually_exclusive_and_complete() -> None:
    outcomes = pd.read_parquet(
        PRIMARY / "first_next_loop_outcomes.parquet",
        columns=["decision_id", "primary_label", "tied_semantic_loop_ids"],
    )
    decisions = pd.read_parquet(
        PRIMARY / "causal_completed_bar_decisions.parquet", columns=["decision_id"]
    )

    metadata = json.loads((PRIMARY / "run_metadata.json").read_text(encoding="utf-8"))

    assert len(outcomes) == len(decisions) == metadata["completed_bar_decisions"]
    assert outcomes["decision_id"].is_unique
    assert outcomes["primary_label"].notna().all()
    tied = outcomes["primary_label"].eq("TIED_REGISTERED_COMPLETION")
    assert outcomes.loc[tied, "tied_semantic_loop_ids"].map(len).ge(2).all()


def test_legacy_and_v2_target_difference_is_quantified_on_every_decision() -> None:
    detail = pd.read_parquet(
        PRIMARY / "legacy_v2_target_comparison_detail.parquet",
        columns=[
            "decision_id",
            "legacy_positive_count",
            "active_prefix_count",
            "registered_event_set_differs",
            "semantics_differ",
            "comparison_available",
        ],
    )

    metadata = json.loads((PRIMARY / "run_metadata.json").read_text(encoding="utf-8"))

    assert len(detail) == metadata["completed_bar_decisions"]
    assert detail["legacy_positive_count"].gt(1).sum() > 0
    assert detail["active_prefix_count"].gt(0).sum() > 0
    assert detail["registered_event_set_differs"].sum() > 0
    assert not detail.loc[~detail["comparison_available"], "semantics_differ"].any()
    assert detail.loc[detail["comparison_available"], "semantics_differ"].any()


def test_duration_artifacts_keep_24_exact_and_terminal_runs_censored() -> None:
    tail = pd.read_csv(PRIMARY / "duration_tail_diagnostics.csv")
    censor = pd.read_csv(PRIMARY / "duration_censoring_audit.csv")

    assert tail["duration_24_is_exact"].all()
    assert tail["duration_25_is_separate"].all()
    assert not tail["forced_exit_at_24"].any()
    assert tail["hazard_at_24"].lt(1.0).all()
    assert censor["exact_duration_24_count"].sum() > 0
    assert censor["duration_greater_than_24_count"].sum() > 0
    metadata = json.loads((PRIMARY / "run_metadata.json").read_text(encoding="utf-8"))
    eligible = censor["censoring_population"].eq("source_complete_duration_fit")
    invalid = censor["censoring_population"].eq("gap_invalidated_excluded_not_censored")

    assert (
        censor.loc[eligible, "right_censored_terminal_runs"].sum() == metadata["session_sequences"]
    )
    assert not censor.loc[invalid, "right_censored_terminal_runs"].any()


def test_state_posterior_ledger_contains_v2_tail_aware_state_age_surface() -> None:
    schema = pq.read_schema(PRIMARY / "state_posterior_ledger.parquet")
    state_type = schema.field("posterior_state_probabilities").type
    age_type = schema.field("state_age_posterior").type
    next_type = schema.field("next_state_probabilities").type

    assert state_type.list_size == 8
    assert age_type.list_size == 8 * 78
    assert next_type.list_size == 8
    metadata = json.loads((PRIMARY / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["state_age_posterior_support"] == 78
    assert (
        pq.ParquetFile(PRIMARY / "state_posterior_ledger.parquet").metadata.num_rows
        == (metadata["completed_bar_decisions"])
    )


def test_every_completed_bar_is_present_and_run_entries_are_a_strict_subset() -> None:
    population = pd.read_csv(PRIMARY / "run_entry_vs_per_bar_population.csv").set_index(
        "population"
    )

    metadata = json.loads((PRIMARY / "run_metadata.json").read_text(encoding="utf-8"))

    assert (
        population.loc["all_completed_regular_session_bars", "rows"]
        == metadata["completed_bar_decisions"]
    )
    assert (
        population.loc["legacy_state_run_entries", "rows"]
        < population.loc["all_completed_regular_session_bars", "rows"]
    )
    assert population["run_entry_is_strict_subset"].all()


def test_completion_ledger_exports_every_registered_event_not_only_the_first() -> None:
    completions = pd.read_parquet(
        PRIMARY / "loop_completion_event_ledger.parquet",
        columns=["decision_id", "semantic_loop_id", "is_primary_completion"],
    )
    outcomes = pd.read_parquet(
        PRIMARY / "first_next_loop_outcomes.parquet",
        columns=["decision_id", "every_completion_within_horizon"],
    )
    exported_ids = (
        completions.groupby("decision_id", sort=False)["semantic_loop_id"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    expected_ids = {
        str(row.decision_id): sorted(set(row.every_completion_within_horizon))
        for row in outcomes.itertuples(index=False)
        if len(row.every_completion_within_horizon) > 0
    }

    assert exported_ids == expected_ids
    assert completions[["decision_id", "semantic_loop_id"]].notna().all().all()
    assert completions["is_primary_completion"].any()
    assert len(completions) >= sum(len(values) for values in expected_ids.values())


def test_candidate_census_covers_full_allowed_lengths_and_explicit_exclusions() -> None:
    candidates = pd.read_parquet(PRIMARY / "dictionary_candidate_census.parquet")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    allowed = set(contract["dictionary"]["allowed_primitive_transition_lengths"]) | set(
        contract["dictionary"]["allowed_composite_transition_lengths"]
    )
    excluded = candidates.loc[~candidates["support_eligible_for_null"]]

    assert set(candidates["full_transition_length"]).issubset(allowed)
    assert candidates["full_transition_length"].ge(6).any()
    assert not excluded.empty
    assert excluded["candidate_status"].eq("excluded_before_null").all()
    assert excluded["exclusion_reason"].notna().all()


def test_primary_null_has_2000_draws_and_no_economic_surface() -> None:
    draws = np.load(PRIMARY / "structural_null_draws.npz")
    nulls = pd.read_csv(PRIMARY / "structural_null_results.csv")

    assert draws["primary_draws"].shape[0] == 2_000
    assert draws["clock_conditioned_draws"].shape[0] == 500
    assert draws["research_only"][0]
    assert not draws["execution_enabled"][0]
    assert set(nulls["null_name"]) == {
        "NULL_A_FITTED_SEMI_MARKOV",
        "NULL_B_CLOCK_CONDITIONED_SEMI_MARKOV",
        "NULL_C_FIRST_ORDER_ANALYTICAL",
        "NULL_D_WHOLE_SESSION_CIRCULAR_CONTROL",
    }
    forbidden = ("payoff", "pnl", "profit", "mfe", "mae", "return")
    assert not any(token in column.lower() for column in nulls for token in forbidden)


def test_independent_auditor_does_not_import_production_v2_modules() -> None:
    source = (WORK_DIR / "audit_loop_event_semantics_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(name.startswith("stocker_research") for name in imported)


def test_new_source_has_no_broker_order_position_or_execution_import() -> None:
    package_dir = REPO_ROOT / "packages" / "stocker_research" / "src" / "stocker_research"
    sources = [
        package_dir / "loop_events_v2.py",
        package_dir / "loop_dictionary_v2.py",
        package_dir / "loop_prefix_automaton_v2.py",
        package_dir / "causal_state_export_v2.py",
        package_dir / "loop_duration_v2.py",
        package_dir / "loop_nulls_v2.py",
        package_dir / "loop_ledger_v2.py",
        WORK_DIR / "run_loop_event_semantics_v2.py",
        WORK_DIR / "audit_loop_event_semantics_v2.py",
    ]
    forbidden = ("broker", "order", "position", "execution", "stocker_execution")
    violations = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        if any(token in name.lower() for name in imports for token in forbidden):
            violations.append(str(source))

    assert not violations


def test_every_generated_artifact_records_the_safety_boundary() -> None:
    failures = []
    for path in PRIMARY.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            if not set(SAFETY).issubset(pd.read_csv(path, nrows=0).columns):
                failures.append(path.name)
        elif path.suffix == ".parquet":
            if not set(SAFETY).issubset(pq.read_schema(path).names):
                failures.append(path.name)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if any(payload.get(key) != value for key, value in SAFETY.items()):
                failures.append(path.name)
        elif path.suffix == ".npz":
            payload = np.load(path)
            if any(key not in payload or payload[key][0] != value for key, value in SAFETY.items()):
                failures.append(path.name)

    assert not failures


def test_frozen_historical_tree_matches_baseline_blobs() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    baseline = contract["source"]["frozen_lineage_baseline_commit"]
    prefix = "research/slrno-v2/20260714-regime-loop-handoff"
    listing = subprocess.run(
        ["git", "ls-tree", "-r", baseline, prefix],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    mismatches = []
    for line in listing:
        metadata, file_name = line.split("\t", 1)
        expected = metadata.split()[2]
        actual = subprocess.run(
            ["git", "hash-object", file_name],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if expected != actual:
            mismatches.append(file_name)

    assert len(listing) == 1_087
    assert not mismatches


def test_report_contains_all_required_sections_and_exact_next_experiment() -> None:
    report = REPORT.read_text(encoding="utf-8")

    assert all(f"## {index}." in report for index in range(1, 32))
    assert "A separately preregistered structural forecast comparing simple baselines" in report
    assert "no payoff or economic target" in report
    assert "Edge found" not in report
    assert "Live ready" not in report


def test_independent_audit_passes() -> None:
    audit = json.loads((PRIMARY / "independent_audit.json").read_text(encoding="utf-8"))

    assert audit["production_v2_imported"] is False
    assert audit["overall_pass"] is True
    assert audit["failed_checks"] == []


def test_exact_rerun_is_byte_identical() -> None:
    assert EXACT.is_dir() and any(EXACT.iterdir())
    excluded = {"artifact_manifest.json", "independent_audit.json", "decision.json"}
    primary_files = {
        path.name: path
        for path in PRIMARY.iterdir()
        if path.is_file() and path.name not in excluded
    }
    exact_files = {
        path.name: path for path in EXACT.iterdir() if path.is_file() and path.name not in excluded
    }

    assert set(primary_files) == set(exact_files)
    assert all(_sha256(primary_files[name]) == _sha256(exact_files[name]) for name in primary_files)
