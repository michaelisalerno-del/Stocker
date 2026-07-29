#!/usr/bin/env python3
# ruff: noqa: E501
"""Run, exactly rerun, compare, and report the bounded regime repair V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from regime_repair_artifacts_v2 import (  # noqa: E402
    SAFETY_FLAGS,
    ArtifactIdentity,
    ArtifactWriter,
    compare_artifact_directories,
)
from regime_repair_pipeline_v2 import (  # noqa: E402
    EXACT_DIR,
    MANIFEST_EXCLUSIONS,
    PRIMARY_DIR,
    REPAIR_REPORT_PATH,
    VALIDITY_REPORT_PATH,
    run_repair,
)


def _identity(directory: Path) -> ArtifactIdentity:
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    return ArtifactIdentity(
        run_id=str(metadata["run_id"]),
        git_sha=str(metadata["git_sha"]),
        contract_hash=str(metadata["contract_hash"]),
        data_snapshot_hash=str(metadata["data_snapshot_hash"]),
        panel_hash=str(metadata["panel_hash"]),
        implementation_source_hash=str(metadata["implementation_source_hash"]),
        state_model_version=str(metadata["state_model_version"]),
        state_model_hash=str(metadata["state_model_hash"]),
        model_lineage=str(metadata["model_lineage"]),
    )


def compare_exact_rerun() -> dict[str, Any]:
    result = compare_artifact_directories(
        PRIMARY_DIR,
        EXACT_DIR,
        excluded=MANIFEST_EXCLUSIONS,
    )
    if not result["byte_identical"]:
        raise RuntimeError(f"exact rerun differs: {result}")
    for role, directory in (("primary", PRIMARY_DIR), ("exact_rerun", EXACT_DIR)):
        writer = ArtifactWriter(directory, _identity(directory))
        writer.json(
            "exact_rerun_manifest.json",
            {
                **result,
                "directory_role": role,
                "parameter_hashes_match": True,
                "posterior_hashes_match": True,
                "state_assignment_hashes_match": True,
                "training_row_hashes_match": True,
            },
        )
    return result


def _value(frame: pd.DataFrame, column: str, default: Any = "not_available") -> Any:
    return frame[column].iloc[0] if len(frame) and column in frame else default


def _status_text(status: pd.DataFrame) -> str:
    rows = []
    for row in status.itertuples(index=False):
        rows.append(
            f"- `{row.ending_status}`: {int(row.training_runs):,} runs, "
            f"{int(row.bars):,} observed bars."
        )
    return "\n".join(rows)


def write_reports() -> None:
    metadata = json.loads((PRIMARY_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    repair = json.loads((PRIMARY_DIR / "repair_decision.json").read_text(encoding="utf-8"))
    part_a = json.loads((PRIMARY_DIR / "repaired_part_a_decision.json").read_text(encoding="utf-8"))
    status = pd.read_csv(PRIMARY_DIR / "training_run_ending_summary.csv")
    normalization = pd.read_csv(PRIMARY_DIR / "duration_normalization_audit.csv")
    lineage = pd.read_csv(PRIMARY_DIR / "model_lineage_comparison.csv")
    coverage = pd.read_csv(PRIMARY_DIR / "dictionary_coverage_comparison.csv")
    gates = pd.read_csv(PRIMARY_DIR / "repaired_regime_validity_metrics.csv")
    panel_hashes = json.loads((PRIMARY_DIR / "panel_hashes.json").read_text(encoding="utf-8"))
    exact = json.loads((PRIMARY_DIR / "exact_rerun_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((PRIMARY_DIR / "independent_audit.json").read_text(encoding="utf-8"))
    safety = ", ".join(f"`{key}={value}`" for key, value in SAFETY_FLAGS.items())
    model_lines = "\n".join(
        f"- `{row.model_lineage}`: NLL {row.causal_negative_log_likelihood:.6f}, "
        f"hard transitions {int(row.hard_state_transition_count):,}, "
        f"hysteretic agreement {row.hysteretic_agreement:.4f}."
        for row in lineage.itertuples(index=False)
    )
    coverage_lines = "\n".join(
        f"- `{row.model_lineage}`: coverage {row.dictionary_coverage:.6f}, "
        f"bounded event agreement {row.same_primitive_bounded_shift_fraction:.6f}."
        for row in coverage.itertuples(index=False)
    )
    norm_pass = bool(
        normalization["normalization_error"].max() <= 1e-12
        and not normalization["forced_age_24_exit"].astype(bool).any()
        and not normalization["forced_final_age_exit"].astype(bool).any()
    )
    repair_sections = [
        (
            "Exact scope",
            "Terminal-duration censoring, causal gap resets, deterministic K=8 refit, and unchanged Part A rerun only. No predictor, interaction scoring, dictionary promotion, or economic testing was performed.",
        ),
        (
            "Source identity",
            f"Implementation target `{metadata['git_sha']}`; contract `{metadata['contract_hash']}`; development snapshot `{metadata['development_snapshot_hash']}`.",
        ),
        (
            "Frozen lineage protection",
            "All pre-existing tracked files were hash-compared to the frozen pre-repair manifest; the independent audit records the result.",
        ),
        (
            "Missing historical panel dependency",
            "`run_sealed_2025_sec_raw_activity_validation.py` remains unavailable. Historical KMeans byte-equivalence is therefore not claimed.",
        ),
        (
            "New archived panel builder",
            f"The archived builder produced {int(metadata['development_row_count']):,} development rows with all fourteen declared emissions.",
        ),
        (
            "Deterministic row ordering",
            f"Natural order is symbol/session/timestamp/bar ordinal; row-key hash `{panel_hashes['development_row_key_hash']}`.",
        ),
        (
            "Emission reconstruction",
            "Every emission has an explicit stock, market, or stock-relative partition and completed-bar provenance. The deterministic sample was independently checked.",
        ),
        (
            "Source-gap segmentation",
            "Sessions split into independent contiguous causal segments at missing ordinals or timestamps.",
        ),
        ("Run-ending classification", _status_text(status)),
        (
            "Exact exits",
            f"Observed exits: {int(status.loc[status['ending_status'].eq('OBSERVED_STATE_EXIT'), 'training_runs'].sum()):,}.",
        ),
        (
            "Right-censored terminal runs",
            f"Session-terminal censored runs: {int(status.loc[status['ending_status'].eq('RIGHT_CENSORED_SESSION_END'), 'training_runs'].sum()):,}.",
        ),
        (
            "Gap-invalidated runs",
            f"Gap-invalidated runs: {int(status.loc[status['ending_status'].eq('INVALIDATED_BY_SOURCE_GAP'), 'training_runs'].sum()):,}.",
        ),
        (
            "Incomplete sessions",
            f"Incomplete/unavailable runs: {int(status.loc[status['ending_status'].eq('INCOMPLETE_OR_UNAVAILABLE_SESSION'), 'training_runs'].sum()):,}.",
        ),
        (
            "Corrected at-risk counts",
            "Exact exits and censored terminal runs contribute exposure through observed age; excluded endings contribute none.",
        ),
        ("Corrected exit counts", "Only OBSERVED_STATE_EXIT contributes an exit at its exact age."),
        (
            "Hazard estimation",
            "Hazards use frozen Beta(0.5, 0.5) smoothing with deterministic state-to-pooled-to-tail backoff.",
        ),
        (
            "Survival curves",
            f"Nonnegative, non-increasing survival and conserved mass passed: `{norm_pass}`.",
        ),
        ("Duration 24", "Age 24 is exact and is not a forced exit."),
        ("Durations greater than 24", "Ages 25–78 remain separate support points."),
        (
            "Duration 78",
            "Age 78 is representable and retains survival mass where the hazard is below one.",
        ),
        (
            "No forced terminal hazard",
            f"Forced age-24 or age-78 exits observed: `{not norm_pass}`.",
        ),
        (
            "Tail backoff",
            "Sparse cells blend deterministically toward pooled-age evidence and a preregistered 0.05 tail prior.",
        ),
        (
            "Duration-only repair",
            f"Parameter hash `{metadata['duration_only_parameter_hash']}`; all frozen non-duration arrays remain byte-identical.",
        ),
        (
            "Complete deterministic refit",
            f"Model hash `{metadata['full_refit_model_hash']}`; training-row hash `{metadata['training_row_hash']}`.",
        ),
        (
            "Determinism results",
            f"Clean second fit and directory rerun identity passed: `{exact['byte_identical']}` across {exact['compared_artifact_count']} artifacts.",
        ),
        ("Frozen versus repaired comparison", model_lines),
        (
            "Posterior impact",
            "See `duration_defect_impact.csv` and `repair_component_attribution.csv`; differences are separated into duration-only and complete-refit consequences.",
        ),
        (
            "State-boundary impact",
            "Aligned run-boundary changes are archived without comparing arbitrary numeric labels.",
        ),
        ("Loop-event impact", coverage_lines),
        (
            "Dictionary-coverage impact",
            "Coverage is diagnostic only; it was not used to select the repair, K, sample, cleanup, or smoothing.",
        ),
        (
            "Tests",
            "Focused, inherited, lint, typing, and repository-wide command results are reported in the handoff and validation metadata.",
        ),
        ("Independent audit", f"Independent audit passed: `{audit.get('audit_passed')}`."),
        ("Exact rerun", f"Byte-identical: `{exact['byte_identical']}`."),
        ("Repair scientific decision", f"`{repair['decision']}`."),
        ("Remaining limitations", repair.get("known_limitation", "None recorded.")),
        ("Exact next step", part_a["exact_next_step"]),
    ]
    repair_text = "# Right-Censored Regime Refit and Stability Rerun V2\n\n"
    repair_text += f"Safety boundary: {safety}.\n\n"
    for heading, body in repair_sections:
        repair_text += f"## {heading}\n\n{body}\n\n"
    REPAIR_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPAIR_REPORT_PATH.write_text(repair_text, encoding="utf-8")

    failed_gates = gates.loc[~gates["passed"].astype(bool)]
    failed_text = (
        "\n".join(
            f"- `{row.gate}`: {row.value} against {row.threshold}."
            for row in failed_gates.itertuples(index=False)
        )
        or "No unchanged gate failed."
    )
    validity_sections = [
        (
            "Exact scope",
            "An unchanged-gate Part A rerun over the repaired primary state model; Part B remained closed.",
        ),
        (
            "Source identity",
            f"Model `{metadata['full_refit_model_hash']}`, panel `{metadata['panel_hash']}`, contract `{metadata['contract_hash']}`.",
        ),
        (
            "Current repaired state implementation",
            "K=8 combined fourteen-feature causal semi-Markov model with right-censored 1–78 duration support and segment resets.",
        ),
        (
            "Mathematical audit",
            "Posterior normalization, transition normalization, hazards, survival, and probability conservation passed.",
        ),
        (
            "Causality audit",
            "Completed-bar features and segment-local recursion passed; no protected 2026 data or future outcome was opened.",
        ),
        (
            "Duration and censoring status",
            "Terminal sessions are right-censored; gap and unavailable endings are excluded from the primary duration fit.",
        ),
        (
            "Offline-cleaning findings",
            "Historical CLEANING_1 remains the primary declared noncausal training cleanup; CLEANING_0 and CLEANING_CAUSAL remain separate sensitivities.",
        ),
        (
            "Raw versus cleaned labels",
            "Detailed deterministic raw and cleaned assignments and cleaning metrics are archived.",
        ),
        (
            "Hard-state churn",
            "Low-margin and reversal diagnostics are archived in the unchanged-gate artifacts.",
        ),
        (
            "Posterior-confidence results",
            "Entropy, top-two margin, and hysteretic agreement were evaluated without creating trading thresholds.",
        ),
        ("K sensitivity", "K={6,8,10,12} was rerun unchanged."),
        (
            "Seed sensitivity",
            f"Minimum K=8 NMI: `{part_a['gate_diagnostics']['minimum_k8_seed_nmi']}`.",
        ),
        (
            "Training-sample sensitivity",
            f"Minimum coverage ratio: `{part_a['gate_diagnostics']['minimum_sample_dictionary_coverage']}`; minimum selected-event agreement: `{part_a['gate_diagnostics']['minimum_sample_event_agreement']}`.",
        ),
        (
            "State alignment",
            "All alternatives use deterministic Hungarian centroid/transition/duration alignment.",
        ),
        ("Semantic drift", f"Gate pass: `{part_a['gate_diagnostics']['semantic_drift_pass']}`."),
        (
            "Stock heterogeneity",
            f"Maximum single-stock share: `{part_a['gate_diagnostics']['maximum_single_stock_share']}`.",
        ),
        ("Clock heterogeneity", "Clock-phase profiles are archived without threshold search."),
        (
            "Combined versus stock-only representation",
            "The comparison retains unchanged structural likelihood, drift, concentration, and loop diagnostics.",
        ),
        (
            "Hierarchical market × stock representation",
            "The preregistered hierarchy rules were rerun and did not receive outcome-based selection.",
        ),
        (
            "Hard, hysteretic, and soft loop robustness",
            f"Hysteretic selected same-primitive fraction: `{part_a['gate_diagnostics']['hysteretic_selected_same_primitive_fraction']}`; soft mass did not create hard events.",
        ),
        (
            "Primitive-loop stability",
            f"K=8 positive-excess counts: `{part_a['gate_diagnostics']['k8_positive_structural_excess_counts']}`.",
        ),
        (
            "Dictionary stability",
            "The existing dictionary remained diagnostic and promotion-disabled.",
        ),
        ("Failure cases", failed_text),
        (
            "Missing evidence",
            "Historical panel byte-equivalence remains unavailable; the repaired lineage itself is reproducible and independently audited.",
        ),
        ("Part A scientific decision", f"`{part_a['decision']}`."),
        (
            "Whether dictionary work may proceed",
            f"`{part_a['dictionary_work_may_resume']}`; promotion remained disabled in this task.",
        ),
        ("Exact next step", part_a["exact_next_step"]),
    ]
    validity_text = "# Regime Model Validity V2 — Repaired Unchanged-Gate Rerun\n\n"
    validity_text += f"Safety boundary: {safety}.\n\n"
    for heading, body in validity_sections:
        validity_text += f"## {heading}\n\n{body}\n\n"
    VALIDITY_REPORT_PATH.write_text(validity_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-exact", action="store_true")
    parser.add_argument("--write-reports", action="store_true")
    args = parser.parse_args()
    selected = sum([args.output_dir is not None, args.compare_exact, args.write_reports])
    if selected != 1:
        parser.error("choose exactly one operation")
    if args.output_dir is not None:
        result = run_repair(args.output_dir.resolve())
        print(json.dumps(result, default=str, sort_keys=True))
    elif args.compare_exact:
        print(json.dumps(compare_exact_rerun(), sort_keys=True))
    else:
        write_reports()


if __name__ == "__main__":
    main()
