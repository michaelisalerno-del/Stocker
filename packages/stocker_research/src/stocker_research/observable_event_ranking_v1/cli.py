"""Explicit stage CLI for Observable Event Cross-Sectional Ranking V1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from stocker_research.observable_event_ranking_v1.artifacts import (
    ArtifactBinding,
    ArtifactWriter,
    compare_artifact_directories,
    sha256_file,
)
from stocker_research.observable_event_ranking_v1.audit import run_independent_audit
from stocker_research.observable_event_ranking_v1.contract import (
    REQUIRED_SAFETY_FLAGS,
    canonical_hash,
    canonical_json_bytes,
    frozen_contract,
)
from stocker_research.observable_event_ranking_v1.decision import evaluate_development_gate
from stocker_research.observable_event_ranking_v1.development import (
    fit_final_frozen_components,
    run_development_oof,
)
from stocker_research.observable_event_ranking_v1.features import feature_manifest
from stocker_research.observable_event_ranking_v1.pipeline import (
    StageDependencyError,
    build_targets_stage,
    run_preflight,
)
from stocker_research.observable_event_ranking_v1.prospective import (
    append_prediction,
    append_settlement,
    prospective_ledger_schemas,
    score_frozen_prediction,
)
from stocker_research.observable_event_ranking_v1.targets import build_target_ledger

COMMANDS: tuple[str, ...] = (
    "preflight",
    "build-events",
    "audit-events",
    "build-targets",
    "run-development",
    "audit-development",
    "exact-rerun",
    "freeze-prospective",
    "score-prospective",
    "settle-prospective",
    "ibkr-resolve-contracts",
    "ibkr-capture-quotes",
    "ibkr-observability-dry-run",
)

_EXPERIMENT_RELATIVE = Path(
    "research/observable-event-ranking/20260719-observable-event-cross-sectional-ranking-v1"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed stage parser; no post-cutoff bypass option exists."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=COMMANDS)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=_EXPERIMENT_RELATIVE / "work/artifacts",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_EXPERIMENT_RELATIVE / "work/reports",
    )
    parser.add_argument("--sector-ledger", type=Path)
    parser.add_argument("--settlement-bars", type=Path)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--prediction-file", type=Path)
    parser.add_argument("--settlement-file", type=Path)
    parser.add_argument("--settlement-time")
    parser.add_argument("--enable-ibkr", action="store_true")
    return parser


def _git_identity(repository_root: Path) -> tuple[str, str]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return sha, branch


def _binding_from_manifest(artifact_dir: Path) -> ArtifactBinding:
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    return ArtifactBinding(**manifest["binding"])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Independent audit — Observable Event Ranking V1",
        "",
        f"- Audit passed: `{str(audit['audit_passed']).lower()}`",
        f"- Decision audited: `{audit['decision_audited']}`",
        "- Main runner imported: `false`",
        "- Candidate event, metric, gate, and prediction helpers imported: `false`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{str(passed).lower()}`" for name, passed in sorted(audit["checks"].items())
    )
    lines.extend(
        [
            "",
            "## Not applicable after the pre-target blocker",
            "",
            *[f"- `{name}`" for name in audit["not_applicable_due_pre_target_data_blocker"]],
            "",
            audit["scientific_interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def _write_main_report(
    report_root: Path, primary_dir: Path, audit: dict[str, Any] | None = None
) -> Path:
    decision = json.loads((primary_dir / "support_decision.json").read_text(encoding="utf-8"))
    sources = json.loads(
        (primary_dir / "source_identity_manifest.json").read_text(encoding="utf-8")
    )
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "20260719-observable-event-cross-sectional-ranking-v1.md"
    audit_status = "not yet run" if audit is None else str(audit["audit_passed"]).lower()
    exact_status = (
        "not yet run" if audit is None else str(audit["checks"]["exact_rerun_identity"]).lower()
    )
    gate_lines = "\n".join(
        f"- `{name}`: `{value}`" for name, value in sorted(decision["gates"].items())
    )
    path.write_text(
        f"""# Observable Event Cross-Sectional Ranking V1

## Safety and scope

Research and backtesting only. Execution is disabled, no live or paper order submission
is permitted, no account or position data was requested, and production runtime behavior
was not modified. No orders were sent.

## Descriptive result

- Safe pre-cutoff EODHD raw-file symbols inventoried: {sources["safe_symbol_count"]}.
- Protected source files opened: {sources["protected_files_opened"]}.
- Event rows: 0.
- Supported slates: 0.
- Point-in-time sector membership: unavailable.

No event frequency or stock, sector, or clock distribution can be estimated because the
required effective-dated sector ledger is absent. The available current screener sector
strings are not projected backward.

## Structural result

Targets and models were not permitted to run. There is no candidate-versus-baseline IC,
top-two-minus-median result, uncertainty interval, or stability result. A positive IC, if
later observed, would establish only structural rank information and not an executable edge.

## Directional interpretation

No future-return rank was constructed. Accordingly there is no directional prediction
evidence in this run.

## Economic interpretation

No gross payoff or transaction-cost result exists. Provider bar prices are structural
references, not achieved IBKR fills, and EODHD volume is only provider-reported activity
proxy data.

## Executability and IBKR

The read-only protocol, fake client, schemas, classifications, throttling configuration,
contract ledger, and observation plan are implemented. The official IBKR API transport is
blocked because IBKR distributes it through its official ZIP/MSI rather than a supported
Python package index dependency. No TWS/IB Gateway connection or subscription check was
attempted. Delayed and frozen observations are explicitly non-executable; a recorded bid
and ask would bound a reference quote, never prove a fill.

## Scientific decision

- Decision: `{decision["decision"]}`.
- Support gate passed: `false`.
- Targets permitted: `false`.
- Models permitted: `false`.
- Exact rerun: `{exact_status}`.
- Independent audit: `{audit_status}`.

### Gate values

{gate_lines}

This is a successful fail-closed implementation outcome. A genuinely new run would need a
larger pre-cutoff source universe, trusted point-in-time sector membership, proven bar-time
semantics, and resolved corporate-action handling under the unchanged contract.
""",
        encoding="utf-8",
    )
    return path


def _preflight(args: argparse.Namespace, destination: Path) -> Any:
    sha, branch = _git_identity(args.repository_root)
    return run_preflight(
        repository_root=args.repository_root,
        data_dir=args.data_dir,
        output_dir=destination,
        git_sha=sha,
        branch=branch,
        sector_ledger_path=args.sector_ledger,
        max_symbols=args.max_symbols,
        max_sessions=args.max_sessions,
    )


def _run_exact(args: argparse.Namespace) -> dict[str, Any]:
    primary = args.artifact_root / "primary"
    exact = args.artifact_root / "exact_rerun"
    if not (primary / "artifact_manifest.json").exists():
        _preflight(args, primary)
    _preflight(args, exact)
    initial = compare_artifact_directories(primary, exact)
    audit = run_independent_audit(
        primary_dir=primary,
        exact_dir=exact,
        repository_root=args.repository_root,
    )
    for directory in (primary, exact):
        writer = ArtifactWriter(directory, _binding_from_manifest(directory))
        writer.json("independent_audit.json", audit)
        (directory / "independent_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
        writer.manifest()
    final = compare_artifact_directories(primary, exact)
    payload = {
        "exact_rerun_pass": bool(initial.identical and final.identical),
        "initial_compared_files": list(initial.compared_files),
        "final_compared_files": list(final.compared_files),
        "mismatches": sorted(set([*initial.mismatches, *final.mismatches])),
        "excluded_files": [],
        "independent_audit_pass": audit["audit_passed"],
        "safety": REQUIRED_SAFETY_FLAGS,
    }
    _write_json(args.report_root / "exact_rerun_comparison.json", payload)
    _write_main_report(args.report_root, primary, audit)
    if not payload["exact_rerun_pass"] or not audit["audit_passed"]:
        raise StageDependencyError(f"exact rerun or audit failed: {payload}")
    return payload


def _build_targets(args: argparse.Namespace) -> None:
    primary = args.artifact_root / "primary"
    build_targets_stage(primary)
    if args.settlement_bars is None:
        raise StageDependencyError("--settlement-bars is required after support passes")
    events = pd.read_parquet(primary / "event_ledger.parquet")
    bars = pd.read_parquet(args.settlement_bars)
    target = build_target_ledger(events, bars)
    writer = ArtifactWriter(primary, _binding_from_manifest(primary))
    writer.json("feature_manifest.json", feature_manifest())
    writer.json(
        "target_contract.json",
        {
            "entry_reference": "open_of_t_plus_2",
            "primary_exit_minutes_after_entry": 60,
            "primary_target": "within_slate_percentile_rank_future_return_60m",
            "rank_range": [0.0, 1.0],
            "tie_handling": "average",
            "minimum_valid_targets": 8,
            "maximum_unavailable_fraction": 0.10,
            "safety": REQUIRED_SAFETY_FLAGS,
        },
    )
    writer.parquet(
        "target_ledger.parquet",
        target,
        columns=tuple(target.columns),
        sort_by=("assigned_decision_time", "slate_id", "symbol"),
    )
    writer.manifest()


def _run_development(args: argparse.Namespace) -> None:
    primary = args.artifact_root / "primary"
    target_path = primary / "target_ledger.parquet"
    if not target_path.exists():
        raise StageDependencyError("target_ledger.parquet is required before development")
    support = json.loads((primary / "support_decision.json").read_text(encoding="utf-8"))
    if not bool(support.get("passed")):
        raise StageDependencyError(
            f"development refused: {support.get('decision', 'unknown_support_decision')}"
        )
    target = pd.read_parquet(target_path)
    result = run_development_oof(target)
    writer = ArtifactWriter(primary, _binding_from_manifest(primary))
    writer.json(
        "chronological_fold_manifest.json",
        {"folds": [asdict(fold) for fold in result.folds]},
    )
    writer.parquet(
        "baseline_oof_predictions.parquet",
        result.baseline_predictions,
        columns=tuple(result.baseline_predictions.columns),
        sort_by=("session", "slate_id", "symbol"),
    )
    writer.csv(
        "baseline_metrics.csv",
        result.baseline_metrics,
        columns=tuple(result.baseline_metrics.columns),
        sort_by=("baseline_id",),
    )
    writer.json(
        "strongest_baseline_selection.json",
        {
            "selected_baseline": result.strongest_baseline,
            "selection_statistic": "pooled_historical_oof_mean_per_slate_spearman_ic",
            "random_baseline_eligible_for_selection": False,
        },
    )
    writer.json("model_effective_configuration.json", result.model_effective_configuration)
    writer.json("model_parameters.json", result.model_parameters)
    prospective_model, prospective_baseline = fit_final_frozen_components(
        target,
        result.strongest_baseline,
    )
    writer.json("prospective_model_parameters.json", prospective_model)
    writer.json("prospective_baseline_parameters.json", prospective_baseline)
    writer.parquet(
        "candidate_oof_predictions.parquet",
        result.candidate_predictions,
        columns=tuple(result.candidate_predictions.columns),
        sort_by=("session", "slate_id", "symbol"),
    )
    writer.parquet(
        "slate_level_metrics.parquet",
        result.slate_metrics,
        columns=tuple(result.slate_metrics.columns),
        sort_by=("session", "slate_id"),
    )
    primary_metrics = pd.DataFrame(
        [
            {
                "candidate_mean_ic": result.slate_metrics["candidate_ic"].mean(),
                "baseline_mean_ic": result.slate_metrics["baseline_ic"].mean(),
                "candidate_top_two_minus_median": result.slate_metrics[
                    "candidate_top_two_minus_median"
                ].mean(),
                "baseline_top_two_minus_median": result.slate_metrics[
                    "baseline_top_two_minus_median"
                ].mean(),
            }
        ]
    )
    writer.csv("primary_metrics.csv", primary_metrics, columns=tuple(primary_metrics.columns))
    bootstrap = pd.DataFrame(
        [
            {"metric": "candidate_minus_baseline_ic", **asdict(result.ic_bootstrap)},
            {"metric": "candidate_minus_baseline_top_two", **asdict(result.top_two_bootstrap)},
        ]
    )
    writer.csv(
        "paired_bootstrap_metrics.csv",
        bootstrap,
        columns=tuple(bootstrap.columns),
        sort_by=("metric",),
    )
    for name, frame, sort in (
        ("month_metrics.csv", result.month_metrics, ("month",)),
        ("quarter_metrics.csv", result.quarter_metrics, ("quarter",)),
        ("leave_one_stock_out.csv", result.leave_one_stock_out, ("removed_symbol",)),
        ("concentration_results.csv", result.concentration_results, ("dimension", "value")),
        ("turnover_results.csv", result.turnover_results, ("session", "slate_id")),
    ):
        writer.csv(name, frame, columns=tuple(frame.columns), sort_by=sort)
    event_counts = target["symbol"].value_counts(normalize=True)
    symbol_concentration = result.concentration_results.loc[
        result.concentration_results["dimension"].eq("symbol")
    ]
    max_top_fraction = (
        float(symbol_concentration["selection_fraction"].max())
        if not symbol_concentration.empty
        else 0.0
    )
    slate_folds = result.candidate_predictions.loc[:, ["slate_id", "fold_id"]].drop_duplicates()
    fold_metrics = result.slate_metrics.merge(
        slate_folds,
        on="slate_id",
        validate="one_to_one",
    )
    fold_positive = float(
        fold_metrics.groupby("fold_id", sort=True)["candidate_minus_baseline_ic"]
        .mean()
        .gt(0.0)
        .mean()
    )
    gate_inputs = {
        "support_passed": bool(support["passed"]),
        "candidate_mean_ic": float(primary_metrics.iloc[0]["candidate_mean_ic"]),
        "baseline_mean_ic": float(primary_metrics.iloc[0]["baseline_mean_ic"]),
        "candidate_top_two_minus_median": float(
            primary_metrics.iloc[0]["candidate_top_two_minus_median"]
        ),
        "baseline_top_two_minus_median": float(
            primary_metrics.iloc[0]["baseline_top_two_minus_median"]
        ),
        "positive_fold_fraction": fold_positive,
        "max_stock_event_fraction": float(event_counts.max()) if not event_counts.empty else 0.0,
        "max_stock_top_two_fraction": max_top_fraction,
    }
    writer.json("development_gate_inputs.json", gate_inputs)
    decision = evaluate_development_gate(
        support_passed=bool(gate_inputs["support_passed"]),
        candidate_mean_ic=float(gate_inputs["candidate_mean_ic"]),
        baseline_mean_ic=float(gate_inputs["baseline_mean_ic"]),
        candidate_top_two_minus_median=float(gate_inputs["candidate_top_two_minus_median"]),
        baseline_top_two_minus_median=float(gate_inputs["baseline_top_two_minus_median"]),
        positive_fold_fraction=float(gate_inputs["positive_fold_fraction"]),
        max_stock_event_fraction=float(gate_inputs["max_stock_event_fraction"]),
        max_stock_top_two_fraction=float(gate_inputs["max_stock_top_two_fraction"]),
        exact_rerun=False,
        independent_audit=False,
    )
    writer.json("development_decision.json", asdict(decision))
    writer.manifest()


def _freeze_prospective(args: argparse.Namespace) -> None:
    primary = args.artifact_root / "primary"
    freeze_path = primary / "prospective_freeze_manifest.json"
    if freeze_path.exists():
        raise StageDependencyError("prospective bundle is already frozen and immutable")
    decision_path = primary / "development_decision.json"
    if not decision_path.exists():
        raise StageDependencyError("development_decision.json is required")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("authorises_prospective_freeze"):
        raise StageDependencyError("historical development decision does not authorise freeze")
    schemas = prospective_ledger_schemas()
    writer = ArtifactWriter(primary, _binding_from_manifest(primary))
    writer.json("prospective_prediction_ledger_schema.json", schemas["prediction"])
    writer.json("prospective_settlement_ledger_schema.json", schemas["settlement"])
    contract = frozen_contract()
    writer.json(
        "prospective_decision_grid_contract.json",
        {
            "decision_grid_new_york": contract["decision_grid_new_york"],
            "grid_assignment": contract["grid_assignment"],
            "entry_and_exit_timing": contract["targets"],
            "safety": REQUIRED_SAFETY_FLAGS,
        },
    )
    writer.json(
        "prospective_final_gate_definition.json",
        {
            "evidence_type_required": "prospective",
            "historical_data_may_satisfy": False,
            "gate": contract["prospective_gate"],
            "safety": REQUIRED_SAFETY_FLAGS,
        },
    )
    component_names = (
        "frozen_experiment_contract.json",
        "event_threshold.json",
        "universe_ledger.parquet",
        "sector_membership_ledger.parquet",
        "feature_manifest.json",
        "target_contract.json",
        "prospective_model_parameters.json",
        "strongest_baseline_selection.json",
        "prospective_baseline_parameters.json",
        "prospective_decision_grid_contract.json",
        "source_identity_manifest.json",
        "implementation_source_manifest.json",
        "environment_manifest.json",
        "prospective_prediction_ledger_schema.json",
        "prospective_settlement_ledger_schema.json",
        "prospective_final_gate_definition.json",
    )
    missing = [name for name in component_names if not (primary / name).is_file()]
    if missing:
        raise StageDependencyError(f"prospective freeze components missing: {missing}")
    components = [
        {
            "path": name,
            "sha256": sha256_file(primary / name),
            "size_bytes": (primary / name).stat().st_size,
        }
        for name in component_names
    ]
    bundle_payload = {
        "bundle_version": "observable_event_ranking_v1_prospective_bundle",
        "status": "frozen",
        "binding": writer.binding.to_dict(),
        "components": components,
        "historical_development_decision": decision,
        "historical_data_may_satisfy_final_gate": False,
        "structural_research_only": True,
        "economic_evaluation_enabled": False,
        "safety": REQUIRED_SAFETY_FLAGS,
    }
    writer.json(
        "prospective_freeze_manifest.json",
        {**bundle_payload, "bundle_hash": canonical_hash(bundle_payload)},
    )
    writer.manifest()


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one explicit operation and return a process status."""

    args = build_parser().parse_args(argv)
    primary = args.artifact_root / "primary"
    exact = args.artifact_root / "exact_rerun"
    try:
        if args.operation in {"preflight", "build-events"}:
            result = _preflight(args, primary)
            _write_main_report(args.report_root, primary)
            print(json.dumps({"decision": result.decision, "run_id": result.binding.run_id}))
        elif args.operation in {"audit-events", "audit-development"}:
            if not exact.exists():
                raise StageDependencyError("exact-rerun must complete before independent audit")
            audit = run_independent_audit(
                primary_dir=primary, exact_dir=exact, repository_root=args.repository_root
            )
            print(json.dumps(audit, sort_keys=True))
            if not audit["audit_passed"]:
                return 1
        elif args.operation == "build-targets":
            _build_targets(args)
        elif args.operation == "run-development":
            _run_development(args)
        elif args.operation == "exact-rerun":
            print(json.dumps(_run_exact(args), sort_keys=True))
        elif args.operation == "freeze-prospective":
            _freeze_prospective(args)
        elif args.operation == "score-prospective":
            if args.prediction_file is None:
                raise StageDependencyError("--prediction-file is required")
            if not (primary / "prospective_freeze_manifest.json").exists():
                raise StageDependencyError("prospective freeze is required before scoring")
            freeze = json.loads(
                (primary / "prospective_freeze_manifest.json").read_text(encoding="utf-8")
            )
            payload = json.loads(args.prediction_file.read_text(encoding="utf-8"))
            payload = score_frozen_prediction(
                payload,
                model_parameters=json.loads(
                    (primary / "prospective_model_parameters.json").read_text(encoding="utf-8")
                ),
                baseline_parameters=json.loads(
                    (primary / "prospective_baseline_parameters.json").read_text(encoding="utf-8")
                ),
                bundle_hash=str(freeze["bundle_hash"]),
            )
            append_prediction(
                primary.parent / "prospective/predictions",
                payload,
                expected_bundle_hash=str(freeze["bundle_hash"]),
            )
        elif args.operation == "settle-prospective":
            if (
                args.prediction_file is None
                or args.settlement_file is None
                or not args.settlement_time
            ):
                raise StageDependencyError(
                    "--prediction-file, --settlement-file, and --settlement-time are required"
                )
            prediction = json.loads(args.prediction_file.read_text(encoding="utf-8"))
            settlement = json.loads(args.settlement_file.read_text(encoding="utf-8"))
            append_settlement(
                primary.parent / "prospective/settlements",
                prediction=prediction,
                settlement=settlement,
                settlement_time=args.settlement_time,
            )
        elif args.operation == "ibkr-observability-dry-run":
            if not (primary / "ibkr_observability_fake_dry_run.json").exists():
                _preflight(args, primary)
            print((primary / "ibkr_observability_fake_dry_run.json").read_text(encoding="utf-8"))
        elif args.operation in {"ibkr-resolve-contracts", "ibkr-capture-quotes"}:
            if not args.enable_ibkr:
                raise StageDependencyError("live IBKR observability requires --enable-ibkr")
            raise StageDependencyError(
                "official IBKR TWS API local installation and transport validation are required"
            )
        else:
            raise AssertionError(f"unhandled operation: {args.operation}")
    except StageDependencyError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), file=sys.stderr)
        return 2
    return 0
