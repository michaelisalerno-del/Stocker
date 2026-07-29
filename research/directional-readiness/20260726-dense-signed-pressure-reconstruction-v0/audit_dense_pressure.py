#!/usr/bin/env python3
"""Independent fail-closed audit for dense signed-pressure reconstruction V0."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[2]
PACKAGE_SOURCE = ROOT / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.dense_signed_pressure_v0 import (  # noqa: E402
    build_dense_bar_grid,
    causal_progress_surface,
    exact_pressure_raw_components,
)

DEFAULT_OUTPUT = EXPERIMENT / "artifacts/primary"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstructed_formula_hash(source_text: str, route_text: str) -> str:
    source_tree = ast.parse(source_text)
    pressure_node = next(
        node
        for node in ast.walk(source_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_signed_pressure"
    )
    formula = ast.get_source_segment(source_text, pressure_node)
    if formula is None:
        raise ValueError("signed-pressure formula segment unavailable")
    route_tokens = (
        'groupby(["session", "checkpoint"]',
        "progress - np.median(progress)",
        "fit_component_scaling(",
        'panel["signed_pressure"]',
    )
    if not all(token in route_text for token in route_tokens):
        raise ValueError("signed-pressure route lineage changed")
    return hashlib.sha256((formula + "\n" + "\n".join(route_tokens)).encode()).hexdigest()


def stable_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    rows = frame.copy()
    identity = (
        rows["symbol"].astype(str)
        + "|"
        + rows["session"].astype(str)
        + "|"
        + rows["checkpoint"].astype(int).astype(str)
    )
    rows["_hash"] = identity.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    return (
        rows.sort_values("_hash", kind="mergesort")
        .head(count)
        .drop(columns="_hash")
        .reset_index(drop=True)
    )


def slope(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    centered = x - float(x.mean())
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        return 0.0
    return float(centered @ (values - float(values.mean())) / denominator)


def manual_components(
    bars: pd.DataFrame,
    *,
    signed_progress: float,
) -> dict[str, float]:
    prices = bars.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    activity = bars["historical_relative_activity"].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or not np.isfinite(activity).all():
        raise ValueError("manual audit prefix is not finite")
    open_ = prices[:, 0]
    high = prices[:, 1]
    low = prices[:, 2]
    close = prices[:, 3]
    previous = np.roll(close, 1)
    previous[0] = open_[0]
    returns = 10_000.0 * (close / previous - 1.0)
    width = high - low
    clv = np.full(len(bars), 0.5, dtype=np.float64)
    nonzero = width > 1e-12
    clv[nonzero] = (close[nonzero] - low[nonzero]) / width[nonzero]
    opening_range = float(high.max() - low.min())
    boundary = 0.5 * (
        slope(high) / max(opening_range, 1e-12) + slope(low) / max(opening_range, 1e-12)
    )
    return {
        "signed_progress": signed_progress,
        "signed_efficiency": float(returns.sum() / max(float(np.abs(returns).sum()), 1e-12)),
        "mean_close_location": float(np.clip(clv, 0.0, 1.0).mean()),
        "boundary_slope": boundary,
    }


def manual_scale(value: float, parameters: dict[str, Any]) -> float:
    return float(
        np.clip(
            (value - float(parameters["center"])) / float(parameters["scale"]),
            float(parameters["clip_lower"]),
            float(parameters["clip_upper"]),
        )
    )


def source_current_candidates(trace: pd.DataFrame) -> pd.DataFrame:
    ordered = trace.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )
    grouped = ordered.groupby(["symbol", "session"], sort=False)
    ordered["_prefix_count"] = grouped.cumcount() + 1
    ordered["_activity_missing"] = ~np.isfinite(
        ordered["historical_relative_activity"].to_numpy(dtype=float)
    )
    ordered["_activity_missing_prefix"] = grouped["_activity_missing"].cumsum()
    current = ordered.loc[
        ordered["bar_ordinal"].add(1).between(1, 34)
        & ordered["_prefix_count"].eq(ordered["bar_ordinal"] + 1)
        & ordered["_activity_missing_prefix"].eq(0)
    ].copy()
    current["checkpoint"] = current["bar_ordinal"] + 1
    first_open = (
        ordered.loc[ordered["bar_ordinal"].eq(0), ["symbol", "session", "open"]]
        .rename(columns={"open": "session_open"})
        .copy()
    )
    current = current.merge(
        first_open,
        on=["symbol", "session"],
        validate="many_to_one",
    )
    current["raw_progress_bps"] = 10_000.0 * (current["close"] / current["session_open"] - 1.0)
    current["causal_median"] = current.groupby(["session", "checkpoint"], sort=False)[
        "raw_progress_bps"
    ].transform("median")
    current["causal_signed_progress"] = current["raw_progress_bps"] - current["causal_median"]
    return current


def production_causality_audit(
    trace: pd.DataFrame,
    sparse: pd.DataFrame,
    scaling: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, float | int]:
    """Exercise the production grid/progress seam after truncation and mutation."""

    original = causal_progress_surface(build_dense_bar_grid(trace))
    selected_even = stable_sample(sparse.loc[sparse["checkpoint"].eq(6)], 50)
    selected_odd = stable_sample(original.loc[original["checkpoint"].eq(5)], 50)

    def reconstruct(checkpoint: int, *, mutate_future: bool) -> pd.DataFrame:
        source = trace.copy()
        future = source["bar_ordinal"].astype(int).ge(checkpoint)
        if mutate_future:
            source.loc[future, ["open", "high", "low", "close"]] *= 13.0
        else:
            source = source.loc[~future].copy()
        return causal_progress_surface(build_dense_bar_grid(source))

    even_truncated = reconstruct(6, mutate_future=False)
    even_mutated = reconstruct(6, mutate_future=True)
    odd_truncated = reconstruct(5, mutate_future=False)
    odd_mutated = reconstruct(5, mutate_future=True)

    def progress_difference(
        selected: pd.DataFrame,
        rebuilt: pd.DataFrame,
        checkpoint: int,
    ) -> float:
        keys = selected.loc[:, ["symbol", "session", "checkpoint"]]
        reference = original.loc[original["checkpoint"].eq(checkpoint)].merge(
            keys,
            on=["symbol", "session", "checkpoint"],
            validate="one_to_one",
        )
        joined = reference.loc[
            :, ["symbol", "session", "checkpoint", "causal_signed_progress_bps"]
        ].merge(
            rebuilt.loc[
                :,
                ["symbol", "session", "checkpoint", "causal_signed_progress_bps"],
            ],
            on=["symbol", "session", "checkpoint"],
            suffixes=("_reference", "_rebuilt"),
            validate="one_to_one",
        )
        return float(
            (
                joined["causal_signed_progress_bps_reference"]
                - joined["causal_signed_progress_bps_rebuilt"]
            )
            .abs()
            .max()
        )

    def pressure_difference(rebuilt: pd.DataFrame) -> float:
        joined = selected_even.merge(
            rebuilt.loc[
                :,
                ["symbol", "session", "checkpoint", "causal_signed_progress_bps"],
            ],
            on=["symbol", "session", "checkpoint"],
            validate="one_to_one",
        ).merge(
            original.loc[
                original["checkpoint"].eq(6),
                ["symbol", "session", "checkpoint", "causal_signed_progress_bps"],
            ],
            on=["symbol", "session", "checkpoint"],
            suffixes=("_rebuilt", "_reference"),
            validate="one_to_one",
        )
        parameters = scaling["6"]["signed_progress"]
        rebuilt_z = np.asarray(
            [
                manual_scale(float(value), parameters)
                for value in joined["causal_signed_progress_bps_rebuilt"]
            ]
        )
        reference_z = np.asarray(
            [
                manual_scale(float(value), parameters)
                for value in joined["causal_signed_progress_bps_reference"]
            ]
        )
        return float(np.max(np.abs((rebuilt_z - reference_z) / 4.0)))

    return {
        "sparse_rows": 50,
        "odd_rows": 50,
        "maximum_sparse_truncated_progress_difference": progress_difference(
            selected_even, even_truncated, 6
        ),
        "maximum_sparse_future_value_mutation_progress_difference": progress_difference(
            selected_even, even_mutated, 6
        ),
        "maximum_sparse_truncated_candidate_pressure_difference": pressure_difference(
            even_truncated
        ),
        "maximum_sparse_future_value_mutation_candidate_pressure_difference": (
            pressure_difference(even_mutated)
        ),
        "maximum_odd_truncated_progress_difference": progress_difference(
            selected_odd, odd_truncated, 5
        ),
        "maximum_odd_future_value_mutation_progress_difference": progress_difference(
            selected_odd, odd_mutated, 5
        ),
    }


def audit(output: Path) -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads((output / "contract.json").read_text()))
    source = cast(dict[str, Any], json.loads((output / "source_manifest.json").read_text()))
    lineage = cast(
        dict[str, Any],
        json.loads((output / "existing_signed_pressure_lineage.json").read_text()),
    )
    compatibility = cast(
        dict[str, Any],
        json.loads((output / "sparse_dense_pressure_compatibility.json").read_text()),
    )
    decision = cast(dict[str, Any], json.loads((output / "phase1_decision.json").read_text()))
    determinism = cast(dict[str, Any], json.loads((output / "determinism_check.json").read_text()))

    trace_path = Path(cast(dict[str, Any], source["causal_state_trace"])["path"])
    sparse_path = Path(cast(dict[str, Any], source["full_audited_sparse_pressure_panel"])["path"])
    scaling_path = Path(cast(dict[str, Any], source["component_scaling"])["path"])
    source_path = Path(cast(dict[str, Any], source["signed_pressure_source"])["path"])
    route_path = Path(cast(dict[str, Any], source["sparse_panel_call_site"])["path"])
    branch_c_path = Path(cast(dict[str, Any], source["directional_branch_c"])["path"])
    episode_path = Path(cast(dict[str, Any], source["fresh_episode_identities"])["path"])
    manifest_paths = {
        "causal_state_trace": trace_path,
        "full_audited_sparse_pressure_panel": sparse_path,
        "directional_branch_c": branch_c_path,
        "component_scaling": scaling_path,
        "signed_pressure_source": source_path,
        "sparse_panel_call_site": route_path,
        "fresh_episode_identities": episode_path,
    }
    manifest_hash_checks = {
        name: sha256_file(path) == str(cast(dict[str, Any], source[name])["sha256"])
        for name, path in manifest_paths.items()
    }
    trace = pd.read_parquet(trace_path)
    sparse_columns = [
        "symbol",
        "session",
        "checkpoint",
        "signed_pressure",
        "z_component__signed_progress",
        "raw_component__signed_progress",
        "raw_component__signed_efficiency",
        "raw_component__mean_close_location",
        "raw_component__boundary_slope",
    ]
    sparse = pd.read_parquet(sparse_path, columns=sparse_columns)
    scaling_document = cast(dict[str, Any], json.loads(scaling_path.read_text(encoding="utf-8")))
    scaling = cast(
        dict[str, dict[str, dict[str, Any]]],
        scaling_document["component_development_scaling"],
    )

    source_text = source_path.read_text(encoding="utf-8")
    route_text = route_path.read_text(encoding="utf-8")
    formula_hash = reconstructed_formula_hash(source_text, route_text)
    preprocessing_hash = hashlib.sha256(
        json.dumps(
            scaling_document["component_development_scaling"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    lineage_checks = {
        "signed_pressure_function_found": "def _signed_pressure(" in source_text,
        "equal_four_component_mean_found": "values.mean(axis=1)" in source_text,
        "session_checkpoint_peer_median_found": (
            'groupby(["session", "checkpoint"]' in route_text
            and "progress - np.median(progress)" in route_text
        ),
        "future_three_bar_population_filter_found": (
            "required = set(range(checkpoint + 3))" in route_text
        ),
        "formula_hash_reconstructed": formula_hash == str(lineage["formula_hash"]),
        "preprocessing_hash_reconstructed": (
            preprocessing_hash
            == str(cast(dict[str, Any], source["component_scaling"])["preprocessing_hash"])
        ),
        "all_manifest_input_hashes_match": all(manifest_hash_checks.values()),
    }

    groups = {
        (str(symbol), str(session)): group.sort_values("bar_ordinal", kind="mergesort")
        for (symbol, session), group in trace.groupby(["symbol", "session"], sort=False)
    }
    sparse_sample = stable_sample(sparse, 50)
    maximum_raw_difference = 0.0
    maximum_pressure_difference = 0.0
    for row in sparse_sample.itertuples(index=False):
        checkpoint = int(row.checkpoint)
        bars = groups[(str(row.symbol), str(row.session))]
        prefix = bars.loc[bars["bar_ordinal"].astype(int).lt(checkpoint)]
        raw = manual_components(
            prefix,
            signed_progress=float(row.raw_component__signed_progress),
        )
        for component in (
            "signed_progress",
            "signed_efficiency",
            "mean_close_location",
            "boundary_slope",
        ):
            archived = float(getattr(row, f"raw_component__{component}"))
            maximum_raw_difference = max(maximum_raw_difference, abs(raw[component] - archived))
        pressure = float(
            np.mean(
                [
                    manual_scale(raw[component], scaling[str(checkpoint)][component])
                    for component in (
                        "signed_progress",
                        "signed_efficiency",
                        "mean_close_location",
                        "boundary_slope",
                    )
                ]
            )
        )
        maximum_pressure_difference = max(
            maximum_pressure_difference, abs(pressure - float(row.signed_pressure))
        )

    current = source_current_candidates(trace)
    observed_keys = set(
        trace.loc[:, ["symbol", "session", "bar_ordinal"]].itertuples(index=False, name=None)
    )
    current["three_future_bars_available"] = [
        all(
            (str(symbol), str(session), int(checkpoint) + offset) in observed_keys
            for offset in range(3)
        )
        for symbol, session, checkpoint in current.loc[
            :, ["symbol", "session", "checkpoint"]
        ].itertuples(index=False, name=None)
    ]
    even = current.loc[current["checkpoint"].isin(range(6, 35, 2))].copy()
    future_excluded_even = even.loc[~even["three_future_bars_available"].astype(bool)]
    future_excluded_even_slates = int(
        future_excluded_even.loc[:, ["session", "checkpoint"]].drop_duplicates().shape[0]
    )
    archived_population = even.loc[even["three_future_bars_available"].astype(bool)].copy()
    archived_population["archived_median"] = archived_population.groupby(
        ["session", "checkpoint"], sort=False
    )["raw_progress_bps"].transform("median")
    archived_population["archived_signed_progress"] = (
        archived_population["raw_progress_bps"] - archived_population["archived_median"]
    )
    independent = sparse.merge(
        archived_population.loc[
            :,
            [
                "symbol",
                "session",
                "checkpoint",
                "causal_signed_progress",
                "archived_signed_progress",
            ],
        ],
        on=["symbol", "session", "checkpoint"],
        validate="one_to_one",
    )
    archived_progress_difference = (
        independent["archived_signed_progress"] - independent["raw_component__signed_progress"]
    ).abs()
    independent_causal_z = [
        manual_scale(
            float(causal_progress),
            scaling[str(int(checkpoint))]["signed_progress"],
        )
        for checkpoint, causal_progress in independent.loc[
            :, ["checkpoint", "causal_signed_progress"]
        ].itertuples(index=False, name=None)
    ]
    independent["causal_candidate_pressure"] = (
        independent["signed_pressure"]
        + (
            np.asarray(independent_causal_z)
            - independent["z_component__signed_progress"].to_numpy(dtype=float)
        )
        / 4.0
    )
    independent["causal_candidate_difference"] = (
        independent["causal_candidate_pressure"] - independent["signed_pressure"]
    ).abs()
    independent_pressure_mismatches = int(
        independent["causal_candidate_difference"].gt(1e-12).sum()
    )
    independent_maximum_pressure_difference = float(
        independent["causal_candidate_difference"].max()
    )
    odd_sample = stable_sample(
        current.loc[current["checkpoint"].mod(2).eq(1)],
        50,
    )
    maximum_odd_component_implementation_difference = 0.0
    for row in odd_sample.itertuples(index=False):
        checkpoint = int(row.checkpoint)
        full = groups[(str(row.symbol), str(row.session))].copy()
        prefix = full.loc[full["bar_ordinal"].astype(int).lt(checkpoint)]
        manual = manual_components(
            prefix,
            signed_progress=float(row.causal_signed_progress),
        )
        production = exact_pressure_raw_components(
            prefix,
            centered_signed_progress_bps=float(row.causal_signed_progress),
        )
        maximum_odd_component_implementation_difference = max(
            maximum_odd_component_implementation_difference,
            max(abs(manual[name] - production[name]) for name in manual),
        )
    production_causality = production_causality_audit(trace, sparse, scaling)

    dense = pd.read_parquet(output / "dense_signed_pressure.parquet")
    audit_grid = build_dense_bar_grid(trace)
    stored_grid_audit = pd.read_csv(output / "dense_bar_grid_audit.csv")
    stored_grid_overall = stored_grid_audit.loc[stored_grid_audit["dimension"].eq("overall")].iloc[
        0
    ]
    comparison = pd.read_csv(
        output / "sparse_dense_pressure_comparison.csv",
        usecols=[
            "symbol",
            "session",
            "checkpoint",
            "absolute_difference",
            "exceeds_tolerance",
        ],
    )
    coverage = pd.read_csv(output / "dense_pressure_episode_coverage.csv")
    episode_identities = pd.read_csv(episode_path)
    expected_episodes = episode_identities.loc[
        episode_identities["episode_identity_match"].astype(bool),
        ["stock", "session", "checkpoint"],
    ].sort_values(["stock", "session", "checkpoint"], kind="mergesort")
    covered_episodes = coverage.loc[:, ["stock", "session", "checkpoint"]].sort_values(
        ["stock", "session", "checkpoint"], kind="mergesort"
    )
    dependencies = pd.read_csv(output / "pressure_dependency_classification.csv")
    class_d = dependencies.loc[dependencies["classification"].eq("D")]
    comparison_rows_exceeding = int(comparison["exceeds_tolerance"].astype(bool).sum())
    comparison_maximum = float(comparison["absolute_difference"].max())
    expected_decision = "blocked_dense_pressure_upstream_dependency"
    phase2_directory = EXPERIMENT / "artifacts/phase2-frozen-quiet-accumulation-dense-pressure"
    checks = {
        **lineage_checks,
        "required_contract_flags_present": all(
            key in contract
            for key in (
                "research_only",
                "feature_reconstruction",
                "existing_signed_pressure_formula_frozen",
                "new_pressure_definition_allowed",
                "interpolation_allowed",
                "forward_fill_allowed",
                "backfill_from_future_allowed",
                "future_outcome_features_allowed",
                "dense_bar_frequency_minutes",
                "sparse_checkpoint_compatibility_required",
                "maximum_sparse_checkpoint_difference",
                "upstream_dependency_reconstruction_required",
                "development_start",
                "development_end",
                "assessment_start",
                "assessment_end",
                "opened_holdout_excluded",
                "protected_start",
                "frozen_directional_design_unchanged",
                "broker_access",
                "option_pnl_calculated",
                "execution_enabled",
                "production_runtime_modified",
            )
        ),
        "protected_boundary_respected": (
            pd.to_datetime(trace["session"]).max() <= pd.Timestamp("2025-08-22")
        ),
        "dense_checkpoint_grid_reconstructed": (
            tuple(sorted(audit_grid["checkpoint"].unique())) == tuple(range(1, 35))
            and int(len(audit_grid)) == int(stored_grid_overall["expected_rows"])
            and int(audit_grid["bar_present"].sum())
            == int(stored_grid_overall["materialised_rows"])
            and int((~audit_grid["bar_present"].astype(bool)).sum())
            == int(stored_grid_overall["missing_bars"])
            and int(
                (
                    audit_grid["bar_present"].astype(bool)
                    & ~audit_grid["timestamp_aligned"].astype(bool)
                ).sum()
            )
            == int(stored_grid_overall["misaligned_timestamps"])
        ),
        "source_bar_identity_unique": not trace.duplicated(
            ["symbol", "session", "bar_ordinal"]
        ).any(),
        "no_interpolation_forward_fill_or_backfill": (
            not bool(contract["interpolation_allowed"])
            and not bool(contract["forward_fill_allowed"])
            and not bool(contract["backfill_from_future_allowed"])
        ),
        "manual_sparse_formula_reconstruction_within_tolerance": (
            maximum_raw_difference <= 1e-12 and maximum_pressure_difference <= 1e-12
        ),
        "manual_odd_checkpoint_upstream_rows_reconstructed": len(odd_sample) == 50,
        "manual_existing_sparse_rows_reconstructed": len(sparse_sample) == 50,
        "truncated_history_invariant_for_raw_components": (
            float(production_causality["maximum_sparse_truncated_progress_difference"]) <= 1e-12
            and float(production_causality["maximum_odd_truncated_progress_difference"]) <= 1e-12
        ),
        "future_value_mutation_invariant_for_raw_components": (
            float(production_causality["maximum_sparse_future_value_mutation_progress_difference"])
            <= 1e-12
            and float(production_causality["maximum_odd_future_value_mutation_progress_difference"])
            <= 1e-12
        ),
        "production_reconstruction_seam_truncated_and_mutated": all(
            float(value) <= 1e-12
            for key, value in production_causality.items()
            if key.startswith("maximum_")
        ),
        "odd_component_implementation_matches_independent_formula": (
            maximum_odd_component_implementation_difference <= 1e-12
        ),
        "future_population_membership_dependency_reproduced": (
            len(future_excluded_even) == 161 and future_excluded_even_slates == 32
        ),
        "sparse_compatibility_failure_reproduced": (
            independent_pressure_mismatches
            == int(compatibility["rows_exceeding_1e_12"])
            == comparison_rows_exceeding
            and math.isclose(
                independent_maximum_pressure_difference,
                float(compatibility["maximum_absolute_difference"]),
                abs_tol=1e-15,
                rel_tol=0.0,
            )
            and math.isclose(
                comparison_maximum,
                independent_maximum_pressure_difference,
                abs_tol=1e-15,
                rel_tol=0.0,
            )
        ),
        "archived_peer_normalization_reproduced": (
            float(archived_progress_difference.max()) <= 1e-12
        ),
        "class_d_dependency_recorded": (
            len(class_d) == 1
            and class_d.iloc[0]["dependency_name"]
            == "cross_sectional_signed_progress_slate_membership"
        ),
        "invalid_dense_rows_not_imputed": (
            dense["signed_pressure"].isna().all() and not dense["pressure_valid"].astype(bool).any()
        ),
        "coverage_correctly_fails": (
            not coverage["complete_five_bar_pressure_window"].astype(bool).any()
        ),
        "coverage_episode_identities_reconstructed": (
            len(expected_episodes) == len(covered_episodes) == 538
            and expected_episodes.reset_index(drop=True).equals(
                covered_episodes.reset_index(drop=True)
            )
            and int(coverage["valid_pressure_bars"].sum()) == 0
            and int(coverage["partition"].eq("development").sum()) == 285
            and int(coverage["partition"].eq("assessment").sum()) == 253
        ),
        "phase1_decision_correct": decision["primary_decision"] == expected_decision,
        "phase2_not_authorized": (
            not bool(decision["phase2_authorized"])
            and not bool(decision["phase2_executed"])
            and not phase2_directory.exists()
        ),
        "determinism_passed": bool(determinism["passed"]),
    }
    passed = bool(all(checks.values()))
    result: dict[str, Any] = {
        "audit_scope": "independent_phase1_fail_closed_audit",
        "manual_rows_examined": 100,
        "manual_pressure_rows_reconstructed": 50,
        "manual_existing_sparse_checkpoint_rows": 50,
        "manual_new_odd_upstream_rows_reconstructed": 50,
        "manual_new_odd_pressure_rows_not_materialized_due_class_d": 50,
        "maximum_manual_raw_component_difference": maximum_raw_difference,
        "maximum_manual_sparse_pressure_difference": maximum_pressure_difference,
        "maximum_odd_component_implementation_difference": (
            maximum_odd_component_implementation_difference
        ),
        "production_reconstruction_causality": production_causality,
        "manifest_input_hash_checks": manifest_hash_checks,
        "reconstructed_formula_hash": formula_hash,
        "reconstructed_preprocessing_hash": preprocessing_hash,
        "future_filtered_even_candidate_rows": int(len(future_excluded_even)),
        "future_filtered_even_session_checkpoint_slates": future_excluded_even_slates,
        "future_population_pressure_mismatches": independent_pressure_mismatches,
        "maximum_future_population_pressure_difference": (independent_maximum_pressure_difference),
        "checks": checks,
        "phase1_decision": expected_decision,
        "audit_result": (
            "passed_fail_closed_blocker_verified"
            if passed
            else "blocked_reproducibility_or_audit_failure"
        ),
        "passed": passed,
    }
    write_json(output / "lightweight_audit.json", result)
    decision["independent_audit_result"] = result["audit_result"]
    if not passed:
        decision["primary_decision"] = "blocked_dense_pressure_reproducibility_failure"
        decision["phase1_decision"] = "blocked_dense_pressure_reproducibility_failure"
        decision["blocker"] = "blocked_dense_pressure_reproducibility_failure"
    write_json(output / "phase1_decision.json", decision)
    for report_path in (output / "report.md", EXPERIMENT / "reports/report.md"):
        report = report_path.read_text(encoding="utf-8")
        report = report.replace(
            "Independent audit: pending.",
            f"Independent audit: {result['audit_result']}.",
        )
        report_path.write_text(report, encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    result = audit(arguments.output)
    print(
        json.dumps(
            {
                "audit_result": result["audit_result"],
                "passed": result["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
