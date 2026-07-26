#!/usr/bin/env python3
"""Reconstruct and audit the frozen sparse signed-pressure lineage on a dense grid.

This runner fails closed when exact sparse compatibility and current-bar causality
cannot both be satisfied.  It never creates an alternative pressure definition.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
    PRESSURE_COMPONENTS,
    SPARSE_CHECKPOINTS,
    SPARSE_TOLERANCE,
    apply_component_scale,
    build_dense_bar_grid,
    causal_progress_surface,
    classify_cross_sectional_dependency,
    compare_causal_candidate_to_sparse,
    discover_signed_pressure_lineage,
    evaluate_coverage_gate,
    exact_pressure_raw_components,
    invalid_dense_pressure_surface,
    phase1_decision,
    pressure_window_audit,
    sha256_file,
    sparse_compatibility_summary,
    stable_json_hash,
    standardized_signed_pressure,
    validate_authorized_sessions,
)

DEFAULT_OUTPUT = EXPERIMENT / "artifacts/primary"
TRACE_PATH = (
    ROOT / "research/route-competition/20260722-route-competition-hazard-quick-v0/"
    "artifacts/primary/causal_state_trace.parquet"
)
BEHAVIOURAL_SOURCE = (
    ROOT / "packages/stocker_research/src/stocker_research/behavioural_state_dimensions_v0.py"
)
ROUTE_RUNNER = (
    ROOT / "research/route-competition/20260722-route-competition-hazard-quick-v0/run_screen_v0.py"
)
SPARSE_PANEL_PATH = (
    ROOT / "research/route-competition/20260722-broad-conflict-advance-hazard-v02/"
    "artifacts/primary/dense_advance_panel.parquet"
)
SCALING_PATH = SPARSE_PANEL_PATH.parent / "model_configurations.json"
PREDECESSOR_EXPERIMENT = (
    ROOT / "research/directional-readiness/20260726-pretrigger-quiet-accumulation-direction-v0"
)
PREDECESSOR_OUTPUT = PREDECESSOR_EXPERIMENT / "artifacts/primary"
PREDECESSOR_SOURCE_MANIFEST = PREDECESSOR_OUTPUT / "source_manifest.json"
EPISODE_IDENTITIES = PREDECESSOR_OUTPUT / "episode_identity_comparison.csv"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def load_contract() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((EXPERIMENT / "contract.json").read_text()))


def load_branch_c_path() -> Path:
    manifest = cast(
        dict[str, Any], json.loads(PREDECESSOR_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    )
    return Path(cast(dict[str, Any], manifest["branch_c"])["path"])


def stable_sample(frame: pd.DataFrame, *, count: int) -> pd.DataFrame:
    rows = frame.copy()
    identities = (
        rows["symbol"].astype(str)
        + "|"
        + rows["session"].astype(str)
        + "|"
        + rows["checkpoint"].astype(int).astype(str)
    )
    rows["_sample_hash"] = identities.map(
        lambda value: __import__("hashlib").sha256(value.encode()).hexdigest()
    )
    return (
        rows.sort_values("_sample_hash", kind="mergesort")
        .head(count)
        .drop(columns="_sample_hash")
        .reset_index(drop=True)
    )


def dense_grid_audit(trace: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(trace.duplicated(["symbol", "session", "bar_ordinal"], keep=False).sum())
    rows: list[dict[str, object]] = []

    def add_row(dimension: str, key: str, group: pd.DataFrame) -> None:
        rows.append(
            {
                "dimension": dimension,
                "key": key,
                "stocks": int(group["symbol"].nunique()),
                "sessions": int(group["session"].nunique()),
                "stock_sessions": int(
                    group.loc[:, ["symbol", "session"]].drop_duplicates().shape[0]
                ),
                "expected_rows": int(len(group)),
                "materialised_rows": int(group["bar_present"].sum()),
                "missing_bars": int((~group["bar_present"].astype(bool)).sum()),
                "duplicate_bars": duplicate_count if dimension == "overall" else 0,
                "misaligned_timestamps": int(
                    (group["bar_present"] & ~group["timestamp_aligned"]).sum()
                ),
                "current_history_eligible_rows": int(group["current_history_available"].sum()),
            }
        )

    add_row("overall", "all", grid)
    for checkpoint, group in grid.groupby("checkpoint", sort=True):
        add_row("checkpoint", str(int(checkpoint)), group)
    month = grid.assign(month=grid["session"].astype(str).str[:7])
    for value, group in month.groupby("month", sort=True):
        add_row("month", str(value), group)
    return pd.DataFrame(rows)


def dependency_classification(
    progress: pd.DataFrame,
    *,
    changed_sparse_rows: int,
    changed_sparse_slates: int,
) -> pd.DataFrame:
    all_dependency = classify_cross_sectional_dependency(
        progress.assign(
            stock=progress["symbol"],
            current_history_available=True,
        )
    )
    even = progress.loc[progress["checkpoint"].isin(SPARSE_CHECKPOINTS)].copy()
    sparse_dependency = classify_cross_sectional_dependency(
        even.assign(stock=even["symbol"], current_history_available=True)
    )
    records: list[dict[str, object]] = [
        {
            "dependency_name": "completed_five_minute_ohlc",
            "source": str(TRACE_PATH),
            "classification": "A",
            "historical_availability": "every materialised completed bar",
            "required_lookback": "current regular-session prefix",
            "causality_status": "causal",
            "reconstruction_method": "exact archived OHLC rows",
            "sparse_compatibility_test": "raw component equality",
            "blocker_reason": "",
        },
        {
            "dependency_name": "historical_relative_activity",
            "source": "causal_state_trace.historical_relative_activity",
            "classification": "A",
            "historical_availability": "bar-level where prior-history baseline exists",
            "required_lookback": "historical same-stock same-bar observations",
            "causality_status": "causal_presence_gate",
            "reconstruction_method": "reuse archived activity proxy; no relabelling",
            "sparse_compatibility_test": "presence equality",
            "blocker_reason": "",
        },
        {
            "dependency_name": "signed_efficiency",
            "source": "opening_raw_components",
            "classification": "C",
            "historical_availability": "sparse even checkpoints",
            "required_lookback": "regular-session prefix through evaluated bar",
            "causality_status": "causal_at_arbitrary_completed_bar",
            "reconstruction_method": "exact sum(return_bps)/sum(abs(return_bps)) formula",
            "sparse_compatibility_test": "manual raw reconstruction",
            "blocker_reason": "",
        },
        {
            "dependency_name": "mean_close_location",
            "source": "opening_raw_components",
            "classification": "C",
            "historical_availability": "sparse even checkpoints",
            "required_lookback": "regular-session prefix through evaluated bar",
            "causality_status": "causal_at_arbitrary_completed_bar",
            "reconstruction_method": "exact archived close-location mean",
            "sparse_compatibility_test": "manual raw reconstruction",
            "blocker_reason": "",
        },
        {
            "dependency_name": "boundary_slope",
            "source": "opening_raw_components",
            "classification": "C",
            "historical_availability": "sparse even checkpoints",
            "required_lookback": "regular-session prefix through evaluated bar",
            "causality_status": "causal_at_arbitrary_completed_bar",
            "reconstruction_method": "exact normalized high/low OLS slope mean",
            "sparse_compatibility_test": "manual raw reconstruction",
            "blocker_reason": "",
        },
        {
            "dependency_name": "checkpoint_parity_guard",
            "source": "opening_raw_components",
            "classification": "C",
            "historical_availability": "existing wider vector rejects odd prefix counts",
            "required_lookback": "current prefix",
            "causality_status": "pressure_four_component_extraction_is_causal",
            "reconstruction_method": (
                "extract only the same four pressure components; do not evaluate "
                "unrelated half-window acceleration components"
            ),
            "sparse_compatibility_test": "even checkpoint raw equality",
            "blocker_reason": "",
        },
        {
            "dependency_name": "checkpoint_component_scaling",
            "source": str(SCALING_PATH),
            "classification": "B",
            "historical_availability": "frozen even-checkpoint 2024 median/IQR parameters",
            "required_lookback": "development partition only",
            "causality_status": "conditional_on_valid_cross_sectional_population",
            "reconstruction_method": "same development median/IQR, scale floor, clip [-5,5]",
            "sparse_compatibility_test": "archived parameter hash and probability equality",
            "blocker_reason": "downstream of class-D population dependency",
        },
        {
            "dependency_name": "cross_sectional_signed_progress_slate_membership",
            "source": "build_raw_decision_panel required=set(range(checkpoint + 3))",
            "classification": "D",
            "historical_availability": "sparse snapshots only",
            "required_lookback": "would require knowing availability of three later bars",
            "causality_status": "future_dependent_population_membership",
            "reconstruction_method": "none permitted",
            "sparse_compatibility_test": (
                f"{changed_sparse_rows} sparse pressure rows change across "
                f"{changed_sparse_slates} session-checkpoint slates"
            ),
            "blocker_reason": (
                "removing or retaining post-checkpoint bars changes the peer median; "
                "causal membership cannot reproduce archived pressure within 1e-12"
            ),
        },
        {
            "dependency_name": "soft_regime_or_posterior_features",
            "source": "causal_state_trace",
            "classification": "A",
            "historical_availability": "bar-level",
            "required_lookback": "not applicable",
            "causality_status": "not_an_input_to_signed_pressure",
            "reconstruction_method": "excluded from pressure calculation",
            "sparse_compatibility_test": "lineage inspection",
            "blocker_reason": "",
        },
    ]
    frame = pd.DataFrame(records)
    frame.attrs["all_dense_dependency"] = asdict(all_dependency)
    frame.attrs["sparse_even_dependency"] = asdict(sparse_dependency)
    return frame


def grouped_difference_records(comparison: pd.DataFrame, column: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    working = comparison.copy()
    if column == "month":
        working["month"] = working["session"].astype(str).str[:7]
    for value, group in working.groupby(column, sort=True):
        result.append(
            {
                column: str(value),
                "rows": int(len(group)),
                "rows_exceeding_1e_12": int(group["exceeds_tolerance"].sum()),
                "maximum_absolute_difference": float(group["absolute_difference"].max()),
                "mean_absolute_difference": float(group["absolute_difference"].mean()),
            }
        )
    return result


def manual_formula_audit(
    trace: pd.DataFrame,
    sparse: pd.DataFrame,
    progress: pd.DataFrame,
    scaling: dict[str, dict[str, dict[str, object]]],
) -> dict[str, Any]:
    trace_groups = {
        (str(symbol), str(session)): group.sort_values("bar_ordinal", kind="mergesort")
        for (symbol, session), group in trace.groupby(["symbol", "session"], sort=False)
    }
    sparse_sample = stable_sample(sparse, count=50)
    maximum_raw_difference = 0.0
    maximum_pressure_difference = 0.0
    for row in sparse_sample.itertuples(index=False):
        key = (str(row.symbol), str(row.session))
        checkpoint = int(row.checkpoint)
        bars = trace_groups[key].loc[trace_groups[key]["bar_ordinal"].astype(int).lt(checkpoint)]
        raw = exact_pressure_raw_components(
            bars,
            centered_signed_progress_bps=float(row.raw_component__signed_progress),
        )
        for component in PRESSURE_COMPONENTS:
            archived = float(getattr(row, f"raw_component__{component}"))
            maximum_raw_difference = max(maximum_raw_difference, abs(raw[component] - archived))
        standardized = {
            component: apply_component_scale(
                raw[component],
                scaling[str(checkpoint)][component],
            )
            for component in PRESSURE_COMPONENTS
        }
        pressure = standardized_signed_pressure(standardized)
        maximum_pressure_difference = max(
            maximum_pressure_difference, abs(pressure - float(row.signed_pressure))
        )

    odd = progress.loc[progress["checkpoint"].mod(2).eq(1)].copy()
    odd_sample = stable_sample(odd, count=50)
    maximum_odd_future_mutation_raw_difference = 0.0
    for row in odd_sample.itertuples(index=False):
        key = (str(row.symbol), str(row.session))
        checkpoint = int(row.checkpoint)
        full = trace_groups[key].copy()
        truncated = full.loc[full["bar_ordinal"].astype(int).lt(checkpoint)].copy()
        centered = float(row.causal_signed_progress_bps)
        original = exact_pressure_raw_components(
            truncated,
            centered_signed_progress_bps=centered,
        )
        mutated = full.copy()
        later = mutated["bar_ordinal"].astype(int).ge(checkpoint)
        mutated.loc[later, ["open", "high", "low", "close"]] = (
            mutated.loc[later, ["open", "high", "low", "close"]] * 7.0
        )
        mutated_prefix = mutated.loc[mutated["bar_ordinal"].astype(int).lt(checkpoint)]
        recalculated = exact_pressure_raw_components(
            mutated_prefix,
            centered_signed_progress_bps=centered,
        )
        maximum_odd_future_mutation_raw_difference = max(
            maximum_odd_future_mutation_raw_difference,
            max(abs(original[name] - recalculated[name]) for name in PRESSURE_COMPONENTS),
        )
    return {
        "deterministic_sample_rows": 100,
        "existing_sparse_checkpoint_rows": 50,
        "new_odd_checkpoint_rows": 50,
        "maximum_manual_raw_component_difference": maximum_raw_difference,
        "maximum_manual_sparse_pressure_difference": maximum_pressure_difference,
        "maximum_odd_future_value_mutation_raw_difference": (
            maximum_odd_future_mutation_raw_difference
        ),
    }


def reconstruction_seam_causality_audit(
    trace: pd.DataFrame,
    progress: pd.DataFrame,
    comparison: pd.DataFrame,
    component_scaling: dict[str, dict[str, dict[str, object]]],
) -> dict[str, Any]:
    affected_even = comparison.loc[
        comparison["checkpoint"].eq(6) & comparison["exceeds_tolerance"].astype(bool)
    ]
    selected_affected = stable_sample(affected_even, count=min(25, len(affected_even)))
    remaining_even = comparison.loc[
        comparison["checkpoint"].eq(6)
        & ~comparison.set_index(["symbol", "session", "checkpoint"]).index.isin(
            selected_affected.set_index(["symbol", "session", "checkpoint"]).index
        )
    ]
    selected_even = pd.concat(
        [
            selected_affected,
            stable_sample(remaining_even, count=50 - len(selected_affected)),
        ],
        ignore_index=True,
    )
    selected_odd = stable_sample(
        progress.loc[progress["checkpoint"].eq(5)],
        count=50,
    )

    def rebuilt_progress(checkpoint: int, *, mutate_future: bool) -> pd.DataFrame:
        source = trace.copy()
        future = source["bar_ordinal"].astype(int).ge(checkpoint)
        if mutate_future:
            source.loc[future, ["open", "high", "low", "close"]] = (
                source.loc[future, ["open", "high", "low", "close"]] * 11.0
            )
        else:
            source = source.loc[~future].copy()
        return causal_progress_surface(build_dense_bar_grid(source))

    even_truncated = rebuilt_progress(6, mutate_future=False)
    even_mutated = rebuilt_progress(6, mutate_future=True)
    odd_truncated = rebuilt_progress(5, mutate_future=False)
    odd_mutated = rebuilt_progress(5, mutate_future=True)

    def maximum_progress_difference(
        selected: pd.DataFrame,
        rebuilt: pd.DataFrame,
        checkpoint: int,
    ) -> float:
        original = progress.loc[progress["checkpoint"].eq(checkpoint)].merge(
            selected.loc[:, ["symbol", "session", "checkpoint"]],
            on=["symbol", "session", "checkpoint"],
            validate="one_to_one",
        )
        joined = original.loc[
            :, ["symbol", "session", "checkpoint", "causal_signed_progress_bps"]
        ].merge(
            rebuilt.loc[
                :,
                ["symbol", "session", "checkpoint", "causal_signed_progress_bps"],
            ],
            on=["symbol", "session", "checkpoint"],
            suffixes=("_original", "_rebuilt"),
            validate="one_to_one",
        )
        return float(
            (
                joined["causal_signed_progress_bps_original"]
                - joined["causal_signed_progress_bps_rebuilt"]
            )
            .abs()
            .max()
        )

    def maximum_even_pressure_difference(rebuilt: pd.DataFrame) -> float:
        joined = selected_even.loc[
            :,
            [
                "symbol",
                "session",
                "checkpoint",
                "existing_sparse_pressure",
                "z_component__signed_progress",
                "dense_pressure",
            ],
        ].merge(
            rebuilt.loc[
                :,
                ["symbol", "session", "checkpoint", "causal_signed_progress_bps"],
            ],
            on=["symbol", "session", "checkpoint"],
            validate="one_to_one",
        )
        rebuilt_z = [
            apply_component_scale(
                float(value),
                component_scaling[str(int(checkpoint))]["signed_progress"],
            )
            for checkpoint, value in joined.loc[
                :, ["checkpoint", "causal_signed_progress_bps"]
            ].itertuples(index=False, name=None)
        ]
        rebuilt_pressure = (
            joined["existing_sparse_pressure"]
            + (np.asarray(rebuilt_z) - joined["z_component__signed_progress"]) / 4.0
        )
        return float((rebuilt_pressure - joined["dense_pressure"]).abs().max())

    return {
        "sparse_checkpoint_causality_rows": 50,
        "odd_checkpoint_causality_rows": 50,
        "maximum_sparse_truncated_candidate_pressure_difference": (
            maximum_even_pressure_difference(even_truncated)
        ),
        "maximum_sparse_future_value_mutation_candidate_pressure_difference": (
            maximum_even_pressure_difference(even_mutated)
        ),
        "maximum_odd_truncated_upstream_progress_difference": (
            maximum_progress_difference(selected_odd, odd_truncated, 5)
        ),
        "maximum_odd_future_value_mutation_upstream_progress_difference": (
            maximum_progress_difference(selected_odd, odd_mutated, 5)
        ),
        "sparse_sample_archived_pressure_rows_changed_by_future_population": int(
            selected_even["exceeds_tolerance"].sum()
        ),
        "odd_pressure_reconstruction_authorized": False,
        "odd_pressure_blocker": "cross_sectional_signed_progress_slate_membership_class_D",
    }


def episode_coverage_frame(dense_pressure: pd.DataFrame) -> pd.DataFrame:
    identities = pd.read_csv(EPISODE_IDENTITIES)
    episodes = identities.loc[
        identities["episode_identity_match"].astype(bool),
        ["stock", "session", "checkpoint", "partition_direct"],
    ].rename(columns={"partition_direct": "partition"})
    return pressure_window_audit(episodes, dense_pressure)


def determinism_audit(
    trace: pd.DataFrame,
    grid: pd.DataFrame,
    progress: pd.DataFrame,
    comparison: pd.DataFrame,
    sparse: pd.DataFrame,
    component_scaling: dict[str, dict[str, dict[str, object]]],
    dense_pressure: pd.DataFrame,
    *,
    lineage_hash: str,
    source_lineage_version: str,
    preprocessing_hash: str,
) -> dict[str, Any]:
    rebuilt_grid = build_dense_bar_grid(trace)
    rebuilt_progress = causal_progress_surface(rebuilt_grid)
    grid_key = ["symbol", "session", "checkpoint"]
    dense_key = ["stock", "session", "checkpoint_index"]
    grid_mismatches = int(not grid.loc[:, grid_key].equals(rebuilt_grid.loc[:, grid_key]))
    progress_joined = progress.merge(
        rebuilt_progress,
        on=grid_key,
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    progress_difference = (
        progress_joined["causal_signed_progress_bps_first"]
        - progress_joined["causal_signed_progress_bps_second"]
    ).abs()
    rebuilt_dense = invalid_dense_pressure_surface(
        rebuilt_grid,
        formula_hash=lineage_hash,
        preprocessing_hash=preprocessing_hash,
        source_lineage_version=source_lineage_version,
    )
    rebuilt_comparison = compare_causal_candidate_to_sparse(
        sparse,
        rebuilt_progress,
        component_scaling,
    )
    pressure_validity_mismatches = int(
        (
            dense_pressure["pressure_valid"].to_numpy()
            != rebuilt_dense["pressure_valid"].to_numpy()
        ).sum()
    )
    sparse_key = ["symbol", "session", "checkpoint"]
    sparse_checkpoint_mismatches = int(
        not comparison.loc[:, sparse_key].equals(rebuilt_comparison.loc[:, sparse_key])
    )
    pressure_difference = comparison["dense_pressure"].to_numpy(dtype=float) - rebuilt_comparison[
        "dense_pressure"
    ].to_numpy(dtype=float)
    dense_comparison_columns = [
        "stock",
        "session",
        "bar_timestamp",
        "checkpoint_index",
        "signed_pressure",
        "upstream_current_history_present",
        "upstream_activity_proxy_present",
        "upstream_cross_sectional_normalization_present",
        "pressure_valid",
        "missing_reason_code",
        "partition",
        "source_lineage_version",
        "formula_hash",
        "preprocessing_hash",
    ]
    dense_surface_metadata_mismatches = int(
        not dense_pressure.loc[:, dense_comparison_columns].equals(
            rebuilt_dense.loc[:, dense_comparison_columns]
        )
    )
    return {
        "dense_bar_identity_mismatches": grid_mismatches,
        "maximum_upstream_feature_difference": float(progress_difference.max()),
        "maximum_pressure_difference": float(np.max(np.abs(pressure_difference))),
        "sparse_checkpoint_mismatches": sparse_checkpoint_mismatches,
        "pressure_validity_mismatches": pressure_validity_mismatches,
        "dense_surface_identity_mismatches": int(
            not dense_pressure.loc[:, dense_key].equals(rebuilt_dense.loc[:, dense_key])
        ),
        "dense_surface_metadata_mismatches": dense_surface_metadata_mismatches,
        "redownloaded_data": False,
        "bootstrap_or_null_samples_redrawn": False,
        "passed": bool(
            grid_mismatches == 0
            and float(progress_difference.max()) <= SPARSE_TOLERANCE
            and float(np.max(np.abs(pressure_difference))) <= SPARSE_TOLERANCE
            and sparse_checkpoint_mismatches == 0
            and pressure_validity_mismatches == 0
            and dense_surface_metadata_mismatches == 0
        ),
    }


def build_report(
    *,
    decision: dict[str, Any],
    grid_summary: pd.DataFrame,
    compatibility: dict[str, Any],
    branch_compatibility: dict[str, Any],
    coverage: pd.DataFrame,
    audit_result: str,
) -> str:
    overall = grid_summary.loc[grid_summary["dimension"].eq("overall")].iloc[0]
    development = coverage.loc[coverage["partition"].eq("development")]
    assessment = coverage.loc[coverage["partition"].eq("assessment")]
    return f"""# Dense Five-Minute Signed-Pressure Reconstruction V0

## Outcome

Phase 1 decision: `{decision["primary_decision"]}`.

The exact sparse formula lineage was found. The archived cross-sectional signed-progress
normalization first filtered each session/checkpoint stock slate using availability of three
later bars. Removing those rows, or their being absent, changes membership and therefore the
peer median; mutating later OHLC values while retaining the rows does not. A current-bar-causal
reconstruction cannot preserve the archived values within the binding `1e-12` tolerance. No
alternative pressure definition was created.

Phase 2 was not authorized and the frozen directional experiment was not rerun.

## Lineage and causality

- Formula: equal mean of development-scaled `signed_progress`, `signed_efficiency`,
  `mean_close_location`, and `boundary_slope`.
- Activity field: `historical_relative_activity`, retained as an activity proxy and not
  described as exchange-verified volume.
- Direct order flow measured: no.
- Interpolation, forward fill, and backfill: none.
- Future-dependent dependency: cross-sectional signed-progress slate membership.
- Affected full sparse rows: {compatibility["rows_exceeding_1e_12"]:,}.
- Maximum causal-versus-sparse difference: {compatibility["maximum_absolute_difference"]:.15g}.
- Directional Branch-C affected rows: {branch_compatibility["rows_exceeding_1e_12"]:,}.
- Directional Branch-C maximum difference:
  {branch_compatibility["maximum_absolute_difference"]:.15g}.

## Dense grid and support

- Stocks: {int(overall["stocks"]):,}.
- Sessions: {int(overall["sessions"]):,}.
- Stock-sessions: {int(overall["stock_sessions"]):,}.
- Expected checkpoint rows: {int(overall["expected_rows"]):,}.
- Materialised completed-bar rows: {int(overall["materialised_rows"]):,}.
- Missing bars: {int(overall["missing_bars"]):,}.
- Misaligned timestamps: {int(overall["misaligned_timestamps"]):,}.
- Valid exact dense pressure rows: 0.
- Development fresh episodes: {len(development):,}; complete five-bar windows:
  {int(development["complete_five_bar_pressure_window"].sum()):,}.
- Assessment fresh episodes: {len(assessment):,}; complete five-bar windows:
  {int(assessment["complete_five_bar_pressure_window"].sum()):,}.

Coverage by stock and month is retained in `dense_pressure_episode_coverage.csv`; no stock or
month has a valid exact five-bar pressure window because the binding upstream dependency
failed before pressure materialization.

## Audit

- Independent audit: {audit_result}.
- Determinism: {decision["determinism_result"]}.
- Phase 2 authorization: false.

This is retrospective, research-only feature-lineage work. It is not institutional
accumulation observation, direct order-flow measurement, option P&L, prospective validation,
paper readiness, live readiness, or a deployable strategy.
"""


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    trace = pd.read_parquet(TRACE_PATH)
    protected = validate_authorized_sessions(trace)
    lineage = discover_signed_pressure_lineage(BEHAVIOURAL_SOURCE, ROUTE_RUNNER)
    scaling_document = cast(dict[str, Any], json.loads(SCALING_PATH.read_text(encoding="utf-8")))
    component_scaling = cast(
        dict[str, dict[str, dict[str, object]]],
        scaling_document["component_development_scaling"],
    )
    preprocessing_hash = stable_json_hash(
        cast(dict[str, Any], scaling_document["component_development_scaling"])
    )
    branch_c_path = load_branch_c_path()
    source_manifest = {
        **{
            key: contract[key]
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
        },
        "causal_state_trace": {
            "path": str(TRACE_PATH),
            "sha256": sha256_file(TRACE_PATH),
            "role": "authorized completed five-minute OHLC and activity-proxy source",
        },
        "full_audited_sparse_pressure_panel": {
            "path": str(SPARSE_PANEL_PATH),
            "sha256": sha256_file(SPARSE_PANEL_PATH),
            "role": "full archived even-checkpoint pressure snapshots",
        },
        "directional_branch_c": {
            "path": str(branch_c_path),
            "sha256": sha256_file(branch_c_path),
            "role": "sparse subset consumed by frozen directional experiment",
        },
        "component_scaling": {
            "path": str(SCALING_PATH),
            "sha256": sha256_file(SCALING_PATH),
            "preprocessing_hash": preprocessing_hash,
        },
        "signed_pressure_source": {
            "path": str(BEHAVIOURAL_SOURCE),
            "sha256": lineage.source_sha256,
            "function": lineage.function_name,
        },
        "sparse_panel_call_site": {
            "path": str(ROUTE_RUNNER),
            "sha256": lineage.route_source_sha256,
            "function": lineage.route_call_site_function,
        },
        "fresh_episode_identities": {
            "path": str(EPISODE_IDENTITIES),
            "sha256": sha256_file(EPISODE_IDENTITIES),
        },
        "activity_proxy": {
            "field": "historical_relative_activity",
            "label": "activity proxy",
            "confirmed_exchange_volume": False,
        },
        "protected_outcomes_read": False,
        "protected_outcomes_materialized": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(
        output / "protected_boundary_audit.json",
        {
            **{
                key: contract[key]
                for key in (
                    "research_only",
                    "feature_reconstruction",
                    "protected_start",
                    "opened_holdout_excluded",
                    "broker_access",
                    "execution_enabled",
                )
            },
            **protected,
            "assessment_end_enforced": "2025-08-22",
            "opened_holdout_rows_materialized": 0,
            "protected_outcome_rows_materialized": 0,
        },
    )

    lineage_payload = {
        **asdict(lineage),
        "raw_component_source_function": "opening_raw_components",
        "raw_bar_component_source_function": "bar_component_frame",
        "call_sites": [
            "derive_behavioural_dimensions",
            "add_development_frozen_baseline_features",
        ],
        "input_columns": list(lineage.input_columns),
        "required_raw_inputs": [
            "regular-session open/high/low/close",
            "historical_relative_activity presence gate",
        ],
        "required_historical_lookback": (
            "completed regular-session prefix; sparse implementation currently "
            "requires even prefix count and three later bars for row inclusion"
        ),
        "required_behavioural_dimensions": [],
        "required_regime_or_posterior_features": [],
        "normalisation": (
            "session x checkpoint cross-sectional median for signed progress, then "
            "checkpoint-specific 2024 median/IQR scaling"
        ),
        "missing_value_rule": "reject non-finite prefix input; no imputation",
        "clipping_rule": "each standardized component clipped to [-5,+5]",
        "sign_convention": "positive is upward bar-derived pressure; negative downward",
        "output_scale": "dimensionless equal mean of four robust-standardized components",
        "calculated_directly_or_vector": "calculated as one wider behavioural vector dimension",
        "checkpoint_specific_structure": True,
        "future_filtered_population_membership_found": True,
        "lineage_unambiguous": True,
    }
    write_json(output / "existing_signed_pressure_lineage.json", lineage_payload)

    grid = build_dense_bar_grid(trace)
    grid_summary = dense_grid_audit(trace, grid)
    write_csv(output / "dense_bar_grid_audit.csv", grid_summary)
    progress = causal_progress_surface(grid)

    sparse_columns = [
        "symbol",
        "session",
        "checkpoint",
        "signed_pressure",
        *(f"raw_component__{name}" for name in PRESSURE_COMPONENTS),
        *(f"z_component__{name}" for name in PRESSURE_COMPONENTS),
    ]
    sparse = pd.read_parquet(SPARSE_PANEL_PATH, columns=sparse_columns)
    comparison = compare_causal_candidate_to_sparse(
        sparse,
        progress,
        component_scaling,
    )
    write_csv(output / "sparse_dense_pressure_comparison.csv", comparison)
    compatibility = sparse_compatibility_summary(comparison)
    changed_slates = int(
        comparison.loc[
            comparison["exceeds_tolerance"],
            ["session", "checkpoint"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    branch_c = pd.read_parquet(
        branch_c_path,
        columns=["symbol", "session", "checkpoint", "signed_pressure"],
    )
    branch_comparison = branch_c.merge(
        comparison.loc[
            :,
            [
                "symbol",
                "session",
                "checkpoint",
                "dense_pressure",
                "absolute_difference",
                "exceeds_tolerance",
            ],
        ],
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"signed_pressure": "existing_sparse_pressure"})
    branch_compatibility = sparse_compatibility_summary(branch_comparison)
    compatibility_payload = {
        **{
            key: contract[key]
            for key in (
                "research_only",
                "feature_reconstruction",
                "existing_signed_pressure_formula_frozen",
                "new_pressure_definition_allowed",
                "interpolation_allowed",
                "forward_fill_allowed",
                "backfill_from_future_allowed",
                "sparse_checkpoint_compatibility_required",
                "maximum_sparse_checkpoint_difference",
                "broker_access",
                "execution_enabled",
            )
        },
        **compatibility,
        "archived_raw_signed_progress_rows_reproduced": int(
            comparison["archived_raw_component_reproduced"].sum()
        ),
        "archived_raw_signed_progress_maximum_difference": float(
            (
                comparison["archived_future_filtered_signed_progress_bps"]
                - comparison["raw_component__signed_progress"]
            )
            .abs()
            .max()
        ),
        "changed_session_checkpoint_slates": changed_slates,
        "dense_pressure_candidate_valid": False,
        "comparison_basis": (
            "causal current-slate diagnostic; not an authorized reconstructed pressure"
        ),
        "directional_branch_c": branch_compatibility,
        "differences_by_stock": grouped_difference_records(comparison, "symbol"),
        "differences_by_checkpoint": grouped_difference_records(comparison, "checkpoint"),
        "differences_by_month": grouped_difference_records(comparison, "month"),
        "failure_cause": (
            "causal current-bar population includes rows that the archive excluded "
            "using availability of three later bars"
        ),
    }
    write_json(output / "sparse_dense_pressure_compatibility.json", compatibility_payload)

    dependencies = dependency_classification(
        progress,
        changed_sparse_rows=int(compatibility["rows_exceeding_1e_12"]),
        changed_sparse_slates=changed_slates,
    )
    write_csv(output / "pressure_dependency_classification.csv", dependencies)
    dependency_manifest = {
        "exact_formula_reused": True,
        "new_formula_created": False,
        "activity_proxy_label": "historical_relative_activity",
        "activity_proxy_is_confirmed_exchange_volume": False,
        "component_extraction": {
            "signed_progress": "same session/checkpoint peer-relative progress",
            "signed_efficiency": "same causal return-efficiency formula",
            "mean_close_location": "same causal close-location mean",
            "boundary_slope": "same causal normalized high/low OLS slope mean",
        },
        "all_dense_dependency_classification": dependencies.attrs["all_dense_dependency"],
        "sparse_even_dependency_classification": dependencies.attrs["sparse_even_dependency"],
        "binding_class_d_dependency": "cross_sectional_signed_progress_slate_membership",
        "complete_upstream_causal_chain_reconstructable": False,
    }
    write_json(output / "dense_upstream_feature_manifest.json", dependency_manifest)
    write_json(
        output / "dense_upstream_causality_audit.json",
        {
            "completed_ohlc_causal": True,
            "historical_relative_activity_causal": True,
            "signed_efficiency_causal": True,
            "mean_close_location_causal": True,
            "boundary_slope_causal": True,
            "cross_sectional_signed_progress_membership_causal": False,
            "future_filtered_current_eligible_rows_all_checkpoints": int(
                dependencies.attrs["all_dense_dependency"]["changed_rows"]
            ),
            "future_filtered_slates_all_checkpoints": int(
                dependencies.attrs["all_dense_dependency"]["changed_slates"]
            ),
            "future_filtered_current_eligible_rows_sparse_checkpoints": int(
                dependencies.attrs["sparse_even_dependency"]["changed_rows"]
            ),
            "future_filtered_slates_sparse_checkpoints": int(
                dependencies.attrs["sparse_even_dependency"]["changed_slates"]
            ),
            "interpolation_used": False,
            "forward_fill_used": False,
            "backfill_used": False,
            "passed": False,
            "blocker": "blocked_dense_pressure_upstream_dependency",
        },
    )

    formula_audit = manual_formula_audit(trace, sparse, progress, component_scaling)
    seam_causality = reconstruction_seam_causality_audit(
        trace,
        progress,
        comparison,
        component_scaling,
    )
    pressure_causality = {
        **formula_audit,
        **seam_causality,
        "maximum_truncated_stock_prefix_raw_component_difference": formula_audit[
            "maximum_manual_sparse_pressure_difference"
        ],
        "maximum_truncated_full_pressure_difference": float(
            compatibility["maximum_absolute_difference"]
        ),
        "truncated_full_pressure_reconstruction_authorized": False,
        "future_mutation_pressure_mismatches": int(compatibility["rows_exceeding_1e_12"]),
        "maximum_future_mutation_pressure_difference": float(
            compatibility["maximum_absolute_difference"]
        ),
        "future_value_mutation_raw_component_mismatches": 0,
        "future_population_mutation_detected": True,
        "passed": False,
        "blocker": "blocked_dense_pressure_upstream_dependency",
        "secondary_causality_gate_status": "failed_due_future_population_membership",
    }
    write_json(output / "dense_pressure_causality_audit.json", pressure_causality)

    dense_pressure = invalid_dense_pressure_surface(
        grid,
        formula_hash=lineage.formula_hash,
        preprocessing_hash=preprocessing_hash,
        source_lineage_version=lineage.source_sha256,
    )
    write_parquet(output / "dense_signed_pressure.parquet", dense_pressure)
    coverage = episode_coverage_frame(dense_pressure)
    write_csv(output / "dense_pressure_episode_coverage.csv", coverage)
    coverage_gate = evaluate_coverage_gate(coverage)

    grid_again = build_dense_bar_grid(trace)
    progress_again = causal_progress_surface(grid_again)
    deterministic_comparison = progress.merge(
        progress_again,
        on=["symbol", "session", "checkpoint"],
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    maximum_repeat_progress_difference = float(
        (
            deterministic_comparison["causal_signed_progress_bps_first"]
            - deterministic_comparison["causal_signed_progress_bps_second"]
        )
        .abs()
        .max()
    )
    determinism = determinism_audit(
        trace,
        grid,
        progress,
        comparison,
        sparse,
        component_scaling,
        dense_pressure,
        lineage_hash=lineage.formula_hash,
        source_lineage_version=lineage.source_sha256,
        preprocessing_hash=preprocessing_hash,
    )
    determinism["maximum_repeat_causal_progress_difference"] = maximum_repeat_progress_difference
    write_json(output / "determinism_check.json", determinism)

    primary_decision = phase1_decision(
        lineage_found=True,
        binding_dependency_class="D",
        compatibility_passed=bool(compatibility["passed"]),
        causality_passed=False,
        coverage_passed=coverage_gate.passed,
        reproducibility_passed=bool(determinism["passed"]),
    )
    decision = {
        **contract,
        "primary_decision": primary_decision,
        "phase1_decision": primary_decision,
        "blocker": primary_decision,
        "blocker_dependency": "cross_sectional_signed_progress_slate_membership",
        "blocker_detail": (
            "the archive defines the session/checkpoint peer median only after requiring "
            "three later bars; removing future bars changes pressure"
        ),
        "lineage_status": "supported",
        "upstream_dependency_status": "blocked",
        "sparse_compatibility_status": "blocked",
        "causality_status": "blocked",
        "coverage_status": "blocked",
        "phase2_authorized": False,
        "phase2_executed": False,
        "frozen_directional_settings_changed": False,
        "dense_bar_rows": int(len(grid)),
        "materialised_completed_bar_rows": int(grid["bar_present"].sum()),
        "valid_dense_pressure_rows": int(dense_pressure["pressure_valid"].sum()),
        "sparse_overlapping_rows": int(compatibility["joined_rows"]),
        "maximum_sparse_checkpoint_difference_observed": float(
            compatibility["maximum_absolute_difference"]
        ),
        "sparse_rows_exceeding_tolerance": int(compatibility["rows_exceeding_1e_12"]),
        "development_total_episodes": int(coverage["partition"].eq("development").sum()),
        "assessment_total_episodes": int(coverage["partition"].eq("assessment").sum()),
        "development_complete_window_episodes": (coverage_gate.development_complete_episodes),
        "assessment_complete_window_episodes": (coverage_gate.assessment_complete_episodes),
        "coverage_gate": asdict(coverage_gate),
        "independent_audit_result": "pending",
        "determinism_result": (
            "passed_fail_closed_reconstruction" if determinism["passed"] else "failed"
        ),
        "direct_order_flow_measured": False,
        "institutional_accumulation_observed": False,
        "option_profitability_tested": False,
        "prospective_validation": False,
        "paper_readiness": False,
        "live_readiness": False,
        "deployable_strategy": False,
    }
    write_json(output / "phase1_decision.json", decision)
    write_json(
        output / "lightweight_audit.json",
        {
            "status": "pending_independent_audit",
            "phase1_decision": primary_decision,
            "passed": False,
        },
    )
    report = build_report(
        decision=decision,
        grid_summary=grid_summary,
        compatibility=compatibility,
        branch_compatibility=branch_compatibility,
        coverage=coverage,
        audit_result="pending",
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "report.md").write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    decision = run(arguments.output)
    print(json.dumps({"phase1_decision": decision["primary_decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
