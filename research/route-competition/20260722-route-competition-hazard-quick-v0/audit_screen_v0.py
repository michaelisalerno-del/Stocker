#!/usr/bin/env python3
"""Independently audit the Route-Competition Completion-Hazard screen artifacts."""

from __future__ import annotations

# ruff: noqa: E402 -- local package roots are resolved before research imports.
import hashlib
import importlib.util
import json
import math
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

from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine
from stocker_research.opening_trajectory_unregistered_families_v0 import (
    canonical_unregistered_path,
    pool_hidden_family,
)
from stocker_research.route_competition_hazard_v0 import (
    BASELINE_FEATURES,
    CHECKPOINTS,
    H1_FEATURES,
    ROUTE_FEATURES,
    assign_route_resolution_state,
    binary_hazard_metrics,
    choose_primary_decision,
    freeze_route_thresholds,
    permute_route_bundle,
    reconstruct_hazard_probability,
    route_increment_passes,
    session_bootstrap_multiplicities,
)

PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
V2_RUNNER = (
    REPO_ROOT
    / "research"
    / "loop-funnel"
    / "20260721-emotion-regime-coarse-loop-family-v0"
    / "run_screen_v0.py"
)
SCREEN_RUNNER = EXPERIMENT_DIR / "run_screen_v0.py"
SAFETY_FLAGS: dict[str, bool] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "registered_completion_hazard": True,
    "route_competition_features": True,
    "exact_route_identity_modelled": False,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
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
COMPONENTS = (
    "activity_effort",
    "range_effort",
    "travel_effort",
    "absolute_efficiency",
    "close_retention",
    "directional_persistence",
    "extreme_rejection",
    "absolute_progress",
    "compression",
    "signed_progress",
    "signed_efficiency",
    "mean_close_location",
    "boundary_slope",
)
LOCAL_FEATURES = (
    "prior_6_mean_range",
    "prior_6_price_travel",
    "prior_6_absolute_net_movement",
    "prior_6_activity_proxy",
    "recent_vs_earlier_range_ratio",
    "recent_vs_earlier_activity_ratio",
    "current_bar_range_vs_prior_6",
    "current_bar_activity_vs_prior_6",
    "current_bar_body_fraction",
    "current_bar_extreme_wick_fraction",
)
HIDDEN_FAMILIES = (
    "unregistered_primitive_like__5-6-5",
    "unregistered_primitive_like__2-3-2",
    "unregistered_primitive_like__2-5-2",
    "unregistered_primitive_like__4-7-4",
)
LEDGER_COLUMNS = (
    "ledger_kind",
    "symbol",
    "session",
    "bar_ordinal",
    "semantic_loop_id",
    "primitive_loop_id",
    "orientation_id",
    "motif_type",
    "repeat_depth",
    "progress_states",
    "transitions_remaining",
    "family_id",
    "available_timestamp_utc",
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str) + "\n"


def stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_ledger_view(frame: pd.DataFrame) -> pd.DataFrame:
    view = frame.loc[:, list(LEDGER_COLUMNS)].copy()
    for column in (
        "ledger_kind",
        "symbol",
        "session",
        "semantic_loop_id",
        "primitive_loop_id",
        "orientation_id",
        "motif_type",
        "family_id",
    ):
        view[column] = view[column].astype("string").fillna("<NA>")
    view["bar_ordinal"] = pd.to_numeric(view["bar_ordinal"], errors="raise").astype(int)
    for column in ("repeat_depth", "progress_states", "transitions_remaining"):
        view[column] = pd.to_numeric(view[column], errors="coerce").fillna(-1).astype(int)
    view["available_timestamp_utc"] = pd.to_datetime(
        view["available_timestamp_utc"], utc=True, errors="raise"
    ).astype(str)
    return view.sort_values(list(LEDGER_COLUMNS), kind="mergesort").reset_index(drop=True)


def reconstruct_structural_ledger(trace: pd.DataFrame, dictionary: Any) -> pd.DataFrame:
    """Independently rebuild every completion and every per-bar active prefix."""

    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    rows: list[dict[str, Any]] = []
    for (symbol, session), group in trace.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        hard = ordered["causal_hard_state"].to_numpy(dtype=int)
        changed = np.concatenate(([True], hard[1:] != hard[:-1]))
        state_events = ordered.loc[changed]
        event_trace = engine.scan_state_events(
            state_events["causal_hard_state"].astype(int).tolist(),
            bar_ordinals=(state_events["bar_ordinal"].astype(int) + 1).tolist(),
            event_timestamps=[
                value.to_pydatetime()
                for value in pd.to_datetime(state_events["bar_start_timestamp"], utc=True)
            ],
            available_timestamps=[
                value.to_pydatetime()
                for value in pd.to_datetime(state_events["bar_complete_timestamp"], utc=True)
            ],
        )
        for event in event_trace.registered_completions:
            rows.append(
                {
                    "ledger_kind": "registered_completion",
                    "symbol": str(symbol),
                    "session": str(session),
                    "bar_ordinal": int(event.completion_bar_ordinal),
                    "semantic_loop_id": str(event.semantic_loop_id),
                    "primitive_loop_id": str(event.primitive_loop_id),
                    "orientation_id": str(event.orientation_id),
                    "motif_type": str(event.motif_type),
                    "repeat_depth": int(event.repeat_depth),
                    "progress_states": np.nan,
                    "transitions_remaining": np.nan,
                    "family_id": None,
                    "available_timestamp_utc": pd.Timestamp(event.completion_available_timestamp),
                }
            )
        for event in event_trace.unregistered_completions:
            canonical = canonical_unregistered_path(event.full_path)
            available = event_trace.state_events[event.completion_event_index].available_timestamp
            rows.append(
                {
                    "ledger_kind": "hidden_completion",
                    "symbol": str(symbol),
                    "session": str(session),
                    "bar_ordinal": int(event.completion_bar_ordinal),
                    "semantic_loop_id": None,
                    "primitive_loop_id": None,
                    "orientation_id": str(canonical.orientation_id),
                    "motif_type": None,
                    "repeat_depth": np.nan,
                    "progress_states": np.nan,
                    "transitions_remaining": np.nan,
                    "family_id": pool_hidden_family(canonical.family_id, HIDDEN_FAMILIES),
                    "available_timestamp_utc": pd.Timestamp(available),
                }
            )
        event_indices = np.cumsum(changed).astype(int) - 1
        for position, bar in enumerate(ordered.itertuples(index=False)):
            completed_count = int(bar.bar_ordinal) + 1
            for prefix in event_trace.prefixes_after_event[int(event_indices[position])]:
                rows.append(
                    {
                        "ledger_kind": "active_prefix",
                        "symbol": str(symbol),
                        "session": str(session),
                        "bar_ordinal": completed_count,
                        "semantic_loop_id": str(prefix.semantic_loop_id),
                        "primitive_loop_id": str(prefix.primitive_loop_id),
                        "orientation_id": str(prefix.orientation_id),
                        "motif_type": str(prefix.motif_type),
                        "repeat_depth": int(prefix.repeat_depth),
                        "progress_states": int(prefix.progress_states),
                        "transitions_remaining": int(prefix.transitions_remaining),
                        "family_id": None,
                        "available_timestamp_utc": pd.Timestamp(bar.bar_complete_timestamp),
                    }
                )
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS).drop_duplicates()
    return ledger.sort_values(
        ["symbol", "session", "bar_ordinal", "ledger_kind", "semantic_loop_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _orientation_signature(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    if not text:
        return None
    if "__o_" not in text:
        return text
    suffix = text.split("__o_", maxsplit=1)[1]
    anchor = suffix.split("-", maxsplit=1)[0]
    return f"anchor_state_{anchor}" if anchor else None


def _snapshot(prefixes: pd.DataFrame, checkpoint: int) -> pd.DataFrame:
    current = prefixes.loc[prefixes["bar_ordinal"].eq(checkpoint)].copy()
    if current.empty:
        current["depth"] = pd.Series(dtype=float)
        return current
    denominator = (
        current["progress_states"].astype(float)
        + current["transitions_remaining"].astype(float)
        - 1.0
    )
    current["depth"] = ((current["progress_states"].astype(float) - 1.0) / denominator).clip(
        0.0, 1.0
    )
    return (
        current.sort_values("depth", ascending=False, kind="mergesort")
        .drop_duplicates(["semantic_loop_id", "orientation_id"], keep="first")
        .reset_index(drop=True)
    )


def manual_route_features(
    prefixes: pd.DataFrame, registered: pd.DataFrame, checkpoint: int
) -> dict[str, float]:
    current = _snapshot(prefixes, checkpoint)
    previous_one = _snapshot(prefixes, checkpoint - 1)
    previous_three = _snapshot(prefixes, checkpoint - 3)
    depths = sorted(current["depth"].astype(float), reverse=True)
    top = depths[0] if depths else 0.0
    second = depths[1] if len(depths) > 1 else 0.0
    family_counts = current["motif_type"].astype(str).value_counts()
    if len(family_counts) < 2:
        entropy = 0.0
    else:
        probabilities = family_counts / family_counts.sum()
        entropy = -sum(float(value) * math.log(float(value)) for value in probabilities)
    orientations = current["orientation_id"].map(_orientation_signature).dropna()
    disagreement = (
        0.0
        if len(orientations) < 2 or len(orientations) != len(current)
        else 1.0 - float(orientations.value_counts().max()) / len(orientations)
    )
    key = lambda frame: set(  # noqa: E731 - compact independent identity projection.
        zip(
            frame["semantic_loop_id"].astype(str),
            frame["orientation_id"].astype(str),
            strict=True,
        )
    )
    current_keys = key(current)
    previous_keys = key(previous_one)
    prior = registered.loc[registered["bar_ordinal"].lt(checkpoint)]
    recent_ids = set(
        prior.loc[prior["bar_ordinal"].ge(checkpoint - 6), "semantic_loop_id"].astype(str)
    )
    older_ids = set(
        prior.loc[
            prior["bar_ordinal"].ge(checkpoint - 12) & prior["bar_ordinal"].lt(checkpoint - 6),
            "semantic_loop_id",
        ].astype(str)
    )
    weighted = []
    for row in current.itertuples(index=False):
        identity = str(row.semantic_loop_id)
        multiplier = 1.0 + float(identity in recent_ids) + 0.5 * float(identity in older_ids)
        weighted.append(float(row.depth) * multiplier)
    return {
        "active_prefix_count": float(len(current)),
        "active_prefix_family_count": float(current["motif_type"].nunique()),
        "top_prefix_depth_fraction": top,
        "second_prefix_depth_fraction": second,
        "top_minus_second_prefix_depth": top - second,
        "prefix_family_entropy": entropy,
        "orientation_disagreement_fraction": disagreement,
        "new_prefixes_last_1_bar": float(len(current_keys - previous_keys)),
        "invalidated_prefixes_last_1_bar": float(len(previous_keys - current_keys)),
        "active_prefix_count_change_last_1_bar": float(len(current) - len(previous_one)),
        "active_prefix_count_change_last_3_bars": float(len(current) - len(previous_three)),
        "top_prefix_depth_change_last_1_bar": top
        - (float(previous_one["depth"].max()) if not previous_one.empty else 0.0),
        "top_prefix_depth_change_last_3_bars": top
        - (float(previous_three["depth"].max()) if not previous_three.empty else 0.0),
        "matching_recent_loop_prefix_count": float(
            current["semantic_loop_id"].astype(str).isin(recent_ids).sum()
        ),
        "recent_loop_memory_weighted_top_depth": max(weighted, default=0.0),
    }


def _top_precision(frame: pd.DataFrame, column: str, threshold: float) -> float:
    selected = frame.loc[frame[column].ge(threshold)]
    return float(
        np.average(
            selected["registered_completion_next_3_bars"],
            weights=selected["row_weight"],
        )
    )


def _increments(
    frame: pd.DataFrame, boundaries: Mapping[str, Mapping[str, float]]
) -> dict[str, float]:
    metrics = {
        model: binary_hazard_metrics(
            frame["registered_completion_next_3_bars"],
            frame[f"{model}_probability"],
            frame["row_weight"],
        )
        for model in ("H0", "H1")
    }
    return {
        "log_loss_improvement": float(metrics["H0"]["log_loss"]) - float(metrics["H1"]["log_loss"]),
        "brier_improvement": float(metrics["H0"]["brier_score"])
        - float(metrics["H1"]["brier_score"]),
        "auc_improvement": float(metrics["H1"]["auc"]) - float(metrics["H0"]["auc"]),
        "average_precision_improvement": float(metrics["H1"]["average_precision"])
        - float(metrics["H0"]["average_precision"]),
        "top_decile_precision_improvement": _top_precision(
            frame,
            "H1_probability",
            float(boundaries["H1"]["top_decile"]),
        )
        - _top_precision(
            frame,
            "H0_probability",
            float(boundaries["H0"]["top_decile"]),
        ),
    }


def run_audit() -> dict[str, Any]:
    contract = read_json(PRIMARY / "contract.json")
    source = read_json(PRIMARY / "source_manifest.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    baseline_manifest = read_json(PRIMARY / "baseline_feature_manifest.json")
    route_manifest = read_json(PRIMARY / "route_competition_feature_manifest.json")
    configurations = read_json(PRIMARY / "model_configurations.json")
    coefficients = read_json(PRIMARY / "model_coefficients.json")
    decision = read_json(PRIMARY / "decision.json")
    determinism = read_json(PRIMARY / "determinism_check.json")
    panel = pd.read_parquet(PRIMARY / "decision_panel.parquet")
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    stored_ledger = pd.read_parquet(PRIMARY / "route_competition_ledger.parquet")
    trace_manifest = cast(Mapping[str, Any], source["causal_state_trace"])
    trace_path = REPO_ROOT / str(trace_manifest["logical_path"])
    trace = pd.read_parquet(trace_path)
    v2_runner = load_module(V2_RUNNER, "route_hazard_audit_v2_runner")
    dictionary, dictionary_manifest = v2_runner.load_loop_dictionary()
    ledger = reconstruct_structural_ledger(trace, dictionary)
    target_registered = ledger.loc[ledger["ledger_kind"].eq("registered_completion")].copy()
    target_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in target_registered.groupby(["symbol", "session"], sort=False)
    }
    empty_target = target_registered.iloc[:0]
    pooled = pd.read_csv(PRIMARY / "pooled_metrics.csv")
    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    null_metrics = pd.read_csv(PRIMARY / "route_null_metrics.csv")

    checks: dict[str, Any] = {}
    checks["safety_flags"] = all(
        contract.get(key) == expected
        and decision.get(key) == expected
        and source.get(key) == expected
        for key, expected in SAFETY_FLAGS.items()
    )
    checks["dates_and_protected_boundary"] = bool(
        panel["session"].astype(str).between("2024-01-01", "2025-08-22").all()
        and trace["session"].astype(str).between("2024-01-01", "2025-08-22").all()
        and int(protected["protected_rows_materialised"]) == 0
        and bool(protected["passed"])
    )
    checks["frozen_cohort"] = set(panel["symbol"].astype(str)) == FROZEN_COHORT
    stored_ledger_view = _canonical_ledger_view(stored_ledger)
    reconstructed_ledger_view = _canonical_ledger_view(ledger)
    stored_ledger_hash = stable_frame_hash(stored_ledger_view, LEDGER_COLUMNS)
    reconstructed_ledger_hash = stable_frame_hash(reconstructed_ledger_view, LEDGER_COLUMNS)
    checks["causal_state_trace_source"] = bool(
        trace_path.is_file()
        and sha256_file(trace_path) == str(trace_manifest["sha256"])
        and len(trace) == int(trace_manifest["rows"])
        and list(trace.columns) == list(trace_manifest["columns"])
        and int(trace["bar_ordinal"].max()) == 37
        and set(trace["symbol"].astype(str)) == FROZEN_COHORT
        and not trace.duplicated(["symbol", "session", "bar_ordinal"]).any()
    )
    checks["frozen_loop_dictionary"] = canonical_json(dictionary_manifest) == canonical_json(
        source["loop_dictionary_manifest"]
    )
    checks["independent_full_structural_ledger_reconstruction"] = bool(
        len(stored_ledger_view) == len(reconstructed_ledger_view)
        and stored_ledger_hash == reconstructed_ledger_hash
    )
    checks["eight_checkpoints"] = tuple(sorted(panel["checkpoint"].unique())) == CHECKPOINTS
    checks["three_bar_target_and_causality"] = True
    for row in panel.itertuples(index=False):
        events = target_groups.get((str(row.symbol), str(row.session)), empty_target)
        counts = events["bar_ordinal"].astype(int).tolist()
        expected = int(any(row.checkpoint < value <= row.checkpoint + 3 for value in counts))
        expected_one = int(row.checkpoint + 1 in counts)
        stored_counts = [
            int(value) for value in json.loads(row.future_registered_completion_counts)
        ]
        if expected != int(row.registered_completion_next_3_bars):
            checks["three_bar_target_and_causality"] = False
            break
        if expected_one != int(row.registered_completion_next_1_bar):
            checks["three_bar_target_and_causality"] = False
            break
        if stored_counts != sorted(
            value for value in counts if row.checkpoint < value <= row.checkpoint + 3
        ):
            checks["three_bar_target_and_causality"] = False
            break
        if pd.Timestamp(row.feature_available_timestamp_utc) <= pd.Timestamp(
            row.checkpoint_timestamp_utc
        ):
            checks["three_bar_target_and_causality"] = False
            break
        future_events = events.loc[
            events["bar_ordinal"].gt(row.checkpoint) & events["bar_ordinal"].le(row.checkpoint + 3)
        ]
        if not future_events.empty and bool(
            pd.to_datetime(future_events["available_timestamp_utc"], utc=True)
            .le(pd.Timestamp(row.feature_available_timestamp_utc))
            .any()
        ):
            checks["three_bar_target_and_causality"] = False
            break

    checks["baseline_feature_surface"] = bool(
        tuple(baseline_manifest["features"]) == BASELINE_FEATURES
        and np.isfinite(panel.loc[:, list(BASELINE_FEATURES)].to_numpy(float)).all()
        and not any("frustration" in value or "exhaustion" in value for value in BASELINE_FEATURES)
    )
    checks["route_feature_surface"] = bool(
        tuple(route_manifest["features"]) == ROUTE_FEATURES
        and tuple(H1_FEATURES) == (*BASELINE_FEATURES, *ROUTE_FEATURES)
        and not any("future" in value or "target_identity" in value for value in H1_FEATURES)
    )

    current_trace = trace.copy()
    current_trace["checkpoint"] = current_trace["bar_ordinal"].astype(int) + 1
    current_trace = current_trace.loc[current_trace["checkpoint"].isin(CHECKPOINTS)]
    regime_columns = (
        "posterior_entropy",
        "transition_probability",
        "persistence_probability",
        "expected_state_age",
        "top_state_probability",
        "top_second_margin",
    )
    regime_join = panel.loc[
        :,
        [
            "row_id",
            "symbol",
            "session",
            "checkpoint",
            "checkpoint_timestamp_utc",
            "feature_available_timestamp_utc",
            *regime_columns,
        ],
    ].merge(
        current_trace.loc[
            :,
            [
                "symbol",
                "session",
                "checkpoint",
                "bar_start_timestamp",
                "bar_complete_timestamp",
                *(f"state_p_{state}" for state in range(8)),
                *regime_columns,
            ],
        ],
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
        suffixes=("_panel", "_trace"),
    )
    posterior = regime_join.loc[:, [f"state_p_{state}" for state in range(8)]].to_numpy(dtype=float)
    ordered_posterior = np.sort(posterior, axis=1)
    maximum_regime_difference = max(
        max(
            float(
                np.max(
                    np.abs(
                        regime_join[f"{column}_panel"].to_numpy(float)
                        - regime_join[f"{column}_trace"].to_numpy(float)
                    )
                )
            )
            for column in regime_columns
        ),
        float(
            np.max(
                np.abs(
                    ordered_posterior[:, -1]
                    - regime_join["top_state_probability_panel"].to_numpy(float)
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    ordered_posterior[:, -1]
                    - ordered_posterior[:, -2]
                    - regime_join["top_second_margin_panel"].to_numpy(float)
                )
            )
        ),
    )
    checks["current_regime_and_completed_bar_causality"] = bool(
        len(regime_join) == len(panel)
        and np.isfinite(posterior).all()
        and np.allclose(posterior.sum(axis=1), 1.0, atol=1e-12)
        and maximum_regime_difference <= 1e-12
        and (
            pd.to_datetime(regime_join["checkpoint_timestamp_utc"], utc=True)
            == pd.to_datetime(regime_join["bar_start_timestamp"], utc=True)
        ).all()
        and (
            pd.to_datetime(regime_join["feature_available_timestamp_utc"], utc=True)
            == pd.to_datetime(regime_join["bar_complete_timestamp"], utc=True)
        ).all()
    )

    maximum_component_difference = 0.0
    scaling = baseline_manifest["behavioural_component_scaling"]
    for checkpoint in CHECKPOINTS:
        subset = panel.loc[panel["checkpoint"].eq(checkpoint)]
        for component in COMPONENTS:
            frozen = scaling[str(checkpoint)][component]
            regenerated = np.clip(
                (subset[f"raw_component__{component}"].to_numpy(float) - float(frozen["center"]))
                / float(frozen["scale"]),
                -5.0,
                5.0,
            )
            maximum_component_difference = max(
                maximum_component_difference,
                float(
                    np.max(
                        np.abs(regenerated - subset[f"z_component__{component}"].to_numpy(float))
                    )
                ),
            )
    regenerated_dimensions = pd.DataFrame(index=panel.index)
    regenerated_dimensions["arousal"] = panel[
        [
            "z_component__activity_effort",
            "z_component__range_effort",
            "z_component__travel_effort",
        ]
    ].mean(axis=1)
    regenerated_dimensions["conviction"] = panel[
        [
            "z_component__absolute_efficiency",
            "z_component__close_retention",
            "z_component__directional_persistence",
        ]
    ].mean(axis=1)
    regenerated_dimensions["tension"] = (
        panel[
            [
                "z_component__activity_effort",
                "z_component__compression",
                "z_component__extreme_rejection",
            ]
        ].mean(axis=1)
        - panel["z_component__absolute_progress"]
    )
    regenerated_dimensions["signed_pressure"] = panel[
        [
            "z_component__signed_progress",
            "z_component__signed_efficiency",
            "z_component__mean_close_location",
            "z_component__boundary_slope",
        ]
    ].mean(axis=1)
    maximum_dimension_difference = float(
        np.max(
            np.abs(
                regenerated_dimensions.to_numpy(float)
                - panel[regenerated_dimensions.columns].to_numpy(float)
            )
        )
    )
    local_scaling = baseline_manifest["stock_and_clock_local_scaling"]
    maximum_local_difference = 0.0
    sample = panel.iloc[np.linspace(0, len(panel) - 1, 1000, dtype=int)]
    for row in sample.itertuples(index=False):
        frozen = local_scaling[f"{row.symbol}|{row.checkpoint}"]
        for feature in LOCAL_FEATURES:
            expected = np.clip(
                (float(getattr(row, f"raw_local__{feature}")) - float(frozen[feature]["center"]))
                / float(frozen[feature]["scale"]),
                -5.0,
                5.0,
            )
            maximum_local_difference = max(
                maximum_local_difference, abs(float(expected) - float(getattr(row, feature)))
            )
    checks["behavioural_and_local_reconstruction"] = bool(
        maximum_component_difference <= 1e-12
        and maximum_dimension_difference <= 1e-12
        and maximum_local_difference <= 1e-12
    )

    prefixes = ledger.loc[ledger["ledger_kind"].eq("active_prefix")]
    registered = ledger.loc[ledger["ledger_kind"].eq("registered_completion")]
    hidden = ledger.loc[ledger["ledger_kind"].eq("hidden_completion")]
    prefix_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in prefixes.groupby(["symbol", "session"], sort=False)
    }
    registered_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in registered.groupby(["symbol", "session"], sort=False)
    }
    hidden_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in hidden.groupby(["symbol", "session"], sort=False)
    }
    empty_prefix = prefixes.iloc[:0]
    empty_registered = registered.iloc[:0]
    empty_hidden = hidden.iloc[:0]
    maximum_route_difference = 0.0
    maximum_structural_memory_difference = 0.0
    causal_prefix_timestamps = True
    route_sample = panel.iloc[np.linspace(0, len(panel) - 1, 240, dtype=int)]
    for row in route_sample.itertuples(index=False):
        key = (str(row.symbol), str(row.session))
        session_prefix = prefix_groups.get(key, empty_prefix)
        session_registered = registered_groups.get(key, empty_registered)
        session_hidden = hidden_groups.get(key, empty_hidden)
        manual = manual_route_features(session_prefix, session_registered, int(row.checkpoint))
        for feature in ROUTE_FEATURES:
            maximum_route_difference = max(
                maximum_route_difference,
                abs(float(manual[feature]) - float(getattr(row, feature))),
            )
        prior_six = session_registered.loc[
            session_registered["bar_ordinal"].ge(row.checkpoint - 6)
            & session_registered["bar_ordinal"].lt(row.checkpoint)
        ]
        prior_twelve = session_registered.loc[
            session_registered["bar_ordinal"].ge(row.checkpoint - 12)
            & session_registered["bar_ordinal"].lt(row.checkpoint)
        ]
        current_ids = set(
            session_prefix.loc[
                session_prefix["bar_ordinal"].eq(row.checkpoint), "semantic_loop_id"
            ].astype(str)
        )
        hidden_six = session_hidden.loc[
            session_hidden["bar_ordinal"].ge(row.checkpoint - 6)
            & session_hidden["bar_ordinal"].lt(row.checkpoint)
        ]
        earlier = session_registered.loc[session_registered["bar_ordinal"].lt(row.checkpoint)]
        structural_expected = {
            "any_registered_completion_prior_6": float(not prior_six.empty),
            "any_registered_completion_prior_12": float(not prior_twelve.empty),
            "same_identity_active_prefix_with_prior_completion": float(
                bool(current_ids.intersection(set(prior_twelve["semantic_loop_id"].astype(str))))
            ),
            "any_hidden_event_prior_6": float(not hidden_six.empty),
            "hidden_2_3_2_prior_6": float(
                hidden_six["family_id"].astype(str).eq("unregistered_primitive_like__2-3-2").any()
            ),
            "bars_since_latest_registered_completion": (
                float(row.checkpoint + 1)
                if earlier.empty
                else float(row.checkpoint - int(earlier["bar_ordinal"].max()))
            ),
        }
        for feature, value in structural_expected.items():
            maximum_structural_memory_difference = max(
                maximum_structural_memory_difference,
                abs(value - float(getattr(row, feature))),
            )
        visible = session_prefix.loc[session_prefix["bar_ordinal"].eq(row.checkpoint)]
        if not visible.empty and bool(
            pd.to_datetime(visible["available_timestamp_utc"], utc=True)
            .gt(pd.Timestamp(row.feature_available_timestamp_utc))
            .any()
        ):
            causal_prefix_timestamps = False
    checks["route_feature_reconstruction"] = maximum_route_difference <= 1e-12
    checks["recent_structural_memory_reconstruction"] = (
        maximum_structural_memory_difference <= 1e-12
    )
    checks["no_future_prefix_transition"] = causal_prefix_timestamps

    thresholds = freeze_route_thresholds(panel.loc[panel["period"].eq("development")])
    regenerated_labels = assign_route_resolution_state(panel, thresholds)
    checks["development_only_bins_and_labels"] = bool(
        all(
            np.allclose(
                thresholds[key],
                route_manifest["development_frozen_bins"]["route_quartiles"][key],
                atol=0.0,
                rtol=0.0,
            )
            for key in thresholds
        )
        and (regenerated_labels.astype(str) == panel["route_resolution_state"].astype(str)).all()
    )
    slate_weight = panel.groupby(["period", "session", "checkpoint"])["row_weight"].sum()
    checks["slate_weighting"] = bool(np.allclose(slate_weight, 1.0, atol=1e-12))

    manual_rows = panel.iloc[np.linspace(0, len(panel) - 1, 200, dtype=int)]
    maximum_probability_difference = 0.0
    for model in ("H0", "H1"):
        specification = coefficients["primary_models"][model]
        regenerated = reconstruct_hazard_probability(manual_rows, specification)
        maximum_probability_difference = max(
            maximum_probability_difference,
            float(
                np.max(np.abs(regenerated - manual_rows[f"{model}_probability"].to_numpy(float)))
            ),
        )
    checks["manual_probability_reconstruction_200_rows"] = maximum_probability_difference <= 1e-12
    checks["model_coefficients_and_convergence"] = all(
        int(coefficients["primary_models"][model]["n_iter"]) < 300
        and tuple(coefficients["primary_models"][model]["feature_names"])
        == (BASELINE_FEATURES if model == "H0" else H1_FEATURES)
        for model in ("H0", "H1")
    )

    maximum_metric_difference = 0.0
    for model in ("H0", "H1"):
        regenerated = binary_hazard_metrics(
            assessment["registered_completion_next_3_bars"],
            assessment[f"{model}_probability"],
            assessment["row_weight"],
        )
        stored = pooled.loc[pooled["model"].eq(model)].iloc[0]
        for metric in ("log_loss", "brier_score", "auc", "average_precision"):
            maximum_metric_difference = max(
                maximum_metric_difference,
                abs(float(regenerated[metric]) - float(stored[metric])),
            )
    checks["pooled_metrics"] = maximum_metric_difference <= 1e-12

    boundaries = configurations["probability_quantile_boundaries"]
    stored_draws = bootstrap.loc[bootstrap["record_type"].eq("draw")].set_index("draw")
    maximum_bootstrap_difference = 0.0
    multiplicities = session_bootstrap_multiplicities(
        assessment["session"], draws=15, seed=20260722
    )
    for draw, multiplicity in enumerate(multiplicities):
        sampled = assessment.copy()
        sampled["row_weight"] *= multiplicity
        sampled = sampled.loc[sampled["row_weight"].gt(0)]
        regenerated = _increments(sampled, boundaries)
        for statistic, value in regenerated.items():
            maximum_bootstrap_difference = max(
                maximum_bootstrap_difference,
                abs(float(value) - float(stored_draws.loc[draw, statistic])),
            )
    stored_intervals = bootstrap.loc[bootstrap["record_type"].eq("interval")]
    for row in stored_intervals.itertuples(index=False):
        values = stored_draws[str(row.statistic)].to_numpy(float)
        alpha = 1.0 - float(row.interval_level)
        maximum_bootstrap_difference = max(
            maximum_bootstrap_difference,
            abs(float(np.quantile(values, alpha / 2.0)) - float(row.lower)),
            abs(float(np.quantile(values, 1.0 - alpha / 2.0)) - float(row.upper)),
        )
    checks["session_bootstrap_15_draws"] = maximum_bootstrap_difference <= 1e-12

    null_draws = null_metrics.loc[null_metrics["record_type"].eq("draw")].set_index("draw")
    maximum_null_probability_difference = 0.0
    maximum_null_metric_difference = 0.0
    h0_metrics = binary_hazard_metrics(
        assessment["registered_completion_next_3_bars"],
        assessment["H0_probability"],
        assessment["row_weight"],
    )
    for draw in range(3):
        permuted = permute_route_bundle(
            panel,
            route_features=ROUTE_FEATURES,
            strata=("period", "session", "checkpoint"),
            seed=20260723 + draw,
        )
        bundle_hash = stable_frame_hash(
            permuted,
            ["period", "session", "checkpoint", "symbol", *ROUTE_FEATURES],
        )
        if bundle_hash != str(null_draws.loc[draw, "route_bundle_hash"]):
            maximum_null_metric_difference = float("inf")
        permuted_assessment = permuted.loc[permuted["period"].eq("assessment")]
        regenerated_probability = reconstruct_hazard_probability(
            permuted_assessment, coefficients["route_null_models"][str(draw)]
        )
        stored_probability = assessment[f"route_null_{draw}_probability"].to_numpy(float)
        maximum_null_probability_difference = max(
            maximum_null_probability_difference,
            float(np.max(np.abs(regenerated_probability - stored_probability))),
        )
        regenerated_metric = binary_hazard_metrics(
            assessment["registered_completion_next_3_bars"],
            stored_probability,
            assessment["row_weight"],
        )
        expected = {
            "log_loss_improvement": float(h0_metrics["log_loss"])
            - float(regenerated_metric["log_loss"]),
            "brier_improvement": float(h0_metrics["brier_score"])
            - float(regenerated_metric["brier_score"]),
            "auc_improvement": float(regenerated_metric["auc"]) - float(h0_metrics["auc"]),
        }
        for statistic, value in expected.items():
            maximum_null_metric_difference = max(
                maximum_null_metric_difference,
                abs(float(value) - float(null_draws.loc[draw, statistic])),
            )
    comparison = null_metrics.loc[null_metrics["record_type"].eq("comparison")].set_index(
        "statistic"
    )
    real_increments = _increments(assessment, boundaries)
    for statistic in (
        "log_loss_improvement",
        "brier_improvement",
        "auc_improvement",
    ):
        expected_count = int((float(real_increments[statistic]) > null_draws[statistic]).sum())
        maximum_null_metric_difference = max(
            maximum_null_metric_difference,
            abs(expected_count - int(comparison.loc[statistic, "real_exceeds_null_count"])),
        )
    checks["route_bundle_null_3_draws_reconstructed_without_refit"] = bool(
        maximum_null_probability_difference <= 1e-12 and maximum_null_metric_difference <= 1e-12
    )

    support = decision["support"]
    support_blocker = None
    if not (
        len(assessment) >= 30000
        and assessment["session"].nunique() >= 140
        and assessment["symbol"].nunique() >= 15
        and assessment["year_month"].nunique() >= 8
        and assessment["registered_completion_next_3_bars"].sum() >= 500
        and float(support["feature_retention"]) >= 0.95
        and float(support["maximum_weighted_stock_share"]) <= 0.10
        and float(support["maximum_target_class_share"]) <= 0.90
    ):
        support_blocker = "blocked_insufficient_support"
    gates = decision["decision_gates"]
    h1_passed = route_increment_passes(gates)
    expected_decision = choose_primary_decision(
        blocker=support_blocker,
        h1_passed=h1_passed,
        route_narrowing_ordered=bool(decision["route_narrowing_ordered"]),
        h0_meaningful=bool(decision["H0_meaningful"]),
    )
    checks["decision_logic"] = expected_decision == decision["primary_decision"]
    checks["determinism_artifact"] = bool(determinism["passed"])

    passed = all(bool(value) for value in checks.values())
    return {
        **SAFETY_FLAGS,
        "auditor": "audit_screen_v0.py",
        "independent_artifact_reload": True,
        "model_refits_performed": 0,
        "bootstrap_refits_performed": 0,
        "route_null_refits_performed": 0,
        "manual_probability_rows": 200,
        "manual_route_rows": 240,
        "independent_structural_ledger_rows": len(ledger),
        "stored_structural_ledger_hash": stored_ledger_hash,
        "reconstructed_structural_ledger_hash": reconstructed_ledger_hash,
        "maximum_regime_difference": maximum_regime_difference,
        "maximum_component_difference": maximum_component_difference,
        "maximum_dimension_difference": maximum_dimension_difference,
        "maximum_local_feature_difference": maximum_local_difference,
        "maximum_route_feature_difference": maximum_route_difference,
        "maximum_structural_memory_difference": maximum_structural_memory_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "maximum_bootstrap_difference": maximum_bootstrap_difference,
        "maximum_null_probability_difference": maximum_null_probability_difference,
        "maximum_null_metric_difference": maximum_null_metric_difference,
        "checks": checks,
        "passed": passed,
    }


def main() -> int:
    result = run_audit()
    if result["passed"]:
        screen_runner = load_module(SCREEN_RUNNER, "route_hazard_report_finalizer")
        report_result = cast(dict[str, Any], screen_runner.finalize_report(PRIMARY, audit=result))
        result["report_sha256"] = report_result["sha256"]
        result["report_copies_match"] = bool(report_result["copies_match"])
        cast(dict[str, Any], result["checks"])["report_copies_match"] = bool(
            report_result["copies_match"]
        )
        result["passed"] = all(
            bool(value) for value in cast(Mapping[str, Any], result["checks"]).values()
        )
    (PRIMARY / "lightweight_audit.json").write_text(canonical_json(result), encoding="utf-8")
    print(canonical_json(result), end="")
    if result["passed"]:
        return 0
    decision = read_json(PRIMARY / "decision.json")
    decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
    decision["blocker"] = "blocked_reproducibility_or_audit_failure"
    (PRIMARY / "decision.json").write_text(canonical_json(decision), encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
