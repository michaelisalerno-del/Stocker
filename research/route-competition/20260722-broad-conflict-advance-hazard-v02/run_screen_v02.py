#!/usr/bin/env python3
"""Run the Broad-Conflict Advance-Hazard Dense-Checkpoint Quick Screen V0.2."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits are fixed before numerical imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-broad-conflict-advance-v02-mpl")

import argparse
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

from stocker_research.broad_conflict_advance_hazard_v02 import (
    BASELINE_NON_CLOCK_FEATURES,
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    DENSE_H1_FEATURES,
    ROUTE_FEATURES,
    advance_increment_passes,
    assign_frozen_route_states,
    broad_conflict_mechanism_passes,
    candidate_normalized_weights,
    choose_broad_conflict_decision,
    predecessor_surface_differences,
    route_bundle_permutation,
    session_bootstrap_multiplicities,
    theoretical_raw_population,
)
from stocker_research.route_competition_hazard_v0 import (
    binary_hazard_metrics,
    fit_hazard_model,
    reject_protected_dates,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
V0_DIR = REPO_ROOT / "research" / "route-competition" / "20260722-route-competition-hazard-quick-v0"
V0_PRIMARY = V0_DIR / "artifacts" / "primary"
V01_DIR = (
    REPO_ROOT / "research" / "route-competition" / "20260722-route-competition-fixed-lead-audit-v01"
)
V01_PRIMARY = V01_DIR / "artifacts" / "primary"
V0_RUNNER = V0_DIR / "run_screen_v0.py"
V01_RUNNER = V01_DIR / "run_screen_v01.py"

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
FROZEN_COHORT = (
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
)
SHARED_CHECKPOINTS = (6, 10, 14, 18, 22, 26, 30, 34)
ROUTE_STATES = (
    "BROAD_CONFLICT",
    "NARROWING",
    "DOMINANT_ROUTE",
    "LOW_ROUTE_SUPPORT",
    "OTHER",
)
DEVELOPMENT_START = "2024-01-01"
ASSESSMENT_START = "2025-01-01"
READ_END = "2025-08-22"
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
BOOTSTRAP_DRAWS = 15
BOOTSTRAP_SEED = 20260722
NULL_REFITS = 3
NULL_SEED = 20260811
EXPECTED_REGISTERED_EVENT_HASH = "e538274d689074d191f57e88f5f99283d6818a4bee7e8f29e3fe7cc1aa8329c2"

EXPECTED_V0_SHA256 = {
    "assessment_predictions.parquet": (
        "2d212cc21991f0c1a92b9c5cb34cf9e324cbd92d56c9de9c6d0fc002ddd46463"
    ),
    "baseline_feature_manifest.json": (
        "cc89c5dbd159ea724a05fc33b461d638e399d32ea90fda5ddd0c6623f1b54b38"
    ),
    "causal_state_trace.parquet": (
        "b58ad2f824ed1a81d0d8846864bb1127330a46b4bd93d5d762ccff1f60857f1d"
    ),
    "checkpoint_manifest.json": "91781bd4a330734576f113b1b58a4a7d125e119c95c41706eb84bfdfabbdcb0f",
    "decision_panel.parquet": "aeec77229470165cf4b547ea19eb72db82ff8f377936a0694a68265222e36083",
    "determinism_check.json": "460d691fb9d857cc2ab737c5f40ffe15f6f7066d5a7f0e71111de2e9fb9d8850",
    "lightweight_audit.json": "44e29438769bed453fb164e8f3d4e93a177c2905de9601f720f61b2a8b87ef28",
    "model_configurations.json": "5cd3bab546aa7484118c595e7d562cea5407c0fa227360d868654c99b15b7014",
    "route_competition_feature_manifest.json": (
        "edcb2ab127be6e9f5bc6f248cf68390447485f3c53952a83b73f81a379386e5c"
    ),
    "route_competition_ledger.parquet": (
        "b3eab7d5b699719220bd8fd084ca0b25a8b2b56851211a093fe04a3cb22f124a"
    ),
    "source_manifest.json": "64cd179d36d9b3a8b9c5869f176e965f43b6faeabf53d013d29b7736698e03a5",
}
EXPECTED_V01_SHA256 = {
    "decision.json": "2aef1eddad38853a26977d31ef6f9f37a21033733739be41e59a8cae35112d8a",
    "determinism_check.json": "eddd13a39a7b1235f88c259cdc5092b098c9d60eba8dc047fad7c8771cab6e28",
    "fixed_lead_panel.parquet": "64e72905833805ea19c82b0b2572278f1af9eb2b9b2aee1f07d9dd2d65e71fe1",
    "lightweight_audit.json": "1ad6a1d6b07841378b81f043e21216dc1175e58f4f3a68aac174badd86352765",
    "predecessor_reconstruction.json": (
        "1e9cfdbb256403021c59ae915e2c341793c501ad85b7c37a94ed9080b26ed103"
    ),
    "prefix_proximity_manifest.json": (
        "8562b765f7bf2d780d45d38baf42529e3aee46bc8c5f462ec682cb8d85758303"
    ),
    "source_manifest.json": "45977529217340be84d9c3d142700769e2c3c131e8b462d96eeb6ae5559abd57",
}
METRIC_COLUMNS = (
    "log_loss",
    "brier_score",
    "auc",
    "average_precision",
    "expected_calibration_error",
    "calibration_intercept",
    "calibration_slope",
    "base_rate",
    "mean_probability_realised_class",
    "rows",
    "positive_outcomes",
    "model",
    "unique_stock_sessions",
    "sessions",
    "stocks",
    "top_decile_probability_boundary",
    "top_decile_precision",
    "top_decile_lift",
    "top_decile_rows",
    "top_quintile_probability_boundary",
    "top_quintile_precision",
    "top_quintile_lift",
    "top_quintile_rows",
)


class ScreenBlocker(RuntimeError):
    """A fail-closed preregistered screen blocker."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return str(value)


def canonical_json(value: Any) -> str:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=_json_default)
        + "\n"
    )


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="mergesort")
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocker("blocked_predecessor_reconstruction_failure", f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"contract safety flag differs: {key}",
            )
    expected_limits = {
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
        "primary_model_fits": 2,
        "session_bootstrap_draws": 15,
        "route_bundle_null_refits": 3,
        "maximum_plots": 1,
    }
    limits = cast(Mapping[str, Any], contract["hard_limits"])
    if any(limits.get(key) != value for key, value in expected_limits.items()):
        raise ScreenBlocker("blocked_quick_broad_conflict_resource_limit", "hard limits differ")
    if tuple(contract["checkpoints"]) != DENSE_CHECKPOINTS:
        raise ScreenBlocker("blocked_quick_broad_conflict_resource_limit", "checkpoint set differs")
    return contract


def _verified_hashes(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    missing = [name for name in expected if not (root / name).is_file()]
    if missing:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            f"predecessor artifacts missing: {missing}",
        )
    actual = {name: sha256_file(root / name) for name in expected}
    if actual != dict(expected):
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "commit-pinned predecessor artifact hashes differ",
        )
    return actual


def load_frozen_sources() -> dict[str, Any]:
    v0_hashes = _verified_hashes(V0_PRIMARY, EXPECTED_V0_SHA256)
    v01_hashes = _verified_hashes(V01_PRIMARY, EXPECTED_V01_SHA256)
    v0_audit = read_json(V0_PRIMARY / "lightweight_audit.json")
    v0_determinism = read_json(V0_PRIMARY / "determinism_check.json")
    v01_audit = read_json(V01_PRIMARY / "lightweight_audit.json")
    v01_determinism = read_json(V01_PRIMARY / "determinism_check.json")
    if not all(
        bool(value["passed"]) for value in (v0_audit, v0_determinism, v01_audit, v01_determinism)
    ):
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "predecessor audit or determinism did not pass",
        )
    return {
        "states": pd.read_parquet(V0_PRIMARY / "causal_state_trace.parquet"),
        "ledger": pd.read_parquet(V0_PRIMARY / "route_competition_ledger.parquet"),
        "v0_panel": pd.read_parquet(V0_PRIMARY / "decision_panel.parquet"),
        "v01_panel": pd.read_parquet(V01_PRIMARY / "fixed_lead_panel.parquet"),
        "v0_source": read_json(V0_PRIMARY / "source_manifest.json"),
        "v0_baseline_manifest": read_json(V0_PRIMARY / "baseline_feature_manifest.json"),
        "v0_route_manifest": read_json(V0_PRIMARY / "route_competition_feature_manifest.json"),
        "v01_source": read_json(V01_PRIMARY / "source_manifest.json"),
        "v01_prefix_manifest": read_json(V01_PRIMARY / "prefix_proximity_manifest.json"),
        "v0_hashes": v0_hashes,
        "v01_hashes": v01_hashes,
    }


def registered_event_hash(ledger: pd.DataFrame) -> str:
    registered = ledger.loc[ledger["ledger_kind"].eq("registered_completion")]
    columns = (
        "symbol",
        "session",
        "bar_ordinal",
        "semantic_loop_id",
        "primitive_loop_id",
        "orientation_id",
        "motif_type",
        "repeat_depth",
        "available_timestamp_utc",
    )
    return stable_frame_hash(registered, columns)


def reconstruct_fixed_lead_labels(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    canonical_routes: pd.DataFrame,
    *,
    checkpoints: Sequence[int],
) -> pd.DataFrame:
    """Vectorize the frozen V0.1 canonical-prefix and three-bar lead semantics."""

    registered = ledger.loc[ledger["ledger_kind"].eq("registered_completion")].copy()
    raw_prefixes = ledger.loc[
        ledger["ledger_kind"].eq("active_prefix")
        & ledger["bar_ordinal"].astype(int).isin(checkpoints)
    ].copy()
    if not set(raw_prefixes["motif_type"].dropna().astype(str)).issubset(
        {"primitive", "repeat", "composite"}
    ):
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "unknown registered prefix motif",
        )
    checked_prefixes = raw_prefixes.merge(
        canonical_routes,
        on=["semantic_loop_id", "orientation_id"],
        how="left",
        validate="many_to_one",
    )
    if checked_prefixes["canonical_total_transitions"].isna().any():
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "active prefix orientation is absent from the frozen dictionary",
        )
    if (
        not checked_prefixes["motif_type"]
        .astype(str)
        .eq(checked_prefixes["dictionary_motif_type"].astype(str))
        .all()
    ):
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "active prefix motif differs from the frozen dictionary",
        )
    progress = checked_prefixes["progress_states"].astype(int)
    declared = checked_prefixes["transitions_remaining"].astype(int)
    checked_prefixes["remaining_required_transitions"] = checked_prefixes[
        "canonical_total_transitions"
    ].astype(int) - (progress - 1)
    if (
        bool(progress.lt(1).any())
        or bool(declared.lt(0).any())
        or bool(checked_prefixes["remaining_required_transitions"].lt(0).any())
        or not checked_prefixes["remaining_required_transitions"].eq(declared).all()
    ):
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "declared prefix remainder differs from canonical route length",
        )
    prefixes = checked_prefixes.drop_duplicates(
        ["symbol", "session", "bar_ordinal", "semantic_loop_id", "orientation_id"]
    ).copy()
    prefixes["one_transition_away"] = prefixes["remaining_required_transitions"].eq(1)
    proximity = (
        prefixes.groupby(["symbol", "session", "bar_ordinal"], sort=True)
        .agg(
            minimum_remaining_transitions=("remaining_required_transitions", "min"),
            number_of_one_transition_away_prefixes=("one_transition_away", "sum"),
        )
        .reset_index()
        .rename(columns={"bar_ordinal": "checkpoint"})
    )
    result = panel.copy().merge(
        proximity,
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    result["number_of_one_transition_away_prefixes"] = (
        result["number_of_one_transition_away_prefixes"].fillna(0).astype(int)
    )
    result["any_prefix_one_transition_from_completion"] = (
        result["number_of_one_transition_away_prefixes"].gt(0).astype(int)
    )
    event_keys = {
        (str(row.symbol), str(row.session), int(row.bar_ordinal))
        for row in registered[["symbol", "session", "bar_ordinal"]].itertuples(index=False)
    }
    identities = {
        (str(symbol), str(session), int(bar)): json.dumps(
            sorted(set(group["semantic_loop_id"].astype(str)))
        )
        for (symbol, session, bar), group in registered.groupby(
            ["symbol", "session", "bar_ordinal"], sort=False
        )
    }
    keys = list(result[["symbol", "session", "checkpoint"]].itertuples(index=False, name=None))
    leads = np.zeros(len(result), dtype=np.int8)
    for lead in (1, 2, 3):
        present = np.fromiter(
            (
                (str(symbol), str(session), int(checkpoint) + lead) in event_keys
                for symbol, session, checkpoint in keys
            ),
            dtype=bool,
            count=len(keys),
        )
        leads[(leads == 0) & present] = lead
    result["first_completion_lead"] = leads.astype(int)
    result["first_completion_semantic_loop_ids"] = [
        "[]" if lead == 0 else identities[(str(symbol), str(session), int(checkpoint) + int(lead))]
        for (symbol, session, checkpoint), lead in zip(keys, leads, strict=True)
    ]
    result["completion_next_1_bar"] = result["first_completion_lead"].eq(1).astype(int)
    result["completion_in_bars_2_or_3"] = result["first_completion_lead"].isin([2, 3]).astype(int)
    result["advance_eligible"] = (
        result["first_completion_lead"].ne(1)
        & result["any_prefix_one_transition_from_completion"].eq(0)
    ).astype(int)
    if (
        not result["completion_next_1_bar"]
        .eq(result["registered_completion_next_1_bar"].astype(int))
        .all()
        or not result["first_completion_lead"]
        .ne(0)
        .eq(result["registered_completion_next_3_bars"].astype(bool))
        .all()
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure",
            "fixed leads do not reproduce the frozen three-bar targets",
        )
    return result.sort_values(
        ["period", "session", "checkpoint", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def build_dense_panel(
    source: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Rebuild only the frozen causal checkpoint surface from predecessor ledgers."""

    states = cast(pd.DataFrame, source["states"])
    ledger = cast(pd.DataFrame, source["ledger"])
    v0_runner = load_module(V0_RUNNER, "broad_conflict_v02_v0_runner")
    v0_runner.CHECKPOINTS = DENSE_CHECKPOINTS
    v0_runner.BASELINE_FEATURES = DENSE_H0_FEATURES
    raw_panel, exclusions, possible_rows = v0_runner.build_raw_decision_panel(states, ledger)
    panel, component_scaling, local_scaling = v0_runner.add_development_frozen_baseline_features(
        raw_panel
    )

    v0_route_manifest = cast(Mapping[str, Any], source["v0_route_manifest"])
    frozen_bins = cast(Mapping[str, Any], v0_route_manifest["development_frozen_bins"])
    route_thresholds = cast(Mapping[str, Sequence[float]], frozen_bins["route_quartiles"])
    panel["route_resolution_state"] = assign_frozen_route_states(panel, route_thresholds)
    panel["transition_probability_half"] = np.where(
        panel["transition_probability"].le(float(frozen_bins["transition_probability_median"])),
        "low",
        "high",
    )
    panel["posterior_entropy_half"] = np.where(
        panel["posterior_entropy"].le(float(frozen_bins["posterior_entropy_median"])),
        "low",
        "high",
    )
    panel["recent_registered_completion_group"] = np.where(
        panel["any_registered_completion_prior_6"].gt(0),
        "recent_registered_completion_within_6",
        "no_recent_registered_completion",
    )
    panel = panel.sort_values(
        ["period", "session", "checkpoint", "symbol"], kind="mergesort"
    ).reset_index(drop=True)

    reference = cast(pd.DataFrame, source["v0_panel"])
    dense_shared = panel.loc[panel["checkpoint"].isin(SHARED_CHECKPOINTS)]
    shared_features = (
        *BASELINE_NON_CLOCK_FEATURES,
        *(f"checkpoint_{checkpoint}" for checkpoint in SHARED_CHECKPOINTS),
        *ROUTE_FEATURES,
    )
    predecessor = {
        **SAFETY_FLAGS,
        **predecessor_surface_differences(reference, dense_shared, feature_columns=shared_features),
        "shared_checkpoints": list(SHARED_CHECKPOINTS),
        "shared_rows": len(dense_shared),
        "registered_event_identity_hash": registered_event_hash(ledger),
        "registered_event_identity_mismatches": int(
            registered_event_hash(ledger) != EXPECTED_REGISTERED_EVENT_HASH
        ),
    }
    predecessor["passed"] = bool(
        int(predecessor["row_identity_mismatches"]) == 0
        and int(predecessor["checkpoint_timestamp_mismatches"]) == 0
        and int(predecessor["target_mismatches"]) == 0
        and int(predecessor["route_resolution_label_mismatches"]) == 0
        and int(predecessor["registered_event_identity_mismatches"]) == 0
        and float(predecessor["maximum_shared_feature_difference"]) <= 1e-12
    )
    if not predecessor["passed"]:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "shared checkpoint surface differs from frozen V0",
        )

    v01_runner = load_module(V01_RUNNER, "broad_conflict_v02_v01_runner")
    v01_source = v01_runner.load_predecessor()
    canonical_routes, dictionary_manifest = v01_runner.load_canonical_route_metadata(v01_source)
    label_columns = (
        "minimum_remaining_transitions",
        "number_of_one_transition_away_prefixes",
        "any_prefix_one_transition_from_completion",
        "first_completion_lead",
        "first_completion_semantic_loop_ids",
        "completion_next_1_bar",
        "completion_in_bars_2_or_3",
        "advance_eligible",
    )
    panel = reconstruct_fixed_lead_labels(
        panel,
        ledger,
        canonical_routes,
        checkpoints=DENSE_CHECKPOINTS,
    )
    frozen_shared = (
        cast(pd.DataFrame, source["v01_panel"])
        .loc[lambda frame: frame["checkpoint"].isin(SHARED_CHECKPOINTS)]
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
    )
    reconstructed_shared = (
        panel.loc[panel["checkpoint"].isin(SHARED_CHECKPOINTS)]
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
    )
    numeric_label_columns = tuple(
        column for column in label_columns if column != "first_completion_semantic_loop_ids"
    )
    label_mismatches = int(
        (
            ~np.isclose(
                frozen_shared.loc[:, list(numeric_label_columns)].to_numpy(float),
                reconstructed_shared.loc[:, list(numeric_label_columns)].to_numpy(float),
                atol=0.0,
                rtol=0.0,
                equal_nan=True,
            )
        ).sum()
        + (
            frozen_shared["first_completion_semantic_loop_ids"].astype(str).to_numpy()
            != reconstructed_shared["first_completion_semantic_loop_ids"].astype(str).to_numpy()
        ).sum()
    )
    predecessor["fixed_lead_label_mismatches"] = label_mismatches
    predecessor["passed"] = bool(predecessor["passed"] and label_mismatches == 0)
    if label_mismatches:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "vectorized fixed-lead labels differ from frozen V0.1",
        )
    required_label_columns = (
        "number_of_one_transition_away_prefixes",
        "any_prefix_one_transition_from_completion",
        "first_completion_lead",
        "completion_next_1_bar",
        "completion_in_bars_2_or_3",
        "advance_eligible",
    )
    if panel[list(required_label_columns)].isna().any(axis=None):
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "one or more dense checkpoint rows lack a fixed-lead label",
        )
    panel = candidate_normalized_weights(panel)
    panel["checkpoint_group"] = pd.cut(
        panel["checkpoint"],
        bins=[5, 14, 24, 34],
        labels=["early_6_14", "middle_16_24", "later_26_34"],
        include_lowest=True,
        right=True,
    ).astype("string")
    reject_protected_dates(panel, column="session")
    advance = panel.loc[panel["advance_eligible"].eq(1)]
    if (
        not np.isfinite(advance.loc[:, list(DENSE_H1_FEATURES)].to_numpy(float)).all()
        or not np.isfinite(advance["row_weight"].to_numpy(float)).all()
        or bool(advance["row_weight"].le(0.0).any())
        or bool(advance["first_completion_lead"].eq(1).any())
        or bool(advance["any_prefix_one_transition_from_completion"].eq(1).any())
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure",
            "advance population violates feature, weight, or chronology constraints",
        )
    prefix_manifest = {
        **SAFETY_FLAGS,
        "definition": "canonical route transitions minus completed prefix transitions",
        "canonical_calculation": (
            "len(frozen registered oriented path) - 1 - (progress_states - 1)"
        ),
        "declared_ledger_remainder_used_only_as_cross_check": True,
        "semantic_loop_dictionary": dictionary_manifest,
        "motif_types": ["primitive", "repeat", "composite"],
        "raw_active_prefix_rows_checked": int(
            ledger.loc[
                ledger["ledger_kind"].eq("active_prefix")
                & ledger["bar_ordinal"].astype(int).isin(DENSE_CHECKPOINTS)
            ].shape[0]
        ),
        "unique_prefix_candidates_after_frozen_deduplication": int(
            ledger.loc[
                ledger["ledger_kind"].eq("active_prefix")
                & ledger["bar_ordinal"].astype(int).isin(DENSE_CHECKPOINTS)
            ]
            .drop_duplicates(
                [
                    "symbol",
                    "session",
                    "bar_ordinal",
                    "semantic_loop_id",
                    "orientation_id",
                ]
            )
            .shape[0]
        ),
        "frozen_deduplication_key": [
            "symbol",
            "session",
            "bar_ordinal",
            "semantic_loop_id",
            "orientation_id",
        ],
        "advance_definition": (
            "first_completion_lead != 1 and no active prefix exactly one transition away"
        ),
    }
    return (
        panel,
        exclusions,
        possible_rows,
        predecessor,
        component_scaling,
        local_scaling,
        prefix_manifest,
    )


def build_weight_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = panel.loc[panel["advance_eligible"].eq(1)].copy()
    stock_session = (
        eligible.groupby(["period", "session", "symbol"], sort=True)
        .agg(
            eligible_advance_rows=("row_id", "size"),
            eligible_stocks_in_session=("eligible_stocks_in_session", "first"),
            total_weight=("row_weight", "sum"),
        )
        .reset_index()
    )
    stock_session["expected_total_weight"] = 1.0 / stock_session[
        "eligible_stocks_in_session"
    ].astype(float)
    stock_session["absolute_weight_difference"] = (
        stock_session["total_weight"] - stock_session["expected_total_weight"]
    ).abs()
    stock_session["record_type"] = "stock_session"
    session = (
        stock_session.groupby(["period", "session"], sort=True)
        .agg(
            eligible_stocks_in_session=("symbol", "nunique"),
            total_weight=("total_weight", "sum"),
            minimum_stock_session_weight=("total_weight", "min"),
            maximum_stock_session_weight=("total_weight", "max"),
        )
        .reset_index()
    )
    session["expected_total_weight"] = 1.0
    session["absolute_weight_difference"] = (session["total_weight"] - 1.0).abs()
    session["record_type"] = "session"
    audit = pd.concat([stock_session, session], ignore_index=True, sort=False)
    summary = {
        **SAFETY_FLAGS,
        "advance_rows": len(eligible),
        "stock_sessions": int(stock_session.shape[0]),
        "sessions": int(session.shape[0]),
        "maximum_stock_session_weight_difference": float(
            stock_session["absolute_weight_difference"].max()
        ),
        "maximum_session_weight_difference": float(session["absolute_weight_difference"].max()),
        "finite_positive_eligible_weights": bool(
            np.isfinite(eligible["row_weight"].to_numpy(float)).all()
            and eligible["row_weight"].gt(0.0).all()
        ),
    }
    summary["passed"] = bool(
        summary["finite_positive_eligible_weights"]
        and float(summary["maximum_stock_session_weight_difference"]) <= 1e-12
        and float(summary["maximum_session_weight_difference"]) <= 1e-12
    )
    if not summary["passed"]:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "candidate-normalized weights differ"
        )
    return audit, summary


def support_and_concentration(
    panel: pd.DataFrame, states: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, str | None]:
    raw_assessment = panel.loc[panel["period"].eq("assessment")]
    advance = raw_assessment.loc[raw_assessment["advance_eligible"].eq(1)]
    assessment_sessions = int(
        states.loc[states["session"].astype(str).ge(ASSESSMENT_START), "session"].nunique()
    )
    theoretical = theoretical_raw_population(eligible_sessions=assessment_sessions)
    raw_retention = len(raw_assessment) / theoretical
    route_retention = float(
        np.isfinite(advance.loc[:, list(ROUTE_FEATURES)].to_numpy(float)).all(axis=1).mean()
    )
    weighted_stock_share = (
        advance.groupby("symbol")["row_weight"].sum() / advance["row_weight"].sum()
    )
    support = {
        "eligible_assessment_sessions": assessment_sessions,
        "theoretical_raw_assessment_rows": theoretical,
        "reconstructed_raw_assessment_rows": len(raw_assessment),
        "raw_retention": raw_retention,
        "raw_sessions": int(raw_assessment["session"].nunique()),
        "raw_stocks": int(raw_assessment["symbol"].nunique()),
        "raw_months": int(raw_assessment["year_month"].nunique()),
        "advance_rows": len(advance),
        "advance_sessions": int(advance["session"].nunique()),
        "advance_stocks": int(advance["symbol"].nunique()),
        "advance_months": int(advance["year_month"].nunique()),
        "advance_positive_outcomes": int(advance["completion_in_bars_2_or_3"].sum()),
        "route_feature_retention": route_retention,
        "maximum_weighted_stock_share": float(weighted_stock_share.max()),
        "base_rate": float(
            np.average(advance["completion_in_bars_2_or_3"], weights=advance["row_weight"])
        ),
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
    rows = [
        {
            "population": "raw",
            "metric": "retention",
            "value": raw_retention,
            "threshold": 0.95,
            "passed": gates["raw_retention"],
        },
        {
            "population": "advance",
            "metric": "rows",
            "value": support["advance_rows"],
            "threshold": 30_000,
            "passed": gates["advance_rows"],
        },
        {
            "population": "advance",
            "metric": "positive_outcomes",
            "value": support["advance_positive_outcomes"],
            "threshold": 400,
            "passed": gates["advance_positives"],
        },
        {
            "population": "advance",
            "metric": "route_feature_retention",
            "value": route_retention,
            "threshold": 0.95,
            "passed": gates["route_feature_retention"],
        },
        {
            "population": "advance",
            "metric": "maximum_weighted_stock_share",
            "value": support["maximum_weighted_stock_share"],
            "threshold": 0.10,
            "passed": gates["concentration"],
        },
    ]
    return support, pd.DataFrame(rows), blocker


def _top_precision(
    frame: pd.DataFrame, *, probability_column: str, target: str, threshold: float
) -> tuple[float, int]:
    selected = frame.loc[frame[probability_column].ge(threshold)]
    if selected.empty:
        return float("nan"), 0
    return (
        float(np.average(selected[target], weights=selected["row_weight"])),
        len(selected),
    )


def model_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    target: str,
    boundaries: Mapping[str, float],
) -> dict[str, Any]:
    probability_column = f"{model}_probability"
    values = binary_hazard_metrics(frame[target], frame[probability_column], frame["row_weight"])
    base_rate = float(values["base_rate"])
    decile, decile_rows = _top_precision(
        frame,
        probability_column=probability_column,
        target=target,
        threshold=float(boundaries["top_decile"]),
    )
    quintile, quintile_rows = _top_precision(
        frame,
        probability_column=probability_column,
        target=target,
        threshold=float(boundaries["top_quintile"]),
    )
    return {
        **values,
        "model": model,
        "unique_stock_sessions": int(frame[["session", "symbol"]].drop_duplicates().shape[0]),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "top_decile_probability_boundary": float(boundaries["top_decile"]),
        "top_decile_precision": decile,
        "top_decile_lift": decile / base_rate if base_rate > 0.0 else float("nan"),
        "top_decile_rows": decile_rows,
        "top_quintile_probability_boundary": float(boundaries["top_quintile"]),
        "top_quintile_precision": quintile,
        "top_quintile_lift": quintile / base_rate if base_rate > 0.0 else float("nan"),
        "top_quintile_rows": quintile_rows,
    }


def pair_increments(
    frame: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    baseline = model_metrics(
        frame,
        model="A0",
        target="completion_in_bars_2_or_3",
        boundaries=boundaries["A0"],
    )
    route = model_metrics(
        frame,
        model="A1",
        target="completion_in_bars_2_or_3",
        boundaries=boundaries["A1"],
    )
    return {
        "log_loss_improvement": float(baseline["log_loss"]) - float(route["log_loss"]),
        "brier_improvement": float(baseline["brier_score"]) - float(route["brier_score"]),
        "auc_improvement": float(route["auc"]) - float(baseline["auc"]),
        "average_precision_improvement": float(route["average_precision"])
        - float(baseline["average_precision"]),
        "top_decile_precision_improvement": float(route["top_decile_precision"])
        - float(baseline["top_decile_precision"]),
    }


def fit_primary_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, float]]]:
    output = panel.copy()
    population = output.loc[output["advance_eligible"].eq(1)]
    development = population.loc[population["period"].eq("development")]
    specifications: dict[str, Any] = {}
    boundaries: dict[str, dict[str, float]] = {}
    for model, features in (("A0", DENSE_H0_FEATURES), ("A1", DENSE_H1_FEATURES)):
        try:
            fitted = fit_hazard_model(
                development,
                features=features,
                target="completion_in_bars_2_or_3",
            )
        except ValueError as error:
            raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
        output.loc[population.index, f"{model}_probability"] = fitted.predict_probability(
            population
        )
        specifications[model] = {
            **fitted.as_dict(),
            "target": "completion_in_bars_2_or_3",
            "population": "advance_eligible",
        }
        development_probability = output.loc[development.index, f"{model}_probability"]
        boundaries[model] = {
            "top_decile": float(development_probability.quantile(0.90)),
            "top_quintile": float(development_probability.quantile(0.80)),
        }
    return output, specifications, boundaries


def weighted_rate(frame: pd.DataFrame, target: str) -> float:
    """Return a candidate-normalized weighted event rate."""

    if frame.empty or float(frame["row_weight"].sum()) <= 0.0:
        return float("nan")
    return float(np.average(frame[target], weights=frame["row_weight"]))


def build_model_tables(
    panel: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assessment = panel.loc[panel["period"].eq("assessment") & panel["advance_eligible"].eq(1)]
    pooled = pd.DataFrame(
        [
            model_metrics(
                assessment,
                model=model,
                target="completion_in_bars_2_or_3",
                boundaries=boundaries[model],
            )
            for model in ("A0", "A1")
        ]
    )
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_group_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    for checkpoint, group in assessment.groupby("checkpoint", sort=True):
        for model in ("A0", "A1"):
            checkpoint_rows.append(
                {
                    "checkpoint": int(checkpoint),
                    **model_metrics(
                        group,
                        model=model,
                        target="completion_in_bars_2_or_3",
                        boundaries=boundaries[model],
                    ),
                }
            )
    for checkpoint_group, group in assessment.groupby("checkpoint_group", sort=True):
        for model in ("A0", "A1"):
            checkpoint_group_rows.append(
                {
                    "checkpoint_group": str(checkpoint_group),
                    **model_metrics(
                        group,
                        model=model,
                        target="completion_in_bars_2_or_3",
                        boundaries=boundaries[model],
                    ),
                }
            )
    for month, group in assessment.groupby("year_month", sort=True):
        for model in ("A0", "A1"):
            monthly_rows.append(
                {
                    "year_month": str(month),
                    **model_metrics(
                        group,
                        model=model,
                        target="completion_in_bars_2_or_3",
                        boundaries=boundaries[model],
                    ),
                }
            )
    for dimension in (
        "transition_probability_half",
        "posterior_entropy_half",
        "recent_registered_completion_group",
        "route_resolution_state",
    ):
        for value, group in assessment.groupby(dimension, sort=True):
            for model in ("A0", "A1"):
                subgroup_rows.append(
                    {
                        "subgroup_dimension": dimension,
                        "subgroup_value": str(value),
                        **model_metrics(
                            group,
                            model=model,
                            target="completion_in_bars_2_or_3",
                            boundaries=boundaries[model],
                        ),
                    }
                )
    return (
        pooled,
        pd.DataFrame(checkpoint_rows),
        pd.DataFrame(checkpoint_group_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(subgroup_rows),
    )


def build_lead_tables(
    panel: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    support_rows: list[dict[str, Any]] = []
    for lead in range(4):
        group = assessment.loc[assessment["first_completion_lead"].eq(lead)]
        support_rows.append(
            {
                "first_completion_lead": lead,
                "rows": len(group),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "months": int(group["year_month"].nunique()),
                "one_transition_away_prefix_rows": int(
                    group["any_prefix_one_transition_from_completion"].sum()
                ),
                "one_transition_away_prefix_prevalence": float(
                    group["any_prefix_one_transition_from_completion"].mean()
                ),
                "advance_eligible_rows": int(group["advance_eligible"].sum()),
            }
        )
    diagnostics: list[dict[str, Any]] = []
    means = (
        "active_prefix_count",
        "active_prefix_family_count",
        "top_prefix_depth_fraction",
        "second_prefix_depth_fraction",
        "top_minus_second_prefix_depth",
        "prefix_family_entropy",
        "orientation_disagreement_fraction",
        "active_prefix_count_change_last_3_bars",
        "recent_loop_memory_weighted_top_depth",
    )
    advance = assessment.loc[assessment["advance_eligible"].eq(1)]
    for lead in (2, 3):
        group = advance.loc[advance["first_completion_lead"].eq(lead)]
        summary: dict[str, Any] = {
            "record_type": "lead_summary",
            "first_completion_lead": lead,
            "route_resolution_state": "ALL",
            "outcomes": len(group),
            "sessions": int(group["session"].nunique()),
            "stocks": int(group["symbol"].nunique()),
            "A0_mean_predicted_probability": float(
                np.average(group["A0_probability"], weights=group["row_weight"])
            )
            if boundaries is not None
            else float("nan"),
            "A1_mean_predicted_probability": float(
                np.average(group["A1_probability"], weights=group["row_weight"])
            )
            if boundaries is not None
            else float("nan"),
            "A1_top_decile_capture_rate": float(
                np.average(
                    group["A1_probability"].ge(boundaries["A1"]["top_decile"]),
                    weights=group["row_weight"],
                )
            )
            if boundaries is not None
            else float("nan"),
        }
        summary.update({f"mean_{name}": float(group[name].mean()) for name in means})
        diagnostics.append(summary)
        for state, state_group in group.groupby("route_resolution_state", sort=True):
            diagnostics.append(
                {
                    "record_type": "route_state_distribution",
                    "first_completion_lead": lead,
                    "route_resolution_state": str(state),
                    "outcomes": len(state_group),
                    "sessions": int(state_group["session"].nunique()),
                    "stocks": int(state_group["symbol"].nunique()),
                    "state_share": len(state_group) / len(group),
                }
            )
    return pd.DataFrame(support_rows), pd.DataFrame(diagnostics)


def build_route_state_metrics(
    panel: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    means = (
        "active_prefix_count",
        "active_prefix_family_count",
        "top_prefix_depth_fraction",
        "second_prefix_depth_fraction",
        "top_minus_second_prefix_depth",
        "prefix_family_entropy",
        "orientation_disagreement_fraction",
        "active_prefix_count_change_last_3_bars",
    )
    for period in ("development", "assessment"):
        population = panel.loc[panel["period"].eq(period) & panel["advance_eligible"].eq(1)]
        for state in ROUTE_STATES:
            group = population.loc[population["route_resolution_state"].eq(state)]
            row: dict[str, Any] = {
                "period": period,
                "route_resolution_state": state,
                "advance_eligible_rows": len(group),
                "positive_outcomes": int(group["completion_in_bars_2_or_3"].sum()),
                "completion_rate": weighted_rate(group, "completion_in_bars_2_or_3"),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "months": int(group["year_month"].nunique()),
                "inference_supported": bool(
                    period == "assessment"
                    and len(group) >= 100
                    and int(group["completion_in_bars_2_or_3"].sum()) >= 25
                ),
                "A1_minus_A0_log_loss_improvement": float("nan"),
                "A1_minus_A0_brier_improvement": float("nan"),
            }
            row.update(
                {
                    f"mean_{name}": float(group[name].mean()) if not group.empty else float("nan")
                    for name in means
                }
            )
            if boundaries is not None and not group.empty:
                increments = pair_increments(group, boundaries)
                row["A1_minus_A0_log_loss_improvement"] = increments["log_loss_improvement"]
                row["A1_minus_A0_brier_improvement"] = increments["brier_improvement"]
            rows.append(row)
    return pd.DataFrame(rows)


def run_bootstrap(
    assessment: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    point_increments = pair_increments(assessment, boundaries)
    broad = assessment.loc[assessment["route_resolution_state"].eq("BROAD_CONFLICT")]
    low = assessment.loc[assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
    point = {
        **point_increments,
        "broad_conflict_completion_rate": weighted_rate(broad, "completion_in_bars_2_or_3"),
        "broad_conflict_minus_pooled_rate": weighted_rate(broad, "completion_in_bars_2_or_3")
        - weighted_rate(assessment, "completion_in_bars_2_or_3"),
        "broad_conflict_minus_low_route_support_rate": weighted_rate(
            broad, "completion_in_bars_2_or_3"
        )
        - weighted_rate(low, "completion_in_bars_2_or_3"),
    }
    model_statistics = tuple(point_increments)
    broad_statistics = (
        "broad_conflict_completion_rate",
        "broad_conflict_minus_pooled_rate",
        "broad_conflict_minus_low_route_support_rate",
    )
    rows: list[dict[str, Any]] = []
    multiplicities = session_bootstrap_multiplicities(
        assessment["session"], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    for draw, multiplicity in enumerate(multiplicities):
        sample = assessment.copy()
        sample["row_weight"] *= multiplicity
        sample = sample.loc[sample["row_weight"].gt(0.0)]
        increments = pair_increments(sample, boundaries)
        sample_broad = sample.loc[sample["route_resolution_state"].eq("BROAD_CONFLICT")]
        sample_low = sample.loc[sample["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
        values = {
            **increments,
            "broad_conflict_completion_rate": weighted_rate(
                sample_broad, "completion_in_bars_2_or_3"
            ),
            "broad_conflict_minus_pooled_rate": weighted_rate(
                sample_broad, "completion_in_bars_2_or_3"
            )
            - weighted_rate(sample, "completion_in_bars_2_or_3"),
            "broad_conflict_minus_low_route_support_rate": weighted_rate(
                sample_broad, "completion_in_bars_2_or_3"
            )
            - weighted_rate(sample_low, "completion_in_bars_2_or_3"),
        }
        for statistic in (*model_statistics, *broad_statistics):
            rows.append(
                {
                    "record_type": "draw",
                    "comparison": "advance" if statistic in model_statistics else "broad_conflict",
                    "draw": draw,
                    "statistic": statistic,
                    "value": values[statistic],
                    "interval_level": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "point_estimate": point[statistic],
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    draws = pd.DataFrame(rows)
    intervals: list[dict[str, Any]] = []
    for (comparison, statistic), group in draws.groupby(["comparison", "statistic"], sort=True):
        values = group["value"].to_numpy(float)
        for level in (0.80, 0.90, 0.95):
            alpha = 1.0 - level
            intervals.append(
                {
                    "record_type": "interval",
                    "comparison": str(comparison),
                    "draw": np.nan,
                    "statistic": str(statistic),
                    "value": np.nan,
                    "interval_level": level,
                    "lower": float(np.quantile(values, alpha / 2.0)),
                    "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
                    "point_estimate": point[str(statistic)],
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    return pd.concat([draws, pd.DataFrame(intervals)], ignore_index=True)


def run_route_nulls(
    panel: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result = panel.copy()
    population = result.loc[result["advance_eligible"].eq(1)]
    assessment = population.loc[population["period"].eq("assessment")]
    real = pair_increments(assessment, boundaries)
    rows: list[dict[str, Any]] = []
    specifications: dict[str, Any] = {}
    for draw in range(NULL_REFITS):
        seed = NULL_SEED + draw
        permuted = route_bundle_permutation(
            population,
            route_features=ROUTE_FEATURES,
            strata=("period", "session", "checkpoint"),
            seed=seed,
        )
        development = permuted.loc[permuted["period"].eq("development")]
        try:
            fitted = fit_hazard_model(
                development,
                features=DENSE_H1_FEATURES,
                target="completion_in_bars_2_or_3",
            )
        except ValueError as error:
            raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
        permuted_assessment = permuted.loc[permuted["period"].eq("assessment")]
        probability = fitted.predict_probability(permuted_assessment)
        column = f"route_null_{draw}_probability"
        result.loc[permuted_assessment.index, column] = probability
        null_frame = assessment.copy()
        null_frame["A1_probability"] = probability
        increment = pair_increments(null_frame, boundaries)
        rows.append(
            {
                "record_type": "draw",
                "draw": draw,
                "seed": seed,
                "route_bundle_hash": stable_frame_hash(
                    permuted,
                    ["period", "session", "checkpoint", "symbol", *ROUTE_FEATURES],
                ),
                **increment,
            }
        )
        specifications[str(draw)] = {
            **fitted.as_dict(),
            "target": "completion_in_bars_2_or_3",
            "seed": seed,
        }
    draw_rows = list(rows)
    for statistic in (
        "log_loss_improvement",
        "brier_improvement",
        "auc_improvement",
        "average_precision_improvement",
    ):
        rows.append(
            {
                "record_type": "comparison",
                "statistic": statistic,
                "real_increment": real[statistic],
                "real_exceeds_null_count": sum(
                    real[statistic] > float(row[statistic]) for row in draw_rows
                ),
                "null_draws": NULL_REFITS,
            }
        )
    return result, pd.DataFrame(rows), specifications


def _bootstrap_lower(
    bootstrap: pd.DataFrame, comparison: str, statistic: str, level: float = 0.80
) -> float:
    row = bootstrap.loc[
        bootstrap["record_type"].eq("interval")
        & bootstrap["comparison"].eq(comparison)
        & bootstrap["statistic"].eq(statistic)
        & bootstrap["interval_level"].eq(level)
    ]
    if len(row) != 1:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"missing bootstrap interval: {comparison}/{statistic}/{level}",
        )
    return float(row.iloc[0]["lower"])


def _model_stability(metrics: pd.DataFrame, key: str) -> tuple[int, int]:
    positive = 0
    adverse = 0
    for _, group in metrics.groupby(key, sort=True):
        indexed = group.set_index("model")
        log_improvement = float(indexed.loc["A0", "log_loss"]) - float(
            indexed.loc["A1", "log_loss"]
        )
        brier_improvement = float(indexed.loc["A0", "brier_score"]) - float(
            indexed.loc["A1", "brier_score"]
        )
        positive += int(log_improvement > 0.0)
        adverse += int(log_improvement < -0.005 or brier_improvement < -0.002)
    return positive, adverse


def _state_rate_stability(panel: pd.DataFrame, key: str) -> tuple[int, int]:
    population = panel.loc[panel["period"].eq("assessment") & panel["advance_eligible"].eq(1)]
    positive = 0
    adverse = 0
    for _, group in population.groupby(key, sort=True):
        broad = group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT")]
        difference = weighted_rate(broad, "completion_in_bars_2_or_3") - weighted_rate(
            group, "completion_in_bars_2_or_3"
        )
        positive += int(difference > 0.0)
        adverse += int(difference < -0.005)
    return positive, adverse


def _real_exceeds_all_nulls(route_nulls: pd.DataFrame) -> bool:
    comparisons = route_nulls.loc[
        route_nulls["record_type"].eq("comparison")
        & route_nulls["statistic"].isin(["log_loss_improvement", "brier_improvement"])
    ]
    return bool(
        len(comparisons) == 2
        and comparisons["real_exceeds_null_count"].astype(int).eq(NULL_REFITS).any()
    )


def evaluate_decision(
    panel: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    checkpoint_group_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    route_nulls: pd.DataFrame,
    support: Mapping[str, Any],
    *,
    blocker: str | None = None,
) -> dict[str, Any]:
    assessment = panel.loc[panel["period"].eq("assessment") & panel["advance_eligible"].eq(1)]
    development = panel.loc[panel["period"].eq("development") & panel["advance_eligible"].eq(1)]
    configurations = cast(
        Mapping[str, Mapping[str, float]],
        {
            "A0": {
                "top_decile": float(
                    pooled_metrics.loc[
                        pooled_metrics["model"].eq("A0"), "top_decile_probability_boundary"
                    ].iloc[0]
                ),
                "top_quintile": float(
                    pooled_metrics.loc[
                        pooled_metrics["model"].eq("A0"), "top_quintile_probability_boundary"
                    ].iloc[0]
                ),
            },
            "A1": {
                "top_decile": float(
                    pooled_metrics.loc[
                        pooled_metrics["model"].eq("A1"), "top_decile_probability_boundary"
                    ].iloc[0]
                ),
                "top_quintile": float(
                    pooled_metrics.loc[
                        pooled_metrics["model"].eq("A1"), "top_quintile_probability_boundary"
                    ].iloc[0]
                ),
            },
        },
    )
    increments = pair_increments(assessment, configurations)
    positive_months, _ = _model_stability(monthly_metrics, "year_month")
    _, adverse_checkpoint_groups = _model_stability(checkpoint_group_metrics, "checkpoint_group")
    advance_gates: dict[str, object] = {
        **increments,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(
            bootstrap, "advance", "log_loss_improvement"
        ),
        "bootstrap_80_brier_lower": _bootstrap_lower(bootstrap, "advance", "brier_improvement"),
        "bootstrap_80_average_precision_lower": _bootstrap_lower(
            bootstrap, "advance", "average_precision_improvement"
        ),
        "positive_months": positive_months,
        "materially_adverse_checkpoint_groups": adverse_checkpoint_groups,
        "real_exceeds_all_nulls": _real_exceeds_all_nulls(route_nulls),
        "support_and_concentration_passed": bool(support["passed"]),
    }
    assessment_broad = assessment.loc[assessment["route_resolution_state"].eq("BROAD_CONFLICT")]
    assessment_low = assessment.loc[assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
    development_broad = development.loc[development["route_resolution_state"].eq("BROAD_CONFLICT")]
    broad_positive_months, broad_adverse_groups = (
        _state_rate_stability(panel, "year_month")[0],
        _state_rate_stability(panel, "checkpoint_group")[1],
    )
    broad_gates: dict[str, object] = {
        "assessment_minus_pooled": weighted_rate(assessment_broad, "completion_in_bars_2_or_3")
        - weighted_rate(assessment, "completion_in_bars_2_or_3"),
        "assessment_minus_low_route_support": weighted_rate(
            assessment_broad, "completion_in_bars_2_or_3"
        )
        - weighted_rate(assessment_low, "completion_in_bars_2_or_3"),
        "bootstrap_80_pooled_difference_lower": _bootstrap_lower(
            bootstrap, "broad_conflict", "broad_conflict_minus_pooled_rate"
        ),
        "bootstrap_80_low_route_difference_lower": _bootstrap_lower(
            bootstrap,
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
    advance_passed = advance_increment_passes(advance_gates)
    broad_passed = broad_conflict_mechanism_passes(broad_gates)
    broad_descriptive = bool(
        float(broad_gates["assessment_minus_pooled"]) > 0.0
        and float(broad_gates["assessment_minus_low_route_support"]) > 0.0
        and float(broad_gates["development_minus_pooled"]) > 0.0
    )
    a0 = pooled_metrics.set_index("model").loc["A0"]
    base_rate = float(a0["base_rate"])
    constant_log_loss = float(
        -np.average(
            assessment["completion_in_bars_2_or_3"] * math.log(max(base_rate, 1e-12))
            + (1 - assessment["completion_in_bars_2_or_3"]) * math.log(max(1.0 - base_rate, 1e-12)),
            weights=assessment["row_weight"],
        )
    )
    baseline_meaningful = bool(float(a0["auc"]) > 0.5 and float(a0["log_loss"]) < constant_log_loss)
    primary = choose_broad_conflict_decision(
        blocker=blocker,
        advance_passed=advance_passed,
        broad_conflict_passed=broad_passed,
        broad_conflict_descriptively_enriched=broad_descriptive,
        baseline_meaningful=baseline_meaningful,
    )
    lead_status: dict[str, str] = {}
    for lead in (2, 3):
        group = assessment.loc[assessment["first_completion_lead"].eq(lead)]
        enough = bool(
            len(group) >= 100
            and group["session"].nunique() >= 100
            and group["symbol"].nunique() >= 15
        )
        if not enough:
            status = "insufficient_support"
        elif advance_passed and float(
            np.average(group["A1_probability"], weights=group["row_weight"])
        ) > float(np.average(group["A0_probability"], weights=group["row_weight"])):
            status = "supported"
        else:
            status = "descriptive_only"
        lead_status[str(lead)] = status
    return {
        **SAFETY_FLAGS,
        "primary_decision": primary,
        "blocker": blocker,
        "advance_model_status": "supported" if advance_passed else "not_supported",
        "broad_conflict_status": "supported"
        if broad_passed
        else ("descriptive_only" if broad_descriptive else "not_supported"),
        "lead_two_status": lead_status["2"],
        "lead_three_status": lead_status["3"],
        "A1_minus_A0_increments": increments,
        "advance_model_gates": advance_gates,
        "broad_conflict_gates": broad_gates,
        "baseline_meaningful": baseline_meaningful,
        "constant_probability_log_loss": constant_log_loss,
        "support": dict(support),
    }


def build_source_and_boundary_manifests(
    source: Mapping[str, Any], panel: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    states = cast(pd.DataFrame, source["states"])
    ledger = cast(pd.DataFrame, source["ledger"])
    state_timestamps = pd.to_datetime(states["bar_complete_timestamp"], utc=True, errors="raise")
    ledger_timestamps = pd.to_datetime(ledger["available_timestamp_utc"], utc=True, errors="raise")
    checkpoint_timestamps = pd.to_datetime(
        panel["checkpoint_timestamp_utc"], utc=True, errors="raise"
    )
    protected_counts = {
        "causal_state_trace": int(state_timestamps.ge(PROTECTED_START).sum()),
        "structural_ledger": int(ledger_timestamps.ge(PROTECTED_START).sum()),
        "dense_checkpoint_panel": int(checkpoint_timestamps.ge(PROTECTED_START).sum()),
    }
    protected_rows = sum(protected_counts.values())
    boundary = {
        **SAFETY_FLAGS,
        "development_start": DEVELOPMENT_START,
        "development_end_inclusive": "2024-12-31",
        "assessment_start": ASSESSMENT_START,
        "assessment_end_inclusive": READ_END,
        "protected_start": str(PROTECTED_START),
        "minimum_timestamp_read": str(state_timestamps.min()),
        "maximum_timestamp_read": str(state_timestamps.max()),
        "maximum_structural_timestamp_read": str(ledger_timestamps.max()),
        "maximum_checkpoint_timestamp": str(checkpoint_timestamps.max()),
        "protected_files_touched": [],
        "protected_rows_by_source": protected_counts,
        "protected_rows_materialised": protected_rows,
        "passed": protected_rows == 0,
    }
    if not boundary["passed"]:
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "protected rows were materialised"
        )
    rows_by_stock = {
        str(key): int(value) for key, value in states.groupby("symbol", sort=True).size().items()
    }
    state_month = pd.to_datetime(states["session"]).dt.to_period("M").astype(str)
    rows_by_month = {
        str(key): int(value) for key, value in state_month.value_counts().sort_index().items()
    }
    v0_source = cast(Mapping[str, Any], source["v0_source"])
    state_source = cast(Mapping[str, Any], v0_source["state_source"])
    manifest = {
        **SAFETY_FLAGS,
        "predecessor_experiments": [
            str(V0_DIR.relative_to(REPO_ROOT)),
            str(V01_DIR.relative_to(REPO_ROOT)),
        ],
        "predecessor_artifact_hashes": {
            "hazard_v0": source["v0_hashes"],
            "fixed_lead_v01": source["v01_hashes"],
        },
        "dates_read": {"start": DEVELOPMENT_START, "end_inclusive": READ_END},
        "minimum_timestamp_read": boundary["minimum_timestamp_read"],
        "maximum_timestamp_read": boundary["maximum_timestamp_read"],
        "frozen_audited_cohort": list(FROZEN_COHORT),
        "rows_by_stock": rows_by_stock,
        "rows_by_month": rows_by_month,
        "source_hashes": state_source["source_hashes"],
        "protected_files_touched": [],
        "protected_rows_materialised": protected_rows,
        "raw_data_downloaded": False,
        "historical_activity_field": "EODHD historical activity proxy",
        "historical_activity_is_exchange_volume": False,
        "timeframe": "5m",
    }
    return manifest, boundary


def build_checkpoint_manifest(
    panel: pd.DataFrame, exclusions: pd.DataFrame, possible_rows: int
) -> dict[str, Any]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in DENSE_CHECKPOINTS:
        group = assessment.loc[assessment["checkpoint"].eq(checkpoint)]
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "rows": len(group),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "minimum_bar_start_timestamp": str(group["checkpoint_timestamp_utc"].min()),
                "maximum_bar_start_timestamp": str(group["checkpoint_timestamp_utc"].max()),
                "minimum_feature_available_timestamp": str(
                    group["feature_available_timestamp_utc"].min()
                ),
                "maximum_feature_available_timestamp": str(
                    group["feature_available_timestamp_utc"].max()
                ),
            }
        )
    assessment_sessions = int(assessment["session"].nunique())
    return {
        **SAFETY_FLAGS,
        "checkpoints": list(DENSE_CHECKPOINTS),
        "all_checkpoints_even": True,
        "shared_predecessor_checkpoints": list(SHARED_CHECKPOINTS),
        "new_dense_checkpoints": [
            value for value in DENSE_CHECKPOINTS if value not in SHARED_CHECKPOINTS
        ],
        "assessment_sessions": assessment_sessions,
        "nominal_theoretical_assessment_rows": theoretical_raw_population(
            eligible_sessions=assessment_sessions
        ),
        "possible_rows_all_periods": possible_rows,
        "raw_rows_all_periods": len(panel),
        "assessment_rows": len(assessment),
        "duplicate_row_ids": int(panel["row_id"].duplicated().sum()),
        "row_identity_sha256": stable_frame_hash(panel, ["row_id"]),
        "causal_feature_surface_sha256": stable_frame_hash(panel, ["row_id", *DENSE_H1_FEATURES]),
        "causal_feature_surface_columns": ["row_id", *DENSE_H1_FEATURES],
        "rows_by_checkpoint": checkpoint_rows,
        "exclusions_by_reason": exclusions.to_dict(orient="records"),
        "timestamp_sources": {
            "checkpoint_timestamp_utc": "frozen causal-state bar_start_timestamp",
            "feature_available_timestamp_utc": (
                "frozen causal-state bar_complete_timestamp; causal model availability time"
            ),
        },
    }


def plot_screen(pooled: pd.DataFrame, route_states: pd.DataFrame, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indexed = pooled.set_index("model")
    increments = [
        float(indexed.loc["A0", "log_loss"]) - float(indexed.loc["A1", "log_loss"]),
        float(indexed.loc["A0", "brier_score"]) - float(indexed.loc["A1", "brier_score"]),
        float(indexed.loc["A1", "average_precision"])
        - float(indexed.loc["A0", "average_precision"]),
    ]
    states = route_states.loc[route_states["period"].eq("assessment")]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(["Log loss", "Brier", "Avg precision"], increments, color="#355f8d")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("A1 minus A0 improvement")
    axes[1].bar(states["route_resolution_state"], states["completion_rate"], color="#d95f59")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("Weighted clean advance completion rate")
    axes[1].set_title("Frozen route-resolution states")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_report(
    decision: Mapping[str, Any],
    support: Mapping[str, Any],
    pooled: pd.DataFrame,
    route_states: pd.DataFrame,
    *,
    audit: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        "# Broad-Conflict Advance-Hazard Dense-Checkpoint Quick Screen V0.2",
        "",
        f"Primary decision: `{decision['primary_decision']}`.",
        "",
        "This retrospective structural screen used only frozen observable five-minute-bar, "
        "regime, behavioural, registered-loop and prefix ledgers. It did not open returns, "
        "direction, options, entries, exits, trading, execution, accounts or broker data.",
        "",
        "## Support",
        "",
        f"- Raw assessment rows: {int(support['reconstructed_raw_assessment_rows']):,} of "
        f"{int(support['theoretical_raw_assessment_rows']):,} "
        f"({float(support['raw_retention']):.4%}).",
        f"- Clean advance rows: {int(support['advance_rows']):,}; positives: "
        f"{int(support['advance_positive_outcomes']):,}; weighted base rate: "
        f"{float(support['base_rate']):.4%}.",
    ]
    blocker = decision.get("blocker")
    if blocker is not None:
        lines.extend(
            [
                "",
                f"The screen stopped at the preregistered blocker `{blocker}` before model "
                "fitting, bootstrap, or route-null refits.",
            ]
        )
        if audit is not None:
            lines.extend(["", f"Independent lightweight audit passed: {bool(audit['passed'])}."])
        lines.extend(
            [
                "",
                "This blocked screen provides no evidence of economic value, directional edge, "
                "options edge, trading utility or deployability.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(["", "## Models", ""])
    for row in pooled.itertuples(index=False):
        lines.append(
            f"- {row.model}: log loss {float(row.log_loss):.8f}; Brier "
            f"{float(row.brier_score):.8f}; AUC {float(row.auc):.8f}; average precision "
            f"{float(row.average_precision):.8f}; top-decile precision "
            f"{float(row.top_decile_precision):.4%}."
        )
    increments = cast(Mapping[str, Any], decision["A1_minus_A0_increments"])
    lines.extend(
        [
            "",
            "A1-minus-A0 improvements: log loss "
            f"{float(increments['log_loss_improvement']):.8f}; Brier "
            f"{float(increments['brier_improvement']):.8f}; AUC "
            f"{float(increments['auc_improvement']):.8f}; average precision "
            f"{float(increments['average_precision_improvement']):.8f}.",
            "",
            "## Frozen route states",
            "",
        ]
    )
    for row in route_states.loc[route_states["period"].eq("assessment")].itertuples(index=False):
        completion_rate = (
            f"{float(row.completion_rate):.4%}"
            if np.isfinite(float(row.completion_rate))
            else "unsupported"
        )
        lines.append(
            f"- {row.route_resolution_state}: {int(row.advance_eligible_rows):,} rows; "
            f"{int(row.positive_outcomes):,} positives; weighted completion rate "
            f"{completion_rate}."
        )
    if audit is not None:
        lines.extend(["", f"Independent lightweight audit passed: {bool(audit['passed'])}."])
    lines.extend(
        [
            "",
            "These findings are not prospective validation and provide no evidence of economic "
            "value, directional edge, options edge, trading utility or deployability.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_report(output: Path, *, audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    report = build_report(
        read_json(output / "decision.json"),
        cast(Mapping[str, Any], read_json(output / "decision.json")["support"]),
        pd.read_csv(output / "pooled_metrics.csv"),
        pd.read_csv(output / "route_resolution_state_metrics.csv"),
        audit=audit,
    )
    primary = output / "report.md"
    copy = REPORTS_DIR / "report.md"
    primary.write_text(report, encoding="utf-8")
    copy.write_text(report, encoding="utf-8")
    return {
        "sha256": sha256_file(primary),
        "copies_match": sha256_file(primary) == sha256_file(copy),
    }


def _registered_leads(panel: pd.DataFrame, ledger: pd.DataFrame) -> np.ndarray:
    completions = {
        (str(row.symbol), str(row.session), int(row.bar_ordinal))
        for row in ledger.loc[
            ledger["ledger_kind"].eq("registered_completion"),
            ["symbol", "session", "bar_ordinal"],
        ].itertuples(index=False)
    }
    leads = np.zeros(len(panel), dtype=np.int8)
    for index, row in enumerate(panel[["symbol", "session", "checkpoint"]].itertuples(index=False)):
        for lead in (1, 2, 3):
            if (str(row.symbol), str(row.session), int(row.checkpoint) + lead) in completions:
                leads[index] = lead
                break
    return leads


def determinism_check(
    output: Path,
    ledger: pd.DataFrame,
    route_thresholds: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    stored = pd.read_parquet(output / "dense_advance_panel.parquet")
    checkpoint_manifest = read_json(output / "dense_checkpoint_manifest.json")
    row_identity_hash = stable_frame_hash(stored, ["row_id"])
    feature_surface_hash = stable_frame_hash(stored, ["row_id", *DENSE_H1_FEATURES])
    duplicate_row_ids = int(stored["row_id"].duplicated().sum())
    row_identity_mismatches = (
        duplicate_row_ids
        + abs(len(stored) - int(checkpoint_manifest["raw_rows_all_periods"]))
        + int(row_identity_hash != checkpoint_manifest["row_identity_sha256"])
    )
    feature_surface_hash_mismatches = int(
        feature_surface_hash != checkpoint_manifest["causal_feature_surface_sha256"]
    )
    regenerated_leads = _registered_leads(stored, ledger)
    regenerated_states = assign_frozen_route_states(stored, route_thresholds)
    regenerated_eligibility = (
        (regenerated_leads != 1)
        & (stored["any_prefix_one_transition_from_completion"].to_numpy(int) == 0)
    ).astype(int)
    regenerated = stored.copy()
    regenerated["first_completion_lead"] = regenerated_leads
    regenerated["advance_eligible"] = regenerated_eligibility
    regenerated["completion_in_bars_2_or_3"] = np.isin(regenerated_leads, [2, 3]).astype(int)
    regenerated["route_resolution_state"] = regenerated_states
    regenerated = candidate_normalized_weights(regenerated)
    refit, refit_specs, refit_boundaries = fit_primary_models(regenerated)
    configurations = read_json(output / "model_configurations.json")
    stored_specs = cast(Mapping[str, Any], read_json(output / "model_coefficients.json"))[
        "primary_models"
    ]
    maximum_probability_difference = 0.0
    maximum_coefficient_difference = 0.0
    for model in ("A0", "A1"):
        population = stored["advance_eligible"].eq(1)
        maximum_probability_difference = max(
            maximum_probability_difference,
            float(
                np.max(
                    np.abs(
                        stored.loc[population, f"{model}_probability"].to_numpy(float)
                        - refit.loc[population, f"{model}_probability"].to_numpy(float)
                    )
                )
            ),
        )
        current = cast(Mapping[str, Any], refit_specs[model])
        archived = cast(Mapping[str, Any], stored_specs[model])
        maximum_coefficient_difference = max(
            maximum_coefficient_difference,
            float(
                np.max(
                    np.abs(
                        np.asarray(current["coefficient"], dtype=float)
                        - np.asarray(archived["coefficient"], dtype=float)
                    )
                )
            ),
            abs(float(current["intercept"]) - float(archived["intercept"])),
        )
    maximum_feature_difference = float(
        np.max(
            np.abs(
                stored.loc[:, list(DENSE_H1_FEATURES)].to_numpy(float)
                - regenerated.loc[:, list(DENSE_H1_FEATURES)].to_numpy(float)
            )
        )
    )
    maximum_weight_difference = float(
        np.nanmax(
            np.abs(stored["row_weight"].to_numpy(float) - regenerated["row_weight"].to_numpy(float))
        )
    )
    boundaries = cast(
        Mapping[str, Mapping[str, float]],
        configurations["probability_quantile_boundaries"],
    )
    boundary_difference = max(
        abs(refit_boundaries[model][quantile] - float(boundaries[model][quantile]))
        for model in ("A0", "A1")
        for quantile in ("top_decile", "top_quintile")
    )
    assessment = refit.loc[refit["period"].eq("assessment") & refit["advance_eligible"].eq(1)]
    regenerated_pooled = pd.DataFrame(
        [
            model_metrics(
                assessment,
                model=model,
                target="completion_in_bars_2_or_3",
                boundaries=boundaries[model],
            )
            for model in ("A0", "A1")
        ]
    )
    archived_pooled = pd.read_csv(output / "pooled_metrics.csv").set_index("model")
    maximum_metric_difference = 0.0
    for row in regenerated_pooled.itertuples(index=False):
        for column in (
            "log_loss",
            "brier_score",
            "auc",
            "average_precision",
            "base_rate",
            "top_decile_precision",
            "top_quintile_precision",
        ):
            maximum_metric_difference = max(
                maximum_metric_difference,
                abs(float(getattr(row, column)) - float(archived_pooled.loc[row.model, column])),
            )
    decision = read_json(output / "decision.json")
    _, _, checkpoint_groups, months, _ = build_model_tables(refit, boundaries)
    regenerated_decision = evaluate_decision(
        refit,
        regenerated_pooled,
        months,
        checkpoint_groups,
        pd.read_csv(output / "bootstrap_metrics.csv"),
        pd.read_csv(output / "route_null_metrics.csv"),
        cast(Mapping[str, Any], decision["support"]),
    )
    lead_mismatches = int(
        (stored["first_completion_lead"].to_numpy(int) != regenerated_leads).sum()
    )
    eligibility_mismatches = int(
        (stored["advance_eligible"].to_numpy(int) != regenerated_eligibility).sum()
    )
    state_mismatches = int(
        (
            stored["route_resolution_state"].astype(str).to_numpy()
            != regenerated_states.astype(str).to_numpy()
        ).sum()
    )
    final_decision_match = regenerated_decision["primary_decision"] == decision["primary_decision"]
    passed = bool(
        row_identity_mismatches == 0
        and lead_mismatches == 0
        and eligibility_mismatches == 0
        and state_mismatches == 0
        and maximum_probability_difference <= 1e-12
        and maximum_coefficient_difference <= 1e-12
        and maximum_feature_difference <= 1e-12
        and feature_surface_hash_mismatches == 0
        and maximum_weight_difference <= 1e-12
        and boundary_difference <= 1e-12
        and maximum_metric_difference <= 1e-12
        and final_decision_match
    )
    return {
        **SAFETY_FLAGS,
        "models_refit": ["A0", "A1"],
        "bootstrap_repeated": False,
        "route_null_refits_repeated": False,
        "row_identity_mismatches": row_identity_mismatches,
        "duplicate_row_ids": duplicate_row_ids,
        "row_identity_sha256": row_identity_hash,
        "row_identity_manifest_match": (
            row_identity_hash == checkpoint_manifest["row_identity_sha256"]
        ),
        "causal_feature_surface_sha256": feature_surface_hash,
        "causal_feature_surface_hash_mismatches": feature_surface_hash_mismatches,
        "lead_label_mismatches": lead_mismatches,
        "advance_eligibility_mismatches": eligibility_mismatches,
        "route_resolution_label_mismatches": state_mismatches,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_weight_difference": maximum_weight_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_coefficient_difference": maximum_coefficient_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "maximum_quantile_boundary_difference": boundary_difference,
        "regenerated_primary_decision": regenerated_decision["primary_decision"],
        "final_decision_match": final_decision_match,
        "passed": passed,
    }


def execute_screen(output: Path) -> dict[str, Any]:
    contract = load_contract()
    output.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    source = load_frozen_sources()
    (
        panel,
        exclusions,
        possible_rows,
        predecessor,
        component_scaling,
        local_scaling,
        prefix_manifest,
    ) = build_dense_panel(source)
    weight_audit, weight_summary = build_weight_audit(panel)
    support, concentration, blocker = support_and_concentration(
        panel, cast(pd.DataFrame, source["states"])
    )
    source_manifest, protected = build_source_and_boundary_manifests(source, panel)
    checkpoint_manifest = build_checkpoint_manifest(panel, exclusions, possible_rows)
    raw_assessment = panel.loc[panel["period"].eq("assessment")]
    lead_support, lead_diagnostics = build_lead_tables(panel)
    prefix_manifest.update(
        {
            "assessment_one_transition_away_prevalence": float(
                raw_assessment["any_prefix_one_transition_from_completion"].mean()
            ),
            "one_transition_away_prevalence_by_lead": {
                str(int(row.first_completion_lead)): float(
                    row.one_transition_away_prefix_prevalence
                )
                for row in lead_support.itertuples(index=False)
            },
            "orientation_preserved": True,
            "semantic_loop_identity_preserved": True,
        }
    )
    lead_manifest = {
        **SAFETY_FLAGS,
        "definition": (
            "earliest registered semantic-loop completion strictly after the checkpoint "
            "within the next three completed five-minute bars"
        ),
        "lead_values": [0, 1, 2, 3],
        "same_future_bar_multiple_completions_are_one_timing_outcome": True,
        "all_completing_semantic_identities_preserved_for_audit": True,
        "exact_identity_used_as_model_feature": False,
        "advance_target": "completion_in_bars_2_or_3",
        "advance_exclusions": [
            "first_completion_lead == 1",
            "any active prefix exactly one transition from completion",
        ],
        "lead_status_rule": (
            "supported only when the pooled A1 gate passes, lead support is at least 100 "
            "outcomes/100 sessions/15 stocks, and mean A1 probability exceeds A0; otherwise "
            "descriptive_only or insufficient_support"
        ),
        "unresolved_target_rows": 0,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected)
    write_json(output / "predecessor_reconstruction.json", predecessor)
    write_json(output / "dense_checkpoint_manifest.json", checkpoint_manifest)
    write_json(output / "lead_target_manifest.json", lead_manifest)
    write_json(output / "prefix_proximity_manifest.json", prefix_manifest)
    write_parquet(output / "dense_advance_panel.parquet", panel)
    write_csv(output / "weight_audit.csv", weight_audit)
    write_csv(output / "lead_support.csv", lead_support)
    write_csv(output / "lead_route_diagnostics.csv", lead_diagnostics)
    write_csv(output / "concentration_metrics.csv", concentration)

    if blocker is not None:
        decision = {
            **SAFETY_FLAGS,
            "primary_decision": blocker,
            "blocker": blocker,
            "advance_model_status": "insufficient_support",
            "broad_conflict_status": "insufficient_support",
            "lead_two_status": "insufficient_support",
            "lead_three_status": "insufficient_support",
            "support": support,
        }
        for name in (
            "pooled_metrics.csv",
            "checkpoint_metrics.csv",
            "checkpoint_group_metrics.csv",
            "monthly_metrics.csv",
            "subgroup_metrics.csv",
            "route_resolution_state_metrics.csv",
            "bootstrap_metrics.csv",
            "route_null_metrics.csv",
        ):
            write_csv(output / name, pd.DataFrame({"stop_reason": [blocker]}))
        write_json(
            output / "model_configurations.json",
            {
                **SAFETY_FLAGS,
                "planned_primary_models": ["A0", "A1"],
                "primary_models_fitted": 0,
                "bootstrap_draws_executed": 0,
                "route_null_refits_executed": 0,
                "stop_reason": blocker,
            },
        )
        write_json(
            output / "model_coefficients.json",
            {**SAFETY_FLAGS, "primary_models": {}, "route_null_models": {}},
        )
        for model in ("A0", "A1"):
            panel[f"{model}_probability"] = np.nan
        write_parquet(
            output / "assessment_predictions.parquet",
            panel.loc[panel["period"].eq("assessment")].reset_index(drop=True),
        )
        write_json(output / "decision.json", decision)
        determinism = {
            **SAFETY_FLAGS,
            "models_refit": [],
            "bootstrap_repeated": False,
            "route_null_refits_repeated": False,
            "primary_decision": blocker,
            "passed": True,
        }
        write_json(output / "determinism_check.json", determinism)
        finalize_report(output)
        return decision

    panel, primary_specs, boundaries = fit_primary_models(panel)
    pooled, checkpoints, checkpoint_groups, months, subgroups = build_model_tables(
        panel, boundaries
    )
    lead_support, lead_diagnostics = build_lead_tables(panel, boundaries)
    route_states = build_route_state_metrics(panel, boundaries)
    assessment_advance = panel.loc[
        panel["period"].eq("assessment") & panel["advance_eligible"].eq(1)
    ]
    bootstrap = run_bootstrap(assessment_advance, boundaries)
    panel, route_nulls, null_specs = run_route_nulls(panel, boundaries)
    decision = evaluate_decision(
        panel,
        pooled,
        months,
        checkpoint_groups,
        bootstrap,
        route_nulls,
        support,
    )
    model_configuration = {
        **SAFETY_FLAGS,
        "primary_models": ["A0", "A1"],
        "primary_models_fitted": 2,
        "target": "completion_in_bars_2_or_3",
        "population": "advance_eligible",
        "A0_features": list(DENSE_H0_FEATURES),
        "A1_features": list(DENSE_H1_FEATURES),
        "route_bundle": list(ROUTE_FEATURES),
        "probability_quantile_boundaries": boundaries,
        "component_development_scaling": component_scaling,
        "local_development_scaling": local_scaling,
        "model": contract["model"],
        "weight": ("1 / (eligible_stocks_in_session * eligible_advance_rows_for_stock_session)"),
        "session_bootstrap_draws": BOOTSTRAP_DRAWS,
        "route_bundle_null_refits": NULL_REFITS,
        "n_jobs": 1,
    }
    coefficients = {
        **SAFETY_FLAGS,
        "primary_models": primary_specs,
        "route_null_models": null_specs,
    }
    write_parquet(output / "dense_advance_panel.parquet", panel)
    write_parquet(
        output / "assessment_predictions.parquet",
        panel.loc[panel["period"].eq("assessment")].reset_index(drop=True),
    )
    write_csv(output / "lead_support.csv", lead_support)
    write_csv(output / "lead_route_diagnostics.csv", lead_diagnostics)
    write_csv(output / "route_resolution_state_metrics.csv", route_states)
    write_json(output / "model_configurations.json", model_configuration)
    write_json(output / "model_coefficients.json", coefficients)
    write_csv(output / "pooled_metrics.csv", pooled)
    write_csv(output / "checkpoint_metrics.csv", checkpoints)
    write_csv(output / "checkpoint_group_metrics.csv", checkpoint_groups)
    write_csv(output / "monthly_metrics.csv", months)
    write_csv(output / "subgroup_metrics.csv", subgroups)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "route_null_metrics.csv", route_nulls)
    write_json(output / "decision.json", decision)
    plot_screen(pooled, route_states, output / "advance_route_screen.png")
    route_thresholds = cast(
        Mapping[str, Sequence[float]],
        cast(Mapping[str, Any], source["v0_route_manifest"])["development_frozen_bins"][
            "route_quartiles"
        ],
    )
    determinism = determinism_check(
        output,
        cast(pd.DataFrame, source["ledger"]),
        route_thresholds,
    )
    if not determinism["passed"]:
        decision = {
            **decision,
            "primary_decision": "blocked_reproducibility_or_audit_failure",
            "blocker": "blocked_reproducibility_or_audit_failure",
        }
        write_json(output / "decision.json", decision)
    write_json(output / "determinism_check.json", determinism)
    finalize_report(output)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--finalize-report",
        action="store_true",
        help="Regenerate the report after the independent audit.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        if arguments.finalize_report:
            audit_path = arguments.output / "lightweight_audit.json"
            result = finalize_report(
                arguments.output,
                audit=read_json(audit_path) if audit_path.is_file() else None,
            )
        else:
            result = execute_screen(arguments.output)
    except ScreenBlocker as error:
        result = {"primary_decision": error.code, "detail": error.detail}
        print(canonical_json(result), end="")
        return 1
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
