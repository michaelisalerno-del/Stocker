#!/usr/bin/env python3
"""Run the Route-Competition Fixed-Lead Audit Quick Screen V0.1."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-route-fixed-lead-audit-mpl")

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

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.route_competition_fixed_lead_v01 import (
    choose_fixed_lead_decision,
    earliest_completion_lead,
    fixed_lead_increment_passes,
    predecessor_surface_differences,
    theoretical_assessment_support,
)
from stocker_research.route_competition_hazard_v0 import (
    BASELINE_FEATURES,
    CHECKPOINTS,
    H1_FEATURES,
    ROUTE_FEATURES,
    binary_hazard_metrics,
    fit_hazard_model,
    permute_route_bundle,
    reconstruct_hazard_probability,
    reject_protected_dates,
    session_bootstrap_multiplicities,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
PREDECESSOR_DIR = (
    REPO_ROOT / "research" / "route-competition" / "20260722-route-competition-hazard-quick-v0"
)
PREDECESSOR = PREDECESSOR_DIR / "artifacts" / "primary"
DICTIONARY_PATH = (
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
EXPECTED_PREDECESSOR_COMMIT = "1001693c70e99f92ae77777b0d6b3633777bf7af"
EXPECTED_DICTIONARY_SHA256 = "9550810616f9249f3a8adf32b08fe17c0e6fdc1cf582466d9d10ee6df639cb7a"
EXPECTED_DICTIONARY_HASH = "497142c8d0ab880e59385da123d9eb2189469e9e3a4a631e0f63eb6fc77030d3"
EXPECTED_PREDECESSOR_SHA256 = {
    "assessment_predictions.parquet": (
        "2d212cc21991f0c1a92b9c5cb34cf9e324cbd92d56c9de9c6d0fc002ddd46463"
    ),
    "baseline_feature_manifest.json": (
        "cc89c5dbd159ea724a05fc33b461d638e399d32ea90fda5ddd0c6623f1b54b38"
    ),
    "checkpoint_manifest.json": (
        "91781bd4a330734576f113b1b58a4a7d125e119c95c41706eb84bfdfabbdcb0f"
    ),
    "decision_panel.parquet": ("aeec77229470165cf4b547ea19eb72db82ff8f377936a0694a68265222e36083"),
    "determinism_check.json": ("460d691fb9d857cc2ab737c5f40ffe15f6f7066d5a7f0e71111de2e9fb9d8850"),
    "lightweight_audit.json": ("44e29438769bed453fb164e8f3d4e93a177c2905de9601f720f61b2a8b87ef28"),
    "model_coefficients.json": ("5b5af4a454e6ecf5f4aa8e28ccfa894dfaa361550354bee93ecb96b4781613e3"),
    "model_configurations.json": (
        "5cd3bab546aa7484118c595e7d562cea5407c0fa227360d868654c99b15b7014"
    ),
    "pooled_metrics.csv": "bd096bbbb42f15669d1a193ace78c3c840d4ed324700ecb58f3224c4e1d2d206",
    "protected_boundary_audit.json": (
        "07d89d7ae1d084255a053f6d1ab8e9a08c14a220c3804af3985d2e9545957794"
    ),
    "route_competition_feature_manifest.json": (
        "edcb2ab127be6e9f5bc6f248cf68390447485f3c53952a83b73f81a379386e5c"
    ),
    "route_competition_ledger.parquet": (
        "b3eab7d5b699719220bd8fd084ca0b25a8b2b56851211a093fe04a3cb22f124a"
    ),
    "source_manifest.json": "64cd179d36d9b3a8b9c5869f176e965f43b6faeabf53d013d29b7736698e03a5",
}

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
BOOTSTRAP_DRAWS = 15
BOOTSTRAP_SEED = 20260722
NULL_DRAWS = 3
NULL_SEEDS = {"immediate": 20260731, "advance": 20260803}
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
    """A fail-closed preregistered experiment blocker."""

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


def load_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"contract safety flag differs: {key}",
            )
    limits = cast(Mapping[str, Any], contract["hard_limits"])
    expected_limits = {
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
        "checkpoints": 8,
        "primary_model_fits": 4,
        "session_bootstrap_draws": 15,
        "route_feature_null_refits_per_target": 3,
        "maximum_plots": 1,
    }
    if any(limits.get(key) != value for key, value in expected_limits.items()):
        raise ScreenBlocker("blocked_quick_fixed_lead_resource_limit", "hard limits differ")
    if tuple(contract["checkpoints"]) != CHECKPOINTS:
        raise ScreenBlocker("blocked_quick_fixed_lead_resource_limit", "checkpoints differ")
    return contract


def load_predecessor() -> dict[str, Any]:
    required = (
        "decision_panel.parquet",
        "assessment_predictions.parquet",
        "route_competition_ledger.parquet",
        "pooled_metrics.csv",
        "model_coefficients.json",
        "model_configurations.json",
        "checkpoint_manifest.json",
        "route_competition_feature_manifest.json",
        "baseline_feature_manifest.json",
        "protected_boundary_audit.json",
        "determinism_check.json",
        "lightweight_audit.json",
        "source_manifest.json",
    )
    missing = [name for name in required if not (PREDECESSOR / name).is_file()]
    if missing:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            f"predecessor artifacts missing: {missing}",
        )
    hashes = {name: sha256_file(PREDECESSOR / name) for name in required}
    if hashes != EXPECTED_PREDECESSOR_SHA256:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "predecessor artifacts differ from the commit-pinned manifest",
        )
    return {
        "panel": pd.read_parquet(PREDECESSOR / "decision_panel.parquet"),
        "assessment": pd.read_parquet(PREDECESSOR / "assessment_predictions.parquet"),
        "ledger": pd.read_parquet(PREDECESSOR / "route_competition_ledger.parquet"),
        "pooled": pd.read_csv(PREDECESSOR / "pooled_metrics.csv"),
        "coefficients": read_json(PREDECESSOR / "model_coefficients.json"),
        "configurations": read_json(PREDECESSOR / "model_configurations.json"),
        "checkpoint_manifest": read_json(PREDECESSOR / "checkpoint_manifest.json"),
        "route_manifest": read_json(PREDECESSOR / "route_competition_feature_manifest.json"),
        "baseline_manifest": read_json(PREDECESSOR / "baseline_feature_manifest.json"),
        "protected": read_json(PREDECESSOR / "protected_boundary_audit.json"),
        "determinism": read_json(PREDECESSOR / "determinism_check.json"),
        "audit": read_json(PREDECESSOR / "lightweight_audit.json"),
        "source": read_json(PREDECESSOR / "source_manifest.json"),
        "hashes": hashes,
    }


def load_canonical_route_metadata(source: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and verify registered orientations independently of the prefix ledger."""

    if not DICTIONARY_PATH.is_file() or sha256_file(DICTIONARY_PATH) != EXPECTED_DICTIONARY_SHA256:
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "frozen semantic-loop dictionary source differs",
        )
    table = pd.read_csv(DICTIONARY_PATH)
    if set(table["dictionary_hash"].astype(str)) != {EXPECTED_DICTIONARY_HASH}:
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "frozen semantic-loop dictionary hash differs",
        )
    predecessor_manifest = cast(Mapping[str, Any], source["source"])
    predecessor_dictionary = cast(
        Mapping[str, Any], predecessor_manifest["loop_dictionary_manifest"]
    )
    if (
        predecessor_dictionary.get("dictionary_hash") != EXPECTED_DICTIONARY_HASH
        or predecessor_dictionary.get("source_sha256") != EXPECTED_DICTIONARY_SHA256
    ):
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "predecessor dictionary provenance differs",
        )
    rows: list[dict[str, Any]] = []
    for row in table.itertuples(index=False):
        semantic_loop_id = str(row.semantic_loop_id)
        motif_type = str(row.motif_type)
        canonical = tuple(int(value) for value in json.loads(str(row.canonical_orientation)))
        valid_paths = [
            tuple(int(value) for value in path)
            for path in json.loads(str(row.all_valid_oriented_paths))
        ]
        if canonical not in valid_paths or any(
            len(path) < 3 or path[0] != path[-1] or len(path) != len(canonical)
            for path in valid_paths
        ):
            raise ScreenBlocker(
                "blocked_prefix_proximity_reconstruction_failure",
                f"canonical orientation metadata differs for {semantic_loop_id}",
            )
        for path in valid_paths:
            rows.append(
                {
                    "semantic_loop_id": semantic_loop_id,
                    "orientation_id": (
                        f"{semantic_loop_id}__o_{'-'.join(str(value) for value in path)}"
                    ),
                    "dictionary_motif_type": motif_type,
                    "canonical_total_transitions": len(path) - 1,
                }
            )
    metadata = pd.DataFrame(rows)
    if metadata.duplicated(["semantic_loop_id", "orientation_id"]).any():
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "canonical orientation identities are not unique",
        )
    manifest = {
        "path": str(DICTIONARY_PATH.relative_to(REPO_ROOT)),
        "source_sha256": EXPECTED_DICTIONARY_SHA256,
        "dictionary_hash": EXPECTED_DICTIONARY_HASH,
        "registered_definition_count": int(table["semantic_loop_id"].nunique()),
        "registered_orientation_count": len(metadata),
    }
    return metadata, manifest


def _top_precision(
    frame: pd.DataFrame, *, probability_column: str, target: str, threshold: float
) -> tuple[float, int]:
    selected = frame.loc[frame[probability_column].ge(threshold)]
    if selected.empty:
        return float("nan"), 0
    return (
        float(np.average(selected[target], weights=selected["row_weight"])),
        int(len(selected)),
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


def verify_predecessor(source: Mapping[str, Any], panel: pd.DataFrame) -> dict[str, Any]:
    if tuple(cast(Mapping[str, Any], source["baseline_manifest"])["features"]) != BASELINE_FEATURES:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure", "H0 feature order differs"
        )
    if tuple(cast(Mapping[str, Any], source["route_manifest"])["features"]) != ROUTE_FEATURES:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure", "route feature order differs"
        )
    assessment_source = cast(pd.DataFrame, source["assessment"])
    assessment_panel = panel.loc[panel["period"].eq("assessment")].reset_index(drop=True)
    surface = predecessor_surface_differences(
        assessment_source,
        assessment_panel,
        feature_columns=(*BASELINE_FEATURES, *ROUTE_FEATURES),
    )
    assessment_row_mismatches = abs(len(assessment_source) - len(assessment_panel)) + sum(
        left != right
        for left, right in zip(
            assessment_source["row_id"].astype(str),
            assessment_panel["row_id"].astype(str),
            strict=False,
        )
    )
    coefficients = cast(Mapping[str, Any], source["coefficients"])
    manual_differences: list[float] = []
    for model in ("H0", "H1"):
        regenerated = reconstruct_hazard_probability(
            panel, cast(Mapping[str, object], coefficients["primary_models"][model])
        )
        manual_differences.append(
            float(np.max(np.abs(regenerated - panel[f"{model}_probability"].to_numpy(float))))
        )
    predecessor_boundaries = cast(
        Mapping[str, Mapping[str, float]],
        cast(Mapping[str, Any], source["configurations"])["probability_quantile_boundaries"],
    )
    recomputed = pd.DataFrame(
        [
            model_metrics(
                assessment_panel,
                model=model,
                target="registered_completion_next_3_bars",
                boundaries=predecessor_boundaries[model],
            )
            for model in ("H0", "H1")
        ]
    )
    stored = cast(pd.DataFrame, source["pooled"])
    metric_names = (
        "log_loss",
        "brier_score",
        "auc",
        "average_precision",
        "top_decile_precision",
        "top_quintile_precision",
    )
    metric_difference = 0.0
    for model in ("H0", "H1"):
        left = recomputed.loc[recomputed["model"].eq(model)].iloc[0]
        right = stored.loc[stored["model"].eq(model)].iloc[0]
        metric_difference = max(
            metric_difference,
            max(abs(float(left[name]) - float(right[name])) for name in metric_names),
        )
    result = {
        **SAFETY_FLAGS,
        **surface,
        "assessment_row_identity_mismatches": int(assessment_row_mismatches),
        "maximum_probability_difference": max(
            float(surface["maximum_probability_difference"]),
            max(manual_differences, default=0.0),
        ),
        "maximum_metric_difference": metric_difference,
        "predecessor_assessment_rows": len(assessment_panel),
        "predecessor_assessment_sessions": assessment_panel["session"].nunique(),
        "predecessor_assessment_stocks": assessment_panel["symbol"].nunique(),
        "predecessor_assessment_months": assessment_panel["year_month"].nunique(),
        "predecessor_three_bar_positives": int(
            assessment_panel["registered_completion_next_3_bars"].sum()
        ),
        "predecessor_determinism_passed": bool(
            cast(Mapping[str, Any], source["determinism"])["passed"]
        ),
        "predecessor_audit_passed": bool(cast(Mapping[str, Any], source["audit"])["passed"]),
        "source_hashes": source["hashes"],
    }
    passed = bool(
        int(result["row_identity_mismatches"]) == 0
        and assessment_row_mismatches == 0
        and int(result["checkpoint_timestamp_mismatches"]) == 0
        and int(result["split_mismatches"]) == 0
        and int(result["target_mismatches"]) == 0
        and float(result["maximum_weight_difference"]) <= 1e-12
        and float(result["maximum_feature_difference"]) <= 1e-12
        and float(result["maximum_probability_difference"]) <= 1e-12
        and metric_difference <= 1e-12
        and bool(result["predecessor_determinism_passed"])
        and bool(result["predecessor_audit_passed"])
    )
    result["passed"] = passed
    if not passed:
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "frozen predecessor panel or predictions differ",
        )
    return result


def build_fixed_lead_panel(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    canonical_routes: pd.DataFrame,
) -> pd.DataFrame:
    registered = ledger.loc[ledger["ledger_kind"].eq("registered_completion")].copy()
    prefixes = ledger.loc[
        ledger["ledger_kind"].eq("active_prefix")
        & ledger["bar_ordinal"].astype(int).isin(CHECKPOINTS)
    ].copy()
    if not set(prefixes["motif_type"].dropna().astype(str)).issubset(
        {"primitive", "repeat", "composite"}
    ):
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "unknown registered prefix motif",
        )
    prefixes = prefixes.drop_duplicates(
        ["symbol", "session", "bar_ordinal", "semantic_loop_id", "orientation_id"]
    )
    prefixes = prefixes.merge(
        canonical_routes,
        on=["semantic_loop_id", "orientation_id"],
        how="left",
        validate="many_to_one",
    )
    if prefixes["canonical_total_transitions"].isna().any():
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "active prefix orientation is absent from the frozen dictionary",
        )
    if (
        not prefixes["motif_type"]
        .astype(str)
        .eq(prefixes["dictionary_motif_type"].astype(str))
        .all()
    ):
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "active prefix motif differs from the frozen dictionary",
        )
    progress = prefixes["progress_states"].astype(int)
    declared_remaining = prefixes["transitions_remaining"].astype(int)
    prefixes["remaining_required_transitions"] = prefixes["canonical_total_transitions"].astype(
        int
    ) - (progress - 1)
    if (
        (progress < 1).any()
        or (declared_remaining < 0).any()
        or (prefixes["remaining_required_transitions"] < 0).any()
        or not prefixes["remaining_required_transitions"].eq(declared_remaining).all()
    ):
        raise ScreenBlocker(
            "blocked_prefix_proximity_reconstruction_failure",
            "declared prefix remainder differs from canonical route length",
        )
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
    proximity["any_prefix_one_transition_from_completion"] = (
        proximity["number_of_one_transition_away_prefixes"].gt(0).astype(int)
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
        result["any_prefix_one_transition_from_completion"].fillna(0).astype(int)
    )

    registered_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in registered.groupby(["symbol", "session"], sort=False)
    }
    empty_registered = registered.iloc[:0]
    leads: list[int] = []
    identities: list[str] = []
    for row in result.itertuples(index=False):
        events = registered_groups.get((str(row.symbol), str(row.session)), empty_registered)
        ordinals = events["bar_ordinal"].astype(int).tolist()
        lead = earliest_completion_lead(int(row.checkpoint), ordinals)
        leads.append(lead)
        if lead == 0:
            identities.append("[]")
        else:
            at_lead = events.loc[
                events["bar_ordinal"].astype(int).eq(int(row.checkpoint) + lead),
                "semantic_loop_id",
            ]
            identities.append(json.dumps(sorted(set(at_lead.astype(str)))))
    result["first_completion_lead"] = np.asarray(leads, dtype=int)
    result["first_completion_semantic_loop_ids"] = identities
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
            "fixed lead does not reproduce predecessor targets",
        )
    result = result.sort_values(
        ["period", "session", "checkpoint", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    reject_protected_dates(result, column="session")
    return result


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].to_numpy(float)
    weights = frame["row_weight"].to_numpy(float)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return float("nan")
    return float(np.average(values[finite], weights=weights[finite]))


def build_lead_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    support_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for lead in (0, 1, 2, 3):
        group = assessment.loc[assessment["first_completion_lead"].eq(lead)]
        one_away = float(group["any_prefix_one_transition_from_completion"].mean())
        support_rows.append(
            {
                "first_completion_lead": lead,
                "rows": len(group),
                "sessions": group["session"].nunique(),
                "stocks": group["symbol"].nunique(),
                "one_transition_away_rows": int(
                    group["any_prefix_one_transition_from_completion"].sum()
                ),
                "one_transition_away_prefix_prevalence": one_away,
            }
        )
        summary = {
            "record_type": "lead_summary",
            "first_completion_lead": lead,
            "route_resolution_state": "ALL",
            "lead_rows": len(group),
            "state_rows": len(group),
            "state_prevalence": 1.0,
            "sessions": group["session"].nunique(),
            "stocks": group["symbol"].nunique(),
            "mean_active_prefix_count": weighted_mean(group, "active_prefix_count"),
            "mean_top_prefix_depth": weighted_mean(group, "top_prefix_depth_fraction"),
            "mean_depth_margin": weighted_mean(group, "top_minus_second_prefix_depth"),
            "mean_prefix_family_entropy": weighted_mean(group, "prefix_family_entropy"),
            "mean_orientation_disagreement": weighted_mean(
                group, "orientation_disagreement_fraction"
            ),
            "mean_recent_loop_memory_weighted_top_depth": weighted_mean(
                group, "recent_loop_memory_weighted_top_depth"
            ),
            "one_transition_away_prefix_prevalence": one_away,
        }
        diagnostic_rows.append(summary)
        for state in ROUTE_STATES:
            state_group = group.loc[group["route_resolution_state"].eq(state)]
            diagnostic_rows.append(
                {
                    **summary,
                    "record_type": "route_state_prevalence",
                    "route_resolution_state": state,
                    "state_rows": len(state_group),
                    "state_prevalence": len(state_group) / len(group) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(support_rows), pd.DataFrame(diagnostic_rows)


def build_route_state_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    assessment = panel.loc[panel["period"].eq("assessment")]
    rows: list[dict[str, Any]] = []
    for state in ROUTE_STATES:
        full = assessment.loc[assessment["route_resolution_state"].eq(state)]
        advance = full.loc[full["advance_eligible"].eq(1)]
        positives = int(advance["completion_in_bars_2_or_3"].sum())
        rows.append(
            {
                "route_resolution_state": state,
                "all_rows": len(full),
                "advance_rows": len(advance),
                "sessions": advance["session"].nunique(),
                "stocks": advance["symbol"].nunique(),
                "bars_two_to_three_positive_outcomes": positives,
                "next_bar_completion_rate": (
                    float(np.average(full["completion_next_1_bar"], weights=full["row_weight"]))
                    if len(full)
                    else np.nan
                ),
                "clean_bars_two_to_three_completion_rate": (
                    float(
                        np.average(
                            advance["completion_in_bars_2_or_3"],
                            weights=advance["row_weight"],
                        )
                    )
                    if len(advance)
                    else np.nan
                ),
                "one_transition_away_prefix_frequency": (
                    float(full["any_prefix_one_transition_from_completion"].mean())
                    if len(full)
                    else np.nan
                ),
                "A1_minus_A0_log_loss_improvement": np.nan,
                "A1_minus_A0_brier_improvement": np.nan,
                "inference_supported": bool(len(advance) >= 100 and positives >= 25),
            }
        )
    return pd.DataFrame(rows)


def maximum_weighted_stock_share(frame: pd.DataFrame) -> float:
    total = float(frame["row_weight"].sum())
    if total <= 0.0:
        return float("nan")
    return float(frame.groupby("symbol")["row_weight"].sum().max() / total)


def support_and_concentration(
    panel: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, str | None]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    advance = assessment.loc[assessment["advance_eligible"].eq(1)]
    theoretical = theoretical_assessment_support(
        sessions=assessment["session"].nunique(),
        stocks=len(FROZEN_COHORT),
        checkpoints=len(CHECKPOINTS),
        retained_rows=len(assessment),
    )
    advance_features_finite = np.isfinite(advance.loc[:, list(H1_FEATURES)].to_numpy(float)).all(
        axis=1
    )
    support = {
        **theoretical,
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "next_bar_rows": len(assessment),
        "next_bar_positive_outcomes": int(assessment["completion_next_1_bar"].sum()),
        "next_bar_maximum_weighted_stock_share": maximum_weighted_stock_share(assessment),
        "advance_rows": len(advance),
        "advance_sessions": int(advance["session"].nunique()),
        "advance_stocks": int(advance["symbol"].nunique()),
        "advance_months": int(advance["year_month"].nunique()),
        "advance_positive_outcomes": int(advance["completion_in_bars_2_or_3"].sum()),
        "advance_feature_retention": float(advance_features_finite.mean()),
        "advance_maximum_weighted_stock_share": maximum_weighted_stock_share(advance),
        "next_bar_base_rate": float(
            np.average(assessment["completion_next_1_bar"], weights=assessment["row_weight"])
        ),
        "advance_base_rate": float(
            np.average(advance["completion_in_bars_2_or_3"], weights=advance["row_weight"])
        ),
    }
    gates = {
        "predecessor_rows": len(assessment) >= 24_000,
        "predecessor_retention": float(theoretical["retention"]) >= 0.95,
        "immediate_rows": len(assessment) >= 24_000,
        "immediate_sessions": assessment["session"].nunique() >= 140,
        "immediate_stocks": assessment["symbol"].nunique() >= 15,
        "immediate_months": assessment["year_month"].nunique() >= 8,
        "immediate_positives": int(assessment["completion_next_1_bar"].sum()) >= 300,
        "immediate_concentration": support["next_bar_maximum_weighted_stock_share"] <= 0.10,
        "advance_rows": len(advance) >= 18_000,
        "advance_sessions": advance["session"].nunique() >= 140,
        "advance_stocks": advance["symbol"].nunique() >= 15,
        "advance_months": advance["year_month"].nunique() >= 8,
        "advance_positives": int(advance["completion_in_bars_2_or_3"].sum()) >= 250,
        "advance_feature_retention": support["advance_feature_retention"] >= 0.90,
        "advance_concentration": support["advance_maximum_weighted_stock_share"] <= 0.10,
    }
    blocker: str | None = None
    if not gates["predecessor_rows"] or not gates["predecessor_retention"]:
        blocker = "blocked_predecessor_reconstruction_failure"
    elif not all(gates[key] for key in gates if key.startswith("immediate_")):
        blocker = "blocked_insufficient_immediate_support"
    elif not all(
        gates[key]
        for key in (
            "advance_rows",
            "advance_sessions",
            "advance_stocks",
            "advance_months",
            "advance_feature_retention",
            "advance_concentration",
        )
    ):
        blocker = "blocked_insufficient_advance_support"
    elif not gates["advance_positives"]:
        blocker = "blocked_insufficient_advance_positive_support"
    support["gates"] = gates
    support["passed"] = blocker is None
    concentration = pd.DataFrame(
        [
            {
                "population": "predecessor",
                "metric": "retained_rows",
                "value": len(assessment),
                "threshold": 24_000,
                "passed": gates["predecessor_rows"],
            },
            {
                "population": "predecessor",
                "metric": "theoretical_population_retention",
                "value": theoretical["retention"],
                "threshold": 0.95,
                "passed": gates["predecessor_retention"],
            },
            {
                "population": "immediate",
                "metric": "positive_outcomes",
                "value": support["next_bar_positive_outcomes"],
                "threshold": 300,
                "passed": gates["immediate_positives"],
            },
            {
                "population": "immediate",
                "metric": "maximum_weighted_stock_share",
                "value": support["next_bar_maximum_weighted_stock_share"],
                "threshold": 0.10,
                "passed": gates["immediate_concentration"],
            },
            {
                "population": "advance",
                "metric": "rows",
                "value": support["advance_rows"],
                "threshold": 18_000,
                "passed": gates["advance_rows"],
            },
            {
                "population": "advance",
                "metric": "positive_outcomes",
                "value": support["advance_positive_outcomes"],
                "threshold": 250,
                "passed": gates["advance_positives"],
            },
            {
                "population": "advance",
                "metric": "feature_retention",
                "value": support["advance_feature_retention"],
                "threshold": 0.90,
                "passed": gates["advance_feature_retention"],
            },
            {
                "population": "advance",
                "metric": "maximum_weighted_stock_share",
                "value": support["advance_maximum_weighted_stock_share"],
                "threshold": 0.10,
                "passed": gates["advance_concentration"],
            },
        ]
    )
    return support, concentration, blocker


def pair_increments(
    frame: pd.DataFrame,
    *,
    baseline_model: str,
    route_model: str,
    target: str,
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    baseline = model_metrics(
        frame,
        model=baseline_model,
        target=target,
        boundaries=boundaries[baseline_model],
    )
    route = model_metrics(
        frame,
        model=route_model,
        target=target,
        boundaries=boundaries[route_model],
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
    result = panel.copy()
    specifications: dict[str, Any] = {}
    boundaries: dict[str, dict[str, float]] = {}
    definitions = {
        "N0": ("completion_next_1_bar", BASELINE_FEATURES, False),
        "N1": ("completion_next_1_bar", H1_FEATURES, False),
        "A0": ("completion_in_bars_2_or_3", BASELINE_FEATURES, True),
        "A1": ("completion_in_bars_2_or_3", H1_FEATURES, True),
    }
    for model, (target, features, advance_only) in definitions.items():
        population = result.loc[result["advance_eligible"].eq(1)] if advance_only else result
        development = population.loc[population["period"].eq("development")]
        try:
            fitted = fit_hazard_model(development, features=features, target=target)
        except ValueError as error:
            raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
        result.loc[population.index, f"{model}_probability"] = fitted.predict_probability(
            population
        )
        specifications[model] = {
            **fitted.as_dict(),
            "target": target,
            "population": "advance_eligible" if advance_only else "all_resolvable_rows",
        }
        development_probability = result.loc[development.index, f"{model}_probability"]
        boundaries[model] = {
            "top_decile": float(development_probability.quantile(0.90)),
            "top_quintile": float(development_probability.quantile(0.80)),
        }
    return result, specifications, boundaries


def build_model_tables(
    panel: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    advance = assessment.loc[assessment["advance_eligible"].eq(1)]
    immediate = pd.DataFrame(
        [
            model_metrics(
                assessment,
                model=model,
                target="completion_next_1_bar",
                boundaries=boundaries[model],
            )
            for model in ("N0", "N1")
        ]
    )
    advance_metrics = pd.DataFrame(
        [
            model_metrics(
                advance,
                model=model,
                target="completion_in_bars_2_or_3",
                boundaries=boundaries[model],
            )
            for model in ("A0", "A1")
        ]
    )
    checkpoint_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    for population_name, population, models, target in (
        ("immediate", assessment, ("N0", "N1"), "completion_next_1_bar"),
        ("advance", advance, ("A0", "A1"), "completion_in_bars_2_or_3"),
    ):
        for checkpoint, group in population.groupby("checkpoint", sort=True):
            for model in models:
                checkpoint_rows.append(
                    {
                        "population": population_name,
                        "checkpoint": int(checkpoint),
                        **model_metrics(
                            group,
                            model=model,
                            target=target,
                            boundaries=boundaries[model],
                        ),
                    }
                )
        for month, group in population.groupby("year_month", sort=True):
            for model in models:
                monthly_rows.append(
                    {
                        "population": population_name,
                        "year_month": str(month),
                        **model_metrics(
                            group,
                            model=model,
                            target=target,
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
            for value, group in population.groupby(dimension, sort=True):
                for model in models:
                    subgroup_rows.append(
                        {
                            "population": population_name,
                            "subgroup_dimension": dimension,
                            "subgroup_value": str(value),
                            **model_metrics(
                                group,
                                model=model,
                                target=target,
                                boundaries=boundaries[model],
                            ),
                        }
                    )
    return (
        immediate,
        advance_metrics,
        pd.DataFrame(checkpoint_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(subgroup_rows),
    )


def run_bootstrap(
    assessment: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    multiplicities = session_bootstrap_multiplicities(
        assessment["session"], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    point_estimates = {
        "immediate": pair_increments(
            assessment,
            baseline_model="N0",
            route_model="N1",
            target="completion_next_1_bar",
            boundaries=boundaries,
        ),
        "advance": pair_increments(
            assessment.loc[assessment["advance_eligible"].eq(1)],
            baseline_model="A0",
            route_model="A1",
            target="completion_in_bars_2_or_3",
            boundaries=boundaries,
        ),
    }
    for draw, multiplicity in enumerate(multiplicities):
        sampled = assessment.copy()
        sampled["row_weight"] *= multiplicity
        sampled = sampled.loc[sampled["row_weight"].gt(0)]
        for comparison, baseline, route, target, statistics in (
            (
                "immediate",
                "N0",
                "N1",
                "completion_next_1_bar",
                (
                    "log_loss_improvement",
                    "brier_improvement",
                    "auc_improvement",
                    "average_precision_improvement",
                ),
            ),
            (
                "advance",
                "A0",
                "A1",
                "completion_in_bars_2_or_3",
                (
                    "log_loss_improvement",
                    "brier_improvement",
                    "auc_improvement",
                    "average_precision_improvement",
                    "top_decile_precision_improvement",
                ),
            ),
        ):
            population = (
                sampled.loc[sampled["advance_eligible"].eq(1)]
                if comparison == "advance"
                else sampled
            )
            increments = pair_increments(
                population,
                baseline_model=baseline,
                route_model=route,
                target=target,
                boundaries=boundaries,
            )
            for statistic in statistics:
                rows.append(
                    {
                        "record_type": "draw",
                        "comparison": comparison,
                        "draw": draw,
                        "statistic": statistic,
                        "value": increments[statistic],
                        "interval_level": np.nan,
                        "lower": np.nan,
                        "upper": np.nan,
                        "point_estimate": point_estimates[comparison][statistic],
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
                    "comparison": comparison,
                    "draw": np.nan,
                    "statistic": statistic,
                    "value": np.nan,
                    "interval_level": level,
                    "lower": float(np.quantile(values, alpha / 2.0)),
                    "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
                    "point_estimate": point_estimates[str(comparison)][str(statistic)],
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
    rows: list[dict[str, Any]] = []
    specifications: dict[str, Any] = {}
    assessment = result.loc[result["period"].eq("assessment")]
    comparisons = (
        ("immediate", "N0", "N1", "completion_next_1_bar", False),
        ("advance", "A0", "A1", "completion_in_bars_2_or_3", True),
    )
    for comparison, baseline, route, target, advance_only in comparisons:
        population = result.loc[result["advance_eligible"].eq(1)] if advance_only else result
        real_assessment = (
            assessment.loc[assessment["advance_eligible"].eq(1)] if advance_only else assessment
        )
        real = pair_increments(
            real_assessment,
            baseline_model=baseline,
            route_model=route,
            target=target,
            boundaries=boundaries,
        )
        for draw in range(NULL_DRAWS):
            seed = NULL_SEEDS[comparison] + draw
            permuted = permute_route_bundle(
                population,
                route_features=ROUTE_FEATURES,
                strata=("period", "session", "checkpoint"),
                seed=seed,
            )
            development = permuted.loc[permuted["period"].eq("development")]
            try:
                fitted = fit_hazard_model(development, features=H1_FEATURES, target=target)
            except ValueError as error:
                raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
            permuted_assessment = permuted.loc[permuted["period"].eq("assessment")]
            probability = fitted.predict_probability(permuted_assessment)
            column = f"{comparison}_route_null_{draw}_probability"
            result.loc[permuted_assessment.index, column] = probability
            null_frame = real_assessment.copy()
            null_frame[f"{route}_probability"] = probability
            null_increment = pair_increments(
                null_frame,
                baseline_model=baseline,
                route_model=route,
                target=target,
                boundaries=boundaries,
            )
            rows.append(
                {
                    "record_type": "draw",
                    "comparison": comparison,
                    "draw": draw,
                    "seed": seed,
                    "route_bundle_hash": stable_frame_hash(
                        permuted,
                        ["period", "session", "checkpoint", "symbol", *ROUTE_FEATURES],
                    ),
                    **{
                        key: null_increment[key]
                        for key in (
                            "log_loss_improvement",
                            "brier_improvement",
                            "auc_improvement",
                        )
                    },
                }
            )
            specifications[f"{comparison}_{draw}"] = {
                **fitted.as_dict(),
                "target": target,
                "seed": seed,
            }
        draw_rows = [row for row in rows if row["comparison"] == comparison]
        for statistic in (
            "log_loss_improvement",
            "brier_improvement",
            "auc_improvement",
        ):
            rows.append(
                {
                    "record_type": "comparison",
                    "comparison": comparison,
                    "draw": np.nan,
                    "seed": np.nan,
                    "route_bundle_hash": None,
                    "statistic": statistic,
                    "real_increment": real[statistic],
                    "real_exceeds_null_count": sum(
                        real[statistic] > float(row[statistic]) for row in draw_rows
                    ),
                    "null_draws": NULL_DRAWS,
                }
            )
    return result, pd.DataFrame(rows), specifications


def empty_metric_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=METRIC_COLUMNS)


def empty_breakdown_frame(key_columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*key_columns, *METRIC_COLUMNS])


def blocked_determinism_check(
    output: Path,
    predecessor_panel: pd.DataFrame,
    ledger: pd.DataFrame,
    canonical_routes: pd.DataFrame,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    stored = pd.read_parquet(output / "fixed_lead_panel.parquet")
    regenerated = build_fixed_lead_panel(predecessor_panel, ledger, canonical_routes)
    ids_left = stored["row_id"].astype(str).tolist()
    ids_right = regenerated["row_id"].astype(str).tolist()
    row_mismatches = abs(len(ids_left) - len(ids_right)) + sum(
        left != right for left, right in zip(ids_left, ids_right, strict=False)
    )
    lead_mismatches = int(
        (
            stored["first_completion_lead"].to_numpy(int)
            != regenerated["first_completion_lead"].to_numpy(int)
        ).sum()
    )
    eligibility_mismatches = int(
        (
            stored["advance_eligible"].to_numpy(int)
            != regenerated["advance_eligible"].to_numpy(int)
        ).sum()
    )
    proximity_mismatches = int(
        (
            stored["number_of_one_transition_away_prefixes"].to_numpy(int)
            != regenerated["number_of_one_transition_away_prefixes"].to_numpy(int)
        ).sum()
    )
    maximum_feature_difference = float(
        np.max(
            np.abs(
                stored.loc[:, [*BASELINE_FEATURES, *ROUTE_FEATURES]].to_numpy(float)
                - regenerated.loc[:, [*BASELINE_FEATURES, *ROUTE_FEATURES]].to_numpy(float)
            )
        )
    )
    stored_decision = read_json(output / "decision.json")
    _, _, regenerated_blocker = support_and_concentration(regenerated)
    regenerated_decision = choose_fixed_lead_decision(
        blocker=regenerated_blocker,
        immediate_passed=False,
        advance_passed=False,
        descriptive_lead_structure=False,
        baseline_meaningful=False,
    )
    final_decision_match = (
        stored_decision["primary_decision"] == decision["primary_decision"] == regenerated_decision
    )
    passed = bool(
        row_mismatches == 0
        and lead_mismatches == 0
        and eligibility_mismatches == 0
        and proximity_mismatches == 0
        and maximum_feature_difference <= 1e-12
        and final_decision_match
    )
    return {
        **SAFETY_FLAGS,
        "pre_model_support_blocker": True,
        "models_refit": [],
        "bootstrap_repeated": False,
        "route_null_refits_repeated": False,
        "row_identity_mismatches": row_mismatches,
        "lead_label_mismatches": lead_mismatches,
        "advance_eligibility_mismatches": eligibility_mismatches,
        "prefix_proximity_mismatches": proximity_mismatches,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_probability_difference": None,
        "probability_check_status": "not_applicable_pre_model_support_stop",
        "probability_comparison_rows": 0,
        "regenerated_primary_decision": regenerated_decision,
        "final_decision_match": final_decision_match,
        "passed": passed,
    }


def build_report(
    decision: Mapping[str, Any],
    lead_support: pd.DataFrame,
    route_states: pd.DataFrame,
    immediate_metrics: pd.DataFrame,
    advance_metrics: pd.DataFrame,
    *,
    audit: Mapping[str, Any] | None = None,
) -> str:
    support = cast(Mapping[str, Any], decision["support"])
    lines = [
        "# Route-Competition Fixed-Lead Audit Quick Screen V0.1",
        "",
        (
            "Retrospective, observable, structural, research-only fixed-lead audit. "
            "No economic, directional, options, execution, broker, or strategy outcome was opened."
        ),
        "",
        f"Primary decision: `{decision['primary_decision']}`.",
        "",
        "## Support",
        "",
        (
            f"Theoretical assessment population: {int(support['theoretical_eligible_rows'])}; "
            f"reconstructed rows: {int(support['retained_rows'])}; retention: "
            f"{float(support['retention']):.6f}."
        ),
        (
            f"Next-bar positives: {int(support['next_bar_positive_outcomes'])}; advance-eligible "
            f"rows: {int(support['advance_rows'])}; clean bars-two-to-three positives: "
            f"{int(support['advance_positive_outcomes'])}."
        ),
        "",
        "## Lead decomposition",
        "",
    ]
    for row in lead_support.itertuples(index=False):
        lines.append(
            f"- Lead {int(row.first_completion_lead)}: rows {int(row.rows)}, sessions "
            f"{int(row.sessions)}, one-transition-away prevalence "
            f"{float(row.one_transition_away_prefix_prevalence):.6f}."
        )
    lines.extend(["", "## Route-resolution states", ""])
    for row in route_states.itertuples(index=False):
        lines.append(
            f"- {row.route_resolution_state}: next-bar rate "
            f"{float(row.next_bar_completion_rate):.6f}; clean two-to-three-bar rate "
            f"{float(row.clean_bars_two_to_three_completion_rate):.6f}; advance rows "
            f"{int(row.advance_rows)}; positives {int(row.bars_two_to_three_positive_outcomes)}."
        )
    if decision.get("blocker") is not None:
        lines.extend(
            [
                "",
                (
                    "The preregistered minimum was 250 clean advance positives. The observed "
                    "support was lower, so the experiment stopped before fitting N0/N1/A0/A1, "
                    "bootstrapping, or route-null refitting."
                ),
            ]
        )
    else:
        lines.extend(["", "## Fixed-lead model results", ""])
        for population, metrics in (
            ("Immediate", immediate_metrics),
            ("Clean advance", advance_metrics),
        ):
            for row in metrics.itertuples(index=False):
                lines.append(
                    f"- {population} {row.model}: log loss {float(row.log_loss):.8f}; "
                    f"Brier {float(row.brier_score):.8f}; AUC {float(row.auc):.8f}; "
                    f"average precision {float(row.average_precision):.8f}."
                )
        immediate_increment = cast(Mapping[str, Any], decision["immediate_increments"])
        advance_increment = cast(Mapping[str, Any], decision["advance_increments"])
        lines.extend(
            [
                "",
                (
                    "N1-minus-N0 improvements: log loss "
                    f"{float(immediate_increment['log_loss_improvement']):.8f}; Brier "
                    f"{float(immediate_increment['brier_improvement']):.8f}; AUC "
                    f"{float(immediate_increment['auc_improvement']):.8f}."
                ),
                (
                    "A1-minus-A0 improvements: log loss "
                    f"{float(advance_increment['log_loss_improvement']):.8f}; Brier "
                    f"{float(advance_increment['brier_improvement']):.8f}; AUC "
                    f"{float(advance_increment['auc_improvement']):.8f}; average precision "
                    f"{float(advance_increment['average_precision_improvement']):.8f}."
                ),
            ]
        )
    if audit is not None:
        lines.append(f"Independent audit passed={bool(audit['passed'])}.")
    lines.extend(
        [
            "",
            (
                "This is not prospective validation and supplies no evidence of economic value, "
                "directional edge, options edge, trading utility, or deployability."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def finalize_report(output: Path, *, audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    decision = read_json(output / "decision.json")
    report = build_report(
        decision,
        pd.read_csv(output / "lead_support.csv"),
        pd.read_csv(output / "route_resolution_state_metrics.csv"),
        pd.read_csv(output / "immediate_metrics.csv"),
        pd.read_csv(output / "advance_metrics.csv"),
        audit=audit,
    )
    primary = output / "report.md"
    copy = REPORTS_DIR / "report.md"
    primary.write_text(report, encoding="utf-8")
    copy.write_text(report, encoding="utf-8")
    if sha256_file(primary) != sha256_file(copy):
        raise ScreenBlocker("blocked_reproducibility_or_audit_failure", "report copies differ")
    return {"sha256": sha256_file(primary), "copies_match": True}


def write_blocked_model_artifacts(
    output: Path,
    panel: pd.DataFrame,
    blocker: str,
) -> None:
    for model in ("N0", "N1", "A0", "A1"):
        panel[f"{model}_probability"] = np.nan
    write_parquet(
        output / "assessment_predictions.parquet",
        panel.loc[panel["period"].eq("assessment")].reset_index(drop=True),
    )
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "planned_primary_models": ["N0", "N1", "A0", "A1"],
            "primary_models_fitted": 0,
            "planned_bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_draws_executed": 0,
            "planned_route_null_refits": {"immediate": 3, "advance": 3},
            "route_null_refits_executed": {"immediate": 0, "advance": 0},
            "stop_reason": blocker,
            "H0_features": list(BASELINE_FEATURES),
            "H1_features": list(H1_FEATURES),
        },
    )
    write_json(
        output / "model_coefficients.json",
        {
            **SAFETY_FLAGS,
            "primary_models": {},
            "route_null_models": {},
            "models_fitted": 0,
            "stop_reason": blocker,
        },
    )
    write_csv(output / "immediate_metrics.csv", empty_metric_frame())
    write_csv(output / "advance_metrics.csv", empty_metric_frame())
    write_csv(
        output / "checkpoint_metrics.csv", empty_breakdown_frame(("population", "checkpoint"))
    )
    write_csv(output / "monthly_metrics.csv", empty_breakdown_frame(("population", "year_month")))
    write_csv(
        output / "subgroup_metrics.csv",
        empty_breakdown_frame(("population", "subgroup_dimension", "subgroup_value")),
    )
    write_csv(
        output / "bootstrap_metrics.csv",
        pd.DataFrame(
            columns=(
                "record_type",
                "comparison",
                "draw",
                "statistic",
                "value",
                "interval_level",
                "lower",
                "upper",
                "point_estimate",
                "draws",
                "seed",
            )
        ),
    )
    write_csv(
        output / "route_null_metrics.csv",
        pd.DataFrame(
            columns=(
                "record_type",
                "comparison",
                "draw",
                "seed",
                "route_bundle_hash",
                "log_loss_improvement",
                "brier_improvement",
                "auc_improvement",
                "statistic",
                "real_increment",
                "real_exceeds_null_count",
                "null_draws",
            )
        ),
    )


def _bootstrap_lower(
    bootstrap: pd.DataFrame, comparison: str, statistic: str, level: float
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
            f"bootstrap interval missing: {comparison} {statistic}",
        )
    return float(row.iloc[0]["lower"])


def stability_gate_values(
    metrics: pd.DataFrame, *, population: str, baseline: str, route: str
) -> tuple[int, int]:
    subset = metrics.loc[metrics["population"].eq(population)]
    group_columns = [column for column in ("year_month", "checkpoint") if column in subset.columns]
    if len(group_columns) != 1:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "stability grouping differs"
        )
    key = group_columns[0]
    positive = 0
    adverse = 0
    for _, group in subset.groupby(key, sort=True):
        left = group.loc[group["model"].eq(baseline)].iloc[0]
        right = group.loc[group["model"].eq(route)].iloc[0]
        log_improvement = float(left["log_loss"]) - float(right["log_loss"])
        brier_improvement = float(left["brier_score"]) - float(right["brier_score"])
        positive += int(log_improvement > 0.0)
        adverse += int(log_improvement < -0.005 or brier_improvement < -0.002)
    return positive, adverse


def null_exceeds_all(route_null: pd.DataFrame, comparison: str) -> bool:
    rows = route_null.loc[
        route_null["record_type"].eq("comparison")
        & route_null["comparison"].eq(comparison)
        & route_null["statistic"].isin(["log_loss_improvement", "brier_improvement"])
    ]
    return bool(len(rows) == 2 and rows["real_exceeds_null_count"].astype(int).eq(NULL_DRAWS).any())


def update_route_state_increments(
    route_states: pd.DataFrame,
    assessment: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    result = route_states.copy()
    for index, row in result.iterrows():
        group = assessment.loc[
            assessment["advance_eligible"].eq(1)
            & assessment["route_resolution_state"].eq(row["route_resolution_state"])
        ]
        if group.empty:
            continue
        increments = pair_increments(
            group,
            baseline_model="A0",
            route_model="A1",
            target="completion_in_bars_2_or_3",
            boundaries=boundaries,
        )
        result.loc[index, "A1_minus_A0_log_loss_improvement"] = increments["log_loss_improvement"]
        result.loc[index, "A1_minus_A0_brier_improvement"] = increments["brier_improvement"]
    return result


def plot_supported_screen(
    immediate: pd.DataFrame,
    advance: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values: list[float] = []
    labels: list[str] = []
    for metrics, baseline, route, prefix in (
        (immediate, "N0", "N1", "N"),
        (advance, "A0", "A1", "A"),
    ):
        indexed = metrics.set_index("model")
        values.extend(
            [
                float(indexed.loc[baseline, "log_loss"]) - float(indexed.loc[route, "log_loss"]),
                float(indexed.loc[baseline, "brier_score"])
                - float(indexed.loc[route, "brier_score"]),
                float(indexed.loc[route, "average_precision"])
                - float(indexed.loc[baseline, "average_precision"]),
            ]
        )
        labels.extend([f"{prefix} log", f"{prefix} Brier", f"{prefix} AP"])
    fig, axis = plt.subplots(figsize=(9.0, 4.5))
    axis.bar(labels, values, color=["#4c78a8"] * 3 + ["#f58518"] * 3)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Route-model increment")
    axis.set_title("Immediate versus clean advance route increments")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def supported_determinism_check(
    output: Path,
    predecessor_panel: pd.DataFrame,
    ledger: pd.DataFrame,
    canonical_routes: pd.DataFrame,
    coefficients: Mapping[str, Any],
    stored_decision: Mapping[str, Any],
) -> dict[str, Any]:
    stored = pd.read_parquet(output / "fixed_lead_panel.parquet")
    labels = build_fixed_lead_panel(predecessor_panel, ledger, canonical_routes)
    stored_ids = stored["row_id"].astype(str).tolist()
    regenerated_ids = labels["row_id"].astype(str).tolist()
    row_mismatches = abs(len(stored_ids) - len(regenerated_ids)) + sum(
        left != right for left, right in zip(stored_ids, regenerated_ids, strict=False)
    )
    lead_mismatches = int(
        (
            stored["first_completion_lead"].to_numpy(int)
            != labels["first_completion_lead"].to_numpy(int)
        ).sum()
    )
    eligibility_mismatches = int(
        (stored["advance_eligible"].to_numpy(int) != labels["advance_eligible"].to_numpy(int)).sum()
    )
    proximity_mismatches = int(
        (
            stored["number_of_one_transition_away_prefixes"].to_numpy(int)
            != labels["number_of_one_transition_away_prefixes"].to_numpy(int)
        ).sum()
    )
    maximum_feature_difference = float(
        np.max(
            np.abs(
                stored.loc[:, [*BASELINE_FEATURES, *ROUTE_FEATURES]].to_numpy(float)
                - labels.loc[:, [*BASELINE_FEATURES, *ROUTE_FEATURES]].to_numpy(float)
            )
        )
    )
    maximum_probability_difference = 0.0
    maximum_coefficient_difference = 0.0
    definitions = {
        "N0": ("completion_next_1_bar", BASELINE_FEATURES, False),
        "N1": ("completion_next_1_bar", H1_FEATURES, False),
        "A0": ("completion_in_bars_2_or_3", BASELINE_FEATURES, True),
        "A1": ("completion_in_bars_2_or_3", H1_FEATURES, True),
    }
    for model, (target, features, advance_only) in definitions.items():
        population = stored.loc[stored["advance_eligible"].eq(1)] if advance_only else stored
        development = population.loc[population["period"].eq("development")]
        refit = fit_hazard_model(development, features=features, target=target)
        regenerated = refit.predict_probability(population)
        maximum_probability_difference = max(
            maximum_probability_difference,
            float(np.max(np.abs(regenerated - population[f"{model}_probability"].to_numpy(float)))),
        )
        serialized = cast(Mapping[str, Any], coefficients["primary_models"])[model]
        maximum_coefficient_difference = max(
            maximum_coefficient_difference,
            float(
                np.max(
                    np.abs(
                        np.asarray(refit.as_dict()["coefficient"], dtype=float)
                        - np.asarray(serialized["coefficient"], dtype=float)
                    )
                )
            ),
            abs(float(refit.as_dict()["intercept"]) - float(serialized["intercept"])),
        )
    configurations = read_json(output / "model_configurations.json")
    boundaries = cast(
        Mapping[str, Mapping[str, float]], configurations["probability_quantile_boundaries"]
    )
    assessment = stored.loc[stored["period"].eq("assessment")]
    assessment_advance = assessment.loc[assessment["advance_eligible"].eq(1)]
    regenerated_immediate = pd.DataFrame(
        [
            model_metrics(
                assessment,
                model=model,
                target="completion_next_1_bar",
                boundaries=boundaries[model],
            )
            for model in ("N0", "N1")
        ]
    )
    regenerated_advance = pd.DataFrame(
        [
            model_metrics(
                assessment_advance,
                model=model,
                target="completion_in_bars_2_or_3",
                boundaries=boundaries[model],
            )
            for model in ("A0", "A1")
        ]
    )
    metric_columns = (
        "log_loss",
        "brier_score",
        "auc",
        "average_precision",
        "expected_calibration_error",
        "calibration_intercept",
        "calibration_slope",
        "base_rate",
        "mean_probability_realised_class",
        "top_decile_precision",
        "top_decile_lift",
        "top_quintile_precision",
        "top_quintile_lift",
    )
    maximum_metric_difference = 0.0
    for regenerated, stored_path in (
        (regenerated_immediate, output / "immediate_metrics.csv"),
        (regenerated_advance, output / "advance_metrics.csv"),
    ):
        archived = pd.read_csv(stored_path).set_index("model")
        for row in regenerated.itertuples(index=False):
            for column in metric_columns:
                left = float(getattr(row, column))
                right = float(archived.loc[str(row.model), column])
                difference = 0.0 if np.isnan(left) and np.isnan(right) else abs(left - right)
                maximum_metric_difference = max(maximum_metric_difference, difference)

    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    monthly = pd.read_csv(output / "monthly_metrics.csv")
    checkpoint = pd.read_csv(output / "checkpoint_metrics.csv")
    route_null = pd.read_csv(output / "route_null_metrics.csv")
    immediate_increments = pair_increments(
        assessment,
        baseline_model="N0",
        route_model="N1",
        target="completion_next_1_bar",
        boundaries=boundaries,
    )
    advance_increments = pair_increments(
        assessment_advance,
        baseline_model="A0",
        route_model="A1",
        target="completion_in_bars_2_or_3",
        boundaries=boundaries,
    )
    immediate_positive_months, _ = stability_gate_values(
        monthly, population="immediate", baseline="N0", route="N1"
    )
    advance_positive_months, _ = stability_gate_values(
        monthly, population="advance", baseline="A0", route="A1"
    )
    _, immediate_adverse = stability_gate_values(
        checkpoint, population="immediate", baseline="N0", route="N1"
    )
    _, advance_adverse = stability_gate_values(
        checkpoint, population="advance", baseline="A0", route="A1"
    )
    immediate_gates: dict[str, object] = {
        **immediate_increments,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(
            bootstrap, "immediate", "log_loss_improvement", 0.80
        ),
        "bootstrap_80_brier_lower": _bootstrap_lower(
            bootstrap, "immediate", "brier_improvement", 0.80
        ),
        "positive_months": immediate_positive_months,
        "materially_adverse_checkpoints": immediate_adverse,
        "real_exceeds_all_nulls": null_exceeds_all(route_null, "immediate"),
        "support_and_concentration_passed": True,
    }
    advance_gates: dict[str, object] = {
        **advance_increments,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(
            bootstrap, "advance", "log_loss_improvement", 0.80
        ),
        "bootstrap_80_brier_lower": _bootstrap_lower(
            bootstrap, "advance", "brier_improvement", 0.80
        ),
        "bootstrap_80_average_precision_lower": _bootstrap_lower(
            bootstrap, "advance", "average_precision_improvement", 0.80
        ),
        "positive_months": advance_positive_months,
        "materially_adverse_checkpoints": advance_adverse,
        "real_exceeds_all_nulls": null_exceeds_all(route_null, "advance"),
        "support_and_concentration_passed": True,
    }
    immediate_passed = fixed_lead_increment_passes(immediate_gates, require_average_precision=False)
    advance_passed = fixed_lead_increment_passes(advance_gates, require_average_precision=True)
    route_states = pd.read_csv(output / "route_resolution_state_metrics.csv")
    supported_state_rates = route_states.loc[
        route_states["inference_supported"], "clean_bars_two_to_three_completion_rate"
    ].dropna()
    descriptive = bool(
        len(supported_state_rates) >= 2
        and float(supported_state_rates.max()) > float(supported_state_rates.min())
    )
    a0 = regenerated_advance.set_index("model").loc["A0"]
    base = float(a0["base_rate"])
    constant_log_loss = float(
        -np.average(
            assessment_advance["completion_in_bars_2_or_3"] * math.log(max(base, 1e-12))
            + (1 - assessment_advance["completion_in_bars_2_or_3"])
            * math.log(max(1.0 - base, 1e-12)),
            weights=assessment_advance["row_weight"],
        )
    )
    baseline_meaningful = bool(float(a0["auc"]) > 0.5 and float(a0["log_loss"]) < constant_log_loss)
    regenerated_decision = choose_fixed_lead_decision(
        blocker=None,
        immediate_passed=immediate_passed,
        advance_passed=advance_passed,
        descriptive_lead_structure=descriptive,
        baseline_meaningful=baseline_meaningful,
    )
    final_decision_match = regenerated_decision == stored_decision["primary_decision"]
    passed = bool(
        maximum_probability_difference <= 1e-12
        and maximum_coefficient_difference <= 1e-12
        and maximum_feature_difference <= 1e-12
        and maximum_metric_difference <= 1e-12
        and row_mismatches == 0
        and lead_mismatches == 0
        and eligibility_mismatches == 0
        and proximity_mismatches == 0
        and final_decision_match
    )
    return {
        **SAFETY_FLAGS,
        "models_refit": ["N0", "N1", "A0", "A1"],
        "bootstrap_repeated": False,
        "route_null_refits_repeated": False,
        "row_identity_mismatches": row_mismatches,
        "lead_label_mismatches": lead_mismatches,
        "advance_eligibility_mismatches": eligibility_mismatches,
        "prefix_proximity_mismatches": proximity_mismatches,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_coefficient_difference": maximum_coefficient_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "regenerated_primary_decision": regenerated_decision,
        "final_decision_match": final_decision_match,
        "passed": passed,
    }


def execute_screen(output: Path) -> dict[str, Any]:
    contract = load_contract()
    output.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    source = load_predecessor()
    predecessor_panel = cast(pd.DataFrame, source["panel"])
    ledger = cast(pd.DataFrame, source["ledger"])
    canonical_routes, dictionary_manifest = load_canonical_route_metadata(source)
    predecessor = verify_predecessor(source, predecessor_panel.copy())
    panel = build_fixed_lead_panel(predecessor_panel, ledger, canonical_routes)
    surface_after_augmentation = predecessor_surface_differences(
        predecessor_panel,
        panel,
        feature_columns=(*BASELINE_FEATURES, *ROUTE_FEATURES),
    )
    if any(
        float(surface_after_augmentation[key]) > 1e-12
        for key in (
            "maximum_weight_difference",
            "maximum_feature_difference",
            "maximum_probability_difference",
        )
    ) or any(
        int(surface_after_augmentation[key]) != 0
        for key in (
            "row_identity_mismatches",
            "checkpoint_timestamp_mismatches",
            "split_mismatches",
            "target_mismatches",
        )
    ):
        raise ScreenBlocker(
            "blocked_predecessor_reconstruction_failure",
            "fixed-lead augmentation altered the predecessor surface",
        )
    support, concentration, blocker = support_and_concentration(panel)
    lead_support, lead_diagnostics = build_lead_tables(panel)
    route_states = build_route_state_metrics(panel)
    protected_start = pd.Timestamp("2025-08-23T00:00:00Z")
    panel_timestamps = pd.to_datetime(panel["checkpoint_timestamp_utc"], utc=True, errors="raise")
    structural_timestamps = pd.to_datetime(
        ledger["available_timestamp_utc"], utc=True, errors="raise"
    )
    protected_decision_rows = int(panel_timestamps.ge(protected_start).sum())
    protected_structural_rows = int(structural_timestamps.ge(protected_start).sum())
    protected_rows_materialised = protected_decision_rows + protected_structural_rows
    protected = {
        **SAFETY_FLAGS,
        "development_start": DEVELOPMENT_START,
        "development_end_inclusive": "2024-12-31",
        "assessment_start": ASSESSMENT_START,
        "assessment_end_inclusive": READ_END,
        "protected_start": str(protected_start),
        "minimum_decision_session": str(panel["session"].min()),
        "maximum_decision_session": str(panel["session"].max()),
        "maximum_decision_timestamp": str(panel_timestamps.max()),
        "maximum_structural_available_timestamp": str(structural_timestamps.max()),
        "protected_decision_rows_materialised": protected_decision_rows,
        "protected_structural_rows_materialised": protected_structural_rows,
        "protected_rows_materialised": protected_rows_materialised,
        "passed": protected_rows_materialised == 0,
    }
    if not protected["passed"]:
        raise ScreenBlocker("blocked_protected_boundary_failure", "protected row materialised")
    source_manifest = {
        **SAFETY_FLAGS,
        "predecessor_experiment": str(PREDECESSOR_DIR.relative_to(REPO_ROOT)),
        "predecessor_commit": EXPECTED_PREDECESSOR_COMMIT,
        "predecessor_artifact_hashes": source["hashes"],
        "semantic_loop_dictionary": dictionary_manifest,
        "dates_read": {"start": DEVELOPMENT_START, "end_inclusive": READ_END},
        "frozen_audited_cohort": list(FROZEN_COHORT),
        "checkpoints": list(CHECKPOINTS),
        "raw_data_downloaded": False,
        "market_rows_materialised_from_predecessor_only": True,
        "protected_rows_materialised": protected_rows_materialised,
    }
    lead_manifest = {
        **SAFETY_FLAGS,
        "definition": (
            "earliest registered semantic-loop completion strictly after checkpoint and within "
            "three completed bars; zero when absent"
        ),
        "lead_values": [0, 1, 2, 3],
        "same_bar_multiple_completions_are_one_positive": True,
        "exact_identity_descriptive_only": True,
        "unresolved_rows_excluded": 0,
    }
    prefix_manifest = {
        **SAFETY_FLAGS,
        "definition": (
            "total canonical transitions required minus transitions completed by active prefix"
        ),
        "canonical_calculation": (
            "len(frozen registered oriented path) - 1 - (progress_states - 1)"
        ),
        "declared_ledger_remainder_used_only_as_cross_check": True,
        "semantic_loop_dictionary": dictionary_manifest,
        "motif_types": ["primitive", "repeat", "composite"],
        "orientation_preserved": True,
        "semantic_loop_identity_preserved": True,
        "active_prefix_rows_checked": int(
            ledger.loc[
                ledger["ledger_kind"].eq("active_prefix")
                & ledger["bar_ordinal"].astype(int).isin(CHECKPOINTS)
            ].shape[0]
        ),
        "assessment_one_transition_away_prevalence": float(
            panel.loc[
                panel["period"].eq("assessment"),
                "any_prefix_one_transition_from_completion",
            ].mean()
        ),
        "advance_definition": (
            "first_completion_lead != 1 and no active prefix exactly one transition away"
        ),
    }
    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected)
    write_json(output / "predecessor_reconstruction.json", predecessor)
    write_json(output / "lead_target_manifest.json", lead_manifest)
    write_json(output / "prefix_proximity_manifest.json", prefix_manifest)
    write_parquet(output / "fixed_lead_panel.parquet", panel)
    write_csv(output / "lead_support.csv", lead_support)
    write_csv(output / "lead_route_diagnostics.csv", lead_diagnostics)
    write_csv(output / "route_resolution_state_metrics.csv", route_states)
    write_csv(output / "concentration_metrics.csv", concentration)

    if blocker is not None:
        immediate_status = (
            "descriptive_only"
            if blocker == "blocked_insufficient_advance_positive_support"
            else "insufficient_support"
        )
        decision = {
            **SAFETY_FLAGS,
            "primary_decision": blocker,
            "blocker": blocker,
            "binding_question": (
                "After excluding lead-one completions and one-transition-away prefixes, does "
                "route competition improve bars-two-to-three completion prediction?"
            ),
            "immediate_completion_status": immediate_status,
            "advance_completion_status": "insufficient_support",
            "non_imminent_route_status": "insufficient_support",
            "support": support,
            "models_fitted": 0,
            "bootstrap_draws_executed": 0,
            "route_null_refits_executed": {"immediate": 0, "advance": 0},
            "detail": (
                f"advance positives {support['advance_positive_outcomes']} are below the fixed "
                "minimum of 250; no model, bootstrap, or null fit was opened"
            ),
        }
        write_json(output / "decision.json", decision)
        write_blocked_model_artifacts(output, panel.copy(), blocker)
        determinism = blocked_determinism_check(
            output,
            predecessor_panel,
            ledger,
            canonical_routes,
            decision,
        )
        write_json(output / "determinism_check.json", determinism)
        if not determinism["passed"]:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure", "determinism check failed"
            )
        finalize_report(output)
        return decision

    panel, primary_models, boundaries = fit_primary_models(panel)
    immediate, advance, checkpoint, monthly, subgroup = build_model_tables(panel, boundaries)
    assessment = panel.loc[panel["period"].eq("assessment")]
    bootstrap = run_bootstrap(assessment, boundaries)
    panel, route_null, null_models = run_route_nulls(panel, boundaries)
    immediate_increments = pair_increments(
        assessment,
        baseline_model="N0",
        route_model="N1",
        target="completion_next_1_bar",
        boundaries=boundaries,
    )
    assessment_advance = assessment.loc[assessment["advance_eligible"].eq(1)]
    advance_increments = pair_increments(
        assessment_advance,
        baseline_model="A0",
        route_model="A1",
        target="completion_in_bars_2_or_3",
        boundaries=boundaries,
    )
    immediate_positive_months, _ = stability_gate_values(
        monthly, population="immediate", baseline="N0", route="N1"
    )
    advance_positive_months, _ = stability_gate_values(
        monthly, population="advance", baseline="A0", route="A1"
    )
    _, immediate_adverse = stability_gate_values(
        checkpoint, population="immediate", baseline="N0", route="N1"
    )
    _, advance_adverse = stability_gate_values(
        checkpoint, population="advance", baseline="A0", route="A1"
    )
    immediate_gates: dict[str, object] = {
        **immediate_increments,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(
            bootstrap, "immediate", "log_loss_improvement", 0.80
        ),
        "bootstrap_80_brier_lower": _bootstrap_lower(
            bootstrap, "immediate", "brier_improvement", 0.80
        ),
        "positive_months": immediate_positive_months,
        "materially_adverse_checkpoints": immediate_adverse,
        "real_exceeds_all_nulls": null_exceeds_all(route_null, "immediate"),
        "support_and_concentration_passed": True,
    }
    advance_gates: dict[str, object] = {
        **advance_increments,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(
            bootstrap, "advance", "log_loss_improvement", 0.80
        ),
        "bootstrap_80_brier_lower": _bootstrap_lower(
            bootstrap, "advance", "brier_improvement", 0.80
        ),
        "bootstrap_80_average_precision_lower": _bootstrap_lower(
            bootstrap, "advance", "average_precision_improvement", 0.80
        ),
        "positive_months": advance_positive_months,
        "materially_adverse_checkpoints": advance_adverse,
        "real_exceeds_all_nulls": null_exceeds_all(route_null, "advance"),
        "support_and_concentration_passed": True,
    }
    immediate_passed = fixed_lead_increment_passes(immediate_gates, require_average_precision=False)
    advance_passed = fixed_lead_increment_passes(advance_gates, require_average_precision=True)
    route_states = update_route_state_increments(route_states, assessment, boundaries)
    supported_state_rates = route_states.loc[
        route_states["inference_supported"], "clean_bars_two_to_three_completion_rate"
    ].dropna()
    descriptive = bool(
        len(supported_state_rates) >= 2
        and float(supported_state_rates.max()) > float(supported_state_rates.min())
    )
    a0 = advance.set_index("model").loc["A0"]
    base = float(a0["base_rate"])
    constant_log_loss = float(
        -np.average(
            assessment_advance["completion_in_bars_2_or_3"] * math.log(max(base, 1e-12))
            + (1 - assessment_advance["completion_in_bars_2_or_3"])
            * math.log(max(1.0 - base, 1e-12)),
            weights=assessment_advance["row_weight"],
        )
    )
    baseline_meaningful = bool(float(a0["auc"]) > 0.5 and float(a0["log_loss"]) < constant_log_loss)
    primary_decision = choose_fixed_lead_decision(
        blocker=None,
        immediate_passed=immediate_passed,
        advance_passed=advance_passed,
        descriptive_lead_structure=descriptive,
        baseline_meaningful=baseline_meaningful,
    )
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": primary_decision,
        "blocker": None,
        "binding_question": (
            "After excluding lead-one completions and one-transition-away prefixes, does route "
            "competition improve bars-two-to-three completion prediction?"
        ),
        "immediate_completion_status": ("supported" if immediate_passed else "not_supported"),
        "advance_completion_status": (
            "supported"
            if advance_passed
            else ("descriptive_only" if descriptive else "not_supported")
        ),
        "non_imminent_route_status": (
            "supported"
            if advance_passed
            else ("descriptive_only" if descriptive else "not_supported")
        ),
        "support": support,
        "immediate_increments": immediate_increments,
        "advance_increments": advance_increments,
        "immediate_gates": immediate_gates,
        "advance_gates": advance_gates,
        "constant_probability_advance_log_loss": constant_log_loss,
        "compressed_transition_baseline_meaningful": baseline_meaningful,
        "models_fitted": 4,
        "bootstrap_draws_executed": BOOTSTRAP_DRAWS,
        "route_null_refits_executed": {"immediate": 3, "advance": 3},
    }
    write_json(output / "decision.json", decision)
    write_parquet(output / "fixed_lead_panel.parquet", panel)
    write_parquet(
        output / "assessment_predictions.parquet",
        panel.loc[panel["period"].eq("assessment")].reset_index(drop=True),
    )
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "primary_models_fitted": 4,
            "primary_models": ["N0", "N1", "A0", "A1"],
            "probability_quantile_boundaries": boundaries,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "route_null_refits": {"immediate": 3, "advance": 3},
            "H0_features": list(BASELINE_FEATURES),
            "H1_features": list(H1_FEATURES),
        },
    )
    coefficient_artifact = {
        **SAFETY_FLAGS,
        "primary_models": primary_models,
        "route_null_models": null_models,
        "models_fitted": 4,
    }
    write_json(output / "model_coefficients.json", coefficient_artifact)
    write_csv(output / "immediate_metrics.csv", immediate)
    write_csv(output / "advance_metrics.csv", advance)
    write_csv(output / "checkpoint_metrics.csv", checkpoint)
    write_csv(output / "monthly_metrics.csv", monthly)
    write_csv(output / "subgroup_metrics.csv", subgroup)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "route_null_metrics.csv", route_null)
    write_csv(output / "route_resolution_state_metrics.csv", route_states)
    plot_supported_screen(
        immediate,
        advance,
        output / "fixed_lead_model_increments.png",
    )
    determinism = supported_determinism_check(
        output,
        predecessor_panel,
        ledger,
        canonical_routes,
        coefficient_artifact,
        decision,
    )
    write_json(output / "determinism_check.json", determinism)
    if not determinism["passed"]:
        raise ScreenBlocker("blocked_reproducibility_or_audit_failure", "determinism check failed")
    finalize_report(output)
    return decision


def write_early_blocker(output: Path, blocker: ScreenBlocker) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "decision.json",
        {
            **SAFETY_FLAGS,
            "primary_decision": blocker.code,
            "blocker": blocker.code,
            "detail": blocker.detail,
            "immediate_completion_status": "insufficient_support",
            "advance_completion_status": "insufficient_support",
            "non_imminent_route_status": "insufficient_support",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = arguments.output.expanduser().resolve()
    try:
        if not output.is_relative_to(REPO_ROOT):
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "output must be inside the repository root",
            )
        decision = execute_screen(output)
        print(canonical_json(decision), end="")
        return 2 if decision.get("blocker") else 0
    except ScreenBlocker as blocker:
        write_early_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
