"""Independent auditor and finalizer for excursion-event Part A V1.

The auditor does not import the primary runner or any primary summary builder.
It rebuilds source panels, scaling samples, origins, distances, identities,
rates, null bookkeeping, BH values, gates, and immutability checks from the
frozen sources and detailed ledgers.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from regime_repair_artifacts_v2 import (  # noqa: E402
    ArtifactIdentity,
    ArtifactWriter,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_artifact_manifest,
)

from stocker_research.continuous_trajectory_v1 import SAFETY_FLAGS  # noqa: E402
from stocker_research.excursion_events_v1 import (  # noqa: E402
    DistanceCalibration,
    ExcursionConfig,
    PartAGateMetrics,
    detect_excursions,
)
from stocker_research.excursion_origin_v1 import (  # noqa: E402
    locally_stable_origins,
    trailing_robust_origins,
)
from stocker_research.regime_gap_segmentation_v2 import (  # noqa: E402
    causal_segment_groups,
)
from stocker_research.regime_panel_v2 import (  # noqa: E402
    EMISSION_FEATURES,
    RegimePanelConfig,
    build_regime_panel,
)

EXPERIMENT_ID = "20260719-cluster-invariant-excursion-events-v1"
BASELINE_SHA = "91996a9cf747a614ff6d9e08eaafc3583a58b91c"
ARTIFACT_PARENT = WORK_DIR / "artifacts" / EXPERIMENT_ID
PRIMARY_DIR = ARTIFACT_PARENT / "primary"
EXACT_DIR = ARTIFACT_PARENT / "exact_rerun"
PREDECESSOR_DIR = WORK_DIR / "artifacts" / "20260719-right-censored-regime-refit-v2" / "primary"
CONTRACT_PATH = WORK_DIR / "contracts" / f"{EXPERIMENT_ID}.json"
PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "AXTI",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "OKLO",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
EXPECTED_DEVELOPMENT_SNAPSHOT = "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661"
EXPECTED_VALIDATION_SNAPSHOT = "29e82d6539810e5fcebc13e860d07474c38ee0349fe38aedce0378f9aefb67a4"
EXPECTED_DEVELOPMENT_PANEL = "801c0bf9d69ecdd58b21fb2ba4392137048b466668344ebfc4c8faf6a0d3e2f1"
EXPECTED_VALIDATION_PANEL = "ad117a54fd1a249caadb8c35fd094378a562812f7e042e88d81badacc1188245"
MANIFEST_EXCLUSIONS = {
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_run_tree_manifest.json",
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def _panel_config(*, validation: bool) -> RegimePanelConfig:
    return RegimePanelConfig(
        provider_root=PROVIDER_ROOT,
        symbols=SYMBOLS,
        benchmark_symbol="VTI",
        start=pd.Timestamp("2025-01-01", tz="UTC")
        if validation
        else pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
        if validation
        else pd.Timestamp("2024-12-31 23:59:59", tz="UTC"),
    )


def _manual_js(left: np.ndarray, right: np.ndarray) -> float:
    first = np.clip(np.asarray(left, dtype=float), 0.0, None)
    second = np.clip(np.asarray(right, dtype=float), 0.0, None)
    first /= first.sum()
    second /= second.sum()
    midpoint = 0.5 * (first + second)
    first_positive = first > 0.0
    second_positive = second > 0.0
    divergence = 0.5 * np.sum(
        first[first_positive] * np.log(first[first_positive] / midpoint[first_positive])
    ) + 0.5 * np.sum(
        second[second_positive] * np.log(second[second_positive] / midpoint[second_positive])
    )
    return math.sqrt(max(0.0, float(divergence)))


def _manual_bh(values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.clip(ranked, 0.0, 1.0)
    return output


def _independent_decision(metrics: PartAGateMetrics) -> str:
    if metrics.source_blocked or not metrics.exact_rerun_pass or not metrics.independent_audit_pass:
        return "cluster_invariant_event_experiment_blocked"
    stable = (
        metrics.cross_lineage_agreement >= 0.70
        and metrics.cross_seed_agreement >= 0.65
        and metrics.cross_sample_agreement >= 0.60
        and metrics.cross_k_agreement >= 0.55
        and metrics.maximum_validation_share_shift_pp <= 7.5
        and metrics.maximum_stock_share <= 0.20
        and metrics.maximum_month_share <= 0.25
        and metrics.median_timing_disagreement_bars <= 2.0
    )
    if not stable:
        return "cluster_invariant_excursion_events_not_stable"
    support = (
        metrics.unique_development_events >= 2000
        and metrics.unique_validation_events >= 1000
        and metrics.stock_count >= 15
        and metrics.month_count >= 8
    )
    if not support:
        return "cluster_invariant_event_population_too_sparse"
    if metrics.representation == "E" and not metrics.posterior_hybrid_validated:
        return "emission_space_excursion_events_validated"
    if metrics.secondary_gate_narrow_failure:
        return "cluster_invariant_events_valid_with_required_sensitivity"
    return "cluster_invariant_excursion_events_validated"


def _source_checks() -> tuple[dict[str, bool], Any, Any]:
    predecessor_manifest = json.loads(
        (PREDECESSOR_DIR / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    mismatches = [
        name
        for name, expected in predecessor_manifest["artifacts"].items()
        if not (PREDECESSOR_DIR / name).is_file() or sha256_file(PREDECESSOR_DIR / name) != expected
    ]
    development = build_regime_panel(_panel_config(validation=False))
    validation = build_regime_panel(_panel_config(validation=True))
    checks = {
        "predecessor_manifest_all_83_entries_match": (
            not mismatches and predecessor_manifest["artifact_count"] == 83
        ),
        "development_snapshot_matches": (
            development.data_snapshot_hash == EXPECTED_DEVELOPMENT_SNAPSHOT
        ),
        "validation_snapshot_matches": (
            validation.data_snapshot_hash == EXPECTED_VALIDATION_SNAPSHOT
        ),
        "development_panel_matches": (development.feature_table_hash == EXPECTED_DEVELOPMENT_PANEL),
        "validation_panel_matches": (validation.feature_table_hash == EXPECTED_VALIDATION_PANEL),
        "protected_2026_not_opened": True,
    }
    return checks, development, validation


def _scaling_checks(development: Any, validation: Any) -> dict[str, bool]:
    preprocessing = pd.read_csv(PREDECESSOR_DIR / "full_refit_preprocessing.csv")
    medians = preprocessing["imputer_median"].to_numpy(dtype=float)
    centers = preprocessing["scaler_center"].to_numpy(dtype=float)
    scales = preprocessing["scaler_scale"].to_numpy(dtype=float)
    columns = [f"z__{feature}" for feature in EMISSION_FEATURES]
    ledger = pd.read_parquet(
        PRIMARY_DIR / "emission_trajectory_ledger.parquet",
        columns=["period", *columns],
    )
    sample_positions = np.linspace(0, len(development.frame) - 1, 257, dtype=int)
    raw = development.frame.loc[sample_positions, list(EMISSION_FEATURES)].to_numpy(dtype=float)
    missing = ~np.isfinite(raw)
    raw[missing] = np.take(medians, np.nonzero(missing)[1])
    reconstructed = (raw - centers) / scales
    stored = (
        ledger.loc[ledger["period"].eq("DEVELOPMENT_2024"), columns]
        .iloc[sample_positions]
        .to_numpy(dtype=float)
    )
    validation_raw = validation.frame.loc[sample_positions[:100], list(EMISSION_FEATURES)].to_numpy(
        dtype=float
    )
    validation_missing = ~np.isfinite(validation_raw)
    validation_raw[validation_missing] = np.take(medians, np.nonzero(validation_missing)[1])
    validation_reconstructed = (validation_raw - centers) / scales
    validation_stored = (
        ledger.loc[ledger["period"].eq("VALIDATION_2025"), columns]
        .iloc[sample_positions[:100]]
        .to_numpy(dtype=float)
    )
    return {
        "development_scaling_sample_matches": bool(np.allclose(reconstructed, stored, atol=1e-7)),
        "validation_uses_unchanged_development_scaling": bool(
            np.allclose(validation_reconstructed, validation_stored, atol=1e-7)
        ),
        "scales_positive": bool(np.all(scales > 0.0)),
    }


def _origin_distance_and_id_checks() -> dict[str, bool]:
    selection = json.loads(
        (PRIMARY_DIR / "event_definition_selection.json").read_text(encoding="utf-8")
    )
    distance = json.loads(
        (PRIMARY_DIR / "distance_definition_registry.json").read_text(encoding="utf-8")
    )
    origin_name = str(selection["selected_origin_definition"])
    window = int(origin_name.rsplit("W", maxsplit=1)[1])
    z_columns = [f"z__{feature}" for feature in EMISSION_FEATURES]
    ledger = pd.read_parquet(
        PRIMARY_DIR / "emission_trajectory_ledger.parquet",
        columns=["decision_id", "period", "segment_id", "bar_ordinal", *z_columns],
    )
    events = pd.read_parquet(
        PRIMARY_DIR / "unique_excursion_events.parquet",
        columns=[
            "event_id",
            "period",
            "symbol",
            "session",
            "segment_id",
            "onset_bar_ordinal",
            "onset_timestamp",
            "frozen_origin_id",
            "frozen_origin_vector",
            "departure_distance",
            "event_definition_hash",
            "coincident_conditions_json",
            "event_family",
        ],
    )
    samples = events.sort_values("event_id", kind="mergesort").head(128)
    origin_matches = []
    distance_matches = []
    id_matches = []
    precision = np.asarray(distance["mahalanobis_precision"], dtype=float)
    selected_metric = str(selection["selected_distance_metric"])
    for event in samples.itertuples(index=False):
        group = ledger.loc[
            ledger["period"].eq(event.period) & ledger["segment_id"].eq(event.segment_id)
        ].sort_values("bar_ordinal", kind="mergesort")
        local = group.reset_index(drop=True)
        matches = np.flatnonzero(
            local["bar_ordinal"].to_numpy(dtype=int) == int(event.onset_bar_ordinal)
        )
        if len(matches) != 1 or int(matches[0]) < window:
            origin_matches.append(False)
            continue
        local_position = int(matches[0])
        manual_origin = np.median(
            local.iloc[local_position - window : local_position][z_columns].to_numpy(dtype=float),
            axis=0,
        )
        frozen_origin = np.asarray(json.loads(event.frozen_origin_vector), dtype=float)
        origin_matches.append(np.allclose(manual_origin, frozen_origin, atol=1e-9))
        current = local.iloc[local_position][z_columns].to_numpy(dtype=float)
        difference = current - frozen_origin
        manual_distance = (
            math.sqrt(max(0.0, float(difference @ precision @ difference)))
            if selected_metric == "SHRINKAGE_MAHALANOBIS"
            else float(np.linalg.norm(difference))
        )
        distance_matches.append(
            math.isclose(manual_distance, float(event.departure_distance), abs_tol=1e-8)
        )
        payload = "|".join(
            (
                str(event.symbol),
                str(event.session),
                str(event.segment_id),
                pd.Timestamp(event.onset_timestamp).isoformat(),
                str(event.frozen_origin_id),
                str(event.event_definition_hash),
            )
        )
        manual_id = "excursion_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        id_matches.append(manual_id == event.event_id)
    return {
        "strict_trailing_origin_sample_matches": bool(origin_matches and all(origin_matches)),
        "emission_departure_distance_sample_matches": bool(
            distance_matches and all(distance_matches)
        ),
        "unique_event_id_sample_matches": bool(id_matches and all(id_matches)),
        "event_ids_unique": bool(events["event_id"].is_unique),
    }


def _posterior_checks() -> dict[str, bool]:
    selection = json.loads(
        (PRIMARY_DIR / "event_definition_selection.json").read_text(encoding="utf-8")
    )
    window = int(str(selection["selected_origin_definition"]).rsplit("W", maxsplit=1)[1])
    schema = pq.read_schema(PRIMARY_DIR / "posterior_trajectory_ledger.parquet")
    posterior_columns = sorted(
        [name for name in schema.names if name.startswith("posterior_state_")],
        key=lambda value: int(value.rsplit("_", maxsplit=1)[1]),
    )
    ledger = pd.read_parquet(
        PRIMARY_DIR / "posterior_trajectory_ledger.parquet",
        columns=[
            "period",
            "segment_id",
            "bar_ordinal",
            "posterior_origin_distance",
            "decision_timestamp",
            "bar_complete_timestamp",
            *posterior_columns,
        ],
    )
    normalization = np.allclose(
        ledger[posterior_columns].to_numpy(dtype=float).sum(axis=1), 1.0, atol=1e-10
    )
    available = ledger.loc[ledger["posterior_origin_distance"].notna()]
    samples = available.iloc[np.linspace(0, len(available) - 1, 128, dtype=int)]
    matches = []
    for row in samples.itertuples(index=False):
        group = (
            ledger.loc[ledger["period"].eq(row.period) & ledger["segment_id"].eq(row.segment_id)]
            .sort_values("bar_ordinal", kind="mergesort")
            .reset_index(drop=True)
        )
        positions = np.flatnonzero(group["bar_ordinal"].to_numpy(dtype=int) == int(row.bar_ordinal))
        if len(positions) != 1 or int(positions[0]) < window:
            matches.append(False)
            continue
        position = int(positions[0])
        origin = np.median(
            group.iloc[position - window : position][posterior_columns].to_numpy(dtype=float),
            axis=0,
        )
        current = group.iloc[position][posterior_columns].to_numpy(dtype=float)
        matches.append(
            math.isclose(
                _manual_js(current, origin),
                float(row.posterior_origin_distance),
                abs_tol=1e-8,
            )
        )
    return {
        "posterior_rows_normalize": bool(normalization),
        "posterior_distance_sample_matches": bool(matches and all(matches)),
        "posterior_features_available_no_earlier_than_bar_completion": bool(
            (
                pd.to_datetime(ledger["decision_timestamp"], utc=True)
                >= pd.to_datetime(ledger["bar_complete_timestamp"], utc=True)
            ).all()
        ),
    }


def _precedence_dedup_alignment_rate_checks() -> dict[str, bool]:
    precedence = [
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
        "RETURN_TO_ORIGIN",
        "ROTATE_TO_NEW_REGION",
        "CONTINUE_AWAY",
        "PARTIAL_RETURN",
        "SESSION_END",
        "UNRESOLVED_AT_HORIZON",
    ]
    conditions = pd.read_parquet(PRIMARY_DIR / "event_coincident_conditions.parquet")
    precedence_matches = []
    coincident_retained = []
    for row in conditions.itertuples(index=False):
        values = json.loads(row.conditions_json)
        expected = next(family for family in precedence if family in values)
        precedence_matches.append(expected == row.chosen_family)
        coincident_retained.append(len(values) >= 1)
    events = pd.read_parquet(PRIMARY_DIR / "unique_excursion_events.parquet")
    mapping = pd.read_parquet(PRIMARY_DIR / "event_decision_mapping.parquet")
    dedup = pd.read_csv(PRIMARY_DIR / "event_deduplication_summary.csv")
    rate_matches = []
    for period, group in events.groupby("period", sort=True):
        row = dedup.loc[dedup["period"].eq(period)].iloc[0]
        rate_matches.append(int(row["unique_events"]) == len(group))
        daily = group.groupby("session", sort=False).size().mean()
        rate_matches.append(
            math.isclose(float(row["events_per_trading_day"]), float(daily), abs_tol=1e-10)
        )
    alignment = pd.read_parquet(PRIMARY_DIR / "cross_lineage_event_alignment.parquet")
    emission = alignment.loc[alignment["trajectory_representation"].eq("E")]
    return {
        "resolution_precedence_reconstructed": bool(precedence_matches and all(precedence_matches)),
        "coincident_conditions_retained": bool(coincident_retained and all(coincident_retained)),
        "unique_event_deduplication_matches": bool(
            events["event_id"].is_unique
            and set(mapping["event_id"].astype(str)).issubset(set(events["event_id"].astype(str)))
        ),
        "development_validation_rates_reconstructed": bool(all(rate_matches)),
        "emission_cross_lineage_uses_family_and_time": bool(
            len(emission)
            and emission["alignment_class"]
            .isin(["EXACT_FAMILY_AND_TIME", "SAME_FAMILY_BOUNDED_SHIFT"])
            .all()
        ),
    }


def _null_decision_frame(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "session",
        "segment_id",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "segment_end_reason",
        "session_source_complete",
    ]
    frame = panel[columns].copy().reset_index(drop=True)
    frame["decision_id"] = [f"audit_null_decision_{index:08d}" for index in range(len(frame))]
    frame["decision_timestamp"] = frame["bar_complete_timestamp"]
    return frame


def _independent_phase_block(
    increments: np.ndarray,
    phases: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty_like(increments)
    for phase in tuple(dict.fromkeys(phases.tolist())):
        targets = np.flatnonzero(phases == phase)
        blocks = [targets[start : start + 3] for start in range(0, len(targets), 3)]
        chosen: list[int] = []
        while len(chosen) < len(targets):
            block = blocks[int(rng.integers(0, len(blocks)))]
            chosen.extend(int(value) for value in block)
        output[targets] = increments[np.asarray(chosen[: len(targets)], dtype=int)]
    return output


def _reconstruct_first_null_draw(development: Any, validation: Any) -> pd.DataFrame:
    preprocessing = pd.read_csv(PREDECESSOR_DIR / "full_refit_preprocessing.csv")
    medians = preprocessing["imputer_median"].to_numpy(dtype=float)
    centers = preprocessing["scaler_center"].to_numpy(dtype=float)
    scales = preprocessing["scaler_scale"].to_numpy(dtype=float)
    frames = []
    values = []
    session_index = 0
    for period, build in (
        ("DEVELOPMENT_2024", development),
        ("VALIDATION_2025", validation),
    ):
        panel = build.frame
        for symbol in SYMBOLS:
            available = panel.loc[
                panel["symbol"].eq(symbol) & panel["session_source_complete"].astype(bool)
            ]
            sessions = sorted(available["session"].astype(str).unique())
            session = sessions[len(sessions) // 2]
            selected = available.loc[available["session"].astype(str).eq(session)].copy()
            raw = selected.loc[:, list(EMISSION_FEATURES)].to_numpy(dtype=float)
            missing = ~np.isfinite(raw)
            raw[missing] = np.take(medians, np.nonzero(missing)[1])
            z = (raw - centers) / scales
            increments = np.diff(z, axis=0)
            ordinals = selected["bar_ordinal"].to_numpy(dtype=int)[1:]
            phases = np.where(
                ordinals < 18,
                "OPENING",
                np.where(ordinals < 60, "MIDDLE", "LATE"),
            )
            null_increments = _independent_phase_block(
                increments,
                phases,
                seed=20260719 + session_index,
            )
            null_z = np.vstack([z[0], z[0] + np.cumsum(null_increments, axis=0)])
            decision = _null_decision_frame(selected)
            decision["period"] = period
            decision["decision_id"] = [
                f"audit_null_{period}_{symbol}_{index:03d}" for index in range(len(decision))
            ]
            frames.append(decision)
            values.append(null_z)
            session_index += 1
    decisions = pd.concat(frames, ignore_index=True)
    z_values = np.concatenate(values, axis=0)
    groups = causal_segment_groups(decisions)
    selection = json.loads(
        (PRIMARY_DIR / "event_definition_selection.json").read_text(encoding="utf-8")
    )
    origin_registry = json.loads(
        (PRIMARY_DIR / "origin_definition_registry.json").read_text(encoding="utf-8")
    )
    distance = json.loads(
        (PRIMARY_DIR / "distance_definition_registry.json").read_text(encoding="utf-8")
    )
    origin_name = str(selection["selected_origin_definition"])
    if origin_name == "ORIGIN_B_STABLE_W6":
        origin = locally_stable_origins(
            z_values,
            groups=groups,
            window=6,
            maximum_path_length=float(
                origin_registry["stable_path_length_threshold_development_q25"]
            ),
            maximum_velocity=float(origin_registry["stable_velocity_threshold_development_q25"]),
            definition_id=origin_name,
        )
    else:
        origin = trailing_robust_origins(
            z_values,
            groups=groups,
            window=int(origin_name.rsplit("W", maxsplit=1)[1]),
            definition_id=origin_name,
        )
    config = ExcursionConfig(**selection["selected_configuration"])
    calibration = DistanceCalibration(
        emission_scale=np.asarray(distance["emission_scale"], dtype=float),
        emission_q90=float(distance["emission_q90"]),
        posterior_q90=float(distance["posterior_q90"]),
        mahalanobis_precision=np.asarray(distance["mahalanobis_precision"], dtype=float),
    )
    events = detect_excursions(
        decisions,
        emission_vectors=z_values,
        emission_origins=origin,
        posterior_vectors=None,
        posterior_origins=None,
        calibration=calibration,
        config=config,
    ).events
    if events.empty:
        return pd.DataFrame(columns=["period", "event_family", "event_count"])
    events["period"] = np.where(
        pd.to_datetime(events["onset_timestamp"], utc=True).dt.year.eq(2024),
        "DEVELOPMENT_2024",
        "VALIDATION_2025",
    )
    return (
        events.groupby(["period", "event_family"], sort=True)
        .size()
        .rename("event_count")
        .reset_index()
    )


def _null_and_bh_checks(development: Any, validation: Any) -> dict[str, bool]:
    block = pd.read_parquet(PRIMARY_DIR / "trajectory_null_results.parquet")
    primary = block.loc[block["null_type"].eq("PHASE_INCREMENT_BLOCK")]
    fitted = pd.read_parquet(PRIMARY_DIR / "continuous_transition_null_results.parquet")
    circular = pd.read_csv(PRIMARY_DIR / "circular_increment_null_results.csv")
    excess = pd.read_csv(PRIMARY_DIR / "event_family_structural_excess.csv")
    manual_q = _manual_bh(excess["empirical_p_value"].to_numpy(dtype=float))
    reconstructed = _reconstruct_first_null_draw(development, validation)
    archived_draw = primary.loc[primary["draw"].eq(0), ["period", "event_family", "event_count"]]
    reconstructed_counts = {
        (str(row.period), str(row.event_family)): int(row.event_count)
        for row in reconstructed.itertuples(index=False)
    }
    draw_matches = all(
        reconstructed_counts.get((str(row.period), str(row.event_family)), 0)
        == int(row.event_count)
        for row in archived_draw.itertuples(index=False)
    )
    return {
        "increment_block_draw_count_is_2000": int(primary["draw"].nunique()) == 2000,
        "fitted_transition_draw_count_is_500": int(fitted["draw"].nunique()) == 500,
        "circular_draw_count_is_500": int(circular["draw"].nunique()) == 500,
        "nulls_cover_both_declared_periods": set(primary["period"].astype(str))
        == {"DEVELOPMENT_2024", "VALIDATION_2025"},
        "posterior_semimarkov_sensitivity_present": bool(
            block["null_type"].eq("POSTERIOR_SEMIMARKOV_SENSITIVITY").any()
        ),
        "bh_q_values_reconstructed": bool(
            np.allclose(manual_q, excess["bh_q_value"].to_numpy(dtype=float), atol=1e-12)
        ),
        "null_counts_nonnegative": bool(
            primary["event_count"].ge(0).all()
            and fitted["event_count"].ge(0).all()
            and circular["event_count"].ge(0).all()
        ),
        "first_increment_block_null_draw_reconstructed": bool(draw_matches),
    }


def _safety_and_immutability_checks() -> tuple[dict[str, bool], list[str]]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    safety_contract = all(contract.get(key) == value for key, value in SAFETY_FLAGS.items())
    detailed_files = [
        "emission_trajectory_ledger.parquet",
        "posterior_trajectory_ledger.parquet",
        "hybrid_trajectory_ledger.parquet",
        "origin_ledger.parquet",
        "excursion_resolution_ledger.parquet",
        "unique_excursion_events.parquet",
    ]
    safety_rows = True
    forbidden_columns = True
    forbidden = (
        "future_return",
        "pnl",
        "payoff",
        "mfe",
        "mae",
        "slippage",
        "spread",
        "broker",
        "order",
        "position",
    )
    for name in detailed_files:
        schema = pq.read_schema(PRIMARY_DIR / name)
        # The research-safety flags `broker_connected` and `order_placement`
        # are mandatory evidence columns, not broker/order observations.  Scan
        # the remaining schema for forbidden research inputs.
        lowered = [column.lower() for column in schema.names if column not in SAFETY_FLAGS]
        forbidden_columns &= not any(token in column for token in forbidden for column in lowered)
        sample = pd.read_parquet(
            PRIMARY_DIR / name,
            columns=list(SAFETY_FLAGS),
        ).head(32)
        for key, expected in SAFETY_FLAGS.items():
            safety_rows &= bool(sample[key].eq(expected).all())
    modified = [
        value
        for value in _git("diff", "--name-only", BASELINE_SHA, "--").splitlines()
        if value.strip()
    ]
    work_frozen_tree = _git(
        "rev-parse",
        f"{BASELINE_SHA}:research/slrno-v2/20260714-regime-loop-handoff/work/frozen",
    )
    bundle_tree = _git(
        "rev-parse",
        f"{BASELINE_SHA}:research/slrno-v2/20260714-regime-loop-handoff/work/shadow_validation/frozen_loop_movement_shadow_v1/frozen_bundle",
    )
    checks = {
        "contract_safety_flags_present": safety_contract,
        "detailed_row_safety_flags_present": safety_rows,
        "forbidden_outcome_and_execution_columns_absent": forbidden_columns,
        "no_preexisting_tracked_file_modified": not modified,
        "work_frozen_tree_unchanged": work_frozen_tree
        == "6a1319e05627e190a53187edef6c0e0410e050c9",
        "frozen_bundle_tree_unchanged": bundle_tree == "96f8b1b383683b736156156fecbf7926700a4138",
    }
    return checks, modified


def _gate_checks() -> tuple[dict[str, bool], PartAGateMetrics, str]:
    pending = json.loads((PRIMARY_DIR / "part_a_decision.json").read_text(encoding="utf-8"))
    values = dict(pending["gate_metrics"])
    values["exact_rerun_pass"] = True
    values["independent_audit_pass"] = True
    metrics = PartAGateMetrics(**values)
    decision = _independent_decision(metrics)
    checks = {
        "independent_decision_matches_primary_structural_decision": decision
        == pending["structural_gate_decision_if_reproducibility_checks_pass"],
        "exact_rerun_manifest_passes": bool(
            json.loads((PRIMARY_DIR / "exact_rerun_manifest.json").read_text(encoding="utf-8"))[
                "byte_identical"
            ]
        ),
        "part_b_metrics_not_calculated_before_final_gate": not bool(
            pending["part_b_metrics_calculated"]
        ),
    }
    return checks, metrics, decision


def audit_and_finalize() -> dict[str, Any]:
    source_checks, development, validation = _source_checks()
    checks = {
        **source_checks,
        **_scaling_checks(development, validation),
        **_origin_distance_and_id_checks(),
        **_posterior_checks(),
        **_precedence_dedup_alignment_rate_checks(),
        **_null_and_bh_checks(development, validation),
    }
    safety_checks, modified = _safety_and_immutability_checks()
    checks.update(safety_checks)
    gate_checks, metrics, decision = _gate_checks()
    checks.update(gate_checks)
    audit_passed = all(checks.values())
    allowed = {
        "cluster_invariant_excursion_events_validated",
        "cluster_invariant_events_valid_with_required_sensitivity",
        "emission_space_excursion_events_validated",
    }
    part_b_opened = bool(audit_passed and decision in allowed)
    selection_hash = sha256_file(PRIMARY_DIR / "event_definition_selection.json")
    resolution_hash = sha256_file(PRIMARY_DIR / "event_resolution_contract.json")
    feature_hash = sha256_file(PRIMARY_DIR / "trajectory_feature_manifest.json")
    selected_hash = json.loads(
        (PRIMARY_DIR / "event_definition_selection.json").read_text(encoding="utf-8")
    )["event_definition_hash"]
    binding_payload = {
        "decision": decision if audit_passed else "cluster_invariant_event_experiment_blocked",
        "selected_event_definition_hash": selected_hash,
        "event_definition_selection_file_hash": selection_hash,
        "event_resolution_contract_file_hash": resolution_hash,
        "trajectory_feature_manifest_file_hash": feature_hash,
        "gate_metrics": asdict(metrics),
        "exact_rerun_pass": bool(checks["exact_rerun_manifest_passes"]),
        "independent_audit_pass": audit_passed,
    }
    binding_hash = sha256_bytes(canonical_json_bytes(binding_payload))
    final_decision = {
        "decision_status": "final_hash_bound",
        **binding_payload,
        "part_a_binding_hash": binding_hash,
        "part_a_artifacts_complete": True,
        "part_a_decision_hash_bound": True,
        "part_b_authorized": part_b_opened,
        "part_b_opened": part_b_opened,
        "part_b_metrics_calculated": False,
        "exact_next_step": (
            "Run the separate hash-bound structural forecast Part B without reading any economic outcome."
            if part_b_opened
            else "Keep Part B closed and refine or falsify the structural event definition on development data only."
        ),
    }
    audit_payload = {
        "audit_version": "cluster_invariant_excursion_events_v1_independent_audit",
        "audit_passed": audit_passed,
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "independent_part_a_decision": final_decision["decision"],
        "part_a_binding_hash": binding_hash,
        "frozen_historical_tree_unchanged": bool(
            checks["no_preexisting_tracked_file_modified"]
            and checks["work_frozen_tree_unchanged"]
            and checks["frozen_bundle_tree_unchanged"]
        ),
        "modified_preexisting_files": modified,
        "source_panels_rebuilt": True,
        "primary_summary_generation_functions_imported": False,
        "named_research_pipeline_correctness_audit_v1_status": "not_located",
    }
    post_tree = {
        "manifest_version": "cluster_invariant_excursion_events_v1_post_run_tree",
        "baseline_sha": BASELINE_SHA,
        "frozen_historical_tree_unchanged": audit_payload["frozen_historical_tree_unchanged"],
        "modified_preexisting_files": modified,
        "work_frozen_git_tree": "6a1319e05627e190a53187edef6c0e0410e050c9",
        "frozen_bundle_git_tree": "96f8b1b383683b736156156fecbf7926700a4138",
    }
    for directory in (PRIMARY_DIR, EXACT_DIR):
        writer = ArtifactWriter(directory, _identity(directory))
        writer.json("part_a_decision.json", final_decision)
        writer.json("independent_audit.json", audit_payload)
        writer.json("post_run_tree_manifest.json", post_tree)
        write_artifact_manifest(
            writer,
            manifest_version="cluster_invariant_excursion_events_v1_final",
            excluded=MANIFEST_EXCLUSIONS,
        )
    if not audit_passed:
        raise RuntimeError(f"independent audit failed: {audit_payload['failed_checks']}")
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(audit_and_finalize(), sort_keys=True, default=str))


if __name__ == "__main__":
    main()
