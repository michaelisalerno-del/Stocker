#!/usr/bin/env python3
"""Independent audit for the research-only Directional Signature Atlas V1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CORE_PATH = WORK / "frozen_loop_movement_shadow_core.py"
BUNDLE_ROOT = WORK / "shadow_validation/frozen_loop_movement_shadow_v1/frozen_bundle"
CONTRACT_PATH = WORK / "contracts/20260717-directional-signature-atlas-v1.json"
FEATURE_SCHEMA_PATH = WORK / "contracts/20260717-directional-signature-atlas-v1-feature-schema.json"
DEFAULT_PRIMARY = WORK / "artifacts/20260717-directional-signature-atlas-v1/primary"
DEFAULT_EXACT = WORK / "artifacts/20260717-directional-signature-atlas-v1/exact_rerun"
IDENTITY_EXCLUSIONS = {
    "independent_audit.json",
    "prospective_forecast_dry_run.json",
    "prospective_settlement_dry_run.json",
    "track_a_independent_audit.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def safe_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def close(left: object, right: object, *, tolerance: float = 1e-8) -> bool:
    left_float = float(cast(Any, left))
    right_float = float(cast(Any, right))
    if math.isnan(left_float) and math.isnan(right_float):
        return True
    return math.isclose(left_float, right_float, rel_tol=tolerance, abs_tol=tolerance)


def add_check(checks: list[dict[str, object]], check_id: str, passed: bool, detail: str) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})


def exact_identity(primary: Path, exact: Path) -> dict[str, object]:
    suffixes = {".parquet", ".csv", ".json", ".png"}
    left = {
        str(path.relative_to(primary)): path
        for path in primary.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in IDENTITY_EXCLUSIONS
    }
    right = {
        str(path.relative_to(exact)): path
        for path in exact.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in IDENTITY_EXCLUSIONS
    }
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    mismatches = sorted(
        name
        for name in set(left) & set(right)
        if sha256_file(left[name]) != sha256_file(right[name])
    )
    return {
        "byte_identical": not missing and not extra and not mismatches,
        "compared_files": len(left),
        "missing_files": missing,
        "extra_files": extra,
        "hash_mismatches": mismatches,
    }


def condition_mask(frame: pd.DataFrame, conditions: Sequence[Mapping[str, Any]]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        feature = str(condition["feature"])
        operator = str(condition["operator"])
        value = condition["value"]
        if feature not in frame:
            return pd.Series(False, index=frame.index)
        if operator == "==":
            mask &= frame[feature].eq(value)
        elif operator == "!=":
            mask &= frame[feature].ne(value) & frame[feature].notna()
        elif operator == ">":
            mask &= pd.to_numeric(frame[feature], errors="coerce").gt(float(value))
        elif operator == "<":
            mask &= pd.to_numeric(frame[feature], errors="coerce").lt(float(value))
        else:
            raise ValueError(f"unsupported condition operator in audit: {operator}")
    return mask.fillna(False)


def adjust_p_values(p_values: Sequence[float], method: str) -> np.ndarray:
    if not p_values:
        return np.asarray([], dtype=float)
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    count = len(values)
    if method == "fdr_bh":
        adjusted_ranked = ranked * count / np.arange(1, count + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    elif method == "holm":
        adjusted_ranked = ranked * (count - np.arange(count))
        adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    else:
        raise ValueError(method)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def expected_target(gross_long_bps: pd.Series, cost_bps: float, multiple: float) -> pd.Series:
    threshold = multiple * cost_bps
    values = np.full(len(gross_long_bps), "NEUTRAL", dtype=object)
    gross = pd.to_numeric(gross_long_bps, errors="coerce").to_numpy(float)
    values[(gross - cost_bps > 0.0) & (gross > threshold)] = "LONG"
    values[(-gross - cost_bps > 0.0) & (gross < -threshold)] = "SHORT"
    values[~np.isfinite(gross)] = "UNAVAILABLE"
    return pd.Series(values, index=gross_long_bps.index)


def verify_manifest(root: Path) -> tuple[bool, str]:
    manifest = cast(dict[str, Any], load_json(root / "artifact_manifest.json"))
    failures: list[str] = []
    for row in cast(list[dict[str, Any]], manifest["files"]):
        path = root / str(row["relative_path"])
        if not path.is_file():
            failures.append(str(row["relative_path"]))
            continue
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
            failures.append(str(row["relative_path"]))
    return not failures, f"{len(manifest['files'])} files; mismatches={failures[:5]}"


def verify_source_identity(
    root: Path, metadata: Mapping[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    identities = cast(dict[str, Any], load_json(root / "source_identities.json"))
    sources = cast(dict[str, dict[str, Any]], identities["sources"])
    failures: list[str] = []
    hashes: dict[str, str] = {}
    for name, specification in sorted(sources.items()):
        path = Path(str(specification["path"]))
        if not path.is_file():
            failures.append(f"missing:{name}")
            continue
        actual = sha256_file(path)
        hashes[name] = actual
        if actual != str(specification["sha256"]):
            failures.append(f"drift:{name}")
    snapshot_payload = (
        json.dumps(hashes, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    snapshot = hashlib.sha256(snapshot_payload.encode("utf-8")).hexdigest()
    if snapshot != str(identities["data_snapshot_sha256"]):
        failures.append("source_snapshot_hash")
    if snapshot != str(metadata["data_snapshot_sha256"]):
        failures.append("metadata_snapshot_hash")
    return not failures, f"{len(sources)} sources; failures={failures[:5]}", identities


def verify_prior_experiment_evidence(root: Path) -> tuple[bool, str]:
    payload = cast(dict[str, Any], load_json(root / "prior_experiment_coverage.json"))
    failures: list[str] = []
    experiments = cast(list[dict[str, Any]], payload.get("experiments", []))
    if (
        len(experiments) != 10
        or payload.get("exact_signature_atlas_previously_tested") is not False
    ):
        failures.append("prior_experiment_census")
    for experiment in experiments:
        for evidence in cast(list[dict[str, Any]], experiment.get("evidence", [])):
            path = Path(str(evidence["path"]))
            if bool(evidence["available"]) != path.is_file():
                failures.append(f"availability:{experiment['experiment']}")
            elif path.is_file() and str(evidence["sha256"]) != sha256_file(path):
                failures.append(f"hash:{experiment['experiment']}:{evidence['role']}")
    return not failures, f"experiments={len(experiments)}; failures={failures[:5]}"


def verify_git_identity(
    metadata: Mapping[str, Any], identities: Mapping[str, Any]
) -> tuple[bool, str]:
    git_sha = str(metadata["git_sha"])
    sources = cast(Mapping[str, Mapping[str, Any]], identities["sources"])
    failures: list[str] = []
    checked = 0
    for name, specification in sorted(sources.items()):
        if not name.startswith("atlas_"):
            continue
        path = Path(str(specification["path"]))
        try:
            relative = str(path.relative_to(REPO))
        except ValueError:
            continue
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{git_sha}:{relative}"],
                cwd=REPO,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            failures.append(f"missing:{name}")
            continue
        actual = hashlib.sha256(committed).hexdigest()
        if actual != str(specification["sha256"]):
            failures.append(f"mismatch:{name}")
        checked += 1
    return not failures, f"git={git_sha}; committed_atlas_sources={checked}; failures={failures}"


def verify_frozen_source_pins(
    contract: Mapping[str, Any], identities: Mapping[str, Any]
) -> tuple[bool, str]:
    pins = cast(Mapping[str, Any], contract["frozen_sources"])
    sources = cast(Mapping[str, Mapping[str, Any]], identities["sources"])
    mapping = {
        "prior_fixed_clock_contract_sha256": "contract",
        "prior_fixed_clock_runner_sha256": "runner",
        "state_preprocessing_sha256": "state_preprocessing",
        "state_parameters_sha256": "state_parameters",
        "fixed_cycles_sha256": "fixed_cycles",
        "loop_path_parameters_sha256": "path_parameters",
        "movement_feature_manifest_sha256": "movement_manifest",
        "movement_parameters_sha256": "movement_parameters",
        "frozen_movement_core_sha256": "frozen_movement_core",
        "vti_provider_sha256": "provider_VTI",
    }
    failures = [
        key
        for key, source_name in mapping.items()
        if str(pins.get(key)) != str(sources.get(source_name, {}).get("sha256"))
    ]
    return not failures, f"pins={len(mapping)}; failures={failures}"


def verify_population(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[bool, str]:
    population = cast(Mapping[str, Any], contract["population"])
    expected_rows = int(population["expected_population_rows"])
    ids_match = set(features["opportunity_id"].astype(str)) == set(
        outcomes["opportunity_id"].astype(str)
    )
    unique = features["opportunity_id"].is_unique and outcomes["opportunity_id"].is_unique
    ordinals = set(pd.to_numeric(features["decision_ordinal"]).astype(int))
    clocks = set(features["decision_clock"].astype(str))
    local = pd.to_datetime(features["decision_timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    expected_times = {"10:30", "12:30"}
    clock_times = set(local.dt.strftime("%H:%M"))
    session_match = local.dt.strftime("%Y-%m-%d").eq(features["session"].astype(str)).all()
    duplicate_clock = features.duplicated(["symbol", "session", "decision_clock"]).any()
    calendar = xcals.get_calendar("XNYS")
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(features["session"]).unique()))
    calendar_sessions = calendar.sessions_in_range(sessions.min(), sessions.max())
    exact_sessions = bool(sessions.isin(calendar_sessions).all())
    stage = features["chronology_stage"].astype(str)
    stage_counts = stage.value_counts().to_dict()
    chronology = cast(Mapping[str, Mapping[str, Any]], contract["chronology"])
    exact_stage = True
    session_dates = pd.to_datetime(features["session"])
    for name in (
        "development_context",
        "discovery",
        "validation",
        "final_opened_holdout",
    ):
        specification = chronology[name]
        expected = session_dates.ge(pd.Timestamp(str(specification["start"]))) & session_dates.lt(
            pd.Timestamp(str(specification["end_exclusive"]))
        )
        exact_stage &= bool(stage.eq(name).eq(expected).all())
    passed = all(
        (
            len(features) == expected_rows,
            len(outcomes) == expected_rows,
            ids_match,
            unique,
            ordinals == {12, 36},
            clocks == {"clock_12", "clock_36"},
            clock_times == expected_times,
            bool(session_match),
            not bool(duplicate_clock),
            exact_sessions,
            exact_stage,
        )
    )
    return passed, (
        f"rows={len(features)}; ordinals={sorted(ordinals)}; "
        f"clocks={sorted(clocks)}; local_times={sorted(clock_times)}; stages={stage_counts}"
    )


def verify_feature_availability(
    features: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[bool, str]:
    decision = pd.to_datetime(features["decision_timestamp"], utc=True, errors="raise")
    failures: list[str] = []
    enabled = [
        str(row["name"])
        for row in cast(list[dict[str, Any]], schema["features"])
        if bool(row.get("condition_enabled"))
    ]
    for feature in enabled:
        availability_column = f"{feature}__available_at"
        if feature not in features or availability_column not in features:
            failures.append(f"missing:{feature}")
            continue
        availability = pd.to_datetime(features[availability_column], utc=True, errors="coerce")
        populated = features[feature].notna()
        if bool((populated & (availability.isna() | availability.gt(decision))).any()):
            failures.append(f"late:{feature}")
    forbidden = {"target", "mfe_long_bps", "mae_long_bps", "future_route"}
    leaked = sorted(forbidden & set(features.columns))
    failures.extend(f"forbidden:{name}" for name in leaked)
    return not failures, f"enabled_features={len(enabled)}; failures={failures[:5]}"


def verify_motifs_and_loop_summaries(features: pd.DataFrame) -> tuple[bool, str]:
    failures: list[str] = []
    current = pd.to_numeric(features["current_state"], errors="coerce")
    previous = pd.to_numeric(features["previous_state"], errors="coerce")
    for length in (2, 3, 4):
        column = f"state_motif_{length}"
        populated = features.loc[features[column].notna(), [column]].copy()
        tokens = populated[column].astype(str).str.split(">")
        if not tokens.map(len).eq(length).all():
            failures.append(f"motif_length_{length}")
        indices = populated.index
        final_tokens = pd.to_numeric(tokens.map(lambda values: values[-1]), errors="coerce")
        if not np.allclose(final_tokens.to_numpy(float), previous.loc[indices].to_numpy(float)):
            failures.append(f"motif_previous_completed_state_{length}")
        if np.isclose(final_tokens.to_numpy(float), current.loc[indices].to_numpy(float)).any():
            failures.append(f"motif_contains_active_state_{length}")
        if tokens.map(
            lambda values: any(
                left == right for left, right in zip(values, values[1:], strict=False)
            )
        ).any():
            failures.append(f"motif_has_uncompressed_adjacent_state_{length}")
    loop_columns = [
        "top_parent_loop",
        "top_loop_orientation",
        "top_loop_score",
        "top_second_margin",
        "compatibility_mass",
        "compatibility_entropy",
        "compatible_loop_count",
    ]
    structural_period = features["chronology_stage"].ne("development_context")
    structural_columns = [
        "current_state",
        "previous_state",
        "state_motif_2",
        "state_motif_3",
        "state_motif_4",
        *loop_columns,
        "predicted_future_range_bps",
        "predicted_absolute_movement_bps",
    ]
    if bool(
        features.loc[~structural_period, structural_columns].notna().any(axis=1).any()
    ):
        failures.append("full_2024_fitted_structural_fields_used_in_development_context")
    allowed = (
        features["state_run_entry_at_decision"].fillna(False).astype(bool)
        & features["frozen_state_inputs_complete"].fillna(False).astype(bool)
        & structural_period
    )
    if bool(features.loc[~allowed, loop_columns].notna().any(axis=1).any()):
        failures.append("loop_summary_outside_exact_causal_run_entry")
    june_29 = features["session"].astype(str).eq("2026-06-29")
    if bool(features.loc[june_29, ["current_state", *loop_columns]].notna().any(axis=1).any()):
        failures.append("vti_boundary_not_failed_closed")
    return (
        not failures,
        f"motif_rows={int(features['state_motif_2'].notna().sum())}; failures={failures}",
    )


def verify_sampled_structural_source_reconstruction(
    features: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[bool, str]:
    """Rebuild one symbol from provider bars and the independently pinned bundle."""

    specification = importlib.util.spec_from_file_location("atlas_audit_frozen_core", CORE_PATH)
    if specification is None or specification.loader is None:
        return False, "unable to load frozen movement core"
    core = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(core)
    provider_root = Path(str(cast(Mapping[str, Any], contract["population"])["provider_root"]))
    population = cast(Mapping[str, Any], contract["population"])
    symbols = sorted(
        set(cast(list[str], population["symbols_2024_2025"]))
        | set(cast(list[str], population["symbols_2026"]))
    )
    symbol = "AAOI" if "AAOI" in symbols else symbols[0]
    start = pd.Timestamp("2024-01-01", tz="UTC")
    as_of = pd.Timestamp("2026-06-29T19:55:00Z")
    panel = pd.concat(
        [core.prepare_symbol_bars(item, provider_root, start, as_of) for item in symbols],
        ignore_index=True,
    )
    panel = panel.loc[
        ~(
            panel["symbol_norm"].eq("AAL")
            & panel["session_date"].astype(str).str.startswith("2026-")
        )
    ].copy()
    panel = panel.sort_values(["symbol_norm", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    vti = core.prepare_symbol_bars("VTI", provider_root, start, as_of)
    vti_max = pd.to_datetime(vti["timestamp"], utc=True).max()
    panel = core.add_market_features(panel, vti)
    panel["frozen_state_inputs_complete"] = pd.to_datetime(
        panel["timestamp"], utc=True
    ).le(vti_max)
    b0 = core.build_causal_b0(symbols, provider_root, start, as_of)
    panel = panel.merge(
        b0[
            [
                "session_date",
                "causal_slow_b0",
                "b0_direction_score",
                "b0_stress_score",
                "b0_stress_box",
            ]
        ],
        on="session_date",
        how="left",
        validate="many_to_one",
    )
    panel["b0_state_numeric"] = panel["causal_slow_b0"].map(
        {"weak_broad_tape": -1.0, "neutral_broad_tape": 0.0, "strong_broad_tape": 1.0}
    )
    panel["b0_high_stress"] = panel["b0_stress_box"].eq("high_stress").astype(float)
    panel = core.add_emission_features(panel)
    panel = panel.sort_values(["symbol_norm", "session_date", "timestamp"], kind="mergesort")
    preprocessing = pd.read_csv(
        BUNDLE_ROOT / "artifacts/state/frozen_emission_preprocessing.csv"
    )
    state_parameters = dict(
        np.load(BUNDLE_ROOT / "artifacts/state/frozen_semimarkov_parameters.npz")
    )
    panel = core.assign_session_states(
        panel.reset_index(drop=True), preprocessing, state_parameters
    )

    anchors: list[dict[str, Any]] = []
    symbol_panel = panel.loc[panel["symbol_norm"].eq(symbol)]
    for session, session_frame in symbol_panel.groupby("session_date", sort=False):
        run_states: list[int] = []
        run_lengths: list[int] = []
        for tuple_row in session_frame.itertuples(index=False):
            row = cast(Any, tuple_row)
            state = int(row.state)
            if not run_states or state != run_states[-1]:
                run_states.append(state)
                run_lengths.append(1)
            else:
                run_lengths[-1] += 1
            ordinal = int(row.bar_index_in_session)
            if ordinal not in {12, 36}:
                continue
            completed = run_states[:-1]
            repeat = 0
            cursor = len(run_states) - 1
            while cursor >= 2 and run_states[cursor] == run_states[cursor - 2]:
                repeat += 1
                cursor -= 2
            anchors.append(
                {
                    **row._asdict(),
                    "symbol": symbol,
                    "session": str(session),
                    "decision_ordinal": ordinal,
                    "current_state": state,
                    "previous_state": run_states[-2] if len(run_states) >= 2 else np.nan,
                    "previous_state_1": run_states[-2] if len(run_states) >= 2 else 8,
                    "previous_state_2": run_states[-3] if len(run_states) >= 3 else 8,
                    "state_motif_2": ">".join(map(str, completed[-2:]))
                    if len(completed) >= 2
                    else None,
                    "state_motif_3": ">".join(map(str, completed[-3:]))
                    if len(completed) >= 3
                    else None,
                    "state_motif_4": ">".join(map(str, completed[-4:]))
                    if len(completed) >= 4
                    else None,
                    "prior_completed_state_dwell_bars": run_lengths[-2]
                    if len(run_lengths) >= 2
                    else np.nan,
                    "same_orientation_repeat_count": repeat,
                    "state_run_entry_at_decision": int(row.age) == 1,
                }
            )
    anchor_frame = pd.DataFrame(anchors)
    ledger = features.loc[
        features["symbol"].eq(symbol)
        & features["chronology_stage"].ne("development_context")
    ].copy()
    comparison = ledger.merge(
        anchor_frame,
        on=["symbol", "session", "decision_ordinal"],
        how="left",
        validate="one_to_one",
        suffixes=("_ledger", "_source"),
    )
    failures: list[str] = []
    complete_comparison = comparison.loc[
        comparison["frozen_state_inputs_complete_source"].astype(bool)
    ]
    sample = complete_comparison.iloc[
        np.linspace(0, len(complete_comparison) - 1, num=12, dtype=int)
    ]
    for column in (
        "state_motif_2",
        "state_motif_3",
        "state_motif_4",
    ):
        left = sample[f"{column}_ledger"].astype(object).where(
            sample[f"{column}_ledger"].notna(), "<NA>"
        )
        right = sample[f"{column}_source"].astype(object).where(
            sample[f"{column}_source"].notna(), "<NA>"
        )
        if not left.astype(str).eq(right.astype(str)).all():
            failures.append(f"sampled_source_{column}")
    for column in (
        "current_state",
        "previous_state",
        "prior_completed_state_dwell_bars",
        "same_orientation_repeat_count",
        "state_run_entry_at_decision",
    ):
        left_numeric = pd.to_numeric(sample[f"{column}_ledger"], errors="coerce")
        right_numeric = pd.to_numeric(sample[f"{column}_source"], errors="coerce")
        if not np.allclose(left_numeric, right_numeric, equal_nan=True):
            failures.append(f"sampled_source_{column}")

    cycles = core.load_cycles(BUNDLE_ROOT / "artifacts/state/fixed_cycle_shuffled_nulls.csv")
    path_parameters = dict(np.load(BUNDLE_ROOT / "artifacts/path/model_parameters.npz"))
    feature_manifest = load_json(BUNDLE_ROOT / "artifacts/price/feature_manifest.json")
    outcome_parameters = dict(
        np.load(BUNDLE_ROOT / "artifacts/price/outcome_model_parameters.npz")
    )
    eligible = anchor_frame.loc[
        anchor_frame["state_run_entry_at_decision"].astype(bool)
        & anchor_frame["frozen_state_inputs_complete"].astype(bool)
        & pd.to_datetime(anchor_frame["session"]).ge(pd.Timestamp("2025-01-01"))
    ].copy()
    eligible["state"] = eligible["current_state"].astype(int)
    eligible["b0_entry_numeric"] = pd.to_numeric(eligible["b0_state_numeric"], errors="coerce")
    eligible["b0_entry_high_stress"] = pd.to_numeric(
        eligible["b0_high_stress"], errors="coerce"
    )
    local = pd.to_datetime(eligible["timestamp"], utc=True).dt.tz_convert("America/New_York")
    phase = 2.0 * np.pi * (local.dt.hour * 60.0 + local.dt.minute - 570.0) / 390.0
    eligible["entry_time_sin"] = np.sin(phase)
    eligible["entry_time_cos"] = np.cos(phase)
    eligible = core.add_loop_scores(eligible, cycles, path_parameters)
    eligible = core.movement_predictions(eligible, feature_manifest, outcome_parameters)
    loop_columns = [f"loop_score_{index:02d}" for index in range(1, 21)]
    structural_samples = eligible.iloc[:3]
    for source_row in structural_samples.itertuples(index=False):
        row = cast(Any, source_row)
        recorded = features.loc[
            features["symbol"].eq(symbol)
            & features["session"].astype(str).eq(str(row.session))
            & features["decision_ordinal"].eq(int(row.decision_ordinal))
        ].iloc[0]
        scores = np.asarray([float(getattr(row, column)) for column in loop_columns])
        order = np.argsort(-scores, kind="mergesort")
        top = int(order[0])
        mass = float(scores.sum())
        probabilities = scores / mass if mass > 0.0 else np.zeros_like(scores)
        positive = probabilities[probabilities > 0.0]
        entropy = (
            float(-(positive * np.log(positive)).sum() / np.log(len(scores)))
            if len(positive)
            else 0.0
        )
        core_states = tuple(int(value) for value in cycles.iloc[top]["core"])
        previous = int(row.previous_state_1)
        current = int(row.current_state)
        forward = any(
            core_states[index] == previous
            and core_states[(index + 1) % len(core_states)] == current
            for index in range(len(core_states))
        )
        reverse = any(
            core_states[index] == previous
            and core_states[(index - 1) % len(core_states)] == current
            for index in range(len(core_states))
        )
        orientation = (
            "bidirectional"
            if len(core_states) == 2
            else "unavailable"
            if previous == 8
            else (
                "forward"
                if forward and not reverse
                else (
                    "reverse"
                    if reverse and not forward
                    else ("ambiguous" if forward or reverse else "incompatible_transition")
                )
            )
        )
        expected_values = {
            "top_parent_loop": f"cycle_{top + 1:02d}",
            "top_loop_orientation": orientation,
            "parent_loop_family": f"transition_length_{len(core_states)}",
            "top_loop_score": float(scores[top]),
            "top_second_margin": float(scores[top] - scores[order[1]]),
            "compatibility_mass": mass,
            "compatibility_entropy": entropy,
            "compatible_loop_count": int(np.count_nonzero(scores > 0.0)),
            "predicted_future_range_bps": float(
                row.loop_scores__future_range_bps_prediction_24
            ),
            "predicted_absolute_movement_bps": float(
                row.loop_scores__absolute_return_bps_prediction_24
            ),
        }
        for column, expected in expected_values.items():
            actual = recorded[column]
            if isinstance(expected, str):
                if str(actual) != expected:
                    failures.append(f"sampled_source_{column}")
            elif not close(actual, expected):
                failures.append(f"sampled_source_{column}")
    return not failures, (
        f"symbol={symbol}; motif_samples={len(sample)}; "
        f"loop_movement_samples={len(structural_samples)}; failures={failures[:5]}"
    )


def verify_outcomes(outcomes: pd.DataFrame, contract: Mapping[str, Any]) -> tuple[bool, str]:
    costs = cast(Mapping[str, Any], contract["costs"])
    cost = float(costs["round_trip_bps"])
    multiple = float(costs["primary_dead_band_cost_multiple"])
    scored = outcomes.loc[outcomes["score_status"].eq("scored")].copy()
    gross = pd.to_numeric(scored["gross_long_return_bps"], errors="raise")
    expected = expected_target(gross, cost, multiple)
    checks = [
        len(scored)
        == int(cast(Mapping[str, Any], contract["population"])["expected_exact_outcome_rows"]),
        np.allclose(scored["gross_short_return_bps"], -gross),
        np.allclose(scored["net_long_return_bps"], gross - cost),
        np.allclose(scored["net_short_return_bps"], -gross - cost),
        np.allclose(scored["directional_move_threshold_bps"], multiple * cost),
        expected.eq(scored["target"].astype(str)).all(),
        outcomes.loc[~outcomes["score_status"].eq("scored"), "target"].eq("UNAVAILABLE").all(),
    ]
    return bool(all(checks)), (
        f"scored={len(scored)}; unavailable={len(outcomes) - len(scored)}; "
        f"targets={scored['target'].value_counts().sort_index().to_dict()}"
    )


def verify_timestamps_and_provider_samples(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    delayed: pd.DataFrame,
    first_touch: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[bool, str]:
    joined = features[["opportunity_id", "decision_timestamp"]].merge(
        outcomes,
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
    )
    scored = joined.loc[joined["score_status"].eq("scored")].copy()
    decision = pd.to_datetime(scored["decision_timestamp"], utc=True)
    entry = pd.to_datetime(scored["entry_timestamp"], utc=True)
    terminal = pd.to_datetime(scored["terminal_timestamp"], utc=True)
    failures: list[str] = []
    if not entry.eq(decision + pd.Timedelta(minutes=5)).all():
        failures.append("next_provider_open_timestamp")
    if not terminal.eq(decision + pd.Timedelta(minutes=120)).all():
        failures.append("fixed_24_bar_terminal_timestamp")
    delayed_joined = features[["opportunity_id", "decision_timestamp"]].merge(
        delayed,
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
    )
    delayed_scored = delayed_joined.loc[delayed_joined["score_status"].eq("scored")]
    delayed_entry = pd.to_datetime(delayed_scored["entry_timestamp"], utc=True)
    delayed_decision = pd.to_datetime(delayed_scored["decision_timestamp"], utc=True)
    delayed_terminal = pd.to_datetime(delayed_scored["terminal_timestamp"], utc=True)
    if not delayed_entry.eq(delayed_decision + pd.Timedelta(minutes=10)).all():
        failures.append("one_bar_delay_entry_timestamp")
    if not delayed_terminal.eq(delayed_decision + pd.Timedelta(minutes=120)).all():
        failures.append("one_bar_delay_restarted_terminal")

    provider_root = Path(str(cast(Mapping[str, Any], contract["population"])["provider_root"]))
    sample_positions = np.linspace(0, len(scored) - 1, num=9, dtype=int)
    checked = 0
    provider_cache: dict[str, pd.DataFrame] = {}
    first_touch_by_id = first_touch.set_index("opportunity_id")
    for position in sample_positions:
        row = scored.iloc[int(position)]
        symbol = str(row["symbol"])
        if symbol not in provider_cache:
            path = provider_root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
            provider = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
            provider["timestamp"] = pd.to_datetime(provider["timestamp"], utc=True, errors="raise")
            provider_cache[symbol] = provider.set_index("timestamp").sort_index()
        provider = provider_cache[symbol]
        entry_timestamp = pd.Timestamp(row["entry_timestamp"])
        terminal_timestamp = pd.Timestamp(row["terminal_timestamp"])
        if entry_timestamp not in provider.index or terminal_timestamp not in provider.index:
            failures.append(f"provider_timestamp:{row['opportunity_id']}")
            continue
        path_rows = provider.loc[entry_timestamp:terminal_timestamp]
        if len(path_rows) != 24:
            failures.append(f"provider_path_length:{row['opportunity_id']}")
            continue
        entry_open = float(path_rows.iloc[0]["open"])
        terminal_close = float(path_rows.iloc[-1]["close"])
        gross = 10_000.0 * (terminal_close / entry_open - 1.0)
        future_range = (
            10_000.0 * (float(path_rows["high"].max()) - float(path_rows["low"].min())) / entry_open
        )
        maximum_high = float(path_rows["high"].max())
        minimum_low = float(path_rows["low"].min())
        mfe_long = 10_000.0 * (maximum_high / entry_open - 1.0)
        mae_long = 10_000.0 * (minimum_low / entry_open - 1.0)
        mfe_short = 10_000.0 * (1.0 - minimum_low / entry_open)
        mae_short = 10_000.0 * (1.0 - maximum_high / entry_open)
        if not all(
            (
                close(entry_open, row["entry_open"]),
                close(terminal_close, row["terminal_close"]),
                close(gross, row["gross_long_return_bps"]),
                close(future_range, row["future_high_low_range_bps"]),
                close(mfe_long, row["mfe_long_bps"]),
                close(mae_long, row["mae_long_bps"]),
                close(mfe_short, row["mfe_short_bps"]),
                close(mae_short, row["mae_short_bps"]),
            )
        ):
            failures.append(f"provider_reconstruction:{row['opportunity_id']}")
        touch = cast(pd.Series, first_touch_by_id.loc[str(row["opportunity_id"])])
        barrier = float(cast(Any, touch["first_touch_barrier_bps"]))
        upper = entry_open * (1.0 + barrier / 10_000.0)
        lower = entry_open * (1.0 - barrier / 10_000.0)
        expected_touch = "NEITHER"
        expected_step: int | None = None
        for step, bar in enumerate(path_rows.itertuples(), start=1):
            tuple_bar = cast(Any, bar)
            if float(tuple_bar.open) >= upper:
                expected_touch, expected_step = "UPPER_FIRST", step
                break
            if float(tuple_bar.open) <= lower:
                expected_touch, expected_step = "LOWER_FIRST", step
                break
            upper_touch = float(tuple_bar.high) >= upper
            lower_touch = float(tuple_bar.low) <= lower
            if upper_touch and lower_touch:
                expected_touch, expected_step = "SAME_BAR_DUAL_TOUCH", step
                break
            if upper_touch:
                expected_touch, expected_step = "UPPER_FIRST", step
                break
            if lower_touch:
                expected_touch, expected_step = "LOWER_FIRST", step
                break
        recorded_step: int | None = (
            None
            if pd.isna(touch["first_touch_step"])
            else int(cast(Any, touch["first_touch_step"]))
        )
        if str(touch["first_touch_target"]) != expected_touch or recorded_step != expected_step:
            failures.append(f"first_touch_reconstruction:{row['opportunity_id']}")
        checked += 1
    return not failures, f"provider_samples={checked}; failures={failures[:5]}"


def verify_movement(features: pd.DataFrame, movement: pd.DataFrame) -> tuple[bool, str]:
    joined = features[
        ["opportunity_id", "predicted_future_range_bps", "movement_permission"]
    ].merge(
        movement[["opportunity_id", "predicted_future_range_bps", "movement_permission"]],
        on="opportunity_id",
        suffixes=("_feature", "_ledger"),
        validate="one_to_one",
    )
    predicted = pd.to_numeric(joined["predicted_future_range_bps_feature"], errors="coerce")
    expected = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    expected.loc[predicted.notna()] = predicted.loc[predicted.notna()].gt(30.0)
    feature_permission = joined["movement_permission_feature"].astype("boolean")
    ledger_permission = joined["movement_permission_ledger"].astype("boolean")
    passed = all(
        (
            np.allclose(
                joined["predicted_future_range_bps_feature"],
                joined["predicted_future_range_bps_ledger"],
                equal_nan=True,
            ),
            expected.equals(feature_permission),
            expected.equals(ledger_permission),
        )
    )
    return bool(
        passed
    ), f"permitted={int(expected.sum())}; unavailable={int(predicted.isna().sum())}"


def verify_candidate_registry(
    root: Path,
    registry: pd.DataFrame,
    contract: Mapping[str, Any],
    schema: Mapping[str, Any],
    scored: pd.DataFrame,
) -> tuple[bool, str]:
    search = cast(Mapping[str, Any], contract["search"])
    counts = registry["stage"].value_counts().to_dict()
    univariate_pairwise = int(counts.get("univariate", 0) + counts.get("pairwise", 0))
    triples = int(counts.get("three_condition", 0))
    trees = int(counts.get("shallow_tree", 0))
    failures: list[str] = []
    if univariate_pairwise > int(search["univariate_and_pairwise_cap"]):
        failures.append("univariate_pairwise_cap")
    if triples > int(search["three_condition_cap"]):
        failures.append("three_condition_cap")
    if trees > int(search["tree_candidate_cap"]):
        failures.append("tree_cap")
    enabled_features = [
        str(row["name"])
        for row in cast(list[dict[str, Any]], schema["features"])
        if bool(row.get("condition_enabled"))
    ]
    discovery = scored.loc[scored["chronology_stage"].eq("discovery")]
    level_counts = {
        feature: int(discovery[feature].dropna().nunique()) for feature in enabled_features
    }
    univariate_space = 2 * sum(level_counts.values())
    pairwise_space = 2 * sum(
        level_counts[left] * level_counts[right]
        for index, left in enumerate(sorted(level_counts))
        for right in sorted(level_counts)[index + 1 :]
    )
    expected_univariate = min(univariate_space, int(search["univariate_and_pairwise_cap"]) // 2)
    expected_pairwise = min(
        pairwise_space, int(search["univariate_and_pairwise_cap"]) - expected_univariate
    )
    if int(counts.get("univariate", 0)) != expected_univariate:
        failures.append("balanced_univariate_allocation")
    if int(counts.get("pairwise", 0)) != expected_pairwise:
        failures.append("balanced_pairwise_allocation")
    search_space = cast(dict[str, Any], load_json(root / "candidate_search_space.json"))
    if not (
        int(search_space["observed_univariate_directional_candidates"]) == univariate_space
        and int(search_space["observed_pairwise_directional_candidates"]) == pairwise_space
        and int(search_space["broad_examined_directional_candidates"])
        == univariate_pairwise
    ):
        failures.append("candidate_search_space_manifest")
    if int(registry["condition_count"].max()) > int(search["maximum_conditions"]):
        failures.append("condition_complexity")
    forbidden = {
        "symbol",
        "symbol_norm",
        "month",
        "hindsight_episode",
        "future_route",
        "realised_child",
        "mfe",
        "mae",
        "target",
        "payoff",
    }
    for row in registry.itertuples(index=False):
        conditions = cast(list[dict[str, Any]], json.loads(str(row.conditions_json)))
        names = {str(condition["feature"]) for condition in conditions}
        if names & forbidden or any(
            name.startswith(("future_route", "mfe_", "mae_", "loop_score_"))
            or "payoff" in name
            or name.startswith("target")
            for name in names
        ):
            failures.append(f"forbidden_condition:{row.signature_id}")
            break
        if len(conditions) > 3:
            failures.append(f"too_complex:{row.signature_id}")
            break
    support_reasons = {
        "insufficient_rows",
        "insufficient_sessions",
        "insufficient_stocks",
        "stock_concentration",
        "insufficient_months",
        "insufficient_directional_outcomes",
    }
    support = cast(Mapping[str, Any], contract["support"])
    for row in registry.itertuples(index=False):
        conditions = cast(list[dict[str, Any]], json.loads(str(row.conditions_json)))
        selected = discovery.loc[condition_mask(discovery, conditions)]
        expected_reasons: set[str] = set()
        if len(selected) < int(support["minimum_rows"]):
            expected_reasons.add("insufficient_rows")
        if selected["session"].nunique() < int(support["minimum_independent_sessions"]):
            expected_reasons.add("insufficient_sessions")
        if selected["symbol"].nunique() < int(support["minimum_independent_stocks"]):
            expected_reasons.add("insufficient_stocks")
        maximum_fraction = (
            float(selected["symbol"].value_counts(normalize=True).max())
            if len(selected)
            else math.nan
        )
        if len(selected) and maximum_fraction > float(
            support["maximum_single_stock_row_fraction"]
        ):
            expected_reasons.add("stock_concentration")
        if selected["session"].astype(str).str[:7].nunique() < int(
            support["minimum_calendar_months"]
        ):
            expected_reasons.add("insufficient_months")
        direction = str(row.direction)
        if int(selected["target"].eq(direction).sum()) < int(
            support["minimum_relevant_direction_outcomes"]
        ):
            expected_reasons.add("insufficient_directional_outcomes")
        recorded_reasons = set(json.loads(str(row.rejection_reasons_json)))
        if recorded_reasons & support_reasons != expected_reasons:
            failures.append(f"support_reconstruction:{row.signature_id}")
            break
        comparisons = {
            "rows": len(selected),
            "sessions": selected["session"].nunique(),
            "stocks": selected["symbol"].nunique(),
            "months": selected["session"].astype(str).str[:7].nunique(),
        }
        if any(int(getattr(row, key)) != int(value) for key, value in comparisons.items()):
            failures.append(f"candidate_metric_reconstruction:{row.signature_id}")
            break
    supported = registry["rejection_reasons_json"].map(
        lambda raw: not bool(set(json.loads(str(raw))) & support_reasons)
    )
    expected_q = np.ones(len(registry), dtype=float)
    supported_p = pd.to_numeric(registry.loc[supported, "raw_p_value"], errors="raise").to_numpy(
        float
    )
    expected_q[np.flatnonzero(supported.to_numpy(bool))] = adjust_p_values(
        supported_p.tolist(), "fdr_bh"
    )
    if not np.allclose(expected_q, registry["fdr_q_value"].to_numpy(float)):
        failures.append("fdr_reconstruction")
    if (
        not registry.loc[~registry["discovery_eligible"].astype(bool), "rejection_reasons_json"]
        .notna()
        .all()
    ):
        failures.append("failed_candidates_missing_reason")
    return not failures, f"counts={counts}; failures={failures[:5]}"


def independent_signature_metrics(
    frame: pd.DataFrame, signature: Mapping[str, Any]
) -> dict[str, object]:
    selected = frame.loc[
        condition_mask(frame, cast(Sequence[Mapping[str, Any]], signature["conditions"]))
    ]
    direction = str(signature["direction"])
    payoff_column = "long_net_bps" if direction == "LONG" else "short_net_bps"
    target_rate = float(selected["target"].eq(direction).mean()) if len(selected) else math.nan
    base_rate = float(frame["target"].eq(direction).mean()) if len(frame) else math.nan
    return {
        "rows": len(selected),
        "sessions": selected["session"].nunique(),
        "stocks": selected["symbol"].nunique(),
        "months": selected["session"].astype(str).str[:7].nunique(),
        "mean_directional_net_bps": float(selected[payoff_column].mean()),
        "directional_lift": target_rate - base_rate,
        "long_count": int(selected["target"].eq("LONG").sum()),
        "short_count": int(selected["target"].eq("SHORT").sum()),
        "neutral_count": int(selected["target"].eq("NEUTRAL").sum()),
    }


def verify_signature_metrics(
    scored: pd.DataFrame,
    discovery_library: list[dict[str, Any]],
    validation_metrics: pd.DataFrame,
) -> tuple[bool, str]:
    failures: list[str] = []
    validation_by_id = (
        validation_metrics.set_index("signature_id")
        if not validation_metrics.empty
        else pd.DataFrame()
    )
    for entry in discovery_library:
        signature = cast(dict[str, Any], entry["signature"])
        signature_id = str(signature["signature_id"])
        discovery = independent_signature_metrics(
            scored.loc[scored["chronology_stage"].eq("discovery")], signature
        )
        recorded_discovery = cast(Mapping[str, Any], entry["discovery_metrics"])
        for key in (
            "rows",
            "sessions",
            "stocks",
            "months",
            "long_count",
            "short_count",
            "neutral_count",
            "mean_directional_net_bps",
            "directional_lift",
        ):
            if not close(discovery[key], recorded_discovery[key]):
                failures.append(f"discovery:{signature_id}:{key}")
                break
        validation = independent_signature_metrics(
            scored.loc[scored["chronology_stage"].eq("validation")], signature
        )
        if validation_by_id.empty or signature_id not in validation_by_id.index:
            failures.append(f"validation_missing:{signature_id}")
            continue
        recorded_validation = validation_by_id.loc[signature_id]
        for key in ("rows", "sessions", "stocks", "mean_directional_net_bps", "directional_lift"):
            if not close(validation[key], recorded_validation[key]):
                failures.append(f"validation:{signature_id}:{key}")
                break
    return not failures, f"reconstructed={len(discovery_library)}; failures={failures[:5]}"


def independent_track_a_lead_qualification(
    library: list[dict[str, Any]],
    final_metrics: pd.DataFrame,
    stress: pd.DataFrame,
    calibration: pd.DataFrame,
    comparators: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Rebuild every provisional-lead criterion from final and stress artifacts."""

    support = cast(Mapping[str, Any], contract["support"])
    final = final_metrics.set_index("signature_id") if not final_metrics.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for entry in library:
        signature_id = str(entry["signature"]["signature_id"])
        direction = str(entry["signature"]["direction"])
        reasons: list[str] = []
        if final.empty or signature_id not in final.index:
            reasons.append("not_scored_in_final_opened_holdout")
        else:
            metric = cast(pd.Series, final.loc[signature_id])
            checks = (
                (float(metric["mean_directional_net_bps"]) <= 0.0, "final_payoff_not_positive"),
                (float(metric["directional_lift"]) <= 0.0, "final_lift_not_positive"),
                (
                    float(metric["twice_cost_mean_net_bps"]) <= 0.0,
                    "final_twice_cost_not_positive",
                ),
                (
                    float(metric["top_stock_absolute_contribution_share"])
                    > float(support["maximum_top_stock_absolute_payoff_share"]),
                    "final_stock_concentration",
                ),
                (
                    float(metric["maximum_single_stock_row_fraction"])
                    > float(support["maximum_single_stock_row_fraction"]),
                    "final_stock_row_concentration",
                ),
                (
                    float(metric["top_month_absolute_contribution_share"])
                    > float(support["maximum_top_month_absolute_payoff_share"]),
                    "final_month_concentration",
                ),
                (
                    float(metric["positive_stock_fraction"])
                    <= float(support["minimum_positive_stock_fraction"]),
                    "final_stock_consistency",
                ),
                (
                    float(metric["positive_month_fraction"])
                    <= float(
                        cast(Mapping[str, Any], contract["validation_survival"])[
                            "positive_month_fraction_strictly_greater_than"
                        ]
                    ),
                    "final_month_consistency",
                ),
                (int(metric["rows"]) < int(support["minimum_rows"]), "final_insufficient_rows"),
                (
                    int(metric["sessions"])
                    < int(support["minimum_independent_sessions"]),
                    "final_insufficient_sessions",
                ),
                (
                    int(metric["stocks"]) < int(support["minimum_independent_stocks"]),
                    "final_insufficient_stocks",
                ),
                (
                    int(metric["months"]) < int(support["minimum_calendar_months"]),
                    "final_insufficient_months",
                ),
            )
            for failed, reason in checks:
                if failed:
                    reasons.append(reason)
            relevant_count = int(
                metric["long_count"] if direction == "LONG" else metric["short_count"]
            )
            if relevant_count < int(support["minimum_relevant_direction_outcomes"]):
                reasons.append("final_insufficient_directional_outcomes")
            signature_stress = stress.loc[stress["signature_id"].eq(signature_id)]
            required = {
                "one_bar_execution_delay_same_terminal": "delay_not_positive",
                "remove_best_stock": "best_stock_removal_not_positive",
                "remove_top_five_stocks": "top_five_stock_removal_not_positive",
            }
            for stage in ("validation", "final_opened_holdout"):
                stage_stress = signature_stress.loc[
                    signature_stress["chronology_stage"].eq(stage)
                ]
                for stress_name, reason in required.items():
                    values = stage_stress.loc[
                        stage_stress["stress"].eq(stress_name),
                        "mean_directional_net_bps",
                    ]
                    if values.empty or not values.gt(0.0).all():
                        reasons.append(f"{stage}_{reason}")
                for episode_name in ("remove_best_episode", "remove_top_five_episodes"):
                    episode = stage_stress.loc[stage_stress["stress"].eq(episode_name)]
                    if episode.empty or not episode["status"].eq("available").all():
                        reasons.append(f"{stage}_{episode_name}_unavailable")
                    elif not episode["mean_directional_net_bps"].gt(0.0).all():
                        reasons.append(f"{stage}_{episode_name}_not_positive")
                neighbours = stage_stress.loc[
                    stage_stress["stress"].eq("adjacent_threshold_neighbour")
                ]
                if len(neighbours) and not neighbours["mean_directional_net_bps"].gt(0.0).all():
                    reasons.append(f"{stage}_adjacent_threshold_incompatible")
            signature_calibration = calibration.loc[
                calibration["signature_id"].eq(signature_id)
            ]
            if signature_calibration.empty or not signature_calibration[
                "reasonably_calibrated"
            ].astype(bool).all():
                reasons.append("probability_not_reasonably_calibrated")
            signature_comparators = comparators.loc[
                comparators["signature_id"].eq(signature_id)
            ]
            if signature_comparators.empty or not (
                signature_comparators["stronger_than_momentum"].astype(bool).all()
                and signature_comparators["stronger_than_reversal"].astype(bool).all()
            ):
                reasons.append("not_stronger_than_momentum_and_reversal")
        rows.append(
            {
                "signature_id": signature_id,
                "direction": direction,
                "provisional_prospective_lead": not reasons,
                "rejection_reasons_json": json.dumps(
                    sorted(set(reasons)), separators=(",", ":")
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "signature_id",
            "direction",
            "provisional_prospective_lead",
            "rejection_reasons_json",
        ],
    )


def verify_libraries_and_chronology(
    root: Path,
    discovery_library: list[dict[str, Any]],
    validation_metrics: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[bool, str, list[dict[str, Any]]]:
    survivors = cast(
        list[dict[str, Any]], load_json(root / "frozen_validation_survivor_library.json")
    )
    long_library = cast(list[dict[str, Any]], load_json(root / "long_signature_library.json"))
    short_library = cast(list[dict[str, Any]], load_json(root / "short_signature_library.json"))
    discovery_ids = {str(entry["signature"]["signature_id"]) for entry in discovery_library}
    survivor_ids = {str(entry["signature"]["signature_id"]) for entry in survivors}
    long_ids = {str(entry["signature"]["signature_id"]) for entry in long_library}
    short_ids = {str(entry["signature"]["signature_id"]) for entry in short_library}
    failures: list[str] = []
    if not survivor_ids <= discovery_ids:
        failures.append("validation_regenerated_rules")
    if (
        len([entry for entry in discovery_library if entry["signature"]["direction"] == "LONG"])
        > 10
    ):
        failures.append("discovery_long_cap")
    if (
        len([entry for entry in discovery_library if entry["signature"]["direction"] == "SHORT"])
        > 10
    ):
        failures.append("discovery_short_cap")
    if len(long_library) > 5 or len(short_library) > 5:
        failures.append("validation_library_cap")
    if long_ids & short_ids:
        failures.append("long_short_library_overlap")
    if not (long_ids | short_ids) <= survivor_ids:
        failures.append("prospective_library_not_validation_survivor")
    final_metrics = safe_csv(root / "final_opened_holdout_signature_metrics.csv")
    stress = safe_csv(root / "cost_and_delay_stress_results.csv")
    calibration = safe_csv(root / "individual_signature_calibration_metrics.csv")
    comparators = safe_csv(root / "individual_signature_baseline_comparison.csv")
    expected_qualification = independent_track_a_lead_qualification(
        survivors,
        final_metrics,
        stress,
        calibration,
        comparators,
        contract,
    ).sort_values(["direction", "signature_id"], kind="mergesort")
    recorded_qualification = safe_csv(
        root / "provisional_prospective_lead_qualification.csv"
    ).sort_values(["direction", "signature_id"], kind="mergesort")
    qualification_columns = [
        "signature_id",
        "direction",
        "provisional_prospective_lead",
        "rejection_reasons_json",
    ]
    if list(recorded_qualification.columns) != qualification_columns or len(
        recorded_qualification
    ) != len(expected_qualification):
        failures.append("lead_qualification_shape")
    elif len(expected_qualification):
        recorded_flags = recorded_qualification["provisional_prospective_lead"].map(
            lambda value: str(value).lower() == "true"
        )
        expected_flags = expected_qualification["provisional_prospective_lead"].astype(bool)
        for column in ("signature_id", "direction", "rejection_reasons_json"):
            if not recorded_qualification[column].astype(str).reset_index(drop=True).equals(
                expected_qualification[column].astype(str).reset_index(drop=True)
            ):
                failures.append(f"lead_qualification:{column}")
        if not recorded_flags.reset_index(drop=True).equals(expected_flags.reset_index(drop=True)):
            failures.append("lead_qualification:flag")
    expected_long_ids = set(
        expected_qualification.loc[
            expected_qualification["provisional_prospective_lead"].astype(bool)
            & expected_qualification["direction"].eq("LONG"),
            "signature_id",
        ].astype(str)
    )
    expected_short_ids = set(
        expected_qualification.loc[
            expected_qualification["provisional_prospective_lead"].astype(bool)
            & expected_qualification["direction"].eq("SHORT"),
            "signature_id",
        ].astype(str)
    )
    if long_ids != expected_long_ids or short_ids != expected_short_ids:
        failures.append("prospective_library_qualification_mismatch")
    prospective_library = cast(
        list[dict[str, Any]], load_json(root / "provisional_prospective_lead_library.json")
    )
    if {
        str(entry["signature"]["signature_id"]) for entry in prospective_library
    } != expected_long_ids | expected_short_ids:
        failures.append("provisional_lead_library_mismatch")
    if not validation_metrics.empty:
        expected_holm = adjust_p_values(
            pd.to_numeric(validation_metrics["raw_p_value"], errors="raise").tolist(), "holm"
        )
        if not np.allclose(expected_holm, validation_metrics["holm_adjusted_p_value"]):
            failures.append("validation_holm_reconstruction")
    chronology = cast(Mapping[str, Mapping[str, Any]], contract["chronology"])
    ordered_names = ("discovery", "validation", "final_opened_holdout")
    boundaries = [
        (
            pd.Timestamp(str(chronology[name]["start"])),
            pd.Timestamp(str(chronology[name]["end_exclusive"])),
        )
        for name in ordered_names
    ]
    if not all(
        start < end and (index == 0 or boundaries[index - 1][1] <= start)
        for index, (start, end) in enumerate(boundaries)
    ):
        failures.append("chronology_contract")
    discovery_seal = cast(dict[str, Any], load_json(root / "discovery_stage_seal.json"))
    validation_seal = cast(dict[str, Any], load_json(root / "validation_stage_seal.json"))
    if (
        str(discovery_seal.get("frozen_discovery_library_sha256"))
        != sha256_file(root / "frozen_discovery_signature_library.json")
        or str(discovery_seal.get("frozen_neutral_discovery_library_sha256"))
        != sha256_file(root / "frozen_neutral_discovery_library.json")
        or discovery_seal.get("validation_or_final_opened_before_seal") is not False
    ):
        failures.append("discovery_stage_seal")
    if (
        str(validation_seal.get("frozen_validation_survivor_library_sha256"))
        != sha256_file(root / "frozen_validation_survivor_library.json")
        or str(validation_seal.get("frozen_neutral_survivor_library_sha256"))
        != sha256_file(root / "neutral_veto_library.json")
        or validation_seal.get("final_opened_before_seal") is not False
    ):
        failures.append("validation_stage_seal")
    movement_root = root / "movement_permitted"
    for seal_name, library_name, hash_field in (
        ("discovery_stage_seal.json", "frozen_discovery_library.json", "frozen_library_sha256"),
        (
            "validation_stage_seal.json",
            "frozen_validation_survivors.json",
            "frozen_survivor_library_sha256",
        ),
    ):
        seal = cast(dict[str, Any], load_json(movement_root / seal_name))
        if str(seal.get(hash_field)) != sha256_file(movement_root / library_name):
            failures.append(f"movement_{seal_name}")
    movement_registry = safe_csv(movement_root / "complete_candidate_registry.csv")
    gate_rediscovered = (
        not movement_registry.empty
        and movement_registry["conditions_json"]
        .astype(str)
        .str.contains('"feature":"movement_permission"', regex=False)
        .any()
    )
    if gate_rediscovered:
        failures.append("movement_surface_rediscovered_gate")
    movement_population = pd.read_parquet(
        movement_root / "movement_permitted_scoring_population.parquet"
    )
    if not movement_population["movement_permission"].astype("boolean").eq(True).all():
        failures.append("movement_surface_contains_nonpermitted_row")
    return (
        not failures,
        f"discovery={len(discovery_library)}; survivors={len(survivors)}",
        survivors,
    )


def verify_atlas_controller(
    features: pd.DataFrame,
    decisions: pd.DataFrame,
    survivors: list[dict[str, Any]],
) -> tuple[bool, str]:
    long_votes = np.zeros(len(features), dtype=int)
    short_votes = np.zeros(len(features), dtype=int)
    long_values = np.zeros(len(features), dtype=float)
    short_values = np.zeros(len(features), dtype=float)
    required_features = {
        str(condition["feature"])
        for entry in survivors
        for condition in cast(list[dict[str, Any]], entry["signature"]["conditions"])
    }
    missing_required = np.zeros(len(features), dtype=bool)
    for feature in sorted(required_features):
        missing_required |= (
            features[feature].isna().to_numpy()
            if feature in features
            else np.ones(len(features), dtype=bool)
        )
    for entry in survivors:
        signature = cast(dict[str, Any], entry["signature"])
        mask = condition_mask(
            features, cast(Sequence[Mapping[str, Any]], signature["conditions"])
        ).to_numpy(bool)
        value = float(entry["conservative_value_bps"])
        if str(signature["direction"]) == "LONG":
            long_votes += mask.astype(int)
            long_values[mask] += value
        else:
            short_votes += mask.astype(int)
            short_values[mask] += value
    movement_available = features["movement_permission"].notna().to_numpy()
    movement = (
        features["movement_permission"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    conflict = (long_votes > 0) & (short_votes > 0)
    states = np.full(len(features), "NEUTRAL", dtype=object)
    states[
        movement
        & ~missing_required
        & ~conflict
        & (long_votes > 0)
        & (short_votes == 0)
        & (long_values / np.maximum(long_votes, 1) > 0.0)
    ] = "LONG"
    states[
        movement
        & ~missing_required
        & ~conflict
        & (short_votes > 0)
        & (long_votes == 0)
        & (short_values / np.maximum(short_votes, 1) > 0.0)
    ] = "SHORT"
    reasons = np.full(len(features), "no_directional_vote", dtype=object)
    reasons[~movement] = "movement_permission_failed"
    reasons[~movement_available] = "required_causal_feature_unavailable"
    reasons[movement & conflict] = "conflicting_votes"
    reasons[
        movement
        & ~conflict
        & ((long_votes > 0) | (short_votes > 0))
        & (states == "NEUTRAL")
    ] = "non_positive_conservative_value"
    reasons[states != "NEUTRAL"] = "supported_directional_vote"
    reasons[missing_required] = "required_causal_feature_unavailable"
    ordered = decisions.set_index("opportunity_id").loc[features["opportunity_id"]]
    passed = all(
        (
            np.array_equal(long_votes, ordered["long_vote_count"].to_numpy(int)),
            np.array_equal(short_votes, ordered["short_vote_count"].to_numpy(int)),
            np.array_equal(conflict, ordered["conflict"].to_numpy(bool)),
            np.array_equal(states, ordered["predicted_state"].astype(str).to_numpy()),
            np.array_equal(reasons, ordered["reason_code"].astype(str).to_numpy()),
        )
    )
    return bool(passed), (
        f"rows={len(features)}; survivors={len(survivors)}; directional_outputs="
        f"{int(np.isin(states, ['LONG', 'SHORT']).sum())}; conflicts={int(conflict.sum())}"
    )


def verify_atlas_aggregate_metrics(
    scored: pd.DataFrame,
    decisions: pd.DataFrame,
    predictive: pd.DataFrame,
    economic: pd.DataFrame,
) -> tuple[bool, str]:
    failures: list[str] = []
    joined = decisions.merge(
        scored[
            [
                "opportunity_id",
                "decision_timestamp",
                "target",
                "long_net_bps",
                "short_net_bps",
                "round_trip_cost_bps",
            ]
        ],
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
    )
    for stage, group in joined.groupby("chronology_stage", sort=True):
        state = group["predicted_state"].astype(str)
        directional = state.isin(["LONG", "SHORT"])
        payoff = np.where(
            state.eq("LONG"),
            group["long_net_bps"],
            np.where(state.eq("SHORT"), group["short_net_bps"], 0.0),
        ).astype(float)
        batches = (
            group.assign(_payoff=payoff)
            .groupby("decision_timestamp", sort=True)["_payoff"]
            .sum()
        )
        cumulative = batches.cumsum().to_numpy(float)
        peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
        drawdown = float(np.max(peaks - cumulative)) if len(cumulative) else math.nan
        recorded_economic = economic.loc[
            economic["model_id"].eq("directional_signature_atlas_v1")
            & economic["chronology_stage"].astype(str).eq(str(stage))
        ]
        if recorded_economic.empty:
            failures.append(f"economic_missing:{stage}")
        else:
            row = recorded_economic.iloc[0]
            comparisons = {
                "opportunities": len(group),
                "directional_outputs": int(directional.sum()),
                "total_net_bps": float(payoff.sum()),
                "net_bps_per_full_opportunity": float(payoff.mean()),
                "maximum_drawdown_bps": drawdown,
            }
            for key, expected in comparisons.items():
                if not close(row[key], expected):
                    failures.append(f"economic:{stage}:{key}")
        recorded_predictive = predictive.loc[
            predictive["model_id"].eq("directional_signature_atlas_v1")
            & predictive["chronology_stage"].astype(str).eq(str(stage))
        ]
        if recorded_predictive.empty:
            failures.append(f"predictive_missing:{stage}")
        else:
            truth = group["target"].astype(str)
            targets = np.column_stack(
                [truth.eq("LONG"), truth.eq("SHORT"), truth.eq("NEUTRAL")]
            ).astype(float)
            probabilities = group[["p_long", "p_short", "p_neutral"]].to_numpy(float)
            macro_brier = float(np.square(probabilities - targets).mean(axis=0).mean())
            if not close(recorded_predictive.iloc[0]["macro_brier"], macro_brier):
                failures.append(f"predictive:{stage}:macro_brier")
    return not failures, f"stages={joined['chronology_stage'].nunique()}; failures={failures[:5]}"


def verify_baselines(features: pd.DataFrame, baselines: pd.DataFrame) -> tuple[bool, str]:
    expected_models = {
        "always_neutral",
        "clock_only_base_rate",
        "prior_static_price_context_multinomial",
        "one_bar_momentum",
        "one_bar_reversal",
        "opening_range_position_sign",
        "current_state_alone",
        "current_state_plus_history",
        "movement_permission_plus_momentum",
        "shallow_logistic_compact_features",
    }
    models = set(baselines["model_id"].astype(str))
    failures: list[str] = []
    if models != expected_models:
        failures.append("baseline_set")
    expected_ids = set(features["opportunity_id"].astype(str))
    for model_id, group in baselines.groupby("model_id", sort=False):
        if set(group["opportunity_id"].astype(str)) != expected_ids:
            failures.append(f"population:{model_id}")
    indexed = features.set_index("opportunity_id")
    for model_id, reverse in (("one_bar_momentum", False), ("one_bar_reversal", True)):
        group = baselines.loc[baselines["model_id"].eq(model_id)].set_index("opportunity_id")
        returns = pd.to_numeric(indexed.loc[group.index, "return_1_scale"], errors="coerce")
        positive = "SHORT" if reverse else "LONG"
        negative = "LONG" if reverse else "SHORT"
        expected = np.where(returns > 0.0, positive, np.where(returns < 0.0, negative, "NEUTRAL"))
        if not np.array_equal(expected, group["predicted_state"].astype(str).to_numpy()):
            failures.append(model_id)
    predecessor = baselines.loc[
        baselines["model_id"].eq("prior_static_price_context_multinomial")
    ]
    if predecessor.empty or "metric_eligible" not in predecessor:
        failures.append("prior_static_price_context_identity")
    else:
        ineligible = predecessor.loc[~predecessor["metric_eligible"].astype(bool)]
        probabilities = ineligible[["p_long", "p_short", "p_neutral"]].to_numpy(float)
        if not (
            predecessor.loc[predecessor["chronology_stage"].eq("development_context"),
                            "metric_eligible"].eq(False).all()
            and np.allclose(probabilities, 1.0 / 3.0)
            and ineligible["predicted_state"].astype(str).eq("NEUTRAL").all()
        ):
            failures.append("prior_static_price_context_prequential_boundary")
    return (
        not failures,
        f"models={len(models)}; rows_per_model={len(baselines) // max(len(models), 1)}",
    )


def verify_null_stress_and_concentration(
    root: Path,
    contract: Mapping[str, Any],
    scored: pd.DataFrame,
    discovery_library: list[dict[str, Any]],
) -> tuple[bool, str]:
    nulls = safe_csv(root / "null_test_results.csv")
    stresses = safe_csv(root / "cost_and_delay_stress_results.csv")
    concentration = safe_csv(root / "concentration_results.csv")
    expected_nulls = set(cast(list[str], contract["nulls"]))
    normalised_expected = {
        value.replace("random_atlas_coverage", "random_atlas_controller_coverage")
        for value in expected_nulls
    }
    failures: list[str] = []
    if set(nulls["null"].astype(str)) != normalised_expected:
        failures.append("null_family_set")
    if bool(nulls["similar_persistent_validation_performance"].astype(bool).any()):
        failures.append("null_persistent_performance")
    required_stresses = {
        "twice_cost",
        "one_bar_execution_delay_same_terminal",
        "dead_band_1x_cost",
        "dead_band_3x_cost",
        "movement_permission_on",
        "remove_best_stock",
        "remove_top_five_stocks",
        "remove_best_month",
        "remove_best_episode",
        "remove_top_five_episodes",
        "exclude_weakest_historical_activity_cohort",
        "coarse_clock_bin",
        "adjacent_threshold_neighbour",
    }
    observed_stresses = set(stresses["stress"].astype(str)) if not stresses.empty else set()
    missing_stresses = sorted(required_stresses - observed_stresses) if discovery_library else []
    if missing_stresses:
        failures.append(f"missing_stresses:{missing_stresses}")
    required_dimensions = {
        "symbol",
        "period",
        "month",
        "decision_clock",
        "current_state",
        "state_motif_3",
        "parent_loop_family",
        "hindsight_episode",
    }
    observed_dimensions = (
        set(concentration["dimension"].astype(str)) if not concentration.empty else set()
    )
    if discovery_library and required_dimensions - observed_dimensions:
        failures.append("concentration_dimensions")
    random_atlas = nulls.loc[nulls["null"].eq("random_atlas_controller_coverage_matched")]
    if random_atlas.empty or int(random_atlas.iloc[0].get("draws", 0)) != int(
        cast(Mapping[str, Any], contract["random_atlas_null"])["draws"]
    ):
        failures.append("random_atlas_draw_count")
    if discovery_library:
        entry = discovery_library[0]
        signature = cast(dict[str, Any], entry["signature"])
        signature_id = str(signature["signature_id"])
        direction = str(signature["direction"])
        payoff_column = "long_net_bps" if direction == "LONG" else "short_net_bps"
        selected = scored.loc[
            condition_mask(
                scored, cast(Sequence[Mapping[str, Any]], signature["conditions"])
            )
        ].copy()
        if not selected.empty:
            sample_symbol = str(sorted(selected["symbol"].astype(str).unique())[0])
            group = selected.loc[selected["symbol"].astype(str).eq(sample_symbol)]
            recorded = concentration.loc[
                concentration["signature_id"].eq(signature_id)
                & concentration["dimension"].eq("symbol")
                & concentration["value"].astype(str).eq(sample_symbol)
            ]
            if recorded.empty or not (
                int(recorded.iloc[0]["rows"]) == len(group)
                and close(recorded.iloc[0]["total_net_bps"], group[payoff_column].sum())
            ):
                failures.append("sampled_concentration_reconstruction")
        stage = "validation"
        stage_frame = scored.loc[scored["chronology_stage"].eq(stage)]
        stage_selected = stage_frame.loc[
            condition_mask(
                stage_frame, cast(Sequence[Mapping[str, Any]], signature["conditions"])
            )
        ]
        twice_mean = float(
            (stage_selected[payoff_column] - stage_selected["round_trip_cost_bps"]).mean()
        )
        recorded_twice = stresses.loc[
            stresses["signature_id"].eq(signature_id)
            & stresses["chronology_stage"].eq(stage)
            & stresses["stress"].eq("twice_cost")
        ]
        if recorded_twice.empty or not close(
            twice_mean, recorded_twice.iloc[0]["mean_directional_net_bps"]
        ):
            failures.append("sampled_twice_cost_reconstruction")
    return not failures, (
        f"nulls={len(nulls)}; stress_rows={len(stresses)}; "
        f"dimensions={sorted(observed_dimensions)}; failures={failures[:3]}"
    )


def independent_relative_persistence_qualification(
    library: list[dict[str, Any]],
    validation_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    stress: pd.DataFrame,
    nulls: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Reconstruct strict Track B persistence without importing runner logic."""

    support = cast(Mapping[str, Any], contract["support"])
    validation = (
        validation_metrics.set_index("signature_id")
        if not validation_metrics.empty
        else pd.DataFrame()
    )
    final = final_metrics.set_index("signature_id") if not final_metrics.empty else pd.DataFrame()
    null_failed = bool(
        not nulls.empty
        and nulls["similar_persistent_validation_performance"].fillna(False).astype(bool).any()
    )
    rows: list[dict[str, Any]] = []
    for entry in library:
        signature_id = str(entry["signature"]["signature_id"])
        direction = str(entry["signature"]["direction"])
        reasons: list[str] = []
        for stage, metrics in (
            ("validation", validation),
            ("final_opened_holdout", final),
        ):
            if metrics.empty or signature_id not in metrics.index:
                reasons.append(f"{stage}_metrics_missing")
                continue
            metric = cast(pd.Series, metrics.loc[signature_id])
            checks = (
                (float(metric["mean_directional_net_bps"]) <= 0.0, "payoff_not_positive"),
                (float(metric["directional_lift"]) <= 0.0, "lift_not_positive"),
                (float(metric["twice_cost_mean_net_bps"]) <= 0.0, "twice_cost_not_positive"),
                (int(metric["rows"]) < int(support["minimum_rows"]), "insufficient_rows"),
                (
                    int(metric["sessions"])
                    < int(support["minimum_independent_sessions"]),
                    "insufficient_sessions",
                ),
                (
                    int(metric["stocks"]) < int(support["minimum_independent_stocks"]),
                    "insufficient_stocks",
                ),
                (
                    int(metric["months"]) < int(support["minimum_calendar_months"]),
                    "insufficient_months",
                ),
                (
                    float(metric["maximum_single_stock_row_fraction"])
                    > float(support["maximum_single_stock_row_fraction"]),
                    "stock_row_concentration",
                ),
                (
                    float(metric["top_stock_absolute_contribution_share"])
                    > float(support["maximum_top_stock_absolute_payoff_share"]),
                    "stock_payoff_concentration",
                ),
                (
                    float(metric["top_month_absolute_contribution_share"])
                    > float(support["maximum_top_month_absolute_payoff_share"]),
                    "month_payoff_concentration",
                ),
                (
                    float(metric["positive_stock_fraction"])
                    <= float(support["minimum_positive_stock_fraction"]),
                    "stock_consistency",
                ),
                (
                    float(metric["positive_month_fraction"])
                    <= float(
                        cast(Mapping[str, Any], contract["validation_survival"])[
                            "positive_month_fraction_strictly_greater_than"
                        ]
                    ),
                    "month_consistency",
                ),
                (
                    float(metric["opposite_direction_excess"])
                    > float(support["maximum_opposite_direction_excess"]),
                    "opposite_direction_not_controlled",
                ),
            )
            for failed, reason in checks:
                if failed:
                    reasons.append(f"{stage}_{reason}")
            relevant_count = int(
                metric["long_count"] if direction == "LONG" else metric["short_count"]
            )
            if relevant_count < int(support["minimum_relevant_direction_outcomes"]):
                reasons.append(f"{stage}_insufficient_directional_outcomes")
            stage_stress = stress.loc[
                stress["signature_id"].eq(signature_id)
                & stress["chronology_stage"].eq(stage)
            ]
            for stress_name in (
                "one_bar_execution_delay_same_terminal",
                "remove_best_stock",
                "remove_top_five_stocks",
                "remove_best_month",
            ):
                stress_rows = stage_stress.loc[stage_stress["stress"].eq(stress_name)]
                if stress_rows.empty or not stress_rows["mean_directional_net_bps"].gt(0.0).all():
                    reasons.append(f"{stage}_{stress_name}_not_positive")
            neighbours = stage_stress.loc[
                stage_stress["stress"].eq("adjacent_threshold_neighbour")
            ]
            if len(neighbours) and not neighbours["mean_directional_net_bps"].gt(0.0).all():
                reasons.append(f"{stage}_adjacent_threshold_incompatible")
            for episode_name in ("remove_best_episode", "remove_top_five_episodes"):
                episode = stage_stress.loc[stage_stress["stress"].eq(episode_name)]
                if episode.empty or not episode["status"].eq("available").all():
                    reasons.append(f"{stage}_{episode_name}_unavailable")
                elif not episode["mean_directional_net_bps"].gt(0.0).all():
                    reasons.append(f"{stage}_{episode_name}_not_positive")
        if null_failed:
            reasons.append("null_family_similar_persistent_validation_performance")
        rows.append(
            {
                "signature_id": signature_id,
                "direction": direction,
                "strict_persistent_relative_signature": not reasons,
                "rejection_reasons_json": json.dumps(
                    sorted(set(reasons)), separators=(",", ":")
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "signature_id",
            "direction",
            "strict_persistent_relative_signature",
            "rejection_reasons_json",
        ],
    )


def verify_relative_atlas_baseline_comparison(
    relative: pd.DataFrame,
    atlas: pd.DataFrame,
    baseline: pd.DataFrame,
    summary: Mapping[str, Any],
) -> tuple[bool, str]:
    """Independently compare controller and strength payoffs on both scored stages."""

    expected_ids = set(relative["opportunity_id"].astype(str))
    failures: list[str] = []
    metrics: dict[str, dict[str, float]] = {}
    if not expected_ids <= set(atlas["opportunity_id"].astype(str)):
        failures.append("atlas_population")
    if not expected_ids <= set(baseline["opportunity_id"].astype(str)):
        failures.append("baseline_population")
    if atlas["opportunity_id"].duplicated().any() or baseline["opportunity_id"].duplicated().any():
        failures.append("duplicate_prediction")
    beats = not failures
    for stage in ("validation", "final_opened_holdout"):
        stage_outcomes = relative.loc[relative["chronology_stage"].eq(stage), [
            "opportunity_id",
            "long_net_bps",
            "short_net_bps",
        ]]
        stage_metrics: dict[str, float] = {}
        for label, predictions in (("atlas", atlas), ("relative_strength", baseline)):
            selected = stage_outcomes.merge(
                predictions[["opportunity_id", "predicted_state"]],
                on="opportunity_id",
                how="left",
                validate="one_to_one",
            )
            state = selected["predicted_state"].astype(str)
            if selected["predicted_state"].isna().any():
                failures.append(f"{stage}_{label}_missing_prediction")
            payoff = np.where(
                state.eq("LONG"),
                selected["long_net_bps"],
                np.where(state.eq("SHORT"), selected["short_net_bps"], 0.0),
            ).astype(float)
            stage_metrics[label] = float(np.mean(payoff)) if len(payoff) else math.nan
        metrics[stage] = stage_metrics
        beats &= bool(
            math.isfinite(stage_metrics["atlas"])
            and math.isfinite(stage_metrics["relative_strength"])
            and stage_metrics["atlas"] > stage_metrics["relative_strength"]
        )
    recorded = bool(summary["atlas_beats_relative_strength_validation_and_final"])
    if recorded != beats:
        failures.append("summary_comparison_tamper")
    return not failures, (
        f"expected={beats}; recorded={recorded}; metrics={metrics}; failures={failures}"
    )


def verify_track_b(root: Path) -> tuple[bool, str]:
    track_root = root / "track_b"
    summary = cast(dict[str, Any], load_json(track_root / "track_b_summary.json"))
    contract = cast(dict[str, Any], load_json(root / "frozen_experiment_contract.json"))
    schema = cast(dict[str, Any], load_json(root / "feature_schema.json"))
    relative = pd.read_parquet(track_root / "relative_outcome_ledger.parquet")
    absolute = pd.read_parquet(root / "primary_economic_outcome_ledger.parquet")
    failures: list[str] = []
    if bool(summary["absolute_profitability_claim_allowed"]):
        failures.append("absolute_profitability_claim")
    relative_absolute = relative[["opportunity_id", "decision_timestamp"]].merge(
        absolute[["opportunity_id", "gross_long_return_bps"]],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    gross = pd.to_numeric(relative_absolute["gross_long_return_bps"], errors="coerce")
    groups = gross.groupby(relative_absolute["decision_timestamp"], sort=False)
    expected_peer_count = groups.transform("count")
    minimum_peers = int(contract["track_b"]["minimum_contemporaneous_peers"])
    expected_universe = groups.transform("mean").where(expected_peer_count.ge(minimum_peers))
    expected_residual = (gross - expected_universe).where(gross.notna())
    expected_rank = expected_residual.groupby(
        relative_absolute["decision_timestamp"], sort=False
    ).rank(method="average")
    expected_percentile = (expected_rank - 1.0) / (expected_peer_count - 1.0).replace(
        0.0, np.nan
    )
    relative_long_fraction = float(contract["track_b"]["relative_long_fraction"])
    relative_short_fraction = float(contract["track_b"]["relative_short_fraction"])
    expected_target = np.where(
        expected_percentile.ge(1.0 - relative_long_fraction),
        "LONG",
        np.where(expected_percentile.le(relative_short_fraction), "SHORT", "NEUTRAL"),
    )
    expected_target = np.where(
        expected_percentile.notna() & expected_peer_count.ge(minimum_peers),
        expected_target,
        "UNAVAILABLE",
    )
    reconstructed_columns = {
        "future_equal_universe_return_bps": expected_universe,
        "future_residual_return_bps": expected_residual,
        "future_residual_percentile": expected_percentile,
    }
    for column, expected_values in reconstructed_columns.items():
        if not np.allclose(
            pd.to_numeric(relative[column], errors="coerce"),
            expected_values,
            equal_nan=True,
            atol=1e-10,
        ):
            failures.append(f"relative_reconstruction:{column}")
    if not relative["peer_count"].eq(expected_peer_count).all():
        failures.append("relative_reconstruction:peer_count")
    if not np.array_equal(expected_target, relative["target"].astype(str).to_numpy()):
        failures.append("relative_target")
    discovery_library = cast(
        list[dict[str, Any]], load_json(track_root / "relative_discovery_library.json")
    )
    survivor_library = cast(
        list[dict[str, Any]], load_json(track_root / "relative_survivor_library.json")
    )
    if int(summary["frozen_exploratory_signatures"]) != len(discovery_library):
        failures.append("track_b_discovery_library_count")
    if int(summary["validation_survivors"]) != len(survivor_library):
        failures.append("track_b_survivor_count")
    if not {
        str(entry["signature"]["signature_id"]) for entry in survivor_library
    } <= {str(entry["signature"]["signature_id"]) for entry in discovery_library}:
        failures.append("track_b_validation_regenerated_rules")
    features = pd.read_parquet(root / "outcome_free_feature_ledger.parquet")
    session_dates = pd.to_datetime(features["session"])
    features["chronology_stage"] = "outside_frozen_chronology"
    chronology = cast(Mapping[str, Mapping[str, Any]], contract["chronology"])
    for stage_name in (
        "development_context",
        "discovery",
        "validation",
        "final_opened_holdout",
    ):
        specification = chronology[stage_name]
        in_stage = session_dates.ge(pd.Timestamp(str(specification["start"]))) & session_dates.lt(
            pd.Timestamp(str(specification["end_exclusive"]))
        )
        features.loc[in_stage, "chronology_stage"] = stage_name
    relative_scored = features.merge(
        relative[
            [
                "opportunity_id",
                "target",
                "long_net_bps",
                "short_net_bps",
                "round_trip_cost_bps",
            ]
        ],
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_relative"),
    )
    relative_scored["round_trip_cost_bps"] = relative_scored.pop(
        "round_trip_cost_bps_relative"
    )
    relative_scored = relative_scored.loc[
        relative_scored["target"].ne("UNAVAILABLE")
    ].copy()
    comparison_ok, comparison_detail = verify_relative_atlas_baseline_comparison(
        relative_scored,
        pd.read_parquet(track_root / "relative_atlas_decisions.parquet"),
        pd.read_parquet(track_root / "relative_strength_baseline_predictions.parquet"),
        summary,
    )
    if not comparison_ok:
        failures.append(f"track_b_atlas_baseline_comparison:{comparison_detail}")
    registry = pd.read_parquet(track_root / "relative_candidate_registry.parquet")
    registry_ok, registry_detail = verify_candidate_registry(
        track_root,
        registry,
        contract,
        schema,
        relative_scored,
    )
    if not registry_ok:
        failures.append(f"track_b_candidate_registry:{registry_detail}")
    validation_metrics = safe_csv(track_root / "relative_validation_metrics.csv")
    if not validation_metrics.empty:
        expected_holm = adjust_p_values(
            pd.to_numeric(validation_metrics["raw_p_value"], errors="raise").tolist(),
            "holm",
        )
        if not np.allclose(expected_holm, validation_metrics["holm_adjusted_p_value"]):
            failures.append("track_b_validation_holm")
        survived_ids = set(
            validation_metrics.loc[
                validation_metrics["validation_survived"].astype(bool), "signature_id"
            ].astype(str)
        )
        if not {
            str(entry["signature"]["signature_id"]) for entry in survivor_library
        } <= survived_ids:
            failures.append("track_b_unqualified_validation_survivor")
    for seal_name, library_name, hash_field in (
        (
            "discovery_stage_seal.json",
            "relative_discovery_library.json",
            "frozen_library_sha256",
        ),
        (
            "validation_stage_seal.json",
            "relative_survivor_library.json",
            "frozen_survivor_library_sha256",
        ),
    ):
        seal = cast(dict[str, Any], load_json(track_root / seal_name))
        if str(seal.get(hash_field)) != sha256_file(track_root / library_name):
            failures.append(f"track_b_{seal_name}")
    loso = safe_csv(track_root / "relative_leave_one_stock_out.csv")
    if len(discovery_library) and (
        loso.empty
        or not (
            loso["relative_outcomes_recomputed"].astype(bool).all()
            and loso["direct_cross_sectional_features_recomputed"].astype(bool).all()
            and loso["stress"].eq("leave_one_stock_out_recomputed").all()
        )
    ):
        failures.append("track_b_leave_one_stock_out_recomputation")
    track_b_nulls = safe_csv(track_root / "relative_null_test_results.csv")
    track_b_stress = safe_csv(track_root / "relative_stability_results.csv")
    if len(track_b_nulls) != 7 or (len(discovery_library) and track_b_stress.empty):
        failures.append("track_b_null_or_stress_outputs")
    final_metrics = safe_csv(track_root / "relative_final_opened_holdout_metrics.csv")
    expected_qualification = independent_relative_persistence_qualification(
        survivor_library,
        validation_metrics,
        final_metrics,
        track_b_stress,
        track_b_nulls,
        contract,
    ).sort_values("signature_id", kind="mergesort")
    recorded_qualification = safe_csv(
        track_root / "relative_persistence_qualification.csv"
    ).sort_values("signature_id", kind="mergesort")
    qualification_columns = [
        "signature_id",
        "direction",
        "strict_persistent_relative_signature",
        "rejection_reasons_json",
    ]
    if list(recorded_qualification.columns) != qualification_columns or len(
        recorded_qualification
    ) != len(expected_qualification):
        failures.append("track_b_persistence_qualification_shape")
    elif len(expected_qualification):
        recorded_flags = recorded_qualification[
            "strict_persistent_relative_signature"
        ].map(lambda value: str(value).lower() == "true")
        expected_flags = expected_qualification[
            "strict_persistent_relative_signature"
        ].astype(bool)
        for column in ("signature_id", "direction", "rejection_reasons_json"):
            if not recorded_qualification[column].astype(str).reset_index(drop=True).equals(
                expected_qualification[column].astype(str).reset_index(drop=True)
            ):
                failures.append(f"track_b_persistence_qualification:{column}")
        if not recorded_flags.reset_index(drop=True).equals(expected_flags.reset_index(drop=True)):
            failures.append("track_b_persistence_qualification:flag")
    persistent_ids = sorted(
        expected_qualification.loc[
            expected_qualification["strict_persistent_relative_signature"].astype(bool),
            "signature_id",
        ].astype(str)
    )
    if (
        int(summary["candidate_signatures_examined"]) != len(registry)
        or int(summary["persistent_relative_final_signatures"]) != len(persistent_ids)
        or sorted(cast(list[str], summary["persistent_relative_signature_ids"]))
        != persistent_ids
    ):
        failures.append("track_b_summary_reconstruction")
    return not failures, f"rows={len(relative)}; summary={summary}; failures={failures}"


def verify_scientific_decision(root: Path, *, include_track_b: bool) -> tuple[bool, str]:
    summary = cast(dict[str, Any], load_json(root / "track_a_summary.json"))
    track_b = cast(dict[str, Any], summary.get("track_b", {})) if include_track_b else {}
    long_count = int(summary.get("provisional_prospective_lead_long", 0))
    short_count = int(summary.get("provisional_prospective_lead_short", 0))
    if long_count and short_count:
        expected = "persistent_long_and_short_signatures_found_prospective_validation_required"
    elif long_count:
        expected = "persistent_long_signatures_only"
    elif short_count:
        expected = "persistent_short_signatures_only"
    elif int(summary.get("neutral_validation_and_final_strict_stable", 0)):
        expected = "neutral_veto_more_reliable_than_direction"
    elif include_track_b and int(track_b.get("persistent_relative_final_signatures", 0)) and bool(
        track_b.get("atlas_beats_relative_strength_validation_and_final")
    ):
        expected = "relative_direction_more_predictable_than_absolute"
    else:
        movement = cast(dict[str, Any], summary.get("movement_permitted_surface", {}))
        if int(movement.get("validation_survivors", 0)) and int(
            movement.get("positive_final_scored_signatures", 0)
        ):
            expected = "movement_permission_useful_direction_unresolved"
        elif int(summary.get("validation_survivor_long", 0)) + int(
            summary.get("validation_survivor_short", 0)
        ):
            expected = "signature_effects_concentrated_or_unstable"
        elif int(summary.get("frozen_discovery_long", 0)) + int(
            summary.get("frozen_discovery_short", 0)
        ):
            expected = "discovery_signatures_failed_validation"
        else:
            expected = "no_persistent_directional_signatures"
    field = "scientific_decision" if include_track_b else "track_a_scientific_decision"
    actual = str(summary.get(field))
    return actual == expected, f"field={field}; actual={actual}; expected={expected}"


def verify_prospective(root: Path, prospective_root: Path | None) -> tuple[bool, str]:
    forecast_schema = cast(
        dict[str, Any], load_json(root / "prospective_forecast_ledger_schema.json")
    )
    settlement_schema = cast(
        dict[str, Any], load_json(root / "prospective_settlement_ledger_schema.json")
    )
    failures: list[str] = []
    if not bool(forecast_schema["append_only"]) or not bool(forecast_schema["outcomes_forbidden"]):
        failures.append("forecast_schema")
    if not bool(settlement_schema["append_only"]) or bool(
        settlement_schema["forecast_overwrite_allowed"]
    ):
        failures.append("settlement_schema")
    forecast_required = set(map(str, forecast_schema["required_fields"]))
    settlement_required = set(map(str, settlement_schema["required_fields"]))
    if {"research_only", "execution_enabled"} - settlement_required:
        failures.append("settlement_safety_fields")
    if {"data_snapshot_hash", "training_data_snapshot_hash"} - forecast_required:
        failures.append("prospective_snapshot_identity_fields")
    metadata = cast(dict[str, Any], load_json(root / "run_metadata.json"))
    long_library = cast(list[dict[str, Any]], load_json(root / "long_signature_library.json"))
    short_library = cast(list[dict[str, Any]], load_json(root / "short_signature_library.json"))
    neutral_library = cast(list[dict[str, Any]], load_json(root / "neutral_veto_library.json"))
    library_hashes = {
        "long_library_hash": canonical_hash(long_library),
        "short_library_hash": canonical_hash(short_library),
        "neutral_library_hash": canonical_hash(neutral_library),
    }
    checked_records = 0
    if prospective_root is not None:
        ledgers: dict[str, list[dict[str, Any]]] = {}
        for name in ("forecast_ledger.jsonl", "settlement_ledger.jsonl"):
            path = prospective_root / name
            records = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
            previous: str | None = None
            for record in records:
                recorded_hash = str(record["record_hash"])
                payload = {key: value for key, value in record.items() if key != "record_hash"}
                if (
                    payload.get("previous_hash") != previous
                    or canonical_hash(payload) != recorded_hash
                ):
                    failures.append(f"hash_chain:{name}")
                    break
                previous = recorded_hash
                checked_records += 1
            ledgers[name] = records
        forecast_rows = ledgers.get("forecast_ledger.jsonl", [])
        settlement_rows = ledgers.get("settlement_ledger.jsonl", [])
        forecast_ids = {str(row["opportunity_id"]) for row in forecast_rows}
        settlement_ids = {str(row["opportunity_id"]) for row in settlement_rows}
        if len(forecast_ids) != len(forecast_rows):
            failures.append("duplicate_forecast")
        if len(settlement_ids) != len(settlement_rows):
            failures.append("duplicate_settlement")
        if not settlement_ids <= forecast_ids:
            failures.append("orphan_settlement")
        forecasts_by_id = {str(row["opportunity_id"]): row for row in forecast_rows}
        forbidden_feature_tokens = (
            "future_return",
            "future_direction",
            "future_state",
            "route_identity",
            "child_identity",
            "mfe",
            "mae",
            "hindsight_episode",
            "terminal_payoff",
            "actual_target",
        )
        for row in forecast_rows:
            opportunity_id = str(row.get("opportunity_id"))
            payload_fields = set(row) - {"previous_hash", "record_hash"}
            if payload_fields != forecast_required:
                failures.append(f"forecast_fields:{opportunity_id}")
                continue
            if row["research_only"] is not True or row["execution_enabled"] is not False:
                failures.append(f"forecast_safety:{opportunity_id}")
            expected_identity = {
                "run_id": metadata["run_id"],
                "git_sha": metadata["git_sha"],
                "contract_hash": metadata["contract_sha256"],
                "feature_schema_hash": metadata["feature_schema_sha256"],
            }
            for field, expected in expected_identity.items():
                if str(row.get(field)) != str(expected):
                    failures.append(f"forecast_identity:{opportunity_id}:{field}")
            if str(row.get("training_data_snapshot_hash")) != str(
                metadata["data_snapshot_sha256"]
            ):
                failures.append(f"forecast_training_snapshot:{opportunity_id}")
            if not str(row.get("data_snapshot_hash", "")).strip() or row.get(
                "data_snapshot_hash"
            ) == row.get("training_data_snapshot_hash"):
                failures.append(f"forecast_input_snapshot:{opportunity_id}")
            for field, expected_hash in library_hashes.items():
                if str(row.get(field)) != expected_hash:
                    failures.append(f"forecast_library_hash:{opportunity_id}:{field}")
            try:
                decision = pd.Timestamp(str(row["decision_timestamp"]))
                freeze = pd.Timestamp(str(row["forecast_freeze_timestamp"]))
                entry = pd.Timestamp(str(row["entry_timestamp"]))
                terminal = pd.Timestamp(str(row["terminal_timestamp"]))
                if any(value.tzinfo is None for value in (decision, freeze, entry, terminal)):
                    raise ValueError("naive")
                if not decision <= freeze < entry < terminal:
                    failures.append(f"forecast_timing:{opportunity_id}")
                if entry != decision + pd.Timedelta(minutes=5):
                    failures.append(f"forecast_next_open:{opportunity_id}")
                if terminal != decision + pd.Timedelta(minutes=120):
                    failures.append(f"forecast_fixed_terminal:{opportunity_id}")
                local = decision.tz_convert("America/New_York")
                expected_clock = {"clock_12": "10:30", "clock_36": "12:30"}
                if local.strftime("%H:%M") != expected_clock.get(str(row["decision_clock"])):
                    failures.append(f"forecast_clock:{opportunity_id}")
            except (TypeError, ValueError):
                failures.append(f"forecast_timestamp:{opportunity_id}")
                continue
            features = row["causal_features"]
            availability = row["feature_availability_timestamps"]
            if not isinstance(features, dict) or not isinstance(availability, dict):
                failures.append(f"forecast_feature_mapping:{opportunity_id}")
                continue
            if set(features) != set(availability):
                failures.append(f"forecast_availability_keys:{opportunity_id}")
            if any(
                any(token in str(feature).lower() for token in forbidden_feature_tokens)
                for feature in features
            ):
                failures.append(f"forecast_outcome_feature:{opportunity_id}")
            for feature, value in features.items():
                available_at = availability.get(feature)
                if value is not None and available_at is None:
                    failures.append(f"forecast_missing_availability:{opportunity_id}:{feature}")
                    continue
                if available_at is None:
                    continue
                timestamp = pd.Timestamp(str(available_at))
                if timestamp.tzinfo is None or timestamp > decision or timestamp > freeze:
                    failures.append(f"forecast_feature_timing:{opportunity_id}:{feature}")
            long_votes = sum(bool(value) for value in row["long_signature_decisions"].values())
            short_votes = sum(bool(value) for value in row["short_signature_decisions"].values())
            if long_votes != int(row["long_vote_count"]) or short_votes != int(
                row["short_vote_count"]
            ):
                failures.append(f"forecast_votes:{opportunity_id}")
            if bool(row["conflict_state"]) != bool(long_votes and short_votes):
                failures.append(f"forecast_conflict:{opportunity_id}")
            for direction, library in (("long", long_library), ("short", short_library)):
                expected_decisions: dict[str, bool] = {}
                for entry_payload in library:
                    signature = cast(dict[str, Any], entry_payload["signature"])
                    signature_id = str(signature["signature_id"])
                    expected_decisions[signature_id] = bool(
                        condition_mask(
                            pd.DataFrame([features]),
                            cast(Sequence[Mapping[str, Any]], signature["conditions"]),
                        ).iloc[0]
                    )
                if row[f"{direction}_signature_decisions"] != expected_decisions:
                    failures.append(f"forecast_signature_decisions:{opportunity_id}:{direction}")
            long_value = sum(
                float(entry.get("conservative_value_bps", 0.0))
                for entry in long_library
                if bool(row["long_signature_decisions"].get(entry["signature"]["signature_id"]))
            )
            short_value = sum(
                float(entry.get("conservative_value_bps", 0.0))
                for entry in short_library
                if bool(row["short_signature_decisions"].get(entry["signature"]["signature_id"]))
            )
            movement = str(row["movement_permission"])
            missing_required = any(
                features.get(str(condition["feature"])) is None
                for entry in [*long_library, *short_library]
                for condition in entry["signature"]["conditions"]
            )
            expected_state = "NEUTRAL"
            if movement == "PASS" and not missing_required and not bool(long_votes and short_votes):
                if long_votes and not short_votes and long_value / long_votes > 0.0:
                    expected_state = "LONG"
                elif short_votes and not long_votes and short_value / short_votes > 0.0:
                    expected_state = "SHORT"
            if str(row["final_atlas_state"]) != expected_state:
                failures.append(f"forecast_final_state:{opportunity_id}")

        for row in settlement_rows:
            opportunity_id = str(row.get("opportunity_id"))
            payload_fields = set(row) - {"previous_hash", "record_hash"}
            if payload_fields != settlement_required:
                failures.append(f"settlement_fields:{opportunity_id}")
                continue
            if row["research_only"] is not True or row["execution_enabled"] is not False:
                failures.append(f"settlement_safety:{opportunity_id}")
            forecast = forecasts_by_id.get(opportunity_id)
            if forecast is None:
                continue
            terminal = pd.Timestamp(str(row["terminal_timestamp"]))
            settled_at = pd.Timestamp(str(row["settlement_timestamp"]))
            frozen_terminal = pd.Timestamp(str(forecast["terminal_timestamp"]))
            if terminal != frozen_terminal or settled_at < terminal:
                failures.append(f"settlement_timing:{opportunity_id}")
            if str(row.get("settlement_status")) == "UNAVAILABLE":
                economics = (
                    "gross_long_payoff_bps",
                    "gross_short_payoff_bps",
                    "costs_bps",
                    "net_long_payoff_bps",
                    "net_short_payoff_bps",
                )
                if (
                    str(row.get("primary_target")) != "UNAVAILABLE"
                    or any(row.get(field) is not None for field in economics)
                    or not str(row.get("unavailable_reason", "")).strip()
                ):
                    failures.append(f"settlement_unavailable:{opportunity_id}")
                continue
            gross_long = float(row["gross_long_payoff_bps"])
            gross_short = float(row["gross_short_payoff_bps"])
            costs = float(row["costs_bps"])
            net_long = float(row["net_long_payoff_bps"])
            net_short = float(row["net_short_payoff_bps"])
            if not (
                math.isclose(gross_short, -gross_long, abs_tol=1e-9)
                and math.isclose(net_long, gross_long - costs, abs_tol=1e-9)
                and math.isclose(net_short, gross_short - costs, abs_tol=1e-9)
                and costs >= 0.0
            ):
                failures.append(f"settlement_economics:{opportunity_id}")
            expected_target = (
                "LONG"
                if gross_long > 2.0 * costs and net_long > 0.0
                else ("SHORT" if gross_long < -2.0 * costs and net_short > 0.0 else "NEUTRAL")
            )
            if str(row["primary_target"]) != expected_target:
                failures.append(f"settlement_target:{opportunity_id}")
    return not failures, f"checked_records={checked_records}; failures={failures}"


def verify_safety(metadata: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[bool, str]:
    safety = cast(Mapping[str, Any], contract["safety"])
    failures: list[str] = []
    if bool(metadata["execution_enabled"]) or not bool(metadata["research_only"]):
        failures.append("metadata_flags")
    if any(bool(value) for value in safety.values()):
        failures.append("contract_safety_flag")
    base = str(contract["implementation_base_git_sha"])
    execution_sha = str(metadata["git_sha"])
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}..{execution_sha}"], cwd=REPO, text=True
    ).splitlines()
    work_prefix = "research/slrno-v2/20260714-regime-loop-handoff/work/"
    allowed_exact = {
        f"{work_prefix}run_directional_signature_atlas_v1.py",
        f"{work_prefix}audit_directional_signature_atlas_v1.py",
        f"{work_prefix}contracts/20260717-directional-signature-atlas-v1.json",
        f"{work_prefix}contracts/20260717-directional-signature-atlas-v1-feature-schema.json",
        f"{work_prefix}reports/20260717-directional-signature-atlas-v1.md",
    }
    allowed_prefixes = (
        "packages/stocker_research/src/stocker_research/directional_signature_atlas/",
        "tests/test_directional_signature_atlas_",
        f"{work_prefix}artifacts/20260717-directional-signature-atlas-v1/",
    )
    prohibited = sorted(
        path
        for path in changed
        if path not in allowed_exact and not path.startswith(allowed_prefixes)
    )
    forbidden_path_tokens = (
        "broker",
        "ig_integration",
        "order",
        "position",
        "deployment",
        "application_runtime",
        "strategy_exit",
    )
    prohibited.extend(
        path
        for path in changed
        if any(token in path.lower() for token in forbidden_path_tokens)
    )
    prohibited = sorted(set(prohibited))
    if prohibited:
        failures.append(f"out_of_scope_paths:{prohibited[:5]}")
    return not failures, f"changed_paths={len(changed)}; failures={failures}"


def run_audit(
    primary: Path,
    exact: Path,
    *,
    prospective_root: Path | None,
    scope: str,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    contract = cast(dict[str, Any], load_json(CONTRACT_PATH))
    schema = cast(dict[str, Any], load_json(FEATURE_SCHEMA_PATH))
    metadata = cast(dict[str, Any], load_json(primary / "run_metadata.json"))
    features = pd.read_parquet(primary / "outcome_free_feature_ledger.parquet")
    outcomes = pd.read_parquet(primary / "primary_economic_outcome_ledger.parquet")
    delayed = pd.read_parquet(primary / "one_bar_delay_outcome_ledger.parquet")
    first_touch = pd.read_parquet(primary / "secondary_first_touch_outcome_ledger.parquet")
    movement = pd.read_parquet(primary / "movement_permission_ledger.parquet")
    registry = pd.read_parquet(primary / "complete_candidate_registry.parquet")
    decisions = pd.read_parquet(primary / "atlas_level_decisions.parquet")
    baselines = pd.read_parquet(primary / "baseline_predictions.parquet")
    predictive_metrics = safe_csv(primary / "predictive_calibration_metrics.csv")
    economic_metrics = safe_csv(primary / "economic_metrics.csv")
    discovery_library = cast(
        list[dict[str, Any]], load_json(primary / "frozen_discovery_signature_library.json")
    )
    validation_metrics = safe_csv(primary / "validation_signature_metrics.csv")
    session_dates = pd.to_datetime(features["session"])
    features["chronology_stage"] = "outside_frozen_chronology"
    chronology = cast(Mapping[str, Mapping[str, Any]], contract["chronology"])
    for stage_name in (
        "development_context",
        "discovery",
        "validation",
        "final_opened_holdout",
    ):
        specification = chronology[stage_name]
        mask = session_dates.ge(pd.Timestamp(str(specification["start"]))) & session_dates.lt(
            pd.Timestamp(str(specification["end_exclusive"]))
        )
        features.loc[mask, "chronology_stage"] = stage_name
    scored = features.merge(
        outcomes[
            [
                "opportunity_id",
                "score_status",
                "target",
                "net_long_return_bps",
                "net_short_return_bps",
            ]
        ],
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "net_long_return_bps": "long_net_bps",
            "net_short_return_bps": "short_net_bps",
        }
    )
    scored = scored.loc[scored["score_status"].eq("scored")].reset_index(drop=True)

    contract_ok = (
        sha256_file(CONTRACT_PATH) == str(metadata["contract_sha256"])
        and (primary / "frozen_experiment_contract.json").read_bytes() == CONTRACT_PATH.read_bytes()
    )
    add_check(checks, "contract_identity", contract_ok, str(metadata["contract_sha256"]))
    schema_ok = (
        sha256_file(FEATURE_SCHEMA_PATH) == str(metadata["feature_schema_sha256"])
        and (primary / "feature_schema.json").read_bytes() == FEATURE_SCHEMA_PATH.read_bytes()
    )
    add_check(checks, "feature_schema_identity", schema_ok, str(metadata["feature_schema_sha256"]))
    manifest_ok, manifest_detail = verify_manifest(primary)
    add_check(checks, "artifact_manifest", manifest_ok, manifest_detail)
    source_ok, source_detail, identities = verify_source_identity(primary, metadata)
    add_check(checks, "data_snapshot_identity", source_ok, source_detail)
    pins_ok, pins_detail = verify_frozen_source_pins(contract, identities)
    add_check(checks, "frozen_source_pins", pins_ok, pins_detail)
    prior_ok, prior_detail = verify_prior_experiment_evidence(primary)
    add_check(checks, "prior_experiment_evidence", prior_ok, prior_detail)
    git_ok, git_detail = verify_git_identity(metadata, identities)
    add_check(checks, "git_identity", git_ok, git_detail)
    feature_hash_ok = sha256_file(primary / "outcome_free_feature_ledger.parquet") == str(
        metadata["feature_ledger_sha256"]
    )
    add_check(
        checks,
        "sealed_feature_ledger_identity",
        feature_hash_ok,
        str(metadata["feature_ledger_sha256"]),
    )
    pre_outcome_manifest = cast(
        dict[str, Any], load_json(primary / "pre_outcome_feature_manifest.json")
    )
    add_check(
        checks,
        "pre_outcome_feature_seal",
        feature_hash_ok
        and str(pre_outcome_manifest.get("feature_ledger_sha256"))
        == str(metadata["feature_ledger_sha256"])
        and pre_outcome_manifest.get("outcomes_joined_at_seal") is False,
        str(pre_outcome_manifest),
    )

    for check_id, result in (
        ("opportunity_population_and_clocks", verify_population(features, outcomes, contract)),
        ("feature_availability_and_leakage", verify_feature_availability(features, schema)),
        ("state_motifs_and_loop_summaries", verify_motifs_and_loop_summaries(features)),
        (
            "sampled_structural_source_reconstruction",
            verify_sampled_structural_source_reconstruction(features, contract),
        ),
        ("cost_aware_labels_and_dead_band", verify_outcomes(outcomes, contract)),
        (
            "entry_terminal_first_touch_and_provider_reconstruction",
            verify_timestamps_and_provider_samples(
                features, outcomes, delayed, first_touch, contract
            ),
        ),
        ("movement_permission", verify_movement(features, movement)),
        (
            "candidate_caps_complexity_and_fdr",
            verify_candidate_registry(primary, registry, contract, schema, scored),
        ),
        (
            "sampled_signature_metrics",
            verify_signature_metrics(scored, discovery_library, validation_metrics),
        ),
    ):
        add_check(checks, check_id, result[0], result[1])

    libraries_ok, libraries_detail, survivors = verify_libraries_and_chronology(
        primary, discovery_library, validation_metrics, contract
    )
    add_check(checks, "chronology_and_frozen_libraries", libraries_ok, libraries_detail)
    controller_ok, controller_detail = verify_atlas_controller(features, decisions, survivors)
    add_check(checks, "atlas_votes_neutral_and_conflict", controller_ok, controller_detail)
    aggregate_ok, aggregate_detail = verify_atlas_aggregate_metrics(
        scored, decisions, predictive_metrics, economic_metrics
    )
    add_check(checks, "atlas_predictive_and_economic_metrics", aggregate_ok, aggregate_detail)
    baseline_ok, baseline_detail = verify_baselines(features, baselines)
    add_check(checks, "baseline_population_and_logic", baseline_ok, baseline_detail)
    robust_ok, robust_detail = verify_null_stress_and_concentration(
        primary, contract, scored, discovery_library
    )
    add_check(checks, "null_stress_and_concentration", robust_ok, robust_detail)
    if scope == "all":
        track_b_ok, track_b_detail = verify_track_b(primary)
        add_check(checks, "cross_sectional_track_b", track_b_ok, track_b_detail)
    decision_ok, decision_detail = verify_scientific_decision(
        primary, include_track_b=scope == "all"
    )
    add_check(checks, "scientific_decision_derivation", decision_ok, decision_detail)
    prospective_ok, prospective_detail = verify_prospective(
        primary, prospective_root if scope == "all" else None
    )
    add_check(checks, "prospective_append_only_integrity", prospective_ok, prospective_detail)
    safety_ok, safety_detail = verify_safety(metadata, contract)
    add_check(checks, "research_only_safety", safety_ok, safety_detail)
    identity = exact_identity(primary, exact)
    add_check(
        checks,
        "primary_exact_rerun_byte_identity",
        bool(identity["byte_identical"]),
        f"compared_files={identity['compared_files']}; mismatches={identity['hash_mismatches']}",
    )
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "audit_id": "20260717-directional-signature-atlas-v1-independent-audit",
        "scope": scope,
        "run_id": metadata["run_id"],
        "contract_id": contract["contract_id"],
        "git_sha": metadata["git_sha"],
        "contract_sha256": metadata["contract_sha256"],
        "data_snapshot_sha256": metadata["data_snapshot_sha256"],
        "feature_schema_sha256": metadata["feature_schema_sha256"],
        "feature_ledger_sha256": metadata["feature_ledger_sha256"],
        "check_count": len(checks),
        "passed_check_count": sum(bool(check["passed"]) for check in checks),
        "passed": passed,
        "checks": checks,
        "exact_rerun_identity": identity,
        "research_only": True,
        "execution_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--prospective-root", type=Path)
    parser.add_argument("--scope", choices=("track-a", "all"), default="all")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(
        args.primary,
        args.exact,
        prospective_root=args.prospective_root,
        scope=str(args.scope).replace("-", "_"),
    )
    output = args.output or args.primary / "independent_audit.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bool(result["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
