#!/usr/bin/env python3
"""Independent audit for the research-only Directional Signature Atlas V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT_PATH = WORK / "contracts/20260717-directional-signature-atlas-v1.json"
FEATURE_SCHEMA_PATH = WORK / "contracts/20260717-directional-signature-atlas-v1-feature-schema.json"
DEFAULT_PRIMARY = WORK / "artifacts/20260717-directional-signature-atlas-v1/primary"
DEFAULT_EXACT = WORK / "artifacts/20260717-directional-signature-atlas-v1/exact_rerun"
IDENTITY_EXCLUSIONS = {
    "independent_audit.json",
    "prospective_forecast_dry_run.json",
    "prospective_settlement_dry_run.json",
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
        )
    )
    return passed, (
        f"rows={len(features)}; ordinals={sorted(ordinals)}; "
        f"clocks={sorted(clocks)}; local_times={sorted(clock_times)}"
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
    for length in (2, 3, 4):
        column = f"state_motif_{length}"
        populated = features.loc[features[column].notna(), [column]].copy()
        tokens = populated[column].astype(str).str.split(">")
        if not tokens.map(len).eq(length).all():
            failures.append(f"motif_length_{length}")
        indices = populated.index
        final_tokens = pd.to_numeric(tokens.map(lambda values: values[-1]), errors="coerce")
        if not np.allclose(final_tokens.to_numpy(float), current.loc[indices].to_numpy(float)):
            failures.append(f"motif_current_state_{length}")
    loop_columns = [
        "top_parent_loop",
        "top_loop_orientation",
        "top_loop_score",
        "top_second_margin",
        "compatibility_mass",
        "compatibility_entropy",
        "compatible_loop_count",
    ]
    allowed = features["state_run_entry_at_decision"].astype(bool) & features[
        "frozen_state_inputs_complete"
    ].astype(bool)
    if bool(features.loc[~allowed, loop_columns].notna().any(axis=1).any()):
        failures.append("loop_summary_outside_exact_causal_run_entry")
    june_29 = features["session"].astype(str).eq("2026-06-29")
    if bool(features.loc[june_29, ["current_state", *loop_columns]].notna().any(axis=1).any()):
        failures.append("vti_boundary_not_failed_closed")
    return (
        not failures,
        f"motif_rows={int(features['state_motif_2'].notna().sum())}; failures={failures}",
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
        if not all(
            (
                close(entry_open, row["entry_open"]),
                close(terminal_close, row["terminal_close"]),
                close(gross, row["gross_long_return_bps"]),
                close(future_range, row["future_high_low_range_bps"]),
            )
        ):
            failures.append(f"provider_reconstruction:{row['opportunity_id']}")
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
    expected = predicted.gt(30.0) & predicted.notna()
    passed = all(
        (
            np.allclose(
                joined["predicted_future_range_bps_feature"],
                joined["predicted_future_range_bps_ledger"],
                equal_nan=True,
            ),
            expected.eq(joined["movement_permission_feature"].astype(bool)).all(),
            expected.eq(joined["movement_permission_ledger"].astype(bool)).all(),
        )
    )
    return bool(
        passed
    ), f"permitted={int(expected.sum())}; unavailable={int(predicted.isna().sum())}"


def verify_candidate_registry(
    registry: pd.DataFrame,
    contract: Mapping[str, Any],
    schema: Mapping[str, Any],
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
        discovery = independent_signature_metrics(scored.loc[scored["period"].eq(2024)], signature)
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
        validation = independent_signature_metrics(scored.loc[scored["period"].eq(2025)], signature)
        if validation_by_id.empty or signature_id not in validation_by_id.index:
            failures.append(f"validation_missing:{signature_id}")
            continue
        recorded_validation = validation_by_id.loc[signature_id]
        for key in ("rows", "sessions", "stocks", "mean_directional_net_bps", "directional_lift"):
            if not close(validation[key], recorded_validation[key]):
                failures.append(f"validation:{signature_id}:{key}")
                break
    return not failures, f"reconstructed={len(discovery_library)}; failures={failures[:5]}"


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
    if {str(entry["signature"]["signature_id"]) for entry in long_library} & {
        str(entry["signature"]["signature_id"]) for entry in short_library
    }:
        failures.append("long_short_library_overlap")
    if not validation_metrics.empty:
        expected_holm = adjust_p_values(
            pd.to_numeric(validation_metrics["raw_p_value"], errors="raise").tolist(), "holm"
        )
        if not np.allclose(expected_holm, validation_metrics["holm_adjusted_p_value"]):
            failures.append("validation_holm_reconstruction")
    chronology = cast(Mapping[str, Any], contract["chronology"])
    periods = {
        int(cast(Mapping[str, Any], chronology["discovery"])["period"]),
        int(cast(Mapping[str, Any], chronology["validation"])["period"]),
        int(cast(Mapping[str, Any], chronology["final_opened_holdout"])["period"]),
    }
    if periods != {2024, 2025, 2026}:
        failures.append("chronology_contract")
    return (
        not failures,
        f"discovery={len(discovery_library)}; survivors={len(survivors)}",
        survivors,
    )


def verify_atlas_controller(
    scored: pd.DataFrame,
    decisions: pd.DataFrame,
    survivors: list[dict[str, Any]],
) -> tuple[bool, str]:
    long_votes = np.zeros(len(scored), dtype=int)
    short_votes = np.zeros(len(scored), dtype=int)
    long_values = np.zeros(len(scored), dtype=float)
    short_values = np.zeros(len(scored), dtype=float)
    for entry in survivors:
        signature = cast(dict[str, Any], entry["signature"])
        mask = condition_mask(
            scored, cast(Sequence[Mapping[str, Any]], signature["conditions"])
        ).to_numpy(bool)
        value = float(entry["conservative_value_bps"])
        if str(signature["direction"]) == "LONG":
            long_votes += mask.astype(int)
            long_values[mask] += value
        else:
            short_votes += mask.astype(int)
            short_values[mask] += value
    movement = scored["movement_permission"].astype(bool).to_numpy()
    conflict = (long_votes > 0) & (short_votes > 0)
    states = np.full(len(scored), "NEUTRAL", dtype=object)
    states[
        movement
        & ~conflict
        & (long_votes > 0)
        & (short_votes == 0)
        & (long_values / np.maximum(long_votes, 1) > 0.0)
    ] = "LONG"
    states[
        movement
        & ~conflict
        & (short_votes > 0)
        & (long_votes == 0)
        & (short_values / np.maximum(short_votes, 1) > 0.0)
    ] = "SHORT"
    ordered = decisions.set_index("opportunity_id").loc[scored["opportunity_id"]]
    passed = all(
        (
            np.array_equal(long_votes, ordered["long_vote_count"].to_numpy(int)),
            np.array_equal(short_votes, ordered["short_vote_count"].to_numpy(int)),
            np.array_equal(conflict, ordered["conflict"].to_numpy(bool)),
            np.array_equal(states, ordered["predicted_state"].astype(str).to_numpy()),
        )
    )
    return bool(passed), (
        f"survivors={len(survivors)}; directional_outputs="
        f"{int(np.isin(states, ['LONG', 'SHORT']).sum())}; conflicts={int(conflict.sum())}"
    )


def verify_baselines(scored: pd.DataFrame, baselines: pd.DataFrame) -> tuple[bool, str]:
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
    expected_ids = set(scored["opportunity_id"].astype(str))
    for model_id, group in baselines.groupby("model_id", sort=False):
        if set(group["opportunity_id"].astype(str)) != expected_ids:
            failures.append(f"population:{model_id}")
    indexed = scored.set_index("opportunity_id")
    for model_id, reverse in (("one_bar_momentum", False), ("one_bar_reversal", True)):
        group = baselines.loc[baselines["model_id"].eq(model_id)].set_index("opportunity_id")
        returns = pd.to_numeric(indexed.loc[group.index, "return_1_scale"], errors="coerce")
        positive = "SHORT" if reverse else "LONG"
        negative = "LONG" if reverse else "SHORT"
        expected = np.where(returns > 0.0, positive, np.where(returns < 0.0, negative, "NEUTRAL"))
        if not np.array_equal(expected, group["predicted_state"].astype(str).to_numpy()):
            failures.append(model_id)
    return (
        not failures,
        f"models={len(models)}; rows_per_model={len(baselines) // max(len(models), 1)}",
    )


def verify_null_stress_and_concentration(
    root: Path, contract: Mapping[str, Any]
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
    missing_stresses = sorted(required_stresses - observed_stresses)
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
    if required_dimensions - observed_dimensions:
        failures.append("concentration_dimensions")
    return not failures, (
        f"nulls={len(nulls)}; stress_rows={len(stresses)}; "
        f"dimensions={sorted(observed_dimensions)}; failures={failures[:3]}"
    )


def verify_track_b(root: Path) -> tuple[bool, str]:
    track_root = root / "track_b"
    summary = cast(dict[str, Any], load_json(track_root / "track_b_summary.json"))
    relative = pd.read_parquet(track_root / "relative_outcome_ledger.parquet")
    failures: list[str] = []
    if bool(summary["absolute_profitability_claim_allowed"]):
        failures.append("absolute_profitability_claim")
    grouped = relative.groupby("decision_timestamp", sort=False)
    residual_means = grouped["future_residual_return_bps"].mean().dropna()
    if not np.allclose(residual_means, 0.0, atol=1e-8):
        failures.append("relative_residual_mean")
    if not relative["peer_count"].eq(grouped["opportunity_id"].transform("size")).all():
        failures.append("peer_count")
    percentile = pd.to_numeric(relative["future_residual_percentile"], errors="coerce")
    expected = np.where(
        percentile.ge(0.80),
        "LONG",
        np.where(percentile.le(0.20), "SHORT", "NEUTRAL"),
    )
    expected = np.where(percentile.notna() & relative["peer_count"].ge(10), expected, "UNAVAILABLE")
    if not np.array_equal(expected, relative["target"].astype(str).to_numpy()):
        failures.append("relative_target")
    if int(summary["validation_survivors"]) != 0:
        failures.append("unexpected_track_b_survivor")
    return not failures, f"rows={len(relative)}; summary={summary}; failures={failures}"


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
        forecast_ids = {
            str(row["opportunity_id"]) for row in ledgers.get("forecast_ledger.jsonl", [])
        }
        settlement_ids = {
            str(row["opportunity_id"]) for row in ledgers.get("settlement_ledger.jsonl", [])
        }
        if len(forecast_ids) != len(ledgers.get("forecast_ledger.jsonl", [])):
            failures.append("duplicate_forecast")
        if len(settlement_ids) != len(ledgers.get("settlement_ledger.jsonl", [])):
            failures.append("duplicate_settlement")
        if not settlement_ids <= forecast_ids:
            failures.append("orphan_settlement")
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
    allowed_prefixes = (
        "research/slrno-v2/20260714-regime-loop-handoff/work/",
        "packages/stocker_research/src/stocker_research/directional_signature_atlas/",
        "tests/test_directional_signature_atlas_",
    )
    prohibited = sorted(path for path in changed if not path.startswith(allowed_prefixes))
    if prohibited:
        failures.append(f"out_of_scope_paths:{prohibited[:5]}")
    return not failures, f"changed_paths={len(changed)}; failures={failures}"


def run_audit(
    primary: Path,
    exact: Path,
    *,
    prospective_root: Path | None,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    contract = cast(dict[str, Any], load_json(CONTRACT_PATH))
    schema = cast(dict[str, Any], load_json(FEATURE_SCHEMA_PATH))
    metadata = cast(dict[str, Any], load_json(primary / "run_metadata.json"))
    features = pd.read_parquet(primary / "outcome_free_feature_ledger.parquet")
    outcomes = pd.read_parquet(primary / "primary_economic_outcome_ledger.parquet")
    delayed = pd.read_parquet(primary / "one_bar_delay_outcome_ledger.parquet")
    movement = pd.read_parquet(primary / "movement_permission_ledger.parquet")
    registry = pd.read_parquet(primary / "complete_candidate_registry.parquet")
    decisions = pd.read_parquet(primary / "atlas_level_decisions.parquet")
    baselines = pd.read_parquet(primary / "baseline_predictions.parquet")
    discovery_library = cast(
        list[dict[str, Any]], load_json(primary / "frozen_discovery_signature_library.json")
    )
    validation_metrics = safe_csv(primary / "validation_signature_metrics.csv")
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

    for check_id, result in (
        ("opportunity_population_and_clocks", verify_population(features, outcomes, contract)),
        ("feature_availability_and_leakage", verify_feature_availability(features, schema)),
        ("state_motifs_and_loop_summaries", verify_motifs_and_loop_summaries(features)),
        ("cost_aware_labels_and_dead_band", verify_outcomes(outcomes, contract)),
        (
            "entry_terminal_and_provider_reconstruction",
            verify_timestamps_and_provider_samples(features, outcomes, delayed, contract),
        ),
        ("movement_permission", verify_movement(features, movement)),
        (
            "candidate_caps_complexity_and_fdr",
            verify_candidate_registry(registry, contract, schema),
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
    controller_ok, controller_detail = verify_atlas_controller(scored, decisions, survivors)
    add_check(checks, "atlas_votes_neutral_and_conflict", controller_ok, controller_detail)
    baseline_ok, baseline_detail = verify_baselines(scored, baselines)
    add_check(checks, "baseline_population_and_logic", baseline_ok, baseline_detail)
    robust_ok, robust_detail = verify_null_stress_and_concentration(primary, contract)
    add_check(checks, "null_stress_and_concentration", robust_ok, robust_detail)
    track_b_ok, track_b_detail = verify_track_b(primary)
    add_check(checks, "cross_sectional_track_b", track_b_ok, track_b_detail)
    prospective_ok, prospective_detail = verify_prospective(primary, prospective_root)
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(
        args.primary,
        args.exact,
        prospective_root=args.prospective_root,
    )
    output = args.output or args.primary / "independent_audit.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bool(result["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
