#!/usr/bin/env python3
"""Rebuild live-runtime parity reports from frozen retrospective evidence only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
ARCHETYPE_ROOT = (
    ROOT / "research" / "directional-readiness" / "20260726-stock-local-directional-archetypes-v0"
)
PRIMARY = ARCHETYPE_ROOT / "artifacts" / "primary"
for package in ("stocker_prospective", "stocker_research", "stocker_data", "stocker_core"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

from stocker_prospective.contract import claims_boundary  # noqa: E402
from stocker_prospective.direction import (  # noqa: E402
    ARCHETYPE_IDS,
    FrozenDirectionRuntime,
)
from stocker_prospective.direction_features import (  # noqa: E402
    DirectionFeatureBar,
    FrozenDirectionFeatureBuilder,
)
from stocker_prospective.frozen_m1c import (  # noqa: E402
    M1C_THRESHOLD,
    FrozenM1CRuntime,
)
from stocker_prospective.m1c_features import (  # noqa: E402
    LiveFeatureBar,
    M1CCausalFeatureBuilder,
)

SCALING = (
    ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
    / "model_configurations.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_research_runner() -> ModuleType:
    path = ARCHETYPE_ROOT / "run_screen_v0.py"
    specification = importlib.util.spec_from_file_location("frozen_archetype_runner", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the frozen archetype runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def select_rows(frame: pd.DataFrame, *, count: int) -> pd.DataFrame:
    """Mix near-threshold and evenly-spaced rows deterministically."""

    ordered = frame.assign(
        _distance=(frame["M1C_probability"].astype(float) - M1C_THRESHOLD).abs()
    ).sort_values(
        ["_distance", "stock", "session", "checkpoint"],
        kind="mergesort",
    )
    close_count = count // 2
    close = ordered.head(close_count)
    remainder = ordered.drop(index=close.index).sort_values(
        ["stock", "session", "checkpoint"],
        kind="mergesort",
    )
    positions = np.linspace(0, max(len(remainder) - 1, 0), count - close_count, dtype=int)
    spaced = remainder.iloc[positions] if len(remainder) else remainder
    return (
        pd.concat([close, spaced], ignore_index=True)
        .drop_duplicates(["stock", "session", "checkpoint"], keep="first")
        .head(count)
    )


def build_m1c_report() -> dict[str, Any]:
    runner = load_research_runner()
    historical, stress, states, source_manifest = runner.load_inputs()
    _m0, original_m1c, original_threshold, scored, _metrics, _audit = runner.phase_zero(
        historical,
        stress,
    )
    stress_scored = stress.copy()
    stress_scored["M1C_probability"] = original_m1c.predict(stress_scored)
    development = scored.loc[scored["period"].astype(str).eq("development")].copy()
    assessment = scored.loc[scored["period"].astype(str).eq("assessment")].copy()
    chosen = pd.concat(
        [
            select_rows(development, count=90).assign(parity_partition="development"),
            select_rows(assessment, count=90).assign(parity_partition="assessment"),
            select_rows(stress_scored, count=70).assign(
                parity_partition="opened_retrospective_stress"
            ),
        ],
        ignore_index=True,
    )
    runtime = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=PRIMARY / "causal_movement_feature_manifest.json",
        threshold_path=PRIMARY / "causal_movement_threshold.json",
    )
    feature_builder = M1CCausalFeatureBuilder.from_scaling_artifact(SCALING)
    state_groups = {
        (str(stock), str(session)): rows.sort_values(
            "bar_ordinal",
            kind="mergesort",
        )
        for (stock, session), rows in states.groupby(["stock", "session"], sort=False)
    }
    maximum_feature_difference = 0.0
    maximum_live_group_i_difference = 0.0
    maximum_raw_component_difference = 0.0
    maximum_raw_local_difference = 0.0
    maximum_probability_difference = 0.0
    threshold_membership_mismatches = 0
    identity_mismatches = 0
    missing_indicator_rows = 0
    partitions: dict[str, int] = {}
    stocks: set[str] = set()
    checkpoints: set[int] = set()
    group_o_names = tuple(
        name
        for name in runtime.numeric_features
        if name not in runtime.causal_group_i_features and not name.startswith("checkpoint_")
    )
    for row in chosen.itertuples(index=False):
        record = row._asdict()
        symbol = str(record["stock"])
        checkpoint = int(record["checkpoint"])
        state_rows = state_groups[(symbol, str(record["session"]))].loc[
            lambda frame, cutoff=checkpoint: frame["bar_ordinal"].astype(int).lt(cutoff)
        ]
        live_bars = tuple(
            LiveFeatureBar(
                symbol=symbol,
                session=date.fromisoformat(str(record["session"])),
                bar_ordinal=int(state.bar_ordinal),
                bar_start_timestamp=pd.Timestamp(state.bar_start_timestamp).to_pydatetime(),
                bar_complete_timestamp=pd.Timestamp(state.bar_complete_timestamp).to_pydatetime(),
                open=float(state.open),
                high=float(state.high),
                low=float(state.low),
                close=float(state.close),
                volume=float(state.volume),
                historical_relative_activity=float(state.historical_relative_activity),
                finalised=True,
                source="historical_parity_fixture",
            )
            for state in state_rows.itertuples(index=False)
        )
        live_features = feature_builder.build(
            symbol=symbol,
            checkpoint=checkpoint,
            completed_bars=live_bars,
        )
        maximum_live_group_i_difference = max(
            maximum_live_group_i_difference,
            max(
                abs(live_features.scaled_features[name] - float(record[name]))
                for name in runtime.causal_group_i_features
            ),
        )
        maximum_raw_component_difference = max(
            maximum_raw_component_difference,
            max(
                abs(live_features.raw_components[name] - float(record[f"raw_component__{name}"]))
                for name in live_features.raw_components
            ),
        )
        maximum_raw_local_difference = max(
            maximum_raw_local_difference,
            max(
                abs(live_features.raw_local_features[name] - float(record[f"raw_local__{name}"]))
                for name in live_features.raw_local_features
            ),
        )
        result = runtime.score(
            symbol=symbol,
            checkpoint=checkpoint,
            group_o_context={name: record[name] for name in group_o_names},
            causal_group_i={name: record[name] for name in runtime.causal_group_i_features},
        )
        original = float(record["M1C_probability"])
        maximum_probability_difference = max(
            maximum_probability_difference,
            abs(result.probability - original),
        )
        original_design = original_m1c.design(pd.DataFrame([record]))
        maximum_feature_difference = max(
            maximum_feature_difference,
            float(
                np.max(
                    np.abs(
                        np.asarray(result.transformed_values, dtype=float)
                        - original_design[0, : len(runtime.numeric_features)]
                    )
                )
            ),
        )
        threshold_membership_mismatches += int(
            result.threshold_passed != (original >= M1C_THRESHOLD)
        )
        expected_identity = f"{symbol}|{record['session']}|{checkpoint}"
        identity_mismatches += int(str(record["row_id"]) != expected_identity)
        missing_indicator_rows += int(
            any(float(record[name]) > 0.0 for name in group_o_names if name.endswith("_missing"))
        )
        partition = str(record["parity_partition"])
        partitions[partition] = partitions.get(partition, 0) + 1
        stocks.add(symbol)
        checkpoints.add(checkpoint)
    passed = bool(
        len(chosen) >= 250
        and maximum_feature_difference <= 1e-12
        and maximum_live_group_i_difference <= 1e-12
        and maximum_raw_component_difference <= 1e-12
        and maximum_raw_local_difference <= 1e-12
        and maximum_probability_difference <= 1e-12
        and threshold_membership_mismatches == 0
        and identity_mismatches == 0
    )
    return {
        "claims_boundary": claims_boundary(),
        "research_only": True,
        "historical_parity_only": True,
        "prospective_rows_inserted": 0,
        "model": "M1C",
        "threshold": M1C_THRESHOLD,
        "artifact_threshold_float": float(original_threshold),
        "rows_replayed": int(len(chosen)),
        "partition_rows": partitions,
        "stocks": sorted(stocks),
        "checkpoints": sorted(checkpoints),
        "near_threshold_rows_included": True,
        "missing_indicator_rows": missing_indicator_rows,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_live_group_i_difference": maximum_live_group_i_difference,
        "maximum_raw_component_difference": maximum_raw_component_difference,
        "maximum_raw_local_difference": maximum_raw_local_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "threshold_membership_mismatches": threshold_membership_mismatches,
        "row_identity_mismatches": identity_mismatches,
        "tolerance": 1e-12,
        "passed": passed,
        "activation_allowed": passed,
        "source_hashes": {
            "causal_movement_feature_manifest.json": sha256(
                PRIMARY / "causal_movement_feature_manifest.json"
            ),
            "causal_movement_threshold.json": sha256(PRIMARY / "causal_movement_threshold.json"),
            "dense_causal_surface": str(source_manifest["sources"][0]["sha256"]),
            "causal_group_i_scaling": sha256(SCALING),
            "opened_stress_state_surface": str(
                next(
                    item["sha256"]
                    for item in source_manifest["sources"]
                    if item["role"] == "completed_five_minute_stock_and_market_bars"
                )
            ),
        },
    }


def build_direction_report() -> dict[str, Any]:
    runtime = FrozenDirectionRuntime.from_artifacts(
        model_configurations_path=PRIMARY / "model_configurations.json",
        normalisation_path=PRIMARY / "stock_local_normalisation_parameters.json",
        thresholds_path=PRIMARY / "frozen_archetype_thresholds.json",
    )
    assessment = pd.read_parquet(PRIMARY / "assessment_predictions.parquet").sort_values(
        ["stock", "session", "checkpoint"],
        kind="mergesort",
    )
    feature_builder = FrozenDirectionFeatureBuilder.from_beta_artifact(
        PRIMARY / "stock_market_beta_parameters.csv"
    )
    runner = load_research_runner()
    _historical, _stress, states, _manifest = runner.load_inputs()
    state_groups = {
        (str(stock), str(session)): rows.sort_values(
            "bar_ordinal",
            kind="mergesort",
        )
        for (stock, session), rows in states.groupby(["stock", "session"], sort=False)
    }
    built_by_identity: dict[tuple[str, str, int], Any] = {}
    maximum_raw_feature_difference = 0.0
    raw_finiteness_mismatches = 0
    for row in assessment.itertuples(index=False):
        record = row._asdict()
        identity = (
            str(record["stock"]),
            str(record["session"]),
            int(record["checkpoint"]),
        )
        state_rows = state_groups[identity[:2]].loc[
            lambda frame, cutoff=identity[2]: frame["bar_ordinal"].astype(int).lt(cutoff)
        ]
        live_bars = tuple(
            DirectionFeatureBar(
                symbol=identity[0],
                session=date.fromisoformat(identity[1]),
                bar_ordinal=int(state.bar_ordinal),
                bar_start_timestamp=pd.Timestamp(state.bar_start_timestamp).to_pydatetime(),
                bar_complete_timestamp=pd.Timestamp(state.bar_complete_timestamp).to_pydatetime(),
                open=float(state.open),
                high=float(state.high),
                low=float(state.low),
                close=float(state.close),
                volume=float(state.volume),
                historical_relative_activity=float(state.historical_relative_activity),
                stock_log_return=float(state.bar_log_return),
                market_log_return=float(state.vti__bar_log_return),
                finalised=True,
            )
            for state in state_rows.itertuples(index=False)
        )
        built = feature_builder.build(
            symbol=identity[0],
            checkpoint=identity[2],
            completed_bars=live_bars,
        )
        built_by_identity[identity] = built
        for name, actual in built.raw_features.items():
            expected = float(record[f"raw__{name}"])
            actual_finite = bool(np.isfinite(actual))
            expected_finite = bool(np.isfinite(expected))
            raw_finiteness_mismatches += int(actual_finite != expected_finite)
            if actual_finite and expected_finite:
                maximum_raw_feature_difference = max(
                    maximum_raw_feature_difference,
                    abs(actual - expected),
                )
    model_results: dict[str, dict[str, Any]] = {}
    for model_id in ARCHETYPE_IDS:
        maximum_normalised_feature_difference = 0.0
        maximum_probability_difference = 0.0
        action_mismatches = 0
        for row in assessment.itertuples(index=False):
            record = row._asdict()
            built = built_by_identity[
                (
                    str(record["stock"]),
                    str(record["session"]),
                    int(record["checkpoint"]),
                )
            ]
            raw = {name: built.raw_features[name] for name in runtime.feature_names(model_id)}
            result = runtime.classify_one(
                model_id=model_id,
                raw_features=raw,
                symbol=str(record["stock"]),
                checkpoint=int(record["checkpoint"]),
                checkpoint_category=str(record["checkpoint_category"]),
                day_of_week=str(record["day_of_week"]),
            )
            expected_normalised = np.asarray(
                [float(record[name]) for name in runtime.feature_names(model_id)],
                dtype=float,
            )
            actual_normalised = np.asarray(
                [result.normalised_features[name] for name in runtime.feature_names(model_id)],
                dtype=float,
            )
            maximum_normalised_feature_difference = max(
                maximum_normalised_feature_difference,
                float(np.max(np.abs(actual_normalised - expected_normalised))),
            )
            maximum_probability_difference = max(
                maximum_probability_difference,
                abs(result.probability_up - float(record[f"{model_id}_probability"])),
            )
            action_mismatches += int(result.action != str(record[f"{model_id}_action"]))
        model_results[model_id] = {
            "rows_replayed": int(len(assessment)),
            "maximum_normalised_feature_difference": (maximum_normalised_feature_difference),
            "maximum_probability_difference": maximum_probability_difference,
            "call_put_abstain_mismatches": action_mismatches,
            "passed": bool(
                len(assessment) >= 200
                and maximum_normalised_feature_difference <= 1e-12
                and maximum_probability_difference <= 1e-12
                and action_mismatches == 0
                and maximum_raw_feature_difference <= 1e-12
                and raw_finiteness_mismatches == 0
            ),
        }
    marker = pd.to_datetime(
        assessment["maximum_direction_feature_timestamp"],
        errors="raise",
        utc=True,
    )
    trigger = pd.to_datetime(assessment["signal_timestamp"], errors="raise", utc=True)
    timing = {
        "direction_marker_bar": "T-1",
        "rows_checked": int(len(assessment)),
        "marker_not_before_trigger_count": int(marker.ge(trigger).sum()),
        "trigger_bar_exclusion_mismatches": int(
            (~assessment["trigger_bar_excluded"].astype(bool)).sum()
        ),
        "passed": bool(
            marker.lt(trigger).all() and assessment["trigger_bar_excluded"].astype(bool).all()
        ),
    }
    passed = bool(all(item["passed"] for item in model_results.values()) and timing["passed"])
    return {
        "claims_boundary": claims_boundary(),
        "research_only": True,
        "prospective_hypothesis_only": True,
        "assessment_rows_read": int(len(assessment)),
        "live_feature_rows_replayed": int(len(built_by_identity)),
        "maximum_raw_feature_difference": maximum_raw_feature_difference,
        "raw_feature_finiteness_mismatches": raw_finiteness_mismatches,
        "models": model_results,
        "timing": timing,
        "tolerance": 1e-12,
        "passed": passed,
        "display_allowed": passed,
        "source_hashes": {
            "model_configurations.json": sha256(PRIMARY / "model_configurations.json"),
            "stock_local_normalisation_parameters.json": sha256(
                PRIMARY / "stock_local_normalisation_parameters.json"
            ),
            "stock_market_beta_parameters.csv": sha256(
                PRIMARY / "stock_market_beta_parameters.csv"
            ),
            "frozen_archetype_thresholds.json": sha256(
                PRIMARY / "frozen_archetype_thresholds.json"
            ),
            "assessment_predictions.parquet": sha256(PRIMARY / "assessment_predictions.parquet"),
        },
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    if not payload.get("passed"):
        raise RuntimeError(f"parity failed; refusing to write passing artifact: {path.name}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    m1c = build_m1c_report()
    direction = build_direction_report()
    write_report(OUTPUT / "m1c_live_parity_report.json", m1c)
    write_report(OUTPUT / "direction_live_parity_report.json", direction)
    print(
        json.dumps(
            {
                "m1c_rows": m1c["rows_replayed"],
                "m1c_max_probability_difference": m1c["maximum_probability_difference"],
                "direction_rows_per_model": direction["assessment_rows_read"],
                "passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
