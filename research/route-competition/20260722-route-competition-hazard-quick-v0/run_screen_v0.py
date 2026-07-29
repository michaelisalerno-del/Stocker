#!/usr/bin/env python3
"""Run the Route-Competition Completion-Hazard Quick Screen V0."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-route-competition-hazard-mpl")

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

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.behavioural_state_dimensions_v0 import (
    apply_component_scaling,
    bar_component_frame,
    fit_component_scaling,
    opening_raw_components,
)
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
    assign_frozen_quartile,
    assign_route_resolution_state,
    binary_hazard_metrics,
    choose_primary_decision,
    completion_targets,
    fit_hazard_model,
    freeze_route_thresholds,
    permute_route_bundle,
    reject_protected_dates,
    route_competition_features_from_ledger,
    route_increment_passes,
    session_bootstrap_multiplicities,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
V2_RUNNER = (
    REPO_ROOT
    / "research"
    / "loop-funnel"
    / "20260721-emotion-regime-coarse-loop-family-v0"
    / "run_screen_v0.py"
)
PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "registered-loop-routes"
    / "20260722-hidden-loop-competing-routes-v0"
    / "artifacts"
    / "primary"
)
STATE_CACHE_PATH = Path("/tmp/stocker-route-competition-hazard-v0-states.parquet")
STATE_CACHE_MANIFEST_PATH = Path("/tmp/stocker-route-competition-hazard-v0-states-source.json")

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
DEVELOPMENT_START = "2024-01-01"
ASSESSMENT_START = "2025-01-01"
READ_END = "2025-08-22"
HIDDEN_2_3_2 = "unregistered_primitive_like__2-3-2"
HIDDEN_FAMILIES = (
    "unregistered_primitive_like__5-6-5",
    HIDDEN_2_3_2,
    "unregistered_primitive_like__2-5-2",
    "unregistered_primitive_like__4-7-4",
)
BOOTSTRAP_DRAWS = 15
NULL_REFITS = 3
BOOTSTRAP_SEED = 20260722
NULL_SEED = 20260723
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
ROUTE_STATES = (
    "BROAD_CONFLICT",
    "NARROWING",
    "DOMINANT_ROUTE",
    "LOW_ROUTE_SUPPORT",
    "OTHER",
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


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
        raise ScreenBlocker("blocked_population_reconstruction_failure", f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_contract() -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"contract safety flag differs: {key}",
            )
    limits = contract["hard_limits"]
    expected_limits = {
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
        "checkpoints": 8,
        "primary_horizons": 1,
        "primary_model_fits": 2,
        "session_bootstrap_draws": 15,
        "route_feature_null_refits": 3,
        "maximum_plots": 1,
    }
    if any(limits.get(key) != value for key, value in expected_limits.items()):
        raise ScreenBlocker("blocked_quick_route_competition_resource_limit", "hard limits differ")
    if tuple(contract["checkpoints"]) != CHECKPOINTS:
        raise ScreenBlocker(
            "blocked_quick_route_competition_resource_limit", "checkpoint set differs"
        )
    return contract


def state_cache_fingerprint(runner: ModuleType, provider_root: Path) -> dict[str, Any]:
    """Bind the bounded state cache to every material source and frozen dependency."""

    symbols = sorted({*cast(Sequence[str], runner.REGIME_CONTEXT_SYMBOLS), "VTI"})
    raw_sources: dict[str, str] = {}
    for symbol in symbols:
        stored_symbol = "VTI.US" if symbol == "VTI" else symbol
        path = provider_root / f"symbol={stored_symbol}" / "timeframe=5m" / "data.parquet"
        if not path.is_file():
            raise ScreenBlocker(
                "blocked_population_reconstruction_failure",
                f"state-cache source missing: {symbol}",
            )
        raw_sources[str(path.relative_to(provider_root))] = sha256_file(path)
    dependency_paths = {
        V2_RUNNER,
        Path(runner.PARAMETERS_PATH),
        Path(runner.PREPROCESSING_PATH),
        Path(runner.DICTIONARY_PATH),
        Path(runner.build_regime_panel.__code__.co_filename),
        Path(runner.causal_filter_summary.__code__.co_filename),
    }
    return {
        "cache_contract_version": 1,
        "provider_root": str(provider_root.resolve()),
        "raw_source_sha256": raw_sources,
        "dependency_sha256": {
            str(path.resolve()): sha256_file(path) for path in sorted(dependency_paths)
        },
        "expected_model_hash": str(runner.EXPECTED_MODEL_HASH),
        "expected_development_panel_hash": str(runner.EXPECTED_PANEL_HASH),
        "development_start": DEVELOPMENT_START,
        "read_end_inclusive": READ_END,
        "maximum_target_bar_ordinal": 37,
        "frozen_cohort": list(FROZEN_COHORT),
    }


def validate_cached_state_frame(states: pd.DataFrame, envelope: Mapping[str, Any]) -> None:
    required = {
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "causal_hard_state",
        "expected_state_age",
        "transition_probability",
        "persistence_probability",
        "posterior_entropy_reproduced",
        "historical_volume_baseline_at_bar",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *(f"state_p_{state}" for state in range(8)),
    }
    missing = sorted(required.difference(states.columns))
    if missing:
        raise ScreenBlocker(
            "blocked_population_reconstruction_failure",
            f"bounded state cache columns missing: {missing}",
        )
    if int(envelope.get("state_rows", -1)) != len(states):
        raise ScreenBlocker(
            "blocked_population_reconstruction_failure",
            "bounded state cache row count differs",
        )
    if list(envelope.get("state_columns", [])) != list(states.columns):
        raise ScreenBlocker(
            "blocked_population_reconstruction_failure",
            "bounded state cache schema differs",
        )
    timestamps = pd.to_datetime(states["bar_start_timestamp"], utc=True, errors="raise")
    if (
        bool(timestamps.lt(pd.Timestamp(DEVELOPMENT_START, tz="UTC")).any())
        or bool(timestamps.ge(pd.Timestamp("2025-08-23", tz="UTC")).any())
        or int(states["bar_ordinal"].max()) != 37
        or int(states["bar_ordinal"].min()) != 0
        or set(states["symbol"].astype(str)) != set(FROZEN_COHORT)
        or states.duplicated(["symbol", "session", "bar_ordinal"]).any()
    ):
        raise ScreenBlocker(
            "blocked_population_reconstruction_failure",
            "bounded state cache population or chronology differs",
        )
    source = cast(Mapping[str, Any], envelope.get("source", {}))
    if (
        int(source.get("protected_rows_materialised", -1)) != 0
        or int(source.get("maximum_target_bar_ordinal", -1)) != 37
        or source.get("development_emission_panel_hash")
        != envelope["fingerprint"]["expected_development_panel_hash"]
    ):
        raise ScreenBlocker(
            "blocked_population_reconstruction_failure",
            "bounded state cache source manifest differs",
        )


def load_v2_states(
    provider_root: Path,
) -> tuple[pd.DataFrame, Any, dict[str, Any], dict[str, Any]]:
    runner = load_module(V2_RUNNER, "route_hazard_v2_runner")
    preprocessing, parameters = runner.load_frozen_model()
    runner.MAX_TARGET_BAR_ORDINAL = 37
    fingerprint = state_cache_fingerprint(runner, provider_root)
    if STATE_CACHE_PATH.is_file() and STATE_CACHE_MANIFEST_PATH.is_file():
        envelope = cast(
            dict[str, Any],
            json.loads(STATE_CACHE_MANIFEST_PATH.read_text(encoding="utf-8")),
        )
        if envelope.get("fingerprint") != fingerprint:
            raise ScreenBlocker(
                "blocked_population_reconstruction_failure",
                "bounded state cache fingerprint differs",
            )
        if envelope.get("state_file_sha256") != sha256_file(STATE_CACHE_PATH):
            raise ScreenBlocker(
                "blocked_population_reconstruction_failure",
                "bounded state cache checksum differs",
            )
        states = pd.read_parquet(STATE_CACHE_PATH)
        validate_cached_state_frame(states, envelope)
        source = cast(dict[str, Any], envelope["source"])
        source["temporary_bounded_state_cache_reused"] = True
    else:
        states, source = runner.build_v2_state_panel(provider_root, preprocessing, parameters)
        cast(pd.DataFrame, states).to_parquet(
            STATE_CACHE_PATH, index=False, engine="pyarrow", compression="zstd"
        )
        envelope = {
            "fingerprint": fingerprint,
            "state_file_sha256": sha256_file(STATE_CACHE_PATH),
            "state_columns": list(cast(pd.DataFrame, states).columns),
            "state_rows": len(cast(pd.DataFrame, states)),
            "source": source,
        }
        validate_cached_state_frame(cast(pd.DataFrame, states), envelope)
        STATE_CACHE_MANIFEST_PATH.write_text(canonical_json(envelope), encoding="utf-8")
        source["temporary_bounded_state_cache_reused"] = False
    source["state_cache_validated"] = True
    source["state_cache_file_sha256"] = str(envelope["state_file_sha256"])
    source["state_cache_fingerprint_sha256"] = hashlib.sha256(
        canonical_json(fingerprint).encode("utf-8")
    ).hexdigest()
    dictionary, dictionary_manifest = runner.load_loop_dictionary()
    states = cast(pd.DataFrame, states)
    states = states.loc[states["symbol"].isin(FROZEN_COHORT)].copy()
    states["posterior_entropy"] = states["posterior_entropy_reproduced"].astype(float)
    probabilities = states.loc[:, [f"state_p_{state}" for state in range(8)]].to_numpy(dtype=float)
    ordered = np.sort(probabilities, axis=1)
    states["top_state_probability"] = ordered[:, -1]
    states["top_second_margin"] = ordered[:, -1] - ordered[:, -2]
    states["historical_relative_activity"] = states["volume"] / states[
        "historical_volume_baseline_at_bar"
    ].replace(0.0, np.nan)
    states = states.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )
    reject_protected_dates(states, column="bar_start_timestamp")
    return (
        states,
        dictionary,
        cast(dict[str, Any], source),
        cast(dict[str, Any], dictionary_manifest),
    )


def causal_state_trace_surface(states: pd.DataFrame) -> pd.DataFrame:
    """Freeze the minimal observable source needed to independently rebuild loops."""

    columns = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "causal_hard_state",
        "posterior_entropy",
        "expected_state_age",
        "transition_probability",
        "persistence_probability",
        "top_state_probability",
        "top_second_margin",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "historical_volume_baseline_at_bar",
        "historical_relative_activity",
        *(f"state_p_{state}" for state in range(8)),
    ]
    trace = states.loc[:, columns].copy()
    return trace.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )


def build_structural_ledger(states: pd.DataFrame, dictionary: Any) -> pd.DataFrame:
    """Reconstruct registered, hidden, and complete active-prefix ledgers once."""

    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    rows: list[dict[str, Any]] = []
    for (symbol, session), group in states.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        hard = ordered["causal_hard_state"].to_numpy(dtype=int)
        changes = np.concatenate(([True], hard[1:] != hard[:-1]))
        event_rows = ordered.loc[changes]
        trace = engine.scan_state_events(
            event_rows["causal_hard_state"].astype(int).tolist(),
            bar_ordinals=(event_rows["bar_ordinal"].astype(int) + 1).tolist(),
            event_timestamps=[
                value.to_pydatetime()
                for value in pd.to_datetime(event_rows["bar_start_timestamp"], utc=True)
            ],
            available_timestamps=[
                value.to_pydatetime()
                for value in pd.to_datetime(event_rows["bar_complete_timestamp"], utc=True)
            ],
        )
        for event in trace.registered_completions:
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
        for event in trace.unregistered_completions:
            canonical = canonical_unregistered_path(event.full_path)
            available = trace.state_events[event.completion_event_index].available_timestamp
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
        event_index = np.cumsum(changes).astype(int) - 1
        for position, bar in enumerate(ordered.itertuples(index=False)):
            completed_count = int(bar.bar_ordinal) + 1
            for prefix in trace.prefixes_after_event[int(event_index[position])]:
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
    columns = (
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
    ledger = pd.DataFrame(rows, columns=columns).drop_duplicates()
    return ledger.sort_values(
        ["symbol", "session", "bar_ordinal", "ledger_kind", "semantic_loop_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def opening_range_baselines(
    states: pd.DataFrame,
) -> dict[tuple[str, str, int], float]:
    rows: list[dict[str, Any]] = []
    for (symbol, session), group in states.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        ordinals = set(ordered["bar_ordinal"].astype(int))
        for checkpoint in CHECKPOINTS:
            if not set(range(checkpoint)).issubset(ordinals):
                continue
            opening = ordered.loc[ordered["bar_ordinal"].lt(checkpoint)]
            session_open = float(opening.iloc[0]["open"])
            value = (
                10_000.0
                * (float(opening["high"].max()) - float(opening["low"].min()))
                / session_open
            )
            rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "checkpoint": checkpoint,
                    "opening_range_bps": value,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["symbol", "checkpoint", "session"], kind="mergesort")
    frame["trailing_median"] = frame.groupby(["symbol", "checkpoint"], sort=False)[
        "opening_range_bps"
    ].transform(lambda values: values.expanding(min_periods=1).median().shift(1))
    return {
        (str(row.symbol), str(row.session), int(row.checkpoint)): float(row.trailing_median)
        for row in frame.itertuples(index=False)
        if np.isfinite(float(row.trailing_median)) and float(row.trailing_median) > 0.0
    }


def _local_raw_features(completed: pd.DataFrame) -> dict[str, float]:
    components = bar_component_frame(completed)
    trailing = components.iloc[-6:].reset_index(drop=True)
    ranges = trailing["true_range_bps"].to_numpy(dtype=float)
    returns = trailing["return_bps"].to_numpy(dtype=float)
    activity = trailing["historical_relative_activity"].to_numpy(dtype=float)
    mean_range = float(ranges.mean())
    mean_activity = float(activity.mean())
    current = trailing.iloc[-1]
    width = float(current["high"] - current["low"])
    body = abs(float(current["close"] - current["open"])) / max(width, 1e-12)
    return {
        "prior_6_mean_range": mean_range,
        "prior_6_price_travel": float(np.abs(returns).sum()),
        "prior_6_absolute_net_movement": abs(float(returns.sum())),
        "prior_6_activity_proxy": mean_activity,
        "recent_vs_earlier_range_ratio": float(ranges[3:].mean())
        / max(float(ranges[:3].mean()), 1e-12),
        "recent_vs_earlier_activity_ratio": float(activity[3:].mean())
        / max(float(activity[:3].mean()), 1e-12),
        "current_bar_range_vs_prior_6": float(current["true_range_bps"]) / max(mean_range, 1e-12),
        "current_bar_activity_vs_prior_6": float(current["historical_relative_activity"])
        / max(mean_activity, 1e-12),
        "current_bar_body_fraction": min(max(body, 0.0), 1.0),
        "current_bar_extreme_wick_fraction": max(
            float(current["upper_wick_fraction"]),
            float(current["lower_wick_fraction"]),
        ),
    }


def _prefix_margin(prefixes: pd.DataFrame, completed_count: int) -> float:
    current = prefixes.loc[prefixes["bar_ordinal"].eq(completed_count)]
    depths: list[float] = []
    for row in current.itertuples(index=False):
        progress = int(row.progress_states)
        remaining = int(row.transitions_remaining)
        denominator = progress + remaining - 1
        depths.append(0.0 if denominator <= 0 else (progress - 1) / denominator)
    depths.sort(reverse=True)
    if not depths:
        return 0.0
    return float(depths[0] - (depths[1] if len(depths) > 1 else 0.0))


def _structural_features(
    prefixes: pd.DataFrame,
    registered: pd.DataFrame,
    hidden: pd.DataFrame,
    *,
    checkpoint: int,
) -> dict[str, Any]:
    registered_input = registered.rename(columns={"bar_ordinal": "completion_bar_ordinal"})
    route = route_competition_features_from_ledger(
        prefixes,
        registered_input.loc[:, ["completion_bar_ordinal", "semantic_loop_id"]],
        checkpoint=checkpoint,
    )
    prior_six = registered.loc[
        registered["bar_ordinal"].ge(checkpoint - 6) & registered["bar_ordinal"].lt(checkpoint)
    ]
    prior_twelve = registered.loc[
        registered["bar_ordinal"].ge(checkpoint - 12) & registered["bar_ordinal"].lt(checkpoint)
    ]
    current_prefix_ids = set(
        prefixes.loc[prefixes["bar_ordinal"].eq(checkpoint), "semantic_loop_id"].astype(str)
    )
    prior_ids = set(prior_twelve["semantic_loop_id"].astype(str))
    hidden_prior = hidden.loc[
        hidden["bar_ordinal"].ge(checkpoint - 6) & hidden["bar_ordinal"].lt(checkpoint)
    ]
    earlier_registered = registered.loc[registered["bar_ordinal"].lt(checkpoint)]
    if earlier_registered.empty:
        bars_since = float(checkpoint + 1)
    else:
        bars_since = float(checkpoint - int(earlier_registered["bar_ordinal"].astype(int).max()))
    route.update(
        {
            "any_registered_completion_prior_6": float(not prior_six.empty),
            "any_registered_completion_prior_12": float(not prior_twelve.empty),
            "same_identity_active_prefix_with_prior_completion": float(
                bool(current_prefix_ids.intersection(prior_ids))
            ),
            "any_hidden_event_prior_6": float(not hidden_prior.empty),
            "hidden_2_3_2_prior_6": float(
                hidden_prior["family_id"].astype(str).eq(HIDDEN_2_3_2).any()
            ),
            "bars_since_latest_registered_completion": bars_since,
            "depth_margin_change_last_3_bars": _prefix_margin(prefixes, checkpoint)
            - _prefix_margin(prefixes, checkpoint - 3),
        }
    )
    return route


def build_raw_decision_panel(
    states: pd.DataFrame, ledger: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    baselines = opening_range_baselines(states)
    prefixes = ledger.loc[ledger["ledger_kind"].eq("active_prefix")].copy()
    registered = ledger.loc[ledger["ledger_kind"].eq("registered_completion")].copy()
    hidden = ledger.loc[ledger["ledger_kind"].eq("hidden_completion")].copy()
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
    empty_prefixes = prefixes.iloc[:0]
    empty_registered = registered.iloc[:0]
    empty_hidden = hidden.iloc[:0]
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    possible_rows = 0
    for (symbol, session), group in states.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
        ordinals = set(ordered["bar_ordinal"].astype(int))
        group_key = (str(symbol), str(session))
        session_prefixes = prefix_groups.get(group_key, empty_prefixes)
        session_registered = registered_groups.get(group_key, empty_registered)
        session_hidden = hidden_groups.get(group_key, empty_hidden)
        completion_counts = session_registered["bar_ordinal"].astype(int).tolist()
        for checkpoint in CHECKPOINTS:
            possible_rows += 1
            required = set(range(checkpoint + 3))
            reason: str | None = None
            if not required.issubset(ordinals):
                reason = "source_data_unavailable"
            current_rows = ordered.loc[ordered["bar_ordinal"].eq(checkpoint - 1)]
            if reason is None and (
                current_rows.empty
                or (
                    "bar_is_complete" in current_rows
                    and not bool(current_rows.iloc[0]["bar_is_complete"])
                )
            ):
                reason = "checkpoint_bar_incomplete"
            if reason is None and checkpoint + 3 > 78:
                reason = "fewer_than_three_future_completed_bars"
            baseline = baselines.get((str(symbol), str(session), checkpoint))
            if reason is None and baseline is None:
                reason = "required_causal_feature_reconstruction_failed"
            if reason is not None:
                exclusions.append(
                    {
                        "symbol": str(symbol),
                        "session": str(session),
                        "checkpoint": checkpoint,
                        "reason": reason,
                    }
                )
                continue

            completed = ordered.loc[ordered["bar_ordinal"].lt(checkpoint)].copy()
            if (
                len(completed) != checkpoint
                or not np.isfinite(
                    completed["historical_relative_activity"].to_numpy(dtype=float)
                ).all()
            ):
                exclusions.append(
                    {
                        "symbol": str(symbol),
                        "session": str(session),
                        "checkpoint": checkpoint,
                        "reason": "required_causal_feature_reconstruction_failed",
                    }
                )
                continue
            components = bar_component_frame(completed)
            first_close = float(components.iloc[checkpoint // 2 - 1]["close"])
            session_open = float(components.iloc[0]["open"])
            last_close = float(components.iloc[-1]["close"])
            progress = 10_000.0 * (last_close / session_open - 1.0)
            earlier = 10_000.0 * (first_close / session_open - 1.0)
            recent = 10_000.0 * (last_close / first_close - 1.0)
            acceleration = recent - earlier
            raw_components = opening_raw_components(
                components,
                trailing_opening_range_median_bps=float(baseline),
                signed_progress_bps=progress,
                signed_progress_acceleration_bps=acceleration,
                return_gap_bps=progress,
            )
            local = _local_raw_features(completed)
            structural = _structural_features(
                session_prefixes,
                session_registered,
                session_hidden,
                checkpoint=checkpoint,
            )
            target = completion_targets(
                checkpoint=checkpoint, completion_ordinals=completion_counts
            )
            future_counts = sorted(
                value for value in completion_counts if checkpoint < value <= checkpoint + 3
            )
            current = current_rows.iloc[0]
            record: dict[str, Any] = {
                "row_id": f"{symbol}|{session}|{checkpoint}",
                "symbol": str(symbol),
                "session": str(session),
                "year_month": str(session)[:7],
                "period": ("development" if str(session) < ASSESSMENT_START else "assessment"),
                "checkpoint": checkpoint,
                "checkpoint_bar_ordinal_zero_based": checkpoint - 1,
                "checkpoint_timestamp_utc": pd.Timestamp(current["bar_start_timestamp"]),
                "feature_available_timestamp_utc": pd.Timestamp(current["bar_complete_timestamp"]),
                "target_horizon_end_completed_count": checkpoint + 3,
                "future_registered_completion_counts": json.dumps(future_counts),
                "first_future_registered_completion_count": (
                    float(future_counts[0]) if future_counts else np.nan
                ),
                "raw_signed_progress": progress,
                "raw_signed_progress_acceleration": acceleration,
                "posterior_entropy": float(current["posterior_entropy"]),
                "transition_probability": float(current["transition_probability"]),
                "persistence_probability": float(current["persistence_probability"]),
                "expected_state_age": float(current["expected_state_age"]),
                "top_state_probability": float(current["top_state_probability"]),
                "top_second_margin": float(current["top_second_margin"]),
                **target,
                **structural,
            }
            for component, value in raw_components.items():
                record[f"raw_component__{component}"] = float(value)
            for feature, value in local.items():
                record[f"raw_local__{feature}"] = float(value)
            records.append(record)
    panel = pd.DataFrame(records)
    exclusion_frame = pd.DataFrame(
        exclusions, columns=["symbol", "session", "checkpoint", "reason"]
    )
    if panel.empty:
        raise ScreenBlocker("blocked_population_reconstruction_failure", "decision panel is empty")
    return panel, exclusion_frame, possible_rows


def add_development_frozen_baseline_features(
    raw_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    panel = raw_panel.copy()
    for _, indices in panel.groupby(["session", "checkpoint"], sort=True).groups.items():
        index = list(indices)
        progress = panel.loc[index, "raw_signed_progress"].to_numpy(dtype=float)
        acceleration = panel.loc[index, "raw_signed_progress_acceleration"].to_numpy(dtype=float)
        panel.loc[index, "raw_component__signed_progress"] = progress - np.median(progress)
        panel.loc[index, "raw_component__absolute_progress"] = np.abs(
            progress - np.median(progress)
        )
        panel.loc[index, "raw_component__signed_progress_acceleration"] = acceleration - np.median(
            acceleration
        )
    component_columns = [f"raw_component__{value}" for value in COMPONENTS]
    component_alias = panel.rename(
        columns={f"raw_component__{value}": value for value in COMPONENTS}
    )
    development = component_alias.loc[component_alias["period"].eq("development")]
    scaling = fit_component_scaling(
        development,
        components=COMPONENTS,
        checkpoint_column="checkpoint",
    )
    scaled = apply_component_scaling(
        component_alias,
        scaling,
        components=COMPONENTS,
        checkpoint_column="checkpoint",
    )
    panel["arousal"] = scaled[["z_activity_effort", "z_range_effort", "z_travel_effort"]].mean(
        axis=1
    )
    panel["conviction"] = scaled[
        ["z_absolute_efficiency", "z_close_retention", "z_directional_persistence"]
    ].mean(axis=1)
    panel["tension"] = (
        scaled[["z_activity_effort", "z_compression", "z_extreme_rejection"]].mean(axis=1)
        - scaled["z_absolute_progress"]
    )
    panel["signed_pressure"] = scaled[
        [
            "z_signed_progress",
            "z_signed_efficiency",
            "z_mean_close_location",
            "z_boundary_slope",
        ]
    ].mean(axis=1)
    for component in COMPONENTS:
        panel[f"z_component__{component}"] = scaled[f"z_{component}"]

    local_scaling: dict[str, Any] = {}
    for (symbol, checkpoint), group in panel.loc[panel["period"].eq("development")].groupby(
        ["symbol", "checkpoint"], sort=True
    ):
        key = f"{symbol}|{int(checkpoint)}"
        local_scaling[key] = {}
        for feature in LOCAL_FEATURES:
            values = group[f"raw_local__{feature}"].astype(float)
            center = float(values.median())
            scale = float(values.quantile(0.75) - values.quantile(0.25))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            local_scaling[key][feature] = {"center": center, "scale": scale}
    for feature in LOCAL_FEATURES:
        panel[feature] = np.nan
    for (symbol, checkpoint), indices in panel.groupby(
        ["symbol", "checkpoint"], sort=True
    ).groups.items():
        key = f"{symbol}|{int(checkpoint)}"
        if key not in local_scaling:
            raise ScreenBlocker(
                "blocked_population_reconstruction_failure",
                f"development stock-clock scaling missing: {key}",
            )
        index = list(indices)
        for feature in LOCAL_FEATURES:
            frozen = local_scaling[key][feature]
            values = (
                panel.loc[index, f"raw_local__{feature}"].to_numpy(dtype=float)
                - float(frozen["center"])
            ) / float(frozen["scale"])
            panel.loc[index, feature] = np.clip(values, -5.0, 5.0)
    for checkpoint in CHECKPOINTS:
        panel[f"checkpoint_{checkpoint}"] = panel["checkpoint"].eq(checkpoint).astype(float)
    component_manifest = {
        str(checkpoint): {
            component: dict(value.as_dict()) for component, value in checkpoint_scaling.items()
        }
        for checkpoint, checkpoint_scaling in scaling.items()
    }
    if (
        component_columns
        and not np.isfinite(panel.loc[:, list(BASELINE_FEATURES)].to_numpy(dtype=float)).all()
    ):
        raise ScreenBlocker(
            "blocked_population_reconstruction_failure", "non-finite baseline features"
        )
    return panel, component_manifest, local_scaling


def add_frozen_route_labels_and_weights(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = panel.copy()
    development = output.loc[output["period"].eq("development")]
    thresholds = freeze_route_thresholds(development)
    output["route_resolution_state"] = assign_route_resolution_state(output, thresholds)
    for column, label in (
        ("top_prefix_depth_fraction", "top_prefix_depth_quartile"),
        ("top_minus_second_prefix_depth", "depth_margin_quartile"),
        ("prefix_family_entropy", "prefix_family_entropy_quartile"),
    ):
        output[label] = assign_frozen_quartile(output[column], thresholds[column])
    output["active_prefix_count_bin"] = pd.cut(
        output["active_prefix_count"],
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=["0-2", "3-5", "6-10", "11+"],
        right=True,
    ).astype("string")
    output["candidate_count_change_3_bar_state"] = np.select(
        [
            output["active_prefix_count_change_last_3_bars"].lt(0),
            output["active_prefix_count_change_last_3_bars"].gt(0),
        ],
        ["decreasing", "increasing"],
        default="unchanged",
    )
    transition_median = float(development["transition_probability"].median())
    entropy_median = float(development["posterior_entropy"].median())
    output["transition_probability_half"] = np.where(
        output["transition_probability"].le(transition_median), "low", "high"
    )
    output["posterior_entropy_half"] = np.where(
        output["posterior_entropy"].le(entropy_median), "low", "high"
    )
    output["recent_registered_completion_group"] = np.where(
        output["any_registered_completion_prior_6"].gt(0),
        "recent_registered_completion_within_6",
        "no_recent_registered_completion",
    )
    slate_size = output.groupby(["period", "session", "checkpoint"], sort=False)[
        "symbol"
    ].transform("size")
    output["eligible_stocks_in_session_checkpoint"] = slate_size.astype(int)
    output["row_weight"] = 1.0 / slate_size.astype(float)
    frozen = {
        "route_quartiles": {key: list(value) for key, value in thresholds.items()},
        "transition_probability_median": transition_median,
        "posterior_entropy_median": entropy_median,
        "route_resolution_definitions": {
            "BROAD_CONFLICT": "entropy >= development Q75 and depth margin <= development Q25",
            "NARROWING": (
                "active-prefix count change over three bars < 0 and depth-margin change > 0"
            ),
            "DOMINANT_ROUTE": "top depth and depth margin both >= development Q75",
            "LOW_ROUTE_SUPPORT": "active prefix count <= 2",
            "OTHER": "all remaining rows",
            "precedence": list(ROUTE_STATES),
        },
    }
    return output, frozen


def _weighted_precision(
    frame: pd.DataFrame, probability_column: str, threshold: float
) -> tuple[float, int]:
    selected = frame.loc[frame[probability_column].ge(threshold)]
    if selected.empty:
        return float("nan"), 0
    weights = selected["row_weight"].to_numpy(dtype=float)
    labels = selected["registered_completion_next_3_bars"].to_numpy(dtype=float)
    return float(np.average(labels, weights=weights)), int(len(selected))


def prediction_boundaries(panel: pd.DataFrame) -> dict[str, dict[str, float]]:
    development = panel.loc[panel["period"].eq("development")]
    return {
        model: {
            "top_decile": float(development[f"{model}_probability"].quantile(0.90)),
            "top_quintile": float(development[f"{model}_probability"].quantile(0.80)),
        }
        for model in ("H0", "H1")
    }


def model_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    probability_column = f"{model}_probability"
    metrics = binary_hazard_metrics(
        frame["registered_completion_next_3_bars"],
        frame[probability_column],
        frame["row_weight"],
    )
    decile_precision, decile_rows = _weighted_precision(
        frame, probability_column, float(boundaries[model]["top_decile"])
    )
    quintile_precision, quintile_rows = _weighted_precision(
        frame, probability_column, float(boundaries[model]["top_quintile"])
    )
    base = float(metrics["base_rate"])
    metrics.update(
        {
            "model": model,
            "sessions": int(frame["session"].nunique()),
            "stocks": int(frame["symbol"].nunique()),
            "top_decile_probability_boundary": float(boundaries[model]["top_decile"]),
            "top_decile_precision": decile_precision,
            "top_decile_lift": decile_precision / base if base > 0.0 else float("nan"),
            "top_decile_rows": decile_rows,
            "top_quintile_probability_boundary": float(boundaries[model]["top_quintile"]),
            "top_quintile_precision": quintile_precision,
            "top_quintile_lift": quintile_precision / base if base > 0.0 else float("nan"),
            "top_quintile_rows": quintile_rows,
        }
    )
    return metrics


def fit_primary_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    development = panel.loc[panel["period"].eq("development")].copy()
    fitted_h0 = fit_hazard_model(development, features=BASELINE_FEATURES)
    fitted_h1 = fit_hazard_model(development, features=H1_FEATURES)
    output = panel.copy()
    output["H0_probability"] = fitted_h0.predict_probability(output)
    output["H1_probability"] = fitted_h1.predict_probability(output)
    specifications = {"H0": fitted_h0.as_dict(), "H1": fitted_h1.as_dict()}
    configurations = {
        **SAFETY_FLAGS,
        "primary_fit_count": 2,
        "model_ladder": {
            "H0": "compressed-transition baseline",
            "H1": "H0 plus the 15 fixed causal route-competition features",
        },
        "binding_comparison": "H1 versus H0",
        "specification": {
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "random_state": 20260722,
            "n_jobs": 1,
        },
    }
    return output, specifications, configurations


def build_metric_tables(
    panel: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    pooled = pd.DataFrame(
        [model_metrics(assessment, model=model, boundaries=boundaries) for model in ("H0", "H1")]
    )
    checkpoint_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    for checkpoint, group in assessment.groupby("checkpoint", sort=True):
        for model in ("H0", "H1"):
            checkpoint_rows.append(
                {
                    "checkpoint": int(checkpoint),
                    **model_metrics(group, model=model, boundaries=boundaries),
                }
            )
    for month, group in assessment.groupby("year_month", sort=True):
        for model in ("H0", "H1"):
            monthly_rows.append(
                {
                    "year_month": str(month),
                    **model_metrics(group, model=model, boundaries=boundaries),
                }
            )
    dimensions = (
        "transition_probability_half",
        "posterior_entropy_half",
        "recent_registered_completion_group",
        "route_resolution_state",
    )
    for dimension in dimensions:
        for value, group in assessment.groupby(dimension, sort=True):
            for model in ("H0", "H1"):
                subgroup_rows.append(
                    {
                        "subgroup_dimension": dimension,
                        "subgroup_value": str(value),
                        **model_metrics(group, model=model, boundaries=boundaries),
                    }
                )
    return (
        pooled,
        pd.DataFrame(checkpoint_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(subgroup_rows),
    )


def build_route_resolution_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    assessment = panel.loc[panel["period"].eq("assessment")]
    rows: list[dict[str, Any]] = []
    diagnostics = (
        ("active_prefix_count", "active_prefix_count_bin", ("0-2", "3-5", "6-10", "11+")),
        ("top_prefix_depth", "top_prefix_depth_quartile", ("Q1", "Q2", "Q3", "Q4")),
        ("depth_margin", "depth_margin_quartile", ("Q1", "Q2", "Q3", "Q4")),
        (
            "prefix_family_entropy",
            "prefix_family_entropy_quartile",
            ("Q1", "Q2", "Q3", "Q4"),
        ),
        (
            "candidate_count_change_3_bars",
            "candidate_count_change_3_bar_state",
            ("decreasing", "unchanged", "increasing"),
        ),
        ("route_resolution_state", "route_resolution_state", ROUTE_STATES),
    )
    for diagnostic, column, expected_bins in diagnostics:
        for value in expected_bins:
            group = assessment.loc[assessment[column].astype(str).eq(value)]
            if group.empty:
                rows.append(
                    {
                        "diagnostic": diagnostic,
                        "bin": value,
                        "rows": 0,
                        "sessions": 0,
                        "stocks": 0,
                        "registered_completion_next_3_rate": np.nan,
                        "registered_completion_next_1_rate": np.nan,
                        "inference_supported": False,
                    }
                )
                continue
            weights = group["row_weight"].to_numpy(dtype=float)
            target = group["registered_completion_next_3_bars"].to_numpy(dtype=float)
            target_one = group["registered_completion_next_1_bar"].to_numpy(dtype=float)
            rows.append(
                {
                    "diagnostic": diagnostic,
                    "bin": str(value),
                    "rows": int(len(group)),
                    "sessions": int(group["session"].nunique()),
                    "stocks": int(group["symbol"].nunique()),
                    "registered_completion_next_3_rate": float(np.average(target, weights=weights)),
                    "registered_completion_next_1_rate": float(
                        np.average(target_one, weights=weights)
                    ),
                    "inference_supported": bool(len(group) >= 100),
                }
            )
    return pd.DataFrame(rows)


def metric_increments(
    frame: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    h0 = model_metrics(frame, model="H0", boundaries=boundaries)
    h1 = model_metrics(frame, model="H1", boundaries=boundaries)
    return {
        "log_loss_improvement": float(h0["log_loss"]) - float(h1["log_loss"]),
        "brier_improvement": float(h0["brier_score"]) - float(h1["brier_score"]),
        "auc_improvement": float(h1["auc"]) - float(h0["auc"]),
        "average_precision_improvement": float(h1["average_precision"])
        - float(h0["average_precision"]),
        "top_decile_precision_improvement": float(h1["top_decile_precision"])
        - float(h0["top_decile_precision"]),
    }


def run_bootstrap(
    assessment: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    draws = session_bootstrap_multiplicities(
        assessment["session"], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    draw_rows: list[dict[str, Any]] = []
    for draw_index, multiplicity in enumerate(draws):
        sampled = assessment.copy()
        sampled["row_weight"] = sampled["row_weight"].to_numpy(dtype=float) * multiplicity
        sampled = sampled.loc[sampled["row_weight"].gt(0)].copy()
        draw_rows.append(
            {
                "record_type": "draw",
                "draw": draw_index,
                **metric_increments(sampled, boundaries),
            }
        )
    draw_frame = pd.DataFrame(draw_rows)
    summary_rows: list[dict[str, Any]] = []
    point = metric_increments(assessment, boundaries)
    for statistic in point:
        values = draw_frame[statistic].to_numpy(dtype=float)
        for level in (0.80, 0.90, 0.95):
            alpha = 1.0 - level
            summary_rows.append(
                {
                    "record_type": "interval",
                    "statistic": statistic,
                    "interval_level": level,
                    "lower": float(np.quantile(values, alpha / 2.0)),
                    "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
                    "point_estimate": float(point[statistic]),
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                }
            )
    return pd.concat([draw_frame, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)


def _binary_subset_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    result = binary_hazard_metrics(
        frame["registered_completion_next_3_bars"],
        frame[probability_column],
        frame["row_weight"],
    )
    return {
        "log_loss": float(result["log_loss"]),
        "brier_score": float(result["brier_score"]),
        "auc": float(result["auc"]),
    }


def run_route_nulls(
    panel: pd.DataFrame,
    real_increments: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output = panel.copy()
    assessment_mask = output["period"].eq("assessment")
    h0_assessment = _binary_subset_metrics(output.loc[assessment_mask], "H0_probability")
    null_rows: list[dict[str, Any]] = []
    null_specifications: dict[str, Any] = {}
    for draw in range(NULL_REFITS):
        seed = NULL_SEED + draw
        permuted = permute_route_bundle(
            output,
            route_features=ROUTE_FEATURES,
            strata=("period", "session", "checkpoint"),
            seed=seed,
        )
        development = permuted.loc[permuted["period"].eq("development")]
        fitted = fit_hazard_model(development, features=H1_FEATURES)
        probability_column = f"route_null_{draw}_probability"
        permuted[probability_column] = fitted.predict_probability(permuted)
        output.loc[assessment_mask, probability_column] = permuted.loc[
            assessment_mask, probability_column
        ].to_numpy(dtype=float)
        assessment_metrics = _binary_subset_metrics(
            permuted.loc[assessment_mask], probability_column
        )
        increment = {
            "log_loss_improvement": h0_assessment["log_loss"] - assessment_metrics["log_loss"],
            "brier_improvement": h0_assessment["brier_score"] - assessment_metrics["brier_score"],
            "auc_improvement": assessment_metrics["auc"] - h0_assessment["auc"],
        }
        bundle_hash = stable_frame_hash(
            permuted,
            ["period", "session", "checkpoint", "symbol", *ROUTE_FEATURES],
        )
        null_rows.append(
            {
                "record_type": "draw",
                "draw": draw,
                "seed": seed,
                "route_bundle_hash": bundle_hash,
                **assessment_metrics,
                **increment,
            }
        )
        null_specifications[str(draw)] = fitted.as_dict()
    null_frame = pd.DataFrame(null_rows)
    comparison_rows = []
    for statistic in (
        "log_loss_improvement",
        "brier_improvement",
        "auc_improvement",
    ):
        real = float(real_increments[statistic])
        comparison_rows.append(
            {
                "record_type": "comparison",
                "statistic": statistic,
                "real_increment": real,
                "real_exceeds_null_count": int((real > null_frame[statistic]).sum()),
                "null_draws": NULL_REFITS,
            }
        )
    metrics = pd.concat([null_frame, pd.DataFrame(comparison_rows)], ignore_index=True, sort=False)
    return output, metrics, null_specifications


def concentration_and_support(
    panel: pd.DataFrame,
    *,
    possible_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    weights = assessment["row_weight"].to_numpy(dtype=float)
    total_weight = float(weights.sum())
    stock_shares = assessment.groupby("symbol", sort=True)["row_weight"].sum() / total_weight
    base_rate = float(
        np.average(
            assessment["registered_completion_next_3_bars"].to_numpy(dtype=float),
            weights=weights,
        )
    )
    feature_retention = float(len(panel) / possible_rows) if possible_rows else 0.0
    values: list[dict[str, Any]] = [
        {
            "metric": "assessment_rows",
            "value": len(assessment),
            "threshold": 30000,
            "passed": len(assessment) >= 30000,
        },
        {
            "metric": "assessment_sessions",
            "value": assessment["session"].nunique(),
            "threshold": 140,
            "passed": assessment["session"].nunique() >= 140,
        },
        {
            "metric": "assessment_stocks",
            "value": assessment["symbol"].nunique(),
            "threshold": 15,
            "passed": assessment["symbol"].nunique() >= 15,
        },
        {
            "metric": "assessment_months",
            "value": assessment["year_month"].nunique(),
            "threshold": 8,
            "passed": assessment["year_month"].nunique() >= 8,
        },
        {
            "metric": "positive_outcomes",
            "value": assessment["registered_completion_next_3_bars"].sum(),
            "threshold": 500,
            "passed": assessment["registered_completion_next_3_bars"].sum() >= 500,
        },
        {
            "metric": "feature_retention",
            "value": feature_retention,
            "threshold": 0.95,
            "passed": feature_retention >= 0.95,
        },
        {
            "metric": "maximum_weighted_stock_share",
            "value": float(stock_shares.max()),
            "threshold": 0.10,
            "passed": float(stock_shares.max()) <= 0.10,
        },
        {
            "metric": "maximum_target_class_share",
            "value": max(base_rate, 1.0 - base_rate),
            "threshold": 0.90,
            "passed": max(base_rate, 1.0 - base_rate) <= 0.90,
        },
    ]
    for state in ROUTE_STATES:
        count = int(assessment["route_resolution_state"].eq(state).sum())
        values.append(
            {
                "metric": f"route_resolution_state_rows__{state}",
                "value": count,
                "threshold": 100,
                "passed": count >= 100,
                "blocks_primary": False,
            }
        )
    primary_rows = pd.DataFrame(values).loc[
        lambda frame: ~frame.get("blocks_primary", False).fillna(False)
    ]
    support_passed = bool(primary_rows["passed"].all())
    summary = {
        "assessment_rows": int(len(assessment)),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "positive_outcomes": int(assessment["registered_completion_next_3_bars"].sum()),
        "feature_retention": feature_retention,
        "maximum_weighted_stock_share": float(stock_shares.max()),
        "maximum_target_class_share": max(base_rate, 1.0 - base_rate),
        "support_passed": support_passed,
    }
    blocker = None if support_passed else "blocked_insufficient_support"
    return pd.DataFrame(values), summary, blocker


def _bootstrap_lower(bootstrap: pd.DataFrame, statistic: str, level: float) -> float:
    row = bootstrap.loc[
        bootstrap["record_type"].eq("interval")
        & bootstrap["statistic"].eq(statistic)
        & bootstrap["interval_level"].eq(level)
    ]
    if len(row) != 1:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "bootstrap interval missing"
        )
    return float(row.iloc[0]["lower"])


def choose_decision(
    panel: pd.DataFrame,
    pooled: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    route_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null_metrics: pd.DataFrame,
    support: Mapping[str, Any],
    blocker: str | None,
) -> dict[str, Any]:
    assessment = panel.loc[panel["period"].eq("assessment")]
    increments = metric_increments(assessment, prediction_boundaries(panel))
    monthly_pivot = monthly_metrics.pivot(index="year_month", columns="model", values="log_loss")
    positive_months = int((monthly_pivot["H0"] - monthly_pivot["H1"] > 0.0).sum())
    checkpoint_log = checkpoint_metrics.pivot(
        index="checkpoint", columns="model", values="log_loss"
    )
    checkpoint_brier = checkpoint_metrics.pivot(
        index="checkpoint", columns="model", values="brier_score"
    )
    adverse = int(
        (
            (checkpoint_log["H1"] - checkpoint_log["H0"] > 0.005)
            | (checkpoint_brier["H1"] - checkpoint_brier["H0"] > 0.002)
        ).sum()
    )
    comparisons = null_metrics.loc[null_metrics["record_type"].eq("comparison")]
    comparison_counts = {
        str(row.statistic): int(row.real_exceeds_null_count)
        for row in comparisons.itertuples(index=False)
    }
    real_exceeds_all = bool(
        comparison_counts.get("log_loss_improvement", 0) == NULL_REFITS
        or comparison_counts.get("brier_improvement", 0) == NULL_REFITS
    )
    gates: dict[str, object] = {
        **increments,
        "bootstrap_80_log_loss_lower": _bootstrap_lower(bootstrap, "log_loss_improvement", 0.80),
        "bootstrap_80_brier_lower": _bootstrap_lower(bootstrap, "brier_improvement", 0.80),
        "positive_months": positive_months,
        "materially_adverse_checkpoints": adverse,
        "real_exceeds_all_nulls": real_exceeds_all,
        "concentration_passed": bool(support["maximum_weighted_stock_share"] <= 0.10)
        and bool(support["maximum_target_class_share"] <= 0.90),
    }
    h1_passed = route_increment_passes(gates)
    state_rows = route_metrics.loc[
        route_metrics["diagnostic"].eq("route_resolution_state")
    ].set_index("bin")
    supported_states = all(
        state in state_rows.index and bool(state_rows.loc[state, "inference_supported"])
        for state in (
            "BROAD_CONFLICT",
            "NARROWING",
            "DOMINANT_ROUTE",
            "LOW_ROUTE_SUPPORT",
        )
    )
    route_ordered = False
    if supported_states:
        rate = state_rows["registered_completion_next_3_rate"].astype(float)
        route_ordered = bool(
            min(rate["NARROWING"], rate["DOMINANT_ROUTE"])
            > max(rate["BROAD_CONFLICT"], rate["LOW_ROUTE_SUPPORT"])
        )
    h0 = pooled.loc[pooled["model"].eq("H0")].iloc[0]
    base = float(h0["base_rate"])
    constant_log_loss = float(
        -np.average(
            assessment["registered_completion_next_3_bars"] * math.log(max(base, 1e-12))
            + (1 - assessment["registered_completion_next_3_bars"])
            * math.log(max(1.0 - base, 1e-12)),
            weights=assessment["row_weight"],
        )
    )
    h0_meaningful = bool(float(h0["auc"]) > 0.5 and float(h0["log_loss"]) < constant_log_loss)
    primary = choose_primary_decision(
        blocker=blocker,
        h1_passed=h1_passed,
        route_narrowing_ordered=route_ordered,
        h0_meaningful=h0_meaningful,
    )
    return {
        **SAFETY_FLAGS,
        "primary_decision": primary,
        "binding_question": (
            "Does causal route competition and candidate-set narrowing improve "
            "three-bar registered-completion hazard beyond H0?"
        ),
        "H1_passed_before_support_blocker": h1_passed,
        "route_narrowing_ordered": route_ordered,
        "H0_meaningful": h0_meaningful,
        "constant_probability_log_loss": constant_log_loss,
        "increments": increments,
        "decision_gates": gates,
        "null_real_exceeds_counts": comparison_counts,
        "support": dict(support),
        "blocker": blocker,
        "material_adversity_definition": {
            "checkpoint_log_loss_deterioration_above": 0.005,
            "checkpoint_brier_deterioration_above": 0.002,
        },
        "descriptive_order_definition": (
            "both NARROWING and DOMINANT_ROUTE rates exceed both BROAD_CONFLICT "
            "and LOW_ROUTE_SUPPORT, with >=100 rows each"
        ),
    }


def verify_predecessor_registered_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    predecessor_path = PREDECESSOR_PRIMARY / "registered_completion_ledger.parquet"
    predecessor = pd.read_parquet(predecessor_path)
    current = ledger.loc[
        ledger["ledger_kind"].eq("registered_completion") & ledger["bar_ordinal"].le(31)
    ].copy()
    current["completion_bar_ordinal"] = current["bar_ordinal"].astype(int) - 1
    identity = (
        "symbol",
        "session",
        "completion_bar_ordinal",
        "semantic_loop_id",
        "orientation_id",
    )
    predecessor_unique = predecessor.loc[:, list(identity)].drop_duplicates()
    current_unique = current.loc[:, list(identity)].drop_duplicates()
    predecessor_hash = stable_frame_hash(predecessor_unique, identity)
    current_hash = stable_frame_hash(current_unique, identity)
    passed = bool(
        len(predecessor_unique) == len(current_unique) and predecessor_hash == current_hash
    )
    if not passed:
        raise ScreenBlocker(
            "blocked_population_reconstruction_failure",
            "registered V2 ledger does not reproduce through zero-based bar 30",
        )
    return {
        "source_path": str(predecessor_path.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(predecessor_path),
        "source_rows": int(len(predecessor_unique)),
        "reconstructed_rows": int(len(current_unique)),
        "source_identity_hash": predecessor_hash,
        "reconstructed_identity_hash": current_hash,
        "passed": passed,
    }


def checkpoint_manifest(
    panel: pd.DataFrame, exclusions: pd.DataFrame, possible_rows: int
) -> dict[str, Any]:
    support = []
    for checkpoint in CHECKPOINTS:
        included = panel.loc[panel["checkpoint"].eq(checkpoint)]
        excluded = exclusions.loc[exclusions["checkpoint"].eq(checkpoint)]
        support.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_bar_ordinal_zero_based": checkpoint - 1,
                "feature_available_after_completed_bar_count": checkpoint,
                "target_completed_bar_counts": [
                    checkpoint + 1,
                    checkpoint + 2,
                    checkpoint + 3,
                ],
                "included_rows": int(len(included)),
                "development_rows": int(included["period"].eq("development").sum()),
                "assessment_rows": int(included["period"].eq("assessment").sum()),
                "sessions": int(included["session"].nunique()),
                "stocks": int(included["symbol"].nunique()),
                "excluded_rows": int(len(excluded)),
                "exclusions_by_reason": {
                    str(key): int(value) for key, value in excluded["reason"].value_counts().items()
                },
                "minimum_feature_timestamp": str(included["feature_available_timestamp_utc"].min()),
                "maximum_feature_timestamp": str(included["feature_available_timestamp_utc"].max()),
            }
        )
    return {
        **SAFETY_FLAGS,
        "checkpoints": list(CHECKPOINTS),
        "checkpoint_count": len(CHECKPOINTS),
        "possible_rows_before_causal_feature_exclusions": possible_rows,
        "included_rows": int(len(panel)),
        "excluded_rows": int(len(exclusions)),
        "support": support,
    }


def plot_screen(route_metrics: pd.DataFrame, pooled: pd.DataFrame, output: Path) -> None:
    states = route_metrics.loc[route_metrics["diagnostic"].eq("route_resolution_state")].set_index(
        "bin"
    )
    ordered_states = [state for state in ROUTE_STATES if state in states.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(
        ordered_states,
        [float(states.loc[state, "registered_completion_next_3_rate"]) for state in ordered_states],
        color="#386cb0",
    )
    axes[0].set_ylabel("Registered completion rate: next 3 bars")
    axes[0].set_title("Frozen route-resolution states")
    axes[0].tick_params(axis="x", rotation=30)
    metrics = pooled.set_index("model")
    positions = np.arange(2)
    width = 0.34
    axes[1].bar(
        positions - width / 2,
        [float(metrics.loc["H0", "log_loss"]), float(metrics.loc["H0", "brier_score"])],
        width,
        label="H0",
        color="#7fc97f",
    )
    axes[1].bar(
        positions + width / 2,
        [float(metrics.loc["H1", "log_loss"]), float(metrics.loc["H1", "brier_score"])],
        width,
        label="H1",
        color="#fdc086",
    )
    axes[1].set_xticks(positions, ["Log loss", "Brier"])
    axes[1].set_title("Pooled assessment proper scores")
    axes[1].legend()
    fig.suptitle("Route-Competition Completion-Hazard Quick Screen V0")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(
    decision: Mapping[str, Any],
    support: Mapping[str, Any],
    pooled: pd.DataFrame,
    route_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null_metrics: pd.DataFrame,
    coefficients: Mapping[str, Any],
    determinism: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
) -> str:
    pooled_index = pooled.set_index("model")
    increments = cast(Mapping[str, Any], decision["increments"])
    route_coefficients = {
        feature: coefficient
        for feature, coefficient in zip(
            coefficients["primary_models"]["H1"]["feature_names"],
            coefficients["primary_models"]["H1"]["coefficient"],
            strict=True,
        )
        if feature in ROUTE_FEATURES
    }
    state_rows = route_metrics.loc[route_metrics["diagnostic"].eq("route_resolution_state")]
    count_change_rows = route_metrics.loc[
        route_metrics["diagnostic"].eq("candidate_count_change_3_bars")
    ].set_index("bin")
    lines = [
        "# Route-Competition Completion-Hazard Quick Screen V0",
        "",
        (
            "Retrospective, observable, structural, research-only quick screen. "
            "No economic, directional, execution, broker, or strategy-promotion "
            "outcome was opened."
        ),
        "",
        f"Primary decision: `{decision['primary_decision']}`.",
        "",
        "## Support",
        "",
        (
            f"Assessment rows: {support['assessment_rows']}; sessions: "
            f"{support['assessment_sessions']}; stocks: {support['assessment_stocks']}; "
            f"months: {support['assessment_months']}; positive outcomes: "
            f"{support['positive_outcomes']}."
        ),
        (
            f"Feature retention: {float(support['feature_retention']):.6f}. The "
            "preregistered 30,000-row assessment gate is "
            f"{'met' if int(support['assessment_rows']) >= 30000 else 'not met'}."
        ),
        (
            "Maximum weighted target-class share: "
            f"{float(support['maximum_target_class_share']):.6f}; maximum weighted "
            f"stock share: {float(support['maximum_weighted_stock_share']):.6f}."
        ),
        "",
        "## Pooled models",
        "",
    ]
    for model in ("H0", "H1"):
        row = pooled_index.loc[model]
        lines.append(
            f"- {model}: log loss {float(row['log_loss']):.6f}; Brier "
            f"{float(row['brier_score']):.6f}; AUC {float(row['auc']):.6f}; "
            f"average precision {float(row['average_precision']):.6f}; "
            f"top-decile precision {float(row['top_decile_precision']):.6f}; "
            f"top-quintile precision {float(row['top_quintile_precision']):.6f}."
        )
    lines.extend(
        [
            "",
            "H1-minus-H0 increments (positive means improvement): "
            + ", ".join(f"{key}={float(value):.8f}" for key, value in increments.items())
            + ".",
            "",
            "## Route-resolution states",
            "",
        ]
    )
    for row in state_rows.itertuples(index=False):
        lines.append(
            f"- {row.bin}: rows {int(row.rows)}, three-bar completion rate "
            f"{float(row.registered_completion_next_3_rate):.6f}, "
            f"supported={bool(row.inference_supported)}."
        )
    decision_gates = cast(Mapping[str, Any], decision["decision_gates"])
    lines.append(
        f"Positive log-loss months: {int(decision_gates['positive_months'])}; "
        "materially adverse checkpoints: "
        f"{int(decision_gates['materially_adverse_checkpoints'])}."
    )
    if all(value in count_change_rows.index for value in ("decreasing", "unchanged", "increasing")):
        decreasing_rate = float(
            count_change_rows.loc["decreasing", "registered_completion_next_3_rate"]
        )
        unchanged_rate = float(
            count_change_rows.loc["unchanged", "registered_completion_next_3_rate"]
        )
        increasing_rate = float(
            count_change_rows.loc["increasing", "registered_completion_next_3_rate"]
        )
        lines.append(
            "Candidate-count change alone: decreasing "
            f"{decreasing_rate:.6f}, unchanged {unchanged_rate:.6f}, "
            f"increasing {increasing_rate:.6f}."
        )
    lines.extend(
        [
            "",
            "## Route coefficients",
            "",
            ", ".join(
                f"{feature}={float(value):.6f}" for feature, value in route_coefficients.items()
            ),
            "",
            "## Resampling",
            "",
        ]
    )
    for row in bootstrap.loc[bootstrap["record_type"].eq("interval")].itertuples(index=False):
        lines.append(
            f"- Bootstrap {row.statistic} "
            f"{int(float(row.interval_level) * 100)}%: "
            f"[{float(row.lower):.8f}, {float(row.upper):.8f}]."
        )
    for row in null_metrics.loc[null_metrics["record_type"].eq("comparison")].itertuples(
        index=False
    ):
        lines.append(
            f"- Real {row.statistic} exceeded "
            f"{int(row.real_exceeds_null_count)} of {int(row.null_draws)} "
            "route-bundle null increments."
        )
    if determinism is not None:
        lines.extend(
            [
                "",
                (
                    "Fast determinism passed="
                    f"{bool(determinism['passed'])}; maximum probability difference "
                    f"{float(determinism['maximum_probability_difference']):.3g}; "
                    "row mismatches "
                    f"{int(determinism['row_identity_mismatches'])}; feature hash "
                    f"match={bool(determinism['feature_hash_match'])}."
                ),
            ]
        )
    if audit is not None:
        lines.append(
            "Independent artifact audit passed="
            f"{bool(audit['passed'])}; maximum route-feature difference "
            f"{float(audit['maximum_route_feature_difference']):.3g}; maximum "
            f"bootstrap difference {float(audit['maximum_bootstrap_difference']):.3g}."
        )
    lines.extend(
        [
            "",
            (
                "This is not prospective validation and provides no evidence of "
                "economic value, directional edge, trading utility, or deployability."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def finalize_report(output: Path, *, audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Regenerate both report copies only from frozen, audited artifacts."""

    decision = read_json(output / "decision.json")
    determinism = read_json(output / "determinism_check.json")
    report = build_report(
        decision,
        cast(Mapping[str, Any], decision["support"]),
        pd.read_csv(output / "pooled_metrics.csv"),
        pd.read_csv(output / "route_resolution_state_metrics.csv"),
        pd.read_csv(output / "bootstrap_metrics.csv"),
        pd.read_csv(output / "route_null_metrics.csv"),
        read_json(output / "model_coefficients.json"),
        determinism=determinism,
        audit=audit,
    )
    primary_report = output / "report.md"
    reports_copy = REPORTS_DIR / "report.md"
    primary_report.write_text(report, encoding="utf-8")
    reports_copy.write_text(report, encoding="utf-8")
    primary_hash = sha256_file(primary_report)
    copy_hash = sha256_file(reports_copy)
    if primary_hash != copy_hash:
        raise ScreenBlocker("blocked_reproducibility_or_audit_failure", "report copies differ")
    return {"sha256": primary_hash, "copies_match": True}


def determinism_check(
    output: Path,
    boundaries: Mapping[str, Mapping[str, float]],
    support: Mapping[str, Any],
    blocker: str | None,
    *,
    expected_row_ids: Sequence[str],
    expected_feature_hash: str,
) -> dict[str, Any]:
    panel = pd.read_parquet(output / "decision_panel.parquet")
    coefficients = cast(
        dict[str, Any], json.loads((output / "model_coefficients.json").read_text())
    )
    feature_columns = [
        "row_id",
        *BASELINE_FEATURES,
        *ROUTE_FEATURES,
        "registered_completion_next_3_bars",
        "row_weight",
        "route_resolution_state",
    ]
    reloaded_feature_hash = stable_frame_hash(panel, feature_columns)
    development = panel.loc[panel["period"].eq("development")]
    refit_h0 = fit_hazard_model(development, features=BASELINE_FEATURES)
    refit_h1 = fit_hazard_model(development, features=H1_FEATURES)
    regenerated_h0 = refit_h0.predict_probability(panel)
    regenerated_h1 = refit_h1.predict_probability(panel)
    maximum_probability_difference = float(
        max(
            np.max(np.abs(regenerated_h0 - panel["H0_probability"].to_numpy(float))),
            np.max(np.abs(regenerated_h1 - panel["H1_probability"].to_numpy(float))),
        )
    )
    coefficient_differences = []
    for model, refit in (("H0", refit_h0), ("H1", refit_h1)):
        stored = coefficients["primary_models"][model]
        coefficient_differences.extend(
            np.abs(
                np.asarray(refit.as_dict()["coefficient"], dtype=float)
                - np.asarray(stored["coefficient"], dtype=float)
            ).tolist()
        )
        coefficient_differences.append(
            abs(float(refit.as_dict()["intercept"]) - float(stored["intercept"]))
        )
    thresholds = freeze_route_thresholds(panel.loc[panel["period"].eq("development")])
    regenerated_labels = assign_route_resolution_state(panel, thresholds)
    label_mismatches = int(
        (regenerated_labels.astype(str) != panel["route_resolution_state"].astype(str)).sum()
    )
    replay = panel.copy()
    replay["H0_probability"] = regenerated_h0
    replay["H1_probability"] = regenerated_h1
    replay_pooled, replay_checkpoint, replay_monthly, _ = build_metric_tables(replay, boundaries)
    stored_pooled = pd.read_csv(output / "pooled_metrics.csv")
    numeric_metrics = (
        "log_loss",
        "brier_score",
        "auc",
        "average_precision",
        "top_decile_precision",
        "top_quintile_precision",
    )
    maximum_metric_difference = 0.0
    for model in ("H0", "H1"):
        left = replay_pooled.loc[replay_pooled["model"].eq(model)].iloc[0]
        right = stored_pooled.loc[stored_pooled["model"].eq(model)].iloc[0]
        maximum_metric_difference = max(
            maximum_metric_difference,
            max(abs(float(left[value]) - float(right[value])) for value in numeric_metrics),
        )
    stored_decision = cast(dict[str, Any], json.loads((output / "decision.json").read_text()))
    replay_decision = choose_decision(
        replay,
        replay_pooled,
        replay_checkpoint,
        replay_monthly,
        pd.read_csv(output / "route_resolution_state_metrics.csv"),
        pd.read_csv(output / "bootstrap_metrics.csv"),
        pd.read_csv(output / "route_null_metrics.csv"),
        support,
        blocker,
    )
    loaded_row_ids = panel["row_id"].astype(str).tolist()
    row_mismatches = abs(len(loaded_row_ids) - len(expected_row_ids)) + sum(
        left != right for left, right in zip(loaded_row_ids, expected_row_ids, strict=False)
    )
    assessment_predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    expected_assessment_ids = (
        panel.loc[panel["period"].eq("assessment"), "row_id"].astype(str).tolist()
    )
    assessment_ids = assessment_predictions["row_id"].astype(str).tolist()
    assessment_row_mismatches = abs(len(assessment_ids) - len(expected_assessment_ids)) + sum(
        left != right for left, right in zip(assessment_ids, expected_assessment_ids, strict=False)
    )
    passed = bool(
        maximum_probability_difference <= 1e-12
        and label_mismatches == 0
        and row_mismatches == 0
        and assessment_row_mismatches == 0
        and reloaded_feature_hash == expected_feature_hash
        and max(coefficient_differences, default=0.0) <= 1e-12
        and maximum_metric_difference <= 1e-12
        and replay_decision["primary_decision"] == stored_decision["primary_decision"]
    )
    return {
        **SAFETY_FLAGS,
        "models_refit": ["H0", "H1"],
        "bootstrap_repeated": False,
        "route_null_refits_repeated": False,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_coefficient_difference": max(coefficient_differences, default=0.0),
        "maximum_pooled_metric_difference": maximum_metric_difference,
        "route_resolution_label_mismatches": label_mismatches,
        "row_identity_mismatches": row_mismatches,
        "assessment_row_identity_mismatches": assessment_row_mismatches,
        "feature_hash_match": reloaded_feature_hash == expected_feature_hash,
        "expected_feature_hash": expected_feature_hash,
        "reloaded_feature_hash": reloaded_feature_hash,
        "final_decision_match": replay_decision["primary_decision"]
        == stored_decision["primary_decision"],
        "required_maximum_probability_difference": 1e-12,
        "passed": passed,
    }


def execute_screen(output: Path, *, provider_root: Path) -> dict[str, Any]:
    contract = load_contract()
    output.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    states, dictionary, state_source, dictionary_manifest = load_v2_states(provider_root)
    ledger = build_structural_ledger(states, dictionary)
    predecessor_reconstruction = verify_predecessor_registered_ledger(ledger)
    raw_panel, exclusions, possible_rows = build_raw_decision_panel(states, ledger)
    panel, component_scaling, local_scaling = add_development_frozen_baseline_features(raw_panel)
    panel, frozen_bins = add_frozen_route_labels_and_weights(panel)
    panel = panel.sort_values(
        ["period", "session", "checkpoint", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    reject_protected_dates(panel, column="session")
    panel, primary_specifications, model_configurations = fit_primary_models(panel)
    boundaries = prediction_boundaries(panel)
    pooled, checkpoint_metrics, monthly_metrics, subgroup_metrics = build_metric_tables(
        panel, boundaries
    )
    route_metrics = build_route_resolution_metrics(panel)
    assessment = panel.loc[panel["period"].eq("assessment")]
    real_increments = metric_increments(assessment, boundaries)
    bootstrap = run_bootstrap(assessment, boundaries)
    panel, null_metrics, null_specifications = run_route_nulls(panel, real_increments)
    concentration, support, blocker = concentration_and_support(panel, possible_rows=possible_rows)
    decision = choose_decision(
        panel,
        pooled,
        checkpoint_metrics,
        monthly_metrics,
        route_metrics,
        bootstrap,
        null_metrics,
        support,
        blocker,
    )

    structural_timestamps = pd.to_datetime(
        ledger["available_timestamp_utc"], utc=True, errors="raise"
    )
    protected_boundary = {
        **SAFETY_FLAGS,
        "development_start": DEVELOPMENT_START,
        "development_end_inclusive": "2024-12-31",
        "assessment_start": ASSESSMENT_START,
        "assessment_end_inclusive": READ_END,
        "protected_start": "2025-08-23",
        "minimum_state_timestamp_read": str(states["bar_start_timestamp"].min()),
        "maximum_state_timestamp_read": str(states["bar_start_timestamp"].max()),
        "maximum_decision_session": str(panel["session"].max()),
        "maximum_structural_available_timestamp": str(structural_timestamps.max()),
        "protected_rows_materialised": 0,
        "passed": bool(
            panel["session"].astype(str).lt("2025-08-23").all()
            and states["bar_start_timestamp"].lt(pd.Timestamp("2025-08-23", tz="UTC")).all()
        ),
    }
    if not protected_boundary["passed"]:
        raise ScreenBlocker("blocked_protected_boundary_failure", "protected row materialised")

    checkpoint_data = checkpoint_manifest(panel, exclusions, possible_rows)
    state_trace = causal_state_trace_surface(states)
    state_trace_path = output / "causal_state_trace.parquet"
    write_parquet(state_trace_path, state_trace)
    source_manifest = {
        **SAFETY_FLAGS,
        "dates_read": {
            "start": DEVELOPMENT_START,
            "end_inclusive": READ_END,
        },
        "provider": "EODHD",
        "timeframe": "5m",
        "historical_activity_field": "EODHD historical activity proxy",
        "historical_activity_is_exchange_volume": False,
        "raw_data_downloaded": False,
        "frozen_audited_cohort": list(FROZEN_COHORT),
        "state_source": state_source,
        "causal_state_trace": {
            "logical_path": str(state_trace_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(state_trace_path),
            "rows": len(state_trace),
            "columns": list(state_trace.columns),
            "maximum_bar_ordinal": int(state_trace["bar_ordinal"].max()),
        },
        "loop_dictionary_manifest": dictionary_manifest,
        "predecessor_registered_ledger_reconstruction": predecessor_reconstruction,
        "predecessor_experiments": [
            "research/registered-loop-precursors/20260721-registered-loop-precursor-hidden-veto-v0",
            "research/registered-loop-routes/20260722-hidden-loop-competing-routes-v0",
            "research/hidden-loop-economics/20260721-hidden-loop-economics-registered-bridge-v0",
            "research/unregistered-loop-families/20260721-opening-trajectory-unregistered-families-v0",
            "research/loop-funnel/20260721-emotion-regime-coarse-loop-family-v0",
            "research/observable-behavioural-state/20260721-behavioural-state-dimensions-screen-v0",
        ],
        "minimum_timestamp_read": str(states["bar_start_timestamp"].min()),
        "maximum_timestamp_read": str(states["bar_start_timestamp"].max()),
        "protected_rows_materialised": 0,
    }
    baseline_manifest = {
        **SAFETY_FLAGS,
        "features": list(BASELINE_FEATURES),
        "feature_count": len(BASELINE_FEATURES),
        "behavioural_features": [
            "arousal",
            "conviction",
            "tension",
            "signed_pressure",
        ],
        "regime_features": [
            "posterior_entropy",
            "transition_probability",
            "persistence_probability",
            "expected_state_age",
            "top_state_probability",
            "top_second_margin",
        ],
        "compression_and_local_trigger_features": list(LOCAL_FEATURES),
        "recent_structural_memory_features": [
            "any_registered_completion_prior_6",
            "any_registered_completion_prior_12",
            "same_identity_active_prefix_with_prior_completion",
            "any_hidden_event_prior_6",
            "hidden_2_3_2_prior_6",
            "bars_since_latest_registered_completion",
        ],
        "clock_indicators": [f"checkpoint_{value}" for value in CHECKPOINTS],
        "trailing_six_definition": (
            "six completed bars ending at the checkpoint; current-bar ratios use "
            "that causal trailing-six denominator"
        ),
        "prior_structural_window_definition": "strictly before the checkpoint completed-bar count",
        "behavioural_component_scaling": component_scaling,
        "stock_and_clock_local_scaling": local_scaling,
        "activity_proxy_name": "historical_activity_proxy",
    }
    route_manifest = {
        **SAFETY_FLAGS,
        "features": list(ROUTE_FEATURES),
        "feature_count": len(ROUTE_FEATURES),
        "prefix_identity": "exact frozen registered semantic_loop_id plus orientation_id",
        "broad_family": "frozen registered motif_type: primitive, repeat, or composite",
        "prefix_depth_fraction": (
            "(progress_states - 1) / (progress_states + transitions_remaining - 1)"
        ),
        "orientation_disagreement": (
            "one minus the modal cross-route orientation-anchor-state share; "
            "zero when fewer than two or unavailable"
        ),
        "recent_loop_memory": (
            "exact identity in previous six bars receives +1; exact identity "
            "7-12 bars ago receives +0.5"
        ),
        "future_target_identity_used": False,
        "development_frozen_bins": frozen_bins,
    }
    model_configurations["probability_quantile_boundaries"] = boundaries
    coefficients = {
        **SAFETY_FLAGS,
        "primary_models": primary_specifications,
        "route_null_models": null_specifications,
    }
    determinism_feature_columns = [
        "row_id",
        *BASELINE_FEATURES,
        *ROUTE_FEATURES,
        "registered_completion_next_3_bars",
        "row_weight",
        "route_resolution_state",
    ]
    expected_row_ids = panel["row_id"].astype(str).tolist()
    expected_feature_hash = stable_frame_hash(panel, determinism_feature_columns)

    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected_boundary)
    write_json(output / "checkpoint_manifest.json", checkpoint_data)
    write_parquet(output / "decision_panel.parquet", panel)
    write_json(output / "baseline_feature_manifest.json", baseline_manifest)
    write_json(output / "route_competition_feature_manifest.json", route_manifest)
    write_parquet(output / "route_competition_ledger.parquet", ledger)
    write_json(output / "model_configurations.json", model_configurations)
    write_json(output / "model_coefficients.json", coefficients)
    write_parquet(
        output / "assessment_predictions.parquet",
        panel.loc[panel["period"].eq("assessment")].reset_index(drop=True),
    )
    write_csv(output / "pooled_metrics.csv", pooled)
    write_csv(output / "checkpoint_metrics.csv", checkpoint_metrics)
    write_csv(output / "monthly_metrics.csv", monthly_metrics)
    write_csv(output / "subgroup_metrics.csv", subgroup_metrics)
    write_csv(output / "route_resolution_state_metrics.csv", route_metrics)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "route_null_metrics.csv", null_metrics)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    plot_screen(route_metrics, pooled, output / "route_completion_by_resolution_state.png")
    determinism = determinism_check(
        output,
        boundaries,
        support,
        blocker,
        expected_row_ids=expected_row_ids,
        expected_feature_hash=expected_feature_hash,
    )
    write_json(output / "determinism_check.json", determinism)
    if not determinism["passed"]:
        raise ScreenBlocker("blocked_reproducibility_or_audit_failure", "determinism check failed")
    finalize_report(output)
    return decision


def write_blocker(output: Path, blocker: ScreenBlocker) -> None:
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": blocker.code,
        "blocker": blocker.code,
        "detail": blocker.detail,
    }
    write_json(output / "decision.json", decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=Path.home()
        / "StockerLocal"
        / "data"
        / "processed"
        / "source=eodhd"
        / "instrument_type=stock",
    )
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
        decision = execute_screen(
            output, provider_root=arguments.provider_root.expanduser().resolve()
        )
        print(canonical_json(decision), end="")
        return 0
    except ScreenBlocker as blocker:
        write_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
