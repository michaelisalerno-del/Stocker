#!/usr/bin/env python3
"""Run the Hidden-Loop Competing Routes quick screen V0."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-hidden-competing-routes-mpl")

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

from stocker_research.hidden_loop_competing_routes_v0 import (
    FROZEN_HIDDEN_FAMILIES,
    HIDDEN_A,
    HIDDEN_B,
    NO_REGISTERED_COMPLETION,
    OTHER_REGISTERED_COMPLETION,
    PREREGISTERED_TARGETS,
    PROTECTED_START,
    SAFETY_FLAGS,
    TARGET_A,
    TARGET_B,
    TARGET_C,
    FittedMultinomial,
    assign_frozen_bin,
    benjamini_hochberg,
    candidate_normalised_weights,
    candidate_threshold,
    choose_primary_decision,
    counterfactual_probability_difference,
    deduplicate_registered_completions,
    fit_multinomial,
    freeze_target_class_mapping,
    frozen_quantile_boundaries,
    hidden_history_features,
    lookback_is_complete,
    matched_control_relations,
    model_feature_sets,
    model_from_json,
    multiclass_metrics,
    next_registered_route,
    permute_hidden_bundle,
    precursor_present,
    predict_multinomial,
    registered_history_features,
    reject_protected_dates,
    sample_matched_pseudo_completions,
    sequential_update_ordinals,
    session_bootstrap_multiplicities,
    stable_frame_hash,
    target_prefix_snapshot,
    transition_hypothesis_manifest,
)
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine
from stocker_research.opening_trajectory_unregistered_families_v0 import (
    canonical_unregistered_path,
    pool_hidden_family,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "registered-loop-precursors"
    / "20260721-registered-loop-precursor-hidden-veto-v0"
)
PREDECESSOR_RUNNER = PREDECESSOR_DIR / "run_screen_v0.py"
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
BRIDGE_PRIMARY = (
    REPO_ROOT
    / "research"
    / "hidden-loop-economics"
    / "20260721-hidden-loop-economics-registered-bridge-v0"
    / "artifacts"
    / "primary"
)
OPENING_PRIMARY = (
    REPO_ROOT
    / "research"
    / "unregistered-loop-families"
    / "20260721-opening-trajectory-unregistered-families-v0"
    / "artifacts"
    / "primary"
)

FROZEN_B0_THRESHOLD = 0.22125611521102
TRANSITION_NULL_DRAWS = 25
BOOTSTRAP_DRAWS = 25
HIDDEN_NULL_REFITS = 5
TRANSITION_SEED = 20260722
BOOTSTRAP_SEED = 20260723
HIDDEN_NULL_SEED = 20260724
MODEL_NAMES = ("C0", "C1", "C2")
TARGET_ALIAS = {TARGET_A: "target_a", TARGET_B: "target_b", TARGET_C: "target_c"}


class ScreenBlocker(RuntimeError):
    """Fail-closed blocker carrying one preregistered decision code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
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


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocker("blocked_reproducibility_or_audit_failure", f"cannot load {path}")
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
    limits = contract["hard_limits"]
    required = {
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
        "opening_checkpoints": [6, 12],
        "maximum_elapsed_bars": 6,
        "maximum_primary_model_fits": 3,
        "transition_null_draws": 25,
        "session_bootstrap_draws": 25,
        "hidden_history_null_refits": 5,
        "maximum_plots": 2,
    }
    if any(limits.get(key) != value for key, value in required.items()):
        raise ScreenBlocker(
            "blocked_quick_competing_route_resource_limit", "hard speed limits differ"
        )
    return contract


def load_frozen_inputs(
    provider_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
    dict[str, Any],
]:
    """Reconstruct the predecessor population and bounded state surface once."""

    predecessor = load_module(PREDECESSOR_RUNNER, "hidden_competing_routes_predecessor")
    (
        opening,
        development_archived,
        assessment_archived,
        frozen_completions,
        opening_reconstruction,
        b0_features,
    ) = predecessor.reconstruct_opening_population()
    states, dictionary, state_source, dictionary_manifest = predecessor.load_v2_states(
        provider_root
    )
    opening = cast(pd.DataFrame, opening)
    development_archived = cast(pd.DataFrame, development_archived)
    assessment_archived = cast(pd.DataFrame, assessment_archived)
    states = cast(pd.DataFrame, states)
    frozen_completions = cast(pd.DataFrame, frozen_completions)
    for frame in (opening, development_archived, assessment_archived, states, frozen_completions):
        reject_protected_dates(frame, column="session")
    state_probabilities = [f"state_p_{value}" for value in range(8)]
    ordered = np.sort(states.loc[:, state_probabilities].to_numpy(float), axis=1)
    states["top_state_probability"] = ordered[:, -1]
    states["top_second_margin"] = ordered[:, -1] - ordered[:, -2]
    states["posterior_entropy"] = states["posterior_entropy_reproduced"].astype(float)
    return (
        opening,
        development_archived,
        assessment_archived,
        frozen_completions,
        states,
        dictionary,
        cast(dict[str, Any], state_source),
        cast(dict[str, Any], dictionary_manifest),
        tuple(str(value) for value in b0_features),
        cast(dict[str, Any], opening_reconstruction),
    )


def build_trace_ledgers(
    states: pd.DataFrame, dictionary: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate registered, hidden, and active-prefix ledgers from frozen identities."""

    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    registered_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    for (symbol, session), group in states.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        hard = ordered["causal_hard_state"].to_numpy(dtype=int)
        changes = np.concatenate(([True], hard[1:] != hard[:-1]))
        event_rows = ordered.loc[changes]
        trace = engine.scan_state_events(
            event_rows["causal_hard_state"].astype(int).tolist(),
            bar_ordinals=event_rows["bar_ordinal"].astype(int).tolist(),
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
            registered_rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "completion_bar_ordinal": int(event.completion_bar_ordinal),
                    "semantic_loop_id": str(event.semantic_loop_id),
                    "orientation_id": str(event.orientation_id),
                }
            )
        for event in trace.unregistered_completions:
            canonical = canonical_unregistered_path(event.full_path)
            available_timestamp = trace.state_events[
                event.completion_event_index
            ].available_timestamp
            hidden_rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "completion_bar_ordinal": int(event.completion_bar_ordinal),
                    "completion_timestamp_utc": pd.Timestamp(event.completion_timestamp),
                    "completion_available_timestamp_utc": pd.Timestamp(available_timestamp),
                    "family_id": str(canonical.family_id),
                    "hidden_family_class": pool_hidden_family(
                        canonical.family_id, FROZEN_HIDDEN_FAMILIES[:-1]
                    ),
                    "orientation_id": str(canonical.orientation_id),
                }
            )
        event_index = np.cumsum(changes).astype(int) - 1
        for position, (_, bar) in enumerate(ordered.iterrows()):
            for prefix in trace.prefixes_after_event[int(event_index[position])]:
                prefix_rows.append(
                    {
                        "symbol": str(symbol),
                        "session": str(session),
                        "bar_ordinal": int(bar["bar_ordinal"]),
                        "semantic_loop_id": str(prefix.semantic_loop_id),
                        "orientation_id": str(prefix.orientation_id),
                        "progress_states": int(prefix.progress_states),
                    }
                )
    registered = pd.DataFrame(registered_rows).drop_duplicates(
        ["symbol", "session", "completion_bar_ordinal", "semantic_loop_id", "orientation_id"]
    )
    hidden = pd.DataFrame(hidden_rows).drop_duplicates(
        ["symbol", "session", "completion_bar_ordinal", "family_id", "orientation_id"]
    )
    prefixes = pd.DataFrame(prefix_rows).drop_duplicates(
        [
            "symbol",
            "session",
            "bar_ordinal",
            "semantic_loop_id",
            "orientation_id",
            "progress_states",
        ]
    )
    return (
        registered.reset_index(drop=True),
        hidden.reset_index(drop=True),
        prefixes.reset_index(drop=True),
    )


def verify_trace_identity(trace: pd.DataFrame, registered: pd.DataFrame) -> dict[str, Any]:
    identity = [
        "symbol",
        "session",
        "completion_bar_ordinal",
        "semantic_loop_id",
        "orientation_id",
    ]
    left = trace.loc[:, identity].drop_duplicates()
    right = registered.loc[:, identity].drop_duplicates()
    left_hash = stable_frame_hash(left, identity)
    right_hash = stable_frame_hash(right, identity)
    passed = len(left) == len(right) and left_hash == right_hash
    if not passed:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure",
            f"registered trace differs from frozen ledger: {len(left)}/{len(right)}",
        )
    return {
        "passed": passed,
        "trace_rows": int(len(left)),
        "frozen_rows": int(len(right)),
        "trace_identity_sha256": left_hash,
        "frozen_identity_sha256": right_hash,
    }


def _group_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (str(symbol), str(session)): group.reset_index(drop=True)
        for (symbol, session), group in frame.groupby(["symbol", "session"], sort=False)
    }


def build_null_eligibility(
    states: pd.DataFrame, registered: pd.DataFrame, *, lookback: int
) -> pd.DataFrame:
    at_timestamp = {
        (str(symbol), str(session), int(ordinal)): tuple(
            sorted(group["semantic_loop_id"].astype(str).unique().tolist())
        )
        for (symbol, session, ordinal), group in registered.groupby(
            ["symbol", "session", "completion_bar_ordinal"], sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    for (symbol, session), group in states.groupby(["symbol", "session"], sort=True):
        ordinals = group["bar_ordinal"].astype(int).tolist()
        for row in group.itertuples(index=False):
            ordinal = int(row.bar_ordinal)
            timestamp = pd.Timestamp(row.bar_start_timestamp)
            rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "year_month": str(session)[:7],
                    "clock_bin": timestamp.tz_convert("America/New_York")
                    .floor("30min")
                    .strftime("%H:%M"),
                    "completion_bar_ordinal": ordinal,
                    "completion_timestamp_utc": timestamp,
                    "full_prior_history": lookback_is_complete(ordinals, ordinal, lookback),
                    "semantic_loop_ids_at_timestamp": at_timestamp.get(
                        (str(symbol), str(session), ordinal), ()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _hypothesis_targets(events: pd.DataFrame, hypothesis: Mapping[str, Any]) -> pd.DataFrame:
    if str(hypothesis["target_identity"]) == "ANY_REGISTERED_COMPLETION":
        return events
    return events.loc[events["semantic_loop_id"].astype(str).eq(str(hypothesis["target_identity"]))]


def build_transition_screen(
    registered: pd.DataFrame,
    hidden: pd.DataFrame,
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the corrected six/12-bar census and exactly 25 matched null draws."""

    hypotheses = transition_hypothesis_manifest()
    registered_groups = _group_lookup(registered)
    hidden_groups = _group_lookup(hidden)
    empty_registered = pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id"])
    empty_hidden = pd.DataFrame(columns=["completion_bar_ordinal", "hidden_family_class"])
    census_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    support: dict[str, Any] = {}
    summary_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for lookback in (6, 12):
        eligibility = build_null_eligibility(states, registered, lookback=lookback)
        eligible_pool = {
            (str(symbol), str(session)): set(
                group.loc[
                    group["full_prior_history"].astype(bool), "completion_bar_ordinal"
                ].astype(int)
            )
            for (symbol, session), group in eligibility.groupby(["symbol", "session"], sort=False)
        }
        period_events: dict[str, pd.DataFrame] = {}
        for period, year in (("development", 2024), ("assessment", 2025)):
            all_period = registered.loc[registered["year"].eq(year)].copy()
            eligible_mask = all_period.apply(
                lambda row, eligible_pool=eligible_pool: (
                    int(row["completion_bar_ordinal"])
                    in eligible_pool.get((str(row["symbol"]), str(row["session"])), set())
                ),
                axis=1,
            )
            eligible_events = all_period.loc[eligible_mask].copy()
            period_events[period] = eligible_events
            support[f"{period}_{lookback}"] = {
                "eligible_events": int(len(eligible_events)),
                "ineligible_events": int(len(all_period) - len(eligible_events)),
                "sessions": int(eligible_events["session"].nunique()),
                "stocks": int(eligible_events["symbol"].nunique()),
                "months": int(eligible_events["year_month"].nunique()),
            }
            for hypothesis in hypotheses:
                if lookback not in cast(list[int], hypothesis["lookbacks"]):
                    continue
                targets_all = _hypothesis_targets(all_period, hypothesis)
                targets = _hypothesis_targets(eligible_events, hypothesis)
                labels: list[bool] = []
                for target in targets.itertuples(index=False):
                    key = (str(target.symbol), str(target.session))
                    label = precursor_present(
                        completion_bar_ordinal=int(target.completion_bar_ordinal),
                        lookback_bars=lookback,
                        precursor_kind=str(hypothesis["precursor_kind"]),
                        precursor_identity=str(hypothesis["precursor_identity"]),
                        registered_events=registered_groups.get(key, empty_registered),
                        hidden_events=hidden_groups.get(key, empty_hidden),
                    )
                    labels.append(label)
                    event_rows.append(
                        {
                            "record_type": "observed",
                            "draw": -1,
                            "period": period,
                            "hypothesis_id": str(hypothesis["hypothesis_id"]),
                            "lookback_bars": lookback,
                            "source_event_id": str(target.event_id),
                            "symbol": str(target.symbol),
                            "session": str(target.session),
                            "completion_bar_ordinal": int(target.completion_bar_ordinal),
                            "semantic_loop_id": str(target.semantic_loop_id),
                            "source_clock_bin": str(target.clock_bin),
                            "clock_bin": str(target.clock_bin),
                            "precursor_kind": str(hypothesis["precursor_kind"]),
                            "precursor_identity": str(hypothesis["precursor_identity"]),
                            "precursor_present": label,
                        }
                    )
                positive = (
                    targets.loc[np.asarray(labels, dtype=bool)] if labels else targets.iloc[:0]
                )
                stock_share = positive["symbol"].value_counts(normalize=True)
                month_share = positive["year_month"].value_counts(normalize=True)
                row = {
                    "period": period,
                    "hypothesis_id": str(hypothesis["hypothesis_id"]),
                    "lookback_bars": lookback,
                    "precursor_kind": str(hypothesis["precursor_kind"]),
                    "precursor_identity": str(hypothesis["precursor_identity"]),
                    "target_identity": str(hypothesis["target_identity"]),
                    "expected_sign": str(hypothesis["expected_sign"]),
                    "eligible_events": int(len(targets)),
                    "ineligible_events": int(len(targets_all) - len(targets)),
                    "precursor_events": int(len(positive)),
                    "observed_prevalence": float(np.mean(labels)) if labels else math.nan,
                    "sessions": int(positive["session"].nunique()),
                    "stocks": int(positive["symbol"].nunique()),
                    "months": int(positive["year_month"].nunique()),
                    "maximum_stock_share": float(stock_share.max())
                    if not stock_share.empty
                    else 1.0,
                    "maximum_month_share": float(month_share.max())
                    if not month_share.empty
                    else 1.0,
                }
                census_rows.append(row)
                summary_lookup[(period, str(hypothesis["hypothesis_id"]), lookback)] = row
        coverage_by_period: dict[str, list[float]] = {"development": [], "assessment": []}
        for draw in range(TRANSITION_NULL_DRAWS):
            for period in ("development", "assessment"):
                targets = period_events[period]
                period_eligibility = eligibility.loc[
                    eligibility["session"]
                    .astype(str)
                    .str[:4]
                    .eq("2024" if period == "development" else "2025")
                ]
                sampled = sample_matched_pseudo_completions(
                    targets, period_eligibility, seed=TRANSITION_SEED + lookback * 100 + draw
                )
                coverage = float(len(sampled) / len(targets)) if len(targets) else 0.0
                coverage_by_period[period].append(coverage)
                source_lookup = targets.set_index("event_id")
                for hypothesis in hypotheses:
                    if lookback not in cast(list[int], hypothesis["lookbacks"]):
                        continue
                    if str(hypothesis["target_identity"]) == "ANY_REGISTERED_COMPLETION":
                        hypothesis_sample = sampled
                    else:
                        source_ids = set(
                            targets.loc[
                                targets["semantic_loop_id"]
                                .astype(str)
                                .eq(str(hypothesis["target_identity"])),
                                "event_id",
                            ].astype(str)
                        )
                        hypothesis_sample = sampled.loc[
                            sampled["source_event_id"].astype(str).isin(source_ids)
                        ]
                    labels = []
                    for pseudo in hypothesis_sample.itertuples(index=False):
                        key = (str(pseudo.symbol), str(pseudo.session))
                        label = precursor_present(
                            completion_bar_ordinal=int(pseudo.completion_bar_ordinal),
                            lookback_bars=lookback,
                            precursor_kind=str(hypothesis["precursor_kind"]),
                            precursor_identity=str(hypothesis["precursor_identity"]),
                            registered_events=registered_groups.get(key, empty_registered),
                            hidden_events=hidden_groups.get(key, empty_hidden),
                        )
                        labels.append(label)
                        source = source_lookup.loc[str(pseudo.source_event_id)]
                        event_rows.append(
                            {
                                "record_type": "null",
                                "draw": draw,
                                "period": period,
                                "hypothesis_id": str(hypothesis["hypothesis_id"]),
                                "lookback_bars": lookback,
                                "source_event_id": str(pseudo.source_event_id),
                                "symbol": str(pseudo.symbol),
                                "session": str(pseudo.session),
                                "completion_bar_ordinal": int(pseudo.completion_bar_ordinal),
                                "semantic_loop_id": str(source["semantic_loop_id"]),
                                "source_clock_bin": str(source["clock_bin"]),
                                "clock_bin": str(pseudo.clock_bin),
                                "precursor_kind": str(hypothesis["precursor_kind"]),
                                "precursor_identity": str(hypothesis["precursor_identity"]),
                                "precursor_present": label,
                            }
                        )
                    observed = summary_lookup[(period, str(hypothesis["hypothesis_id"]), lookback)][
                        "observed_prevalence"
                    ]
                    null_prevalence = float(np.mean(labels)) if labels else math.nan
                    null_rows.append(
                        {
                            "record_type": "draw",
                            "period": period,
                            "hypothesis_id": str(hypothesis["hypothesis_id"]),
                            "lookback_bars": lookback,
                            "draw": draw,
                            "observed_prevalence": observed,
                            "null_prevalence": null_prevalence,
                            "enrichment": float(observed - null_prevalence),
                            "matched_events": int(len(hypothesis_sample)),
                        }
                    )
        for period in ("development", "assessment"):
            support[f"{period}_{lookback}"]["mean_matched_null_coverage"] = float(
                np.mean(coverage_by_period[period])
            )
            support[f"{period}_{lookback}"]["minimum_matched_null_coverage"] = float(
                np.min(coverage_by_period[period])
            )
    census = pd.DataFrame(census_rows)
    null_draws = pd.DataFrame(null_rows)
    summary_rows: list[dict[str, Any]] = []
    for key, group in null_draws.groupby(["period", "hypothesis_id", "lookback_bars"], sort=True):
        period, hypothesis_id, lookback = key
        observed = float(group["observed_prevalence"].iloc[0])
        values = group["null_prevalence"].to_numpy(float)
        summary_rows.append(
            {
                "record_type": "summary",
                "period": period,
                "hypothesis_id": hypothesis_id,
                "lookback_bars": int(lookback),
                "draw": -1,
                "observed_prevalence": observed,
                "mean_null_prevalence": float(np.mean(values)),
                "enrichment": float(observed - np.mean(values)),
                "null_percentile": float(100.0 * np.mean(values <= observed)),
                "null_10th_percentile": float(np.quantile(values, 0.10, method="linear")),
                "null_90th_percentile": float(np.quantile(values, 0.90, method="linear")),
                "matched_events": int(group["matched_events"].min()),
            }
        )
    null_summary = pd.DataFrame(summary_rows)
    null_metrics = pd.concat([null_draws, null_summary], ignore_index=True, sort=False)
    multiplicity_rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis["hypothesis_id"])
        assessment = null_summary.loc[
            null_summary["period"].eq("assessment")
            & null_summary["hypothesis_id"].eq(hypothesis_id)
        ]
        p_values: list[float] = []
        for row in assessment.itertuples(index=False):
            draws = null_draws.loc[
                null_draws["period"].eq("assessment")
                & null_draws["hypothesis_id"].eq(hypothesis_id)
                & null_draws["lookback_bars"].eq(int(row.lookback_bars)),
                "null_prevalence",
            ].to_numpy(float)
            if str(hypothesis["expected_sign"]) == "positive":
                p_value = float((1 + np.sum(draws >= float(row.observed_prevalence))) / 26)
            else:
                p_value = float((1 + np.sum(draws <= float(row.observed_prevalence))) / 26)
            p_values.append(p_value)
        multiplicity_rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "expected_sign": str(hypothesis["expected_sign"]),
                "lookbacks": json.dumps(hypothesis["lookbacks"]),
                "p_value": max(p_values),
                "joint_rule": "maximum one-sided p across required lookbacks",
            }
        )
    multiplicity = pd.DataFrame(multiplicity_rows)
    multiplicity["q_value"] = benjamini_hochberg(multiplicity["p_value"].tolist())
    multiplicity["q_le_0_10"] = multiplicity["q_value"].le(0.10)
    return census, null_metrics, multiplicity, pd.DataFrame(event_rows), support


def reconstruct_candidates(
    opening: pd.DataFrame,
    development_archived: pd.DataFrame,
    assessment_archived: pd.DataFrame,
    b0_features: tuple[str, ...],
    registered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Independently refit predecessor B0 folds and recover its exact high slate."""

    predecessor = load_module(PREDECESSOR_RUNNER, "hidden_competing_routes_predecessor_b0")
    valid, fold_manifest, threshold = predecessor.build_b0_crossfit(opening, b0_features)
    valid = cast(pd.DataFrame, valid)
    fold_manifest = cast(pd.DataFrame, fold_manifest)
    threshold = float(threshold)
    key = ["symbol", "session", "decision_ordinal"]
    development = development_archived.merge(
        valid.loc[:, [*key, "B0_oof_probability"]],
        on=key,
        how="inner",
        validate="one_to_one",
    )
    development = development.loc[
        development["B0_oof_probability"].astype(float).ge(threshold)
    ].copy()
    development["candidate_B0_probability"] = development["B0_oof_probability"].astype(float)
    development["period"] = "development"
    assessment = assessment_archived.loc[
        assessment_archived["B0_probability"].astype(float).ge(threshold)
    ].copy()
    assessment["candidate_B0_probability"] = assessment["B0_probability"].astype(float)
    assessment["period"] = "assessment"
    candidates = pd.concat([development, assessment], ignore_index=True, sort=False)
    candidates["candidate_id"] = (
        candidates["symbol"].astype(str)
        + "|"
        + candidates["session"].astype(str)
        + "|"
        + candidates["decision_ordinal"].astype(int).astype(str).str.zfill(2)
    )
    candidates["candidate_total_weight"] = candidates["row_weight"].astype(float)
    registered_groups = _group_lookup(registered)
    original_classes: list[str] = []
    registered_flags: list[int] = []
    first_identities: list[str | None] = []
    for row in candidates.itertuples(index=False):
        group = registered_groups.get(
            (str(row.symbol), str(row.session)),
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "orientation_id"]),
        )
        target_class, identity, _ = next_registered_route(
            group,
            update_ordinal=int(row.repo_bar_start_ordinal),
            horizon_end_ordinal=int(row.repo_bar_start_ordinal) + 12,
            retained_targets=PREREGISTERED_TARGETS,
        )
        original_classes.append(target_class)
        registered_flags.append(int(identity is not None))
        first_identities.append(identity)
    candidates["original_target_class_all_exact"] = original_classes
    candidates["original_registered_completion"] = registered_flags
    candidates["original_first_registered_semantic_loop_id"] = first_identities
    recalculated = candidate_threshold(valid["B0_oof_probability"].dropna())
    assessment_count = int(len(assessment))
    reconstruction = {
        **SAFETY_FLAGS,
        "frozen_threshold": FROZEN_B0_THRESHOLD,
        "recalculated_threshold": recalculated,
        "predecessor_reported_threshold": threshold,
        "maximum_threshold_difference": max(
            abs(recalculated - FROZEN_B0_THRESHOLD), abs(threshold - FROZEN_B0_THRESHOLD)
        ),
        "threshold_verified_to_1e_12": max(
            abs(recalculated - FROZEN_B0_THRESHOLD), abs(threshold - FROZEN_B0_THRESHOLD)
        )
        <= 1e-12,
        "development_oof_rows": int(len(valid)),
        "development_candidate_rows_before_archived_intersection": int(
            valid["B0_oof_probability"].ge(threshold).sum()
        ),
        "development_candidate_rows": int(len(development)),
        "assessment_candidate_rows": assessment_count,
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_registered_candidates": int(
            candidates.loc[
                candidates["period"].eq("assessment"), "original_registered_completion"
            ].sum()
        ),
        "candidate_membership_sha256": stable_frame_hash(candidates, ["candidate_id", "period"]),
        "fold_manifest": fold_manifest.to_dict(orient="records"),
        "assessment_outcomes_inspected_before_threshold_freeze": False,
        "expected_assessment_population_match": assessment_count == 960
        and assessment["session"].nunique() == 154
        and assessment["symbol"].nunique() == 20,
    }
    if (
        not reconstruction["threshold_verified_to_1e_12"]
        or not reconstruction["expected_assessment_population_match"]
    ):
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            f"threshold/population differs: {recalculated}/{assessment_count}",
        )
    export_columns = [
        "period",
        "candidate_id",
        "symbol",
        "session",
        "year_month",
        "decision_ordinal",
        "repo_bar_start_ordinal",
        "feature_available_timestamp_utc",
        "candidate_B0_probability",
        "candidate_total_weight",
        "original_target_class_all_exact",
        "original_registered_completion",
        "original_first_registered_semantic_loop_id",
    ]
    oof_export = valid.loc[
        :,
        ["symbol", "session", "decision_ordinal", "slate_id", "row_weight", "B0_oof_probability"],
    ].copy()
    return candidates, candidates.loc[:, export_columns].copy(), reconstruction, oof_export


def _definition_metadata(dictionary: Any, retained_targets: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for target in retained_targets:
        if target not in dictionary.definitions:
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure", f"target missing from dictionary: {target}"
            )
        definition = dictionary.definitions[target]
        metadata[target] = {
            "canonical_orientation": list(definition.canonical_orientation),
            "canonical_orientation_id": definition.orientation_id_for(
                definition.canonical_orientation
            ),
            "full_transition_length": int(definition.full_transition_length),
            "motif_type": str(definition.motif_type.value),
            "repeat_depth": int(definition.repeat_depth),
        }
    return metadata


def build_sequential_panel(
    candidates: pd.DataFrame,
    states: pd.DataFrame,
    registered: pd.DataFrame,
    hidden: pd.DataFrame,
    prefixes: pd.DataFrame,
    *,
    retained_targets: Sequence[str],
    dictionary: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct causal elapsed-0..6 rows within each original opening horizon."""

    state_groups = _group_lookup(states)
    registered_groups = _group_lookup(registered)
    hidden_groups = _group_lookup(hidden)
    prefix_groups = _group_lookup(prefixes)
    metadata = _definition_metadata(dictionary, retained_targets)
    empty_registered = pd.DataFrame(
        columns=["completion_bar_ordinal", "semantic_loop_id", "orientation_id"]
    )
    empty_hidden = pd.DataFrame(columns=["completion_bar_ordinal", "hidden_family_class"])
    empty_prefix = pd.DataFrame(
        columns=["bar_ordinal", "semantic_loop_id", "orientation_id", "progress_states"]
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates.sort_values(
        ["period", "session", "decision_ordinal", "symbol"], kind="mergesort"
    ).itertuples(index=False):
        key = (str(candidate.symbol), str(candidate.session))
        state_group = state_groups.get(key)
        if state_group is None:
            raise ScreenBlocker(
                "blocked_predecessor_population_not_reconstructable",
                f"candidate state group missing: {key}",
            )
        state_group = state_group.sort_values("bar_ordinal", kind="mergesort")
        if state_group["bar_ordinal"].duplicated().any():
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure", f"duplicate state bars: {key}"
            )
        state_index = state_group.set_index(state_group["bar_ordinal"].astype(int), drop=False)
        registered_group = registered_groups.get(key, empty_registered)
        hidden_group = hidden_groups.get(key, empty_hidden)
        prefix_group = prefix_groups.get(key, empty_prefix)
        opening_ordinal = int(candidate.repo_bar_start_ordinal)
        horizon_end = opening_ordinal + 12
        future = registered_group.loc[
            registered_group["completion_bar_ordinal"].astype(int).gt(opening_ordinal)
            & registered_group["completion_bar_ordinal"].astype(int).le(horizon_end)
        ]
        first_completion = int(future["completion_bar_ordinal"].min()) if not future.empty else None
        update_ordinals = sequential_update_ordinals(
            opening_ordinal=opening_ordinal,
            first_completion_ordinal=first_completion,
            available_ordinals=state_group["bar_ordinal"].astype(int).tolist(),
        )
        for current_ordinal in update_ordinals:
            current = state_index.loc[current_ordinal]
            elapsed = current_ordinal - opening_ordinal
            target_class, target_identity, target_ordinal = next_registered_route(
                registered_group,
                update_ordinal=current_ordinal,
                horizon_end_ordinal=horizon_end,
                retained_targets=retained_targets,
            )
            probabilities = np.sort(
                np.asarray([float(current[f"state_p_{value}"]) for value in range(8)])
            )
            path = state_group.loc[
                state_group["bar_ordinal"].astype(int).between(opening_ordinal, current_ordinal),
                "causal_hard_state",
            ].astype(int)
            transitions = int((path.to_numpy()[1:] != path.to_numpy()[:-1]).sum())
            row: dict[str, Any] = {
                "period": str(candidate.period),
                "candidate_id": str(candidate.candidate_id),
                "symbol": str(candidate.symbol),
                "session": str(candidate.session),
                "year_month": str(candidate.session)[:7],
                "opening_checkpoint": int(candidate.decision_ordinal),
                "opening_bar_ordinal": opening_ordinal,
                "elapsed_bar": elapsed,
                "update_bar_ordinal": current_ordinal,
                "opening_update_timestamp_utc": pd.Timestamp(
                    candidate.feature_available_timestamp_utc
                ),
                "update_timestamp_utc": pd.Timestamp(current["bar_complete_timestamp"]),
                "original_horizon_end_ordinal": horizon_end,
                "first_registered_completion_bar_ordinal": first_completion,
                "next_completion_bar_ordinal": target_ordinal,
                "next_registered_semantic_loop_id": target_identity,
                "next_registered_route": target_class,
                "candidate_B0_probability": float(candidate.candidate_B0_probability),
                "candidate_total_weight": float(candidate.candidate_total_weight),
                "checkpoint_12": float(int(candidate.decision_ordinal) == 12),
                "opening_transition_probability": float(candidate.transition_probability),
                "opening_posterior_entropy": float(candidate.posterior_entropy),
                "opening_top_state_probability": float(candidate.top_state_probability),
                "opening_state_margin": float(candidate.top_second_margin),
                "current_posterior_entropy": float(current["posterior_entropy"]),
                "current_top_state_probability": float(probabilities[-1]),
                "current_top_second_margin": float(probabilities[-1] - probabilities[-2]),
                "current_expected_state_age": float(current["expected_state_age"]),
                "current_persistence_probability": float(current["persistence_probability"]),
                "current_transition_probability": float(current["transition_probability"]),
                "regime_transitions_since_opening": float(transitions),
            }
            for value in range(1, 7):
                row[f"elapsed_bar_{value}"] = float(elapsed == value)
            for value in range(8):
                row[f"current_state_p_{value}"] = float(current[f"state_p_{value}"])
            maximum_depth = 0.0
            current_prefix = prefix_group.loc[
                pd.to_numeric(prefix_group["bar_ordinal"], errors="raise")
                .astype(int)
                .eq(current_ordinal)
            ]
            for target in retained_targets:
                alias = TARGET_ALIAS[target]
                target_metadata = metadata[target]
                snapshot = target_prefix_snapshot(
                    prefix_group,
                    current_ordinal=current_ordinal,
                    target_identity=target,
                    canonical_orientation_id=str(target_metadata["canonical_orientation_id"]),
                    transition_length=int(target_metadata["full_transition_length"]),
                )
                for name, value in snapshot.items():
                    row[
                        f"{alias}_{'prefix_' if not name.startswith('conflicting') else ''}{name}"
                    ] = value
                maximum_depth = max(maximum_depth, snapshot["depth"])
            row.update(
                registered_history_features(
                    registered_group,
                    opening_ordinal=opening_ordinal,
                    current_ordinal=current_ordinal,
                )
            )
            row.update(
                hidden_history_features(
                    hidden_group,
                    opening_ordinal=opening_ordinal,
                    current_ordinal=current_ordinal,
                )
            )
            hidden_now = hidden_group.loc[
                hidden_group["completion_bar_ordinal"].astype(int).eq(current_ordinal)
                & hidden_group["completion_bar_ordinal"].astype(int).gt(opening_ordinal)
            ]
            registered_now = registered_group.loc[
                registered_group["completion_bar_ordinal"].astype(int).eq(current_ordinal)
                & registered_group["completion_bar_ordinal"].astype(int).gt(opening_ordinal)
            ]
            row["new_hidden_5_6_5_completion"] = float(
                hidden_now["hidden_family_class"].astype(str).eq(HIDDEN_A).any()
            )
            row["new_hidden_2_3_2_completion"] = float(
                hidden_now["hidden_family_class"].astype(str).eq(HIDDEN_B).any()
            )
            row["new_registered_4_6_4_completion"] = float(
                registered_now["semantic_loop_id"].astype(str).eq(TARGET_C).any()
            )
            row["current_prefix_any"] = float(not current_prefix.empty)
            row["current_max_target_prefix_depth"] = maximum_depth
            matching_alias = TARGET_ALIAS.get(str(target_class))
            matching = (
                float(row[f"{matching_alias}_prefix_active"]) if matching_alias is not None else 0.0
            )
            row["matching_target_prefix_active"] = matching
            row["conflicting_prefix_active_breakdown"] = float(
                not current_prefix.empty and not bool(matching)
            )
            row["prefix_state_group"] = (
                "NO_ACTIVE_PREFIX"
                if current_prefix.empty
                else "MATCHING_TARGET_PREFIX_ACTIVE"
                if bool(matching)
                else "CONFLICTING_PREFIX_ACTIVE"
            )
            row["original_registered_candidate"] = float(
                bool(candidate.original_registered_completion)
            )
            row["sequential_row_id"] = f"{candidate.candidate_id}|E{elapsed}"
            rows.append(row)
    panel = pd.DataFrame(rows).sort_values(
        ["period", "session", "symbol", "update_timestamp_utc", "elapsed_bar"],
        kind="mergesort",
    )
    before_deduplication = len(panel)
    panel = panel.sort_values(
        ["period", "session", "symbol", "update_timestamp_utc", "elapsed_bar"],
        kind="mergesort",
    ).drop_duplicates(["period", "symbol", "session", "update_timestamp_utc"], keep="first")
    overlap_rows_removed = before_deduplication - len(panel)
    panel["sequential_row_weight"] = candidate_normalised_weights(panel)
    if (
        panel["sequential_row_id"].duplicated().any()
        or panel.duplicated(["period", "symbol", "session", "update_timestamp_utc"]).any()
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "sequential update rows are duplicated"
        )
    if bool(
        panel["update_bar_ordinal"]
        .astype(int)
        .gt(panel["first_registered_completion_bar_ordinal"].fillna(math.inf))
        .any()
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "update exists after first completion"
        )
    if bool(
        panel["next_completion_bar_ordinal"]
        .dropna()
        .astype(int)
        .gt(
            panel.loc[
                panel["next_completion_bar_ordinal"].notna(), "original_horizon_end_ordinal"
            ].astype(int)
        )
        .any()
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "target escaped original horizon"
        )
    reject_protected_dates(panel)
    manifest = {
        **SAFETY_FLAGS,
        "rows_before_cross_checkpoint_timestamp_deduplication": before_deduplication,
        "overlapping_timestamp_rows_removed": overlap_rows_removed,
        "deduplication_precedence": "lowest elapsed bar (newest opening decision)",
        "rows": int(len(panel)),
        "candidates": int(panel["candidate_id"].nunique()),
        "sessions": int(panel["session"].nunique()),
        "stocks": int(panel["symbol"].nunique()),
        "maximum_elapsed_bar": int(panel["elapsed_bar"].max()),
        "duplicate_stock_session_timestamps": int(
            panel.duplicated(["period", "symbol", "session", "update_timestamp_utc"]).sum()
        ),
        "updates_after_first_completion": 0,
        "targets_outside_original_horizon": 0,
        "target_metadata": metadata,
    }
    return panel.reset_index(drop=True), manifest


def sequential_weight_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, group in panel.groupby("candidate_id", sort=True):
        expected = float(group["candidate_total_weight"].iloc[0])
        observed = float(group["sequential_row_weight"].sum())
        rows.append(
            {
                "grouping": "candidate",
                "group_value": str(candidate_id),
                "rows": int(len(group)),
                "candidates": 1,
                "weight_sum": observed,
                "expected_weight_sum": expected,
                "absolute_difference": abs(observed - expected),
            }
        )
    for grouping, column in (
        ("session", "session"),
        ("checkpoint", "opening_checkpoint"),
        ("target_class", "next_registered_route"),
        ("elapsed_bar", "elapsed_bar"),
    ):
        for value, group in panel.groupby(column, sort=True):
            rows.append(
                {
                    "grouping": grouping,
                    "group_value": str(value),
                    "rows": int(len(group)),
                    "candidates": int(group["candidate_id"].nunique()),
                    "weight_sum": float(group["sequential_row_weight"].sum()),
                    "expected_weight_sum": math.nan,
                    "absolute_difference": math.nan,
                }
            )
    return pd.DataFrame(rows)


def freeze_bins(panel: pd.DataFrame) -> dict[str, Any]:
    development = panel.loc[panel["period"].eq("development")]
    positive_depth = development.loc[
        development["current_max_target_prefix_depth"].gt(0),
        "current_max_target_prefix_depth",
    ]
    depth_median = float(positive_depth.median()) if not positive_depth.empty else 0.0
    return {
        **SAFETY_FLAGS,
        "fit_period": "2024_only",
        "b0_decile_boundaries": frozen_quantile_boundaries(
            development.drop_duplicates("candidate_id")["candidate_B0_probability"],
            [value / 10 for value in range(1, 10)],
        ),
        "current_transition_probability_median": float(
            development["current_transition_probability"].median()
        ),
        "positive_maximum_target_prefix_depth_median": depth_median,
        "assessment_outcomes_inspected_before_freeze": False,
    }


def apply_bins(panel: pd.DataFrame, bins: Mapping[str, Any]) -> pd.DataFrame:
    result = panel.copy()
    result["b0_probability_decile"] = assign_frozen_bin(
        result["candidate_B0_probability"], cast(Sequence[float], bins["b0_decile_boundaries"])
    )
    transition_median = float(bins["current_transition_probability_median"])
    result["current_transition_half"] = np.where(
        result["current_transition_probability"].le(transition_median), "LOW", "HIGH"
    )
    depth_median = float(bins["positive_maximum_target_prefix_depth_median"])
    result["current_max_target_prefix_depth_bin"] = np.where(
        result["current_max_target_prefix_depth"].eq(0),
        "NO_PREFIX",
        np.where(result["current_max_target_prefix_depth"].le(depth_median), "LOW", "HIGH"),
    )
    result["elapsed_bar_group"] = pd.cut(
        result["elapsed_bar"],
        bins=[-1, 2, 4, 6],
        labels=["ELAPSED_0_2", "ELAPSED_3_4", "ELAPSED_5_6"],
    ).astype(str)
    return result


def fit_model_ladder(
    panel: pd.DataFrame, feature_sets: Mapping[str, Sequence[str]]
) -> tuple[dict[str, FittedMultinomial], pd.DataFrame]:
    development = panel.loc[panel["period"].eq("development")].copy()
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    models: dict[str, FittedMultinomial] = {}
    for name in MODEL_NAMES:
        model = fit_multinomial(name, development, feature_names=feature_sets[name])
        models[name] = model
        probabilities = predict_multinomial(model, assessment)
        for index, target_class in enumerate(model.classes):
            assessment[f"{name}_probability__{target_class}"] = probabilities[:, index]
    return models, assessment


def _probability_matrix(
    frame: pd.DataFrame, model_name: str, classes: Sequence[str]
) -> np.ndarray[Any, Any]:
    return frame.loc[:, [f"{model_name}_probability__{value}" for value in classes]].to_numpy(float)


def metric_rows(
    frame: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
    *,
    scope_type: str,
    scope_value: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    support = {
        str(key): int(value)
        for key, value in frame["next_registered_route"].value_counts().sort_index().items()
    }
    for name in MODEL_NAMES:
        model = models[name]
        metrics = multiclass_metrics(
            frame["next_registered_route"].astype(str).tolist(),
            _probability_matrix(frame, name, model.classes),
            model.classes,
            frame["sequential_row_weight"].astype(float).tolist(),
        )
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "model": name,
                **metrics,
                "rows": int(len(frame)),
                "unique_candidates": int(frame["candidate_id"].nunique()),
                "sessions": int(frame["session"].nunique()),
                "stocks": int(frame["symbol"].nunique()),
                "target_class_support": json.dumps(support, sort_keys=True),
                "weight_sum": float(frame["sequential_row_weight"].sum()),
            }
        )
    return rows


def model_metric_tables(
    assessment: pd.DataFrame, models: Mapping[str, FittedMultinomial]
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["pooled_metrics"] = pd.DataFrame(
        metric_rows(assessment, models, scope_type="pooled", scope_value="POOLED_ASSESSMENT")
    )
    specifications = {
        "target_class_metrics": ("next_registered_route", "target_class"),
        "checkpoint_metrics": ("opening_checkpoint", "opening_checkpoint"),
        "elapsed_bar_metrics": ("elapsed_bar_group", "elapsed_bar_group"),
        "monthly_metrics": ("year_month", "assessment_month"),
    }
    for table_name, (column, scope_type) in specifications.items():
        rows: list[dict[str, Any]] = []
        for value, group in assessment.groupby(column, sort=True, dropna=False):
            rows.extend(metric_rows(group, models, scope_type=scope_type, scope_value=str(value)))
        tables[table_name] = pd.DataFrame(rows)
    prefix_rows: list[dict[str, Any]] = []
    for column, scope_type in (
        ("prefix_state_group", "prefix_state"),
        ("current_transition_half", "current_transition_half"),
    ):
        for value, group in assessment.groupby(column, sort=True, dropna=False):
            prefix_rows.extend(
                metric_rows(group, models, scope_type=scope_type, scope_value=str(value))
            )
    tables["prefix_state_metrics"] = pd.DataFrame(prefix_rows)
    return tables


def pooled_increments(pooled: pd.DataFrame) -> dict[str, float]:
    lookup = pooled.set_index("model")
    return {
        "C1_minus_C0_log_loss_improvement": float(
            lookup.loc["C0", "multiclass_log_loss"] - lookup.loc["C1", "multiclass_log_loss"]
        ),
        "C1_minus_C0_brier_improvement": float(
            lookup.loc["C0", "multiclass_brier"] - lookup.loc["C1", "multiclass_brier"]
        ),
        "C1_minus_C0_top_two_change": float(
            lookup.loc["C1", "top_two_accuracy"] - lookup.loc["C0", "top_two_accuracy"]
        ),
        "C2_minus_C1_log_loss_improvement": float(
            lookup.loc["C1", "multiclass_log_loss"] - lookup.loc["C2", "multiclass_log_loss"]
        ),
        "C2_minus_C1_brier_improvement": float(
            lookup.loc["C1", "multiclass_brier"] - lookup.loc["C2", "multiclass_brier"]
        ),
        "C2_minus_C1_top_two_change": float(
            lookup.loc["C2", "top_two_accuracy"] - lookup.loc["C1", "top_two_accuracy"]
        ),
    }


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    vector = np.asarray(values, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    if vector.size == 0 or sample_weight.sum() <= 0.0:
        return math.nan
    return float(np.sum(vector * sample_weight) / sample_weight.sum())


def _contrast_scope_groups(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    groups: list[tuple[str, str, pd.DataFrame]] = [("pooled", "POOLED_ASSESSMENT", frame)]
    for column, scope in (
        ("opening_checkpoint", "opening_checkpoint"),
        ("elapsed_bar_group", "elapsed_bar_group"),
        ("year_month", "assessment_month"),
        ("current_transition_half", "current_transition_half"),
        ("prefix_state_group", "prefix_state"),
        ("next_registered_route", "target_class"),
    ):
        groups.extend(
            (scope, str(value), group)
            for value, group in frame.groupby(column, sort=True, dropna=False)
        )
    return groups


def target_specific_contrasts(
    assessment: pd.DataFrame, model: FittedMultinomial
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = assessment.copy()
    registered_classes = [value for value in model.classes if value != NO_REGISTERED_COMPLETION]
    specifications = [
        {
            "contrast_id": "A_hidden_5_6_5_to_target_a",
            "observed_column": "hidden_5_6_5_seen_since_opening",
            "zero_features": ["hidden_5_6_5_seen_since_opening"],
            "target_classes": [TARGET_A],
        },
        {
            "contrast_id": "B_hidden_5_6_5_to_target_b",
            "observed_column": "hidden_5_6_5_seen_since_opening",
            "zero_features": ["hidden_5_6_5_seen_since_opening"],
            "target_classes": [TARGET_B],
        },
        {
            "contrast_id": "C_recent_4_6_4_to_target_c",
            "observed_column": "loop_p_4_6_4_completed_previous_6_bars",
            "zero_features": [
                "prior_target_c_within_6",
                "loop_p_4_6_4_completed_previous_6_bars",
            ],
            "target_classes": [TARGET_C],
        },
        {
            "contrast_id": "D_hidden_2_3_2_to_any_registered",
            "observed_column": "hidden_2_3_2_seen_since_opening",
            "zero_features": ["hidden_2_3_2_seen_since_opening"],
            "target_classes": registered_classes,
        },
    ]
    rows: list[dict[str, Any]] = []
    for specification in specifications:
        contrast_id = str(specification["contrast_id"])
        observed_column = str(specification["observed_column"])
        targets = cast(list[str], specification["target_classes"])
        effect_column = f"contrast_effect__{contrast_id}"
        result[effect_column] = math.nan
        treated = result.loc[result[observed_column].astype(float).gt(0)].copy()
        supported = bool(targets) and all(value in model.classes for value in targets)
        if supported and not treated.empty:
            effects = counterfactual_probability_difference(
                model,
                treated,
                zero_features=cast(list[str], specification["zero_features"]),
                target_classes=targets,
            )
            result.loc[treated.index, effect_column] = effects
        for scope_type, scope_value, group in _contrast_scope_groups(result):
            matched = group.loc[group[observed_column].astype(float).gt(0)].copy()
            if targets == registered_classes:
                outcomes = matched["next_registered_route"].astype(str).ne(NO_REGISTERED_COMPLETION)
            else:
                outcomes = matched["next_registered_route"].astype(str).isin(targets)
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "feature_observed": observed_column,
                    "features_zeroed": json.dumps(specification["zero_features"]),
                    "target_classes": json.dumps(targets),
                    "probability_effect_original_minus_counterfactual": _weighted_mean(
                        matched[effect_column].dropna(),
                        matched.loc[matched[effect_column].notna(), "sequential_row_weight"],
                    ),
                    "observed_outcome_rate": _weighted_mean(
                        outcomes.astype(float), matched["sequential_row_weight"]
                    )
                    if not matched.empty
                    else math.nan,
                    "rows": int(len(matched)),
                    "unique_candidates": int(matched["candidate_id"].nunique()),
                    "sessions": int(matched["session"].nunique()),
                    "stocks": int(matched["symbol"].nunique()),
                    "target_probability_available": supported,
                }
            )
    return pd.DataFrame(rows), result


def _outcome_indicator(frame: pd.DataFrame, outcome: str) -> pd.Series:
    if outcome == "ANY_REGISTERED_COMPLETION":
        return frame["next_registered_route"].astype(str).ne(NO_REGISTERED_COMPLETION).astype(float)
    return frame["next_registered_route"].astype(str).eq(outcome).astype(float)


def matched_candidate_comparisons(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = [
        "year_month",
        "opening_checkpoint",
        "elapsed_bar",
        "b0_probability_decile",
        "current_transition_half",
        "current_max_target_prefix_depth_bin",
    ]
    specifications = [
        (
            "hidden_5_6_5",
            "new_hidden_5_6_5_completion",
            "hidden_5_6_5_seen_since_opening",
            [TARGET_A, TARGET_B, "ANY_REGISTERED_COMPLETION"],
        ),
        (
            "hidden_2_3_2",
            "new_hidden_2_3_2_completion",
            "hidden_2_3_2_seen_since_opening",
            ["ANY_REGISTERED_COMPLETION"],
        ),
        (
            "registered_4_6_4",
            "new_registered_4_6_4_completion",
            "any_registered_loop_completed_since_opening",
            [TARGET_C],
        ),
    ]
    panel_lookup = assessment.set_index("sequential_row_id", drop=False)
    relation_rows: list[pd.DataFrame] = []
    metric_rows_output: list[dict[str, Any]] = []
    for precursor, treated_column, history_column, outcomes in specifications:
        relations = matched_control_relations(
            assessment,
            treated_column=treated_column,
            untreated_history_column=history_column,
            stratum_columns=strata,
            minimum_controls=5,
        )
        if relations.empty:
            for outcome in outcomes:
                metric_rows_output.append(
                    {
                        "precursor": precursor,
                        "outcome": outcome,
                        "treated_rows": 0,
                        "control_relation_rows": 0,
                        "treated_completion_rate": math.nan,
                        "matched_control_completion_rate": math.nan,
                        "treated_minus_control_rate": math.nan,
                        "minimum_controls_required": 5,
                    }
                )
            continue
        relations["precursor"] = precursor
        relation_rows.append(relations)
        treated_ids = relations["treated_row_id"].drop_duplicates().astype(str)
        for outcome in outcomes:
            treated_values: list[float] = []
            control_values: list[float] = []
            treated_weights: list[float] = []
            for treated_id in treated_ids:
                source = panel_lookup.loc[treated_id]
                source_relations = relations.loc[relations["treated_row_id"].eq(treated_id)]
                controls = panel_lookup.loc[source_relations["control_row_id"].astype(str)]
                treated_values.append(
                    float(_outcome_indicator(source.to_frame().T, outcome).iloc[0])
                )
                control_indicator = _outcome_indicator(controls, outcome).to_numpy(float)
                control_weight = source_relations["control_weight_within_treated"].to_numpy(float)
                control_values.append(float(np.sum(control_indicator * control_weight)))
                treated_weights.append(float(source["sequential_row_weight"]))
            treated_rate = _weighted_mean(treated_values, treated_weights)
            control_rate = _weighted_mean(control_values, treated_weights)
            metric_rows_output.append(
                {
                    "precursor": precursor,
                    "outcome": outcome,
                    "treated_rows": int(len(treated_ids)),
                    "control_relation_rows": int(len(relations)),
                    "treated_completion_rate": treated_rate,
                    "matched_control_completion_rate": control_rate,
                    "treated_minus_control_rate": float(treated_rate - control_rate),
                    "minimum_controls_required": 5,
                }
            )
    relation_frame = (
        pd.concat(relation_rows, ignore_index=True, sort=False)
        if relation_rows
        else pd.DataFrame(
            columns=[
                "precursor",
                "treated_row_id",
                "control_row_id",
                "control_weight_within_treated",
            ]
        )
    )
    return pd.DataFrame(metric_rows_output), relation_frame


def _matched_relation_bootstrap_counts(
    assessment: pd.DataFrame,
    matched_relations: pd.DataFrame,
    session_counts: Mapping[str, int],
) -> dict[str, int]:
    if matched_relations.empty:
        return {
            "matched_relation_rows": 0,
            "matched_relation_weighted_rows": 0,
            "matched_treated_rows": 0,
            "matched_treated_weighted_rows": 0,
        }
    row_session = assessment.set_index("sequential_row_id")["session"].astype(str)
    relation_multiplicity = (
        matched_relations["treated_row_id"]
        .astype(str)
        .map(row_session)
        .map(session_counts)
        .fillna(0)
    )
    preserved = matched_relations.loc[relation_multiplicity.gt(0)]
    treated_multiplicity = (
        preserved[["treated_row_id"]]
        .drop_duplicates()["treated_row_id"]
        .astype(str)
        .map(row_session)
        .map(session_counts)
        .fillna(0)
    )
    return {
        "matched_relation_rows": int(len(preserved)),
        "matched_relation_weighted_rows": int(relation_multiplicity.loc[preserved.index].sum()),
        "matched_treated_rows": int(len(treated_multiplicity)),
        "matched_treated_weighted_rows": int(treated_multiplicity.sum()),
    }


def bootstrap_metrics(
    assessment: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
    matched_relations: pd.DataFrame,
) -> pd.DataFrame:
    """Run exactly 25 fixed-prediction whole-session bootstrap draws."""

    multiplicities = session_bootstrap_multiplicities(
        assessment["session"].astype(str).tolist(), draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    metric_names = [
        "C1_minus_C0_log_loss_improvement",
        "C1_minus_C0_brier_improvement",
        "C1_minus_C0_top_two_change",
        "C2_minus_C1_log_loss_improvement",
        "C2_minus_C1_brier_improvement",
        "C2_minus_C1_top_two_change",
        "A_hidden_5_6_5_to_target_a_probability_effect",
        "B_hidden_5_6_5_to_target_b_probability_effect",
        "C_recent_4_6_4_to_target_c_probability_effect",
        "D_hidden_2_3_2_to_any_registered_probability_effect",
        "candidate_level_any_registered_rate",
    ]
    draw_rows: list[dict[str, Any]] = []
    candidates = assessment.sort_values("elapsed_bar", kind="mergesort").drop_duplicates(
        "candidate_id"
    )
    for draw, counts in enumerate(multiplicities):
        sampled = assessment.copy()
        sampled["bootstrap_multiplicity"] = sampled["session"].astype(str).map(counts).fillna(0)
        sampled = sampled.loc[sampled["bootstrap_multiplicity"].gt(0)].copy()
        sampled["bootstrap_weight"] = (
            sampled["sequential_row_weight"] * sampled["bootstrap_multiplicity"]
        )
        model_values: dict[str, dict[str, float]] = {}
        for name in MODEL_NAMES:
            model = models[name]
            model_values[name] = multiclass_metrics(
                sampled["next_registered_route"].astype(str).tolist(),
                _probability_matrix(sampled, name, model.classes),
                model.classes,
                sampled["bootstrap_weight"].tolist(),
            )
        values = {
            "C1_minus_C0_log_loss_improvement": model_values["C0"]["multiclass_log_loss"]
            - model_values["C1"]["multiclass_log_loss"],
            "C1_minus_C0_brier_improvement": model_values["C0"]["multiclass_brier"]
            - model_values["C1"]["multiclass_brier"],
            "C1_minus_C0_top_two_change": model_values["C1"]["top_two_accuracy"]
            - model_values["C0"]["top_two_accuracy"],
            "C2_minus_C1_log_loss_improvement": model_values["C1"]["multiclass_log_loss"]
            - model_values["C2"]["multiclass_log_loss"],
            "C2_minus_C1_brier_improvement": model_values["C1"]["multiclass_brier"]
            - model_values["C2"]["multiclass_brier"],
            "C2_minus_C1_top_two_change": model_values["C2"]["top_two_accuracy"]
            - model_values["C1"]["top_two_accuracy"],
        }
        contrast_map = {
            "A_hidden_5_6_5_to_target_a_probability_effect": (
                "contrast_effect__A_hidden_5_6_5_to_target_a",
                "hidden_5_6_5_seen_since_opening",
            ),
            "B_hidden_5_6_5_to_target_b_probability_effect": (
                "contrast_effect__B_hidden_5_6_5_to_target_b",
                "hidden_5_6_5_seen_since_opening",
            ),
            "C_recent_4_6_4_to_target_c_probability_effect": (
                "contrast_effect__C_recent_4_6_4_to_target_c",
                "loop_p_4_6_4_completed_previous_6_bars",
            ),
            "D_hidden_2_3_2_to_any_registered_probability_effect": (
                "contrast_effect__D_hidden_2_3_2_to_any_registered",
                "hidden_2_3_2_seen_since_opening",
            ),
        }
        for metric, (effect_column, observed_column) in contrast_map.items():
            group = sampled.loc[
                sampled[observed_column].astype(float).gt(0) & sampled[effect_column].notna()
            ]
            values[metric] = _weighted_mean(group[effect_column], group["bootstrap_weight"])
        sampled_candidates = candidates.copy()
        sampled_candidates["bootstrap_multiplicity"] = (
            sampled_candidates["session"].astype(str).map(counts).fillna(0)
        )
        sampled_candidates = sampled_candidates.loc[
            sampled_candidates["bootstrap_multiplicity"].gt(0)
        ]
        candidate_weights = (
            sampled_candidates["candidate_total_weight"]
            * sampled_candidates["bootstrap_multiplicity"]
        )
        values["candidate_level_any_registered_rate"] = _weighted_mean(
            sampled_candidates["original_registered_candidate"], candidate_weights
        )
        relation_counts = _matched_relation_bootstrap_counts(assessment, matched_relations, counts)
        for metric in metric_names:
            draw_rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "interval_level": math.nan,
                    "value": float(values.get(metric, math.nan)),
                    "lower": math.nan,
                    "upper": math.nan,
                    **relation_counts,
                }
            )
    draws = pd.DataFrame(draw_rows)
    summaries: list[dict[str, Any]] = []
    for metric, group in draws.groupby("metric", sort=True):
        values = group["value"].dropna().to_numpy(float)
        point = float(values.mean()) if values.size else math.nan
        for level in (0.80, 0.90, 0.95):
            alpha = (1.0 - level) / 2.0
            summaries.append(
                {
                    "record_type": "summary",
                    "draw": -1,
                    "metric": metric,
                    "interval_level": level,
                    "value": point,
                    "lower": float(np.quantile(values, alpha, method="linear"))
                    if values.size
                    else math.nan,
                    "upper": float(np.quantile(values, 1.0 - alpha, method="linear"))
                    if values.size
                    else math.nan,
                }
            )
    return pd.concat([draws, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def hidden_history_null_metrics(
    panel: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    c1_model: FittedMultinomial,
    real_increments: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, int], bool]:
    """Run exactly five within-stage hidden-bundle null refits."""

    hidden_bundle = [value for value in feature_sets["C2"] if value not in feature_sets["C1"]]
    development = panel.loc[panel["period"].eq("development")].copy()
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    c1_probabilities = predict_multinomial(c1_model, assessment)
    c1_metrics = multiclass_metrics(
        assessment["next_registered_route"].astype(str).tolist(),
        c1_probabilities,
        c1_model.classes,
        assessment["sequential_row_weight"].tolist(),
    )
    rows: list[dict[str, Any]] = []
    converged = True
    for draw in range(HIDDEN_NULL_REFITS):
        shuffled_development = permute_hidden_bundle(
            development,
            bundle_columns=hidden_bundle,
            group_columns=["period", "session", "opening_checkpoint", "elapsed_bar"],
            seed=HIDDEN_NULL_SEED + draw,
        )
        shuffled_assessment = permute_hidden_bundle(
            assessment,
            bundle_columns=hidden_bundle,
            group_columns=["period", "session", "opening_checkpoint", "elapsed_bar"],
            seed=HIDDEN_NULL_SEED + draw,
        )
        null_model = fit_multinomial(
            f"C2_NULL_{draw}", shuffled_development, feature_names=feature_sets["C2"]
        )
        converged = converged and null_model.converged
        null_metrics = multiclass_metrics(
            shuffled_assessment["next_registered_route"].astype(str).tolist(),
            predict_multinomial(null_model, shuffled_assessment),
            null_model.classes,
            shuffled_assessment["sequential_row_weight"].tolist(),
        )
        values = {
            "multiclass_log_loss_improvement": c1_metrics["multiclass_log_loss"]
            - null_metrics["multiclass_log_loss"],
            "multiclass_brier_improvement": c1_metrics["multiclass_brier"]
            - null_metrics["multiclass_brier"],
            "top_two_accuracy_change": null_metrics["top_two_accuracy"]
            - c1_metrics["top_two_accuracy"],
        }
        for metric, value in values.items():
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "null_increment": value,
                    "real_increment": (
                        real_increments["C2_minus_C1_log_loss_improvement"]
                        if metric == "multiclass_log_loss_improvement"
                        else real_increments["C2_minus_C1_brier_improvement"]
                        if metric == "multiclass_brier_improvement"
                        else real_increments["C2_minus_C1_top_two_change"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    exceeded: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for metric, group in frame.groupby("metric", sort=True):
        real = float(group["real_increment"].iloc[0])
        count = int((real > group["null_increment"].astype(float)).sum())
        exceeded[str(metric)] = count
        summaries.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "null_increment": float(group["null_increment"].mean()),
                "real_increment": real,
                "null_draws_exceeded": count,
            }
        )
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True), exceeded, converged


def concentration_and_support(
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    retained_targets: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assessment_candidates = candidates.loc[candidates["period"].eq("assessment")].copy()
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    candidate_stock = assessment_candidates["symbol"].value_counts(normalize=True)
    weighted_stock = assessment.groupby("symbol")["sequential_row_weight"].sum()
    weighted_stock = weighted_stock / weighted_stock.sum()
    weighted_target = assessment.groupby("next_registered_route")["sequential_row_weight"].sum()
    weighted_target = weighted_target / weighted_target.sum()
    candidate_target_support = {
        target: int(
            assessment_candidates["original_first_registered_semantic_loop_id"]
            .astype(str)
            .eq(target)
            .sum()
        )
        for target in retained_targets
    }
    values = {
        "assessment_candidate_rows": int(len(assessment_candidates)),
        "assessment_candidate_sessions": int(assessment_candidates["session"].nunique()),
        "assessment_candidate_stocks": int(assessment_candidates["symbol"].nunique()),
        "assessment_registered_candidates": int(
            assessment_candidates["original_registered_completion"].astype(int).sum()
        ),
        "assessment_sequential_rows": int(len(assessment)),
        "assessment_rows_after_hidden_event": int(
            assessment["any_hidden_event_since_opening"].astype(bool).sum()
        ),
        "maximum_candidate_stock_share": float(candidate_stock.max()),
        "maximum_weighted_stock_share": float(weighted_stock.max()),
        "maximum_weighted_target_class_share": float(weighted_target.max()),
        "exact_target_candidate_support": candidate_target_support,
    }
    checks = {
        "candidate_rows_at_least_850": values["assessment_candidate_rows"] >= 850,
        "candidate_sessions_at_least_140": values["assessment_candidate_sessions"] >= 140,
        "candidate_stocks_at_least_15": values["assessment_candidate_stocks"] >= 15,
        "registered_candidates_at_least_180": values["assessment_registered_candidates"] >= 180,
        "sequential_rows_at_least_4000": values["assessment_sequential_rows"] >= 4000,
        "rows_after_hidden_event_at_least_150": values["assessment_rows_after_hidden_event"] >= 150,
        "no_stock_above_10pct_candidate_rows": values["maximum_candidate_stock_share"] <= 0.10,
        "no_stock_above_10pct_weighted_rows": values["maximum_weighted_stock_share"] <= 0.10,
        "no_target_class_above_85pct_weighted_rows": values["maximum_weighted_target_class_share"]
        <= 0.85,
    }
    value_checks = {
        "assessment_candidate_rows": checks["candidate_rows_at_least_850"],
        "assessment_candidate_sessions": checks["candidate_sessions_at_least_140"],
        "assessment_candidate_stocks": checks["candidate_stocks_at_least_15"],
        "assessment_registered_candidates": checks["registered_candidates_at_least_180"],
        "assessment_sequential_rows": checks["sequential_rows_at_least_4000"],
        "assessment_rows_after_hidden_event": checks["rows_after_hidden_event_at_least_150"],
        "maximum_candidate_stock_share": checks["no_stock_above_10pct_candidate_rows"],
        "maximum_weighted_stock_share": checks["no_stock_above_10pct_weighted_rows"],
        "maximum_weighted_target_class_share": checks["no_target_class_above_85pct_weighted_rows"],
    }
    rows = [
        {
            "scope": "assessment",
            "gate": key,
            "value": float(value) if isinstance(value, (int, float)) else json.dumps(value),
            "passed": value_checks[key],
        }
        for key, value in values.items()
        if key != "exact_target_candidate_support"
    ]
    for key, passed in checks.items():
        rows.append(
            {
                "scope": "assessment_support",
                "gate": key,
                "value": float(passed),
                "passed": passed,
            }
        )
    for target, count in candidate_target_support.items():
        rows.append(
            {
                "scope": "exact_target_assessment_support",
                "gate": target,
                "value": count,
                "passed": count >= 100,
            }
        )
    return pd.DataFrame(rows), {
        **values,
        "checks": checks,
        "support_passed": all(checks.values()),
        "exact_target_inference_supported": {
            target: count >= 100 for target, count in candidate_target_support.items()
        },
    }


def transition_support_gate(support: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for lookback in (6, 12):
        values = cast(Mapping[str, Any], support[f"assessment_{lookback}"])
        checks[f"lookback_{lookback}_eligible_at_least_500"] = int(values["eligible_events"]) >= 500
        checks[f"lookback_{lookback}_sessions_at_least_120"] = int(values["sessions"]) >= 120
        checks[f"lookback_{lookback}_stocks_at_least_15"] = int(values["stocks"]) >= 15
        checks[f"lookback_{lookback}_months_at_least_6"] = int(values["months"]) >= 6
        checks[f"lookback_{lookback}_matched_null_coverage_at_least_0_90"] = (
            float(values["minimum_matched_null_coverage"]) >= 0.90
        )
    return {"lookbacks": dict(support), "checks": checks, "support_passed": all(checks.values())}


def _bootstrap_interval(bootstrap: pd.DataFrame, metric: str, *, level: float, bound: str) -> float:
    row = bootstrap.loc[
        bootstrap["record_type"].eq("summary")
        & bootstrap["metric"].eq(metric)
        & bootstrap["interval_level"].astype(float).eq(level)
    ]
    if len(row) != 1:
        return math.nan
    return float(row.iloc[0][bound])


def _hypothesis_row(
    null_metrics: pd.DataFrame, period: str, hypothesis: str, lookback: int
) -> pd.Series:
    group = null_metrics.loc[
        null_metrics["record_type"].eq("summary")
        & null_metrics["period"].eq(period)
        & null_metrics["hypothesis_id"].eq(hypothesis)
        & null_metrics["lookback_bars"].eq(lookback)
    ]
    if len(group) != 1:
        raise ValueError(f"transition summary missing: {period}/{hypothesis}/{lookback}")
    return group.iloc[0]


def _precursor_support(
    census: pd.DataFrame, hypothesis: str, lookbacks: Sequence[int]
) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    for period in ("development", "assessment"):
        for lookback in lookbacks:
            group = census.loc[
                census["period"].eq(period)
                & census["hypothesis_id"].eq(hypothesis)
                & census["lookback_bars"].eq(lookback)
            ]
            if len(group) != 1:
                return False, {"rows": rows, "checks": {"row_available": False}}
            value = group.iloc[0]
            rows.append(value.to_dict())
            minimum = 30 if period == "development" else 20
            checks[f"{period}_{lookback}_events"] = int(value["precursor_events"]) >= minimum
            checks[f"{period}_{lookback}_sessions"] = int(value["sessions"]) >= 10
            checks[f"{period}_{lookback}_stocks"] = int(value["stocks"]) >= 8
            checks[f"{period}_{lookback}_months"] = int(value["months"]) >= 4
            checks[f"{period}_{lookback}_stock_concentration"] = (
                float(value["maximum_stock_share"]) <= 0.30
            )
            checks[f"{period}_{lookback}_month_concentration"] = (
                float(value["maximum_month_share"]) <= 0.30
            )
    return all(checks.values()), {"rows": rows, "checks": checks}


def evaluate_decision(
    *,
    census: pd.DataFrame,
    transition_null: pd.DataFrame,
    multiplicity: pd.DataFrame,
    transition_support: Mapping[str, Any],
    model_support: Mapping[str, Any],
    metrics: Mapping[str, pd.DataFrame],
    increments: Mapping[str, float],
    contrasts: pd.DataFrame,
    matched: pd.DataFrame,
    bootstrap: pd.DataFrame,
    hidden_null_exceeded: Mapping[str, int],
    converged: bool,
) -> tuple[dict[str, str], dict[str, Any], str | None]:
    q_lookup = multiplicity.set_index("hypothesis_id")["q_value"].to_dict()
    contrast_lookup = contrasts.loc[contrasts["scope_type"].eq("pooled")].set_index("contrast_id")
    matched_lookup = matched.set_index(["precursor", "outcome"])
    target_inference = cast(Mapping[str, bool], model_support["exact_target_inference_supported"])
    hypothesis_gates: dict[str, Any] = {}
    specifications = {
        "H1": {
            "lookbacks": [12],
            "contrast": "A_hidden_5_6_5_to_target_a",
            "bootstrap": "A_hidden_5_6_5_to_target_a_probability_effect",
            "matched": ("hidden_5_6_5", TARGET_A),
            "target": TARGET_A,
            "sign": "positive",
        },
        "H2": {
            "lookbacks": [12],
            "contrast": "B_hidden_5_6_5_to_target_b",
            "bootstrap": "B_hidden_5_6_5_to_target_b_probability_effect",
            "matched": ("hidden_5_6_5", TARGET_B),
            "target": TARGET_B,
            "sign": "positive",
        },
        "H3": {
            "lookbacks": [6, 12],
            "contrast": "C_recent_4_6_4_to_target_c",
            "bootstrap": "C_recent_4_6_4_to_target_c_probability_effect",
            "matched": ("registered_4_6_4", TARGET_C),
            "target": TARGET_C,
            "sign": "positive",
        },
        "H4": {
            "lookbacks": [6],
            "contrast": "D_hidden_2_3_2_to_any_registered",
            "bootstrap": "D_hidden_2_3_2_to_any_registered_probability_effect",
            "matched": ("hidden_2_3_2", "ANY_REGISTERED_COMPLETION"),
            "target": None,
            "sign": "negative",
        },
    }
    for hypothesis, specification in specifications.items():
        lookbacks = cast(list[int], specification["lookbacks"])
        support_passed, support_detail = _precursor_support(census, hypothesis, lookbacks)
        transition_checks: dict[str, bool] = {}
        for period in ("development", "assessment"):
            for lookback in lookbacks:
                row = _hypothesis_row(transition_null, period, hypothesis, lookback)
                if specification["sign"] == "positive":
                    transition_checks[f"{period}_{lookback}_signed_enrichment"] = (
                        float(row["enrichment"]) > 0.0
                    )
                    if period == "assessment":
                        transition_checks[f"{period}_{lookback}_tail"] = float(
                            row["observed_prevalence"]
                        ) > float(row["null_90th_percentile"])
                else:
                    transition_checks[f"{period}_{lookback}_signed_enrichment"] = (
                        float(row["enrichment"]) < 0.0
                    )
                    if period == "assessment":
                        transition_checks[f"{period}_{lookback}_tail"] = float(
                            row["observed_prevalence"]
                        ) < float(row["null_10th_percentile"])
        contrast_value = float(
            contrast_lookup.loc[
                str(specification["contrast"]),
                "probability_effect_original_minus_counterfactual",
            ]
        )
        matched_key = cast(tuple[str, str], specification["matched"])
        matched_value = (
            float(matched_lookup.loc[matched_key, "treated_minus_control_rate"])
            if matched_key in matched_lookup.index
            else math.nan
        )
        if specification["sign"] == "positive":
            contrast_sign = contrast_value > 0.0
            interval_sign = (
                _bootstrap_interval(
                    bootstrap, str(specification["bootstrap"]), level=0.80, bound="lower"
                )
                >= 0.0
            )
            matched_sign = matched_value > 0.0
        else:
            contrast_sign = contrast_value < 0.0
            interval_sign = (
                _bootstrap_interval(
                    bootstrap, str(specification["bootstrap"]), level=0.80, bound="upper"
                )
                <= 0.0
            )
            matched_sign = matched_value < 0.0
        checks = {
            **transition_checks,
            "q_le_0_10": float(q_lookup[hypothesis]) <= 0.10,
            "precursor_support": support_passed,
            "contrast_signed": contrast_sign,
            "contrast_80pct_interval_signed": interval_sign,
        }
        if hypothesis != "H3":
            checks["matched_rate_signed"] = matched_sign
        if hypothesis == "H3":
            checks["C1_improves_log_loss"] = increments["C1_minus_C0_log_loss_improvement"] > 0.0
            checks["C1_improves_brier"] = increments["C1_minus_C0_brier_improvement"] > 0.0
        target = cast(str | None, specification["target"])
        target_supported = target is None or bool(target_inference.get(target, False))
        if (
            not bool(transition_support["support_passed"])
            or not bool(model_support["support_passed"])
            or not target_supported
        ):
            status = "insufficient_support"
        elif all(checks.values()):
            status = "supported"
        elif any(transition_checks.values()) or contrast_sign or matched_sign:
            status = "descriptive_only"
        else:
            status = "not_supported"
        hypothesis_gates[hypothesis] = {
            "status": status,
            "checks": checks,
            "support": support_detail,
            "contrast": contrast_value,
            "matched_rate_difference": matched_value,
            "target_assessment_inference_supported": target_supported,
        }
    pooled = metrics["pooled_metrics"].set_index("model")
    monthly = metrics["monthly_metrics"]
    checkpoint = metrics["checkpoint_metrics"]

    def increments_by_scope(frame: pd.DataFrame, scope_value: str) -> tuple[float, float]:
        group = frame.loc[frame["scope_value"].astype(str).eq(scope_value)].set_index("model")
        return (
            float(group.loc["C1", "multiclass_log_loss"] - group.loc["C2", "multiclass_log_loss"]),
            float(group.loc["C1", "multiclass_brier"] - group.loc["C2", "multiclass_brier"]),
        )

    positive_months = 0
    for month in monthly["scope_value"].unique():
        log_increment, _ = increments_by_scope(monthly, str(month))
        positive_months += int(log_increment > 0.0)
    checkpoint_not_adverse = True
    for checkpoint_value in checkpoint["scope_value"].unique():
        log_increment, brier_increment = increments_by_scope(checkpoint, str(checkpoint_value))
        checkpoint_not_adverse &= log_increment >= -0.001 and brier_increment >= -0.001
    hidden_checks = {
        "C2_improves_log_loss": increments["C2_minus_C1_log_loss_improvement"] > 0.0,
        "C2_improves_brier": increments["C2_minus_C1_brier_improvement"] > 0.0,
        "top_two_not_reduced_more_than_0_002": increments["C2_minus_C1_top_two_change"] >= -0.002,
        "log_loss_80pct_lower_nonnegative": _bootstrap_interval(
            bootstrap, "C2_minus_C1_log_loss_improvement", level=0.80, bound="lower"
        )
        >= 0.0,
        "brier_80pct_lower_nonnegative": _bootstrap_interval(
            bootstrap, "C2_minus_C1_brier_improvement", level=0.80, bound="lower"
        )
        >= 0.0,
        "positive_in_at_least_five_months": positive_months >= 5,
        "neither_checkpoint_materially_adverse": checkpoint_not_adverse,
        "real_increment_exceeds_four_of_five_nulls": (
            hidden_null_exceeded.get("multiclass_log_loss_improvement", 0) >= 4
            or hidden_null_exceeded.get("multiclass_brier_improvement", 0) >= 4
        ),
        "concentration_gates_pass": bool(model_support["support_passed"]),
    }
    registered_checks = {
        "C1_improves_log_loss": increments["C1_minus_C0_log_loss_improvement"] > 0.0,
        "C1_improves_brier": increments["C1_minus_C0_brier_improvement"] > 0.0,
        "top_two_not_reduced_more_than_0_002": increments["C1_minus_C0_top_two_change"] >= -0.002,
        "log_loss_80pct_lower_nonnegative": _bootstrap_interval(
            bootstrap, "C1_minus_C0_log_loss_improvement", level=0.80, bound="lower"
        )
        >= 0.0,
        "brier_80pct_lower_nonnegative": _bootstrap_interval(
            bootstrap, "C1_minus_C0_brier_improvement", level=0.80, bound="lower"
        )
        >= 0.0,
    }
    if not bool(model_support["support_passed"]):
        hidden_status = "insufficient_support"
        registered_status = "insufficient_support"
    else:
        hidden_status = (
            "supported"
            if all(hidden_checks.values())
            else "descriptive_only"
            if increments["C2_minus_C1_log_loss_improvement"] > 0
            or increments["C2_minus_C1_brier_improvement"] > 0
            else "not_supported"
        )
        registered_status = (
            "supported"
            if all(registered_checks.values())
            else "descriptive_only"
            if increments["C1_minus_C0_log_loss_improvement"] > 0
            or increments["C1_minus_C0_brier_improvement"] > 0
            else "not_supported"
        )
    blocker: str | None = None
    if not bool(transition_support["support_passed"]):
        blocker = "blocked_transition_census_support_failure"
    elif not bool(model_support["support_passed"]):
        blocker = "blocked_sequential_model_support_failure"
    elif not converged:
        blocker = "blocked_model_convergence_failure"
    statuses = {
        "target_a_precursor_status": str(hypothesis_gates["H1"]["status"]),
        "target_b_precursor_status": str(hypothesis_gates["H2"]["status"]),
        "target_c_recurrence_status": str(hypothesis_gates["H3"]["status"]),
        "hidden_2_3_2_diversion_status": str(hypothesis_gates["H4"]["status"]),
        "registered_history_increment_status": registered_status,
        "hidden_history_increment_status": hidden_status,
    }
    gates = {
        "hypotheses": hypothesis_gates,
        "registered_history_increment": registered_checks,
        "hidden_history_increment": hidden_checks,
        "positive_hidden_increment_months": positive_months,
        "checkpoint_not_materially_adverse": checkpoint_not_adverse,
        "pooled_metrics": pooled.reset_index().to_dict(orient="records"),
    }
    return statuses, gates, blocker


def _candidate_ids(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["symbol"].astype(str)
        + "|"
        + frame["session"].astype(str)
        + "|"
        + frame["decision_ordinal"].astype(int).astype(str).str.zfill(2)
    )


def _expected_candidate_membership(oof: pd.DataFrame, threshold: float) -> set[tuple[str, str]]:
    key = ["symbol", "session", "decision_ordinal"]
    development_keys = oof.loc[oof["B0_oof_probability"].astype(float).ge(threshold), key]
    archived_development = pd.read_parquet(
        BRIDGE_PRIMARY / "bridge_development_panel.parquet", columns=key
    )
    development = archived_development.merge(
        development_keys, on=key, how="inner", validate="one_to_one"
    )
    assessment = pd.read_parquet(
        BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet",
        columns=[*key, "B0_probability"],
    )
    assessment = assessment.loc[assessment["B0_probability"].astype(float).ge(threshold)]
    return {
        *(("development", value) for value in _candidate_ids(development)),
        *(("assessment", value) for value in _candidate_ids(assessment)),
    }


def _expected_sequential_rows(candidates: pd.DataFrame, registered: pd.DataFrame) -> pd.DataFrame:
    registered_groups = _group_lookup(registered)
    empty = pd.DataFrame(columns=["completion_bar_ordinal"])
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        opening = int(candidate.repo_bar_start_ordinal)
        group = registered_groups.get((str(candidate.symbol), str(candidate.session)), empty)
        future_ordinals = pd.to_numeric(
            group.loc[
                group["completion_bar_ordinal"].astype(int).gt(opening)
                & group["completion_bar_ordinal"].astype(int).le(opening + 12),
                "completion_bar_ordinal",
            ],
            errors="raise",
        ).astype(int)
        stop = (
            min(opening + 6, int(future_ordinals.min()))
            if not future_ordinals.empty
            else opening + 6
        )
        for update in range(opening, stop + 1):
            elapsed = update - opening
            rows.append(
                {
                    "period": str(candidate.period),
                    "candidate_id": str(candidate.candidate_id),
                    "symbol": str(candidate.symbol),
                    "session": str(candidate.session),
                    "elapsed_bar": elapsed,
                    "update_timestamp_utc": pd.Timestamp(candidate.feature_available_timestamp_utc)
                    + pd.Timedelta(minutes=5 * elapsed),
                    "sequential_row_id": f"{candidate.candidate_id}|E{elapsed}",
                }
            )
    expected = pd.DataFrame(rows)
    return (
        expected.sort_values(
            ["period", "session", "symbol", "update_timestamp_utc", "elapsed_bar"],
            kind="mergesort",
        )
        .drop_duplicates(["period", "symbol", "session", "update_timestamp_utc"], keep="first")
        .reset_index(drop=True)
    )


def _maximum_transition_enrichment_difference(
    transition_panel: pd.DataFrame,
    transition_census: pd.DataFrame,
    transition_null: pd.DataFrame,
) -> float:
    differences: list[float] = []
    observed = transition_panel.loc[transition_panel["record_type"].eq("observed")]
    for row in transition_census.itertuples(index=False):
        group = observed.loc[
            observed["period"].eq(row.period)
            & observed["hypothesis_id"].eq(row.hypothesis_id)
            & observed["lookback_bars"].eq(row.lookback_bars)
        ]
        differences.append(abs(float(group["precursor_present"].mean()) - row.observed_prevalence))
    draw_rows = transition_null.loc[transition_null["record_type"].eq("draw")]
    null_panel = transition_panel.loc[transition_panel["record_type"].eq("null")]
    for row in draw_rows.itertuples(index=False):
        group = null_panel.loc[
            null_panel["period"].eq(row.period)
            & null_panel["hypothesis_id"].eq(row.hypothesis_id)
            & null_panel["lookback_bars"].eq(row.lookback_bars)
            & null_panel["draw"].eq(row.draw)
        ]
        null_prevalence = float(group["precursor_present"].mean())
        differences.extend(
            [
                abs(null_prevalence - float(row.null_prevalence)),
                abs(float(row.observed_prevalence) - null_prevalence - float(row.enrichment)),
            ]
        )
    return float(max(differences, default=0.0))


def determinism_check(output: Path) -> dict[str, Any]:
    """Refit only deterministic models and labels; never repeat draws."""

    panel = pd.read_parquet(output / "sequential_candidate_panel.parquet")
    predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    registered = pd.read_parquet(output / "registered_completion_ledger.parquet")
    hidden = pd.read_parquet(output / "hidden_completion_ledger.parquet")
    transition_panel = pd.read_parquet(output / "transition_event_panel.parquet")
    transition_census = pd.read_csv(output / "transition_census.csv")
    transition_null = pd.read_csv(output / "transition_null_metrics.csv")
    candidate_population = pd.read_parquet(output / "candidate_population.parquet")
    oof = pd.read_parquet(output / "b0_development_oof_predictions.parquet")
    mapping = read_json(output / "target_class_mapping.json")
    coefficient_specs = read_json(output / "model_coefficients.json")
    configurations = read_json(output / "model_configurations.json")
    decision = read_json(output / "decision.json")
    feature_sets = {
        name: tuple(str(value) for value in configurations[name]["features"])
        for name in MODEL_NAMES
    }
    models: dict[str, FittedMultinomial] = {}
    coefficient_differences: list[float] = []
    probability_differences: list[float] = []
    development = panel.loc[panel["period"].eq("development")]
    assessment = panel.loc[panel["period"].eq("assessment")]
    for name in MODEL_NAMES:
        model = fit_multinomial(name, development, feature_names=feature_sets[name])
        models[name] = model
        archived = model_from_json(cast(Mapping[str, Any], coefficient_specs[name]))
        coefficient_differences.extend(
            np.abs(
                np.asarray(model.coefficients, dtype=float)
                - np.asarray(archived.coefficients, dtype=float)
            ).ravel()
        )
        coefficient_differences.extend(
            np.abs(np.asarray(model.intercept) - np.asarray(archived.intercept)).ravel()
        )
        probabilities = predict_multinomial(model, assessment)
        archived_probabilities = _probability_matrix(predictions, name, model.classes)
        probability_differences.extend(np.abs(probabilities - archived_probabilities).ravel())
    maximum_probability_difference = float(max(probability_differences, default=0.0))
    maximum_coefficient_difference = float(max(coefficient_differences, default=0.0))
    archived_pooled = pd.read_csv(output / "pooled_metrics.csv").set_index("model")
    metric_differences: list[float] = []
    for name, model in models.items():
        recalculated = multiclass_metrics(
            predictions["next_registered_route"].astype(str).tolist(),
            _probability_matrix(predictions, name, model.classes),
            model.classes,
            predictions["sequential_row_weight"].tolist(),
        )
        for metric, value in recalculated.items():
            metric_differences.append(abs(value - float(archived_pooled.loc[name, metric])))
    registered_groups = _group_lookup(registered)
    hidden_groups = _group_lookup(hidden)
    empty_registered = pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id"])
    empty_hidden = pd.DataFrame(columns=["completion_bar_ordinal", "hidden_family_class"])
    label_cache: dict[tuple[str, str, int, int, str, str], bool] = {}
    mismatches = 0
    for row in transition_panel.itertuples(index=False):
        key = (
            str(row.symbol),
            str(row.session),
            int(row.completion_bar_ordinal),
            int(row.lookback_bars),
            str(row.precursor_kind),
            str(row.precursor_identity),
        )
        if key not in label_cache:
            group_key = (key[0], key[1])
            label_cache[key] = precursor_present(
                completion_bar_ordinal=key[2],
                lookback_bars=key[3],
                precursor_kind=key[4],
                precursor_identity=key[5],
                registered_events=registered_groups.get(group_key, empty_registered),
                hidden_events=hidden_groups.get(group_key, empty_hidden),
            )
        mismatches += int(label_cache[key] != bool(row.precursor_present))
    expected_rows = _expected_sequential_rows(candidate_population, registered)
    actual_row_ids = set(panel["sequential_row_id"].astype(str))
    expected_row_ids = set(expected_rows["sequential_row_id"].astype(str))
    expected_timestamps = expected_rows.set_index("sequential_row_id")["update_timestamp_utc"]
    actual_timestamps = panel.set_index("sequential_row_id")["update_timestamp_utc"]
    common_ids = sorted(actual_row_ids.intersection(expected_row_ids))
    timestamp_mismatches = int(
        (
            pd.to_datetime(actual_timestamps.loc[common_ids], utc=True).to_numpy()
            != pd.to_datetime(expected_timestamps.loc[common_ids], utc=True).to_numpy()
        ).sum()
    )
    sequential_row_mismatches = int(
        len(actual_row_ids.symmetric_difference(expected_row_ids)) + timestamp_mismatches
    )
    candidate_weight_mismatches = int(
        panel.groupby("candidate_id")["sequential_row_weight"]
        .sum()
        .sub(panel.groupby("candidate_id")["candidate_total_weight"].first())
        .abs()
        .gt(1e-12)
        .sum()
    )
    recalculated_mapping = freeze_target_class_mapping(registered.loc[registered["year"].eq(2024)])
    target_mapping_match = (
        recalculated_mapping["retained_exact_targets"] == mapping["retained_exact_targets"]
        and recalculated_mapping["final_target_classes"] == mapping["final_target_classes"]
    )
    recalculated_threshold = candidate_threshold(oof["B0_oof_probability"])
    actual_membership = set(
        zip(
            candidate_population["period"].astype(str),
            candidate_population["candidate_id"].astype(str),
            strict=True,
        )
    )
    expected_membership = _expected_candidate_membership(oof, recalculated_threshold)
    candidate_membership_mismatches = len(
        actual_membership.symmetric_difference(expected_membership)
    )
    maximum_transition_enrichment_difference = _maximum_transition_enrichment_difference(
        transition_panel, transition_census, transition_null
    )
    inputs = cast(Mapping[str, Any], decision["decision_inputs"])
    reconstructed_decision = choose_primary_decision(
        blocker=cast(str | None, inputs["blocker"]),
        target_a_status=str(inputs["target_a_precursor_status"]),
        target_b_status=str(inputs["target_b_precursor_status"]),
        target_c_status=str(inputs["target_c_recurrence_status"]),
        diversion_status=str(inputs["hidden_2_3_2_diversion_status"]),
        hidden_increment_status=str(inputs["hidden_history_increment_status"]),
    )
    final_decision_match = reconstructed_decision == str(decision["primary_decision"])
    maximum_metric_difference = float(max(metric_differences, default=0.0))
    passed = (
        maximum_probability_difference <= 1e-12
        and maximum_coefficient_difference <= 1e-12
        and maximum_metric_difference <= 1e-12
        and mismatches == 0
        and sequential_row_mismatches == 0
        and candidate_weight_mismatches == 0
        and candidate_membership_mismatches == 0
        and maximum_transition_enrichment_difference <= 1e-12
        and target_mapping_match
        and final_decision_match
    )
    return {
        **SAFETY_FLAGS,
        "passed": passed,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_coefficient_difference": maximum_coefficient_difference,
        "maximum_pooled_metric_difference": maximum_metric_difference,
        "transition_label_mismatches": mismatches,
        "sequential_row_mismatches": sequential_row_mismatches,
        "candidate_weight_mismatches": candidate_weight_mismatches,
        "candidate_membership_mismatches": candidate_membership_mismatches,
        "maximum_transition_enrichment_difference": maximum_transition_enrichment_difference,
        "target_class_mapping_match": target_mapping_match,
        "final_decision_match": final_decision_match,
        "models_refitted": 3,
        "bootstrap_repeated": False,
        "transition_null_repeated": False,
        "hidden_history_null_repeated": False,
    }


def create_plots(
    transition_null: pd.DataFrame,
    assessment: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
    output: Path,
) -> list[str]:
    plots: list[str] = []
    summary = transition_null.loc[
        transition_null["record_type"].eq("summary") & transition_null["period"].eq("assessment")
    ].copy()
    labels = [f"{row.hypothesis_id}/{int(row.lookback_bars)}" for row in summary.itertuples()]
    x = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(x - 0.18, summary["observed_prevalence"], 0.36, label="Observed")
    axis.bar(x + 0.18, summary["mean_null_prevalence"], 0.36, label="Matched null")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("Precursor prevalence")
    axis.set_title("Fixed target-specific precursor census")
    axis.legend()
    figure.tight_layout()
    first = output / "target_specific_precursor_enrichment.png"
    figure.savefig(first, dpi=140)
    plt.close(figure)
    plots.append(str(first))

    plot_specs = [
        ("5→6→5 / target A", "hidden_5_6_5_seen_since_opening", [TARGET_A]),
        ("5→6→5 / target B", "hidden_5_6_5_seen_since_opening", [TARGET_B]),
        (
            "4→6→4 / target C",
            "loop_p_4_6_4_completed_previous_6_bars",
            [TARGET_C],
        ),
        (
            "2→3→2 / any registered",
            "hidden_2_3_2_seen_since_opening",
            [value for value in models["C2"].classes if value != NO_REGISTERED_COMPLETION],
        ),
    ]
    values = np.zeros((len(plot_specs), len(MODEL_NAMES)), dtype=float)
    for row_index, (_, observed_column, targets) in enumerate(plot_specs):
        group = assessment.loc[assessment[observed_column].astype(float).gt(0)]
        for column_index, name in enumerate(MODEL_NAMES):
            model = models[name]
            available = [value for value in targets if value in model.classes]
            if group.empty or not available:
                values[row_index, column_index] = math.nan
                continue
            indices = [model.classes.index(value) for value in available]
            probability = _probability_matrix(group, name, model.classes)[:, indices].sum(axis=1)
            values[row_index, column_index] = _weighted_mean(
                probability, group["sequential_row_weight"]
            )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(plot_specs))
    width = 0.24
    for index, name in enumerate(MODEL_NAMES):
        axis.bar(x + (index - 1) * width, values[:, index], width, label=name)
    axis.set_xticks(x, [value[0] for value in plot_specs], rotation=25, ha="right")
    axis.set_ylabel("Mean target probability")
    axis.set_title("Route probabilities after causally observed precursors")
    axis.legend()
    figure.tight_layout()
    second = output / "model_probabilities_after_precursors.png"
    figure.savefig(second, dpi=140)
    plt.close(figure)
    plots.append(str(second))
    return plots


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 40) -> str:
    available = [value for value in columns if value in frame.columns]
    if frame.empty or not available:
        return "No rows."

    def format_cell(value: Any) -> str:
        if value is None or bool(pd.isna(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.9g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(available) + " |"]
    rows.append("| " + " | ".join("---" for _ in available) + " |")
    for values in frame.loc[:, available].head(limit).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(format_cell(value) for value in values) + " |")
    return "\n".join(rows)


def build_report(
    decision: Mapping[str, Any],
    mapping: Mapping[str, Any],
    transition_census: pd.DataFrame,
    transition_null: pd.DataFrame,
    multiplicity: pd.DataFrame,
    pooled: pd.DataFrame,
    contrasts: pd.DataFrame,
    matched: pd.DataFrame,
    bootstrap: pd.DataFrame,
    hidden_null: pd.DataFrame,
) -> str:
    summary = transition_null.loc[transition_null["record_type"].eq("summary")]
    boot = bootstrap.loc[
        bootstrap["record_type"].eq("summary") & bootstrap["interval_level"].eq(0.80)
    ]
    null_summary = hidden_null.loc[hidden_null["record_type"].eq("summary")]
    census_table = markdown_table(
        transition_census,
        [
            "period",
            "hypothesis_id",
            "lookback_bars",
            "eligible_events",
            "ineligible_events",
            "precursor_events",
            "observed_prevalence",
            "sessions",
            "stocks",
            "months",
        ],
    )
    transition_table = markdown_table(
        summary,
        [
            "period",
            "hypothesis_id",
            "lookback_bars",
            "observed_prevalence",
            "mean_null_prevalence",
            "enrichment",
            "null_percentile",
            "null_10th_percentile",
            "null_90th_percentile",
        ],
    )
    pooled_table = markdown_table(
        pooled,
        [
            "model",
            "multiclass_log_loss",
            "multiclass_brier",
            "top_one_accuracy",
            "top_two_accuracy",
            "mean_reciprocal_rank",
            "mean_probability_realised_class",
            "expected_calibration_error",
            "prediction_entropy",
            "effective_candidate_count",
            "rows",
            "unique_candidates",
        ],
    )
    contrast_table = markdown_table(
        contrasts.loc[contrasts["scope_type"].eq("pooled")],
        [
            "contrast_id",
            "probability_effect_original_minus_counterfactual",
            "observed_outcome_rate",
            "rows",
            "unique_candidates",
            "sessions",
            "stocks",
        ],
    )
    matched_table = markdown_table(
        matched,
        [
            "precursor",
            "outcome",
            "treated_rows",
            "control_relation_rows",
            "treated_completion_rate",
            "matched_control_completion_rate",
            "treated_minus_control_rate",
        ],
    )
    null_table = markdown_table(
        null_summary,
        ["metric", "real_increment", "null_increment", "null_draws_exceeded"],
    )
    return f"""# Hidden-Loop Competing Routes and Registered-Loop Recurrence Quick Screen V0

Decision: `{decision["primary_decision"]}`.

- Target A precursor: `{decision["target_a_precursor_status"]}`
- Target B precursor: `{decision["target_b_precursor_status"]}`
- Target C recurrence: `{decision["target_c_recurrence_status"]}`
- Hidden 2→3→2 diversion: `{decision["hidden_2_3_2_diversion_status"]}`
- Registered-history increment: `{decision["registered_history_increment_status"]}`
- Hidden-history increment: `{decision["hidden_history_increment_status"]}`
- Protected rows materialised: `{decision["protected_rows_materialised"]}`
- Determinism: `{decision["determinism_check_passed"]}`
- Independent audit: `{decision["lightweight_audit_passed"]}`

This is retrospective, research-only, observable structural feasibility evidence. Economic and
directional outcomes stayed closed. It is not prospective validation, trading utility, or a
deployable strategy.

## Development-frozen route classes

Retained exact targets: `{json.dumps(mapping["retained_exact_targets"])}`.

Final classes: `{json.dumps(mapping["final_target_classes"])}`.

## Corrected transition census

{census_table}

## Matched transition null

{transition_table}

Multiplicity across the four fixed hypotheses:

{markdown_table(multiplicity, ["hypothesis_id", "p_value", "q_value", "q_le_0_10"])}

## C0/C1/C2 pooled assessment metrics

{pooled_table}

## Target-specific model contrasts

{contrast_table}

## Same-stage matched route comparisons

{matched_table}

## 80% whole-session bootstrap intervals

{markdown_table(boot, ["metric", "value", "lower", "upper"])}

## Five-draw hidden-history null

{null_table}

## Boundary

No account, position, order, broker, P&L, MFE, MAE, direction, entry, exit, stop, target,
portfolio-sizing, deployment, or production runtime surface was accessed or modified.
"""


def execute_screen(output: Path, *, provider_root: Path) -> dict[str, Any]:
    contract = load_contract()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    print("reconstructing predecessor population and bounded V2 state surface", flush=True)
    (
        opening,
        development_archived,
        assessment_archived,
        frozen_completions,
        states,
        dictionary,
        state_source,
        dictionary_manifest,
        b0_features,
        opening_reconstruction,
    ) = load_frozen_inputs(provider_root)
    registered = deduplicate_registered_completions(frozen_completions)
    mapping = freeze_target_class_mapping(registered.loc[registered["year"].eq(2024)])
    if not mapping["at_least_one_exact_target_supported"]:
        raise ScreenBlocker(
            "blocked_exact_target_support_failure", "no exact target passes development support"
        )
    retained_targets = cast(list[str], mapping["retained_exact_targets"])
    print("reconstructing frozen registered/hidden/prefix traces", flush=True)
    trace_registered, hidden, prefixes = build_trace_ledgers(states, dictionary)
    trace_manifest = verify_trace_identity(trace_registered, registered)
    print("running corrected six/12-bar transition census and 25 matched null draws", flush=True)
    transition_census, transition_null, multiplicity, transition_panel, transition_support_raw = (
        build_transition_screen(registered, hidden, states)
    )
    transition_support = transition_support_gate(transition_support_raw)
    print("independently reconstructing frozen B0 threshold and candidate population", flush=True)
    candidates, candidate_export, candidate_reconstruction, oof = reconstruct_candidates(
        opening,
        development_archived,
        assessment_archived,
        b0_features,
        registered,
    )
    candidates["original_target_class"] = candidates[
        "original_first_registered_semantic_loop_id"
    ].map(
        lambda value: (
            NO_REGISTERED_COMPLETION
            if pd.isna(value)
            else str(value)
            if str(value) in retained_targets
            else OTHER_REGISTERED_COMPLETION
        )
    )
    candidate_export["original_target_class"] = candidates["original_target_class"].to_numpy()
    print("building causal elapsed-bar candidate panel", flush=True)
    panel, sequential_manifest = build_sequential_panel(
        candidates,
        states,
        registered,
        hidden,
        prefixes,
        retained_targets=retained_targets,
        dictionary=dictionary,
    )
    bins = freeze_bins(panel)
    panel = apply_bins(panel, bins)
    weight_audit = sequential_weight_audit(panel)
    if (
        float(
            weight_audit.loc[weight_audit["grouping"].eq("candidate"), "absolute_difference"].max()
        )
        > 1e-12
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "candidate-normalised weights differ"
        )
    feature_sets = model_feature_sets(retained_targets)
    feature_manifest = {
        **SAFETY_FLAGS,
        "C0": list(feature_sets["C0"]),
        "C1_additions": [value for value in feature_sets["C1"] if value not in feature_sets["C0"]],
        "C2_additions": [value for value in feature_sets["C2"] if value not in feature_sets["C1"]],
        "all_preprocessing_fit_on": "2024_only",
        "all_features_available_by_update_timestamp": True,
        "original_horizon_restarted": False,
        "prefix_metadata": sequential_manifest["target_metadata"],
        "development_frozen_bins": bins,
    }
    print("fitting exactly three primary multinomial models", flush=True)
    models, predictions = fit_model_ladder(panel, feature_sets)
    model_converged = all(model.converged for model in models.values())
    metric_tables = model_metric_tables(predictions, models)
    increments = pooled_increments(metric_tables["pooled_metrics"])
    contrasts, predictions = target_specific_contrasts(predictions, models["C2"])
    matched, matched_relations = matched_candidate_comparisons(predictions)
    print("running exactly 25 fixed-prediction whole-session bootstrap draws", flush=True)
    bootstrap = bootstrap_metrics(predictions, models, matched_relations)
    print("running exactly five within-stage hidden-history null refits", flush=True)
    hidden_null, hidden_null_exceeded, null_converged = hidden_history_null_metrics(
        panel, feature_sets, models["C1"], increments
    )
    concentration, model_support = concentration_and_support(candidates, panel, retained_targets)
    statuses, gates, blocker = evaluate_decision(
        census=transition_census,
        transition_null=transition_null,
        multiplicity=multiplicity,
        transition_support=transition_support,
        model_support=model_support,
        metrics=metric_tables,
        increments=increments,
        contrasts=contrasts,
        matched=matched,
        bootstrap=bootstrap,
        hidden_null_exceeded=hidden_null_exceeded,
        converged=model_converged and null_converged,
    )
    primary_decision = choose_primary_decision(
        blocker=blocker,
        target_a_status=statuses["target_a_precursor_status"],
        target_b_status=statuses["target_b_precursor_status"],
        target_c_status=statuses["target_c_recurrence_status"],
        diversion_status=statuses["hidden_2_3_2_diversion_status"],
        hidden_increment_status=statuses["hidden_history_increment_status"],
    )
    decision_inputs = {**statuses, "blocker": blocker}
    decision: dict[str, Any] = {
        **SAFETY_FLAGS,
        "primary_decision": primary_decision,
        **statuses,
        "decision_inputs": decision_inputs,
        "decision_gates": gates,
        "transition_support": transition_support,
        "sequential_model_support": model_support,
        "increments": increments,
        "candidate_threshold": FROZEN_B0_THRESHOLD,
        "protected_rows_materialised": 0,
        "determinism_check_passed": False,
        "lightweight_audit_passed": False,
        "retrospective_only": True,
        "prospective_validation": False,
        "permission_to_trade": False,
    }
    model_configurations = {
        **SAFETY_FLAGS,
        **{
            name: {"features": list(feature_sets[name]), "model": models[name].to_json()["model"]}
            for name in MODEL_NAMES
        },
        "primary_model_fits": 3,
        "determinism_refits": 3,
        "hidden_history_null_refits": 5,
        "transition_null_draws": 25,
        "session_bootstrap_draws": 25,
    }
    coefficient_specs = {
        **SAFETY_FLAGS,
        **{name: model.to_json() for name, model in models.items()},
    }
    source_manifest = {
        **SAFETY_FLAGS,
        "development_period": ["2024-01-01", "2024-12-31"],
        "assessment_period": ["2025-01-01", "2025-08-22"],
        "minimum_timestamp_read": state_source["minimum_timestamp_read"],
        "maximum_timestamp_read": state_source["maximum_timestamp_read"],
        "protected_rows_materialised": 0,
        "date_predicate_applied_before_materialisation": True,
        "v2_state_source": state_source,
        "semantic_dictionary": dictionary_manifest,
        "trace_reconstruction": trace_manifest,
        "opening_population_reconstruction": opening_reconstruction,
        "predecessor_artifacts": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                BRIDGE_PRIMARY / "bridge_development_panel.parquet",
                BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet",
                BRIDGE_PRIMARY / "registered_completion_ledger.parquet",
                OPENING_PRIMARY / "unregistered_path_ledger.parquet",
                OPENING_PRIMARY / "hidden_family_mapping.json",
                PREDECESSOR_PRIMARY / "candidate_threshold_manifest.json",
                PREDECESSOR_PRIMARY / "decision.json",
            )
        },
    }
    protected_audit = {
        **SAFETY_FLAGS,
        "development_start": "2024-01-01",
        "development_end_inclusive": "2024-12-31",
        "assessment_start": "2025-01-01",
        "assessment_end_inclusive": "2025-08-22",
        "protected_start": "2025-08-23",
        "minimum_timestamp_read": source_manifest["minimum_timestamp_read"],
        "maximum_timestamp_read": source_manifest["maximum_timestamp_read"],
        "protected_rows_materialised": 0,
        "passed": pd.Timestamp(source_manifest["maximum_timestamp_read"]) < PROTECTED_START,
    }
    if not protected_audit["passed"]:
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "protected boundary cannot be proved"
        )
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected_audit)
    write_json(output / "target_class_mapping.json", mapping)
    write_json(
        output / "transition_hypothesis_manifest.json",
        {
            **SAFETY_FLAGS,
            "hypotheses": transition_hypothesis_manifest(),
            "multiplicity_family_size": 4,
        },
    )
    write_json(output / "candidate_population_reconstruction.json", candidate_reconstruction)
    write_json(output / "feature_manifest.json", feature_manifest)
    write_json(output / "model_configurations.json", model_configurations)
    write_json(output / "model_coefficients.json", coefficient_specs)
    write_parquet(output / "registered_completion_ledger.parquet", registered)
    write_parquet(output / "hidden_completion_ledger.parquet", hidden)
    write_parquet(output / "prefix_activity_ledger.parquet", prefixes)
    write_parquet(output / "transition_event_panel.parquet", transition_panel)
    write_csv(output / "transition_census.csv", transition_census)
    write_csv(output / "transition_null_metrics.csv", transition_null)
    write_csv(output / "transition_multiplicity.csv", multiplicity)
    write_parquet(output / "candidate_population.parquet", candidate_export)
    write_parquet(output / "b0_development_oof_predictions.parquet", oof)
    write_parquet(output / "sequential_candidate_panel.parquet", panel)
    write_csv(output / "sequential_weight_audit.csv", weight_audit)
    write_parquet(output / "assessment_predictions.parquet", predictions)
    for name, frame in metric_tables.items():
        write_csv(output / f"{name}.csv", frame)
    write_csv(output / "target_specific_contrasts.csv", contrasts)
    write_csv(output / "matched_candidate_route_metrics.csv", matched)
    write_parquet(output / "matched_candidate_relations.parquet", matched_relations)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "hidden_history_null_metrics.csv", hidden_null)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    print("performing fast deterministic refits and label reconstruction", flush=True)
    determinism = determinism_check(output)
    decision["determinism_check_passed"] = bool(determinism["passed"])
    if not determinism["passed"]:
        decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
    write_json(output / "determinism_check.json", determinism)
    write_json(output / "decision.json", decision)
    print("running independent lightweight artifact audit", flush=True)
    auditor = load_module(EXPERIMENT_DIR / "audit_screen_v0.py", "hidden_competing_routes_auditor")
    audit = cast(dict[str, Any], auditor.audit_artifacts(output, provider_root=provider_root))
    decision["lightweight_audit_passed"] = bool(audit["passed"])
    if not audit["passed"]:
        decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
    write_json(output / "lightweight_audit.json", audit)
    write_json(output / "decision.json", decision)
    plots = create_plots(transition_null, predictions, models, output)
    decision["plots"] = plots
    write_json(output / "decision.json", decision)
    report = build_report(
        decision,
        mapping,
        transition_census,
        transition_null,
        multiplicity,
        metric_tables["pooled_metrics"],
        contrasts,
        matched,
        bootstrap,
        hidden_null,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def write_blocker(output: Path, blocker: ScreenBlocker) -> None:
    output.mkdir(parents=True, exist_ok=True)
    contract = read_json(CONTRACT_PATH)
    write_json(output / "contract.json", contract)
    statuses = {
        "target_a_precursor_status": "insufficient_support",
        "target_b_precursor_status": "insufficient_support",
        "target_c_recurrence_status": "insufficient_support",
        "hidden_2_3_2_diversion_status": "insufficient_support",
        "registered_history_increment_status": "insufficient_support",
        "hidden_history_increment_status": "insufficient_support",
    }
    write_json(
        output / "decision.json",
        {
            **SAFETY_FLAGS,
            "primary_decision": blocker.code,
            **statuses,
            "decision_inputs": {**statuses, "blocker": blocker.code},
            "blocker_detail": blocker.detail,
            "protected_rows_materialised": 0,
            "determinism_check_passed": False,
            "lightweight_audit_passed": False,
        },
    )


def _clock_bin_for_bar_ordinal(ordinal: int) -> str:
    bar_start_minutes = 9 * 60 + 30 + 5 * int(ordinal)
    floored = (bar_start_minutes // 30) * 30
    return f"{floored // 60:02d}:{floored % 60:02d}"


def _refresh_existing_verification_artifacts(
    output: Path, *, provider_root: Path
) -> dict[str, Any]:
    """Tighten audit metadata without repeating fitted predictions, nulls, or intervals."""

    registered = pd.read_parquet(output / "registered_completion_ledger.parquet")
    transition_panel = pd.read_parquet(output / "transition_event_panel.parquet")
    source_clock = registered.set_index("event_id")["clock_bin"].astype(str)
    transition_panel["source_clock_bin"] = transition_panel["source_event_id"].map(source_clock)
    transition_panel["clock_bin"] = (
        transition_panel["completion_bar_ordinal"].astype(int).map(_clock_bin_for_bar_ordinal)
    )
    write_parquet(output / "transition_event_panel.parquet", transition_panel)

    predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    relations = pd.read_parquet(output / "matched_candidate_relations.parquet")
    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    multiplicities = session_bootstrap_multiplicities(
        predictions["session"].astype(str).tolist(), draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    for draw, counts in enumerate(multiplicities):
        relation_counts = _matched_relation_bootstrap_counts(predictions, relations, counts)
        mask = bootstrap["record_type"].eq("draw") & bootstrap["draw"].eq(draw)
        for name, value in relation_counts.items():
            bootstrap.loc[mask, name] = value
    write_csv(output / "bootstrap_metrics.csv", bootstrap)

    candidates = pd.read_parquet(output / "candidate_population.parquet")
    panel = pd.read_parquet(output / "sequential_candidate_panel.parquet")
    mapping = read_json(output / "target_class_mapping.json")
    concentration, _ = concentration_and_support(
        candidates, panel, cast(list[str], mapping["retained_exact_targets"])
    )
    write_csv(output / "concentration_metrics.csv", concentration)

    decision = read_json(output / "decision.json")
    inputs = cast(Mapping[str, Any], decision["decision_inputs"])
    decision["primary_decision"] = choose_primary_decision(
        blocker=cast(str | None, inputs["blocker"]),
        target_a_status=str(inputs["target_a_precursor_status"]),
        target_b_status=str(inputs["target_b_precursor_status"]),
        target_c_status=str(inputs["target_c_recurrence_status"]),
        diversion_status=str(inputs["hidden_2_3_2_diversion_status"]),
        hidden_increment_status=str(inputs["hidden_history_increment_status"]),
    )
    write_json(output / "decision.json", decision)
    determinism = determinism_check(output)
    decision["determinism_check_passed"] = bool(determinism["passed"])
    if not determinism["passed"]:
        decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
    write_json(output / "determinism_check.json", determinism)
    write_json(output / "decision.json", decision)
    auditor = load_module(EXPERIMENT_DIR / "audit_screen_v0.py", "hidden_competing_routes_auditor")
    audit = cast(dict[str, Any], auditor.audit_artifacts(output, provider_root=provider_root))
    decision["lightweight_audit_passed"] = bool(audit["passed"])
    if not audit["passed"]:
        decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
    write_json(output / "lightweight_audit.json", audit)
    write_json(output / "decision.json", decision)
    return decision


def finalize_existing_report(output: Path, *, provider_root: Path) -> dict[str, Any]:
    """Render the report from completed bounded artifacts without repeating scientific draws."""

    decision = _refresh_existing_verification_artifacts(output, provider_root=provider_root)
    report = build_report(
        decision,
        read_json(output / "target_class_mapping.json"),
        pd.read_csv(output / "transition_census.csv"),
        pd.read_csv(output / "transition_null_metrics.csv"),
        pd.read_csv(output / "transition_multiplicity.csv"),
        pd.read_csv(output / "pooled_metrics.csv"),
        pd.read_csv(output / "target_specific_contrasts.csv"),
        pd.read_csv(output / "matched_candidate_route_metrics.csv"),
        pd.read_csv(output / "bootstrap_metrics.csv"),
        pd.read_csv(output / "hidden_history_null_metrics.csv"),
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


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
    parser.add_argument(
        "--finalize-existing-report",
        action="store_true",
        help="render a report from completed artifacts without repeating bootstrap/null draws",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = arguments.output.expanduser().resolve()
    try:
        if arguments.finalize_existing_report:
            decision = finalize_existing_report(
                output, provider_root=arguments.provider_root.expanduser().resolve()
            )
        else:
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
