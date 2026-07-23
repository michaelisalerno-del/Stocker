#!/usr/bin/env python3
"""Run the repaired EODHD Fixed Overnight Options Strategy Quick Screen V0.1."""

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
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_data.calendars import get_market_calendar
from stocker_research.eodhd_fixed_options_strategy_v0 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT,
    MAXIMUM_MONTH_SHARE,
    PROTECTED_START,
    ZERO_BASED_BAR_ORDINAL,
    OptionSelectionError,
    build_matched_controls,
    expiry_intrinsic_values,
    option_position_pnl,
    previous_close_option_state,
    quote_integrity_reason,
    reject_protected_dates,
    select_atm_straddle,
    session_bootstrap_intervals,
    standard_contract_multiplier,
    strategy_metrics,
    validate_checkpoint_timing,
    validate_expiration_session,
)
from stocker_research.eodhd_fixed_options_strategy_v01 import (
    OVERALL_DECISIONS_V01,
    SAFETY_FLAGS_V01,
    STRATEGY_STATUSES_V01,
    ExactDateFilterResult,
    assert_safety_flags_v01,
    choose_overall_decision_v01,
    filter_exact_observation_date,
    validate_observation_and_expiration_boundary,
)
from stocker_research.eodhd_options_downloader_v0 import (
    CANONICAL_OPTION_COLUMNS,
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsDownloadError,
    OptionsRequest,
    OptionsResourceLimitExceeded,
    canonicalize_response_records,
    resolve_canonical_duplicates,
    stable_request_id,
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
ROUTE_RUNNER = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-hazard-quick-v0"
    / "run_screen_v0.py"
)
PRIOR_OPTION_PRICE_AUDIT = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260722-broad-conflict-prior-close-iv-v0"
    / "artifacts"
    / "primary"
    / "option_underlying_price_audit.csv"
)
PRIOR_PROBE_DIR = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260722-broad-conflict-prior-close-iv-v01-probe"
    / "artifacts"
    / "primary"
)
PRIOR_PROBE_RESULTS = PRIOR_PROBE_DIR / "contract_history_probe_results.csv"
PRIOR_PROBE_MANIFEST = PRIOR_PROBE_DIR / "contract_history_probe_manifest.json"
OPTIONS_DOWNLOAD_RUNNER = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260722-broad-conflict-prior-close-iv-v0"
    / "download_options.py"
)
STATE_CACHE = Path("/tmp/stocker-eodhd-fixed-options-strategy-v0-states.parquet")
STATE_CACHE_MANIFEST = Path("/tmp/stocker-eodhd-fixed-options-strategy-v0-states.json")
V0_OPTIONS_RUN_DIR_NAME = "fixed-overnight-options-v0"
V01_OPTIONS_RUN_DIR_NAME = "fixed-overnight-options-v01"

DEVELOPMENT_START = date(2024, 1, 1)
ASSESSMENT_START = date(2025, 1, 1)
READ_END = date(2025, 8, 22)
MAX_STATE_BAR_ORDINAL = 77
HIDDEN_2_3_2 = "unregistered_primitive_like__2-3-2"
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
STRATEGIES = ("S1", "S2_ALL", "S2_VETO", "S3")
BOOTSTRAP_STATISTICS = (
    "s1_mean_return_on_debit",
    "s1_matched_control_excess",
    "s2_all_mean_return_on_debit",
    "s2_veto_mean_return_on_debit",
    "s2_veto_minus_all_return_difference",
    "s2_veto_minus_all_win_rate_difference",
    "s3_mean_return_on_debit",
    "s3_matched_control_excess",
)


class ScreenBlocker(RuntimeError):
    """A fail-closed experiment-level blocker."""

    def __init__(
        self,
        decision: str,
        detail: str,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail
        self.evidence = dict(evidence or {})


def canonical_json(value: Any) -> str:
    """Return deterministic, human-readable JSON."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.15g")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    ordered = frame.loc[:, list(columns) if columns is not None else list(frame.columns)].copy()
    content = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def explicit_boolean(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be an explicit non-null boolean")


def strict_boolean_series(values: pd.Series, *, name: str) -> pd.Series:
    return values.map(lambda value: explicit_boolean(value, name=name)).astype(bool)


def audit_numeric(value: object) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def audit_relative_spread(row: Mapping[str, Any]) -> float:
    bid = audit_numeric(row.get("bid"))
    ask = audit_numeric(row.get("ask"))
    midpoint = audit_numeric(row.get("midpoint"))
    if not all(math.isfinite(value) for value in (bid, ask, midpoint)) or midpoint <= 0.0:
        return math.nan
    return (ask - bid) / midpoint


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", f"cannot load frozen runner: {path}"
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_contract() -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    assert_safety_flags_v01(contract)
    assert_safety_flags_v01(cast(Mapping[str, object], contract.get("safety", {})))
    if (
        tuple(contract.get("frozen_cohort", ())) != FROZEN_COHORT
        or contract.get("stock_signal", {}).get("completed_five_minute_bar_ordinal") != CHECKPOINT
        or contract.get("resource_limits", {}).get("option_structures") != 3
        or contract.get("resource_limits", {}).get("directional_veto_rules") != 1
        or contract.get("bootstrap", {}).get("draws") != BOOTSTRAP_DRAWS
        or contract.get("resource_limits", {}).get("n_jobs") != 1
        or contract.get("resource_limits", {}).get("processes") != 1
        or contract.get("resource_limits", {}).get("maximum_new_option_records") != 500_000
    ):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "frozen contract identity or limits differ"
        )
    structures = cast(list[dict[str, Any]], contract.get("structures", []))
    if [item.get("id") for item in structures] != ["S1", "S2", "S3"]:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "exactly three frozen structures required"
        )
    if set(contract.get("allowed_overall_decisions", ())) != set(OVERALL_DECISIONS_V01):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "allowed decision vocabulary differs"
        )
    if set(contract.get("allowed_strategy_statuses", ())) != set(STRATEGY_STATUSES_V01):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            "allowed strategy-status vocabulary differs",
        )
    return contract


def _validate_state_frame(states: pd.DataFrame) -> None:
    required = {
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "causal_hard_state",
        "posterior_entropy_reproduced",
        "transition_probability",
        "historical_volume_baseline_at_bar",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *(f"state_p_{state}" for state in range(8)),
    }
    if missing := sorted(required.difference(states.columns)):
        raise ScreenBlocker(
            "blocked_options_contract_reconstruction_failure",
            f"late-day state cache columns missing: {missing}",
        )
    timestamps = pd.to_datetime(states["bar_start_timestamp"], utc=True, errors="raise")
    if (
        states.duplicated(["symbol", "session", "bar_ordinal"]).any()
        or set(states["symbol"].astype(str)) != set(FROZEN_COHORT)
        or int(states["bar_ordinal"].min()) != 0
        or int(states["bar_ordinal"].max()) != MAX_STATE_BAR_ORDINAL
        or bool(timestamps.ge(pd.Timestamp(PROTECTED_START, tz="UTC")).any())
    ):
        raise ScreenBlocker(
            "blocked_protected_boundary_failure",
            "late-day state cache population, identity, or boundary differs",
        )


def load_or_build_states(
    provider_root: Path,
) -> tuple[pd.DataFrame, Any, dict[str, Any], dict[str, Any]]:
    runner = load_module(V2_RUNNER, "fixed_options_v2_runner")
    runner.MAX_TARGET_BAR_ORDINAL = MAX_STATE_BAR_ORDINAL
    cache_reused = False
    source: dict[str, Any]
    if STATE_CACHE.is_file() and STATE_CACHE_MANIFEST.is_file():
        envelope = cast(
            dict[str, Any], json.loads(STATE_CACHE_MANIFEST.read_text(encoding="utf-8"))
        )
        if (
            envelope.get("provider_root") == str(provider_root.resolve())
            and envelope.get("maximum_target_bar_ordinal") == MAX_STATE_BAR_ORDINAL
            and envelope.get("state_file_sha256") == sha256_file(STATE_CACHE)
            and envelope.get("expected_panel_hash") == str(runner.EXPECTED_PANEL_HASH)
            and envelope.get("expected_model_hash") == str(runner.EXPECTED_MODEL_HASH)
        ):
            states = pd.read_parquet(STATE_CACHE)
            source = cast(dict[str, Any], envelope["source"])
            cache_reused = True
        else:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "existing bounded state cache fingerprint differs",
            )
    else:
        preprocessing, parameters = runner.load_frozen_model()
        try:
            states, source = runner.build_v2_state_panel(provider_root, preprocessing, parameters)
        except Exception as error:
            raise ScreenBlocker(
                "blocked_options_contract_reconstruction_failure",
                f"frozen V2 state reconstruction failed: {type(error).__name__}: {error}",
            ) from error
        states = cast(pd.DataFrame, states)
        states = states.loc[states["symbol"].isin(FROZEN_COHORT)].copy()
        _validate_state_frame(states)
        write_parquet(STATE_CACHE, states)
        envelope = {
            "provider_root": str(provider_root.resolve()),
            "maximum_target_bar_ordinal": MAX_STATE_BAR_ORDINAL,
            "expected_panel_hash": str(runner.EXPECTED_PANEL_HASH),
            "expected_model_hash": str(runner.EXPECTED_MODEL_HASH),
            "state_file_sha256": sha256_file(STATE_CACHE),
            "source": source,
        }
        write_json(STATE_CACHE_MANIFEST, envelope)
    states = cast(pd.DataFrame, states)
    _validate_state_frame(states)
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
    dictionary, dictionary_manifest = runner.load_loop_dictionary()
    source = dict(source)
    source["temporary_bounded_state_cache_reused"] = cache_reused
    source["state_cache_file_sha256"] = sha256_file(STATE_CACHE)
    source["maximum_target_bar_ordinal"] = MAX_STATE_BAR_ORDINAL
    return states, dictionary, source, cast(dict[str, Any], dictionary_manifest)


def build_sparse_structural_ledger(
    states: pd.DataFrame, dictionary: Any, route: ModuleType
) -> pd.DataFrame:
    """Rebuild frozen events while retaining only route-required prefix snapshots."""

    engine = route.FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    prefix_counts = frozenset((CHECKPOINT - 3, CHECKPOINT - 1, CHECKPOINT))
    rows: list[dict[str, Any]] = []
    for (symbol, session), group in states.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
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
            canonical = route.canonical_unregistered_path(event.full_path)
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
                    "family_id": route.pool_hidden_family(
                        canonical.family_id, route.HIDDEN_FAMILIES
                    ),
                    "available_timestamp_utc": pd.Timestamp(available),
                }
            )
        event_indices = np.cumsum(changes).astype(int) - 1
        for position, bar in enumerate(ordered.itertuples(index=False)):
            completed_count = int(bar.bar_ordinal) + 1
            if completed_count not in prefix_counts:
                continue
            for prefix in trace.prefixes_after_event[int(event_indices[position])]:
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
    columns = [
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
    ]
    ledger = pd.DataFrame(rows, columns=columns).drop_duplicates()
    return ledger.sort_values(
        ["symbol", "session", "bar_ordinal", "ledger_kind", "semantic_loop_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _top_prefixes(ledger: pd.DataFrame) -> pd.DataFrame:
    current = ledger.loc[
        ledger["ledger_kind"].eq("active_prefix") & ledger["bar_ordinal"].eq(CHECKPOINT)
    ].copy()
    if current.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session",
                "top_prefix_identity",
                "top_prefix_orientation",
                "top_prefix_depth",
                "top_prefix_tie_count",
                "top_prefix_orientation_unambiguous",
            ]
        )
    denominator = (
        pd.to_numeric(current["progress_states"], errors="raise")
        + pd.to_numeric(current["transitions_remaining"], errors="raise")
        - 1
    )
    current["calculated_depth"] = np.where(
        denominator.gt(0),
        (pd.to_numeric(current["progress_states"], errors="raise") - 1) / denominator,
        0.0,
    )
    current["prefix_identity"] = (
        current["semantic_loop_id"].astype(str) + "|" + current["orientation_id"].astype(str)
    )
    rows: list[dict[str, object]] = []
    for (symbol, session), group in current.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values(
            ["calculated_depth", "prefix_identity"],
            ascending=[False, True],
            kind="mergesort",
        )
        first = ordered.iloc[0]
        maximum = float(first["calculated_depth"])
        tied = ordered.loc[
            np.isclose(
                ordered["calculated_depth"].to_numpy(dtype=float),
                maximum,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        rows.append(
            {
                "symbol": str(symbol),
                "session": str(session),
                "top_prefix_identity": str(first["prefix_identity"]),
                "top_prefix_orientation": str(first["orientation_id"]),
                "top_prefix_depth": maximum,
                "top_prefix_tie_count": len(tied),
                "top_prefix_orientation_unambiguous": (
                    len(set(tied["orientation_id"].astype(str))) == 1
                ),
            }
        )
    return pd.DataFrame(rows)


def _regular_sessions() -> tuple[pd.DataFrame, dict[str, str | None], dict[str, str | None]]:
    calendar = get_market_calendar("NYSE")
    expanded = calendar.schedule(
        start_date=pd.Timestamp(DEVELOPMENT_START) - pd.Timedelta(days=15),
        end_date=pd.Timestamp(READ_END) + pd.Timedelta(days=15),
    )
    dates = [timestamp.date() for timestamp in expanded.index]
    previous: dict[str, str | None] = {}
    following: dict[str, str | None] = {}
    for index, session_date in enumerate(dates):
        previous[session_date.isoformat()] = dates[index - 1].isoformat() if index else None
        following[session_date.isoformat()] = (
            dates[index + 1].isoformat() if index + 1 < len(dates) else None
        )
    schedule = expanded.loc[
        (expanded.index.date >= DEVELOPMENT_START) & (expanded.index.date <= READ_END)
    ].copy()
    schedule = schedule.reset_index(names="schedule_date")
    schedule["session"] = schedule["schedule_date"].map(
        lambda value: pd.Timestamp(value).date().isoformat()
    )
    schedule = schedule.loc[:, ["session", "market_open", "market_close"]]
    return schedule, previous, following


def _underlying_daily(provider_root: Path, symbol: str) -> pd.DataFrame:
    path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
    if not path.is_file():
        raise ScreenBlocker(
            "blocked_options_contract_reconstruction_failure",
            f"underlying five-minute source missing: {symbol}",
        )
    raw = pd.read_parquet(
        path,
        columns=["timestamp", "open", "close"],
        filters=[("timestamp", "<", pd.Timestamp(PROTECTED_START, tz="UTC"))],
    )
    timestamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    working = raw.assign(
        timestamp=timestamps,
        session=timestamps.dt.tz_convert("America/New_York").dt.date,
    ).sort_values("timestamp", kind="mergesort")
    daily = (
        working.groupby("session", sort=True)
        .agg(first_open=("open", "first"), unadjusted_close=("close", "last"))
        .reset_index()
    )
    daily["previous_unadjusted_close"] = daily["unadjusted_close"].shift(1)
    ratio = daily["first_open"] / daily["previous_unadjusted_close"]
    daily["inferred_corporate_action_boundary"] = ratio.lt(0.55) | ratio.gt(1.80)
    return daily


def underlying_price_audit(
    skeleton: pd.DataFrame,
    *,
    provider_root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for symbol in FROZEN_COHORT:
        path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        hashes[symbol] = sha256_file(path)
        daily = _underlying_daily(provider_root, symbol).set_index("session")
        targets = skeleton.loc[skeleton["symbol"].eq(symbol)]
        for item in targets.itertuples(index=False):
            signal_date = date.fromisoformat(str(item.session))
            selection_date = date.fromisoformat(str(item.contract_selection_date))
            exit_date = (
                date.fromisoformat(str(item.unprotected_next_session))
                if str(item.unprotected_next_session) < PROTECTED_START.isoformat()
                else None
            )
            available = (
                signal_date in daily.index
                and selection_date in daily.index
                and exit_date is not None
                and exit_date in daily.index
            )
            boundary = (
                bool(daily.loc[signal_date, "inferred_corporate_action_boundary"])
                if signal_date in daily.index
                else False
            )
            exit_boundary = (
                bool(daily.loc[exit_date, "inferred_corporate_action_boundary"])
                if exit_date is not None and exit_date in daily.index
                else False
            )
            rows.append(
                {
                    "symbol": symbol,
                    "session": signal_date.isoformat(),
                    "contract_selection_date": selection_date.isoformat(),
                    "previous_close_underlying_price": (
                        float(daily.loc[selection_date, "unadjusted_close"])
                        if available
                        else math.nan
                    ),
                    "entry_underlying_close": (
                        float(daily.loc[signal_date, "unadjusted_close"])
                        if signal_date in daily.index
                        else math.nan
                    ),
                    "exit_underlying_close": (
                        float(daily.loc[exit_date, "unadjusted_close"])
                        if exit_date is not None and exit_date in daily.index
                        else math.nan
                    ),
                    "underlying_price_source": "repository_unadjusted_eodhd_5m_close",
                    "underlying_source_available": available,
                    "inferred_split_on_signal_date": boundary,
                    "inferred_split_on_exit_date": exit_boundary,
                    "split_boundary_ambiguous": boundary or exit_boundary,
                }
            )
    result = (
        pd.DataFrame(rows)
        .sort_values(["symbol", "session"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(result) != len(skeleton):
        raise ScreenBlocker(
            "blocked_options_contract_reconstruction_failure",
            "underlying price audit population differs",
        )
    return result, hashes


def build_signal_ledger(
    states: pd.DataFrame,
    dictionary: Any,
    *,
    provider_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    route = load_module(ROUTE_RUNNER, "fixed_options_route_runner")
    route.CHECKPOINTS = (CHECKPOINT,)
    route.BASELINE_FEATURES = (
        *(
            feature
            for feature in route.BASELINE_FEATURES
            if not str(feature).startswith("checkpoint_")
        ),
        f"checkpoint_{CHECKPOINT}",
    )
    ledger = build_sparse_structural_ledger(states, dictionary, route)
    raw, exclusions, possible_rows = route.build_raw_decision_panel(states, ledger)
    scaled, component_scaling, local_scaling = route.add_development_frozen_baseline_features(raw)
    labelled, frozen_route = route.add_frozen_route_labels_and_weights(scaled)
    top = _top_prefixes(ledger)
    labelled = labelled.merge(top, on=["symbol", "session"], how="left", validate="one_to_one")
    schedule, previous, following = _regular_sessions()
    skeleton = pd.MultiIndex.from_product(
        [schedule["session"].astype(str).tolist(), FROZEN_COHORT],
        names=["session", "symbol"],
    ).to_frame(index=False)
    skeleton["contract_selection_date"] = skeleton["session"].map(previous)
    skeleton["unprotected_next_session"] = skeleton["session"].map(following)
    if skeleton["contract_selection_date"].isna().any():
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "previous US session mapping failed"
        )
    price_audit, underlying_hashes = underlying_price_audit(skeleton, provider_root=provider_root)
    skeleton = skeleton.merge(schedule, on="session", how="left", validate="many_to_one").merge(
        price_audit,
        on=["symbol", "session", "contract_selection_date"],
        how="left",
        validate="one_to_one",
    )
    keep = [
        "row_id",
        "symbol",
        "session",
        "period",
        "checkpoint",
        "checkpoint_bar_ordinal_zero_based",
        "checkpoint_timestamp_utc",
        "feature_available_timestamp_utc",
        "route_resolution_state",
        "active_prefix_count",
        "active_prefix_count_change_last_3_bars",
        "top_prefix_depth_fraction",
        "top_minus_second_prefix_depth",
        "prefix_family_entropy",
        "depth_margin_change_last_3_bars",
        "top_prefix_orientation",
        "top_prefix_identity",
        "top_prefix_tie_count",
        "top_prefix_orientation_unambiguous",
        "signed_pressure",
        "transition_probability",
        "posterior_entropy",
        "any_registered_completion_prior_6",
        "hidden_2_3_2_prior_6",
        "any_hidden_event_prior_6",
        "top_prefix_depth_quartile",
        "depth_margin_quartile",
        "prefix_family_entropy_quartile",
    ]
    output = skeleton.merge(
        labelled.loc[:, keep],
        on=["symbol", "session"],
        how="left",
        validate="one_to_one",
    )
    output["ordinal_72_structural_available"] = output["row_id"].notna()
    output["period"] = np.where(
        output["session"].lt(ASSESSMENT_START.isoformat()), "development", "assessment"
    )
    output["checkpoint"] = CHECKPOINT
    output["checkpoint_bar_ordinal_zero_based"] = ZERO_BASED_BAR_ORDINAL
    output["bars_remaining_before_close"] = np.nan
    available = output["ordinal_72_structural_available"]
    for index, item in output.loc[available].iterrows():
        output.loc[index, "bars_remaining_before_close"] = validate_checkpoint_timing(
            checkpoint=CHECKPOINT,
            zero_based_bar_ordinal=ZERO_BASED_BAR_ORDINAL,
            feature_available_timestamp=pd.Timestamp(item["feature_available_timestamp_utc"]),
            scheduled_close_timestamp=pd.Timestamp(item["market_close"]),
        )
    output["route_resolution_state"] = output["route_resolution_state"].fillna("UNAVAILABLE")
    output["top_prefix_depth"] = pd.to_numeric(output["top_prefix_depth_fraction"], errors="coerce")
    output["top_minus_second_depth"] = pd.to_numeric(
        output["top_minus_second_prefix_depth"], errors="coerce"
    )
    for label in ("BROAD_CONFLICT", "NARROWING", "DOMINANT_ROUTE", "LOW_ROUTE_SUPPORT"):
        output[label] = output["route_resolution_state"].eq(label)
    for column in (
        "any_registered_completion_prior_6",
        "hidden_2_3_2_prior_6",
        "any_hidden_event_prior_6",
    ):
        output[column] = output[column].fillna(0.0).gt(0.0)
    output["recent_registered_completion_prior_6"] = output["any_registered_completion_prior_6"]
    next_dates = pd.to_datetime(output["unprotected_next_session"], errors="coerce").dt.date
    output["exit_session"] = [
        value.isoformat() if pd.notna(value) and cast(date, value) < PROTECTED_START else None
        for value in next_dates
    ]
    output["chronology_eligible"] = output["exit_session"].notna()
    output["direction_candidate_pre_mapping"] = output["route_resolution_state"].isin(
        ["NARROWING", "DOMINANT_ROUTE"]
    ) & (output["top_prefix_orientation_unambiguous"].astype("boolean").fillna(False).astype(bool))
    output["direction_eligibility_status"] = np.where(
        output["direction_candidate_pre_mapping"],
        "blocked_direction_mapping_unavailable",
        "not_route_eligible",
    )
    output["broad_conflict_candidate_pre_iv"] = (
        output["BROAD_CONFLICT"]
        & output["ordinal_72_structural_available"]
        & output["chronology_eligible"]
        & output["underlying_source_available"].fillna(False).astype(bool)
        & ~output["split_boundary_ambiguous"].fillna(True).astype(bool)
    )
    output["cheap_iv"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    output["s1_eligible"] = False
    output["s2_all_eligible"] = False
    output["s2_veto_eligible"] = False
    output["s3_eligible"] = False
    output = output.sort_values(["session", "symbol"], kind="mergesort").reset_index(drop=True)
    reject_protected_dates(output, ["session", "contract_selection_date", "exit_session"])
    if len(output) != len(schedule) * len(FROZEN_COHORT):
        raise ScreenBlocker(
            "blocked_options_contract_reconstruction_failure",
            "stock-state ledger is not exactly stock × regular session",
        )
    manifest = {
        "possible_source_rows": possible_rows,
        "structural_rows_available": int(available.sum()),
        "structural_rows_unavailable": int((~available).sum()),
        "structural_exclusions": exclusions["reason"].value_counts().sort_index().to_dict(),
        "component_scaling": component_scaling,
        "local_scaling_stock_clock_groups": len(local_scaling),
        "frozen_route": frozen_route,
        "sparse_structural_ledger_rows": len(ledger),
        "underlying_source_sha256": underlying_hashes,
        "checkpoint_prefix_snapshots_retained": [
            CHECKPOINT - 3,
            CHECKPOINT - 1,
            CHECKPOINT,
        ],
    }
    return output, manifest, ledger


def direction_mapping_audit() -> dict[str, Any]:
    evidence_paths = [
        REPO_ROOT
        / "research"
        / "route-competition"
        / "20260722-route-competition-hazard-quick-v0"
        / "contract.json",
        REPO_ROOT
        / "research"
        / "registered-loop-routes"
        / "20260722-hidden-loop-competing-routes-v0"
        / "contract.json",
        REPO_ROOT
        / "research"
        / "slrno-v2"
        / "20260714-regime-loop-handoff"
        / "work"
        / "artifacts"
        / "20260718-loop-event-semantics-v2"
        / "primary"
        / "implementation_census.csv",
    ]
    evidence = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for path in evidence_paths
        if path.is_file()
    ]
    return {
        **SAFETY_FLAGS_V01,
        "audited_orientation_to_price_direction_mapping_available": False,
        "audited_mapping": {},
        "directional_strategy_blocker": "blocked_direction_mapping_unavailable",
        "reason": (
            "frozen orientation IDs encode structural path orientation; no audited artifact "
            "maps them one-to-one to bullish or bearish price direction"
        ),
        "direction_invented": False,
        "evidence": evidence,
    }


def required_option_dates(signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = signals.loc[signals["broad_conflict_candidate_pre_iv"].astype(bool)]
    for item in candidates.itertuples(index=False):
        for strategy, scope in (
            (
                "S1",
                "nearest expiry with entry DTE 7-14; one common ATM call/put strike",
            ),
            ("S3", "expiry with entry DTE 1; one common ATM call/put strike"),
        ):
            for role, option_date in (
                ("contract_preselection", item.contract_selection_date),
                ("entry_quote", item.session),
                ("exit_quote", item.exit_session),
            ):
                rows.append(
                    {
                        "symbol": str(item.symbol),
                        "signal_session": str(item.session),
                        "strategy": strategy,
                        "option_date": str(option_date),
                        "role": role,
                        "required_contract_scope": scope,
                    }
                )
    frame = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "signal_session",
            "strategy",
            "option_date",
            "role",
            "required_contract_scope",
        ],
    )
    if not frame.empty:
        reject_protected_dates(frame, ["signal_session", "option_date"])
        frame = frame.sort_values(
            ["option_date", "symbol", "signal_session", "strategy", "role"],
            kind="mergesort",
        ).reset_index(drop=True)
    return frame


_OCC_ID = re.compile(r"^[A-Z]{1,6}(\d{6})[CP]\d{8}$")


def _contract_expiration(contract_id: str) -> date | None:
    match = _OCC_ID.fullmatch(contract_id)
    if match is None:
        return None
    encoded = match.group(1)
    try:
        return date(2000 + int(encoded[:2]), int(encoded[2:4]), int(encoded[4:6]))
    except ValueError:
        return None


def bounded_download_cache_summary(
    options_cache: Path,
) -> tuple[dict[str, int], set[tuple[str, str, str]]]:
    """Audit the experiment-specific raw cache and exact-chain receipts."""

    data_dir = options_cache / "fixed-overnight-options-v0"
    completed_dir = data_dir / "manifests" / "completed"
    manifest_paths = sorted(completed_dir.glob("*.json")) if completed_dir.is_dir() else []
    manifest_rows: list[Mapping[str, object]] = []
    for path in manifest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["manifest_rows"]
        except (KeyError, json.JSONDecodeError, OSError) as error:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"bounded option request manifest is unreadable: {path}",
            ) from error
        if not isinstance(rows, list):
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"bounded option request manifest rows are invalid: {path}",
            )
        manifest_rows.extend(
            cast(list[Mapping[str, object]], [row for row in rows if isinstance(row, Mapping)])
        )

    logical_records = 0
    logical_bytes = 0
    unique_raw: dict[str, Path] = {}
    data_root = data_dir.resolve()
    for row in manifest_rows:
        response_hash = str(row.get("response_hash", ""))
        cache_path_value = row.get("cache_path")
        if not response_hash or not isinstance(cache_path_value, str):
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "bounded option request manifest lacks raw-cache identity",
            )
        cache_path = Path(cache_path_value).resolve()
        if not cache_path.is_relative_to(data_root) or not cache_path.is_file():
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "bounded option raw-cache path is missing or outside the experiment cache",
            )
        if sha256_file(cache_path) != response_hash:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"bounded option raw-cache hash differs: {cache_path}",
            )
        logical_records += int(row.get("record_count", 0))
        logical_bytes += cache_path.stat().st_size
        unique_raw.setdefault(response_hash, cache_path)

    receipt_path = data_dir / "bounded_download_manifest.json"
    queries: list[Mapping[str, object]] = []
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            query_value = receipt.get("queries", [])
        except (json.JSONDecodeError, OSError) as error:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "bounded option download receipt is unreadable",
            ) from error
        if not isinstance(query_value, list):
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "bounded option download receipt query rows are invalid",
            )
        queries = cast(
            list[Mapping[str, object]],
            [row for row in query_value if isinstance(row, Mapping)],
        )

    complete_coverage: set[tuple[str, str, str]] = set()
    for row in queries:
        if row.get("status") != "complete":
            continue
        option_date = date.fromisoformat(str(row["option_date"]))
        strategy = str(row["strategy"])
        if option_date >= PROTECTED_START or strategy not in {"S1", "S3"}:
            raise ScreenBlocker(
                "blocked_protected_boundary_failure",
                "bounded option receipt escaped the protected date or frozen strategies",
            )
        complete_coverage.add((str(row["symbol"]), option_date.isoformat(), strategy))

    return (
        {
            "bounded_request_manifests": len(manifest_paths),
            "bounded_manifest_rows": len(manifest_rows),
            "bounded_cached_provider_records": logical_records,
            "bounded_cached_logical_response_bytes": logical_bytes,
            "bounded_unique_raw_responses": len(unique_raw),
            "bounded_unique_raw_bytes": sum(path.stat().st_size for path in unique_raw.values()),
            "bounded_complete_queries": len(complete_coverage),
            "bounded_blocked_queries": sum(row.get("status") == "blocked" for row in queries),
        },
        complete_coverage,
    )


@dataclass(frozen=True)
class PlannedOptionRequest:
    """One strategy-specific bounded chain request with stable identities."""

    symbol: str
    option_date: date
    strategy: str
    roles: tuple[str, ...]
    signal_sessions: tuple[str, ...]
    request: OptionsRequest
    request_id: str
    resume_id: str


class CachedPaginationFailure(ValueError):
    """A completed-cache manifest cannot prove pagination completion."""


class CachedResponseSchemaFailure(ValueError):
    """A completed-cache response cannot be parsed safely."""


def plan_option_requests(
    required: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    page_limit: int = 1000,
) -> list[PlannedOptionRequest]:
    """Rebuild the exact frozen request identities used by V0."""

    signals_by_key = {
        (str(row.symbol), str(row.session)): row for row in signals.itertuples(index=False)
    }
    planned: list[PlannedOptionRequest] = []
    groups = required.groupby(["symbol", "option_date", "strategy"], sort=True)
    for (symbol_value, option_date_value, strategy_value), group in groups:
        symbol = str(symbol_value)
        strategy = str(strategy_value)
        if strategy not in {"S1", "S3"}:
            continue
        option_date = date.fromisoformat(str(option_date_value))
        if option_date >= PROTECTED_START:
            raise ScreenBlocker(
                "blocked_protected_boundary_failure",
                "bounded option request reached a protected observation date",
            )
        source_rows = [
            signals_by_key[(symbol, str(session))]
            for session in sorted(set(group["signal_session"].astype(str)))
        ]
        previous_closes = [float(row.previous_close_underlying_price) for row in source_rows]
        entry_dates = [date.fromisoformat(str(row.session)) for row in source_rows]
        if strategy == "S1":
            expiration_from = cast(date, min(value + pd.Timedelta(days=7) for value in entry_dates))
            expiration_to = cast(date, max(value + pd.Timedelta(days=14) for value in entry_dates))
        else:
            expiration_from = cast(date, min(value + pd.Timedelta(days=1) for value in entry_dates))
            expiration_to = cast(date, max(value + pd.Timedelta(days=1) for value in entry_dates))
        request = OptionsRequest(
            underlying_symbol=symbol,
            trade_date_from=option_date,
            trade_date_to=option_date,
            strike_from=0.70 * min(previous_closes),
            strike_to=1.30 * max(previous_closes),
            expiration_from=expiration_from,
            expiration_to=expiration_to,
            compact=False,
        )
        request_id = hashlib.sha256(
            canonical_json(
                {
                    "symbol": symbol,
                    "option_date": option_date,
                    "strategy": strategy,
                    "strike_from": request.strike_from,
                    "strike_to": request.strike_to,
                    "expiration_from": expiration_from,
                    "expiration_to": expiration_to,
                }
            ).encode("utf-8")
        ).hexdigest()
        resume_id = stable_request_id(
            request.endpoint,
            request.parameters(offset=0, limit=page_limit),
        )
        planned.append(
            PlannedOptionRequest(
                symbol=symbol,
                option_date=option_date,
                strategy=strategy,
                roles=tuple(sorted(set(group["role"].astype(str)))),
                signal_sessions=tuple(sorted(set(group["signal_session"].astype(str)))),
                request=request,
                request_id=request_id,
                resume_id=resume_id,
            )
        )
    return planned


def _next_offset(links: Mapping[str, Any]) -> int | None:
    value = links.get("next")
    if value in {None, ""}:
        return None
    if not isinstance(value, str):
        raise CachedPaginationFailure("pagination next link is not a string")
    offsets = parse_qs(urlparse(value).query).get("page[offset]")
    if offsets is None or len(offsets) != 1:
        raise CachedPaginationFailure("pagination next link lacks one offset")
    try:
        result = int(offsets[0])
    except ValueError as error:
        raise CachedPaginationFailure("pagination next offset is invalid") from error
    if result < 0:
        raise CachedPaginationFailure("pagination next offset is negative")
    return result


def _read_completed_cached_response(
    manifest_path: Path,
    *,
    allowed_cache_root: Path,
) -> tuple[list[Mapping[str, Any]], str]:
    """Read and independently validate every page in one completed request."""

    try:
        envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_rows = envelope["manifest_rows"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CachedResponseSchemaFailure("completed request manifest is unreadable") from error
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise CachedResponseSchemaFailure("completed request manifest has no pages")
    ordered = sorted(
        (cast(Mapping[str, Any], row) for row in manifest_rows if isinstance(row, Mapping)),
        key=lambda row: int(row.get("offset", -1)),
    )
    if len(ordered) != len(manifest_rows):
        raise CachedResponseSchemaFailure("completed request manifest page schema differs")

    records: list[Mapping[str, Any]] = []
    response_hashes: list[str] = []
    expected_offset = 0
    expected_total: int | None = None
    pagination_mode: bool | None = None
    for index, row in enumerate(ordered):
        try:
            row_offset = int(row["offset"])
            row_limit = int(row["limit"])
            recorded_count = int(row["record_count"])
            response_hash = str(row["response_hash"])
            raw_path = Path(str(row["cache_path"])).resolve()
        except (KeyError, TypeError, ValueError) as error:
            raise CachedResponseSchemaFailure("completed request page metadata differ") from error
        if row_offset != expected_offset or row_limit < 1:
            raise CachedPaginationFailure("cached pagination offsets are not contiguous")
        if not raw_path.is_relative_to(allowed_cache_root.resolve()) or not raw_path.is_file():
            raise CachedResponseSchemaFailure("cached raw response is missing or out of scope")
        raw_content = raw_path.read_bytes()
        if hashlib.sha256(raw_content).hexdigest() != response_hash:
            raise CachedResponseSchemaFailure("cached raw response hash differs")
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as error:
            raise CachedResponseSchemaFailure("cached raw response is not JSON") from error
        if (
            not isinstance(payload, Mapping)
            or not isinstance(payload.get("data"), list)
            or not isinstance(payload.get("meta"), Mapping)
            or not isinstance(payload.get("links"), Mapping)
        ):
            raise CachedResponseSchemaFailure("cached raw response schema differs")
        page_records = cast(list[Mapping[str, Any]], payload["data"])
        meta = cast(Mapping[str, Any], payload["meta"])
        links = cast(Mapping[str, Any], payload["links"])
        try:
            meta_offset = int(meta["offset"])
            meta_limit = int(meta["limit"])
        except (KeyError, TypeError, ValueError) as error:
            raise CachedPaginationFailure("cached pagination metadata are invalid") from error
        if meta_offset != row_offset or meta_limit != row_limit:
            raise CachedPaginationFailure("cached pagination metadata changed")
        if len(page_records) != recorded_count or len(page_records) > row_limit:
            raise CachedPaginationFailure("cached page record count differs")
        has_total = meta.get("total") is not None
        if pagination_mode is None:
            pagination_mode = has_total
        elif pagination_mode != has_total:
            raise CachedPaginationFailure("cached pagination mode changed")
        if has_total:
            try:
                total = int(meta["total"])
            except (TypeError, ValueError) as error:
                raise CachedPaginationFailure("cached pagination total is invalid") from error
            if total < 0 or (expected_total is not None and total != expected_total):
                raise CachedPaginationFailure("cached pagination total changed")
            expected_total = total
        next_offset = _next_offset(links)
        final_page = index == len(ordered) - 1
        if final_page and next_offset is not None:
            raise CachedPaginationFailure("cached pagination is incomplete")
        if not final_page and next_offset != row_offset + row_limit:
            raise CachedPaginationFailure("cached next offset is not contiguous")
        expected_offset = row_offset + row_limit
        records.extend(page_records)
        response_hashes.append(response_hash)
    if expected_total is not None and len(records) != expected_total:
        raise CachedPaginationFailure("cached response is truncated before total")
    logical_hash = (
        response_hashes[0]
        if len(response_hashes) == 1
        else hashlib.sha256(
            json.dumps(response_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return records, logical_hash


def _exact_date_audit_row(
    planned: PlannedOptionRequest,
    result: ExactDateFilterResult,
) -> dict[str, object]:
    return {
        "request_id": planned.request_id,
        "symbol": planned.symbol,
        "contract_id": "",
        "strategy": planned.strategy,
        "requested_observation_date": planned.option_date.isoformat(),
        "returned_observation_dates": json.dumps(
            [value.isoformat() for value in result.returned_observation_dates],
            separators=(",", ":"),
        ),
        "requested_date_present": result.requested_date_present,
        "exact_date_record_count": result.exact_date_record_count,
        "discarded_other_date_record_count": result.discarded_other_date_record_count,
        "discarded_post_boundary_record_count": result.discarded_post_boundary_record_count,
        "pagination_complete": result.pagination_complete,
        "response_hash": result.response_hash,
        "exact_date_hash": result.exact_date_hash,
        "discarded_fragment_hash": result.discarded_fragment_hash,
        "status": result.status,
    }


def _canonicalise_exact_records(
    planned: PlannedOptionRequest,
    result: ExactDateFilterResult,
) -> tuple[pd.DataFrame, int, int, int, int]:
    if result.status not in {"exact_date_complete", "extra_dates_discarded"}:
        return pd.DataFrame(), 0, 0, 0, 0
    normalised_records: list[Mapping[str, Any]] = []
    original_hash_by_identity: dict[tuple[str, date], str] = {}
    provider_dte_disagreements = 0
    unquotable_records_discarded = 0
    for record in result.retained_records:
        attributes_value = record.get("attributes")
        if not isinstance(attributes_value, Mapping):
            normalised_records.append(record)
            continue
        attributes = dict(attributes_value)
        if any(attributes.get(field) is None for field in ("bid", "ask", "bid_date", "ask_date")):
            unquotable_records_discarded += 1
            continue
        resource_id = record.get("id")
        contract_id = attributes.get("contract")
        expiration_value = attributes.get("exp_date")
        if (
            isinstance(resource_id, str)
            and isinstance(contract_id, str)
            and isinstance(expiration_value, str)
        ):
            try:
                observation_date = date.fromisoformat(resource_id[-10:])
                expiration_date = date.fromisoformat(expiration_value[:10])
                provider_dte_value = attributes.get("dte")
                if (
                    isinstance(provider_dte_value, (int, float))
                    and not isinstance(provider_dte_value, bool)
                    and float(provider_dte_value).is_integer()
                    and int(provider_dte_value) != (expiration_date - observation_date).days
                ):
                    provider_dte_disagreements += 1
                    attributes["dte"] = None
                original_hash_by_identity[(contract_id, observation_date)] = hashlib.sha256(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
            except ValueError:
                pass
        normalised_records.append({**record, "attributes": attributes})
    canonical = canonicalize_response_records(
        normalised_records,
        request_id=planned.request_id,
        provider_schema_version="openapi-2.0.0-exact-date-repair-v01",
    )
    if canonical.rejections:
        return (
            pd.DataFrame(),
            len(canonical.rejections),
            0,
            provider_dte_disagreements,
            unquotable_records_discarded,
        )
    for row in canonical.records:
        identity = (str(row["contract_id"]), cast(date, row["trade_date"]))
        if identity in original_hash_by_identity:
            row["raw_record_hash"] = original_hash_by_identity[identity]
    deduplicated = resolve_canonical_duplicates(canonical.records)
    if deduplicated.conflicting_duplicate_groups:
        return (
            pd.DataFrame(),
            0,
            int(deduplicated.conflicting_duplicate_groups),
            provider_dte_disagreements,
            unquotable_records_discarded,
        )
    stored: list[dict[str, Any]] = []
    for row in deduplicated.records:
        trade_date = cast(date, row["trade_date"])
        expiration_date = cast(date, row["expiration_date"])
        if trade_date != planned.option_date:
            raise ScreenBlocker(
                "blocked_exact_date_filter_failure",
                "canonical record escaped exact requested-date isolation",
            )
        validate_observation_and_expiration_boundary(
            observation_date=trade_date,
            expiration_date=expiration_date,
        )
        contract_id = str(row["contract_id"])
        try:
            multiplier = standard_contract_multiplier(
                contract_id,
                underlying_symbol=planned.symbol,
                strike=float(row["strike"]),
                adjusted_contract=False,
                deliverable_resolved=True,
            )
            adjusted = False
            deliverable = True
        except ValueError:
            multiplier = 0
            adjusted = True
            deliverable = False
        stored.append(
            {
                **row,
                "adjusted_contract": adjusted,
                "deliverable_resolved": deliverable,
                "contract_multiplier": multiplier,
                "settlement_style": "standard_equity_pm" if deliverable else "ambiguous",
                "chain_complete": True,
                "cache_source": "v01_exact_date_repair",
                "request_strategy": planned.strategy,
            }
        )
    return (
        pd.DataFrame(stored),
        0,
        0,
        provider_dte_disagreements,
        unquotable_records_discarded,
    )


def reprocess_cached_responses(
    *,
    required: pd.DataFrame,
    signals: pd.DataFrame,
    options_cache: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Recover V0/V0.1 raw responses through exact-date pre-materialisation filtering."""

    plans = plan_option_requests(required, signals)
    predecessor_receipt_path = (
        options_cache / V0_OPTIONS_RUN_DIR_NAME / "bounded_download_manifest.json"
    )
    predecessor_queries: dict[str, Mapping[str, object]] = {}
    if predecessor_receipt_path.is_file():
        envelope = json.loads(predecessor_receipt_path.read_text(encoding="utf-8"))
        rows = envelope.get("queries", []) if isinstance(envelope, Mapping) else []
        if isinstance(rows, list):
            predecessor_queries = {
                str(row["request_id"]): cast(Mapping[str, object], row)
                for row in rows
                if isinstance(row, Mapping) and isinstance(row.get("request_id"), str)
            }

    exact_audit_rows: list[dict[str, object]] = []
    canonical_frames: list[pd.DataFrame] = []
    receipt_rows: list[dict[str, object]] = []
    examined = 0
    with_requested_date = 0
    recovered = 0
    recovered_records = 0
    extra_discarded = 0
    post_boundary_discarded = 0
    previously_missing_recovered = 0
    canonical_rejections = 0
    conflicting_groups = 0
    provider_dte_disagreements = 0
    unquotable_records_discarded = 0
    sources_used: dict[str, int] = {V0_OPTIONS_RUN_DIR_NAME: 0, V01_OPTIONS_RUN_DIR_NAME: 0}
    raw_records_by_source: dict[str, int] = {
        V0_OPTIONS_RUN_DIR_NAME: 0,
        V01_OPTIONS_RUN_DIR_NAME: 0,
    }
    exact_records_by_source: dict[str, int] = {
        V0_OPTIONS_RUN_DIR_NAME: 0,
        V01_OPTIONS_RUN_DIR_NAME: 0,
    }

    for planned in plans:
        manifest_path: Path | None = None
        source_name = ""
        for candidate_source in (V01_OPTIONS_RUN_DIR_NAME, V0_OPTIONS_RUN_DIR_NAME):
            candidate = (
                options_cache
                / candidate_source
                / "manifests"
                / "completed"
                / f"{planned.resume_id}.json"
            )
            if candidate.is_file():
                manifest_path = candidate
                source_name = candidate_source
                break
        if manifest_path is None:
            continue
        examined += 1
        sources_used[source_name] += 1
        raw_record_count = 0
        try:
            records, response_hash = _read_completed_cached_response(
                manifest_path,
                allowed_cache_root=options_cache,
            )
            raw_record_count = len(records)
            result = filter_exact_observation_date(
                records,
                requested_observation_date=planned.option_date,
                response_hash=response_hash,
                pagination_complete=True,
            )
        except CachedPaginationFailure:
            result = filter_exact_observation_date(
                (),
                requested_observation_date=planned.option_date,
                response_hash=sha256_file(manifest_path),
                pagination_complete=False,
            )
        except CachedResponseSchemaFailure:
            empty = filter_exact_observation_date(
                (),
                requested_observation_date=planned.option_date,
                response_hash=sha256_file(manifest_path),
                pagination_complete=True,
            )
            result = ExactDateFilterResult(
                status="schema_failure",
                retained_records=(),
                returned_observation_dates=(),
                requested_date_present=False,
                exact_date_record_count=0,
                discarded_other_date_record_count=0,
                discarded_post_boundary_record_count=0,
                pagination_complete=True,
                response_hash=empty.response_hash,
                exact_date_hash=empty.exact_date_hash,
                discarded_fragment_hash=empty.discarded_fragment_hash,
            )
        audit_row = _exact_date_audit_row(planned, result)
        raw_records_by_source[source_name] += raw_record_count
        exact_records_by_source[source_name] += result.exact_date_record_count
        (
            frame,
            rejected,
            conflicts,
            dte_disagreements,
            unquotable_discarded,
        ) = _canonicalise_exact_records(planned, result)
        canonical_rejections += rejected
        conflicting_groups += conflicts
        provider_dte_disagreements += dte_disagreements
        unquotable_records_discarded += unquotable_discarded
        audit_row["provider_dte_disagreement_record_count"] = dte_disagreements
        audit_row["unquotable_exact_date_record_count"] = unquotable_discarded
        if rejected or conflicts:
            audit_row["status"] = "schema_failure"
            frame = pd.DataFrame()
        exact_audit_rows.append(audit_row)
        safe = audit_row["status"] in {"exact_date_complete", "extra_dates_discarded"}
        if result.requested_date_present:
            with_requested_date += 1
        if safe:
            recovered += 1
            recovered_records += len(frame)
            if not frame.empty:
                canonical_frames.append(frame)
            previous = predecessor_queries.get(planned.request_id)
            if previous is not None and previous.get("status") != "complete":
                previously_missing_recovered += 1
        extra_discarded += result.discarded_other_date_record_count
        post_boundary_discarded += result.discarded_post_boundary_record_count
        receipt_rows.append(
            {
                "request_id": planned.request_id,
                "resume_id": planned.resume_id,
                "symbol": planned.symbol,
                "option_date": planned.option_date.isoformat(),
                "strategy": planned.strategy,
                "roles": ";".join(planned.roles),
                "signal_sessions": len(planned.signal_sessions),
                "cache_source": source_name,
                "status": audit_row["status"],
                "exact_date_records": len(frame),
                "network_request_required": False,
            }
        )
    if canonical_frames:
        combined_columns = list(
            dict.fromkeys(column for frame in canonical_frames for column in frame.columns)
        )
        compact_frames = [
            frame.loc[:, [column for column in frame if not frame[column].isna().all()]]
            for frame in canonical_frames
        ]
        combined = pd.concat(compact_frames, ignore_index=True, sort=False)
        for column in combined_columns:
            if column not in combined:
                combined[column] = pd.NA
        combined = combined.loc[:, combined_columns]
        deduplicated = resolve_canonical_duplicates(
            cast(list[Mapping[str, Any]], combined.to_dict(orient="records"))
        )
        conflicting_groups += int(deduplicated.conflicting_duplicate_groups)
        if deduplicated.conflicting_duplicate_groups:
            raise ScreenBlocker(
                "blocked_exact_date_filter_failure",
                "exact-date cache conflicts after deterministic deduplication",
            )
        canonical_records = pd.DataFrame(deduplicated.records)
        canonical_records = canonical_records.sort_values(
            ["underlying_symbol", "trade_date", "expiration_date", "strike", "option_type"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        canonical_records = pd.DataFrame(columns=[*CANONICAL_OPTION_COLUMNS])
    exact_audit = pd.DataFrame(
        exact_audit_rows,
        columns=[
            "request_id",
            "symbol",
            "contract_id",
            "strategy",
            "requested_observation_date",
            "returned_observation_dates",
            "requested_date_present",
            "exact_date_record_count",
            "discarded_other_date_record_count",
            "discarded_post_boundary_record_count",
            "pagination_complete",
            "response_hash",
            "exact_date_hash",
            "discarded_fragment_hash",
            "provider_dte_disagreement_record_count",
            "unquotable_exact_date_record_count",
            "status",
        ],
    )
    receipts = pd.DataFrame(receipt_rows)
    summary: dict[str, object] = {
        "cached_responses_examined": examined,
        "cached_responses_with_requested_date": with_requested_date,
        "cached_responses_recovered": recovered,
        "cached_exact_date_records_recovered": recovered_records,
        "extra_date_records_discarded": extra_discarded,
        "post_boundary_records_discarded": post_boundary_discarded,
        "previously_missing_chains_recovered": previously_missing_recovered,
        "canonical_rejections": canonical_rejections,
        "conflicting_duplicate_groups": conflicting_groups,
        "provider_dte_disagreement_records": provider_dte_disagreements,
        "unquotable_exact_date_records_discarded": unquotable_records_discarded,
        "cache_sources_examined": sources_used,
        "raw_provider_records_examined_by_source": raw_records_by_source,
        "exact_date_records_by_source": exact_records_by_source,
        "protected_option_observations_materialised": 0,
    }
    return canonical_records, exact_audit, receipts, summary


def unresolved_option_requests(
    required: pd.DataFrame,
    signals: pd.DataFrame,
    exact_audit: pd.DataFrame,
) -> list[PlannedOptionRequest]:
    """Return only requests with no terminal cached exact-date outcome."""

    terminal_ids = set(
        exact_audit.loc[
            exact_audit["status"].isin(
                {
                    "exact_date_complete",
                    "extra_dates_discarded",
                    "exact_date_absent",
                    "ambiguous_date_mapping",
                    "incomplete_pagination",
                    "schema_failure",
                }
            ),
            "request_id",
        ].astype(str)
    )
    return [
        planned
        for planned in plan_option_requests(required, signals)
        if planned.request_id not in terminal_ids
    ]


def v01_download_inventory(options_cache: Path) -> dict[str, int]:
    """Count cumulative V0.1 acquisition across resumable invocations."""

    data_dir = options_cache / V01_OPTIONS_RUN_DIR_NAME
    manifest_paths = sorted((data_dir / "manifests" / "completed").glob("*.json"))
    logical_records = 0
    logical_bytes = 0
    manifest_page_rows = 0
    raw_by_hash: dict[str, Path] = {}
    for path in manifest_paths:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        rows = envelope.get("manifest_rows", [])
        if not isinstance(rows, list) or not rows:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "V0.1 completed download manifest is invalid",
            )
        for value in rows:
            if not isinstance(value, Mapping):
                raise ScreenBlocker(
                    "blocked_reproducibility_or_audit_failure",
                    "V0.1 completed download page row is invalid",
                )
            raw_path = Path(str(value["cache_path"])).resolve()
            response_hash = str(value["response_hash"])
            if (
                not raw_path.is_relative_to(data_dir.resolve())
                or not raw_path.is_file()
                or sha256_file(raw_path) != response_hash
            ):
                raise ScreenBlocker(
                    "blocked_reproducibility_or_audit_failure",
                    "V0.1 raw response identity or scope differs",
                )
            manifest_page_rows += 1
            logical_records += int(value["record_count"])
            logical_bytes += raw_path.stat().st_size
            raw_by_hash.setdefault(response_hash, raw_path)
    unique_bytes = sum(path.stat().st_size for path in raw_by_hash.values())
    if logical_records > 500_000 or unique_bytes > 5 * 1024**3:
        raise ScreenBlocker(
            "blocked_quick_options_strategy_resource_limit",
            "cumulative V0.1 option acquisition exceeded a frozen resource limit",
        )
    return {
        "new_logical_requests": len(manifest_paths),
        "new_manifest_page_rows": manifest_page_rows,
        "newly_downloaded_records": logical_records,
        "newly_downloaded_logical_bytes": logical_bytes,
        "new_unique_raw_responses": len(raw_by_hash),
        "newly_downloaded_unique_raw_bytes": unique_bytes,
    }


def option_gaps_after_repair(
    required: pd.DataFrame,
    signals: pd.DataFrame,
    exact_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Classify every required request without treating exact absence as leakage."""

    status_by_request = (
        exact_audit.set_index("request_id")["status"].astype(str).to_dict()
        if not exact_audit.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for planned in plan_option_requests(required, signals):
        status = status_by_request.get(planned.request_id, "response_missing")
        if status in {"exact_date_complete", "extra_dates_discarded"}:
            continue
        rows.append(
            {
                "request_id": planned.request_id,
                "symbol": planned.symbol,
                "option_date": planned.option_date.isoformat(),
                "roles": ";".join(planned.roles),
                "strategies": planned.strategy,
                "signal_sessions": len(planned.signal_sessions),
                "exact_date_status": status,
                "terminal_provider_absence": status == "exact_date_absent",
                "download_still_required": status == "response_missing",
                "gap_reason": status,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "request_id",
            "symbol",
            "option_date",
            "roles",
            "strategies",
            "signal_sessions",
            "exact_date_status",
            "terminal_provider_absence",
            "download_still_required",
            "gap_reason",
        ],
    )


def download_unresolved_option_requests(
    *,
    token: str,
    plans: Sequence[PlannedOptionRequest],
    options_cache: Path,
) -> dict[str, object]:
    """Download only cache-missing plans and classify each exact-date outcome."""

    data_dir = options_cache / V01_OPTIONS_RUN_DIR_NAME
    receipt_path = data_dir / "bounded_download_manifest.json"
    if not plans:
        outcome = {
            **SAFETY_FLAGS_V01,
            "status": "not_required",
            "requests_considered": 0,
            "network_requests_made": 0,
            "newly_downloaded_records": 0,
            "newly_downloaded_bytes": 0,
            "queries": [],
            "credential_recorded": False,
        }
        write_json(receipt_path, outcome)
        return outcome
    if not token:
        return {
            **SAFETY_FLAGS_V01,
            "status": "token_unavailable",
            "requests_considered": len(plans),
            "network_requests_made": 0,
            "newly_downloaded_records": 0,
            "newly_downloaded_bytes": 0,
            "queries": [],
            "credential_recorded": False,
        }

    transport_module = load_module(OPTIONS_DOWNLOAD_RUNNER, "fixed_options_v01_transport")
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=data_dir,
            page_limit=1000,
            requests_per_minute=600,
            maximum_raw_records=500_000,
            maximum_download_bytes=5 * 1024**3,
        ),
        transport=transport_module.RequestsTransport(),
    )
    query_rows: list[dict[str, object]] = []
    network_requests = 0
    network_records = 0
    network_bytes = 0
    stopped_by_resource_limit = False
    stopped_by_download_failure = False

    def current_outcome(status: str) -> dict[str, object]:
        return {
            **SAFETY_FLAGS_V01,
            "status": status,
            "requests_considered": len(plans),
            "requests_completed": len(query_rows),
            "network_requests_made": network_requests,
            "newly_downloaded_records": network_records,
            "newly_downloaded_bytes": network_bytes,
            "stopped_by_resource_limit": stopped_by_resource_limit,
            "stopped_by_download_failure": stopped_by_download_failure,
            "queries": query_rows,
            "credential_recorded": False,
        }

    for planned in plans:
        manifest_path = data_dir / "manifests" / "completed" / f"{planned.resume_id}.json"
        request_was_cached = manifest_path.is_file()
        try:
            result = downloader.download(planned.request)
        except OptionsResourceLimitExceeded as error:
            stopped_by_resource_limit = True
            query_rows.append(
                {
                    "request_id": planned.request_id,
                    "resume_id": planned.resume_id,
                    "symbol": planned.symbol,
                    "option_date": planned.option_date.isoformat(),
                    "strategy": planned.strategy,
                    "status": "blocked_resource_limit",
                    "detail": str(error).replace(token, "[REDACTED]"),
                }
            )
            write_json(receipt_path, current_outcome("blocked_resource_limit"))
            break
        except OptionsDownloadError as error:
            stopped_by_download_failure = True
            query_rows.append(
                {
                    "request_id": planned.request_id,
                    "resume_id": planned.resume_id,
                    "symbol": planned.symbol,
                    "option_date": planned.option_date.isoformat(),
                    "strategy": planned.strategy,
                    "status": "download_failure",
                    "detail": str(error).replace(token, "[REDACTED]"),
                }
            )
            write_json(receipt_path, current_outcome("download_failure"))
            break
        result_bytes = sum(Path(row.cache_path).stat().st_size for row in result.manifest_rows)
        if not request_was_cached:
            network_requests += 1
            network_records += len(result.records)
            network_bytes += result_bytes
        response_hashes = [row.response_hash for row in result.manifest_rows]
        response_hash = (
            response_hashes[0]
            if len(response_hashes) == 1
            else hashlib.sha256(
                json.dumps(response_hashes, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        filtered = filter_exact_observation_date(
            result.records,
            requested_observation_date=planned.option_date,
            response_hash=response_hash,
            pagination_complete=True,
        )
        query_rows.append(
            {
                "request_id": planned.request_id,
                "resume_id": planned.resume_id,
                "symbol": planned.symbol,
                "option_date": planned.option_date.isoformat(),
                "strategy": planned.strategy,
                "roles": ";".join(planned.roles),
                "records_returned": len(result.records),
                "exact_date_records": filtered.exact_date_record_count,
                "discarded_other_date_records": filtered.discarded_other_date_record_count,
                "discarded_post_boundary_records": (filtered.discarded_post_boundary_record_count),
                "response_bytes": result_bytes,
                "request_was_cached": request_was_cached,
                "status": filtered.status,
            }
        )
        write_json(receipt_path, current_outcome("in_progress"))
    final_status = (
        "blocked_resource_limit"
        if stopped_by_resource_limit
        else "download_failure"
        if stopped_by_download_failure
        else "completed"
    )
    outcome = current_outcome(final_status)
    write_json(receipt_path, outcome)
    return outcome


def load_canonical_option_cache(
    options_cache: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only pre-boundary canonical rows from the supplied cache.

    Complete canonical chain Parquets are read with a pushed-down protected-date
    predicate. Existing contract histories are opened only when the OCC identity
    proves the contract expired before the protected boundary.
    """

    frames: list[pd.DataFrame] = []
    parquet_files = sorted(options_cache.rglob("*.parquet")) if options_cache.is_dir() else []
    canonical_columns = list(CANONICAL_OPTION_COLUMNS)
    optional_columns = [
        "adjusted_contract",
        "deliverable_resolved",
        "contract_multiplier",
        "settlement_style",
        "chain_complete",
        "cache_source",
        "request_strategy",
    ]
    parquet_rows = 0
    complete_strategy_date_coverage: set[tuple[str, str, str]] = set()
    bounded_summary, bounded_receipt_coverage = bounded_download_cache_summary(options_cache)
    complete_strategy_date_coverage.update(bounded_receipt_coverage)
    for path in parquet_files:
        try:
            schema_frame = pd.read_parquet(path, columns=[])
            del schema_frame
            parquet_file = pq.ParquetFile(path)
            available_columns = set(parquet_file.schema.names)
        except Exception as error:
            raise ScreenBlocker(
                "blocked_options_quote_integrity_failure",
                f"canonical option Parquet schema unreadable: {path}: {error}",
            ) from error
        required_columns = set(CANONICAL_OPTION_COLUMNS)
        if not required_columns.issubset(available_columns):
            continue
        trade_date_type = parquet_file.schema_arrow.field("trade_date").type
        columns = [
            column
            for column in (*canonical_columns, *optional_columns)
            if column in available_columns
        ]
        if pa.types.is_date(trade_date_type):
            protected_filter_value: object = PROTECTED_START
        elif pa.types.is_string(trade_date_type) or pa.types.is_large_string(trade_date_type):
            protected_filter_value = PROTECTED_START.isoformat()
        elif pa.types.is_timestamp(trade_date_type):
            protected_filter_value = pd.Timestamp(PROTECTED_START)
        else:
            raise ScreenBlocker(
                "blocked_protected_boundary_failure",
                f"unsupported option trade_date type in {path}: {trade_date_type}",
            )
        try:
            frame = pd.read_parquet(
                path,
                columns=columns,
                filters=[("trade_date", "<", protected_filter_value)],
            )
        except Exception as error:
            raise ScreenBlocker(
                "blocked_protected_boundary_failure",
                f"protected-date predicate could not be pushed into {path}: {error}",
            ) from error
        if not frame.empty:
            try:
                frame["chain_complete"] = (
                    strict_boolean_series(frame["chain_complete"], name="chain_complete")
                    if "chain_complete" in frame
                    else False
                )
            except ValueError as error:
                raise ScreenBlocker(
                    "blocked_options_quote_integrity_failure",
                    f"chain-completeness metadata is invalid in {path}: {error}",
                ) from error
            frame["cache_source"] = (
                frame["cache_source"].astype(str)
                if "cache_source" in frame
                else f"canonical_parquet:{path.relative_to(options_cache)}"
            )
            if "request_strategy" in frame:
                for (symbol, trade_date_value, strategy), group in frame.groupby(
                    ["underlying_symbol", "trade_date", "request_strategy"],
                    dropna=False,
                    sort=False,
                ):
                    strategy_value = str(strategy)
                    if strategy_value in {"S1", "S3"} and group["chain_complete"].all():
                        complete_strategy_date_coverage.add(
                            (
                                str(symbol),
                                str(trade_date_value)[:10],
                                strategy_value,
                            )
                        )
            frames.append(frame)
            parquet_rows += len(frame)

    prior_manifest = (
        cast(dict[str, Any], json.loads(PRIOR_PROBE_MANIFEST.read_text(encoding="utf-8")))
        if PRIOR_PROBE_MANIFEST.is_file()
        else {}
    )
    local_json = (
        {path.stem: path for path in options_cache.rglob("*.json")}
        if options_cache.is_dir()
        else {}
    )
    history_metadata: dict[str, tuple[str, int]] = {}
    actual_raw_records = 0
    accounted_hashes: set[str] = set()
    for row in cast(list[dict[str, Any]], prior_manifest.get("manifest_rows", [])):
        response_hash = str(row.get("response_hash", ""))
        if response_hash not in local_json or response_hash in accounted_hashes:
            continue
        accounted_hashes.add(response_hash)
        actual_raw_records += int(row.get("record_count", 0))
        contract_id = row.get("contract_id")
        if isinstance(contract_id, str):
            history_metadata[response_hash] = (contract_id, int(row.get("record_count", 0)))

    raw_history_rows = 0
    raw_payloads_opened = 0
    canonical_rejections = 0
    for response_hash, (contract_id, _record_count) in sorted(history_metadata.items()):
        expiration = _contract_expiration(contract_id)
        if expiration is None or expiration >= PROTECTED_START:
            continue
        path = local_json[response_hash]
        if sha256_file(path) != response_hash:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"option cache content hash differs: {path}",
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ScreenBlocker(
                "blocked_options_quote_integrity_failure",
                f"contract history payload schema differs: {path}",
            )
        result = canonicalize_response_records(
            cast(list[Mapping[str, Any]], payload["data"]),
            request_id=response_hash,
            provider_schema_version="openapi-2.0.0-contract-history",
        )
        canonical_rejections += len(result.rejections)
        safe_records = [
            {
                **record,
                "adjusted_contract": False,
                "deliverable_resolved": True,
                "contract_multiplier": 100,
                "settlement_style": "standard_equity_pm",
                "chain_complete": False,
                "cache_source": "bounded_prior_contract_history",
            }
            for record in result.records
            if cast(date, record["trade_date"]) < PROTECTED_START
        ]
        if len(safe_records) != len(result.records):
            raise ScreenBlocker(
                "blocked_protected_boundary_failure",
                "a supposedly pre-boundary contract history contained a protected row",
            )
        if safe_records:
            frames.append(pd.DataFrame(safe_records))
            raw_history_rows += len(safe_records)
        raw_payloads_opened += 1

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="raise").dt.date
        combined["expiration_date"] = pd.to_datetime(
            combined["expiration_date"], errors="raise"
        ).dt.date
        reject_protected_dates(combined, ["trade_date"])
        deduplicated = resolve_canonical_duplicates(
            cast(list[Mapping[str, Any]], combined.to_dict(orient="records"))
        )
        records = pd.DataFrame(deduplicated.records)
    else:
        records = pd.DataFrame(columns=[*CANONICAL_OPTION_COLUMNS, *optional_columns])
        deduplicated = None
    if not records.empty:
        records = records.sort_values(
            ["underlying_symbol", "trade_date", "expiration_date", "strike", "option_type"],
            kind="mergesort",
        ).reset_index(drop=True)
    cache_files = (
        [path for path in options_cache.rglob("*") if path.is_file()]
        if options_cache.is_dir()
        else []
    )
    summary = {
        "cache_root": str(options_cache),
        "cache_json_files": sum(path.suffix == ".json" for path in cache_files),
        "cache_parquet_files": len(parquet_files),
        "cache_bytes": sum(path.stat().st_size for path in cache_files),
        "cached_raw_provider_records": actual_raw_records,
        "canonical_pre_boundary_records": len(records),
        "canonical_parquet_rows": parquet_rows,
        "canonical_safe_history_rows": raw_history_rows,
        "canonical_rejections": canonical_rejections,
        "canonical_duplicate_records": (
            int(deduplicated.duplicate_records) if deduplicated is not None else 0
        ),
        "canonical_conflicting_duplicate_groups": (
            int(deduplicated.conflicting_duplicate_groups) if deduplicated is not None else 0
        ),
        "newly_downloaded_records": 0,
        "newly_downloaded_bytes": 0,
        "protected_option_rows_materialised": 0,
        "raw_option_payloads_opened": raw_payloads_opened,
        **bounded_summary,
        "experiment_downloaded_records": bounded_summary["bounded_cached_provider_records"],
        "experiment_downloaded_bytes": bounded_summary["bounded_cached_logical_response_bytes"],
        "complete_strategy_date_coverage": [
            {
                "symbol": symbol,
                "option_date": option_date,
                "strategy": strategy,
            }
            for symbol, option_date, strategy in sorted(complete_strategy_date_coverage)
        ],
    }
    return records, summary


def inspect_existing_option_cache(
    required: pd.DataFrame, options_cache: Path
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    canonical, summary = load_canonical_option_cache(options_cache)
    cached_groups = {
        (str(symbol), str(trade_date)): group
        for (symbol, trade_date), group in canonical.groupby(
            ["underlying_symbol", "trade_date"], sort=False
        )
    }
    gap_rows: list[dict[str, object]] = []
    complete_coverage = {
        (str(row["symbol"]), str(row["option_date"]), str(row["strategy"]))
        for row in cast(
            list[Mapping[str, object]],
            summary.get("complete_strategy_date_coverage", []),
        )
    }
    for (symbol, option_date, strategy), group in required.groupby(
        ["symbol", "option_date", "strategy"], sort=True
    ):
        cached = cached_groups.get((str(symbol), str(option_date)))
        cached_rows = 0 if cached is None else len(cached)
        chain_complete = (
            str(symbol),
            str(option_date),
            str(strategy),
        ) in complete_coverage
        if chain_complete:
            continue
        gap_rows.append(
            {
                "symbol": str(symbol),
                "option_date": str(option_date),
                "roles": ";".join(sorted(set(group["role"].astype(str)))),
                "strategies": str(strategy),
                "signal_sessions": int(group["signal_session"].nunique()),
                "cached_canonical_rows": cached_rows,
                "complete_required_chain_cached": False,
                "gap_reason": (
                    f"{strategy}_bounded_chain_coverage_absent"
                    if cached_rows
                    else "exact_symbol_date_chain_absent"
                ),
            }
        )
    gaps = pd.DataFrame(
        gap_rows,
        columns=[
            "symbol",
            "option_date",
            "roles",
            "strategies",
            "signal_sessions",
            "cached_canonical_rows",
            "complete_required_chain_cached",
            "gap_reason",
        ],
    )
    quote_rows = [
        {
            "scope": "supplied_canonical_cache",
            "check": "pre_boundary_canonical_integrity",
            "passed": bool(
                summary["canonical_conflicting_duplicate_groups"] == 0
                and summary["protected_option_rows_materialised"] == 0
            ),
            "rows": int(summary["canonical_pre_boundary_records"]),
            "detail": "daily option high/low fields were never selected into the canonical frame",
        },
        {
            "scope": "current_quick_screen",
            "check": "complete_required_chain_cache",
            "passed": bool(gaps.empty),
            "rows": int(len(gaps)),
            "detail": "no contract substitution or forward fill was attempted",
        },
    ]
    return gaps, summary, pd.DataFrame(quote_rows), canonical


def download_bounded_option_gaps(
    *,
    token: str,
    required: pd.DataFrame,
    signals: pd.DataFrame,
    options_cache: Path,
) -> dict[str, int]:
    """Download only exact required dates and frozen strike/expiry ranges."""

    if not token:
        raise ScreenBlocker(
            "blocked_options_contract_reconstruction_failure",
            "EODHD_API_TOKEN is unavailable for unresolved bounded requests",
        )
    transport_module = load_module(OPTIONS_DOWNLOAD_RUNNER, "fixed_options_download_transport")
    data_dir = options_cache / V01_OPTIONS_RUN_DIR_NAME
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=data_dir,
            page_limit=1000,
            requests_per_minute=300,
            maximum_raw_records=500_000,
            maximum_download_bytes=5 * 1024**3,
        ),
        transport=transport_module.RequestsTransport(),
    )
    signals_by_key = {
        (str(row.symbol), str(row.session)): row for row in signals.itertuples(index=False)
    }
    canonical_dir = options_cache / "canonical" / V01_OPTIONS_RUN_DIR_NAME
    canonical_dir.mkdir(parents=True, exist_ok=True)
    processed_records = 0
    processed_bytes = 0
    network_records = 0
    network_bytes = 0
    receipt_path = data_dir / "bounded_download_manifest.json"
    query_by_id: dict[str, dict[str, object]] = {}
    if receipt_path.is_file():
        prior_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        prior_queries = prior_receipt.get("queries", [])
        if isinstance(prior_queries, list):
            query_by_id = {
                str(row["request_id"]): cast(dict[str, object], row)
                for row in prior_queries
                if isinstance(row, dict) and isinstance(row.get("request_id"), str)
            }

    def accounting_evidence() -> dict[str, object]:
        return {
            "processed_provider_records": processed_records,
            "processed_logical_response_bytes": processed_bytes,
            "current_invocation_network_records": network_records,
            "current_invocation_network_bytes": network_bytes,
        }

    def write_receipt(status: str) -> None:
        write_json(
            receipt_path,
            {
                **SAFETY_FLAGS_V01,
                "status": status,
                **accounting_evidence(),
                "queries": sorted(
                    query_by_id.values(),
                    key=lambda row: (
                        str(row.get("option_date", "")),
                        str(row.get("symbol", "")),
                        str(row.get("strategy", "")),
                        str(row.get("request_id", "")),
                    ),
                ),
                "credential_recorded": False,
            },
        )

    groups = required.groupby(["symbol", "option_date", "strategy"], sort=True)
    for (symbol, option_date_value, strategy), group in groups:
        target_date = date.fromisoformat(str(option_date_value))
        if target_date >= PROTECTED_START:
            raise ScreenBlocker(
                "blocked_protected_boundary_failure",
                "bounded option request reached protected date",
            )
        source_rows = [
            signals_by_key[(str(symbol), str(session))]
            for session in sorted(set(group["signal_session"].astype(str)))
        ]
        previous_closes = [float(row.previous_close_underlying_price) for row in source_rows]
        strike_from = 0.70 * min(previous_closes)
        strike_to = 1.30 * max(previous_closes)
        entry_dates = [date.fromisoformat(str(row.session)) for row in source_rows]
        if str(strategy) == "S1":
            expiration_from = min(value + pd.Timedelta(days=7) for value in entry_dates)
            expiration_to = max(value + pd.Timedelta(days=14) for value in entry_dates)
        elif str(strategy) == "S3":
            expiration_from = min(value + pd.Timedelta(days=1) for value in entry_dates)
            expiration_to = max(value + pd.Timedelta(days=1) for value in entry_dates)
        else:
            continue
        request = OptionsRequest(
            underlying_symbol=str(symbol),
            trade_date_from=target_date,
            trade_date_to=target_date,
            strike_from=strike_from,
            strike_to=strike_to,
            expiration_from=cast(date, expiration_from),
            expiration_to=cast(date, expiration_to),
            compact=False,
        )
        resume_id = stable_request_id(
            request.endpoint,
            request.parameters(offset=0, limit=downloader.config.page_limit),
        )
        request_was_cached = (data_dir / "manifests" / "completed" / f"{resume_id}.json").is_file()
        try:
            result = downloader.download(request)
        except OptionsResourceLimitExceeded as error:
            write_receipt("blocked")
            raise ScreenBlocker(
                "blocked_quick_options_strategy_resource_limit",
                str(error),
                evidence=accounting_evidence(),
            ) from error
        except OptionsDownloadError as error:
            write_receipt("blocked")
            raise ScreenBlocker(
                "blocked_options_contract_reconstruction_failure",
                str(error),
                evidence=accounting_evidence(),
            ) from error
        result_records = len(result.records)
        result_bytes = sum(Path(row.cache_path).stat().st_size for row in result.manifest_rows)
        processed_records += result_records
        processed_bytes += result_bytes
        if not request_was_cached:
            network_records += result_records
            network_bytes += result_bytes
        request_identity = hashlib.sha256(
            canonical_json(
                {
                    "symbol": symbol,
                    "option_date": target_date,
                    "strategy": strategy,
                    "strike_from": strike_from,
                    "strike_to": strike_to,
                    "expiration_from": expiration_from,
                    "expiration_to": expiration_to,
                }
            ).encode("utf-8")
        ).hexdigest()
        query_row: dict[str, object] = {
            "request_id": request_identity,
            "symbol": str(symbol),
            "option_date": target_date.isoformat(),
            "strategy": str(strategy),
            "records": result_records,
            "bytes": result_bytes,
            "request_was_cached": request_was_cached,
        }
        response_hashes = [row.response_hash for row in result.manifest_rows]
        response_hash = (
            response_hashes[0]
            if len(response_hashes) == 1
            else hashlib.sha256(
                json.dumps(response_hashes, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        filtered = filter_exact_observation_date(
            result.records,
            requested_observation_date=target_date,
            response_hash=response_hash,
            pagination_complete=True,
        )
        if filtered.status not in {"exact_date_complete", "extra_dates_discarded"}:
            query_by_id[request_identity] = {
                **query_row,
                "status": filtered.status,
            }
            write_receipt("in_progress")
            continue
        canonical = canonicalize_response_records(
            list(filtered.retained_records),
            request_id=request_identity,
            provider_schema_version="openapi-2.0.0-exact-bounded-chain",
        )
        if canonical.rejections:
            query_by_id[request_identity] = {
                **query_row,
                "status": "blocked",
                "blocker": "blocked_options_quote_integrity_failure",
                "canonical_rejections": len(canonical.rejections),
            }
            write_receipt("blocked")
            raise ScreenBlocker(
                "blocked_options_quote_integrity_failure",
                f"canonical option rejections in bounded request: {len(canonical.rejections)}",
                evidence={
                    **accounting_evidence(),
                    "symbol": str(symbol),
                    "requested_option_observation_date": target_date.isoformat(),
                    "strategy": str(strategy),
                    "canonical_rejections": len(canonical.rejections),
                },
            )
        observed_dates = [value.isoformat() for value in filtered.returned_observation_dates]
        stored: list[dict[str, Any]] = []
        for row in canonical.records:
            contract_id = str(row["contract_id"])
            try:
                multiplier = standard_contract_multiplier(
                    contract_id,
                    underlying_symbol=str(symbol),
                    strike=float(row["strike"]),
                    adjusted_contract=False,
                    deliverable_resolved=True,
                )
                adjusted = False
                deliverable = True
            except ValueError:
                multiplier = 0
                adjusted = True
                deliverable = False
            stored.append(
                {
                    **row,
                    "adjusted_contract": adjusted,
                    "deliverable_resolved": deliverable,
                    "contract_multiplier": multiplier,
                    "settlement_style": ("standard_equity_pm" if deliverable else "ambiguous"),
                    "chain_complete": True,
                    "cache_source": "exact_bounded_live_download",
                    "request_strategy": str(strategy),
                }
            )
        destination = canonical_dir / f"{request_identity}.parquet"
        if stored:
            write_parquet(destination, pd.DataFrame(stored))
        query_by_id[request_identity] = {
            **query_row,
            "records": len(stored),
            "status": "complete",
            "observed_option_observation_dates": observed_dates,
        }
        write_receipt("in_progress")
    if processed_records > 500_000 or processed_bytes > 5 * 1024**3:
        write_receipt("blocked")
        raise ScreenBlocker(
            "blocked_quick_options_strategy_resource_limit",
            "bounded option download exceeded frozen records or bytes",
            evidence=accounting_evidence(),
        )
    write_receipt("completed")
    return {
        "processed_provider_records": processed_records,
        "processed_logical_response_bytes": processed_bytes,
        "newly_downloaded_records": network_records,
        "newly_downloaded_bytes": network_bytes,
    }


def empty_trade_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.DataFrame(
        columns=[
            "selection_id",
            "strategy",
            "symbol",
            "signal_session",
            "contract_selection_date",
            "expiration_date",
            "entry_dte",
            "direction",
            "long_contract_id",
            "short_contract_id",
            "call_contract_id",
            "put_contract_id",
            "strike",
            "selection_status",
            "rejection_reason",
        ]
    )
    trades = pd.DataFrame(
        columns=[
            "trade_id",
            "strategy",
            "period",
            "symbol",
            "session",
            "calendar_month",
            "entry_dte",
            "entry_dte_bin",
            "previous_close_atm_iv_quartile",
            "spread_group",
            "entry_debit",
            "exit_credit",
            "contract_multiplier",
            "commission_per_contract_side",
            "commissions",
            "total_initial_cash_debit",
            "net_pnl",
            "return_on_entry_debit",
            "matched_control_excess",
            "hidden_2_3_2_prior_6",
            "any_hidden_event_prior_6",
            "win",
        ]
    )
    return selected, trades


def empty_selection_quote_audit() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scope",
            "check",
            "passed",
            "rows",
            "detail",
            "selection_id",
            "strategy",
            "symbol",
            "signal_session",
            "contract_id",
            "quote_role",
            "quote_date",
            "rejection_reason",
            "bid",
            "ask",
            "midpoint",
            "relative_spread",
        ]
    )


def _option_groups(
    options: pd.DataFrame,
) -> tuple[
    dict[tuple[str, date], pd.DataFrame],
    dict[tuple[str, date], Mapping[str, Any]],
]:
    chains: dict[tuple[str, date], pd.DataFrame] = {}
    quotes: dict[tuple[str, date], Mapping[str, Any]] = {}
    if options.empty:
        return chains, quotes
    for (symbol, trade_date), group in options.groupby(
        ["underlying_symbol", "trade_date"], sort=False
    ):
        chains[(str(symbol), cast(date, trade_date))] = group.copy()
    for (contract_id, trade_date), group in options.groupby(
        ["contract_id", "trade_date"], sort=False
    ):
        if len(group) != 1:
            raise ScreenBlocker(
                "blocked_options_quote_integrity_failure",
                f"non-unique canonical contract-date quote: {contract_id} {trade_date}",
            )
        quotes[(str(contract_id), cast(date, trade_date))] = cast(
            Mapping[str, Any], group.iloc[0].to_dict()
        )
    return chains, quotes


def _empty_metrics() -> dict[str, float | int]:
    return {
        "trades": 0,
        "sessions": 0,
        "stocks": 0,
        "months": 0,
        "total_entry_debit": 0.0,
        "mean_net_pnl": math.nan,
        "median_net_pnl": math.nan,
        "trimmed_mean_net_pnl": math.nan,
        "mean_return_on_debit": math.nan,
        "median_return_on_debit": math.nan,
        "win_rate": math.nan,
        "full_loss_rate": math.nan,
        "maximum_gain": math.nan,
        "maximum_loss": math.nan,
        "return_p05": math.nan,
        "return_p25": math.nan,
        "return_p75": math.nan,
        "return_p95": math.nan,
        "matched_control_excess": math.nan,
        "maximum_stock_share": math.nan,
        "maximum_month_share": math.nan,
        "top_5pct_positive_pnl_contribution": math.nan,
    }


def _metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return _empty_metrics()
    values = _empty_metrics()
    values.update(strategy_metrics(frame))
    return values


def _dte_bin(strategy: str, entry_dte: int) -> str:
    if strategy == "S3":
        return "1"
    if entry_dte <= 9:
        return "7-9"
    if entry_dte <= 12:
        return "10-12"
    return "13-14"


def construct_straddle_economics(
    signals: pd.DataFrame,
    options: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build S1/S3 selections, controls, and treated trades from a complete cache."""

    chains, quotes = _option_groups(options)
    selection_rows: list[dict[str, object]] = []
    base_rows = signals.loc[
        signals["ordinal_72_structural_available"].astype(bool)
        & signals["chronology_eligible"].astype(bool)
        & signals["underlying_source_available"].astype(bool)
        & ~signals["split_boundary_ambiguous"].astype(bool)
    ]
    for signal in base_rows.itertuples(index=False):
        entry_date = date.fromisoformat(str(signal.session))
        selection_date = date.fromisoformat(str(signal.contract_selection_date))
        chain = chains.get((str(signal.symbol), selection_date))
        for strategy, minimum_dte, maximum_dte in (
            ("S1", 7, 14),
            ("S3", 1, 1),
        ):
            common: dict[str, object] = {
                "selection_id": f"{strategy}|{signal.symbol}|{signal.session}",
                "strategy": strategy,
                "symbol": str(signal.symbol),
                "signal_session": str(signal.session),
                "period": str(signal.period),
                "contract_selection_date": str(signal.contract_selection_date),
                "exit_session": str(signal.exit_session),
                "previous_close_underlying_price": float(signal.previous_close_underlying_price),
                "route_resolution_state": str(signal.route_resolution_state),
                "broad_conflict": bool(signal.BROAD_CONFLICT),
                "recent_registered_completion_prior_6": bool(
                    signal.recent_registered_completion_prior_6
                ),
                "hidden_2_3_2_prior_6": bool(signal.hidden_2_3_2_prior_6),
                "any_hidden_event_prior_6": bool(signal.any_hidden_event_prior_6),
                "entry_underlying_close": float(signal.entry_underlying_close),
                "exit_underlying_close": float(signal.exit_underlying_close),
            }
            if chain is None or chain.empty:
                selection_rows.append(
                    {
                        **common,
                        "selection_status": "rejected",
                        "rejection_reason": "exact_previous_session_chain_absent",
                    }
                )
                continue
            try:
                complete_chain = (
                    "chain_complete" in chain
                    and strict_boolean_series(chain["chain_complete"], name="chain_complete").all()
                )
            except ValueError as error:
                selection_rows.append(
                    {
                        **common,
                        "selection_status": "rejected",
                        "rejection_reason": f"chain_complete_metadata_invalid:{error}",
                    }
                )
                continue
            if not complete_chain:
                selection_rows.append(
                    {
                        **common,
                        "selection_status": "rejected",
                        "rejection_reason": "previous_session_chain_not_complete",
                    }
                )
                continue
            try:
                selection = select_atm_straddle(
                    chain,
                    selection_date=selection_date,
                    entry_date=entry_date,
                    underlying_close=float(signal.previous_close_underlying_price),
                    entry_dte_min=minimum_dte,
                    entry_dte_max=maximum_dte,
                )
            except OptionSelectionError as error:
                selection_rows.append(
                    {
                        **common,
                        "selection_status": "rejected",
                        "rejection_reason": f"selection_error:{type(error).__name__}:{error}",
                    }
                )
                continue
            if not selection.available:
                selection_rows.append(
                    {
                        **common,
                        "selection_status": "rejected",
                        "rejection_reason": selection.reason,
                        "expiration_date": selection.expiration_date,
                        "entry_dte": selection.entry_dte,
                        "strike": selection.strike,
                        "call_contract_id": selection.call_contract_id,
                        "put_contract_id": selection.put_contract_id,
                    }
                )
                continue
            state = previous_close_option_state(
                selection,
                underlying_close=float(signal.previous_close_underlying_price),
            )
            selection_rows.append(
                {
                    **common,
                    "selection_status": "selected",
                    "rejection_reason": "",
                    "expiration_date": selection.expiration_date,
                    "entry_dte": selection.entry_dte,
                    "strike": selection.strike,
                    "call_contract_id": selection.call_contract_id,
                    "put_contract_id": selection.put_contract_id,
                    **state,
                }
            )
    selected = pd.DataFrame(selection_rows)
    if selected.empty:
        empty_selected, empty_trades = empty_trade_artifacts()
        return (
            empty_selected,
            empty_trades,
            empty_trades.copy(),
            empty_selection_quote_audit(),
        )
    selected["stock_specific_2024_median_atm_iv"] = np.nan
    selected["previous_close_atm_iv_quartile"] = pd.Series(
        pd.NA, index=selected.index, dtype="string"
    )
    selected["spread_group"] = pd.Series(pd.NA, index=selected.index, dtype="string")
    selected["cheap_iv"] = False
    valid = selected["selection_status"].eq("selected")
    for _group_key, indices in (
        selected.loc[valid].groupby(["strategy", "symbol"], sort=True).groups.items()
    ):
        all_index = list(indices)
        development_index = [
            index for index in all_index if selected.loc[index, "period"] == "development"
        ]
        if not development_index:
            continue
        development_iv = pd.to_numeric(selected.loc[development_index, "atm_iv"], errors="raise")
        median_iv = float(development_iv.median())
        quartiles = development_iv.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
        development_spread = pd.to_numeric(
            selected.loc[development_index, "combined_relative_spread"],
            errors="raise",
        )
        spread_median = float(development_spread.median())
        values = pd.to_numeric(selected.loc[all_index, "atm_iv"], errors="raise").to_numpy(
            dtype=float
        )
        selected.loc[all_index, "stock_specific_2024_median_atm_iv"] = median_iv
        selected.loc[all_index, "cheap_iv"] = values <= median_iv
        bins = np.searchsorted(quartiles, values, side="left")
        selected.loc[all_index, "previous_close_atm_iv_quartile"] = np.asarray(
            ("Q1", "Q2", "Q3", "Q4"), dtype=object
        )[bins]
        selected.loc[all_index, "spread_group"] = np.where(
            pd.to_numeric(
                selected.loc[all_index, "combined_relative_spread"], errors="raise"
            ).to_numpy(dtype=float)
            <= spread_median,
            "tight",
            "wide",
        )

    calendar = get_market_calendar("NYSE")
    expiration_schedule = calendar.schedule(
        start_date=str(selected["signal_session"].min()),
        end_date=str(selected["exit_session"].max()),
    )
    close_by_date = {
        timestamp.date(): pd.Timestamp(row.market_close)
        for timestamp, row in expiration_schedule.iterrows()
    }
    candidate_rows: list[dict[str, object]] = []
    quote_audit_rows: list[dict[str, object]] = []
    economics_outcomes: dict[str, tuple[str, str]] = {}
    for item in selected.loc[selected["selection_status"].eq("selected")].itertuples(index=False):
        selection_id = str(item.selection_id)
        if pd.isna(item.stock_specific_2024_median_atm_iv):
            reason = "stock_specific_2024_iv_median_unavailable"
            economics_outcomes[selection_id] = ("rejected", reason)
            quote_audit_rows.append(
                {
                    "scope": "selected_contract",
                    "check": "entry_exit_quote_set",
                    "passed": False,
                    "rows": 0,
                    "detail": reason,
                    "selection_id": selection_id,
                    "strategy": str(item.strategy),
                    "symbol": str(item.symbol),
                    "signal_session": str(item.signal_session),
                    "contract_id": "",
                    "quote_role": "selection_summary",
                    "quote_date": "",
                    "rejection_reason": reason,
                    "bid": math.nan,
                    "ask": math.nan,
                    "midpoint": math.nan,
                    "relative_spread": math.nan,
                }
            )
            continue
        entry_date = date.fromisoformat(str(item.signal_session))
        exit_date = date.fromisoformat(str(item.exit_session))
        call_entry = quotes.get((str(item.call_contract_id), entry_date))
        put_entry = quotes.get((str(item.put_contract_id), entry_date))
        call_exit = quotes.get((str(item.call_contract_id), exit_date))
        put_exit = quotes.get((str(item.put_contract_id), exit_date))
        quote_set = {
            "call_entry": call_entry,
            "put_entry": put_entry,
            "call_exit": call_exit,
            "put_exit": put_exit,
        }
        contract_ids = {
            "call_entry": str(item.call_contract_id),
            "put_entry": str(item.put_contract_id),
            "call_exit": str(item.call_contract_id),
            "put_exit": str(item.put_contract_id),
        }
        quote_dates = {
            "call_entry": entry_date,
            "put_entry": entry_date,
            "call_exit": exit_date,
            "put_exit": exit_date,
        }
        quote_failures: list[str] = []
        for role, quote in quote_set.items():
            reason = (
                "missing_quote"
                if quote is None
                else quote_integrity_reason(quote, require_open_interest=False)
            )
            if reason is not None:
                quote_failures.append(f"{role}:{reason}")
            quote_audit_rows.append(
                {
                    "scope": "selected_contract",
                    "check": "entry_exit_leg_quote",
                    "passed": reason is None,
                    "rows": 0 if quote is None else 1,
                    "detail": reason or "valid_frozen_eod_bid_ask_quote",
                    "selection_id": selection_id,
                    "strategy": str(item.strategy),
                    "symbol": str(item.symbol),
                    "signal_session": str(item.signal_session),
                    "contract_id": contract_ids[role],
                    "quote_role": role,
                    "quote_date": quote_dates[role].isoformat(),
                    "rejection_reason": reason or "",
                    "bid": math.nan if quote is None else audit_numeric(quote.get("bid")),
                    "ask": math.nan if quote is None else audit_numeric(quote.get("ask")),
                    "midpoint": (
                        math.nan if quote is None else audit_numeric(quote.get("midpoint"))
                    ),
                    "relative_spread": (
                        math.nan if quote is None else audit_relative_spread(quote)
                    ),
                }
            )
        if quote_failures:
            reason = ";".join(quote_failures)
            economics_outcomes[selection_id] = ("rejected", reason)
            quote_audit_rows.append(
                {
                    "scope": "selected_contract",
                    "check": "entry_exit_quote_set",
                    "passed": False,
                    "rows": 4 - sum(value is None for value in quote_set.values()),
                    "detail": reason,
                    "selection_id": selection_id,
                    "strategy": str(item.strategy),
                    "symbol": str(item.symbol),
                    "signal_session": str(item.signal_session),
                    "contract_id": "",
                    "quote_role": "selection_summary",
                    "quote_date": "",
                    "rejection_reason": reason,
                    "bid": math.nan,
                    "ask": math.nan,
                    "midpoint": math.nan,
                    "relative_spread": math.nan,
                }
            )
            continue
        typed_quotes = cast(dict[str, Mapping[str, Any]], quote_set)
        try:
            call_multiplier = standard_contract_multiplier(
                str(item.call_contract_id),
                underlying_symbol=str(item.symbol),
                strike=float(item.strike),
                adjusted_contract=explicit_boolean(
                    typed_quotes["call_entry"].get("adjusted_contract"),
                    name="call adjusted_contract",
                ),
                deliverable_resolved=explicit_boolean(
                    typed_quotes["call_entry"].get("deliverable_resolved"),
                    name="call deliverable_resolved",
                ),
            )
            put_multiplier = standard_contract_multiplier(
                str(item.put_contract_id),
                underlying_symbol=str(item.symbol),
                strike=float(item.strike),
                adjusted_contract=explicit_boolean(
                    typed_quotes["put_entry"].get("adjusted_contract"),
                    name="put adjusted_contract",
                ),
                deliverable_resolved=explicit_boolean(
                    typed_quotes["put_entry"].get("deliverable_resolved"),
                    name="put deliverable_resolved",
                ),
            )
            if call_multiplier != put_multiplier:
                raise ValueError("straddle leg multipliers differ")
            for role, multiplier in (
                ("call_entry", call_multiplier),
                ("put_entry", put_multiplier),
                ("call_exit", call_multiplier),
                ("put_exit", put_multiplier),
            ):
                provider_multiplier = typed_quotes[role].get("contract_multiplier")
                if provider_multiplier is not None and int(provider_multiplier) != multiplier:
                    raise ValueError(f"{role} provider multiplier differs")
            if str(item.strategy) == "S3":
                for leg in ("call_entry", "put_entry"):
                    validate_expiration_session(
                        expiration_date=cast(date, item.expiration_date),
                        exit_session=exit_date,
                        settlement_style=str(
                            typed_quotes[leg].get("settlement_style", "standard_equity_pm")
                        ),
                        scheduled_close_timestamp=close_by_date[exit_date],
                        adjusted_contract=explicit_boolean(
                            typed_quotes[leg].get("adjusted_contract"),
                            name=f"{leg} adjusted_contract",
                        ),
                        deliverable_resolved=explicit_boolean(
                            typed_quotes[leg].get("deliverable_resolved"),
                            name=f"{leg} deliverable_resolved",
                        ),
                    )
            entry_quotes = {
                "call_ask": float(typed_quotes["call_entry"]["ask"]),
                "put_ask": float(typed_quotes["put_entry"]["ask"]),
            }
            exit_quotes = {
                "call_bid": float(typed_quotes["call_exit"]["bid"]),
                "put_bid": float(typed_quotes["put_exit"]["bid"]),
            }
            primary = option_position_pnl(
                structure="long_straddle",
                entry_quotes=entry_quotes,
                exit_quotes=exit_quotes,
                multiplier=call_multiplier,
                commission_per_contract_side=0.75,
            )
            sensitivity = option_position_pnl(
                structure="long_straddle",
                entry_quotes=entry_quotes,
                exit_quotes=exit_quotes,
                multiplier=call_multiplier,
                commission_per_contract_side=1.00,
            )
        except (KeyError, TypeError, ValueError) as error:
            reason = f"construction_error:{type(error).__name__}:{error}"
            economics_outcomes[selection_id] = ("rejected", reason)
            quote_audit_rows.append(
                {
                    "scope": "selected_contract",
                    "check": "entry_exit_quote_set",
                    "passed": False,
                    "rows": 4,
                    "detail": reason,
                    "selection_id": selection_id,
                    "strategy": str(item.strategy),
                    "symbol": str(item.symbol),
                    "signal_session": str(item.signal_session),
                    "contract_id": "",
                    "quote_role": "selection_summary",
                    "quote_date": "",
                    "rejection_reason": reason,
                    "bid": math.nan,
                    "ask": math.nan,
                    "midpoint": math.nan,
                    "relative_spread": math.nan,
                }
            )
            continue
        economics_outcomes[selection_id] = ("constructed", "")
        quote_audit_rows.append(
            {
                "scope": "selected_contract",
                "check": "entry_exit_quote_set",
                "passed": True,
                "rows": 4,
                "detail": "all four frozen EOD quotes and contract checks passed",
                "selection_id": selection_id,
                "strategy": str(item.strategy),
                "symbol": str(item.symbol),
                "signal_session": str(item.signal_session),
                "contract_id": "",
                "quote_role": "selection_summary",
                "quote_date": "",
                "rejection_reason": "",
                "bid": math.nan,
                "ask": math.nan,
                "midpoint": math.nan,
                "relative_spread": math.nan,
            }
        )
        underlying_return = (
            float(item.exit_underlying_close) / float(item.entry_underlying_close) - 1.0
        )
        intrinsic = (
            expiry_intrinsic_values(
                underlying_close=float(item.exit_underlying_close),
                strike=float(item.strike),
            )
            if str(item.strategy) == "S3"
            else {"call_intrinsic": math.nan, "put_intrinsic": math.nan}
        )
        candidate_rows.append(
            {
                "trade_id": f"{item.strategy}|{item.symbol}|{item.signal_session}",
                "strategy": str(item.strategy),
                "period": str(item.period),
                "symbol": str(item.symbol),
                "session": str(item.signal_session),
                "calendar_month": str(item.signal_session)[:7],
                "weekday": entry_date.weekday(),
                "entry_dte": int(item.entry_dte),
                "entry_dte_bin": _dte_bin(str(item.strategy), int(item.entry_dte)),
                "previous_close_atm_iv": float(item.atm_iv),
                "previous_close_atm_iv_quartile": str(item.previous_close_atm_iv_quartile),
                "spread_group": str(item.spread_group),
                "valid_strategy_construction": bool(item.cheap_iv),
                "qualifying_signal": bool(item.broad_conflict and item.cheap_iv),
                "cheap_iv": bool(item.cheap_iv),
                "route_resolution_state": str(item.route_resolution_state),
                "recent_registered_completion_prior_6": bool(
                    item.recent_registered_completion_prior_6
                ),
                "hidden_2_3_2_prior_6": bool(item.hidden_2_3_2_prior_6),
                "any_hidden_event_prior_6": bool(item.any_hidden_event_prior_6),
                "underlying_next_session_return": underlying_return,
                "underlying_next_session_absolute_return": abs(underlying_return),
                "call_contract_id": str(item.call_contract_id),
                "put_contract_id": str(item.put_contract_id),
                "expiration_date": str(item.expiration_date),
                "strike": float(item.strike),
                "contract_multiplier": call_multiplier,
                "call_entry_ask": entry_quotes["call_ask"],
                "put_entry_ask": entry_quotes["put_ask"],
                "call_exit_bid": exit_quotes["call_bid"],
                "put_exit_bid": exit_quotes["put_bid"],
                "commission_per_contract_side": 0.75,
                "sensitivity_commission_per_contract_side": 1.00,
                "sensitivity_net_pnl": sensitivity["net_pnl"],
                "sensitivity_return_on_entry_debit": sensitivity["return_on_entry_debit"],
                **intrinsic,
                "win": primary["net_pnl"] > 0.0,
                **primary,
            }
        )
    selected["economics_status"] = selected["selection_id"].map(
        lambda value: economics_outcomes.get(str(value), ("not_applicable", ""))[0]
    )
    selected["economics_rejection_reason"] = selected["selection_id"].map(
        lambda value: economics_outcomes.get(str(value), ("not_applicable", ""))[1]
    )
    quote_audit = pd.DataFrame(
        quote_audit_rows,
        columns=empty_selection_quote_audit().columns,
    )
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        empty_trades = empty_trade_artifacts()[1]
        return selected, empty_trades, empty_trades.copy(), quote_audit
    candidates["large_underlying_movement"] = False
    candidates["previous_close_iv_group"] = "low"
    for _group_key, indices in candidates.groupby(["strategy", "symbol"], sort=True).groups.items():
        index = list(indices)
        development = candidates.loc[index,].loc[candidates.loc[index, "period"].eq("development")]
        if development.empty:
            continue
        movement_q75 = float(development["underlying_next_session_absolute_return"].quantile(0.75))
        iv_q25 = float(development["previous_close_atm_iv"].quantile(0.25))
        candidates.loc[index, "large_underlying_movement"] = candidates.loc[
            index, "underlying_next_session_absolute_return"
        ].ge(movement_q75)
        candidates.loc[index, "previous_close_iv_group"] = np.where(
            candidates.loc[index, "previous_close_atm_iv"].le(iv_q25),
            "very_low",
            "low",
        )
    treated_ids = candidates.loc[candidates["qualifying_signal"].astype(bool), "trade_id"].astype(
        str
    )
    if treated_ids.empty:
        return selected, candidates.iloc[:0].copy(), candidates, quote_audit
    trades = candidates.loc[candidates["trade_id"].isin(treated_ids)].copy()
    trades["control_count"] = 0
    trades["matched"] = False
    trades["control_trade_ids"] = "[]"
    trades["control_mean_return"] = np.nan
    trades["matched_control_excess"] = np.nan
    return (
        selected.sort_values(["strategy", "signal_session", "symbol"], kind="mergesort"),
        trades.sort_values(["strategy", "session", "symbol"], kind="mergesort").reset_index(
            drop=True
        ),
        candidates.sort_values(["strategy", "session", "symbol"], kind="mergesort").reset_index(
            drop=True
        ),
        quote_audit,
    )


def _support_gate_reasons(trades: pd.DataFrame, strategy: str) -> list[str]:
    assessment = trades.loc[trades["strategy"].eq(strategy) & trades["period"].eq("assessment")]
    reasons: list[str] = []
    maximum_stock_share = (
        float(assessment["symbol"].value_counts(normalize=True).max())
        if not assessment.empty
        else math.inf
    )
    if strategy == "S1":
        gates = (
            (len(assessment) >= 60, "minimum_60_trades"),
            (assessment["session"].nunique() >= 40, "minimum_40_sessions"),
            (assessment["symbol"].nunique() >= 10, "minimum_10_stocks"),
            (assessment["calendar_month"].nunique() >= 5, "minimum_5_months"),
            (maximum_stock_share <= 0.20, "maximum_stock_share_20pct"),
        )
    else:
        gates = (
            (len(assessment) >= 40, "minimum_40_trades"),
            (assessment["session"].nunique() >= 30, "minimum_30_sessions"),
            (assessment["symbol"].nunique() >= 8, "minimum_8_stocks"),
            (assessment["calendar_month"].nunique() >= 4, "minimum_4_months"),
            (maximum_stock_share <= 0.25, "maximum_stock_share_25pct"),
        )
    for passed, name in gates:
        if not passed:
            reasons.append(name)
    return reasons


def attach_supported_matched_controls(
    trades: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Run frozen matching only for S1/S3 after their support gates pass."""

    output = trades.copy()
    executed = {"S1": False, "S3": False}
    for strategy in ("S1", "S3"):
        if _support_gate_reasons(output, strategy):
            continue
        treated_ids = output.loc[output["strategy"].eq(strategy), "trade_id"].astype(str)
        if treated_ids.empty:
            continue
        strategy_candidates = candidates.loc[candidates["strategy"].eq(strategy)].copy()
        matches = build_matched_controls(
            strategy_candidates,
            treated_trade_ids=treated_ids.tolist(),
        ).set_index("treated_trade_id")
        mask = output["strategy"].eq(strategy)
        for column in (
            "control_count",
            "matched",
            "control_trade_ids",
            "control_mean_return",
            "matched_control_excess",
        ):
            output.loc[mask, column] = output.loc[mask, "trade_id"].map(matches[column])
        executed[strategy] = True
    return output, executed


def _support_and_positive(
    trades: pd.DataFrame,
    bootstrap: pd.DataFrame,
    strategy: str,
) -> tuple[bool, bool, list[str]]:
    assessment = trades.loc[trades["strategy"].eq(strategy) & trades["period"].eq("assessment")]
    development = trades.loc[trades["strategy"].eq(strategy) & trades["period"].eq("development")]
    reasons = _support_gate_reasons(trades, strategy)
    support = not reasons
    if not support or assessment.empty or development.empty:
        return support, False, reasons
    assessment_metrics = strategy_metrics(assessment)
    development_metrics = strategy_metrics(development)
    statistic = "s1_mean_return_on_debit" if strategy == "S1" else "s3_mean_return_on_debit"
    maximum_stock_share = float(assessment["symbol"].value_counts(normalize=True).max())
    maximum_month_share = float(assessment["calendar_month"].value_counts(normalize=True).max())
    months_positive = int(
        assessment.groupby("calendar_month", sort=True)["return_on_entry_debit"]
        .mean()
        .gt(0.0)
        .sum()
    )
    bootstrap_row = bootstrap.loc[
        bootstrap["statistic"].eq(statistic) & bootstrap["level"].eq(0.80)
    ]
    lower = float(bootstrap_row.iloc[0]["lower"]) if not bootstrap_row.empty else math.nan
    coverage = float(assessment["matched"].fillna(False).astype(bool).mean())
    matched_excess = pd.to_numeric(assessment["matched_control_excess"], errors="coerce").mean()
    positive = (
        float(assessment_metrics["mean_return_on_debit"]) > 0.0
        and float(assessment_metrics["median_return_on_debit"]) >= -0.05
        and np.sign(float(assessment_metrics["mean_return_on_debit"]))
        == np.sign(float(development_metrics["mean_return_on_debit"]))
        and months_positive >= 4
        and math.isfinite(lower)
        and lower >= 0.0
        and (coverage < 0.70 or (pd.notna(matched_excess) and float(matched_excess) > 0.0))
        and maximum_stock_share <= (0.20 if strategy == "S1" else 0.25)
        and maximum_month_share <= MAXIMUM_MONTH_SHARE
        and float(assessment_metrics["top_5pct_positive_pnl_contribution"]) < 1.0
    )
    return support, bool(positive), reasons


def economic_metric_artifacts(
    trades: pd.DataFrame,
    *,
    direction_blocked: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, bool], pd.DataFrame]:
    support_by_strategy = {
        strategy: not _support_gate_reasons(trades, strategy) for strategy in ("S1", "S3")
    }
    supported_names = {strategy for strategy, supported in support_by_strategy.items() if supported}
    bootstrap_input = trades.loc[trades["strategy"].isin(supported_names)].copy()
    if "matched_control_excess" not in bootstrap_input:
        bootstrap_input["matched_control_excess"] = np.nan
    full_bootstrap = session_bootstrap_intervals(bootstrap_input)
    allowed_bootstrap_statistics = {
        statistic
        for strategy in supported_names
        for statistic in (
            (
                "s1_mean_return_on_debit",
                "s1_matched_control_excess",
            )
            if strategy == "S1"
            else (
                "s3_mean_return_on_debit",
                "s3_matched_control_excess",
            )
        )
    }
    bootstrap = full_bootstrap.loc[
        full_bootstrap["statistic"].isin(allowed_bootstrap_statistics)
    ].reset_index(drop=True)
    strategy_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    stock_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    statuses: dict[str, str] = {
        "S1": "insufficient_support",
        "S2_ALL": (
            "blocked_direction_mapping_unavailable" if direction_blocked else "insufficient_support"
        ),
        "S2_VETO": (
            "blocked_direction_mapping_unavailable" if direction_blocked else "insufficient_support"
        ),
        "S3": "insufficient_support",
    }
    positives = {"S1": False, "S2": False, "S3": False, "S2_hidden_veto": False}
    for strategy in STRATEGIES:
        for period in ("development", "assessment"):
            frame = trades.loc[trades["strategy"].eq(strategy) & trades["period"].eq(period)]
            blocked_direction = strategy.startswith("S2") and direction_blocked
            values = (
                {name: math.nan for name in _empty_metrics()}
                if blocked_direction
                else _metric_values(frame)
            )
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    "scope": "overall",
                    "group_type": "overall",
                    "group_value": "all",
                    "status": statuses[strategy],
                    "blocker": (
                        "blocked_direction_mapping_unavailable"
                        if strategy.startswith("S2") and direction_blocked
                        else ""
                    ),
                    **values,
                }
            )
            subgroup_columns = (
                "entry_dte_bin",
                "previous_close_atm_iv_quartile",
                "spread_group",
                "large_underlying_movement",
                "previous_close_iv_group",
                "recent_registered_completion_prior_6",
            )
            for column in subgroup_columns:
                if column not in frame:
                    continue
                for value, group in frame.groupby(column, dropna=False, sort=True):
                    strategy_rows.append(
                        {
                            "strategy": strategy,
                            "period": period,
                            "scope": "subgroup",
                            "group_type": column,
                            "group_value": str(value),
                            "status": statuses[strategy],
                            "blocker": "",
                            **_metric_values(group),
                        }
                    )
        assessment = trades.loc[trades["strategy"].eq(strategy) & trades["period"].eq("assessment")]
        for month in pd.period_range("2025-01", "2025-08", freq="M").astype(str):
            group = assessment.loc[assessment["calendar_month"].eq(month)]
            blocked_direction = strategy.startswith("S2") and direction_blocked
            monthly_rows.append(
                {
                    "strategy": strategy,
                    "period": "assessment",
                    "calendar_month": month,
                    "trades": math.nan if blocked_direction else len(group),
                    "mean_return_on_debit": (
                        float(group["return_on_entry_debit"].mean())
                        if not group.empty
                        else math.nan
                    ),
                    "mean_net_pnl": (
                        float(group["net_pnl"].mean()) if not group.empty else math.nan
                    ),
                    "status": statuses[strategy],
                    "blocker": (
                        "blocked_direction_mapping_unavailable"
                        if strategy.startswith("S2") and direction_blocked
                        else ""
                    ),
                }
            )
        for symbol, group in assessment.groupby("symbol", sort=True):
            if len(group) < 10:
                continue
            stock_rows.append(
                {
                    "strategy": strategy,
                    "period": "assessment",
                    "symbol": str(symbol),
                    "status": statuses[strategy],
                    **_metric_values(group),
                }
            )
        coverage = (
            float(assessment["matched"].fillna(False).astype(bool).mean())
            if not assessment.empty
            else 0.0
        )
        matched_rows.append(
            {
                "strategy": strategy,
                "period": "assessment",
                "trades": (
                    math.nan if strategy.startswith("S2") and direction_blocked else len(assessment)
                ),
                "matched_trades": (
                    math.nan
                    if strategy.startswith("S2") and direction_blocked
                    else int(assessment["matched"].fillna(False).astype(bool).sum())
                    if not assessment.empty
                    else 0
                ),
                "coverage": (
                    math.nan if strategy.startswith("S2") and direction_blocked else coverage
                ),
                "claim_coverage_passed": (
                    False if strategy.startswith("S2") and direction_blocked else coverage >= 0.70
                ),
                "matched_control_excess": (
                    float(
                        pd.to_numeric(assessment["matched_control_excess"], errors="coerce").mean()
                    )
                    if not assessment.empty
                    else math.nan
                ),
                "status": statuses[strategy],
                "blocker": (
                    "blocked_direction_mapping_unavailable"
                    if strategy.startswith("S2") and direction_blocked
                    else ""
                ),
            }
        )
        metrics = _metric_values(assessment)
        if strategy.startswith("S2") and direction_blocked:
            metrics = {name: math.nan for name in _empty_metrics()}
        maximum_stock_allowed = 0.20 if strategy == "S1" else 0.25 if strategy == "S3" else 0.0
        concentration_passed = bool(
            not assessment.empty
            and float(metrics["maximum_stock_share"]) <= maximum_stock_allowed
            and float(metrics["maximum_month_share"]) <= MAXIMUM_MONTH_SHARE
            and float(metrics["top_5pct_positive_pnl_contribution"]) < 1.0
        )
        concentration_rows.append(
            {
                "strategy": strategy,
                "period": "assessment",
                "maximum_stock_share": metrics["maximum_stock_share"],
                "maximum_month_share": metrics["maximum_month_share"],
                "top_5pct_positive_pnl_contribution": metrics["top_5pct_positive_pnl_contribution"],
                "concentration_gate_passed": concentration_passed,
                "status": statuses[strategy],
                "blocker": (
                    "blocked_direction_mapping_unavailable"
                    if strategy.startswith("S2") and direction_blocked
                    else ""
                ),
            }
        )
    for strategy in ("S1", "S3"):
        support, positive, _reasons = _support_and_positive(trades, bootstrap, strategy)
        positives[strategy] = positive
        statuses[strategy] = (
            "supported" if positive else "not_supported" if support else "insufficient_support"
        )
    strategy_frame = pd.DataFrame(strategy_rows)
    for strategy, status in statuses.items():
        strategy_frame.loc[strategy_frame["strategy"].eq(strategy), "status"] = status
    monthly_frame = pd.DataFrame(monthly_rows)
    for strategy, status in statuses.items():
        monthly_frame.loc[monthly_frame["strategy"].eq(strategy), "status"] = status
    matched_frame = pd.DataFrame(matched_rows)
    concentration_frame = pd.DataFrame(concentration_rows)
    stock_frame = pd.DataFrame(stock_rows)
    for target in (stock_frame, matched_frame, concentration_frame):
        for strategy, status in statuses.items():
            if not target.empty:
                target.loc[target["strategy"].eq(strategy), "status"] = status
    veto = pd.DataFrame(
        [
            {
                "period": "assessment",
                "s2_all_trades": math.nan,
                "s2_veto_trades": math.nan,
                "mean_return_increment": math.nan,
                "win_rate_increment": math.nan,
                "positive_assessment_months": math.nan,
                "veto_passed": False,
                "status": statuses["S2_VETO"],
                "blocker": ("blocked_direction_mapping_unavailable" if direction_blocked else ""),
            }
        ]
    )
    artifacts = {
        "strategy_metrics.csv": strategy_frame,
        "monthly_metrics.csv": monthly_frame,
        "stock_metrics.csv": stock_frame,
        "matched_control_metrics.csv": matched_frame,
        "veto_metrics.csv": veto,
        "bootstrap_metrics.csv": bootstrap,
        "concentration_metrics.csv": concentration_frame,
    }
    return artifacts, statuses, positives, bootstrap


def blocked_metric_artifacts(
    blocker: str,
) -> dict[str, pd.DataFrame]:
    metric_names = [
        "trades",
        "sessions",
        "stocks",
        "months",
        "total_entry_debit",
        "mean_net_pnl",
        "median_net_pnl",
        "trimmed_mean_net_pnl",
        "mean_return_on_debit",
        "median_return_on_debit",
        "win_rate",
        "full_loss_rate",
        "maximum_gain",
        "maximum_loss",
        "return_p05",
        "return_p25",
        "return_p75",
        "return_p95",
        "matched_control_excess",
        "maximum_stock_share",
        "maximum_month_share",
        "top_5pct_positive_pnl_contribution",
    ]
    strategy_rows: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        for period in ("development", "assessment"):
            groups: list[tuple[str, str]] = [("overall", "all")]
            groups.extend(
                ("entry_dte_bin", value)
                for value in (("1",) if strategy == "S3" else ("7-9", "10-12", "13-14"))
            )
            groups.extend(("cheap_iv_quartile", value) for value in ("Q1", "Q2", "Q3", "Q4"))
            groups.extend(("spread_group", value) for value in ("tight", "wide"))
            if strategy in {"S1", "S3"}:
                groups.extend(
                    (
                        ("next_session_large_underlying_movement", "yes"),
                        ("next_session_large_underlying_movement", "no"),
                        ("previous_close_iv_group", "low"),
                        ("previous_close_iv_group", "very_low"),
                        ("recent_registered_completion", "yes"),
                        ("recent_registered_completion", "no"),
                    )
                )
            else:
                groups.extend(
                    (
                        ("direction", "bullish"),
                        ("direction", "bearish"),
                        ("directional_agreement_strength", "lower"),
                        ("directional_agreement_strength", "higher"),
                        ("hidden_2_3_2_prior_6", "yes"),
                        ("hidden_2_3_2_prior_6", "no"),
                        ("any_hidden_event_prior_6", "yes"),
                        ("any_hidden_event_prior_6", "no"),
                        ("recent_registered_completion", "yes"),
                        ("recent_registered_completion", "no"),
                    )
                )
            for group_type, group_value in groups:
                row: dict[str, object] = {
                    "strategy": strategy,
                    "period": period,
                    "scope": "overall" if group_type == "overall" else "subgroup",
                    "group_type": group_type,
                    "group_value": group_value,
                    "status": "blocked",
                    "blocker": blocker,
                }
                row.update(
                    {
                        metric: (
                            0 if metric in {"trades", "sessions", "stocks", "months"} else np.nan
                        )
                        for metric in metric_names
                    }
                )
                strategy_rows.append(row)
    months = pd.period_range("2025-01", "2025-08", freq="M").astype(str)
    monthly = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "period": "assessment",
                "calendar_month": month,
                "trades": 0,
                "mean_return_on_debit": np.nan,
                "mean_net_pnl": np.nan,
                "status": "blocked",
                "blocker": blocker,
            }
            for strategy in STRATEGIES
            for month in months
        ]
    )
    stock = pd.DataFrame(
        columns=[
            "strategy",
            "period",
            "symbol",
            "trades",
            "mean_return_on_debit",
            "mean_net_pnl",
            "status",
            "blocker",
        ]
    )
    matched = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "period": "assessment",
                "trades": 0,
                "matched_trades": 0,
                "coverage": 0.0,
                "claim_coverage_passed": False,
                "matched_control_excess": np.nan,
                "status": "blocked",
                "blocker": blocker,
            }
            for strategy in STRATEGIES
        ]
    )
    veto = pd.DataFrame(
        [
            {
                "period": "assessment",
                "s2_all_trades": 0,
                "s2_veto_trades": 0,
                "mean_return_increment": np.nan,
                "win_rate_increment": np.nan,
                "positive_assessment_months": 0,
                "veto_passed": False,
                "status": "blocked",
                "blocker": "blocked_direction_mapping_unavailable",
            }
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "statistic": statistic,
                "level": level,
                "lower": np.nan,
                "upper": np.nan,
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "status": "not_run_blocked_no_trades",
                "blocker": blocker,
            }
            for statistic in BOOTSTRAP_STATISTICS
            for level in (0.80, 0.90, 0.95)
        ]
    )
    concentration = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "period": "assessment",
                "maximum_stock_share": np.nan,
                "maximum_month_share": np.nan,
                "top_5pct_positive_pnl_contribution": np.nan,
                "concentration_gate_passed": False,
                "status": "blocked",
                "blocker": blocker,
            }
            for strategy in STRATEGIES
        ]
    )
    return {
        "strategy_metrics.csv": pd.DataFrame(strategy_rows),
        "monthly_metrics.csv": monthly,
        "stock_metrics.csv": stock,
        "matched_control_metrics.csv": matched,
        "veto_metrics.csv": veto,
        "bootstrap_metrics.csv": bootstrap,
        "concentration_metrics.csv": concentration,
    }


def contract_preselection_manifest(
    *,
    required: pd.DataFrame,
    gaps: pd.DataFrame,
    blocker: str,
    selected_count: int,
) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS_V01,
        "contract_selection_date_rule": "exact_previous_us_trading_session_D_minus_1",
        "same_session_closing_chain_used_for_identity_selection": False,
        "same_session_greeks_used_for_selection": False,
        "older_chain_forward_filled": False,
        "selected_contract_replacement_at_entry": False,
        "S1": {
            "entry_dte": [7, 14],
            "selection_date_expected_dte": [8, 15],
            "strike_rule": "minimum_abs_log_strike_over_previous_close",
            "tie_break": [
                "highest_minimum_open_interest",
                "narrowest_combined_relative_spread",
                "smallest_iv_gap",
                "strike",
                "contract_ids_lexicographic",
            ],
        },
        "S2": {
            "status": "blocked_direction_mapping_unavailable",
            "both_sides_preselected": True,
            "long_absolute_delta": 0.50,
            "short_absolute_delta": 0.25,
            "maximum_absolute_delta_error": 0.10,
            "tie_break": [
                "highest_minimum_open_interest",
                "narrowest_combined_relative_spread",
                "smallest_total_delta_error",
                "contract_ids_lexicographic",
            ],
        },
        "S3": {
            "entry_dte": [1, 1],
            "selection_date_expected_dte": [2, 2],
            "strike_rule": "same_frozen_atm_rule_as_S1",
        },
        "required_option_date_rows": len(required),
        "incomplete_strategy_date_chains": len(gaps),
        "unique_incomplete_symbol_dates": (
            int(gaps[["symbol", "option_date"]].drop_duplicates().shape[0]) if not gaps.empty else 0
        ),
        "selected_contracts": selected_count,
        "blocker": blocker,
    }


def report_text(
    *,
    decision: Mapping[str, Any],
    signals: pd.DataFrame,
    required: pd.DataFrame,
    gaps: pd.DataFrame,
    cache: Mapping[str, Any],
    structural_manifest: Mapping[str, Any],
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    metric_artifacts: Mapping[str, pd.DataFrame],
) -> str:
    route_counts = (
        signals.loc[
            signals["ordinal_72_structural_available"].astype(bool), "route_resolution_state"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    period_counts = (
        signals.loc[signals["ordinal_72_structural_available"].astype(bool), "period"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    blocked = decision.get("primary_blocker") is not None
    blocker_detail = str(decision.get("blocker_detail") or "unspecified fail-closed blocker")
    decision_summary = (
        f"S1 and S3 ended with a shared pipeline blocker: {blocker_detail}.\n"
        "S2 independently remains blocked because the repository has no audited\n"
        "orientation-to-price-direction mapping. Any partially constructable rows are\n"
        "descriptive diagnostics and are not used for an economic claim."
        if blocked
        else (
            f"Constructed {len(trades)} treated strategy trades from "
            f"{int(selected.get('selection_status', pd.Series(dtype=str)).eq('selected').sum())} "
            "causally preselected contract pairs. S2 remains blocked because the repository "
            "has no audited orientation-to-price-direction mapping."
        )
    )
    economics_summary = (
        "No economic feasibility claim is made under the shared blocker. Blocked S2\n"
        "artifacts contain status fields and null numerical values rather than\n"
        "fabricated zero results."
        if blocked
        else (
            "The strategy, monthly, stock, matched-control, veto, bootstrap, and "
            "concentration artifacts contain the frozen, untuned economics. "
            "All option entries/exits use the prescribed bid/ask sides."
        )
    )
    extra_discarded = cache["extra_date_records_discarded"]
    post_boundary_discarded = cache["post_boundary_records_discarded"]
    records_by_source = cast(Mapping[str, int], cache["exact_date_records_by_source"])

    def report_value(value: object, *, kind: str = "number") -> str:
        try:
            number = float(cast(Any, value))
        except (TypeError, ValueError):
            return "NA"
        if not math.isfinite(number):
            return "NA"
        if kind == "count":
            return str(int(number))
        if kind == "money":
            return f"${number:,.2f}"
        if kind == "percent":
            return f"{100.0 * number:.2f}%"
        return f"{number:.4f}"

    strategy_frame = metric_artifacts["strategy_metrics.csv"]
    overall_metrics = strategy_frame.loc[
        strategy_frame["scope"].eq("overall") & strategy_frame["strategy"].isin(STRATEGIES)
    ]
    economic_rows = [
        "| Strategy | Period | Status | Trades | Mean P&L | Mean return | "
        "Median return | Win rate | $1 mean P&L | $1 mean return |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    strategy_order = {strategy: index for index, strategy in enumerate(STRATEGIES)}
    period_order = {"development": 0, "assessment": 1}
    ordered_overall = overall_metrics.assign(
        _strategy_order=overall_metrics["strategy"].map(strategy_order),
        _period_order=overall_metrics["period"].map(period_order),
    ).sort_values(["_strategy_order", "_period_order"], kind="mergesort")
    for item in ordered_overall.itertuples(index=False):
        sensitivity = trades.loc[
            trades["strategy"].astype(str).eq(str(item.strategy))
            & trades["period"].astype(str).eq(str(item.period))
        ]
        sensitivity_mean_pnl = (
            pd.to_numeric(sensitivity["sensitivity_net_pnl"], errors="raise").mean()
            if not sensitivity.empty
            else math.nan
        )
        sensitivity_mean_return = (
            pd.to_numeric(sensitivity["sensitivity_return_on_entry_debit"], errors="raise").mean()
            if not sensitivity.empty
            else math.nan
        )
        economic_rows.append(
            "| "
            + " | ".join(
                (
                    str(item.strategy),
                    str(item.period),
                    str(item.status),
                    report_value(item.trades, kind="count"),
                    report_value(item.mean_net_pnl, kind="money"),
                    report_value(item.mean_return_on_debit, kind="percent"),
                    report_value(item.median_return_on_debit, kind="percent"),
                    report_value(item.win_rate, kind="percent"),
                    report_value(sensitivity_mean_pnl, kind="money"),
                    report_value(sensitivity_mean_return, kind="percent"),
                )
            )
            + " |"
        )
    economic_table = "\n".join(economic_rows)

    monthly_frame = metric_artifacts["monthly_metrics.csv"]
    monthly_assessment = monthly_frame.loc[
        monthly_frame["period"].eq("assessment")
        & monthly_frame["strategy"].isin(["S1", "S3"])
        & pd.to_numeric(monthly_frame["trades"], errors="coerce").gt(0)
    ].sort_values(["strategy", "calendar_month"], kind="mergesort")
    monthly_rows = [
        "| Strategy | Month | Trades | Mean return | Mean P&L |",
        "|---|---|---:|---:|---:|",
    ]
    for item in monthly_assessment.itertuples(index=False):
        monthly_rows.append(
            "| "
            + " | ".join(
                (
                    str(item.strategy),
                    str(item.calendar_month),
                    report_value(item.trades, kind="count"),
                    report_value(item.mean_return_on_debit, kind="percent"),
                    report_value(item.mean_net_pnl, kind="money"),
                )
            )
            + " |"
        )
    if len(monthly_rows) == 2:
        monthly_rows.append("| NA | NA | 0 | NA | NA |")
    monthly_table = "\n".join(monthly_rows)

    matched_frame = metric_artifacts["matched_control_metrics.csv"]
    concentration_frame = metric_artifacts["concentration_metrics.csv"]
    coverage_rows = [
        "| Strategy | Status | Matched coverage | Matched excess | Max stock | "
        "Max month | Top-5% positive P&L |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in matched_frame.loc[matched_frame["period"].eq("assessment")].itertuples(index=False):
        concentration = concentration_frame.loc[
            concentration_frame["strategy"].eq(str(item.strategy))
            & concentration_frame["period"].eq("assessment")
        ]
        concentration_row = concentration.iloc[0] if len(concentration) == 1 else {}
        coverage_rows.append(
            "| "
            + " | ".join(
                (
                    str(item.strategy),
                    str(item.status),
                    report_value(item.coverage, kind="percent"),
                    report_value(item.matched_control_excess, kind="percent"),
                    report_value(
                        cast(Any, concentration_row).get("maximum_stock_share"),
                        kind="percent",
                    ),
                    report_value(
                        cast(Any, concentration_row).get("maximum_month_share"),
                        kind="percent",
                    ),
                    report_value(
                        cast(Any, concentration_row).get("top_5pct_positive_pnl_contribution"),
                        kind="percent",
                    ),
                )
            )
            + " |"
        )
    coverage_table = "\n".join(coverage_rows)

    bootstrap_frame = metric_artifacts["bootstrap_metrics.csv"]
    bootstrap_rows = [
        "| Statistic | Interval | Lower | Upper | Draws |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in bootstrap_frame.itertuples(index=False):
        bootstrap_rows.append(
            "| "
            + " | ".join(
                (
                    str(item.statistic),
                    report_value(item.level, kind="percent"),
                    report_value(item.lower, kind="percent"),
                    report_value(item.upper, kind="percent"),
                    report_value(item.draws, kind="count"),
                )
            )
            + " |"
        )
    if len(bootstrap_rows) == 2:
        bootstrap_rows.append("| No supported strategy | NA | NA | NA | 0 |")
    bootstrap_table = "\n".join(bootstrap_rows)

    diagnostic_frame = strategy_frame.loc[
        strategy_frame["period"].eq("assessment")
        & strategy_frame["group_type"].isin(["entry_dte_bin", "spread_group"])
        & pd.to_numeric(strategy_frame["trades"], errors="coerce").gt(0)
    ].sort_values(["strategy", "group_type", "group_value"], kind="mergesort")
    diagnostic_rows = [
        "| Strategy | Diagnostic | Group | Trades | Mean return | Median return |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in diagnostic_frame.itertuples(index=False):
        diagnostic_rows.append(
            "| "
            + " | ".join(
                (
                    str(item.strategy),
                    str(item.group_type),
                    str(item.group_value),
                    report_value(item.trades, kind="count"),
                    report_value(item.mean_return_on_debit, kind="percent"),
                    report_value(item.median_return_on_debit, kind="percent"),
                )
            )
            + " |"
        )
    if len(diagnostic_rows) == 2:
        diagnostic_rows.append("| NA | NA | NA | 0 | NA | NA |")
    diagnostic_table = "\n".join(diagnostic_rows)

    return f"""# EODHD Fixed Overnight Options Strategy Quick Screen V0.1

## Decision

`{decision["overall_decision"]}`

{decision_summary}

## Frozen stock signal

- Cohort: {len(FROZEN_COHORT)} stocks.
- Regular-session stock-state rows: {len(signals)}.
- Ordinal-72 structural rows: {int(signals["ordinal_72_structural_available"].sum())}.
- Period support: `{json.dumps(period_counts, sort_keys=True)}`.
- Route states: `{json.dumps(route_counts, sort_keys=True)}`.
- Broad-conflict candidates before IV: {int(signals["broad_conflict_candidate_pre_iv"].sum())}.
- Source exclusions: `{json.dumps(structural_manifest["structural_exclusions"], sort_keys=True)}`.
- Protected market or option rows materialised: 0.

Ordinal 72 is the completed-bar count: zero-based bar 71 starts at 15:25 New
York time and becomes available at 15:30. Normal sessions retain six complete
five-minute bars before the scheduled close; sessions without the required
late bar are retained as unavailable rows.

## Options coverage

- Required option date rows: {len(required)}.
- Remaining unavailable strategy-date chains: {len(gaps)}.
- Cached raw responses examined: {cache["cached_responses_examined"]}.
- Cached responses recovered: {cache["cached_responses_recovered"]}.
- Exact-date records recovered: {cache["cached_exact_date_records_recovered"]}.
- V0 cached exact-date records: {records_by_source[V0_OPTIONS_RUN_DIR_NAME]}.
- V0.1 acquired exact-date records: {records_by_source[V01_OPTIONS_RUN_DIR_NAME]}.
- Other-date records discarded before materialisation: {extra_discarded}.
- Post-boundary records discarded before materialisation: {post_boundary_discarded}.
- Provider DTE disagreements audited: {cache["provider_dte_disagreement_records"]}.
- Unquotable exact-date rows discarded: {cache["unquotable_exact_date_records_discarded"]}.
- Safe canonical pre-boundary option observations: {cache["canonical_pre_boundary_records"]}.
- Cumulative V0.1 repair download records: {cache["newly_downloaded_records"]}.
- Cumulative V0.1 repair download bytes: {cache["newly_downloaded_bytes"]}.

Contract expiration beyond 2025-08-22 was retained only as causal metadata.
No post-boundary quote observation was materialised, and no contract was
reselected, replaced, or forward-filled.

## Economic results

{economics_summary}

{economic_table}

The `$1` columns are the frozen descriptive commission sensitivity per contract,
per side, per leg. Blocked S2 rows contain no fabricated numerical results.

### Assessment months

{monthly_table}

### Matched controls and concentration

{coverage_table}

### Fixed-seed whole-session bootstrap

{bootstrap_table}

Ten draws are a coarse stability diagnostic, not precise inference. `NA`
matched-control intervals reflect failed matching coverage, not zero excess.

### DTE and spread diagnostics

{diagnostic_table}

This report is a retrospective research screen. It does not claim intraday
option execution, IBKR fills, realised profits, prospective validation, or a
deployable strategy.
"""


def evaluate_cached_economics(
    *,
    signals: pd.DataFrame,
    options: pd.DataFrame,
    strategy_blockers: Mapping[str, str | None],
    global_blocker: tuple[str, str] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
    str | None,
]:
    """Evaluate S1/S3 independently while preserving the frozen S2 blocker."""

    selected, trades, candidates, selection_quote_audit = construct_straddle_economics(
        signals, options
    )
    trades, matching_executed = attach_supported_matched_controls(trades, candidates)
    metrics, statuses, positives, _bootstrap = economic_metric_artifacts(
        trades, direction_blocked=True
    )
    for strategy in ("S1", "S3"):
        blocker = strategy_blockers.get(strategy)
        if blocker is None:
            continue
        statuses[strategy] = blocker
        positives[strategy] = False
        for frame in metrics.values():
            if "strategy" not in frame:
                continue
            mask = frame["strategy"].astype(str).eq(strategy)
            frame.loc[mask, "status"] = blocker
            if "blocker" in frame:
                frame.loc[mask, "blocker"] = blocker
    fatal_name = global_blocker[0] if global_blocker is not None else None
    fatal_detail = global_blocker[1] if global_blocker is not None else None
    if fatal_name is None:
        s1_blocker = strategy_blockers.get("S1")
        s3_blocker = strategy_blockers.get("S3")
        if s1_blocker is not None and s1_blocker == s3_blocker:
            fatal_name = {
                "blocked_resource_limit": "blocked_quick_options_strategy_resource_limit",
                "blocked_contract_coverage": ("blocked_options_contract_reconstruction_failure"),
                "blocked_quote_integrity": "blocked_options_quote_integrity_failure",
                "blocked_chronology": "blocked_protected_boundary_failure",
            }.get(s1_blocker)
            if fatal_name is not None:
                fatal_detail = f"S1 and S3 independently ended with {s1_blocker}"
    overall = choose_overall_decision_v01(
        statuses=statuses,
        strategy_positive={
            "S1": positives["S1"],
            "S2": positives["S2"],
            "S3": positives["S3"],
        },
        hidden_veto_positive=positives["S2_hidden_veto"],
        fatal_blocker=fatal_name,
    )
    decision = {
        **SAFETY_FLAGS_V01,
        "overall_decision": overall,
        "s1_overnight_straddle_status": statuses["S1"],
        "s2_directional_spread_status": statuses["S2_ALL"],
        "s2_hidden_veto_status": statuses["S2_VETO"],
        "s3_dte1_straddle_status": statuses["S3"],
        "primary_blocker": fatal_name,
        "blocker_detail": fatal_detail,
        "additional_blockers": ["blocked_direction_mapping_unavailable"],
        "rough_screen_positive": positives,
        "matched_controls_executed": matching_executed,
        "strategy_status_reasons": {
            "S1": strategy_blockers.get("S1") or statuses["S1"],
            "S2": "blocked_direction_mapping_unavailable",
            "S2_hidden_veto": "blocked_direction_mapping_unavailable",
            "S3": strategy_blockers.get("S3") or statuses["S3"],
        },
    }
    return (
        selected,
        trades,
        candidates,
        selection_quote_audit,
        metrics,
        decision,
        fatal_name,
    )


def run(
    *,
    provider_root: Path,
    options_cache: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    repair_manifest = cast(
        dict[str, Any],
        json.loads((EXPERIMENT_DIR / "repair_manifest.json").read_text(encoding="utf-8")),
    )
    assert_safety_flags_v01(repair_manifest)
    write_json(output / "repair_manifest.json", repair_manifest)

    states, dictionary, state_source, dictionary_manifest = load_or_build_states(provider_root)
    signals, structural_manifest, _ledger = build_signal_ledger(
        states, dictionary, provider_root=provider_root
    )
    direction_audit = direction_mapping_audit()
    required = required_option_dates(signals)
    (
        canonical_options,
        exact_date_audit,
        cache_receipts,
        cache_reprocessing,
    ) = reprocess_cached_responses(
        required=required,
        signals=signals,
        options_cache=options_cache,
    )
    cached_reprocessing_before_download = dict(cache_reprocessing)
    initially_unresolved = unresolved_option_requests(required, signals, exact_date_audit)
    token = os.environ.get("EODHD_API_TOKEN", "").strip()
    token_available = bool(token)
    download_outcome = download_unresolved_option_requests(
        token=token,
        plans=initially_unresolved,
        options_cache=options_cache,
    )
    (
        canonical_options,
        exact_date_audit,
        cache_receipts,
        cache_reprocessing,
    ) = reprocess_cached_responses(
        required=required,
        signals=signals,
        options_cache=options_cache,
    )
    download_inventory = v01_download_inventory(options_cache)
    gaps = option_gaps_after_repair(required, signals, exact_date_audit)
    unresolved_gaps = gaps.loc[gaps["download_still_required"].astype(bool)]
    strategy_blockers: dict[str, str | None] = {"S1": None, "S3": None}
    for strategy in ("S1", "S3"):
        strategy_unresolved = unresolved_gaps["strategies"].astype(str).eq(strategy)
        if not bool(strategy_unresolved.any()):
            continue
        strategy_blockers[strategy] = (
            "blocked_resource_limit"
            if download_outcome.get("status") == "blocked_resource_limit"
            else "blocked_contract_coverage"
        )

    canonical_dir = options_cache / "canonical" / V01_OPTIONS_RUN_DIR_NAME
    canonical_path = canonical_dir / "exact-date-cache.parquet"
    write_parquet(canonical_path, canonical_options)

    quote_integrity = pd.DataFrame(
        [
            {
                "scope": "exact_date_cache",
                "check": "extra_dates_discarded_before_materialisation",
                "passed": bool(
                    cache_reprocessing["conflicting_duplicate_groups"] == 0
                    and cache_reprocessing["canonical_rejections"] == 0
                ),
                "rows": int(len(canonical_options)),
                "detail": (
                    f"{cache_reprocessing['extra_date_records_discarded']} other-date "
                    "records discarded before canonicalisation"
                ),
            },
            {
                "scope": "exact_date_cache",
                "check": "protected_observation_boundary",
                "passed": bool(
                    cache_reprocessing["protected_option_observations_materialised"] == 0
                ),
                "rows": int(cache_reprocessing["protected_option_observations_materialised"]),
                "detail": "expiration metadata was not treated as an observation date",
            },
            {
                "scope": "exact_date_cache",
                "check": "provider_dte_disagreement_audit",
                "passed": True,
                "rows": int(cache_reprocessing["provider_dte_disagreement_records"]),
                "detail": "DTE was derived from observation and expiration dates",
            },
            {
                "scope": "exact_date_cache",
                "check": "unquotable_rows_discarded_before_materialisation",
                "passed": True,
                "rows": int(cache_reprocessing["unquotable_exact_date_records_discarded"]),
                "detail": "rows without bid/ask timestamps cannot pass frozen quote gates",
            },
            {
                "scope": "strategy_specific_coverage",
                "check": "unresolved_requests_after_repair",
                "passed": bool(unresolved_gaps.empty),
                "rows": len(unresolved_gaps),
                "detail": "failed exact-date outcomes are per-stock-date exclusions",
            },
        ]
    )
    (
        selected,
        trades,
        control_candidates,
        selection_quote_integrity,
        metric_artifacts,
        decision,
        blocker,
    ) = evaluate_cached_economics(
        signals=signals,
        options=canonical_options,
        strategy_blockers=strategy_blockers,
    )
    quote_integrity = pd.concat(
        [quote_integrity, selection_quote_integrity],
        ignore_index=True,
        sort=False,
    )
    source_manifest = {
        **SAFETY_FLAGS_V01,
        "provider": "EODHD",
        "stock_timeframe": "5m",
        "stock_source_root": str(provider_root),
        "frozen_cohort": list(FROZEN_COHORT),
        "stock_date_minimum": DEVELOPMENT_START.isoformat(),
        "stock_date_maximum": READ_END.isoformat(),
        "stock_state_rows": len(signals),
        "protected_rows_materialised": 0,
        "protected_market_rows_materialised": 0,
        "protected_option_observations_materialised": 0,
        "state_source": state_source,
        "loop_dictionary": dictionary_manifest,
        "structural_manifest": structural_manifest,
        "implementation_sha256": {
            "run_screen_v01.py": sha256_file(Path(__file__).resolve()),
            "audit_screen_v01.py": sha256_file(EXPERIMENT_DIR / "audit_screen_v01.py"),
            "contract.json": sha256_file(CONTRACT_PATH),
            "repair_manifest.json": sha256_file(EXPERIMENT_DIR / "repair_manifest.json"),
            "eodhd_fixed_options_strategy_v01.py": sha256_file(
                REPO_ROOT
                / "packages"
                / "stocker_research"
                / "src"
                / "stocker_research"
                / "eodhd_fixed_options_strategy_v01.py"
            ),
        },
        "direction_mapping_available": False,
        "options_cache": {
            **cache_reprocessing,
            **download_inventory,
            "canonical_pre_boundary_records": len(canonical_options),
            "canonical_cache_path": str(canonical_path),
            "newly_downloaded_bytes": download_inventory["newly_downloaded_unique_raw_bytes"],
            "remaining_chain_gaps": len(gaps),
            "remaining_unresolved_requests": len(unresolved_gaps),
        },
        "option_download_attempted": bool(download_outcome["network_requests_made"]),
        "cumulative_option_download_attempted": bool(download_inventory["new_logical_requests"]),
        "option_download_outcome": download_outcome,
        "api_token_present": token_available,
        "api_token_present_for_current_invocation": token_available,
        "api_token_value_recorded": False,
        "corporate_action_handling": {
            "underlying_prices": "unadjusted EODHD five-minute session closes",
            "split_boundary_ratio_bounds": [0.55, 1.8],
            "ambiguous_boundaries_rejected": True,
            "prior_audit_path": str(PRIOR_OPTION_PRICE_AUDIT.relative_to(REPO_ROOT)),
            "prior_audit_sha256": sha256_file(PRIOR_OPTION_PRICE_AUDIT),
        },
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
    }
    protected = {
        **SAFETY_FLAGS_V01,
        "protected_start": PROTECTED_START.isoformat(),
        "maximum_allowed_observation_date": READ_END.isoformat(),
        "maximum_stock_signal_date": str(signals["session"].max()),
        "maximum_required_option_date": (
            str(required["option_date"].max()) if not required.empty else None
        ),
        "maximum_materialised_option_observation_date": (
            str(canonical_options["trade_date"].max()) if not canonical_options.empty else None
        ),
        "maximum_contract_expiration_metadata_date": (
            str(canonical_options["expiration_date"].max()) if not canonical_options.empty else None
        ),
        "expiration_crossing_contract_rows_retained": (
            int(pd.to_datetime(canonical_options["expiration_date"]).dt.date.gt(READ_END).sum())
            if not canonical_options.empty
            else 0
        ),
        "protected_market_rows_materialised": 0,
        "protected_option_observations_materialised": 0,
        "protected_rows_materialised": 0,
        "passed": True,
    }
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected)
    write_csv(output / "exact_date_filter_audit.csv", exact_date_audit)
    write_json(
        output / "cached_response_reprocessing.json",
        {
            **SAFETY_FLAGS_V01,
            **cache_reprocessing,
            **download_inventory,
            "before_network": cached_reprocessing_before_download,
            "initially_unresolved_requests": len(initially_unresolved),
            "remaining_unresolved_requests": len(unresolved_gaps),
        },
    )
    write_parquet(output / "late_day_stock_signal_ledger.parquet", signals)
    write_json(output / "direction_mapping_audit.json", direction_audit)
    write_csv(output / "required_option_dates.csv", required)
    write_csv(output / "options_download_gap_after_repair.csv", gaps)
    write_json(
        output / "options_download_manifest.json",
        {
            **SAFETY_FLAGS_V01,
            "cached_receipts": cache_receipts.to_dict(orient="records"),
            "network_outcome": download_outcome,
            "cumulative_v01_download": download_inventory,
            "api_token_value_recorded": False,
        },
    )
    write_json(
        output / "contract_preselection_manifest.json",
        contract_preselection_manifest(
            required=required,
            gaps=gaps,
            blocker=blocker or "none",
            selected_count=int(selected["selection_status"].eq("selected").sum())
            if "selection_status" in selected
            else 0,
        ),
    )
    write_parquet(output / "selected_contracts.parquet", selected)
    write_csv(output / "quote_integrity.csv", quote_integrity)
    write_parquet(output / "strategy_trade_ledger.parquet", trades)
    for filename, frame in metric_artifacts.items():
        write_csv(output / filename, frame)
    write_json(output / "decision.json", decision)

    first_signal_hash = frame_hash(signals)
    reloaded_states = pd.read_parquet(STATE_CACHE)
    _validate_state_frame(reloaded_states)
    reloaded_states["posterior_entropy"] = reloaded_states["posterior_entropy_reproduced"].astype(
        float
    )
    probabilities = reloaded_states.loc[:, [f"state_p_{state}" for state in range(8)]].to_numpy(
        dtype=float
    )
    ordered = np.sort(probabilities, axis=1)
    reloaded_states["top_state_probability"] = ordered[:, -1]
    reloaded_states["top_second_margin"] = ordered[:, -1] - ordered[:, -2]
    reloaded_states["historical_relative_activity"] = reloaded_states["volume"] / reloaded_states[
        "historical_volume_baseline_at_bar"
    ].replace(0.0, np.nan)
    rebuilt_signals, _rebuilt_manifest, _rebuilt_ledger = build_signal_ledger(
        reloaded_states, dictionary, provider_root=provider_root
    )
    second_signal_hash = frame_hash(rebuilt_signals)
    rebuilt_required = required_option_dates(rebuilt_signals)
    (
        reloaded_options,
        rebuilt_exact_audit,
        _rebuilt_receipts,
        _rebuilt_cache_summary,
    ) = reprocess_cached_responses(
        required=rebuilt_required,
        signals=rebuilt_signals,
        options_cache=options_cache,
    )
    rebuilt_gaps = option_gaps_after_repair(
        rebuilt_required,
        rebuilt_signals,
        rebuilt_exact_audit,
    )
    cached_reloaded = pd.read_parquet(canonical_path)
    cached_reloaded["trade_date"] = pd.to_datetime(
        cached_reloaded["trade_date"], errors="raise"
    ).dt.date
    cached_reloaded["expiration_date"] = pd.to_datetime(
        cached_reloaded["expiration_date"], errors="raise"
    ).dt.date
    (
        selected_rebuilt,
        trades_rebuilt,
        control_candidates_rebuilt,
        selection_quote_integrity_rebuilt,
        rebuilt_metric_artifacts,
        rebuilt_decision,
        rebuilt_blocker,
    ) = evaluate_cached_economics(
        signals=rebuilt_signals,
        options=reloaded_options,
        strategy_blockers=strategy_blockers,
    )
    quote_columns = [
        column for column in trades.columns if column.endswith("_bid") or column.endswith("_ask")
    ]
    pnl_columns = [
        column
        for column in (
            "entry_debit",
            "exit_credit",
            "commissions",
            "net_pnl",
            "total_initial_cash_debit",
            "return_on_entry_debit",
            "sensitivity_net_pnl",
            "sensitivity_return_on_entry_debit",
        )
        if column in trades
    ]

    def maximum_numeric_difference(columns: Sequence[str]) -> float:
        if not columns or trades.empty or trades_rebuilt.empty:
            return 0.0
        left = trades.set_index("trade_id").sort_index()
        right = trades_rebuilt.set_index("trade_id").sort_index()
        if list(left.index) != list(right.index):
            return math.inf
        differences = left.loc[:, list(columns)].to_numpy(dtype=float) - right.loc[
            :, list(columns)
        ].to_numpy(dtype=float)
        return float(np.max(np.abs(differences))) if differences.size else 0.0

    selected_mismatches = int(frame_hash(selected) != frame_hash(selected_rebuilt))
    exact_date_record_mismatches = int(
        frame_hash(canonical_options) != frame_hash(reloaded_options)
        or frame_hash(canonical_options) != frame_hash(cached_reloaded)
        or frame_hash(exact_date_audit) != frame_hash(rebuilt_exact_audit)
        or frame_hash(gaps) != frame_hash(rebuilt_gaps)
    )
    trade_identity_mismatches = int(
        sorted(trades.get("trade_id", pd.Series(dtype=str)).astype(str))
        != sorted(trades_rebuilt.get("trade_id", pd.Series(dtype=str)).astype(str))
    )
    metric_mismatches = sum(
        frame_hash(metric_artifacts[name]) != frame_hash(rebuilt_metric_artifacts[name])
        for name in sorted(metric_artifacts)
    )
    final_decision_mismatches = int(decision != rebuilt_decision or blocker != rebuilt_blocker)
    control_candidate_mismatches = int(
        frame_hash(control_candidates) != frame_hash(control_candidates_rebuilt)
    )
    quote_integrity_mismatches = int(
        frame_hash(selection_quote_integrity) != frame_hash(selection_quote_integrity_rebuilt)
    )
    maximum_quote_difference = maximum_numeric_difference(quote_columns)
    maximum_pnl_difference = maximum_numeric_difference(pnl_columns)
    deterministic = (
        first_signal_hash == second_signal_hash
        and exact_date_record_mismatches == 0
        and selected_mismatches == 0
        and trade_identity_mismatches == 0
        and control_candidate_mismatches == 0
        and quote_integrity_mismatches == 0
        and metric_mismatches == 0
        and maximum_quote_difference == 0.0
        and maximum_pnl_difference <= 1e-10
        and final_decision_mismatches == 0
    )
    determinism = {
        **SAFETY_FLAGS_V01,
        "cached_records_reloaded": True,
        "option_redownloaded": False,
        "late_day_signal_hash_first": first_signal_hash,
        "late_day_signal_hash_second": second_signal_hash,
        "late_day_signal_mismatches": int(first_signal_hash != second_signal_hash),
        "exact_date_record_mismatches": exact_date_record_mismatches,
        "selected_contract_mismatches": selected_mismatches,
        "trade_identity_mismatches": trade_identity_mismatches,
        "control_candidate_mismatches": control_candidate_mismatches,
        "quote_integrity_mismatches": quote_integrity_mismatches,
        "strategy_metric_mismatches": metric_mismatches,
        "maximum_quote_difference": maximum_quote_difference,
        "maximum_pnl_difference": maximum_pnl_difference,
        "final_decision_mismatches": final_decision_mismatches,
        "passed": deterministic,
    }
    write_json(output / "determinism_check.json", determinism)
    lightweight = {
        **SAFETY_FLAGS_V01,
        "audit_stage": "runner_self_checks",
        "contract_safety_exact": True,
        "exact_requested_date_filtering_verified": True,
        "extra_provider_dates_discarded_before_materialisation": True,
        "strategies_evaluated_independently": True,
        "one_stock_state_row_per_stock_regular_session": True,
        "ordinal_72_timing_verified": True,
        "frozen_route_labels_reconstructed": True,
        "direction_mapping_rejected": True,
        "previous_session_preselection_frozen": True,
        "same_close_contract_selection_prohibited": True,
        "daily_option_high_low_used": False,
        "protected_rows_materialised": 0,
        "determinism_passed": bool(determinism["passed"]),
        "independent_audit_pending": True,
        "passed": bool(determinism["passed"]),
    }
    write_json(output / "lightweight_audit.json", lightweight)
    report = report_text(
        decision=decision,
        signals=signals,
        required=required,
        gaps=gaps,
        cache=cast(Mapping[str, Any], source_manifest["options_cache"]),
        structural_manifest=structural_manifest,
        trades=trades,
        selected=selected,
        metric_artifacts=metric_artifacts,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    if not determinism["passed"]:
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["primary_blocker"] = "blocked_reproducibility_or_audit_failure"
        write_json(output / "decision.json", decision)
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "determinism reconstruction differs"
        )
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=Path(
            os.environ.get(
                "STOCKER_EODHD_STOCK_ROOT",
                str(
                    Path.home()
                    / "StockerLocal"
                    / "data"
                    / "processed"
                    / "source=eodhd"
                    / "instrument_type=stock"
                ),
            )
        ),
    )
    parser.add_argument(
        "--options-cache",
        type=Path,
        default=REPO_ROOT / "data" / "vendor" / "eodhd" / "options",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        decision = run(
            provider_root=arguments.provider_root.resolve(),
            options_cache=arguments.options_cache.resolve(),
            output=arguments.output.resolve(),
        )
    except ScreenBlocker as error:
        print(error.decision)
        print(error.detail)
        return 2
    print(decision["overall_decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
