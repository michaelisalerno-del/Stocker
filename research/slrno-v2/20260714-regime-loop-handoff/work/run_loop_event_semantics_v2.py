#!/usr/bin/env python3
"""Bounded research-only reconstruction for loop-event semantics V2.

The runner reads only structural five-minute bars and frozen state lineage.  It
does not read an economic outcome, fit a payoff model, or expose any execution
surface.  Provider timestamps are bar starts; every decision is stamped at
``bar_start + five minutes``.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import frozen_loop_movement_shadow_core as frozen_core  # noqa: E402

from stocker_research.causal_state_export_v2 import (  # noqa: E402
    HysteresisConfig,
    audit_legacy_run_context,
    build_completed_bar_decisions,
    build_hard_state_runs_v2,
    causal_semimarkov_filter_v2,
    churn_diagnostics,
    expand_duration_hazard_v2,
)
from stocker_research.loop_dictionary_v2 import (  # noqa: E402
    ALLOWED_COMPOSITE_TRANSITION_LENGTHS,
    ALLOWED_PRIMITIVE_TRANSITION_LENGTHS,
    MAX_EVENT_TRANSITIONS,
    DictionaryCandidateMetrics,
    LegacyCycleRecord,
    LoopDefinition,
    LoopDictionary,
    MotifType,
    UnsupportedLoopError,
    candidate_selection_score,
    decompose_closed_path,
    loop_complexity_penalty,
    select_dictionary_candidates,
)
from stocker_research.loop_duration_v2 import (  # noqa: E402
    DiscreteSurvivalDurationModel,
    DurationObservation,
)
from stocker_research.loop_events_v2 import safety_flags  # noqa: E402
from stocker_research.loop_ledger_v2 import (  # noqa: E402
    build_loop_event_ledgers,
    compare_legacy_targets_to_v2_outcomes,
    session_source_is_complete,
)
from stocker_research.loop_nulls_v2 import (  # noqa: E402
    ClockConditionedSemiMarkovNull,
    SemiMarkovNull,
    SessionRunSequence,
    benjamini_hochberg,
    circular_session_control,
    count_candidate_paths,
    empirical_p_values,
    first_order_expected_counts,
    session_phase,
    simulate_null_counts,
)

CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-loop-event-semantics-v2.json"
ARTIFACT_PARENT = WORK_DIR / "artifacts" / "20260718-loop-event-semantics-v2"
CENSUS_SCOPE = WORK_DIR / "contracts" / "20260718-loop-implementation-census-scope.json"
FROZEN_BUNDLE = WORK_DIR / "shadow_validation" / "frozen_loop_movement_shadow_v1" / "frozen_bundle"
STATE_DIR = FROZEN_BUNDLE / "artifacts" / "state"
MODEL_PATH = STATE_DIR / "frozen_semimarkov_parameters.npz"
PREPROCESSING_PATH = STATE_DIR / "frozen_emission_preprocessing.csv"
LEGACY_CYCLE_PATH = STATE_DIR / "fixed_cycle_shuffled_nulls.csv"
HISTORICAL_RUNNER_PATH = (
    FROZEN_BUNDLE / "provenance" / "lineage" / "run_causal_semimarkov_regime_loops.py"
)
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
STATE_COUNT = 8
BAR_DURATION = pd.Timedelta(minutes=5)
DEVELOPMENT_START = pd.Timestamp("2024-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
STATE_MODEL_VERSION = "causal_semimarkov_posterior_export_v2_tail78"
DICTIONARY_VERSION = "semantic_loop_dictionary_v2"
ARTIFACT_IDENTITY: dict[str, Any] = {}


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    if pd.isna(value):
        return None
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _with_safety(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for key, value in safety_flags().items():
        result[key] = value
    return result


def _with_artifact_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for key, value in ARTIFACT_IDENTITY.items():
        if key not in result:
            result[key] = value
    for key in ("symbol", "session", "timestamp", "semantic_loop_id", "legacy_cycle_id"):
        if key not in result:
            result[key] = "not_applicable"
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    merged = {**ARTIFACT_IDENTITY, **payload, **safety_flags()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            merged,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ).encode("utf-8")
        + b"\n"
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    output = _with_safety(_with_artifact_identity(frame))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        output.to_parquet(path, index=False, compression="zstd", compression_level=9)
    elif path.suffix == ".csv":
        output.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    else:
        raise ValueError(f"unsupported frame artifact: {path}")


def _provider_file(symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return PROVIDER_ROOT / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def _bounded_provider_hash(path: Path) -> tuple[str, int]:
    """Hash only the authorised development rows used by this reconstruction."""

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("timestamp", ">=", DEVELOPMENT_START.to_pydatetime()),
            ("timestamp", "<=", DEVELOPMENT_END.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].gt(DEVELOPMENT_END).any():
        raise AssertionError("bounded provider hash admitted a post-development row")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return _sha256_bytes(sink.getvalue().to_pybytes()), len(frame)


def _source_hashes() -> tuple[dict[str, str], str]:
    files = {symbol: _provider_file(symbol) for symbol in (*SYMBOLS, "VTI")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required structural source is missing: {missing}")
    hashes = {symbol: _bounded_provider_hash(path)[0] for symbol, path in files.items()}
    snapshot_hash = _sha256_bytes(_canonical_json_bytes(hashes))
    return hashes, snapshot_hash


def _load_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if any(
        (
            contract["research_only"] is not True,
            contract["execution_enabled"] is not False,
            contract["order_placement"] != "disabled",
            contract["broker_connected"] is not False,
            contract["strategy_promotion"] is not False,
        )
    ):
        raise AssertionError("contract safety boundary is not closed")
    return contract, _sha256_file(CONTRACT_PATH)


def _prepare_panel(
    source_hash_by_symbol: dict[str, str],
    *,
    data_snapshot_hash: str,
    contract_hash: str,
) -> pd.DataFrame:
    start = DEVELOPMENT_START
    end = DEVELOPMENT_END
    parts = [
        frozen_core.prepare_symbol_bars(symbol, PROVIDER_ROOT, start, end) for symbol in SYMBOLS
    ]
    panel = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["symbol_norm", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    vti = frozen_core.prepare_symbol_bars("VTI", PROVIDER_ROOT, start, end)
    panel = frozen_core.add_market_features(panel, vti)
    b0 = frozen_core.build_causal_b0(list(SYMBOLS), PROVIDER_ROOT, start, end)
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
        {
            "weak_broad_tape": -1.0,
            "neutral_broad_tape": 0.0,
            "strong_broad_tape": 1.0,
        }
    )
    panel["b0_high_stress"] = panel["b0_stress_box"].map({"normal_stress": 0.0, "high_stress": 1.0})
    panel = frozen_core.add_emission_features(panel)
    panel = panel.rename(
        columns={
            "symbol_norm": "symbol",
            "session_date": "session",
            "bar_index_in_session": "bar_ordinal",
            "timestamp": "bar_start_timestamp",
        }
    )
    panel["bar_complete_timestamp"] = panel["bar_start_timestamp"] + BAR_DURATION
    panel["bar_is_complete"] = True
    panel["clock_phase"] = panel["bar_ordinal"].map(
        lambda value: session_phase(min(int(value), 77))
    )
    phase = 2.0 * np.pi * panel["bar_ordinal"].to_numpy(dtype=float) * 5.0 / 390.0
    panel["clock_sin"] = np.sin(phase)
    panel["clock_cos"] = np.cos(phase)
    session_last = panel.groupby("session", sort=True)["bar_complete_timestamp"].max().sort_index()
    previous_close = session_last.shift(1)
    panel["b0_source_timestamp"] = panel["session"].map(previous_close)
    panel["b0_available_timestamp"] = panel["b0_source_timestamp"]
    session_last_ordinal = panel.groupby("session", sort=True)["bar_ordinal"].max().sort_index()
    panel["b0_source_bar_ordinal"] = panel["session"].map(session_last_ordinal.shift(1))
    panel["stock_source_artifact_hash"] = panel["symbol"].map(source_hash_by_symbol)
    panel["market_source_artifact_hash"] = source_hash_by_symbol["VTI"]
    panel["b0_source_artifact_hash"] = data_snapshot_hash
    panel["clock_source_artifact_hash"] = contract_hash
    panel["source_artifact_hash"] = data_snapshot_hash

    stock_fields = (
        "current_bar_log_return",
        "return_sum_6",
        "return_sum_12",
        "mean_abs_return_12",
        "bar_range_pct",
        "session_return",
        "regime_log_activity_3",
        "regime_log_activity_12",
        "regime_log_bar_range",
        "log_relative_historical_volume",
    )
    for field in stock_fields:
        valid = panel[field].notna()
        panel[f"{field}__causal_valid"] = valid
        panel[f"{field}__missing_reason"] = np.where(
            valid, None, "source_value_missing_or_insufficient_causal_history"
        )
    for field in ("b0_state_numeric", "b0_high_stress"):
        valid = (
            panel[field].notna()
            & panel["b0_source_timestamp"].notna()
            & panel["b0_available_timestamp"].le(panel["bar_complete_timestamp"])
        )
        panel[f"{field}__causal_valid"] = valid
        panel[f"{field}__missing_reason"] = np.where(
            valid, None, "b0_source_missing_or_initial_warmup"
        )
    for field in ("clock_phase", "clock_sin", "clock_cos"):
        panel[f"{field}__causal_valid"] = True
        panel[f"{field}__missing_reason"] = None

    source_complete = np.zeros(len(panel), dtype=bool)
    for (_, _), group in panel.groupby(["symbol", "session"], sort=False):
        positions = group.index.to_numpy(dtype=int)
        source_complete[positions] = session_source_is_complete(group)
    panel["source_sequence_complete"] = source_complete
    panel["source_sequence_missing_reason"] = np.where(
        source_complete, None, "incomplete_or_ambiguous_in_session_source_sequence"
    )
    if panel[["symbol", "session", "bar_ordinal"]].duplicated().any():
        raise AssertionError("bar identity is not unique")
    if not panel["bar_complete_timestamp"].gt(panel["bar_start_timestamp"]).all():
        raise AssertionError("completed-bar timestamps are not explicit")
    return panel.reset_index(drop=True)


def _contiguous_causal_groups(panel: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Split posterior recursion at every session boundary or missing bar."""

    groups: list[np.ndarray] = []
    for (_, _), group in panel.groupby(["symbol", "session"], sort=False):
        positions = group.index.to_numpy(dtype=int)
        ordinals = group["bar_ordinal"].to_numpy(dtype=int)
        starts = pd.to_datetime(group["bar_start_timestamp"], utc=True)
        breaks = (
            np.flatnonzero(
                (np.diff(ordinals) != 1)
                | (starts.diff().iloc[1:].to_numpy() != np.timedelta64(5, "m"))
            )
            + 1
        )
        for segment in np.split(positions, breaks):
            if len(segment):
                groups.append(segment)
    return tuple(groups)


def _state_inference(
    panel: pd.DataFrame,
) -> tuple[np.ndarray, Any, dict[str, np.ndarray], np.ndarray]:
    preprocessing = pd.read_csv(PREPROCESSING_PATH)
    model = {key: value for key, value in np.load(MODEL_PATH).items()}
    scaled = frozen_core.scale_emissions(panel, preprocessing)
    log_emissions = frozen_core.log_emission(scaled, model)
    legacy_groups = frozen_core.group_positions(
        panel.rename(columns={"symbol": "symbol_norm", "session": "session_date"})
    )
    legacy_labels, _, _ = frozen_core.causal_filter(log_emissions, legacy_groups, model)
    v2_model = expand_duration_hazard_v2(model, maximum_age=78, tail_window=6)
    export = causal_semimarkov_filter_v2(
        log_emissions,
        session_groups=_contiguous_causal_groups(panel),
        model=v2_model,
        bar_start_timestamps=tuple(
            pd.Timestamp(value).to_pydatetime() for value in panel["bar_start_timestamp"]
        ),
        bar_duration=BAR_DURATION.to_pytimedelta(),
    )
    return legacy_labels, export, v2_model, log_emissions


def _context_audit(
    panel: pd.DataFrame,
    legacy_labels: np.ndarray,
    *,
    run_id: str,
    git_sha: str,
    contract_hash: str,
    data_snapshot_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context_fields = (
        "b0_state_numeric",
        "b0_high_stress",
        "clock_sin",
        "clock_cos",
    )
    runs = build_hard_state_runs_v2(panel, legacy_labels, context_fields=context_fields)
    session_complete = (
        panel.groupby(["symbol", "session"], sort=False)["source_sequence_complete"]
        .first()
        .rename("source_sequence_complete")
        .reset_index()
    )
    runs = runs.merge(
        session_complete,
        on=["symbol", "session"],
        how="left",
        validate="many_to_one",
    )
    is_terminal = runs.groupby(["symbol", "session"], sort=False).cumcount(ascending=False).eq(0)
    runs["right_censored"] = is_terminal & runs["source_sequence_complete"]
    runs["censoring_status"] = np.select(
        [~runs["source_sequence_complete"], runs["right_censored"]],
        ["gap_invalidated_excluded", "regular_session_terminal_right_censored"],
        default="observed_exit",
    )
    audit = audit_legacy_run_context(panel, legacy_labels, context_fields=context_fields)
    source_columns = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "b0_source_timestamp",
        "b0_source_bar_ordinal",
        "b0_available_timestamp",
        "stock_source_artifact_hash",
        "b0_source_artifact_hash",
        "clock_source_artifact_hash",
        "source_sequence_complete",
        *[f"{field}__causal_valid" for field in context_fields],
        *[f"{field}__missing_reason" for field in context_fields],
    ]
    source_at_entry = panel[source_columns].rename(
        columns={
            "bar_ordinal": "start_bar_ordinal",
            "bar_start_timestamp": "entry_bar_start_timestamp",
            "bar_complete_timestamp": "entry_bar_complete_timestamp",
        }
    )
    audit = audit.merge(
        source_at_entry,
        on=["symbol", "session", "start_bar_ordinal"],
        how="left",
        validate="many_to_one",
    )
    audit["run_id_v2"] = run_id
    audit["git_sha"] = git_sha
    audit["contract_hash"] = contract_hash
    audit["data_snapshot_hash"] = data_snapshot_hash
    audit["state_model_version"] = STATE_MODEL_VERSION
    audit["dictionary_version"] = DICTIONARY_VERSION
    audit["timestamp"] = audit["start_timestamp"]
    audit["source_field"] = audit["field"]
    is_b0 = audit["field"].isin(["b0_state_numeric", "b0_high_stress"])
    is_clock = audit["field"].isin(["clock_sin", "clock_cos"])
    audit["source_artifact_hash"] = audit["stock_source_artifact_hash"]
    audit.loc[is_b0, "source_artifact_hash"] = audit.loc[is_b0, "b0_source_artifact_hash"]
    audit.loc[is_clock, "source_artifact_hash"] = audit.loc[is_clock, "clock_source_artifact_hash"]
    audit["source_timestamp"] = audit["entry_bar_start_timestamp"]
    audit.loc[is_b0, "source_timestamp"] = audit.loc[is_b0, "b0_source_timestamp"]
    audit["source_bar_ordinal"] = audit["start_bar_ordinal"]
    audit.loc[is_b0, "source_bar_ordinal"] = audit.loc[is_b0, "b0_source_bar_ordinal"]
    audit["available_timestamp"] = audit["entry_bar_complete_timestamp"]
    audit.loc[is_b0, "available_timestamp"] = audit.loc[is_b0, "b0_available_timestamp"]
    audit.loc[is_clock, "available_timestamp"] = audit.loc[is_clock, "entry_bar_start_timestamp"]
    audit["decision_timestamp"] = audit["start_timestamp"] + BAR_DURATION
    audit["causal_valid"] = False
    audit["missing_reason"] = None
    for field in context_fields:
        mask = audit["field"].eq(field)
        audit.loc[mask, "causal_valid"] = audit.loc[mask, f"{field}__causal_valid"]
        audit.loc[mask, "missing_reason"] = audit.loc[mask, f"{field}__missing_reason"]
    audit["causal_valid"] &= audit["available_timestamp"].le(audit["decision_timestamp"])
    audit["legacy_stored_value_evidence"] = "reconstructed_from_frozen_builder_last_row_assignment"
    summary = (
        audit.groupby("field", sort=True)
        .agg(
            audited_runs=("run_id", "size"),
            runs_start_end_differ=("start_end_differ", "sum"),
        )
        .reset_index()
    )
    differ = audit.loc[audit["start_end_differ"]]
    affected = (
        differ.groupby("field", sort=True)
        .agg(
            anchors_affected=("run_id", "nunique"),
            symbols_affected=("symbol", "nunique"),
            periods_affected=(
                "start_timestamp",
                lambda values: pd.to_datetime(values, utc=True).dt.strftime("%Y-%m").nunique(),
            ),
        )
        .reset_index()
    )
    summary = summary.merge(affected, on="field", how="left", validate="one_to_one")
    summary["anchors_affected"] = summary["anchors_affected"].fillna(0).astype(int)
    summary["symbols_affected"] = summary["symbols_affected"].fillna(0).astype(int)
    summary["periods_affected"] = summary["periods_affected"].fillna(0).astype(int)
    summary["experiments_affected"] = summary["field"].map(
        {
            "b0_state_numeric": "none empirically in 2024; code-position limitation remains",
            "b0_high_stress": "none empirically in 2024; code-position limitation remains",
            "clock_sin": "run_regime_utility_ablation_v1 and raw-run context consumers",
            "clock_cos": "run_regime_utility_ablation_v1 and raw-run context consumers",
        }
    )
    for identity, value in (
        ("run_id_v2", run_id),
        ("git_sha", git_sha),
        ("contract_hash", contract_hash),
        ("data_snapshot_hash", data_snapshot_hash),
        ("state_model_version", STATE_MODEL_VERSION),
        ("dictionary_version", DICTIONARY_VERSION),
    ):
        summary[identity] = value
    runs["run_id_v2"] = run_id
    runs["git_sha"] = git_sha
    runs["contract_hash"] = contract_hash
    runs["data_snapshot_hash"] = data_snapshot_hash
    runs["state_model_version"] = STATE_MODEL_VERSION
    runs["dictionary_version"] = DICTIONARY_VERSION
    runs["timestamp"] = runs["start_timestamp"]
    return runs, audit, summary


def _duration_artifacts(
    runs: pd.DataFrame,
) -> tuple[DiscreteSurvivalDurationModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible_runs = runs.loc[runs["source_sequence_complete"].astype(bool)].copy()
    if eligible_runs.empty:
        raise RuntimeError("no source-complete runs remain for duration fitting")
    observations = tuple(
        DurationObservation(
            state=int(row.state),
            duration=int(row.duration),
            right_censored=bool(row.right_censored),
        )
        for row in eligible_runs.itertuples(index=False)
    )
    model = DiscreteSurvivalDurationModel.fit(
        observations,
        state_count=STATE_COUNT,
        maximum_duration=78,
        smoothing_strength=8.0,
    )
    summary_rows = []
    for state in range(STATE_COUNT):
        distribution = model.duration_distribution(state)
        survival = 1.0
        for age in range(1, model.maximum_duration + 1):
            summary_rows.append(
                {
                    "state": state,
                    "age": age,
                    "at_risk_count": int(model.at_risk_counts[state, age - 1]),
                    "exit_count": int(model.exit_counts[state, age - 1]),
                    "censored_count": int(model.censored_counts[state, age - 1]),
                    "hazard": float(model.hazard[state, age - 1]),
                    "survival_at_start": survival,
                    "exact_duration_probability": float(distribution.exact_pmf[age - 1]),
                }
            )
            survival *= 1.0 - float(model.hazard[state, age - 1])
    summary = pd.DataFrame(summary_rows)
    tail_rows = []
    for state in range(STATE_COUNT):
        distribution = model.duration_distribution(state)
        tail_rows.append(
            {
                "state": state,
                "probability_duration_23": float(distribution.exact_pmf[22]),
                "probability_duration_24": float(distribution.exact_pmf[23]),
                "probability_duration_25": float(distribution.exact_pmf[24]),
                "probability_duration_greater_than_24": float(
                    distribution.exact_pmf[24:].sum() + distribution.survival_tail
                ),
                "probability_beyond_regular_session_support": float(distribution.survival_tail),
                "hazard_at_24": float(model.hazard[state, 23]),
                "forced_exit_at_24": False,
                "duration_24_is_exact": True,
                "duration_25_is_separate": True,
            }
        )
    tail = pd.DataFrame(tail_rows)
    censor = (
        eligible_runs.groupby("state", sort=True)
        .agg(
            observed_runs=("duration", "size"),
            exact_exits=("right_censored", lambda values: int((~values).sum())),
            right_censored_terminal_runs=("right_censored", "sum"),
            maximum_observed_duration=("duration", "max"),
            exact_duration_24_count=("duration", lambda values: int((values == 24).sum())),
            duration_25_count=("duration", lambda values: int((values == 25).sum())),
            duration_greater_than_24_count=(
                "duration",
                lambda values: int((values > 24).sum()),
            ),
        )
        .reset_index()
    )
    censor["censoring_population"] = "source_complete_duration_fit"
    invalid = (
        runs.loc[~runs["source_sequence_complete"].astype(bool)]
        .groupby("state", sort=True)
        .agg(
            observed_runs=("duration", "size"),
            maximum_observed_duration=("duration", "max"),
        )
        .reset_index()
    )
    if not invalid.empty:
        invalid["exact_exits"] = 0
        invalid["right_censored_terminal_runs"] = 0
        invalid["exact_duration_24_count"] = 0
        invalid["duration_25_count"] = 0
        invalid["duration_greater_than_24_count"] = 0
        invalid["censoring_population"] = "gap_invalidated_excluded_not_censored"
        invalid = invalid[censor.columns]
        censor = pd.concat([censor, invalid], ignore_index=True)
    return model, summary, tail, censor


def _session_sequences(runs: pd.DataFrame) -> tuple[SessionRunSequence, ...]:
    rows = []
    eligible = runs.loc[runs["source_sequence_complete"].astype(bool)]
    for (symbol, session), group in eligible.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("start_bar_ordinal", kind="mergesort")
        rows.append(
            SessionRunSequence(
                symbol=str(symbol),
                session=str(session),
                states=tuple(ordered["state"].astype(int)),
                durations=tuple(ordered["duration"].astype(int)),
                terminal_right_censored=bool(ordered.iloc[-1]["right_censored"]),
            )
        )
    return tuple(rows)


def _discover_candidates(
    sessions: Sequence[SessionRunSequence],
) -> tuple[dict[str, LoopDefinition], dict[str, dict[str, Any]]]:
    definitions: dict[str, LoopDefinition] = {}
    details: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "observed": 0,
            "symbols": set(),
            "months": set(),
            "phases": set(),
            "quarters": Counter(),
        }
    )
    for session in sessions:
        month = session.session[:7]
        timestamp = pd.Timestamp(session.session)
        quarter = f"{timestamp.year}_q{timestamp.quarter}"
        run_starts = np.r_[0, np.cumsum(session.durations[:-1])]
        allowed_discovery_lengths = (
            ALLOWED_PRIMITIVE_TRANSITION_LENGTHS | ALLOWED_COMPOSITE_TRANSITION_LENGTHS
        )
        if max(allowed_discovery_lengths) != MAX_EVENT_TRANSITIONS:
            raise AssertionError("allowed discovery lengths disagree with maximum event length")
        for transitions in sorted(allowed_discovery_lengths):
            width = transitions + 1
            for start in range(len(session.states) - width + 1):
                path = tuple(session.states[start : start + width])
                if path[0] != path[-1]:
                    continue
                try:
                    definition = decompose_closed_path(path)
                except UnsupportedLoopError:
                    continue
                semantic_id = definition.semantic_loop_id
                definitions.setdefault(semantic_id, definition)
                record = details[semantic_id]
                record["observed"] += 1
                record["symbols"].add(session.symbol)
                record["months"].add(month)
                record["phases"].add(session_phase(min(int(run_starts[start]), 77)))
                record["quarters"][quarter] += 1
    return definitions, details


def _balanced_session_sample(
    sessions: Sequence[SessionRunSequence], per_symbol: int
) -> tuple[SessionRunSequence, ...]:
    grouped: dict[str, list[SessionRunSequence]] = defaultdict(list)
    for session in sessions:
        grouped[session.symbol].append(session)
    selected = []
    for symbol in sorted(grouped):
        values = sorted(grouped[symbol], key=lambda item: item.session)
        if len(values) <= per_symbol:
            selected.extend(values)
            continue
        indices = np.linspace(0, len(values) - 1, per_symbol, dtype=int)
        selected.extend(values[int(index)] for index in indices)
    return tuple(selected)


def _oriented_candidate_matrix(
    definitions: Sequence[LoopDefinition],
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    paths: list[tuple[int, ...]] = []
    owner: list[int] = []
    for candidate_index, definition in enumerate(definitions):
        for path in definition.oriented_paths:
            paths.append(path)
            owner.append(candidate_index)
    return tuple(paths), np.asarray(owner, dtype=int)


def _aggregate_orientation_counts(
    values: np.ndarray, owner: np.ndarray, candidate_count: int
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return np.bincount(owner, weights=array, minlength=candidate_count)
    output = np.zeros((array.shape[0], candidate_count), dtype=float)
    for candidate_index in range(candidate_count):
        output[:, candidate_index] = array[:, owner == candidate_index].sum(axis=1)
    return output


def _eligible_anchor_count(
    sessions: Sequence[SessionRunSequence], definition: LoopDefinition
) -> int:
    width = definition.full_transition_length + 1
    starting_states = {path[0] for path in definition.oriented_paths}
    return sum(
        sum(
            session.states[start] in starting_states
            for start in range(max(0, len(session.states) - width + 1))
        )
        for session in sessions
    )


def _second_order_expected_counts(
    sessions: Sequence[SessionRunSequence], paths: Sequence[tuple[int, ...]]
) -> np.ndarray:
    first = np.full((STATE_COUNT, STATE_COUNT), 0.5, dtype=float)
    np.fill_diagonal(first, 0.0)
    second = np.full((STATE_COUNT, STATE_COUNT, STATE_COUNT), 0.5, dtype=float)
    for previous in range(STATE_COUNT):
        for current in range(STATE_COUNT):
            second[previous, current, current] = 0.0
    for session in sessions:
        for origin, destination in zip(session.states[:-1], session.states[1:], strict=True):
            first[origin, destination] += 1.0
        for previous, current, destination in zip(
            session.states[:-2],
            session.states[1:-1],
            session.states[2:],
            strict=True,
        ):
            second[previous, current, destination] += 1.0
    first /= first.sum(axis=1, keepdims=True)
    second /= second.sum(axis=2, keepdims=True)
    output = np.zeros(len(paths), dtype=float)
    for index, path in enumerate(paths):
        anchors = sum(
            sum(
                session.states[start] == path[0]
                for start in range(max(0, len(session.states) - len(path) + 1))
            )
            for session in sessions
        )
        probability = first[path[0], path[1]]
        for previous, current, destination in zip(path[:-2], path[1:-1], path[2:], strict=True):
            probability *= second[previous, current, destination]
        output[index] = anchors * probability
    return output


def _dictionary_research(
    sessions: Sequence[SessionRunSequence],
    contract: dict[str, Any],
    *,
    output_dir: Path,
    run_id: str,
    git_sha: str,
    contract_hash: str,
    data_snapshot_hash: str,
) -> tuple[
    LoopDictionary,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    definitions_by_id, details = _discover_candidates(sessions)
    definitions = [definitions_by_id[key] for key in sorted(definitions_by_id)]
    minimum_occurrences = int(contract["dictionary"]["minimum_candidate_occurrences"])
    minimum_stock_breadth = int(contract["dictionary"]["minimum_stock_breadth"])
    minimum_month_breadth = int(contract["dictionary"]["minimum_month_breadth"])
    exclusion_reasons: dict[str, list[str]] = {}
    for definition in definitions:
        record = details[definition.semantic_loop_id]
        reasons = []
        if record["observed"] < minimum_occurrences:
            reasons.append("below_minimum_occurrences")
        if len(record["symbols"]) < minimum_stock_breadth:
            reasons.append("below_minimum_stock_breadth")
        if len(record["months"]) < minimum_month_breadth:
            reasons.append("below_minimum_month_breadth")
        exclusion_reasons[definition.semantic_loop_id] = reasons
    supported = [
        definition
        for definition in definitions
        if not exclusion_reasons[definition.semantic_loop_id]
    ]
    if not supported:
        raise RuntimeError("no structurally supported V2 dictionary candidates")
    sample = _balanced_session_sample(
        sessions, int(contract["nulls"]["primary_empirical_sessions_per_symbol"])
    )
    oriented_paths, owner = _oriented_candidate_matrix(supported)
    base_null = SemiMarkovNull.fit(sessions, state_count=STATE_COUNT, maximum_duration=78)
    primary_draws = simulate_null_counts(
        base_null,
        session_lengths=[sum(item.durations) for item in sample],
        candidates=oriented_paths,
        draws=int(contract["nulls"]["primary_draws"]),
        seed=int(contract["nulls"]["seed"]),
    )
    primary_candidate_draws = _aggregate_orientation_counts(primary_draws, owner, len(supported))
    sample_observed = _aggregate_orientation_counts(
        count_candidate_paths(sample, oriented_paths), owner, len(supported)
    )
    p_values = empirical_p_values(sample_observed, primary_candidate_draws)
    q_values = benjamini_hochberg(p_values)

    clock_null = ClockConditionedSemiMarkovNull.fit(
        sessions, state_count=STATE_COUNT, maximum_duration=78
    )
    clock_draws = simulate_null_counts(
        clock_null,
        session_lengths=[sum(item.durations) for item in sample],
        candidates=oriented_paths,
        draws=int(contract["nulls"]["clock_conditioned_draws"]),
        seed=int(contract["nulls"]["seed"]) + 1,
    )
    clock_candidate_draws = _aggregate_orientation_counts(clock_draws, owner, len(supported))

    analytical_orientation = first_order_expected_counts(sessions, base_null, oriented_paths)
    analytical = _aggregate_orientation_counts(analytical_orientation, owner, len(supported))
    second_order = _aggregate_orientation_counts(
        _second_order_expected_counts(sessions, oriented_paths),
        owner,
        len(supported),
    )
    sessions_by_quarter: dict[str, list[SessionRunSequence]] = defaultdict(list)
    for session in sessions:
        timestamp = pd.Timestamp(session.session)
        sessions_by_quarter[f"{timestamp.year}_q{timestamp.quarter}"].append(session)
    quarter_excess: list[np.ndarray] = []
    for quarter in sorted(sessions_by_quarter):
        quarter_sessions = tuple(sessions_by_quarter[quarter])
        quarter_observed = _aggregate_orientation_counts(
            count_candidate_paths(quarter_sessions, oriented_paths),
            owner,
            len(supported),
        )
        quarter_expected = _aggregate_orientation_counts(
            first_order_expected_counts(quarter_sessions, base_null, oriented_paths),
            owner,
            len(supported),
        )
        quarter_excess.append(quarter_observed > quarter_expected)
    period_consistency = np.mean(np.vstack(quarter_excess), axis=0)
    bar_scale = sum(sum(item.durations) for item in sessions) / sum(
        sum(item.durations) for item in sample
    )

    metric_records: list[DictionaryCandidateMetrics] = []
    candidate_rows = []
    excluded_candidate_rows = []
    for definition in definitions:
        reasons = exclusion_reasons[definition.semantic_loop_id]
        if not reasons:
            continue
        record = details[definition.semantic_loop_id]
        excluded_candidate_rows.append(
            {
                "semantic_loop_id": definition.semantic_loop_id,
                "primitive_loop_id": definition.primitive_loop_id,
                "motif_type": definition.motif_type.value,
                "repeat_depth": definition.repeat_depth,
                "canonical_path": "->".join(map(str, definition.canonical_orientation)),
                "full_transition_length": definition.full_transition_length,
                "eligible_anchor_count": _eligible_anchor_count(sessions, definition),
                "observed_completions": int(record["observed"]),
                "stock_breadth": len(record["symbols"]),
                "month_breadth": len(record["months"]),
                "clock_breadth": len(record["phases"]),
                "support_eligible_for_null": False,
                "candidate_status": "excluded_before_null",
                "exclusion_reason": ";".join(reasons),
                "complexity_penalty": loop_complexity_penalty(definition),
            }
        )
    null_rows = []
    for index, definition in enumerate(supported):
        record = details[definition.semantic_loop_id]
        observed = int(record["observed"])
        empirical_expected = float(primary_candidate_draws[:, index].mean())
        expected_full = empirical_expected * bar_scale
        eligible = max(1, _eligible_anchor_count(sessions, definition))
        observed_rate = min((observed + 0.5) / (eligible + 1.0), 1.0 - 1e-12)
        expected_rate = min((expected_full + 0.5) / (eligible + 1.0), 1.0 - 1e-12)
        information = observed_rate * math.log2(
            max(observed_rate / max(expected_rate, 1e-12), 1e-12)
        ) + (1.0 - observed_rate) * math.log2(
            max((1.0 - observed_rate) / max(1.0 - expected_rate, 1e-12), 1e-12)
        )
        increment_current = max(
            0.0,
            observed_rate
            * math.log2((observed + 0.5) / max(float(analytical[index]) + 0.5, 1e-12)),
        )
        increment_history = max(
            0.0,
            observed_rate
            * math.log2((observed + 0.5) / max(float(second_order[index]) + 0.5, 1e-12)),
        )
        rate_ratio = (observed + 0.5) / (expected_full + 0.5)
        metrics = DictionaryCandidateMetrics(
            definition=definition,
            eligible_anchor_count=eligible,
            observed_completions=observed,
            expected_completions=expected_full,
            empirical_p_value=float(p_values[index]),
            fdr_q_value=float(q_values[index]),
            conditional_information_gain=max(0.0, float(information)),
            increment_beyond_current_state=increment_current,
            increment_beyond_previous_state_history=increment_history,
            stock_breadth=len(record["symbols"]),
            month_breadth=len(record["months"]),
            clock_breadth=len(record["phases"]),
            period_consistency=float(period_consistency[index]),
        )
        metric_records.append(metrics)
        candidate_rows.append(
            {
                "semantic_loop_id": definition.semantic_loop_id,
                "primitive_loop_id": definition.primitive_loop_id,
                "motif_type": definition.motif_type.value,
                "repeat_depth": definition.repeat_depth,
                "canonical_path": "->".join(map(str, definition.canonical_orientation)),
                "full_transition_length": definition.full_transition_length,
                "eligible_anchor_count": eligible,
                "observed_completions": observed,
                "expected_completions_semi_markov": expected_full,
                "expected_completions_first_order": float(analytical[index]),
                "expected_completions_second_order": float(second_order[index]),
                "excess_completions": observed - expected_full,
                "rate_ratio": rate_ratio,
                "empirical_p_value": float(p_values[index]),
                "fdr_q_value": float(q_values[index]),
                "conditional_information_gain": metrics.conditional_information_gain,
                "increment_beyond_current_state": increment_current,
                "increment_beyond_previous_state_history": increment_history,
                "stock_breadth": metrics.stock_breadth,
                "month_breadth": metrics.month_breadth,
                "clock_breadth": metrics.clock_breadth,
                "period_consistency": metrics.period_consistency,
                "complexity_penalty": loop_complexity_penalty(definition),
                "selection_score": candidate_selection_score(metrics),
                "support_eligible_for_null": True,
                "candidate_status": "null_evaluated",
                "exclusion_reason": None,
            }
        )
        for null_name, observed_value, draws_matrix in (
            ("NULL_A_FITTED_SEMI_MARKOV", sample_observed[index], primary_candidate_draws),
            ("NULL_B_CLOCK_CONDITIONED_SEMI_MARKOV", sample_observed[index], clock_candidate_draws),
        ):
            draws_column = draws_matrix[:, index]
            null_rows.append(
                {
                    "null_name": null_name,
                    "population": "deterministic_balanced_2024_session_sample",
                    "draws": len(draws_column),
                    "semantic_loop_id": definition.semantic_loop_id,
                    "observed_count": int(observed_value),
                    "expected_count": float(draws_column.mean()),
                    "excess_count": float(observed_value - draws_column.mean()),
                    "rate_ratio": float((observed_value + 0.5) / (draws_column.mean() + 0.5)),
                    "empirical_p_value": float(
                        empirical_p_values(np.asarray([observed_value]), draws_column[:, None])[0]
                    ),
                    "fdr_q_value": float(q_values[index])
                    if null_name.startswith("NULL_A")
                    else None,
                    "stock_breadth": len(record["symbols"]),
                    "month_breadth": len(record["months"]),
                    "clock_breadth": len(record["phases"]),
                    "period_consistency": metrics.period_consistency,
                }
            )
        null_rows.append(
            {
                "null_name": "NULL_C_FIRST_ORDER_ANALYTICAL",
                "population": "full_2024_structural_sessions",
                "draws": 0,
                "semantic_loop_id": definition.semantic_loop_id,
                "observed_count": observed,
                "expected_count": float(analytical[index]),
                "excess_count": observed - float(analytical[index]),
                "rate_ratio": float((observed + 0.5) / (float(analytical[index]) + 0.5)),
                "empirical_p_value": None,
                "fdr_q_value": None,
                "stock_breadth": len(record["symbols"]),
                "month_breadth": len(record["months"]),
                "clock_breadth": len(record["phases"]),
                "period_consistency": metrics.period_consistency,
            }
        )

    eligible_metrics = [
        metrics
        for metrics in metric_records
        if metrics.fdr_q_value <= float(contract["dictionary"]["maximum_fdr_q_value"])
        and (metrics.observed_completions + 0.5) / (metrics.expected_completions + 0.5)
        >= float(contract["dictionary"]["minimum_rate_ratio"])
        and metrics.increment_beyond_previous_state_history > 0.0
    ]
    selected_metrics = select_dictionary_candidates(
        eligible_metrics,
        maximum_entries=int(contract["dictionary"]["maximum_selected_entries"]),
    )
    if not selected_metrics:
        raise RuntimeError("semi-Markov null retained no semantic dictionary entries")
    dictionary = LoopDictionary.from_definitions(
        (item.definition for item in selected_metrics), version=DICTIONARY_VERSION
    )
    selected_ids = {item.definition.semantic_loop_id for item in selected_metrics}
    selection = pd.DataFrame(candidate_rows)
    selection = selection.loc[selection["semantic_loop_id"].isin(selected_ids)].copy()
    score_order = {
        item.definition.semantic_loop_id: index + 1 for index, item in enumerate(selected_metrics)
    }
    selection["selection_rank"] = selection["semantic_loop_id"].map(score_order)
    selection["selected"] = True
    selection = selection.sort_values("selection_rank", kind="mergesort")

    circular_sessions = [item for item in sample if len(item.states) > 1]
    circular = tuple(
        circular_session_control(item, offset=max(1, len(item.states) // 2))
        for item in circular_sessions
    )
    circular_counts = _aggregate_orientation_counts(
        count_candidate_paths(circular, oriented_paths), owner, len(supported)
    )
    for index, definition in enumerate(supported):
        null_rows.append(
            {
                "null_name": "NULL_D_WHOLE_SESSION_CIRCULAR_CONTROL",
                "population": "eligible_balanced_2024_sessions",
                "draws": 1,
                "semantic_loop_id": definition.semantic_loop_id,
                "observed_count": int(sample_observed[index]),
                "expected_count": float(circular_counts[index]),
                "excess_count": float(sample_observed[index] - circular_counts[index]),
                "rate_ratio": float(
                    (sample_observed[index] + 0.5) / (circular_counts[index] + 0.5)
                ),
                "empirical_p_value": None,
                "fdr_q_value": None,
                "stock_breadth": len(details[definition.semantic_loop_id]["symbols"]),
                "month_breadth": len(details[definition.semantic_loop_id]["months"]),
                "clock_breadth": len(details[definition.semantic_loop_id]["phases"]),
                "period_consistency": next(
                    item.period_consistency
                    for item in metric_records
                    if item.definition.semantic_loop_id == definition.semantic_loop_id
                ),
            }
        )

    np.savez_compressed(
        output_dir / "structural_null_draws.npz",
        semantic_loop_ids=np.asarray([item.semantic_loop_id for item in supported], dtype="U64"),
        primary_draws=primary_candidate_draws.astype(np.int32),
        clock_conditioned_draws=clock_candidate_draws.astype(np.int32),
        sample_observed=sample_observed.astype(np.int32),
        research_only=np.asarray([True]),
        execution_enabled=np.asarray([False]),
        order_placement=np.asarray(["disabled"]),
        broker_connected=np.asarray([False]),
        strategy_promotion=np.asarray([False]),
        run_id_v2=np.asarray([run_id]),
        git_sha=np.asarray([git_sha]),
        contract_hash=np.asarray([contract_hash]),
        data_snapshot_hash=np.asarray([data_snapshot_hash]),
        source_artifact_hash=np.asarray([data_snapshot_hash]),
        dictionary_version=np.asarray([DICTIONARY_VERSION]),
        state_model_version=np.asarray([STATE_MODEL_VERSION]),
    )
    return (
        dictionary,
        pd.concat(
            [pd.DataFrame(candidate_rows), pd.DataFrame(excluded_candidate_rows)],
            ignore_index=True,
            sort=False,
        ),
        selection,
        pd.DataFrame(null_rows),
        primary_candidate_draws,
    )


def _legacy_dictionary() -> tuple[LoopDictionary, pd.DataFrame]:
    source = pd.read_csv(LEGACY_CYCLE_PATH)
    records = []
    output = source.copy()
    output.insert(
        0,
        "legacy_cycle_id",
        [f"cycle_{index:02d}" for index in range(1, len(source) + 1)],
    )
    output.insert(1, "discovery_rank", np.arange(1, len(source) + 1))
    for row in output.itertuples(index=False):
        path = tuple(int(value) for value in str(row.cycle).split("->"))
        records.append(
            LegacyCycleRecord(
                legacy_cycle_id=str(row.legacy_cycle_id),
                closed_path=path,
                discovery_rank=int(row.discovery_rank),
            )
        )
    return LoopDictionary.from_legacy(records, version="legacy_dictionary_v1"), output


def _dictionary_tables(
    dictionary: LoopDictionary,
    selection: pd.DataFrame,
    legacy_dictionary: LoopDictionary,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    for definition in dictionary.definitions.values():
        selected_rows.append(
            {
                "dictionary_version": dictionary.version,
                "dictionary_hash": dictionary.dictionary_hash,
                "semantic_loop_id": definition.semantic_loop_id,
                "primitive_loop_id": definition.primitive_loop_id,
                "motif_type": definition.motif_type.value,
                "repeat_depth": definition.repeat_depth,
                "primitive_core": list(definition.primitive_core),
                "primitive_transition_length": definition.primitive_transition_length,
                "full_core": list(definition.full_core),
                "full_transition_length": definition.full_transition_length,
                "canonical_orientation": list(definition.canonical_orientation),
                "all_valid_oriented_paths": [list(path) for path in definition.oriented_paths],
                "component_primitive_ids": list(definition.component_primitive_ids),
                "component_boundaries": [list(item) for item in definition.component_boundaries],
            }
        )
    duplicate_columns = [
        column
        for column in ("primitive_loop_id", "motif_type", "repeat_depth")
        if column in selection
    ]
    semantic = pd.DataFrame(selected_rows).merge(
        selection.drop(columns=duplicate_columns),
        on="semantic_loop_id",
        how="left",
        validate="one_to_one",
    )
    migrations = pd.DataFrame(legacy_dictionary.migration_rows())
    all_definitions: dict[str, LoopDefinition] = dict(legacy_dictionary.definitions)
    all_definitions.update(dictionary.definitions)
    decomposition = pd.DataFrame(
        [
            {
                "semantic_loop_id": item.semantic_loop_id,
                "primitive_loop_id": item.primitive_loop_id,
                "motif_type": item.motif_type.value,
                "primitive_core": list(item.primitive_core),
                "primitive_transition_length": item.primitive_transition_length,
                "repeat_depth": item.repeat_depth,
                "full_core": list(item.full_core),
                "full_transition_length": item.full_transition_length,
                "canonical_orientation": list(item.canonical_orientation),
                "all_valid_oriented_paths": [list(path) for path in item.oriented_paths],
            }
            for item in sorted(all_definitions.values(), key=lambda value: value.semantic_loop_id)
        ]
    )
    composites = pd.DataFrame(
        [
            {
                "semantic_loop_id": item.semantic_loop_id,
                "component_primitive_ids": list(item.component_primitive_ids),
                "component_boundaries": [list(value) for value in item.component_boundaries],
                "component_count": len(item.component_primitive_ids),
            }
            for item in sorted(all_definitions.values(), key=lambda value: value.semantic_loop_id)
            if item.motif_type is MotifType.COMPOSITE
        ]
    )
    return semantic, migrations, decomposition, composites


def _feature_manifest() -> dict[str, Any]:
    stock_fields = (
        "current_bar_log_return",
        "return_sum_6",
        "return_sum_12",
        "mean_abs_return_12",
        "bar_range_pct",
        "session_return",
        "regime_log_activity_3",
        "regime_log_activity_12",
        "regime_log_bar_range",
        "log_relative_historical_volume",
    )
    fields = {
        field: {
            "source_timestamp_column": "bar_start_timestamp",
            "source_bar_ordinal_column": "bar_ordinal",
            "available_timestamp_column": "bar_complete_timestamp",
            "decision_timestamp_column": "decision_timestamp",
            "causal_valid_column": f"{field}__causal_valid",
            "missing_reason_column": f"{field}__missing_reason",
            "source_field": field,
            "source_artifact_hash_column": "stock_source_artifact_hash",
            "causal_invariant": "bar_complete_timestamp <= decision_timestamp",
            "missing_policy": "preserve",
        }
        for field in stock_fields
    }
    for field in ("clock_phase", "clock_sin", "clock_cos"):
        fields[field] = {
            "source_timestamp_column": "bar_start_timestamp",
            "source_bar_ordinal_column": "bar_ordinal",
            "available_timestamp_column": "bar_start_timestamp",
            "decision_timestamp_column": "decision_timestamp",
            "causal_valid_column": f"{field}__causal_valid",
            "missing_reason_column": f"{field}__missing_reason",
            "source_field": field,
            "source_artifact_hash_column": "clock_source_artifact_hash",
            "causal_invariant": "bar_start_timestamp <= decision_timestamp",
            "missing_policy": "preserve",
        }
    for field in ("b0_state_numeric", "b0_high_stress"):
        fields[field] = {
            "source_timestamp_column": "b0_source_timestamp",
            "source_bar_ordinal_column": "b0_source_bar_ordinal",
            "available_timestamp_column": "b0_available_timestamp",
            "decision_timestamp_column": "decision_timestamp",
            "causal_valid_column": f"{field}__causal_valid",
            "missing_reason_column": f"{field}__missing_reason",
            "source_field": field,
            "source_artifact_hash_column": "b0_source_artifact_hash",
            "causal_invariant": "b0_available_timestamp <= decision_timestamp",
            "missing_policy": "preserve",
        }
    state_fields = (
        "posterior_state_probabilities",
        "posterior_entropy",
        "top_state",
        "second_state",
        "top_second_margin",
        "expected_state_age",
        "probability_state_persists_next_bar",
        "transition_probability_next_bar",
        "next_state_probabilities",
        "hard_state_posterior_map",
        "hard_state_hysteretic",
    )
    for field in state_fields:
        fields[field] = {
            "source_timestamp_column": "posterior_source_timestamp",
            "source_bar_ordinal_column": "bar_ordinal",
            "available_timestamp_column": "posterior_available_timestamp",
            "decision_timestamp_column": "decision_timestamp",
            "causal_valid_column": "posterior_causal_valid",
            "missing_reason_column": "posterior_missing_reason",
            "source_field": "forward_only_state_age_posterior",
            "source_artifact_hash_column": "state_source_artifact_hash",
            "causal_invariant": "posterior_available_timestamp <= decision_timestamp",
            "missing_policy": "fail_closed_on_incomplete_session",
        }
    structural_fields = (
        "previous_completed_state_1",
        "previous_completed_state_2",
        "previous_completed_state_3",
        "previous_completed_state_4",
        "previous_primitive_loop_1",
        "previous_primitive_loop_2",
        "bars_since_previous_loop",
        "same_loop_repeat_depth",
        "active_prefix_count",
        "active_primitive_prefixes",
        "active_repeat_prefixes",
        "active_composite_prefixes",
        "shortest_transitions_remaining",
        "highest_soft_prefix_probability",
        "soft_completion_probabilities",
    )
    for field in structural_fields:
        fields[field] = {
            "source_timestamp_column": "bar_start_timestamp",
            "source_bar_ordinal_column": "bar_ordinal",
            "available_timestamp_column": "bar_complete_timestamp",
            "decision_timestamp_column": "decision_timestamp",
            "causal_valid_column": f"{field}__causal_valid",
            "missing_reason_column": f"{field}__missing_reason",
            "source_field": "causal_state_event_history",
            "source_artifact_hash_column": "structural_source_artifact_hash",
            "causal_invariant": "bar_complete_timestamp <= decision_timestamp",
            "missing_policy": "fail_closed_on_incomplete_session",
        }
    return {
        "manifest_version": "feature_availability_manifest_v2",
        "provider_timestamp_semantics": "bar_start",
        "bar_duration_minutes": 5,
        "fields": fields,
        "forbidden_feature_tokens": [
            "future",
            "payoff",
            "MFE",
            "MAE",
            "route_completion",
            "order",
            "position",
        ],
    }


def _decision_input(panel: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "bar_is_complete",
        "source_artifact_hash",
        "stock_source_artifact_hash",
        "market_source_artifact_hash",
        "b0_source_artifact_hash",
        "clock_source_artifact_hash",
        "source_sequence_complete",
        "source_sequence_missing_reason",
        "b0_source_timestamp",
        "b0_source_bar_ordinal",
        "b0_available_timestamp",
        "b0_state_numeric",
        "b0_high_stress",
        "current_bar_log_return",
        "return_sum_6",
        "return_sum_12",
        "mean_abs_return_12",
        "bar_range_pct",
        "session_return",
        "clock_phase",
        "clock_sin",
        "clock_cos",
        "regime_log_activity_3",
        "regime_log_activity_12",
        "regime_log_bar_range",
        "log_relative_historical_volume",
    ]
    provenance_fields = [
        field
        for context_field in (
            "current_bar_log_return",
            "return_sum_6",
            "return_sum_12",
            "mean_abs_return_12",
            "bar_range_pct",
            "session_return",
            "clock_phase",
            "clock_sin",
            "clock_cos",
            "regime_log_activity_3",
            "regime_log_activity_12",
            "regime_log_bar_range",
            "log_relative_historical_volume",
            "b0_state_numeric",
            "b0_high_stress",
        )
        for field in (
            f"{context_field}__causal_valid",
            f"{context_field}__missing_reason",
        )
    ]
    fields.extend(provenance_fields)
    return panel[fields].copy()


def _write_state_posterior_ledger(
    path: Path,
    decisions: pd.DataFrame,
    state_export: Any,
) -> None:
    metadata_columns = [
        "decision_id",
        "run_id_v2",
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "dictionary_version",
        "state_model_version",
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "decision_timestamp",
        "source_artifact_hash",
        "stock_source_artifact_hash",
        "state_source_artifact_hash",
        "structural_source_artifact_hash",
        "source_sequence_complete",
        "source_sequence_missing_reason",
        "posterior_causal_valid",
        "posterior_missing_reason",
        "hard_state_legacy",
        "hard_state_posterior_map",
        "hard_state_hysteretic",
        "hard_run_age",
        "posterior_map_run_age",
        "expected_state_age",
        "posterior_entropy",
        "top_state",
        "second_state",
        "top_second_margin",
        "probability_state_persists_next_bar",
        "transition_probability_next_bar",
        "posterior_source_timestamp",
        "posterior_available_timestamp",
        "research_only",
        "execution_enabled",
        "order_placement",
        "broker_connected",
        "strategy_promotion",
    ]
    table = pa.Table.from_pandas(decisions[metadata_columns], preserve_index=False)
    state_values = pa.array(
        np.asarray(state_export.state_probabilities, dtype=np.float32).reshape(-1),
        type=pa.float32(),
    )
    state_array = pa.FixedSizeListArray.from_arrays(state_values, STATE_COUNT)
    age_width = int(np.prod(state_export.state_age_probabilities.shape[1:]))
    age_values = pa.array(
        np.asarray(state_export.state_age_probabilities, dtype=np.float32).reshape(-1),
        type=pa.float32(),
    )
    age_array = pa.FixedSizeListArray.from_arrays(age_values, age_width)
    next_values = pa.array(
        np.asarray(state_export.next_state_probabilities, dtype=np.float32).reshape(-1),
        type=pa.float32(),
    )
    next_array = pa.FixedSizeListArray.from_arrays(next_values, STATE_COUNT)
    table = table.append_column("posterior_state_probabilities", state_array)
    table = table.append_column("state_age_posterior", age_array)
    table = table.append_column("next_state_probabilities", next_array)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
    )


def _implementation_census_v2() -> pd.DataFrame:
    """AST-census every active or frozen loop implementation in declared scope."""

    scope = json.loads(CENSUS_SCOPE.read_text(encoding="utf-8"))
    roots = [REPO_ROOT / value for value in scope["roots"]]
    filename_tokens = tuple(str(value).lower() for value in scope["python_filename_tokens"])
    content_tokens = tuple(str(value).lower() for value in scope["python_content_tokens"])
    excluded = tuple(str(value) for value in scope["exclude_directory_tokens"])
    baseline_paths = set(
        _git_output(
            "ls-tree",
            "-r",
            "--name-only",
            str(scope["frozen_baseline_commit"]),
        ).splitlines()
    )
    corpus: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if any(token in path.parts for token in excluded):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            lower = text.lower()
            if any(token in path.name.lower() for token in filename_tokens) or any(
                token in lower for token in content_tokens
            ):
                corpus[relative] = text
    rows: list[dict[str, Any]] = []
    for relative, text in sorted(corpus.items()):
        lower = text.lower()
        path = REPO_ROOT / relative
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError:
            nodes: list[ast.AST] = []
        else:
            nodes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            ]
        consumers = sorted(
            other
            for other, other_text in corpus.items()
            if other != relative and path.stem in other_text
        )
        status = (
            "active_v2"
            if relative not in baseline_paths
            else "frozen_historical"
            if relative.startswith("research/slrno-v2/20260714-regime-loop-handoff/")
            else "active_legacy"
        )
        affected = {
            "affected_b0_provenance": "b0" in lower
            and any(token in lower for token in ("iloc[-1]", "last_row", "last =")),
            "affected_overlapping_target": any(
                token in lower for token in ("compatible_cycle", "cycle_target", "legacy_positive")
            ),
            "affected_primitive_repeat_mix": "cycle" in lower and "primitive_root" not in lower,
            "affected_rank_id": "cycle_" in lower and "semantic_loop_id" not in lower,
            "affected_length_mismatch": "cycle" in lower
            and any(token in lower for token in ("range(2, 6)", "{2, 3, 4}")),
            "affected_duration_tail": "duration" in lower
            and any(token in lower for token in (">= 24", "max_duration", "hazard[:, -1]")),
            "affected_hard_state_only": "hard_state" in lower and "posterior_state" not in lower,
            "affected_run_entry_only": any(
                token in lower
                for token in (
                    "run_entry",
                    "entry_run",
                    "run_start",
                    "build_online_runs",
                    "start_pos",
                )
            )
            and "every completed bar" not in lower,
            "affected_static_future_context": any(
                token in lower for token in ("anchor_context", "anchor_b0", "static_context")
            ),
            "affected_weak_null": any(
                token in lower for token in ("permutation", "permuted", "shuffle")
            )
            and "semi-markov null" not in lower,
            "affected_raw_frequency": "cycle_count" in lower and "information_gain" not in lower,
        }
        target_semantics = (
            "prefix-aware mutually exclusive first structural event"
            if "firstnextloopeventengine" in lower or "first_next_loop" in lower
            else "legacy overlapping compatible-cycle path"
            if affected["affected_overlapping_target"]
            else "structural loop/cycle lineage; no target in this component"
        )
        anchor = (
            "every completed bar"
            if "build_completed_bar_decisions" in lower
            else "state-run entry"
            if affected["affected_run_entry_only"]
            else "component-specific or not applicable"
        )
        loop_definition = (
            "semantic primitive/repeat/composite closed path"
            if "semantic_loop_id" in lower
            else "legacy discovered closed state cycle"
            if "cycle" in lower or "loop" in lower
            else "not applicable"
        )
        row_nodes = nodes or [None]
        for node in sorted(
            row_nodes,
            key=lambda value: (getattr(value, "lineno", 0), getattr(value, "name", "")),
        ):
            doc = ast.get_docstring(node, clean=True) if node is not None else None
            rows.append(
                {
                    "file": relative,
                    "function_or_class": getattr(node, "name", "<module>"),
                    "line": int(getattr(node, "lineno", 1)),
                    "purpose": (doc.splitlines()[0] if doc else path.stem.replace("_", " ")),
                    "active_versus_frozen_status": status,
                    "input_population": (
                        "bar-level structural panel"
                        if "read_parquet" in lower or "prepare_symbol_bars" in lower
                        else "hard state-run/event sequence"
                        if "state" in lower
                        else "component-specific"
                    ),
                    "forecast_anchor": anchor,
                    "loop_definition": loop_definition,
                    "orientation_handling": (
                        "explicit orientation metadata"
                        if "orientation_id" in lower
                        else "rotation/route handling embedded or absent"
                    ),
                    "target_semantics": target_semantics,
                    "timing_semantics": (
                        "explicit bar completion availability"
                        if "bar_complete_timestamp" in lower
                        else "historical timestamp convention"
                    ),
                    "feature_availability_convention": (
                        "explicit source and availability timestamps"
                        if "available_timestamp" in lower
                        else "historical/implicit"
                    ),
                    "known_consumers": consumers[:20],
                    **affected,
                }
            )
    census = pd.DataFrame(rows)
    required_names = (
        "run_causal_semimarkov_regime_loops.py",
        "run_causal_loop_prefix_path_forecast.py",
        "run_joint_history_semimarkov_loop_forecast.py",
        "factor_conditioned_loop_occurrence_core.py",
        "factor_conditioned_loop_occurrence_eval.py",
        "run_factor_conditioned_loop_occurrence_v1.py",
        "run_loop_burst_mechanism_v1.py",
        "frozen_loop_movement_shadow_core.py",
        "per_loop_quality_shadow_core.py",
    )
    absent = [name for name in required_names if not census["file"].str.endswith(name).any()]
    if absent:
        raise RuntimeError(f"implementation census omitted required lineage: {absent}")
    return census.sort_values(["file", "line", "function_or_class"], kind="mergesort")


def _historical_lineage_impact_v2(census: pd.DataFrame, *, b0_changed_runs: int) -> pd.DataFrame:
    """Conservatively disposition every prior loop-related source file."""

    historical = census.loc[census["active_versus_frozen_status"].ne("active_v2")].drop_duplicates(
        "file"
    )
    rows = []
    for row in historical.itertuples(index=False):
        overlapping = bool(row.affected_overlapping_target)
        run_entry = bool(row.affected_run_entry_only)
        b0 = bool(row.affected_b0_provenance)
        static = bool(row.affected_static_future_context)
        duration = bool(row.affected_duration_tail)
        hard = bool(row.affected_hard_state_only)
        rank = bool(row.affected_rank_id)
        if b0 and run_entry:
            severity = "provenance limitation"
        elif static or overlapping or duration or hard or rank:
            severity = "semantic limitation"
        else:
            severity = "none"
        rows.append(
            {
                "experiment_name": Path(row.file).stem,
                "historical_file_or_report": row.file,
                "target_type": row.target_semantics,
                "anchor_type": row.forecast_anchor,
                "uses_legacy_overlapping_labels": overlapping,
                "uses_state_run_entry_only": run_entry,
                "uses_b0_run_end_field_as_entry_context": b0 and run_entry,
                "uses_static_context_along_hypothetical_path": static,
                "uses_legacy_duration_overflow": duration,
                "uses_hard_state_only": hard,
                "uses_rank_based_cycle_id": rank,
                "affected_or_unaffected": "unaffected" if severity == "none" else "affected",
                "severity": severity,
                "historical_result_interpretability": (
                    "fully interpretable"
                    if severity == "none"
                    else "structurally interpretable only"
                ),
                "requires_rerun_for_v2_comparison": severity != "none",
                "b0_empirically_changed_runs_2024": b0_changed_runs,
                "b0_result_note": (
                    "Frozen builder assignment is last-row; reconstructed 2024 B0 values are "
                    "session-constant, while clock context changes within runs."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["experiment_name", "historical_file_or_report"], kind="mergesort"
    )


def _legacy_run_assignment_audit() -> pd.DataFrame:
    """Extract first/last-row field assignments from the frozen builder AST."""

    text = HISTORICAL_RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(HISTORICAL_RUNNER_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_online_runs"
    )
    records = []
    for mapping in (node for node in ast.walk(function) if isinstance(node, ast.Dict)):
        for key, value in zip(mapping.keys, mapping.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            expression = ast.unparse(value)
            if "frame.at[first" in expression:
                row_source = "first_row"
            elif "frame.at[last" in expression:
                row_source = "last_row"
            elif "previous_states" in expression:
                row_source = "causal_prior_run_history"
            else:
                row_source = "derived_or_literal"
            records.append(
                {
                    "file": HISTORICAL_RUNNER_PATH.relative_to(REPO_ROOT).as_posix(),
                    "function_or_class": "build_online_runs",
                    "field": key.value,
                    "assignment_expression": expression,
                    "row_source": row_source,
                    "source_line": int(getattr(value, "lineno", 0)),
                    "source_file_hash": _sha256_file(HISTORICAL_RUNNER_PATH),
                    "historical_stored_ledger_available": False,
                    "evidence_status": "frozen_source_AST_and_direct_bar_reconstruction",
                }
            )
    return pd.DataFrame(records).drop_duplicates("field").sort_values("source_line")


def _legacy_entry_context_consumers(census: pd.DataFrame) -> pd.DataFrame:
    """Enumerate later source uses of frozen run-context fields."""

    fields = ("b0_state_numeric", "b0_high_stress", "time_sin", "time_cos")
    rows = []
    for relative in sorted(census["file"].unique()):
        path = REPO_ROOT / relative
        if not path.is_file() or path == HISTORICAL_RUNNER_PATH:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        lower_file = "\n".join(lines).lower()
        for field in fields:
            for line_number, line in enumerate(lines, start=1):
                if field not in line:
                    continue
                rows.append(
                    {
                        "file": relative,
                        "field": field,
                        "source_line": line_number,
                        "source_expression": line.strip(),
                        "interprets_as_run_entry_context": any(
                            token in lower_file
                            for token in (
                                "run_entry",
                                "entry_run",
                                "run_start",
                                "anchor_run",
                                "entry_context",
                            )
                        ),
                        "consumer_status": (
                            "entry_context_consumer"
                            if any(
                                token in lower_file
                                for token in ("run_entry", "entry_run", "run_start")
                            )
                            else "context_use_not_proven_as_entry_anchor"
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["file", "field", "source_line"])


def _frozen_tree_hashes(baseline: str) -> pd.DataFrame:
    prefix = "research/slrno-v2/20260714-regime-loop-handoff"
    listing = _git_output("ls-tree", "-r", baseline, prefix).splitlines()
    rows = []
    for line in listing:
        metadata, path_text = line.split("\t", 1)
        blob = metadata.split()[2]
        path = REPO_ROOT / path_text
        if not path.is_file():
            raise FileNotFoundError(f"frozen baseline file disappeared: {path_text}")
        current_blob = _git_output("hash-object", path_text)
        rows.append(
            {
                "baseline_commit": baseline,
                "file": path_text,
                "baseline_git_blob": blob,
                "current_git_blob": current_blob,
                "byte_unchanged": blob == current_blob,
                "current_sha256": _sha256_file(path),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or not result["byte_unchanged"].all():
        changed = result.loc[~result["byte_unchanged"], "file"].tolist()
        raise AssertionError(f"frozen historical file changed: {changed}")
    return result


def _churn_table(decisions: pd.DataFrame, dictionary: LoopDictionary) -> pd.DataFrame:
    eligible_decisions = decisions.loc[decisions["structural_event_eligibility"].astype(bool)]
    rows = []
    totals = Counter()
    for (_, _), group in eligible_decisions.groupby(["symbol", "session"], sort=False):
        diagnostic = churn_diagnostics(
            group["hard_state_legacy"].to_numpy(dtype=int),
            margins=group["top_second_margin"].to_numpy(dtype=float),
            entropy=group["posterior_entropy"].to_numpy(dtype=float),
            low_margin_threshold=0.10,
        )
        for key in (
            "rows",
            "hard_transitions",
            "low_margin_hard_transitions",
            "one_bar_reversals",
            "two_bar_reversals",
        ):
            totals[key] += int(diagnostic[key])
    for key, value in totals.items():
        rows.append({"representation": "LEGACY_HARD_MAP", "metric": key, "value": value})
    hard_change = eligible_decisions.groupby(["symbol", "session"], sort=False)[
        "hard_state_legacy"
    ].transform(lambda values: values.ne(values.shift(1)))
    low_confidence_loop_dependency = int(
        (
            hard_change
            & eligible_decisions["top_second_margin"].lt(0.10)
            & eligible_decisions["previous_primitive_loop_1"].notna()
        ).sum()
    )
    rows.append(
        {
            "representation": "LEGACY_HARD_MAP",
            "metric": "loop_rows_dependent_on_low_margin_transition",
            "value": low_confidence_loop_dependency,
        }
    )
    for representation, state_column in (
        ("LEGACY_HARD_MAP", "hard_state_legacy"),
        ("CAUSAL_HYSTERETIC_STATE", "hard_state_hysteretic"),
    ):
        for definition in dictionary.definitions.values():
            count = 0
            for (_, _), group in eligible_decisions.groupby(["symbol", "session"], sort=False):
                states = group[state_column].to_numpy(dtype=int)
                compressed = states[np.r_[True, states[1:] != states[:-1]]]
                count += sum(
                    tuple(compressed[start : start + len(path)]) == path
                    for path in definition.oriented_paths
                    for start in range(len(compressed) - len(path) + 1)
                )
            rows.append(
                {
                    "representation": representation,
                    "metric": f"dictionary_count__{definition.semantic_loop_id}",
                    "value": count,
                }
            )
    rows.append(
        {
            "representation": "SOFT_POSTERIOR",
            "metric": "rows_with_sampled_soft_prefix_probability",
            "value": int(decisions["highest_soft_prefix_probability"].notna().sum()),
        }
    )
    rows.append(
        {
            "representation": "SOFT_POSTERIOR",
            "metric": "hard_completions_created",
            "value": 0,
        }
    )
    sampled_soft = eligible_decisions.loc[
        eligible_decisions["soft_completion_probabilities"].notna()
    ]
    for definition in dictionary.definitions.values():
        expected_count = sum(
            float(value.get(definition.semantic_loop_id, 0.0))
            for value in sampled_soft["soft_completion_probabilities"]
            if isinstance(value, dict)
        )
        rows.append(
            {
                "representation": "SOFT_POSTERIOR",
                "metric": (f"dictionary_expected_completion_count__{definition.semantic_loop_id}"),
                "value": expected_count,
                "denominator": "bounded_deterministic_soft_session_sample",
            }
        )
    return pd.DataFrame(rows)


def _v2_source_hashes(source_commit: str) -> pd.DataFrame:
    paths = [
        *sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "packages/stocker_research/src/stocker_research").glob(
                "*v2.py"
            )
        ),
        Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        Path(WORK_DIR / "audit_loop_event_semantics_v2.py").relative_to(REPO_ROOT).as_posix(),
        CONTRACT_PATH.relative_to(REPO_ROOT).as_posix(),
        CENSUS_SCOPE.relative_to(REPO_ROOT).as_posix(),
        Path(WORK_DIR / "tests/test_loop_event_semantics_v2.py").relative_to(REPO_ROOT).as_posix(),
    ]
    rows = []
    for relative in sorted(set(paths)):
        path = REPO_ROOT / relative
        try:
            commit_blob = _git_output("rev-parse", f"{source_commit}:{relative}")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"V2 source is not committed at reconstruction SHA: {relative}"
            ) from error
        rows.append(
            {
                "file": relative,
                "source_sha256": _sha256_file(path),
                "source_commit": source_commit,
                "commit_blob": commit_blob,
            }
        )
    return pd.DataFrame(rows)


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "file": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {"artifacts": rows, "artifact_count": len(rows)}


def _target_comparison_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the detailed Parquet ledger into a compact CSV report surface."""

    rows: list[dict[str, Any]] = []
    populations = {
        "all_completed_bars": pd.Series(True, index=detail.index),
        "eligible_completed_bars": detail["comparison_available"].astype(bool),
        "legacy_run_entries": detail["is_run_entry"].astype(bool),
    }
    for population, mask in populations.items():
        sample = detail.loc[mask]
        rows.append(
            {
                "record_type": "aggregate",
                "population": population,
                "v2_first_event": None,
                "decisions": len(sample),
                "comparison_available_decisions": int(sample["comparison_available"].sum()),
                "unavailable_decisions": int((~sample["comparison_available"]).sum()),
                "legacy_positive_decisions": int(sample["legacy_positive_count"].gt(0).sum()),
                "simultaneous_legacy_positive_decisions": int(
                    sample["legacy_positive_count"].gt(1).sum()
                ),
                "active_prefix_decisions": int(sample["active_prefix_count"].gt(0).sum()),
                "registered_event_set_difference_decisions": int(
                    sample["registered_event_set_differs"].sum()
                ),
                "semantic_difference_decisions": int(sample["semantics_differ"].sum()),
            }
        )
        for label, count in sample["v2_first_event"].value_counts().sort_index().items():
            rows.append(
                {
                    "record_type": "v2_first_event_count",
                    "population": population,
                    "v2_first_event": label,
                    "decisions": int(count),
                    "comparison_available_decisions": None,
                    "unavailable_decisions": None,
                    "legacy_positive_decisions": None,
                    "simultaneous_legacy_positive_decisions": None,
                    "active_prefix_decisions": None,
                    "registered_event_set_difference_decisions": None,
                    "semantic_difference_decisions": None,
                }
            )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> None:
    ARTIFACT_IDENTITY.clear()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, contract_hash = _load_contract()
    git_sha = _git_output("rev-parse", "HEAD")
    source_hash_by_symbol, data_snapshot_hash = _source_hashes()
    run_id = _sha256_bytes(
        _canonical_json_bytes(
            {
                "contract_hash": contract_hash,
                "git_sha": git_sha,
                "data_snapshot_hash": data_snapshot_hash,
                "dictionary_version": DICTIONARY_VERSION,
                "state_model_version": STATE_MODEL_VERSION,
            }
        )
    )[:24]

    panel = _prepare_panel(
        source_hash_by_symbol,
        data_snapshot_hash=data_snapshot_hash,
        contract_hash=contract_hash,
    )
    legacy_labels, state_export, v2_state_model, _ = _state_inference(panel)
    runs, b0_audit, b0_summary = _context_audit(
        panel,
        legacy_labels,
        run_id=run_id,
        git_sha=git_sha,
        contract_hash=contract_hash,
        data_snapshot_hash=data_snapshot_hash,
    )
    duration_model, duration_summary, duration_tail, duration_censor = _duration_artifacts(runs)
    sessions = _session_sequences(runs)
    legacy_dictionary, legacy_cycles = _legacy_dictionary()
    dictionary, candidate_census, selection, null_results, _ = _dictionary_research(
        sessions,
        contract,
        output_dir=output_dir,
        run_id=run_id,
        git_sha=git_sha,
        contract_hash=contract_hash,
        data_snapshot_hash=data_snapshot_hash,
    )
    ARTIFACT_IDENTITY.update(
        {
            "run_id_v2": run_id,
            "git_sha": git_sha,
            "contract_hash": contract_hash,
            "data_snapshot_hash": data_snapshot_hash,
            "source_artifact_hash": data_snapshot_hash,
            "dictionary_version": dictionary.version,
            "dictionary_hash": dictionary.dictionary_hash,
            "state_model_version": STATE_MODEL_VERSION,
        }
    )
    semantic_dictionary, migration, decomposition, composites = _dictionary_tables(
        dictionary, selection, legacy_dictionary
    )

    state_source_hash = _sha256_bytes(
        _canonical_json_bytes(
            {
                "model_file_hash": _sha256_file(MODEL_PATH),
                "preprocessing_file_hash": _sha256_file(PREPROCESSING_PATH),
                "duration_hazard_v2_hash": _sha256_bytes(
                    np.asarray(v2_state_model["duration_hazard"], dtype=np.float64).tobytes()
                ),
                "data_snapshot_hash": data_snapshot_hash,
                "state_model_version": STATE_MODEL_VERSION,
            }
        )
    )
    structural_source_hash = _sha256_bytes(
        _canonical_json_bytes(
            {
                "data_snapshot_hash": data_snapshot_hash,
                "state_source_hash": state_source_hash,
                "dictionary_hash": dictionary.dictionary_hash,
            }
        )
    )
    decision_input = _decision_input(panel)
    decision_input["state_source_artifact_hash"] = state_source_hash
    decision_input["structural_source_artifact_hash"] = structural_source_hash
    decisions = build_completed_bar_decisions(
        decision_input,
        state_export,
        legacy_hard_states=legacy_labels,
        git_sha=git_sha,
        contract_hash=contract_hash,
        data_snapshot_hash=data_snapshot_hash,
        dictionary_version=dictionary.version,
        state_model_version=STATE_MODEL_VERSION,
        hysteresis_config=HysteresisConfig(
            switch_probability=float(
                contract["state_representations"]["causal_hysteretic_state"]["switch_probability"]
            ),
            switch_margin=float(
                contract["state_representations"]["causal_hysteretic_state"]["switch_margin"]
            ),
        ),
        include_state_age_posterior=False,
    )
    decisions["posterior_causal_valid"] = decisions["source_sequence_complete"].astype(
        bool
    ) & decisions["posterior_available_timestamp"].le(decisions["decision_timestamp"])
    decisions["posterior_missing_reason"] = np.where(
        decisions["posterior_causal_valid"],
        None,
        "incomplete_source_sequence_posterior_segment_reset",
    )
    soft_sample = _balanced_session_sample(
        sessions, int(contract["nulls"]["primary_empirical_sessions_per_symbol"])
    )
    soft_keys = frozenset((item.symbol, item.session) for item in soft_sample)
    source_hashes = (
        ("frozen_model", _sha256_file(MODEL_PATH)),
        ("frozen_preprocessing", _sha256_file(PREPROCESSING_PATH)),
        ("data_snapshot", data_snapshot_hash),
        ("dictionary", dictionary.dictionary_hash),
    )

    legacy_bundle = build_loop_event_ledgers(
        decisions,
        dictionary=legacy_dictionary,
        horizon_bars=int(contract["event_contract"]["forecast_horizon_bars"]),
        allowed_states=frozenset(range(STATE_COUNT)),
        state_column="hard_state_legacy",
        source_hashes=source_hashes,
        soft_prefix_session_keys=frozenset(),
    )
    v2_bundle = build_loop_event_ledgers(
        decisions,
        dictionary=dictionary,
        horizon_bars=int(contract["event_contract"]["forecast_horizon_bars"]),
        allowed_states=frozenset(range(STATE_COUNT)),
        state_column="hard_state_legacy",
        source_hashes=source_hashes,
        soft_prefix_session_keys=soft_keys,
    )
    decisions = v2_bundle.decisions
    legacy_v2_comparison = compare_legacy_targets_to_v2_outcomes(
        legacy_bundle.legacy_targets,
        v2_bundle.outcomes,
        decisions,
    )

    for frame in (
        decisions,
        v2_bundle.prefixes,
        v2_bundle.completions,
        v2_bundle.outcomes,
    ):
        for identity, value in (
            ("run_id_v2", run_id),
            ("git_sha", git_sha),
            ("contract_hash", contract_hash),
            ("data_snapshot_hash", data_snapshot_hash),
            ("dictionary_version", dictionary.version),
            ("state_model_version", STATE_MODEL_VERSION),
        ):
            if identity not in frame:
                frame[identity] = value

    for frame in (legacy_bundle.legacy_targets,):
        for identity, value in (
            ("run_id_v2", run_id),
            ("git_sha", git_sha),
            ("contract_hash", contract_hash),
            ("data_snapshot_hash", data_snapshot_hash),
            ("dictionary_version", legacy_dictionary.version),
            ("semantic_dictionary_version", dictionary.version),
            ("comparison_dictionary_basis", "migrated_legacy_dictionary_v1"),
            ("state_model_version", STATE_MODEL_VERSION),
        ):
            if identity not in frame:
                frame[identity] = value

    for identity, value in (
        ("run_id_v2", run_id),
        ("git_sha", git_sha),
        ("contract_hash", contract_hash),
        ("data_snapshot_hash", data_snapshot_hash),
        ("dictionary_version", dictionary.version),
        ("legacy_dictionary_version", legacy_dictionary.version),
        ("comparison_dictionary_basis", "legacy_dictionary_v1_vs_semantic_dictionary_v2"),
        ("state_model_version", STATE_MODEL_VERSION),
    ):
        if identity not in legacy_v2_comparison:
            legacy_v2_comparison[identity] = value

    _write_state_posterior_ledger(
        output_dir / "state_posterior_ledger.parquet", decisions, state_export
    )
    del state_export
    gc.collect()

    b0_changed = int(
        b0_audit.loc[
            b0_audit["field"].isin(["b0_state_numeric", "b0_high_stress"]),
            "start_end_differ",
        ].sum()
    )
    census = _implementation_census_v2()
    lineage_impact = _historical_lineage_impact_v2(census, b0_changed_runs=b0_changed)
    legacy_assignment_audit = _legacy_run_assignment_audit()
    legacy_context_consumers = _legacy_entry_context_consumers(census)
    churn = _churn_table(decisions, dictionary)
    population = pd.DataFrame(
        [
            {
                "population": "all_completed_regular_session_bars",
                "rows": len(decisions),
                "symbols": decisions["symbol"].nunique(),
                "sessions": decisions.groupby(["symbol", "session"]).ngroups,
            },
            {
                "population": "legacy_state_run_entries",
                "rows": int(decisions["is_run_entry"].sum()),
                "symbols": decisions.loc[decisions["is_run_entry"], "symbol"].nunique(),
                "sessions": decisions.loc[decisions["is_run_entry"]]
                .groupby(["symbol", "session"])
                .ngroups,
            },
        ]
    )
    population["run_entry_is_strict_subset"] = int(decisions["is_run_entry"].sum()) < len(decisions)
    blockers = pd.DataFrame(
        [
            {
                "item": "initial causal B0 warm-up",
                "affected_rows": int(decisions["b0_state_numeric"].isna().sum()),
                "status": "known_missingness_preserved",
                "blocks_event_reconstruction": False,
            },
            {
                "item": "incomplete or ambiguous in-session source sequence",
                "affected_rows": int((~decisions["structural_event_eligibility"]).sum()),
                "status": (
                    "posterior_reset_at_gap; excluded_from_duration_dictionary_and_null; "
                    "failed_closed_as_UNAVAILABLE"
                ),
                "blocks_event_reconstruction": False,
            },
            {
                "item": "state-age posterior support",
                "affected_rows": len(decisions),
                "status": (
                    "V2_posterior_uses_78_age_support_with_explicit_geometric_tail; "
                    "legacy_hard_map_remains_separate"
                ),
                "blocks_event_reconstruction": False,
            },
            {
                "item": "soft prefix computation",
                "affected_rows": int(decisions["highest_soft_prefix_probability"].isna().sum()),
                "status": "bounded_deterministic_sample_by_contract",
                "blocks_event_reconstruction": False,
            },
        ]
    )
    frozen_hashes = _frozen_tree_hashes(str(contract["source"]["frozen_lineage_baseline_commit"]))
    v2_source_hashes = _v2_source_hashes(git_sha)
    source_files = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "source_file": str(_provider_file(symbol)),
                "source_file_hash": digest,
                "source_hash_scope": "filtered_2024_development_rows_only",
                "bounded_source_rows": _bounded_provider_hash(_provider_file(symbol))[1],
                "development_start": DEVELOPMENT_START,
                "development_end": DEVELOPMENT_END,
                "data_snapshot_hash": data_snapshot_hash,
            }
            for symbol, digest in sorted(source_hash_by_symbol.items())
        ]
    )

    _write_frame(output_dir / "implementation_census.csv", census)
    _write_frame(output_dir / "historical_lineage_impact.csv", lineage_impact)
    _write_frame(
        output_dir / "legacy_run_field_assignment_audit.csv",
        legacy_assignment_audit,
    )
    _write_frame(
        output_dir / "legacy_entry_context_consumers.csv",
        legacy_context_consumers,
    )
    _write_frame(output_dir / "b0_feature_provenance_audit.parquet", b0_audit)
    _write_frame(output_dir / "b0_start_end_difference_summary.csv", b0_summary)
    _write_json(output_dir / "feature_availability_manifest.json", _feature_manifest())
    _write_frame(output_dir / "legacy_cycle_dictionary.csv", legacy_cycles)
    _write_frame(output_dir / "semantic_loop_dictionary_v2.csv", semantic_dictionary)
    _write_frame(output_dir / "legacy_to_v2_loop_mapping.csv", migration)
    _write_frame(output_dir / "primitive_loop_decomposition.csv", decomposition)
    _write_frame(output_dir / "composite_component_mapping.csv", composites)
    _write_frame(output_dir / "causal_completed_bar_decisions.parquet", decisions)
    _write_frame(output_dir / "active_prefix_ledger.parquet", v2_bundle.prefixes)
    _write_frame(output_dir / "loop_completion_event_ledger.parquet", v2_bundle.completions)
    _write_frame(output_dir / "first_next_loop_outcomes.parquet", v2_bundle.outcomes)
    _write_frame(output_dir / "legacy_overlapping_targets.parquet", legacy_bundle.legacy_targets)
    _write_frame(
        output_dir / "legacy_v2_target_comparison_detail.parquet",
        legacy_v2_comparison,
    )
    _write_frame(
        output_dir / "legacy_v2_target_comparison.csv",
        _target_comparison_summary(legacy_v2_comparison),
    )
    _write_frame(output_dir / "duration_model_summary.csv", duration_summary)
    _write_frame(output_dir / "duration_tail_diagnostics.csv", duration_tail)
    _write_frame(output_dir / "duration_censoring_audit.csv", duration_censor)
    _write_frame(output_dir / "structural_null_results.csv", null_results)
    _write_frame(output_dir / "dictionary_candidate_census.parquet", candidate_census)
    _write_frame(output_dir / "semantic_dictionary_selection.csv", selection)
    _write_frame(output_dir / "state_churn_diagnostics.csv", churn)
    _write_frame(output_dir / "run_entry_vs_per_bar_population.csv", population)
    _write_frame(output_dir / "missingness_and_blockers.csv", blockers)
    _write_frame(output_dir / "structural_session_runs.parquet", runs)
    _write_frame(output_dir / "frozen_historical_tree_hashes.csv", frozen_hashes)
    _write_frame(output_dir / "v2_source_hashes.csv", v2_source_hashes)
    _write_frame(output_dir / "source_file_hashes.csv", source_files)
    np.savez_compressed(
        output_dir / "duration_model_v2.npz",
        hazard=duration_model.hazard,
        at_risk_counts=duration_model.at_risk_counts,
        exit_counts=duration_model.exit_counts,
        censored_counts=duration_model.censored_counts,
        research_only=np.asarray([True]),
        execution_enabled=np.asarray([False]),
        order_placement=np.asarray(["disabled"]),
        broker_connected=np.asarray([False]),
        strategy_promotion=np.asarray([False]),
        **{key: np.asarray([value]) for key, value in ARTIFACT_IDENTITY.items()},
    )
    metadata = {
        "run_id": run_id,
        "source_commit": git_sha,
        "contract_hash": contract_hash,
        "data_snapshot_hash": data_snapshot_hash,
        "dictionary_version": dictionary.version,
        "dictionary_hash": dictionary.dictionary_hash,
        "state_model_version": STATE_MODEL_VERSION,
        "state_age_posterior_support": 78,
        "state_age_terminal_bucket": "age_78_and_later_geometric_tail_not_forced_exit",
        "event_duration_support": 78,
        "symbols": list(SYMBOLS),
        "period": "2024 development",
        "completed_bar_decisions": len(decisions),
        "session_sequences": len(sessions),
        "semantic_dictionary_entries": len(dictionary.definitions),
        "legacy_dictionary_entries": len(legacy_dictionary.definitions),
        "primary_null_draws": int(contract["nulls"]["primary_draws"]),
        "economic_outcomes_read": False,
        "predictive_model_trained": False,
        "frozen_historical_files_unchanged": bool(frozen_hashes["byte_unchanged"].all()),
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    decision_payload = {
        "scientific_decision": "implementation_audit_incomplete",
        "ready_for_next_loop_forecast": False,
        "pending_requirement": "independent_audit_and_exact_rerun_identity",
        "known_limitations": blockers["status"].tolist(),
        "edge_claimed": False,
        "predictor_built": False,
        "exact_next_experiment": contract["exact_next_experiment"],
    }
    _write_json(output_dir / "decision.json", decision_payload)
    _write_json(output_dir / "artifact_manifest.json", _artifact_manifest(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Versioned primary or exact-rerun artifact directory.",
    )
    arguments = parser.parse_args()
    resolved = arguments.output.resolve()
    if ARTIFACT_PARENT.resolve() not in resolved.parents:
        raise SystemExit("output must be inside the versioned V2 artifact directory")
    run(resolved)


if __name__ == "__main__":
    main()
