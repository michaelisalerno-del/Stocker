from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

EXPERIMENT_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = EXPERIMENT_DIR.parents[2]
PRIMARY: Final[Path] = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS: Final[Path] = EXPERIMENT_DIR / "reports"
CONTRACT_PATH: Final[Path] = EXPERIMENT_DIR / "contract.json"

TAIL_PRIMARY: Final[Path] = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-tail-phase-v1"
    / "artifacts"
    / "primary"
)
TAIL_EPISODES_PATH: Final[Path] = TAIL_PRIMARY / "fresh_episode_results_v1.parquet"
TAIL_CHECKPOINTS_PATH: Final[Path] = TAIL_PRIMARY / "checkpoint_results_v1.parquet"
TAIL_PROVENANCE_PATH: Final[Path] = TAIL_PRIMARY / "provenance_manifest_v1.json"
PRIOR_PRIMARY: Final[Path] = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-signed-market-shock-transition-v1"
    / "artifacts"
    / "primary"
)
PRIOR_EPISODES_PATH: Final[Path] = (
    PRIOR_PRIMARY / "episode_market_state_response_v1.parquet"
)
PRIOR_MARKET_STATES_PATH: Final[Path] = (
    PRIOR_PRIMARY / "market_state_surface_v1.parquet"
)
PRIOR_TAIL_DIAGNOSTICS_PATH: Final[Path] = (
    PRIOR_PRIMARY / "tail_phase_diagnostics_v1.csv"
)
PRIOR_PROVENANCE_PATH: Final[Path] = PRIOR_PRIMARY / "provenance_manifest_v1.json"
STATE_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/Documents/Codex/"
    "2026-07-23-you-are-working-in-the-github-5/data/cache/"
    "minimal-intraday-iv-excess-holdout-v0/frozen_state_surface.parquet"
)
VTI_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock/symbol=VTI.US/timeframe=5m/data.parquet"
)
EXPECTED_HASHES: Final[dict[Path, str]] = {
    TAIL_EPISODES_PATH: (
        "a843fedf4c5df712237fd374c5efb3ff8925b575c875395ea648d786b589d9a3"
    ),
    TAIL_CHECKPOINTS_PATH: (
        "8dd0ef53d9c5493b70f600a28d6f77e8ffabd5e7b48a5378cf0bb4411382cb8f"
    ),
    STATE_PATH: (
        "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0"
    ),
    PRIOR_EPISODES_PATH: (
        "5abcff67f6c17ec6ab99c8bbc9d37e769594eb9cc90513882d3dd1e0f7ecf9a7"
    ),
    PRIOR_MARKET_STATES_PATH: (
        "d9cdb0bb99886968d72d464b24d53a4b330470156a9a918cf01b2ed0722966ed"
    ),
    PRIOR_TAIL_DIAGNOSTICS_PATH: (
        "ca811abf83d81fa74abfd52875d411623dbc9a681af8a343525e2db911bfae6b"
    ),
    PRIOR_PROVENANCE_PATH: (
        "a5d7eb4f68e16324ea45b85f5f490bf7c47ff76dec2a219d015739e8940ce75f"
    ),
}

sys.path.insert(0, str(REPO_ROOT / "packages" / "stocker_prospective" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "stocker_research" / "src"))

from stocker_prospective.opening_market_transition_v1 import (  # noqa: E402
    EXPECTED_OPENING_BAR_COUNT_V1,
    OPENING_MARKET_PROXY_V1,
    OpeningCalibrationPeriodV1,
    OpeningCalibrationQuantilesV1,
    OpeningMarketTransitionStateResultV1,
    OpeningPreEntryWindowV1,
    OpeningTransitionThresholdManifestV1,
    OpeningTransitionThresholdsV1,
    calculate_opening_preentry_window_v1,
    calculate_stock_opening_response_v1,
    classify_opening_market_transition_v1,
)
from stocker_prospective.signed_market_shock_v1 import (  # noqa: E402
    MarketShockBarV1,
    assert_unprotected_sessions_v1,
    frozen_material_move_v1,
    partition_material_endpoint_v1,
)
from stocker_research.m1c_opening_market_transition_v1 import (  # noqa: E402
    M1C_HIGH_MOVEMENT_THRESHOLD_V1,
    MINIMUM_PREDICTOR_SUPPORT_V1,
    FrozenOpeningResponseQuintilesV1,
    assign_opening_response_quintile_v1,
    freeze_opening_response_quintiles_v1,
    freeze_opening_thresholds_v1,
    validate_prior_population_reconciliation_v1,
)

DEVELOPMENT_START: Final[str] = "2024-01-01"
DEVELOPMENT_END: Final[str] = "2024-12-31"
ASSESSMENT_START: Final[str] = "2025-01-01"
ASSESSMENT_END: Final[str] = "2025-08-22"
STRESS_START: Final[str] = "2025-09-01"
STRESS_END: Final[str] = "2025-12-31"
PROTECTED_START: Final[str] = "2026-01-01"
BOOTSTRAP_DRAWS: Final[int] = 5000
SESSION_BOOTSTRAP_SEED: Final[int] = 2026072806
EVENT_BOOTSTRAP_SEED: Final[int] = 2026072807
NULL_DRAWS: Final[int] = 1000
NULL_SEED: Final[int] = 2026072808
IDENTITY: Final[list[str]] = ["stock", "session", "checkpoint"]
SEVERE_STATES: Final[tuple[str, str]] = (
    "NEGATIVE_SEVERE_OPENING_TRANSITION",
    "POSITIVE_SEVERE_OPENING_TRANSITION",
)
STATE_ORDER: Final[tuple[str, ...]] = (
    "NEGATIVE_SEVERE_OPENING_TRANSITION",
    "POSITIVE_SEVERE_OPENING_TRANSITION",
    "ELEVATED_OPENING_RANGE_NONDIRECTIONAL",
    "NORMAL_OPENING",
    "UNKNOWN_INCOMPLETE",
)
RUN_COMMAND: Final[str] = (
    "rtk uv run python research/directional-readiness/"
    "20260728-m1c-opening-market-transition-v1/run_experiment.py"
)


class ExperimentBlocked(RuntimeError):
    """A scientific or operational prerequisite failed closed."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_frame(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["rtk", "git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _partition(session: str) -> str | None:
    if DEVELOPMENT_START <= session <= DEVELOPMENT_END:
        return "development"
    if ASSESSMENT_START <= session <= ASSESSMENT_END:
        return "assessment"
    if STRESS_START <= session <= STRESS_END:
        return "stress"
    return None


def _verify_sources() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise ExperimentBlocked(f"required source missing: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ExperimentBlocked(f"required source hash drifted: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": observed,
                "bytes": path.stat().st_size,
                "hash_scope": "whole_file_known_to_end_by_2025-12-31",
            }
        )
    if not VTI_PATH.is_file():
        raise ExperimentBlocked("blocked_market_data")
    prior = cast(
        dict[str, Any],
        json.loads(PRIOR_PROVENANCE_PATH.read_text(encoding="utf-8")),
    )
    confirmations = cast(dict[str, Any], prior["confirmations"])
    if any(
        (
            bool(confirmations["protected_2026_outcomes_accessed"]),
            bool(confirmations["broker_accessed"]),
            bool(confirmations["order_routing_enabled"]),
            bool(confirmations["order_placed"]),
            not bool(confirmations["m1c_unchanged"]),
            not bool(confirmations["tail_phase_v1_unchanged"]),
            not bool(confirmations["a1_unchanged"]),
            bool(confirmations["contaminated_fields_used"]),
        )
    ):
        raise ExperimentBlocked("inherited frozen-system provenance violates V1")
    return records


def _opened_session_read(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=None if columns is None else list(columns),
        filters=[("session", "<", PROTECTED_START)],
    )
    assert_unprotected_sessions_v1(frame["session"])
    if frame["session"].astype(str).ge(PROTECTED_START).any():
        raise ExperimentBlocked(f"protected row admitted by bounded reader: {path}")
    return frame


def _load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    episodes = _opened_session_read(PRIOR_EPISODES_PATH)
    tail_episodes = _opened_session_read(TAIL_EPISODES_PATH)
    checkpoints = _opened_session_read(TAIL_CHECKPOINTS_PATH)
    prior_market_states = _opened_session_read(PRIOR_MARKET_STATES_PATH)
    state_columns = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "bar_is_complete",
        "open",
        "high",
        "low",
        "close",
        "source_gap_before",
        "source_gap_after",
        "source_data_error_in_session",
        "benchmark_available_timestamp",
        "vti__bar_log_return",
        "vti__bar_range_pct",
    ]
    bars = pd.read_parquet(
        STATE_PATH,
        columns=state_columns,
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", STRESS_END),
        ],
    ).rename(columns={"symbol": "stock"})
    assert_unprotected_sessions_v1(bars["session"])
    if bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ExperimentBlocked("stock bar identities are not unique")

    vti = pd.read_parquet(
        VTI_PATH,
        columns=[
            "source",
            "symbol",
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ],
        filters=[
            ("timestamp", ">=", datetime(2024, 1, 1, tzinfo=UTC)),
            ("timestamp", "<", datetime(2026, 1, 1, tzinfo=UTC)),
        ],
    )
    timestamps = pd.to_datetime(vti["timestamp"], utc=True, errors="raise")
    if timestamps.ge(pd.Timestamp(PROTECTED_START, tz=UTC)).any():
        raise ExperimentBlocked("protected VTI row admitted by bounded reader")
    return episodes, tail_episodes, checkpoints, prior_market_states, bars, vti


def _market_schedule() -> pd.DataFrame:
    import pandas_market_calendars as market_calendars

    calendar = market_calendars.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date="2023-12-01",
        end_date=STRESS_END,
    ).reset_index(names="session")
    schedule["session"] = pd.to_datetime(schedule["session"]).dt.strftime(
        "%Y-%m-%d"
    )
    schedule["previous_session_v1"] = schedule["session"].shift(1)
    schedule["partition"] = schedule["session"].map(_partition)
    return schedule.reset_index(drop=True)


def _prepare_vti_bars(vti: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    frame = vti.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    regular = minute.ge(570) & minute.lt(960)
    on_grid = (
        ((minute - 570) % 5).eq(0)
        & local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
    )
    frame = frame.loc[regular & on_grid].copy()
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame["session"] = local.dt.strftime("%Y-%m-%d")
    frame["bar_ordinal"] = ((minute - 570) // 5).astype(int)
    frame["bar_start_timestamp"] = frame["timestamp"]
    frame["bar_complete_timestamp"] = frame["timestamp"] + pd.Timedelta(minutes=5)
    frame["symbol"] = OPENING_MARKET_PROXY_V1
    frame["partition"] = frame["session"].map(_partition)
    if frame.duplicated(["session", "bar_ordinal"]).any():
        raise ExperimentBlocked("canonical VTI bars are not unique")
    prices = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    valid = (
        np.isfinite(prices.to_numpy(float)).all(axis=1)
        & prices.gt(0.0).all(axis=1)
        & prices["high"].ge(prices[["open", "close", "low"]].max(axis=1))
        & prices["low"].le(prices[["open", "close", "high"]].min(axis=1))
    )
    frame["finalised"] = valid.to_numpy(bool)
    frame["source_ohlc_valid_v1"] = valid.to_numpy(bool)
    frame = frame.sort_values(
        ["session", "bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    hash_columns = [
        "source",
        "symbol",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]
    return frame, _sha256_frame(vti.loc[:, hash_columns].reset_index(drop=True))


def _market_bar_groups(
    frame: pd.DataFrame,
) -> dict[str, tuple[MarketShockBarV1, ...]]:
    groups: dict[str, tuple[MarketShockBarV1, ...]] = {}
    for session, group in frame.groupby("session", sort=False):
        groups[str(session)] = tuple(
            MarketShockBarV1(
                symbol=OPENING_MARKET_PROXY_V1,
                session=date.fromisoformat(str(session)),
                bar_ordinal=int(row.bar_ordinal),
                bar_start_timestamp=pd.Timestamp(
                    row.bar_start_timestamp
                ).to_pydatetime(),
                bar_complete_timestamp=pd.Timestamp(
                    row.bar_complete_timestamp
                ).to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                finalised=bool(row.finalised),
            )
            for row in group.itertuples(index=False)
        )
    return groups


def _previous_session_close(
    *,
    previous_session: str,
    market_close: datetime,
    groups: Mapping[str, tuple[MarketShockBarV1, ...]],
) -> float | None:
    bars = [
        bar
        for bar in groups.get(previous_session, ())
        if bar.finalised and bar.bar_complete_timestamp <= market_close
    ]
    if not bars:
        return None
    final = max(bars, key=lambda bar: bar.bar_complete_timestamp)
    if final.bar_complete_timestamp != market_close:
        return None
    return float(final.close)


def _build_market_predictors(vti_bars: pd.DataFrame) -> pd.DataFrame:
    groups = _market_bar_groups(vti_bars)
    schedule = _market_schedule()
    schedule_lookup = {
        str(row.session): row for row in schedule.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    target_schedule = schedule.loc[schedule["partition"].notna()]
    for raw in target_schedule.itertuples(index=False):
        row = cast(Any, raw)
        session = str(row.session)
        previous_session = str(row.previous_session_v1)
        previous_schedule = schedule_lookup.get(previous_session)
        previous_close = (
            None
            if previous_schedule is None
            else _previous_session_close(
                previous_session=previous_session,
                market_close=pd.Timestamp(
                    previous_schedule.market_close
                ).to_pydatetime(),
                groups=groups,
            )
        )
        market_open = pd.Timestamp(row.market_open).to_pydatetime()
        signal = market_open + timedelta(
            minutes=5 * EXPECTED_OPENING_BAR_COUNT_V1
        )
        measurement = calculate_opening_preentry_window_v1(
            market_proxy=OPENING_MARKET_PROXY_V1,
            session=date.fromisoformat(session),
            previous_session=date.fromisoformat(previous_session),
            session_open_timestamp=market_open,
            signal_timestamp=signal,
            entry_timestamp=signal,
            completed_bars=groups.get(session, ()),
            prior_regular_session_close=previous_close,
        )
        records.append(
            {
                "session": session,
                "partition": str(row.partition),
                "checkpoint": 6,
                **measurement.model_dump(mode="python"),
            }
        )
    return pd.DataFrame(records)


def _window_from_row(row: Any) -> OpeningPreEntryWindowV1:
    return OpeningPreEntryWindowV1(
        market_proxy_v1=str(row.market_proxy_v1),
        session=date.fromisoformat(str(row.session)),
        previous_session_v1=date.fromisoformat(str(row.previous_session_v1)),
        checkpoint_v1=6,
        session_open_timestamp_v1=pd.Timestamp(
            row.session_open_timestamp_v1
        ).to_pydatetime(),
        signal_timestamp_v1=pd.Timestamp(row.signal_timestamp_v1).to_pydatetime(),
        entry_timestamp_v1=pd.Timestamp(row.entry_timestamp_v1).to_pydatetime(),
        opening_bar_ordinals_v1=tuple(row.opening_bar_ordinals_v1),
        expected_opening_bar_count_v1=int(
            row.expected_opening_bar_count_v1
        ),
        observed_opening_bar_count_v1=int(
            row.observed_opening_bar_count_v1
        ),
        final_complete_pre_entry_bar_start_v1=(
            None
            if pd.isna(row.final_complete_pre_entry_bar_start_v1)
            else pd.Timestamp(
                row.final_complete_pre_entry_bar_start_v1
            ).to_pydatetime()
        ),
        entry_bar_ordinal_v1=int(row.entry_bar_ordinal_v1),
        entry_bar_included_v1=False,
        market_session_open_v1=(
            None if pd.isna(row.market_session_open_v1) else float(row.market_session_open_v1)
        ),
        market_prior_regular_session_close_v1=(
            None
            if pd.isna(row.market_prior_regular_session_close_v1)
            else float(row.market_prior_regular_session_close_v1)
        ),
        market_last_complete_pre_entry_close_v1=(
            None
            if pd.isna(row.market_last_complete_pre_entry_close_v1)
            else float(row.market_last_complete_pre_entry_close_v1)
        ),
        market_opening_return_v1=(
            None
            if pd.isna(row.market_opening_return_v1)
            else float(row.market_opening_return_v1)
        ),
        market_opening_range_v1=(
            None
            if pd.isna(row.market_opening_range_v1)
            else float(row.market_opening_range_v1)
        ),
        market_overnight_gap_v1=(
            None
            if pd.isna(row.market_overnight_gap_v1)
            else float(row.market_overnight_gap_v1)
        ),
        market_total_transition_v1=(
            None
            if pd.isna(row.market_total_transition_v1)
            else float(row.market_total_transition_v1)
        ),
        market_gap_open_alignment_v1=str(row.market_gap_open_alignment_v1),
        maximum_market_timestamp_v1=(
            None
            if pd.isna(row.maximum_market_timestamp_v1)
            else pd.Timestamp(row.maximum_market_timestamp_v1).to_pydatetime()
        ),
        complete_v1=bool(row.complete_v1),
        missing_reasons_v1=tuple(row.missing_reasons_v1),
    )


def _apply_market_states(
    predictors: pd.DataFrame,
    thresholds: OpeningTransitionThresholdsV1,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for raw in predictors.itertuples(index=False):
        row = cast(Any, raw)
        window = _window_from_row(row)
        state = classify_opening_market_transition_v1(
            window=window,
            thresholds=thresholds,
        )
        records.append(
            {
                **window.model_dump(mode="python"),
                "session": str(row.session),
                "partition": str(row.partition),
                "checkpoint": 6,
                "market_window_complete_v1": window.complete_v1,
                "market_window_missing_reasons_v1": window.missing_reasons_v1,
                **state.model_dump(mode="python"),
                **{
                    f"threshold_{key}": value
                    for key, value in thresholds.model_dump(
                        mode="python"
                    ).items()
                },
            }
        )
    return pd.DataFrame(records)


def _stock_bar_cache(
    bars: pd.DataFrame,
    identities: pd.DataFrame,
) -> dict[tuple[str, str], tuple[MarketShockBarV1, ...]]:
    needed = identities[["stock", "session"]].drop_duplicates()
    selected = bars.merge(
        needed,
        on=["stock", "session"],
        how="inner",
        validate="many_to_one",
    )
    cache: dict[tuple[str, str], tuple[MarketShockBarV1, ...]] = {}
    for (stock, session), group in selected.groupby(
        ["stock", "session"],
        sort=False,
    ):
        cache[(str(stock), str(session))] = tuple(
            MarketShockBarV1(
                symbol=str(stock),
                session=date.fromisoformat(str(session)),
                bar_ordinal=int(row.bar_ordinal),
                bar_start_timestamp=pd.Timestamp(
                    row.bar_start_timestamp
                ).to_pydatetime(),
                bar_complete_timestamp=pd.Timestamp(
                    row.bar_complete_timestamp
                ).to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                finalised=bool(row.bar_is_complete)
                and not bool(row.source_data_error_in_session)
                and not bool(row.source_gap_before)
                and not bool(row.source_gap_after),
            )
            for row in group.sort_values(
                "bar_ordinal",
                kind="mergesort",
            ).itertuples(index=False)
        )
    return cache


def _validate_endpoint_partition(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    signed = pd.to_numeric(output["future_signed_return_15m"], errors="coerce")
    threshold = pd.to_numeric(output["threshold_15m"], errors="coerce")
    valid = (
        np.isfinite(signed.to_numpy(float))
        & np.isfinite(threshold.to_numpy(float))
        & threshold.gt(0.0).to_numpy(bool)
    )
    states = pd.Series("UNKNOWN_INCOMPLETE", index=output.index, dtype="object")
    states.loc[valid] = [
        partition_material_endpoint_v1(
            signed_return=float(observed),
            threshold_15m=float(scale),
        )
        for observed, scale in zip(
            signed.loc[valid],
            threshold.loc[valid],
            strict=True,
        )
    ]
    strict = pd.Series(
        [
            frozen_material_move_v1(
                signed_return=float(observed),
                threshold_15m=float(scale),
            )
            for observed, scale in zip(
                signed.loc[valid],
                threshold.loc[valid],
                strict=True,
            )
        ],
        index=output.index[valid],
    )
    partition_event = states.loc[valid].isin(["MATERIAL_UP", "MATERIAL_DOWN"])
    if not partition_event.equals(strict):
        raise ExperimentBlocked("strict target partition equivalence failed")
    archived = output.loc[valid, "primary_outcome_state_v1"].astype(str)
    if not archived.equals(states.loc[valid]):
        raise ExperimentBlocked("prior strict endpoint partition drifted")
    output["primary_outcome_state_v1"] = states
    output["primary_outcome_complete_v1"] = valid
    output["future_absolute_movement_15m_v1"] = np.abs(signed)
    output["future_iv_residual_15m_v1"] = np.abs(signed) - threshold
    output["future_exceed_iv_15m_v1"] = states.isin(
        ["MATERIAL_UP", "MATERIAL_DOWN"]
    ).where(valid)
    return output


def _state_result_from_row(row: Any) -> OpeningMarketTransitionStateResultV1:
    return OpeningMarketTransitionStateResultV1(
        opening_market_transition_state_v1=str(
            row.opening_market_transition_state_v1
        ),
        opening_transition_sign_v1=(
            None
            if pd.isna(row.opening_transition_sign_v1)
            else int(row.opening_transition_sign_v1)
        ),
        opening_transition_event_id_v1=(
            None
            if pd.isna(row.opening_transition_event_id_v1)
            else str(row.opening_transition_event_id_v1)
        ),
        complete_v1=bool(row.opening_transition_state_complete_v1),
        missing_reasons_v1=tuple(
            row.opening_transition_state_missing_reasons_v1
        ),
    )


def _attach_stock_responses(
    panel: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    output = panel.copy()
    cache = _stock_bar_cache(bars, output)
    records: list[dict[str, Any]] = []
    for raw in output.itertuples(index=False):
        row = cast(Any, raw)
        result = calculate_stock_opening_response_v1(
            symbol=str(row.stock),
            session=date.fromisoformat(str(row.session)),
            session_open_timestamp=pd.Timestamp(
                row.session_open_timestamp_v1
            ).to_pydatetime(),
            signal_timestamp=pd.Timestamp(row.signal_timestamp).to_pydatetime(),
            completed_stock_bars=cache.get(
                (str(row.stock), str(row.session)),
                (),
            ),
            market_opening_return_v1=(
                None
                if pd.isna(row.market_opening_return_v1)
                else float(row.market_opening_return_v1)
            ),
            opening_transition_state_v1=_state_result_from_row(row),
            threshold_15m=(
                None if pd.isna(row.threshold_15m) else float(row.threshold_15m)
            ),
        )
        stock_bars = {
            bar.bar_ordinal: bar
            for bar in cache.get((str(row.stock), str(row.session)), ())
        }
        previous = stock_bars.get(4)
        latest = stock_bars.get(5)
        recent = (
            math.log(latest.close / previous.close)
            if previous is not None
            and latest is not None
            and previous.close > 0.0
            and latest.close > 0.0
            and previous.finalised
            and latest.finalised
            else math.nan
        )
        records.append(
            {
                **result.model_dump(mode="python"),
                "recent_stock_return_5m_v1": recent,
            }
        )
    response = pd.DataFrame(records, index=output.index).rename(
        columns={
            "complete_v1": "stock_opening_response_complete_v1",
            "missing_reasons_v1": (
                "stock_opening_response_missing_reasons_v1"
            ),
        }
    )
    for column in response:
        output[column] = response[column]
    return output


def _prepare_episode_panel(
    episodes: pd.DataFrame,
    market_states: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    source = episodes.loc[
        pd.to_numeric(episodes["checkpoint"], errors="coerce").eq(6)
        & episodes["m1c_tail_phase_v1"].eq("FIRST_ENTRY")
    ].copy()
    if source.duplicated(IDENTITY).any():
        raise ExperimentBlocked("checkpoint-6 fresh episode identities are not unique")
    market = market_states.drop(columns="partition").rename(
        columns={
            "complete_v1": "opening_transition_state_complete_v1",
            "missing_reasons_v1": (
                "opening_transition_state_missing_reasons_v1"
            ),
        }
    )
    output = source.merge(
        market,
        on=["session", "checkpoint"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_market"),
    )
    source_signal = pd.to_datetime(
        output["signal_timestamp"],
        utc=True,
        errors="coerce",
    )
    market_signal = pd.to_datetime(
        output["signal_timestamp_v1"],
        utc=True,
        errors="coerce",
    )
    entry = pd.to_datetime(
        output["prospective_entry_timestamp"],
        utc=True,
        errors="coerce",
    )
    if not source_signal.equals(market_signal) or not source_signal.equals(entry):
        raise ExperimentBlocked("checkpoint-6 signal, market, and entry timestamps differ")
    output = _validate_endpoint_partition(output)
    output = _attach_stock_responses(output, bars)
    output["month_v1"] = output["session"].astype(str).str[:7]
    signed = pd.to_numeric(output["future_signed_return_15m"], errors="coerce")
    sign = pd.to_numeric(output["opening_transition_sign_v1"], errors="coerce")
    output["market_follow_aligned_return_v1"] = sign * signed
    output["amplification_aligned_return_v1"] = sign * signed
    output["resistance_aligned_return_v1"] = -sign * signed
    output["followed_opening_transition_v1"] = pd.Series(
        pd.NA,
        index=output.index,
        dtype="Int64",
    )
    material = output["primary_outcome_state_v1"].isin(
        ["MATERIAL_UP", "MATERIAL_DOWN"]
    ) & sign.notna()
    direction = np.where(
        output.loc[material, "primary_outcome_state_v1"].eq("MATERIAL_UP"),
        1,
        -1,
    )
    output.loc[material, "followed_opening_transition_v1"] = (
        direction == sign.loc[material].to_numpy(float)
    ).astype(int)
    output["market_follow_action_v1"] = "ABSTAIN"
    output.loc[sign.eq(1), "market_follow_action_v1"] = "CALL"
    output.loc[sign.eq(-1), "market_follow_action_v1"] = "PUT"
    output["amplification_action_v1"] = "ABSTAIN"
    amplify = output["stock_opening_response_class_v1"].eq("AMPLIFYING")
    output.loc[amplify & sign.eq(1), "amplification_action_v1"] = "CALL"
    output.loc[amplify & sign.eq(-1), "amplification_action_v1"] = "PUT"
    output["resistance_action_v1"] = "ABSTAIN"
    resist = output["stock_opening_response_class_v1"].eq("RESISTING")
    output.loc[resist & sign.eq(1), "resistance_action_v1"] = "PUT"
    output.loc[resist & sign.eq(-1), "resistance_action_v1"] = "CALL"
    output["final_eligible_v1"] = (
        output["m1c_high_tail_v1"].astype(bool)
        & output["opening_transition_state_complete_v1"].astype(bool)
        & output["stock_opening_response_complete_v1"].astype(bool)
        & output["primary_outcome_complete_v1"].astype(bool)
    )
    output["opening_regime_cluster_v1"] = np.where(
        output["opening_transition_event_id_v1"].notna(),
        output["opening_transition_event_id_v1"],
        "opening-regime-"
        + output["session"].astype(str)
        + "-"
        + output["opening_market_transition_state_v1"].astype(str),
    )
    return output


def _frozen_regression(
    tail_episodes: pd.DataFrame,
    inherited_episodes: pd.DataFrame,
) -> dict[str, Any]:
    fields = [
        "M1C_probability",
        "m1c_high_tail_v1",
        "m1c_tail_phase_v1",
        "A1_probability_up_v1",
        "A1_action_v1",
        "episode_id",
        "existing_fresh_episode_identifier",
    ]
    left = tail_episodes.loc[:, [*IDENTITY, *fields]].copy()
    right = inherited_episodes.loc[:, [*IDENTITY, *fields]].copy()
    comparison = left.merge(
        right,
        on=IDENTITY,
        how="inner",
        validate="one_to_one",
        suffixes=("_tail", "_inherited"),
    )
    if len(comparison) != len(left) or len(comparison) != len(right):
        raise ExperimentBlocked("fresh episode identities changed")
    for field in fields:
        first = comparison[f"{field}_tail"]
        second = comparison[f"{field}_inherited"]
        if pd.api.types.is_numeric_dtype(first):
            numeric_first = pd.to_numeric(first, errors="coerce")
            numeric_second = pd.to_numeric(second, errors="coerce")
            if not np.allclose(
                numeric_first.to_numpy(float),
                numeric_second.to_numpy(float),
                rtol=0.0,
                atol=1e-15,
                equal_nan=True,
            ):
                raise ExperimentBlocked(f"frozen regression changed {field}")
        elif not first.fillna("<NA>").astype(str).equals(
            second.fillna("<NA>").astype(str)
        ):
            raise ExperimentBlocked(f"frozen regression changed {field}")
    return {
        "passed": True,
        "rows_compared": int(len(comparison)),
        "fields": fields,
        "tolerance": 1e-15,
    }


def _prior_population_reconciliation(
    checkpoints: pd.DataFrame,
    tail_episodes: pd.DataFrame,
    prior_episodes: pd.DataFrame,
    prior_market_states: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market = prior_market_states[
        ["session", "checkpoint", "market_shock_state_v1"]
    ].copy()
    high_tail = checkpoints.loc[
        checkpoints["m1c_high_tail_v1"].astype(bool)
        & checkpoints["partition"].isin(["assessment", "stress"])
    ].merge(
        market,
        on=["session", "checkpoint"],
        how="left",
        validate="many_to_one",
    )
    candidates = high_tail.loc[
        high_tail["market_shock_state_v1"].isin(
            ["NEGATIVE_SHOCK_ONSET", "POSITIVE_SHOCK_ONSET"]
        )
        & high_tail["m1c_tail_phase_v1"].isin(["FIRST_ENTRY", "RE_ENTRY"])
    ].copy()
    primary = prior_episodes.loc[
        prior_episodes["partition"].isin(["assessment", "stress"])
        & prior_episodes["market_shock_state_v1"].isin(
            ["NEGATIVE_SHOCK_ONSET", "POSITIVE_SHOCK_ONSET"]
        )
        & prior_episodes["shock_response_complete_v1"].astype(bool)
        & prior_episodes["primary_outcome_complete_v1"].astype(bool)
        & prior_episodes["shock_response_class_v1"].isin(
            ["AMPLIFYING", "RESISTING", "NEUTRAL_EXACT"]
        ),
        [
            *IDENTITY,
            "episode_id",
            "existing_fresh_episode_identifier",
            "shock_response_complete_v1",
            "primary_outcome_complete_v1",
            "shock_response_class_v1",
        ],
    ].copy()
    if primary.duplicated(IDENTITY).any():
        raise ExperimentBlocked("prior primary shock population is duplicated")
    canonical = tail_episodes[
        [
            *IDENTITY,
            "episode_id",
            "existing_fresh_episode_identifier",
            "signal_timestamp",
        ]
    ].copy()
    if canonical.duplicated(IDENTITY).any():
        raise ExperimentBlocked("canonical fresh episode population is duplicated")

    merged = candidates.merge(
        primary,
        on=IDENTITY,
        how="left",
        validate="one_to_one",
        suffixes=("", "_primary"),
    )
    canonical_keys = canonical.set_index(IDENTITY)
    records: list[dict[str, Any]] = []
    for raw in merged.sort_values(IDENTITY, kind="mergesort").itertuples(
        index=False
    ):
        row = cast(Any, raw)
        key = (str(row.stock), str(row.session), int(row.checkpoint))
        included = not pd.isna(row.episode_id)
        fresh_episode_id: str | None = (
            None if not included else str(row.episode_id)
        )
        previous_id: str | None = None
        spacing: float | None = None
        if included:
            reason = (
                "included:canonical_fresh_episode;"
                "complete_stock_response;complete_directional_outcome;"
                "valid_response_class"
            )
        elif key in canonical_keys.index:
            canonical_row = canonical_keys.loc[key]
            fresh_episode_id = str(canonical_row["episode_id"])
            reason = "different_population_definition:unexpected_primary_filter"
        else:
            same_session = canonical.loc[
                canonical["stock"].astype(str).eq(str(row.stock))
                & canonical["session"].astype(str).eq(str(row.session))
            ].copy()
            candidate_signal = pd.Timestamp(row.signal_timestamp)
            same_session["_signal"] = pd.to_datetime(
                same_session["signal_timestamp"],
                utc=True,
            )
            previous = same_session.loc[
                same_session["_signal"].lt(candidate_signal)
            ].sort_values("_signal", kind="mergesort")
            if len(previous):
                nearest = previous.iloc[-1]
                previous_id = str(nearest["episode_id"])
                spacing = (
                    candidate_signal
                    - pd.Timestamp(nearest["signal_timestamp"])
                ).total_seconds() / 60.0
            reason = "different_population_definition:"
            if spacing is not None and spacing < 30.0:
                reason += (
                    "tail_phase_re_entry_not_frozen_fresh_episode;"
                    f"minimum_episode_spacing_not_met:{spacing:g}<30"
                )
            else:
                reason += "not_present_in_canonical_fresh_episode_population"
        records.append(
            {
                "period": str(row.partition),
                "stock": str(row.stock),
                "session": str(row.session),
                "checkpoint": int(row.checkpoint),
                "fresh_episode_id": fresh_episode_id,
                "prior_canonical_fresh_episode_id_v1": previous_id,
                "minutes_since_prior_canonical_fresh_episode_v1": spacing,
                "tail_phase_v1": str(row.m1c_tail_phase_v1),
                "market_shock_state_v1": str(row.market_shock_state_v1),
                "included_in_primary_signed_shock_population_v1": included,
                "included_in_tail_phase_diagnostics_v1": True,
                "inclusion_exclusion_reason_v1": reason,
            }
        )
    result = pd.DataFrame(records)
    expected_tail = {
        (
            str(row.period),
            str(row.stock),
            str(row.session),
            int(row.checkpoint),
        )
        for row in result.itertuples(index=False)
    }
    expected_primary = set(primary["episode_id"].astype(str))
    validate_prior_population_reconciliation_v1(
        result,
        expected_tail_keys=expected_tail,
        expected_primary_episode_ids=expected_primary,
    )
    observed = (
        result.groupby("period", sort=False)
        .agg(
            tail_diagnostic_candidate_count=(
                "included_in_tail_phase_diagnostics_v1",
                "sum",
            ),
            primary_signed_shock_count=(
                "included_in_primary_signed_shock_population_v1",
                "sum",
            ),
        )
        .reset_index()
    )
    counts = {
        str(row.period): {
            "tail_phase_candidate_count": int(
                row.tail_diagnostic_candidate_count
            ),
            "primary_signed_shock_count": int(row.primary_signed_shock_count),
            "excluded_by_population_definition": int(
                row.tail_diagnostic_candidate_count
                - row.primary_signed_shock_count
            ),
        }
        for row in observed.itertuples(index=False)
    }
    expected_counts = {
        "assessment": (15, 9, 6),
        "stress": (34, 29, 5),
    }
    for period, expected in expected_counts.items():
        item = counts.get(period, {})
        observed_tuple = (
            item.get("tail_phase_candidate_count"),
            item.get("primary_signed_shock_count"),
            item.get("excluded_by_population_definition"),
        )
        if observed_tuple != expected:
            raise ExperimentBlocked(
                f"blocked_population_accounting:{period}:{observed_tuple}"
            )
    excluded = result.loc[
        ~result["included_in_primary_signed_shock_population_v1"].astype(bool)
    ]
    if (
        len(excluded) != 11
        or not pd.to_numeric(
            excluded["minutes_since_prior_canonical_fresh_episode_v1"],
            errors="coerce",
        ).eq(20.0).all()
    ):
        raise ExperimentBlocked("blocked_population_accounting")
    summary = {
        "status": "fully_reconciled",
        "counts": counts,
        "explanation": (
            "The Tail Phase diagnostic used every high-M1C FIRST_ENTRY or "
            "RE_ENTRY checkpoint row, while the primary study used canonical "
            "fresh episodes with the frozen 30-minute spacing rule."
        ),
        "excluded_row_count": 11,
        "excluded_reason": (
            "Each excluded RE_ENTRY occurred 20 minutes after a canonical "
            "fresh episode and therefore failed the frozen 30-minute spacing."
        ),
        "incomplete_stock_response_exclusions": 0,
        "invalid_iv_scale_exclusions": 0,
        "incomplete_outcome_exclusions": 0,
        "invalid_response_class_exclusions": 0,
        "duplicate_or_dependent_episode_exclusions": 0,
        "reporting_bug": False,
        "population_construction_bug": False,
        "terminology_ambiguity": True,
        "prior_scientific_conclusion_changed": False,
    }
    return result, summary


def _proxy_alignment_audit(
    vti_bars: pd.DataFrame,
    stock_bars: pd.DataFrame,
) -> dict[str, Any]:
    proxy = vti_bars[
        ["session", "bar_ordinal", "open", "high", "low", "close"]
    ].copy()
    previous_close = proxy.groupby("session", sort=False)["close"].shift(1)
    previous_ordinal = proxy.groupby("session", sort=False)["bar_ordinal"].shift(1)
    denominator = previous_close.where(
        proxy["bar_ordinal"].eq(previous_ordinal + 1),
        proxy["open"],
    )
    proxy["_return"] = np.log(proxy["close"] / denominator)
    proxy["_range"] = (proxy["high"] - proxy["low"]) / proxy["open"]
    archived = stock_bars[
        [
            "session",
            "bar_ordinal",
            "vti__bar_log_return",
            "vti__bar_range_pct",
            "benchmark_available_timestamp",
            "bar_complete_timestamp",
        ]
    ].drop_duplicates(["session", "bar_ordinal"])
    consistency = stock_bars.groupby(
        ["session", "bar_ordinal"],
        sort=False,
    )[["vti__bar_log_return", "vti__bar_range_pct"]].nunique(dropna=False)
    if (consistency > 1).any().any():
        raise ExperimentBlocked("archived VTI fields differ across stocks")
    comparison = proxy.merge(
        archived,
        on=["session", "bar_ordinal"],
        how="inner",
        validate="one_to_one",
    )
    valid = (
        comparison["vti__bar_log_return"].notna()
        & comparison["vti__bar_range_pct"].notna()
        & comparison["_return"].notna()
        & comparison["_range"].notna()
    )
    return_difference = np.abs(
        comparison.loc[valid, "_return"]
        - comparison.loc[valid, "vti__bar_log_return"]
    )
    range_difference = np.abs(
        comparison.loc[valid, "_range"]
        - comparison.loc[valid, "vti__bar_range_pct"]
    )
    available_after = (
        pd.to_datetime(comparison["benchmark_available_timestamp"], utc=True)
        > pd.to_datetime(comparison["bar_complete_timestamp"], utc=True)
    )
    maximum_return = (
        float(return_difference.max()) if len(return_difference) else math.nan
    )
    maximum_range = (
        float(range_difference.max()) if len(range_difference) else math.nan
    )
    if (
        not math.isfinite(maximum_return)
        or maximum_return > 1e-12
        or not math.isfinite(maximum_range)
        or maximum_range > 1e-12
        or available_after.any()
    ):
        raise ExperimentBlocked("canonical VTI causal alignment regression failed")
    return {
        "rows_compared": int(valid.sum()),
        "maximum_absolute_bar_return_difference": maximum_return,
        "maximum_absolute_bar_range_difference": maximum_range,
        "benchmark_available_after_stock_bar_count": int(available_after.sum()),
        "passed": True,
    }


def _timing_audit(
    market_states: pd.DataFrame,
) -> dict[str, Any]:
    sample = market_states.loc[
        market_states["session"].eq("2025-01-02")
    ].iloc[0]
    open_timestamp = pd.Timestamp(sample["session_open_timestamp_v1"])
    final_start = pd.Timestamp(
        sample["final_complete_pre_entry_bar_start_v1"]
    )
    signal = pd.Timestamp(sample["signal_timestamp_v1"])
    entry = pd.Timestamp(sample["entry_timestamp_v1"])
    expected_ordinals = tuple(range(EXPECTED_OPENING_BAR_COUNT_V1))
    if (
        tuple(sample["opening_bar_ordinals_v1"]) != expected_ordinals
        or final_start != open_timestamp + pd.Timedelta(minutes=25)
        or signal != open_timestamp + pd.Timedelta(minutes=30)
        or entry != signal
        or bool(sample["entry_bar_included_v1"])
    ):
        raise ExperimentBlocked("checkpoint-6 timing audit failed")
    local_open = open_timestamp.tz_convert("America/New_York")
    local_final = final_start.tz_convert("America/New_York")
    local_signal = signal.tz_convert("America/New_York")
    return {
        "checkpoint": 6,
        "checkpoint_meaning": (
            "six complete five-minute regular-session bars are available"
        ),
        "session_open_timestamp_example_utc": open_timestamp,
        "session_open_timestamp_local": local_open,
        "final_complete_bar_start_example_utc": final_start,
        "final_complete_bar_start_local": local_final,
        "checkpoint_6_signal_timestamp_example_utc": signal,
        "checkpoint_6_signal_timestamp_local": local_signal,
        "next_bar_open_entry_timestamp_example_utc": entry,
        "next_bar_open_entry_timestamp_local": local_signal,
        "complete_regular_session_bars_before_entry": 6,
        "bar_ordinals": list(expected_ordinals),
        "bar_local_intervals": [
            "09:30-09:35",
            "09:35-09:40",
            "09:40-09:45",
            "09:45-09:50",
            "09:50-09:55",
            "09:55-10:00",
        ],
        "entry_bar_ordinal": 6,
        "entry_bar_local_interval": "10:00-10:05",
        "entry_bar_excluded": True,
        "partial_bars_included": False,
        "expected_opening_bar_count": 6,
    }


def _market_proxy_audit(
    vti_raw: pd.DataFrame,
    vti_bars: pd.DataFrame,
    market_states: pd.DataFrame,
    bounded_hash: str,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    raw_timestamps = pd.to_datetime(vti_raw["timestamp"], utc=True)
    complete = market_states["market_window_complete_v1"].astype(bool)
    current_six_available = (
        pd.to_numeric(
            market_states["observed_opening_bar_count_v1"],
            errors="coerce",
        ).eq(EXPECTED_OPENING_BAR_COUNT_V1)
    )
    by_period = []
    for period in ("development", "assessment", "stress"):
        group = market_states.loc[market_states["partition"].eq(period)]
        by_period.append(
            {
                "period": period,
                "scheduled_sessions": int(len(group)),
                "six_current_session_bars_available": int(
                    current_six_available.loc[group.index].sum()
                ),
                "complete_with_valid_prior_session_close": int(
                    complete.loc[group.index].sum()
                ),
                "incomplete": int((~complete.loc[group.index]).sum()),
            }
        )
    return {
        "canonical_proxy_available": True,
        "proxy_identifier": "VTI",
        "source": "EODHD processed local five-minute stock bars",
        "source_path": str(VTI_PATH),
        "frequency": "5m",
        "timestamp_semantics": "UTC bar-start timestamp",
        "finality_semantics": (
            "bar is causal only after its five-minute interval completes"
        ),
        "trading_calendar": "NYSE regular-session calendar",
        "adjustment_convention": "raw/unadjusted EODHD OHLC",
        "imputation": False,
        "partial_bars_allowed": False,
        "new_external_dataset_acquired": False,
        "alternative_proxies_tested": False,
        "bounded_filter": (
            "timestamp >= 2024-01-01T00:00:00Z and "
            "timestamp < 2026-01-01T00:00:00Z"
        ),
        "bounded_opened_row_count": int(len(vti_raw)),
        "bounded_min_timestamp": raw_timestamps.min(),
        "bounded_max_timestamp": raw_timestamps.max(),
        "bounded_arrow_sha256": bounded_hash,
        "bounded_rows_with_missing_ohlc": int(
            vti_raw[["open", "high", "low", "close"]].isna().any(axis=1).sum()
        ),
        "regular_session_bar_count": int(len(vti_bars)),
        "checkpoint_6_availability": by_period,
        "existing_causal_alignment": dict(alignment),
        "previous_zero_difference_audit_reused": True,
        "protected_2026_rows_opened": False,
    }


def _population_accounting(
    checkpoints: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for period in ("development", "assessment", "stress"):
        checkpoint_rows = checkpoints.loc[
            checkpoints["partition"].eq(period)
            & pd.to_numeric(checkpoints["checkpoint"], errors="coerce").eq(6)
        ].copy()
        probabilities = pd.to_numeric(
            checkpoint_rows["M1C_probability"],
            errors="coerce",
        )
        valid_probability = np.isfinite(probabilities.to_numpy(float))
        high = valid_probability & probabilities.ge(
            M1C_HIGH_MOVEMENT_THRESHOLD_V1
        )
        episode_rows = panel.loc[panel["partition"].eq(period)].copy()
        stages = [
            ("all_checkpoint_6_rows", len(checkpoint_rows)),
            ("valid_checkpoint_6_m1c_probability", int(valid_probability.sum())),
            ("high_m1c_checkpoint_6_rows", int(high.sum())),
            ("canonical_fresh_first_entry_rows", len(episode_rows)),
            (
                "complete_market_transition_state",
                int(
                    episode_rows[
                        "opening_transition_state_complete_v1"
                    ].astype(bool).sum()
                ),
            ),
            (
                "complete_stock_opening_response",
                int(
                    episode_rows[
                        "stock_opening_response_complete_v1"
                    ].astype(bool).sum()
                ),
            ),
            (
                "complete_15m_outcome",
                int(
                    episode_rows["primary_outcome_complete_v1"].astype(bool).sum()
                ),
            ),
            (
                "final_eligible_rows",
                int(episode_rows["final_eligible_v1"].astype(bool).sum()),
            ),
        ]
        rows.extend(
            {
                "period": period,
                "stage": stage,
                "row_count": int(count),
            }
            for stage, count in stages
        )
        for raw in episode_rows.itertuples(index=False):
            row = cast(Any, raw)
            reasons: list[str] = []
            if not bool(row.m1c_high_tail_v1):
                reasons.append("not_high_m1c")
            if not bool(row.opening_transition_state_complete_v1):
                reasons.extend(
                    str(value)
                    for value in row.opening_transition_state_missing_reasons_v1
                )
            if not bool(row.stock_opening_response_complete_v1):
                reasons.extend(
                    str(value)
                    for value in row.stock_opening_response_missing_reasons_v1
                )
            if not bool(row.primary_outcome_complete_v1):
                reasons.append("incomplete_15m_outcome")
            details.append(
                {
                    "period": period,
                    "stock": str(row.stock),
                    "session": str(row.session),
                    "checkpoint": int(row.checkpoint),
                    "fresh_episode_id": str(row.episode_id),
                    "final_eligible_v1": bool(row.final_eligible_v1),
                    "exclusion_reasons_v1": (
                        "included"
                        if not reasons
                        else ";".join(dict.fromkeys(reasons))
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(details)


def _maximum_share(values: pd.Series) -> float:
    if len(values) == 0:
        return math.nan
    counts = values.fillna("<NA>").astype(str).value_counts()
    return float(counts.max() / counts.sum()) if len(counts) else math.nan


def _distribution(values: pd.Series) -> str:
    counts = values.fillna("<NA>").astype(str).value_counts(sort=False).sort_index()
    total = int(counts.sum())
    return json.dumps(
        {
            str(key): {
                "count": int(count),
                "rate": float(count / total) if total else math.nan,
            }
            for key, count in counts.items()
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _positive_group_rate(
    frame: pd.DataFrame,
    *,
    group_column: str,
    return_column: str,
) -> float:
    if len(frame) == 0:
        return math.nan
    grouped = frame.groupby(group_column, sort=False)[return_column].mean()
    return float(grouped.gt(0.0).mean()) if len(grouped) else math.nan


def _outcome_metrics(
    frame: pd.DataFrame,
    *,
    period: str,
    population: str,
) -> dict[str, Any]:
    count = len(frame)
    signed = pd.to_numeric(frame["future_signed_return_15m"], errors="coerce")
    absolute = pd.to_numeric(
        frame["future_absolute_movement_15m_v1"],
        errors="coerce",
    )
    residual = pd.to_numeric(
        frame["future_iv_residual_15m_v1"],
        errors="coerce",
    )
    state = frame["primary_outcome_state_v1"].astype(str)
    up = int(state.eq("MATERIAL_UP").sum())
    down = int(state.eq("MATERIAL_DOWN").sum())
    no_move = int(state.eq("NO_MATERIAL_MOVE").sum())
    material = up + down
    event_values = frame["opening_transition_event_id_v1"].dropna().astype(str)
    return {
        "period": period,
        "population": population,
        "episode_count": int(count),
        "unique_session_count": int(frame["session"].nunique()),
        "unique_transition_event_count": int(event_values.nunique()),
        "stock_count": int(frame["stock"].nunique()),
        "material_up_count": up,
        "material_up_rate": float(up / count) if count else math.nan,
        "material_down_count": down,
        "material_down_rate": float(down / count) if count else math.nan,
        "no_material_move_count": no_move,
        "no_material_move_rate": float(no_move / count) if count else math.nan,
        "material_move_count": material,
        "material_move_rate": float(material / count) if count else math.nan,
        "mean_signed_15m_return": (
            float(signed.mean()) if signed.notna().any() else math.nan
        ),
        "median_signed_15m_return": (
            float(signed.median()) if signed.notna().any() else math.nan
        ),
        "mean_absolute_15m_movement": (
            float(absolute.mean()) if absolute.notna().any() else math.nan
        ),
        "median_absolute_15m_movement": (
            float(absolute.median()) if absolute.notna().any() else math.nan
        ),
        "mean_iv_residual": (
            float(residual.mean()) if residual.notna().any() else math.nan
        ),
        "median_iv_residual": (
            float(residual.median()) if residual.notna().any() else math.nan
        ),
        "exceed_iv_rate": (
            float(frame["future_exceed_iv_15m_v1"].astype(float).mean())
            if count
            else math.nan
        ),
        "mean_post_entry_local_range_share": (
            float(
                pd.to_numeric(
                    frame["post_share_of_local_range_v1"],
                    errors="coerce",
                ).mean()
            )
            if count
            else math.nan
        ),
        "positive_session_rate": _positive_group_rate(
            frame.assign(_return=signed),
            group_column="session",
            return_column="_return",
        ),
        "positive_month_rate": _positive_group_rate(
            frame.assign(_return=signed),
            group_column="month_v1",
            return_column="_return",
        ),
        "maximum_stock_share": _maximum_share(frame["stock"]),
        "maximum_month_share": _maximum_share(frame["month_v1"]),
        "maximum_session_share": _maximum_share(frame["session"]),
        "maximum_transition_event_share": (
            _maximum_share(event_values) if len(event_values) else math.nan
        ),
        "tail_phase_composition_json": _distribution(
            frame["m1c_tail_phase_v1"]
        ),
        "stock_distribution_json": _distribution(frame["stock"]),
        "month_distribution_json": _distribution(frame["month_v1"]),
        "session_distribution_json": _distribution(frame["session"]),
        "transition_event_distribution_json": (
            _distribution(event_values) if len(event_values) else "{}"
        ),
    }


def _outcome_populations(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    severe = frame.loc[
        frame["opening_market_transition_state_v1"].isin(SEVERE_STATES)
    ]
    populations: list[tuple[str, pd.DataFrame]] = [
        ("all_checkpoint_6_fresh_first_entry", frame),
        ("all_severe_opening_transitions", severe),
        (
            "normal_opening",
            frame.loc[
                frame["opening_market_transition_state_v1"].eq("NORMAL_OPENING")
            ],
        ),
        (
            "negative_severe_opening_transition",
            frame.loc[
                frame["opening_market_transition_state_v1"].eq(
                    "NEGATIVE_SEVERE_OPENING_TRANSITION"
                )
            ],
        ),
        (
            "positive_severe_opening_transition",
            frame.loc[
                frame["opening_market_transition_state_v1"].eq(
                    "POSITIVE_SEVERE_OPENING_TRANSITION"
                )
            ],
        ),
        (
            "amplifying",
            severe.loc[
                severe["stock_opening_response_class_v1"].eq("AMPLIFYING")
            ],
        ),
        (
            "resisting",
            severe.loc[
                severe["stock_opening_response_class_v1"].eq("RESISTING")
            ],
        ),
    ]
    for state in SEVERE_STATES:
        state_rows = severe.loc[
            severe["opening_market_transition_state_v1"].eq(state)
        ]
        for response_class in ("AMPLIFYING", "RESISTING", "NEUTRAL_EXACT"):
            populations.append(
                (
                    f"{state.lower()}__{response_class.lower()}",
                    state_rows.loc[
                        state_rows["stock_opening_response_class_v1"].eq(
                            response_class
                        )
                    ],
                )
            )
    return populations


def _required_outcome_table(
    frame: pd.DataFrame,
    *,
    period: str,
) -> pd.DataFrame:
    source = frame.loc[
        frame["partition"].eq(period) & frame["final_eligible_v1"].astype(bool)
    ].copy()
    return pd.DataFrame(
        [
            _outcome_metrics(group, period=period, population=label)
            for label, group in _outcome_populations(source)
        ]
    )


def _gap_alignment_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        for alignment in (
            "ALIGNED_POSITIVE",
            "ALIGNED_NEGATIVE",
            "GAP_UP_OPENING_DOWN",
            "GAP_DOWN_OPENING_UP",
            "ZERO_OR_NEUTRAL",
            "UNKNOWN_INCOMPLETE",
        ):
            group = source.loc[
                source["market_gap_open_alignment_v1"].eq(alignment)
            ]
            records.append(
                _outcome_metrics(
                    group,
                    period=period,
                    population=f"gap_open_alignment:{alignment}",
                )
            )
    return pd.DataFrame(records)


def _unique_transition_events(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    severe = panel.loc[
        panel["final_eligible_v1"].astype(bool)
        & panel["opening_market_transition_state_v1"].isin(SEVERE_STATES)
    ].copy()
    event_rows: list[dict[str, Any]] = []
    for event_id, group in severe.groupby(
        "opening_transition_event_id_v1",
        sort=True,
    ):
        event_rows.append(
            {
                "opening_transition_event_id_v1": str(event_id),
                "period": str(group["partition"].iloc[0]),
                "session": str(group["session"].iloc[0]),
                "checkpoint": 6,
                "market_proxy_v1": "VTI",
                "opening_transition_sign_v1": int(
                    group["opening_transition_sign_v1"].iloc[0]
                ),
                "opening_market_transition_state_v1": str(
                    group["opening_market_transition_state_v1"].iloc[0]
                ),
                "stock_episode_count": int(len(group)),
                "stock_count": int(group["stock"].nunique()),
                "stocks_json": json.dumps(
                    sorted(group["stock"].astype(str).unique().tolist()),
                    separators=(",", ":"),
                ),
                "amplifying_count": int(
                    group["stock_opening_response_class_v1"]
                    .eq("AMPLIFYING")
                    .sum()
                ),
                "resisting_count": int(
                    group["stock_opening_response_class_v1"]
                    .eq("RESISTING")
                    .sum()
                ),
                "material_up_count": int(
                    group["primary_outcome_state_v1"].eq("MATERIAL_UP").sum()
                ),
                "material_down_count": int(
                    group["primary_outcome_state_v1"].eq("MATERIAL_DOWN").sum()
                ),
                "no_material_move_count": int(
                    group["primary_outcome_state_v1"]
                    .eq("NO_MATERIAL_MOVE")
                    .sum()
                ),
            }
        )
    return pd.DataFrame(event_rows)


def _event_accounting(
    panel: pd.DataFrame,
    market_states: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("development", "assessment", "stress"):
        stock_rows = panel.loc[
            panel["partition"].eq(period)
            & panel["final_eligible_v1"].astype(bool)
        ]
        severe = stock_rows.loc[
            stock_rows["opening_market_transition_state_v1"].isin(SEVERE_STATES)
        ]
        markets = market_states.loc[market_states["partition"].eq(period)]
        records.append(
            {
                "period": period,
                "stock_episode_count": int(len(stock_rows)),
                "severe_stock_episode_count": int(len(severe)),
                "unique_session_count": int(severe["session"].nunique()),
                "unique_opening_transition_event_count": int(
                    severe["opening_transition_event_id_v1"].nunique()
                ),
                "negative_transition_event_count": int(
                    markets["opening_market_transition_state_v1"]
                    .eq("NEGATIVE_SEVERE_OPENING_TRANSITION")
                    .sum()
                ),
                "positive_transition_event_count": int(
                    markets["opening_market_transition_state_v1"]
                    .eq("POSITIVE_SEVERE_OPENING_TRANSITION")
                    .sum()
                ),
                "complete_normal_opening_event_count": int(
                    markets["opening_market_transition_state_v1"]
                    .eq("NORMAL_OPENING")
                    .sum()
                ),
                "incomplete_event_count": int(
                    markets["opening_market_transition_state_v1"]
                    .eq("UNKNOWN_INCOMPLETE")
                    .sum()
                ),
                "mean_stocks_per_transition_event": (
                    float(
                        severe.groupby(
                            "opening_transition_event_id_v1",
                            sort=False,
                        ).size().mean()
                    )
                    if len(severe)
                    else math.nan
                ),
                "maximum_stocks_per_transition_event": (
                    int(
                        severe.groupby(
                            "opening_transition_event_id_v1",
                            sort=False,
                        ).size().max()
                    )
                    if len(severe)
                    else 0
                ),
            }
        )
    return pd.DataFrame(records)


def _sign_prediction(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[numeric.gt(0.0)] = 1.0
    result.loc[numeric.lt(0.0)] = -1.0
    return result


def _action_prediction(actions: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=actions.index, dtype=float)
    result.loc[actions.astype(str).eq("CALL")] = 1.0
    result.loc[actions.astype(str).eq("PUT")] = -1.0
    return result


def _policy_metrics(
    eligible: pd.DataFrame,
    prediction: pd.Series,
    *,
    period: str,
    mechanism: str,
    policy: str,
    evaluation_scope: str,
) -> dict[str, Any]:
    predicted = pd.to_numeric(
        prediction.reindex(eligible.index),
        errors="coerce",
    )
    acted_mask = predicted.isin([-1.0, 1.0])
    acted = eligible.loc[acted_mask].copy()
    predicted = predicted.loc[acted_mask]
    signed = pd.to_numeric(
        acted["future_signed_return_15m"],
        errors="coerce",
    )
    aligned = predicted * signed
    state = acted["primary_outcome_state_v1"].astype(str)
    actual = pd.Series(np.nan, index=acted.index, dtype=float)
    actual.loc[state.eq("MATERIAL_UP")] = 1.0
    actual.loc[state.eq("MATERIAL_DOWN")] = -1.0
    material = actual.notna()
    correct = actual.loc[material].eq(predicted.loc[material])
    material_count = int(material.sum())
    acted_count = len(acted)
    session_aligned = pd.DataFrame(
        {"session": acted["session"], "aligned": aligned}
    )
    month_aligned = pd.DataFrame(
        {"month_v1": acted["month_v1"], "aligned": aligned}
    )
    return {
        "period": period,
        "mechanism": mechanism,
        "policy": policy,
        "evaluation_scope": evaluation_scope,
        "eligible_episode_count": int(len(eligible)),
        "acted_episode_count": int(acted_count),
        "abstention_count": int(len(eligible) - acted_count),
        "unique_session_count": int(acted["session"].nunique()),
        "unique_transition_event_count": int(
            acted["opening_transition_event_id_v1"].nunique()
        ),
        "stock_count": int(acted["stock"].nunique()),
        "call_count": int(predicted.eq(1.0).sum()),
        "put_count": int(predicted.eq(-1.0).sum()),
        "material_up_count": int(state.eq("MATERIAL_UP").sum()),
        "material_down_count": int(state.eq("MATERIAL_DOWN").sum()),
        "no_material_move_count": int(state.eq("NO_MATERIAL_MOVE").sum()),
        "material_direction_accuracy": (
            float(correct.mean()) if material_count else math.nan
        ),
        "accuracy_counting_no_move_as_failure": (
            float(correct.sum() / acted_count) if acted_count else math.nan
        ),
        "material_prediction_following_rate": (
            float(correct.mean()) if material_count else math.nan
        ),
        "material_prediction_opposing_rate": (
            float((~correct).mean()) if material_count else math.nan
        ),
        "material_following_minus_opposing_rate": (
            float(correct.mean() - (~correct).mean())
            if material_count
            else math.nan
        ),
        "mean_aligned_return": (
            float(aligned.mean()) if aligned.notna().any() else math.nan
        ),
        "median_aligned_return": (
            float(aligned.median()) if aligned.notna().any() else math.nan
        ),
        "mean_raw_signed_return": (
            float(signed.mean()) if signed.notna().any() else math.nan
        ),
        "positive_session_rate": (
            float(session_aligned.groupby("session")["aligned"].mean().gt(0.0).mean())
            if acted_count
            else math.nan
        ),
        "positive_month_rate": (
            float(
                month_aligned.groupby("month_v1")["aligned"].mean().gt(0.0).mean()
            )
            if acted_count
            else math.nan
        ),
        "maximum_stock_share": _maximum_share(acted["stock"]),
        "maximum_month_share": _maximum_share(acted["month_v1"]),
        "maximum_session_share": _maximum_share(acted["session"]),
        "maximum_transition_event_share": (
            _maximum_share(
                acted["opening_transition_event_id_v1"].dropna().astype(str)
            )
            if acted["opening_transition_event_id_v1"].notna().any()
            else math.nan
        ),
    }


def _mechanism_frames(
    source: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    severe = source.loc[
        source["opening_market_transition_state_v1"].isin(SEVERE_STATES)
    ]
    sign = pd.to_numeric(
        severe["opening_transition_sign_v1"],
        errors="coerce",
    )
    amplifying = severe.loc[
        severe["stock_opening_response_class_v1"].eq("AMPLIFYING")
    ]
    resisting = severe.loc[
        severe["stock_opening_response_class_v1"].eq("RESISTING")
    ]
    return {
        "market_following": (severe, sign),
        "amplification_continuation": (
            amplifying,
            pd.to_numeric(
                amplifying["opening_transition_sign_v1"],
                errors="coerce",
            ),
        ),
        "resistance_reversal": (
            resisting,
            -pd.to_numeric(
                resisting["opening_transition_sign_v1"],
                errors="coerce",
            ),
        ),
    }


def _mechanism_results(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: dict[str, list[dict[str, Any]]] = {
        "market_following": [],
        "amplification_continuation": [],
        "resistance_reversal": [],
    }
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        for mechanism, (acted, prediction) in _mechanism_frames(source).items():
            records[mechanism].append(
                _policy_metrics(
                    acted,
                    prediction,
                    period=period,
                    mechanism=mechanism,
                    policy=mechanism,
                    evaluation_scope="fixed_acted_population",
                )
            )
    return (
        pd.DataFrame(records["market_following"]),
        pd.DataFrame(records["amplification_continuation"]),
        pd.DataFrame(records["resistance_reversal"]),
    )


def _baseline_tables(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        severe = source.loc[
            source["opening_market_transition_state_v1"].isin(SEVERE_STATES)
        ]
        scopes = {
            "all_severe_opening_episodes": severe,
            "amplification_acted_episodes": severe.loc[
                severe["stock_opening_response_class_v1"].eq("AMPLIFYING")
            ],
            "resistance_acted_episodes": severe.loc[
                severe["stock_opening_response_class_v1"].eq("RESISTING")
            ],
        }
        for scope, eligible in scopes.items():
            transition_sign = pd.to_numeric(
                eligible["opening_transition_sign_v1"],
                errors="coerce",
            )
            policies = {
                "follow_vti_opening_transition": transition_sign,
                "oppose_vti_opening_transition": -transition_sign,
                "recent_stock_5m_momentum": _sign_prediction(
                    eligible["recent_stock_return_5m_v1"]
                ),
                "stock_opening_window_momentum": _sign_prediction(
                    eligible["stock_opening_return_v1"]
                ),
                "frozen_A1": _action_prediction(eligible["A1_action_v1"]),
                "existing_clean_market_direction_baseline": _sign_prediction(
                    eligible["pre_entry_broad_market_signed_return_10m_v1"]
                ),
                "always_CALL": pd.Series(1.0, index=eligible.index),
                "always_PUT": pd.Series(-1.0, index=eligible.index),
            }
            for policy, prediction in policies.items():
                records.append(
                    _policy_metrics(
                        eligible,
                        prediction,
                        period=period,
                        mechanism="baseline",
                        policy=policy,
                        evaluation_scope=scope,
                    )
                )
            blocked = _policy_metrics(
                eligible.iloc[:0],
                pd.Series(dtype=float),
                period=period,
                mechanism="baseline",
                policy="frozen_D2",
                evaluation_scope=scope,
            )
            blocked["status"] = "blocked_contaminated_or_unreproducible_lineage"
            records.append(blocked)
    output = pd.DataFrame(records)
    if "status" not in output:
        output["status"] = "evaluated"
    output["status"] = output["status"].fillna("evaluated")
    return output


def _safe_auc(target: pd.Series, score: pd.Series) -> float:
    valid = target.notna() & score.notna()
    target_values = pd.to_numeric(target.loc[valid], errors="coerce")
    score_values = pd.to_numeric(score.loc[valid], errors="coerce")
    finite = np.isfinite(target_values.to_numpy(float)) & np.isfinite(
        score_values.to_numpy(float)
    )
    target_values = target_values.loc[finite]
    score_values = score_values.loc[finite]
    if len(target_values) < 2 or target_values.nunique() < 2:
        return math.nan
    return float(roc_auc_score(target_values, score_values))


def _safe_spearman(
    left: pd.Series,
    right: pd.Series,
) -> tuple[float, float]:
    valid = left.notna() & right.notna()
    left_values = pd.to_numeric(left.loc[valid], errors="coerce")
    right_values = pd.to_numeric(right.loc[valid], errors="coerce")
    finite = np.isfinite(left_values.to_numpy(float)) & np.isfinite(
        right_values.to_numpy(float)
    )
    left_values = left_values.loc[finite]
    right_values = right_values.loc[finite]
    if len(left_values) < 3 or left_values.nunique() < 2 or right_values.nunique() < 2:
        return math.nan, math.nan
    result = spearmanr(left_values, right_values)
    return float(result.statistic), float(result.pvalue)


def _ranking_tables(
    frame: pd.DataFrame,
    frozen: FrozenOpeningResponseQuintilesV1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_records: list[dict[str, Any]] = []
    quintile_records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        severe = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
            & frame["opening_market_transition_state_v1"].isin(SEVERE_STATES)
        ].copy()
        material = severe.loc[
            severe["primary_outcome_state_v1"].isin(
                ["MATERIAL_UP", "MATERIAL_DOWN"]
            )
        ]
        auc = _safe_auc(
            material["followed_opening_transition_v1"],
            material["stock_relative_opening_response_v1"],
        )
        spearman, spearman_p = _safe_spearman(
            severe["stock_relative_opening_response_v1"],
            severe["market_follow_aligned_return_v1"],
        )
        ranking_records.append(
            {
                "period": period,
                "material_episode_count": int(len(material)),
                "session_count": int(material["session"].nunique()),
                "transition_event_count": int(
                    material["opening_transition_event_id_v1"].nunique()
                ),
                "roc_auc_followed_opening_transition_v1": auc,
                "spearman_stock_relative_response_vs_market_follow_return": (
                    spearman
                ),
                "spearman_p_value_descriptive": spearman_p,
                "probability_model_fitted": False,
                "score_calibrated": False,
            }
        )
        for quintile in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            group = severe.loc[severe["opening_response_quintile_v1"].eq(quintile)]
            group_material = group.loc[
                group["followed_opening_transition_v1"].notna()
            ]
            quintile_records.append(
                {
                    "period": period,
                    "response_quintile_v1": quintile,
                    "episode_count": int(len(group)),
                    "material_episode_count": int(len(group_material)),
                    "followed_transition_count": int(
                        pd.to_numeric(
                            group_material[
                                "followed_opening_transition_v1"
                            ],
                            errors="coerce",
                        )
                        .eq(1)
                        .sum()
                    ),
                    "followed_transition_rate_among_material": (
                        float(
                            pd.to_numeric(
                                group_material[
                                    "followed_opening_transition_v1"
                                ],
                                errors="coerce",
                            ).mean()
                        )
                        if len(group_material)
                        else math.nan
                    ),
                    "mean_market_follow_aligned_return": (
                        float(group["market_follow_aligned_return_v1"].mean())
                        if len(group)
                        else math.nan
                    ),
                    "no_material_move_rate": (
                        float(
                            group["primary_outcome_state_v1"]
                            .eq("NO_MATERIAL_MOVE")
                            .mean()
                        )
                        if len(group)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(ranking_records), pd.DataFrame(quintile_records)


def _regime_policy_metrics(
    frame: pd.DataFrame,
    *,
    period: str,
    regime: str,
) -> dict[str, Any]:
    if regime == "severe_opening":
        group = frame.loc[
            frame["opening_market_transition_state_v1"].isin(SEVERE_STATES)
        ]
        prediction = pd.to_numeric(
            group["opening_transition_sign_v1"],
            errors="coerce",
        )
    elif regime == "normal_opening":
        group = frame.loc[
            frame["opening_market_transition_state_v1"].eq("NORMAL_OPENING")
        ]
        prediction = _sign_prediction(group["market_opening_return_v1"])
    else:
        raise ValueError(f"unknown regime: {regime}")
    metrics = _policy_metrics(
        group,
        prediction,
        period=period,
        mechanism="severe_vs_normal",
        policy="market_direction",
        evaluation_scope=regime,
    )
    acted = group.loc[prediction.isin([-1.0, 1.0])]
    aligned = prediction.loc[acted.index] * pd.to_numeric(
        acted["future_signed_return_15m"],
        errors="coerce",
    )
    metrics.update(
        {
            "regime": regime,
            "mean_market_aligned_return": (
                float(aligned.mean()) if len(aligned) else math.nan
            ),
            "material_following_rate": metrics[
                "material_prediction_following_rate"
            ],
            "material_opposing_rate": metrics[
                "material_prediction_opposing_rate"
            ],
            "no_move_rate": (
                float(
                    acted["primary_outcome_state_v1"]
                    .eq("NO_MATERIAL_MOVE")
                    .mean()
                )
                if len(acted)
                else math.nan
            ),
            "iv_excess_rate": (
                float(acted["future_exceed_iv_15m_v1"].astype(float).mean())
                if len(acted)
                else math.nan
            ),
            "mean_absolute_movement": (
                float(acted["future_absolute_movement_15m_v1"].mean())
                if len(acted)
                else math.nan
            ),
        }
    )
    return metrics


def _severe_vs_normal(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        severe = _regime_policy_metrics(
            source,
            period=period,
            regime="severe_opening",
        )
        normal = _regime_policy_metrics(
            source,
            period=period,
            regime="normal_opening",
        )
        records.extend([severe, normal])
        records.append(
            {
                "period": period,
                "mechanism": "severe_vs_normal",
                "policy": "market_direction",
                "evaluation_scope": "severe_minus_normal",
                "regime": "severe_minus_normal",
                "acted_episode_count": int(
                    severe["acted_episode_count"] + normal["acted_episode_count"]
                ),
                "material_direction_accuracy": (
                    float(severe["material_direction_accuracy"])
                    - float(normal["material_direction_accuracy"])
                ),
                "accuracy_counting_no_move_as_failure": (
                    float(severe["accuracy_counting_no_move_as_failure"])
                    - float(normal["accuracy_counting_no_move_as_failure"])
                ),
                "mean_market_aligned_return": (
                    float(severe["mean_market_aligned_return"])
                    - float(normal["mean_market_aligned_return"])
                ),
                "material_following_rate": (
                    float(severe["material_following_rate"])
                    - float(normal["material_following_rate"])
                ),
                "material_opposing_rate": (
                    float(severe["material_opposing_rate"])
                    - float(normal["material_opposing_rate"])
                ),
                "no_move_rate": (
                    float(severe["no_move_rate"]) - float(normal["no_move_rate"])
                ),
                "iv_excess_rate": (
                    float(severe["iv_excess_rate"])
                    - float(normal["iv_excess_rate"])
                ),
                "mean_absolute_movement": (
                    float(severe["mean_absolute_movement"])
                    - float(normal["mean_absolute_movement"])
                ),
            }
        )
    return pd.DataFrame(records)


def _bootstrap_statistic(sample: pd.DataFrame) -> dict[str, float]:
    states = sample["primary_outcome_state_v1"].astype(str).to_numpy()
    returns = pd.to_numeric(
        sample["future_signed_return_15m"],
        errors="coerce",
    ).to_numpy(float)
    signs = pd.to_numeric(
        sample["opening_transition_sign_v1"],
        errors="coerce",
    ).to_numpy(float)
    responses = sample["stock_opening_response_class_v1"].astype(str).to_numpy()
    market_returns = pd.to_numeric(
        sample["market_opening_return_v1"],
        errors="coerce",
    ).to_numpy(float)
    severe = np.isin(
        sample["opening_market_transition_state_v1"].astype(str).to_numpy(),
        SEVERE_STATES,
    )
    actual = np.full(len(sample), np.nan, dtype=float)
    actual[states == "MATERIAL_UP"] = 1.0
    actual[states == "MATERIAL_DOWN"] = -1.0

    def policy(mask: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
        acted = mask & np.isfinite(prediction) & np.isfinite(returns)
        if not acted.any():
            return math.nan, math.nan, math.nan
        aligned = prediction[acted] * returns[acted]
        material = acted & np.isfinite(actual)
        correct = actual[material] == prediction[material]
        return (
            float(np.mean(aligned)),
            float(np.mean(correct)) if material.any() else math.nan,
            float(np.sum(correct) / np.sum(acted)),
        )

    output: dict[str, float] = {}
    mechanism_inputs = {
        "market_follow": (severe, signs),
        "amplification": (severe & (responses == "AMPLIFYING"), signs),
        "resistance": (severe & (responses == "RESISTING"), -signs),
    }
    for prefix, (mask, prediction) in mechanism_inputs.items():
        mean_return, material_accuracy, all_accuracy = policy(mask, prediction)
        output[f"{prefix}_mean_aligned_return"] = mean_return
        output[f"{prefix}_material_direction_accuracy"] = material_accuracy
        output[f"{prefix}_accuracy_including_no_move"] = all_accuracy

    material_severe = severe & np.isfinite(actual) & np.isfinite(signs)
    followed = (actual[material_severe] == signs[material_severe]).astype(int)
    score = pd.to_numeric(
        sample["stock_relative_opening_response_v1"],
        errors="coerce",
    ).to_numpy(float)[material_severe]
    valid_score = np.isfinite(score)
    output["continuous_ranking_auc"] = (
        float(roc_auc_score(followed[valid_score], score[valid_score]))
        if valid_score.sum() >= 2
        and np.unique(followed[valid_score]).size == 2
        else math.nan
    )
    amp_material = material_severe & (responses == "AMPLIFYING")
    resist_material = material_severe & (responses == "RESISTING")
    output["amplifying_minus_resisting_follow_rate"] = (
        float(np.mean(actual[amp_material] == signs[amp_material]))
        - float(np.mean(actual[resist_material] == signs[resist_material]))
        if amp_material.any() and resist_material.any()
        else math.nan
    )
    normal = (
        sample["opening_market_transition_state_v1"]
        .astype(str)
        .eq("NORMAL_OPENING")
        .to_numpy()
    )
    normal_prediction = np.sign(market_returns)
    severe_policy = policy(severe, signs)
    normal_policy = policy(normal, normal_prediction)
    output["severe_minus_normal_accuracy_including_no_move"] = (
        severe_policy[2] - normal_policy[2]
    )
    output["severe_minus_normal_mean_market_aligned_return"] = (
        severe_policy[0] - normal_policy[0]
    )
    return output


def _cluster_bootstrap(
    source: pd.DataFrame,
    *,
    period: str,
    cluster_column: str,
    cluster_type: str,
    seed: int,
) -> pd.DataFrame:
    clusters = source[cluster_column].dropna().astype(str).unique()
    if len(clusters) == 0:
        return pd.DataFrame()
    groups = {
        key: source.loc[source[cluster_column].astype(str).eq(key)]
        for key in clusters
    }
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat(
            [groups[str(key)] for key in chosen],
            ignore_index=True,
        )
        records.append(
            {
                "period": period,
                "cluster_type": cluster_type,
                "draw": draw,
                "seed": seed,
                "cluster_count": int(len(clusters)),
                **_bootstrap_statistic(sample),
            }
        )
    return pd.DataFrame(records)


def _bootstrap_summary(draws: pd.DataFrame) -> pd.DataFrame:
    identity = {"period", "cluster_type", "draw", "seed", "cluster_count"}
    records: list[dict[str, Any]] = []
    for (period, cluster_type), group in draws.groupby(
        ["period", "cluster_type"],
        sort=False,
    ):
        for statistic in sorted(set(group.columns).difference(identity)):
            values = pd.to_numeric(group[statistic], errors="coerce").dropna()
            records.append(
                {
                    "period": str(period),
                    "cluster_type": str(cluster_type),
                    "statistic": statistic,
                    "valid_draw_count": int(len(values)),
                    "mean": (
                        float(values.mean()) if len(values) else math.nan
                    ),
                    "lower_95": (
                        float(np.quantile(values, 0.025, method="linear"))
                        if len(values)
                        else math.nan
                    ),
                    "upper_95": (
                        float(np.quantile(values, 0.975, method="linear"))
                        if len(values)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _support_result(
    frame: pd.DataFrame,
    *,
    mechanism: str,
) -> dict[str, Any]:
    sign = pd.to_numeric(frame["opening_transition_sign_v1"], errors="coerce")
    if mechanism == "resistance_reversal":
        calls = int(sign.eq(-1).sum())
        puts = int(sign.eq(1).sum())
    else:
        calls = int(sign.eq(1).sum())
        puts = int(sign.eq(-1).sum())
    event_values = frame["opening_transition_event_id_v1"].dropna().astype(str)
    checks = {
        "minimum_30_episodes": len(frame) >= 30,
        "minimum_15_sessions": frame["session"].nunique() >= 15,
        "minimum_15_transition_events": event_values.nunique() >= 15,
        "minimum_8_call_predictions": calls >= 8,
        "minimum_8_put_predictions": puts >= 8,
        "minimum_8_stocks": frame["stock"].nunique() >= 8,
        "maximum_stock_share_25pct": _maximum_share(frame["stock"]) <= 0.25,
        "maximum_session_share_20pct": _maximum_share(frame["session"]) <= 0.20,
        "maximum_event_share_20pct": (
            _maximum_share(event_values) <= 0.20 if len(event_values) else False
        ),
    }
    return {
        "support_pass": all(checks.values()),
        "checks": checks,
        "episode_count": int(len(frame)),
        "session_count": int(frame["session"].nunique()),
        "transition_event_count": int(event_values.nunique()),
        "call_count": calls,
        "put_count": puts,
        "stock_count": int(frame["stock"].nunique()),
        "maximum_stock_share": _maximum_share(frame["stock"]),
        "maximum_session_share": _maximum_share(frame["session"]),
        "maximum_event_share": (
            _maximum_share(event_values) if len(event_values) else math.nan
        ),
    }


def _winsorised_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) == 0:
        return math.nan
    lower, upper = np.quantile(
        numeric.to_numpy(float),
        [0.01, 0.99],
        method="linear",
    )
    return float(numeric.clip(lower=lower, upper=upper).mean())


def _primary_results(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        for mechanism, (acted, prediction) in _mechanism_frames(source).items():
            metrics = _policy_metrics(
                acted,
                prediction,
                period=period,
                mechanism=mechanism,
                policy=mechanism,
                evaluation_scope="fixed_acted_population",
            )
            aligned = prediction * pd.to_numeric(
                acted["future_signed_return_15m"],
                errors="coerce",
            )
            support = _support_result(acted, mechanism=mechanism)
            metrics.update(
                {
                    "raw_mean_aligned_return": metrics["mean_aligned_return"],
                    "winsorised_1pct_mean_aligned_return": _winsorised_mean(
                        aligned
                    ),
                    "support_status": (
                        "pass"
                        if support["support_pass"]
                        else "blocked_insufficient_support"
                    ),
                    "support_checks_json": json.dumps(
                        support["checks"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
            records.append(metrics)
    return pd.DataFrame(records)


def _leave_one_out(
    frame: pd.DataFrame,
    *,
    dependency: str,
    dependency_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        for mechanism, (acted, prediction) in _mechanism_frames(source).items():
            for omitted in sorted(
                acted[dependency_column].dropna().astype(str).unique()
            ):
                keep = ~acted[dependency_column].astype(str).eq(omitted)
                sample = acted.loc[keep]
                sample_prediction = prediction.loc[sample.index]
                metrics = _policy_metrics(
                    sample,
                    sample_prediction,
                    period=period,
                    mechanism=mechanism,
                    policy=mechanism,
                    evaluation_scope=f"leave_one_{dependency}_out",
                )
                records.append(
                    {
                        "period": period,
                        "mechanism": mechanism,
                        "dependency": dependency,
                        "omitted_value": omitted,
                        "remaining_episode_count": int(len(sample)),
                        "remaining_session_count": int(
                            sample["session"].nunique()
                        ),
                        "remaining_transition_event_count": int(
                            sample["opening_transition_event_id_v1"].nunique()
                        ),
                        "mean_aligned_return": metrics["mean_aligned_return"],
                        "material_direction_accuracy": metrics[
                            "material_direction_accuracy"
                        ],
                        "accuracy_counting_no_move_as_failure": metrics[
                            "accuracy_counting_no_move_as_failure"
                        ],
                    }
                )
    return pd.DataFrame(records)


def _concentration_report(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    dimensions = {
        "stock": "stock",
        "month": "month_v1",
        "session": "session",
        "transition_event": "opening_transition_event_id_v1",
    }
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        for mechanism, (acted, _) in _mechanism_frames(source).items():
            for dimension, column in dimensions.items():
                counts = (
                    acted[column]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .rename_axis("value")
                    .reset_index(name="episode_count")
                )
                total = int(counts["episode_count"].sum())
                for rank, row in enumerate(counts.itertuples(index=False), start=1):
                    records.append(
                        {
                            "period": period,
                            "mechanism": mechanism,
                            "dimension": dimension,
                            "rank": rank,
                            "value": str(row.value),
                            "episode_count": int(row.episode_count),
                            "episode_share": (
                                float(row.episode_count / total)
                                if total
                                else math.nan
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _recompute_outcome_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    signed = pd.to_numeric(output["future_signed_return_15m"], errors="coerce")
    sign = pd.to_numeric(output["opening_transition_sign_v1"], errors="coerce")
    output["market_follow_aligned_return_v1"] = sign * signed
    output["amplification_aligned_return_v1"] = sign * signed
    output["resistance_aligned_return_v1"] = -sign * signed
    output["followed_opening_transition_v1"] = pd.Series(
        pd.NA,
        index=output.index,
        dtype="Int64",
    )
    material = output["primary_outcome_state_v1"].isin(
        ["MATERIAL_UP", "MATERIAL_DOWN"]
    ) & sign.notna()
    direction = np.where(
        output.loc[material, "primary_outcome_state_v1"].eq("MATERIAL_UP"),
        1,
        -1,
    )
    output.loc[material, "followed_opening_transition_v1"] = (
        direction == sign.loc[material].to_numpy(float)
    ).astype(int)
    return output


def _primary_null(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcome_columns = [
        "future_signed_return_15m",
        "primary_outcome_state_v1",
        "future_absolute_movement_15m_v1",
        "future_iv_residual_15m_v1",
        "future_exceed_iv_15m_v1",
    ]
    all_draws: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ].copy()
        source = source.reset_index(drop=True)
        candidates: dict[int, np.ndarray] = {}
        for index, row in source.iterrows():
            valid = source.index[
                source["stock"].astype(str).eq(str(row["stock"]))
                & ~source["session"].astype(str).eq(str(row["session"]))
            ].to_numpy(int)
            if len(valid) == 0:
                raise ExperimentBlocked(
                    f"null donor unavailable:{period}:{row['stock']}"
                )
            candidates[index] = valid
        rng = np.random.default_rng(NULL_SEED)
        records: list[dict[str, Any]] = []
        for draw in range(NULL_DRAWS):
            donors = np.asarray(
                [
                    int(rng.choice(candidates[index]))
                    for index in range(len(source))
                ],
                dtype=int,
            )
            if (
                source.loc[donors, "session"].reset_index(drop=True)
                == source["session"].reset_index(drop=True)
            ).any():
                raise ExperimentBlocked("primary null reused the same session")
            sample = source.copy()
            for column in outcome_columns:
                sample[column] = source.loc[
                    donors,
                    column,
                ].reset_index(drop=True)
            sample = _recompute_outcome_derivatives(sample)
            records.append(
                {
                    "period": period,
                    "draw": draw,
                    "seed": NULL_SEED,
                    **_bootstrap_statistic(sample),
                }
            )
        draws = pd.DataFrame(records)
        all_draws.append(draws)
        summaries[period] = {
            "draw_count": int(len(draws)),
            "seed": NULL_SEED,
            "same_session_reassignment_allowed": False,
            "preserved_fields": [
                "stock",
                "checkpoint_6",
                "period",
                "outcome_completeness",
            ],
        }
    return pd.concat(all_draws, ignore_index=True), summaries


def _temporal_placebo(
    frame: pd.DataFrame,
    *,
    period: str,
) -> dict[str, Any]:
    source = frame.loc[
        frame["partition"].eq(period)
        & frame["final_eligible_v1"].astype(bool)
    ].copy()
    source["_order"] = pd.to_datetime(
        source["signal_timestamp"],
        utc=True,
        errors="raise",
    )
    source = source.sort_values(
        ["stock", "_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    outcome_columns = [
        "future_signed_return_15m",
        "primary_outcome_state_v1",
        "future_absolute_movement_15m_v1",
        "future_iv_residual_15m_v1",
        "future_exceed_iv_15m_v1",
    ]
    next_session = source.groupby("stock", sort=False)["session"].shift(-1)
    next_order = source.groupby("stock", sort=False)["_order"].shift(-1)
    valid = next_session.notna() & next_order.notna()
    placebo = source.loc[valid].copy()
    for column in outcome_columns:
        placebo[column] = (
            source.groupby("stock", sort=False)[column]
            .shift(-1)
            .loc[valid]
            .to_numpy()
        )
    if not next_order.loc[valid].gt(source.loc[valid, "_order"]).all():
        raise ExperimentBlocked("temporal placebo did not move forward")
    if next_session.loc[valid].astype(str).ge(PROTECTED_START).any():
        raise ExperimentBlocked("temporal placebo crossed into protected chronology")
    placebo = _recompute_outcome_derivatives(placebo)
    return {
        "period": period,
        "pair_count": int(len(placebo)),
        "source_episode_count": int(len(source)),
        "same_stock": True,
        "same_checkpoint": True,
        "same_period": True,
        "chronology_crossed": False,
        "protected_boundary_crossed": False,
        "statistics": _bootstrap_statistic(placebo),
    }


def _one_sided_null_p(
    values: pd.Series,
    observed: float,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if not math.isfinite(observed) or len(numeric) == 0:
        return math.nan
    return float((1 + numeric.ge(observed).sum()) / (len(numeric) + 1))


def _holm_two(p_values: Mapping[str, float]) -> dict[str, float]:
    valid = {
        key: float(value)
        for key, value in p_values.items()
        if math.isfinite(float(value))
    }
    if len(valid) != 2:
        return {key: math.nan for key in p_values}
    ordered = sorted(valid.items(), key=lambda item: item[1])
    first_key, first_value = ordered[0]
    second_key, second_value = ordered[1]
    first_adjusted = min(1.0, 2.0 * first_value)
    second_adjusted = min(1.0, max(first_adjusted, second_value))
    return {
        first_key: first_adjusted,
        second_key: second_adjusted,
    }


def _transition_sign_stratification(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        for mechanism, (acted, prediction) in _mechanism_frames(source).items():
            for sign, label in (
                (-1, "negative_severe_opening_transition"),
                (1, "positive_severe_opening_transition"),
            ):
                group = acted.loc[
                    pd.to_numeric(
                        acted["opening_transition_sign_v1"],
                        errors="coerce",
                    ).eq(sign)
                ]
                metrics = _policy_metrics(
                    group,
                    prediction.loc[group.index],
                    period=period,
                    mechanism=mechanism,
                    policy=mechanism,
                    evaluation_scope=label,
                )
                metrics["transition_sign"] = sign
                records.append(metrics)
    return pd.DataFrame(records)


def _bootstrap_lookup(
    summary: pd.DataFrame,
    *,
    period: str,
    cluster_type: str,
    statistic: str,
    bound: str = "lower_95",
) -> float:
    selected = summary.loc[
        summary["period"].eq(period)
        & summary["cluster_type"].eq(cluster_type)
        & summary["statistic"].eq(statistic),
        bound,
    ]
    return float(selected.iloc[0]) if len(selected) == 1 else math.nan


def _primary_metric(
    primary: pd.DataFrame,
    *,
    period: str,
    mechanism: str,
    column: str,
) -> float:
    selected = primary.loc[
        primary["period"].eq(period)
        & primary["mechanism"].eq(mechanism),
        column,
    ]
    return float(selected.iloc[0]) if len(selected) == 1 else math.nan


def _loo_positive(
    tables: Sequence[pd.DataFrame],
    *,
    mechanism: str,
) -> bool:
    relevant = pd.concat(
        [
            table.loc[
                table["mechanism"].eq(mechanism)
                & table["period"].isin(["assessment", "stress"])
            ]
            for table in tables
        ],
        ignore_index=True,
    )
    values = pd.to_numeric(
        relevant["mean_aligned_return"],
        errors="coerce",
    )
    return bool(len(values) and values.notna().all() and values.gt(0.0).all())


def _same_sign_positive(
    sign_table: pd.DataFrame,
    *,
    mechanism: str,
) -> bool:
    rows = sign_table.loc[
        sign_table["mechanism"].eq(mechanism)
        & sign_table["period"].isin(["assessment", "stress"])
    ]
    values = pd.to_numeric(rows["mean_aligned_return"], errors="coerce")
    return bool(len(rows) == 4 and values.notna().all() and values.gt(0.0).all())


def _decision_contract(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    null_draws: pd.DataFrame,
    sign_table: pd.DataFrame,
    severe_normal: pd.DataFrame,
    ranking: pd.DataFrame,
    leave_month: pd.DataFrame,
    leave_stock: pd.DataFrame,
    leave_event: pd.DataFrame,
) -> dict[str, Any]:
    support: dict[str, dict[str, Any]] = {}
    for period in ("assessment", "stress"):
        source = frame.loc[
            frame["partition"].eq(period)
            & frame["final_eligible_v1"].astype(bool)
        ]
        for mechanism, (acted, _) in _mechanism_frames(source).items():
            support[f"{period}:{mechanism}"] = _support_result(
                acted,
                mechanism=mechanism,
            )

    observed = {
        mechanism: _primary_metric(
            primary,
            period="assessment",
            mechanism=mechanism,
            column="mean_aligned_return",
        )
        for mechanism in (
            "market_following",
            "amplification_continuation",
            "resistance_reversal",
        )
    }
    assessment_null = null_draws.loc[
        null_draws["period"].eq("assessment")
    ]
    null_columns = {
        "market_following": "market_follow_mean_aligned_return",
        "amplification_continuation": "amplification_mean_aligned_return",
        "resistance_reversal": "resistance_mean_aligned_return",
    }
    null_p = {
        mechanism: _one_sided_null_p(
            assessment_null[column],
            observed[mechanism],
        )
        for mechanism, column in null_columns.items()
    }
    adjusted = _holm_two(
        {
            "amplification_continuation": null_p[
                "amplification_continuation"
            ],
            "resistance_reversal": null_p["resistance_reversal"],
        }
    )
    bootstrap_prefix = {
        "market_following": "market_follow",
        "amplification_continuation": "amplification",
        "resistance_reversal": "resistance",
    }
    checks_by_mechanism: dict[str, dict[str, bool]] = {}
    for mechanism, prefix in bootstrap_prefix.items():
        means_positive = all(
            _primary_metric(
                primary,
                period=period,
                mechanism=mechanism,
                column="mean_aligned_return",
            )
            > 0.0
            for period in ("assessment", "stress")
        )
        accuracy_positive = all(
            _primary_metric(
                primary,
                period=period,
                mechanism=mechanism,
                column="material_direction_accuracy",
            )
            > 0.5
            for period in ("assessment", "stress")
        )
        checks_by_mechanism[mechanism] = {
            "assessment_support_pass": bool(
                support[f"assessment:{mechanism}"]["support_pass"]
            ),
            "stress_support_pass": bool(
                support[f"stress:{mechanism}"]["support_pass"]
            ),
            "mean_aligned_return_positive_both_periods": means_positive,
            "assessment_session_cluster_lower_above_zero": (
                _bootstrap_lookup(
                    bootstrap_summary,
                    period="assessment",
                    cluster_type="session",
                    statistic=f"{prefix}_mean_aligned_return",
                )
                > 0.0
            ),
            "assessment_event_cluster_lower_above_zero": (
                _bootstrap_lookup(
                    bootstrap_summary,
                    period="assessment",
                    cluster_type="opening_transition_event",
                    statistic=f"{prefix}_mean_aligned_return",
                )
                > 0.0
            ),
            "material_direction_accuracy_above_half_both_periods": (
                accuracy_positive
            ),
            "positive_and_negative_transitions_same_mechanism": (
                _same_sign_positive(sign_table, mechanism=mechanism)
            ),
            "leave_one_dependency_robust": _loo_positive(
                [leave_month, leave_stock, leave_event],
                mechanism=mechanism,
            ),
        }

    normal_accuracy = {
        period: float(
            severe_normal.loc[
                severe_normal["period"].eq(period)
                & severe_normal["regime"].eq("normal_opening"),
                "accuracy_counting_no_move_as_failure",
            ].iloc[0]
        )
        for period in ("assessment", "stress")
    }
    broad_checks = checks_by_mechanism["market_following"]
    broad_checks.update(
        {
            "accuracy_including_no_move_beats_normal_both_periods": all(
                _primary_metric(
                    primary,
                    period=period,
                    mechanism="market_following",
                    column="accuracy_counting_no_move_as_failure",
                )
                > normal_accuracy[period]
                for period in ("assessment", "stress")
            ),
            "assessment_null_p_below_0_05": (
                null_p["market_following"] < 0.05
            ),
        }
    )
    all_severe_means = {
        period: _primary_metric(
            primary,
            period=period,
            mechanism="market_following",
            column="mean_aligned_return",
        )
        for period in ("assessment", "stress")
    }
    amplification_checks = checks_by_mechanism["amplification_continuation"]
    amplification_checks.update(
        {
            "selection_beats_following_all_severe_both_periods": all(
                _primary_metric(
                    primary,
                    period=period,
                    mechanism="amplification_continuation",
                    column="mean_aligned_return",
                )
                > all_severe_means[period]
                for period in ("assessment", "stress")
            ),
            "holm_adjusted_assessment_null_p_below_0_05": (
                adjusted["amplification_continuation"] < 0.05
            ),
        }
    )
    resistance_checks = checks_by_mechanism["resistance_reversal"]
    resistance_checks.update(
        {
            "opposing_beats_following_same_resisting_both_periods": all(
                _primary_metric(
                    primary,
                    period=period,
                    mechanism="resistance_reversal",
                    column="mean_aligned_return",
                )
                > -_primary_metric(
                    primary,
                    period=period,
                    mechanism="resistance_reversal",
                    column="mean_aligned_return",
                )
                for period in ("assessment", "stress")
            ),
            "holm_adjusted_assessment_null_p_below_0_05": (
                adjusted["resistance_reversal"] < 0.05
            ),
        }
    )

    def supported(checks: Mapping[str, bool]) -> bool:
        return all(bool(value) for value in checks.values())

    if supported(broad_checks):
        broad_decision = "opening_market_direction_supported_retrospectively"
    elif not (
        broad_checks["assessment_support_pass"]
        and broad_checks["stress_support_pass"]
    ):
        broad_decision = "blocked_insufficient_support"
    elif broad_checks["mean_aligned_return_positive_both_periods"]:
        broad_decision = "opening_market_direction_descriptive_only"
    else:
        broad_decision = "no_incremental_opening_market_directional_signal"

    assessment_auc = float(
        ranking.loc[
            ranking["period"].eq("assessment"),
            "roc_auc_followed_opening_transition_v1",
        ].iloc[0]
    )
    stress_auc = float(
        ranking.loc[
            ranking["period"].eq("stress"),
            "roc_auc_followed_opening_transition_v1",
        ].iloc[0]
    )
    ranking_reproducible = (
        assessment_auc > 0.5
        and stress_auc > 0.5
        and _bootstrap_lookup(
            bootstrap_summary,
            period="assessment",
            cluster_type="session",
            statistic="continuous_ranking_auc",
        )
        > 0.5
        and _bootstrap_lookup(
            bootstrap_summary,
            period="assessment",
            cluster_type="opening_transition_event",
            statistic="continuous_ranking_auc",
        )
        > 0.5
    )
    if supported(amplification_checks):
        amplification_decision = (
            "opening_amplification_continuation_supported_retrospectively"
        )
    elif not (
        amplification_checks["assessment_support_pass"]
        and amplification_checks["stress_support_pass"]
    ):
        amplification_decision = "blocked_insufficient_support"
    elif ranking_reproducible:
        amplification_decision = "opening_amplification_ranking_only"
    elif amplification_checks["mean_aligned_return_positive_both_periods"]:
        amplification_decision = "opening_amplification_descriptive_only"
    else:
        amplification_decision = "no_opening_amplification_signal"

    if supported(resistance_checks):
        resistance_decision = (
            "opening_resistance_reversal_supported_retrospectively"
        )
    elif not (
        resistance_checks["assessment_support_pass"]
        and resistance_checks["stress_support_pass"]
    ):
        resistance_decision = "blocked_insufficient_support"
    elif resistance_checks["mean_aligned_return_positive_both_periods"]:
        resistance_decision = "opening_resistance_descriptive_only"
    else:
        resistance_decision = "no_opening_resistance_signal"

    supported_decisions = {
        "opening_market_direction_supported_retrospectively",
        "opening_amplification_continuation_supported_retrospectively",
        "opening_resistance_reversal_supported_retrospectively",
    }
    mechanism_decisions = {
        broad_decision,
        amplification_decision,
        resistance_decision,
    }
    if mechanism_decisions.intersection(supported_decisions):
        overall = "opening_transition_direction_supported_retrospectively"
    elif mechanism_decisions == {"blocked_insufficient_support"}:
        overall = "blocked_insufficient_support"
    else:
        movement_rows = severe_normal.loc[
            severe_normal["regime"].eq("severe_minus_normal")
        ]
        movement_only = bool(
            pd.to_numeric(
                movement_rows["mean_absolute_movement"],
                errors="coerce",
            )
            .gt(0.0)
            .all()
            and pd.to_numeric(
                movement_rows["iv_excess_rate"],
                errors="coerce",
            )
            .gt(0.0)
            .all()
        )
        descriptive = any(
            "descriptive_only" in value or "ranking_only" in value
            for value in mechanism_decisions
        )
        if descriptive:
            overall = "opening_transition_descriptive_only"
        elif movement_only:
            overall = "opening_transition_movement_only"
        else:
            overall = "no_incremental_opening_transition_directional_signal"
    return {
        "broad_market_following_decision": broad_decision,
        "amplification_continuation_decision": amplification_decision,
        "resistance_reversal_decision": resistance_decision,
        "overall_decision": overall,
        "support": support,
        "contract_checks": checks_by_mechanism,
        "assessment_null_p_values": null_p,
        "holm_adjusted_response_mechanism_p_values": adjusted,
        "continuous_ranking_reproducible": ranking_reproducible,
        "more_conservative_cluster_conclusion_governs": True,
    }


def _threshold_manifest(
    thresholds: OpeningTransitionThresholdsV1,
    quintiles: FrozenOpeningResponseQuintilesV1,
    *,
    configuration_hash: str,
) -> OpeningTransitionThresholdManifestV1:
    if not thresholds.calibration_complete_v1:
        raise ExperimentBlocked("blocked_data_completeness:opening_thresholds")
    if not quintiles.calibration_complete_v1:
        raise ExperimentBlocked("blocked_data_completeness:response_quintiles")
    boundaries = (
        quintiles.q20_v1,
        quintiles.q40_v1,
        quintiles.q60_v1,
        quintiles.q80_v1,
    )
    if any(value is None for value in boundaries):
        raise ExperimentBlocked("blocked_data_completeness:response_quintiles")
    return OpeningTransitionThresholdManifestV1(
        schema_version="m1c-opening-market-transition-thresholds-v1",
        market_proxy_v1="VTI",
        checkpoint_v1=6,
        expected_opening_bar_count_v1=6,
        calibration_period=OpeningCalibrationPeriodV1(
            start="2024-01-01",
            end="2024-12-31",
            predictors_only=True,
            future_stock_outcomes_accessed_for_thresholds=False,
            option_outcomes_accessed_for_thresholds=False,
        ),
        quantiles=OpeningCalibrationQuantilesV1(
            signed_return_lower=0.10,
            signed_return_upper=0.90,
            range=0.75,
            method="numpy_linear",
        ),
        minimum_predictor_support_v1=MINIMUM_PREDICTOR_SUPPORT_V1,
        pooling_fallback_used=False,
        configuration_hash=configuration_hash,
        thresholds=thresholds,
        stock_relative_response_quintile_boundaries_v1=tuple(
            float(value) for value in boundaries if value is not None
        ),
        stock_relative_response_quintile_support_v1=quintiles.support_v1,
    )


def _calibration_table(
    thresholds: OpeningTransitionThresholdsV1,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "checkpoint": 6,
                "calibration_start": DEVELOPMENT_START,
                "calibration_end": DEVELOPMENT_END,
                "predictors_only": True,
                "quantile_method": "numpy_linear",
                **thresholds.model_dump(mode="python"),
            }
        ]
    )


def _missingness_table(
    panel: pd.DataFrame,
    market_states: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("development", "assessment", "stress"):
        markets = market_states.loc[market_states["partition"].eq(period)]
        for reasons in markets.loc[
            ~markets["complete_v1"].astype(bool),
            "missing_reasons_v1",
        ]:
            for reason in tuple(reasons):
                records.append(
                    {
                        "period": period,
                        "category": "market_transition",
                        "reason": str(reason),
                        "episode_count": 1,
                    }
                )
        episodes = panel.loc[panel["partition"].eq(period)]
        for reasons in episodes.loc[
            ~episodes["stock_opening_response_complete_v1"].astype(bool),
            "stock_opening_response_missing_reasons_v1",
        ]:
            for reason in tuple(reasons):
                records.append(
                    {
                        "period": period,
                        "category": "stock_response",
                        "reason": str(reason),
                        "episode_count": 1,
                    }
                )
        incomplete_outcome = int(
            (~episodes["primary_outcome_complete_v1"].astype(bool)).sum()
        )
        if incomplete_outcome:
            records.append(
                {
                    "period": period,
                    "category": "directional_outcome",
                    "reason": "incomplete_15m_outcome",
                    "episode_count": incomplete_outcome,
                }
            )
    if not records:
        return pd.DataFrame(
            columns=["period", "category", "reason", "episode_count"]
        )
    return (
        pd.DataFrame(records)
        .groupby(["period", "category", "reason"], as_index=False, sort=True)[
            "episode_count"
        ]
        .sum()
    )


def _enrich_primary_results(
    primary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    decision: Mapping[str, Any],
) -> pd.DataFrame:
    output = primary.copy()
    prefix = {
        "market_following": "market_follow",
        "amplification_continuation": "amplification",
        "resistance_reversal": "resistance",
    }
    for index, row in output.iterrows():
        mechanism = str(row["mechanism"])
        statistic = f"{prefix[mechanism]}_mean_aligned_return"
        for cluster_type, label in (
            ("session", "session_cluster"),
            ("opening_transition_event", "event_cluster"),
        ):
            output.loc[index, f"{label}_lower_95"] = _bootstrap_lookup(
                bootstrap_summary,
                period=str(row["period"]),
                cluster_type=cluster_type,
                statistic=statistic,
                bound="lower_95",
            )
            output.loc[index, f"{label}_upper_95"] = _bootstrap_lookup(
                bootstrap_summary,
                period=str(row["period"]),
                cluster_type=cluster_type,
                statistic=statistic,
                bound="upper_95",
            )
        p_values = cast(
            Mapping[str, float],
            decision["assessment_null_p_values"],
        )
        adjusted = cast(
            Mapping[str, float],
            decision["holm_adjusted_response_mechanism_p_values"],
        )
        output.loc[index, "assessment_null_p_value"] = (
            float(p_values[mechanism])
            if str(row["period"]) == "assessment"
            else math.nan
        )
        output.loc[index, "holm_adjusted_assessment_null_p_value"] = (
            float(adjusted[mechanism])
            if mechanism in adjusted and str(row["period"]) == "assessment"
            else math.nan
        )
    return output


def _attach_ranking_intervals(
    ranking: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = ranking.copy()
    for index, row in output.iterrows():
        for cluster_type, label in (
            ("session", "session_cluster"),
            ("opening_transition_event", "event_cluster"),
        ):
            output.loc[index, f"{label}_auc_lower_95"] = _bootstrap_lookup(
                bootstrap_summary,
                period=str(row["period"]),
                cluster_type=cluster_type,
                statistic="continuous_ranking_auc",
                bound="lower_95",
            )
            output.loc[index, f"{label}_auc_upper_95"] = _bootstrap_lookup(
                bootstrap_summary,
                period=str(row["period"]),
                cluster_type=cluster_type,
                statistic="continuous_ranking_auc",
                bound="upper_95",
            )
    return output


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    if len(frame) == 0:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(
            _format_number(getattr(row, column))
            for column in columns
        )
        + " |"
        for row in frame.loc[:, list(columns)].itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def _report_markdown(
    *,
    reconciliation_summary: Mapping[str, Any],
    timing: Mapping[str, Any],
    thresholds: OpeningTransitionThresholdsV1,
    population: pd.DataFrame,
    event_accounting: pd.DataFrame,
    assessment: pd.DataFrame,
    stress: pd.DataFrame,
    primary: pd.DataFrame,
    ranking: pd.DataFrame,
    severe_normal: pd.DataFrame,
    baseline: pd.DataFrame,
    decision: Mapping[str, Any],
    null_placebo: Mapping[str, Any],
) -> str:
    population_view = population.loc[
        population["stage"].isin(
            [
                "canonical_fresh_first_entry_rows",
                "complete_market_transition_state",
                "complete_stock_opening_response",
                "complete_15m_outcome",
                "final_eligible_rows",
            ]
        )
    ]
    outcome_view = pd.concat(
        [
            assessment.loc[
                assessment["population"].isin(
                    [
                        "all_severe_opening_transitions",
                        "normal_opening",
                        "amplifying",
                        "resisting",
                    ]
                )
            ],
            stress.loc[
                stress["population"].isin(
                    [
                        "all_severe_opening_transitions",
                        "normal_opening",
                        "amplifying",
                        "resisting",
                    ]
                )
            ],
        ],
        ignore_index=True,
    )
    baseline_view = baseline.loc[
        baseline["evaluation_scope"].isin(
            [
                "amplification_acted_episodes",
                "resistance_acted_episodes",
            ]
        )
        & baseline["policy"].isin(
            [
                "follow_vti_opening_transition",
                "recent_stock_5m_momentum",
                "stock_opening_window_momentum",
                "frozen_A1",
                "existing_clean_market_direction_baseline",
            ]
        )
    ]
    population_table = _markdown_table(
        population_view,
        ["period", "stage", "row_count"],
    )
    event_table = _markdown_table(
        event_accounting,
        [
            "period",
            "severe_stock_episode_count",
            "unique_session_count",
            "unique_opening_transition_event_count",
            "negative_transition_event_count",
            "positive_transition_event_count",
            "complete_normal_opening_event_count",
            "incomplete_event_count",
        ],
    )
    outcome_table = _markdown_table(
        outcome_view,
        [
            "period",
            "population",
            "episode_count",
            "unique_transition_event_count",
            "material_move_rate",
            "no_material_move_rate",
            "exceed_iv_rate",
            "mean_absolute_15m_movement",
            "mean_iv_residual",
        ],
    )
    primary_table = _markdown_table(
        primary,
        [
            "period",
            "mechanism",
            "acted_episode_count",
            "unique_transition_event_count",
            "mean_aligned_return",
            "session_cluster_lower_95",
            "event_cluster_lower_95",
            "material_direction_accuracy",
            "accuracy_counting_no_move_as_failure",
            "support_status",
        ],
    )
    ranking_table = _markdown_table(
        ranking,
        [
            "period",
            "material_episode_count",
            "roc_auc_followed_opening_transition_v1",
            "session_cluster_auc_lower_95",
            "event_cluster_auc_lower_95",
            "spearman_stock_relative_response_vs_market_follow_return",
        ],
    )
    severe_normal_table = _markdown_table(
        severe_normal.loc[
            severe_normal["regime"].isin(
                ["severe_opening", "normal_opening", "severe_minus_normal"]
            )
        ],
        [
            "period",
            "regime",
            "acted_episode_count",
            "material_direction_accuracy",
            "accuracy_counting_no_move_as_failure",
            "mean_market_aligned_return",
            "no_move_rate",
            "iv_excess_rate",
            "mean_absolute_movement",
        ],
    )
    baseline_table = _markdown_table(
        baseline_view,
        [
            "period",
            "evaluation_scope",
            "policy",
            "acted_episode_count",
            "mean_aligned_return",
            "material_direction_accuracy",
            "accuracy_counting_no_move_as_failure",
        ],
    )
    null_p = cast(Mapping[str, Any], decision["assessment_null_p_values"])
    market_null_p = _format_number(null_p["market_following"])
    amplification_null_p = _format_number(
        null_p["amplification_continuation"]
    )
    resistance_null_p = _format_number(null_p["resistance_reversal"])
    holm_json = json.dumps(
        _json_safe(decision["holm_adjusted_response_mechanism_p_values"]),
        sort_keys=True,
    )
    temporal_json = json.dumps(
        _json_safe(null_placebo["temporal_placebo"]),
        sort_keys=True,
    )
    gap_thresholds = (
        f"{thresholds.market_overnight_gap_q10_v1:.12g} / "
        f"{thresholds.market_overnight_gap_q90_v1:.12g}"
    )
    total_thresholds = (
        f"{thresholds.market_total_transition_q10_v1:.12g} / "
        f"{thresholds.market_total_transition_q90_v1:.12g}"
    )
    return f"""# M1C Opening Market Transition V1

## Scope and interpretation

This is a fixed retrospective underlying-direction experiment on previously
opened 2025 periods. It is not untouched confirmation, an option-edge test, or
a tradeability claim. M1C, its threshold, its 15-minute endpoint, the frozen
20-stock cohort, freshness, Tail Phase V1, and A1 were not changed.

## Population-accounting audit

Status: `{reconciliation_summary["status"]}`.

The prior apparent 9-versus-15 assessment and 29-versus-34 stress differences
were different population definitions, not missing outcomes or a construction
bug. Tail diagnostics admitted every high-M1C `FIRST_ENTRY`/`RE_ENTRY`
checkpoint row. The primary signed-shock study admitted only canonical fresh
episodes. All 11 extra `RE_ENTRY` rows occurred 20 minutes after the preceding
fresh episode and failed the frozen 30-minute spacing rule (6 assessment, 5
stress). The prior scientific conclusion does not change. The prior
`fresh_tail_entries` label was terminologically ambiguous; the exact
episode-ID reconciliation is now explicit.

## Exact checkpoint-6 timing

Checkpoint 6 means six complete five-minute bars. The fixed opening window is
09:30-10:00 New York time: ordinals 0 through 5, with the final included bar
starting 09:55 and completing at the 10:00 signal. Frozen M1C entry is the
10:00 next-bar open. Bar 6 (10:00-10:05), partial bars, future bars, and prior
sessions are excluded. Expected bar count: `{timing["expected_opening_bar_count"]}`.

The prior two-window shock experiment excluded checkpoint 6 because W1 needed
a close reference before the regular-session open. This separately versioned
experiment uses only the fixed same-session opening window for its primary
state; the previous regular-session close is used solely to audit the gap and
total-transition identity.

## Canonical VTI and frozen thresholds

The canonical proxy was already available: raw/unadjusted EODHD VTI
five-minute OHLC, UTC bar-start timestamps, final after each five-minute
interval, aligned to the NYSE calendar. No alternative proxy or new dataset was
tested.

- Opening return q10: `{thresholds.market_opening_return_q10_v1:.12g}`
- Opening return q90: `{thresholds.market_opening_return_q90_v1:.12g}`
- Opening range q75: `{thresholds.market_opening_range_q75_v1:.12g}`
- Overnight gap q10/q90 (descriptive): `{gap_thresholds}`
- Total transition q10/q90 (descriptive): `{total_thresholds}`
- Complete 2024 predictor support: `{thresholds.market_opening_return_support_v1}`

The severe state uses only opening return and opening range. Negative is
`return <= q10 and range >= q75`; positive is `return >= q90 and range >=
q75`. Elevated range without either signed tail is nondirectional; other
complete rows are normal. Equality is inclusive.

## Population and structural opening-regime evidence

{population_table}

{event_table}

All primary episode rows are checkpoint-6 `FIRST_ENTRY`. Later `PERSISTENT`
rows and `RE_ENTRY` rows are not independent support.

## Absolute-movement evidence

{outcome_table}

## Directional evidence

{primary_table}

Continuous response ranking:

{ranking_table}

Severe versus normal opening:

{severe_normal_table}

Selected same-timestamp baselines:

{baseline_table}

Frozen D2 is `blocked_contaminated_or_unreproducible_lineage`; it was not
approximately reconstructed. No option outcome was used.

## Nulls, placebo, and shared-event uncertainty

The primary null used `{NULL_DRAWS}` fixed-seed reassignments to a different
session while preserving stock, checkpoint 6, period, and outcome
completeness. The temporal placebo used the next eligible checkpoint-6 fresh
`FIRST_ENTRY` outcome for the same stock and period. Assessment null p-values:
market following `{market_null_p}`, amplification `{amplification_null_p}`,
resistance `{resistance_null_p}`.
Holm-adjusted amplification/resistance p-values are
`{holm_json}`.

Session and whole opening-transition-event bootstraps each used
`{BOOTSTRAP_DRAWS}` replications. At checkpoint 6 each severe event is one
session/sign, so the two cluster units coincide for severe-only mechanisms;
both are nevertheless persisted and the more conservative conclusion governs.
Null/placebo metadata: `{temporal_json}`.

## Decisions

- Broad market following: `{decision["broad_market_following_decision"]}`
- Amplification continuation: `{decision["amplification_continuation_decision"]}`
- Resistance reversal: `{decision["resistance_reversal_decision"]}`
- Overall: `{decision["overall_decision"]}`

These are separate frozen mechanisms. They were not merged and the better arm
was not selected as a policy.

## Tail Phase

The experiment addresses the dominant checkpoint-6 `FIRST_ENTRY` population
excluded by the prior two-window design. Tail Phase V1 is attached unchanged
for provenance. No phase gate, phase interaction, later persistence, or
`RE_ENTRY` support is used.

## Prospective recorder integration

The existing IBKR recorder stores the frozen VTI opening fields after core
checkpoint and episode processing. The integration is logging-only and
failure-contained; it does not change M1C scoring, episode inclusion,
promotion priority, subscriptions, recorder capacity, option selection,
direction decisions, or routing. The first 20 transfer sessions remain
`engineering_transfer` and cannot recalibrate these definitions.

## Option profitability

**Not tested**

Prior-close ATM IV is used only for the unchanged strict M1C movement threshold
and the stock-local response scale.

## Execution realism and remaining unknowns

Five-minute historical bars cannot observe bid withdrawal, ask withdrawal,
replenishment, trade impact, spread changes, queue behaviour, executable
option outcomes, slippage, fill probability, market impact, or prospective
behavioural stability. No broker was accessed, no routing path was enabled,
and no order was placed.

## Operational blockers

Pipeline/data blockers are separated from scientific negative results. The
canonical VTI source, checkpoint-6 bars, stock bars, IV scales, and outcomes
were available for the reported eligible rows. Protected 2026 outcomes were
not opened, calculated, inspected, or displayed.
"""


def _validate_before_write(
    *,
    panel: pd.DataFrame,
    market_states: pd.DataFrame,
    reconciliation: pd.DataFrame,
    primary: pd.DataFrame,
    null_draws: pd.DataFrame,
) -> None:
    assert_unprotected_sessions_v1(panel["session"])
    assert_unprotected_sessions_v1(market_states["session"])
    if panel["session"].astype(str).ge(PROTECTED_START).any():
        raise ExperimentBlocked("protected episode entered final panel")
    if panel.duplicated(IDENTITY).any():
        raise ExperimentBlocked("final episode identities are duplicated")
    if not panel["checkpoint"].eq(6).all():
        raise ExperimentBlocked("non-checkpoint-6 row entered primary panel")
    if not panel["m1c_tail_phase_v1"].eq("FIRST_ENTRY").all():
        raise ExperimentBlocked("non-FIRST_ENTRY row entered primary panel")
    if not panel["m1c_high_tail_v1"].astype(bool).all():
        raise ExperimentBlocked("non-high-M1C row entered primary panel")
    if (
        pd.to_datetime(panel["maximum_market_timestamp_v1"], utc=True)
        > pd.to_datetime(panel["signal_timestamp"], utc=True)
    ).any():
        raise ExperimentBlocked("future market timestamp entered predictors")
    if (
        pd.to_datetime(panel["maximum_stock_timestamp_v1"], utc=True)
        > pd.to_datetime(panel["signal_timestamp"], utc=True)
    ).any():
        raise ExperimentBlocked("future stock timestamp entered predictors")
    if len(reconciliation) != 49:
        raise ExperimentBlocked("prior reconciliation row count drifted")
    if len(null_draws) != 2 * NULL_DRAWS:
        raise ExperimentBlocked("primary null replication count drifted")
    if set(primary["mechanism"]) != {
        "market_following",
        "amplification_continuation",
        "resistance_reversal",
    }:
        raise ExperimentBlocked("primary mechanisms were merged or omitted")


def _write_outputs(
    *,
    frozen_configuration: Mapping[str, Any],
    threshold_manifest: OpeningTransitionThresholdManifestV1,
    calibration: pd.DataFrame,
    reconciliation: pd.DataFrame,
    reconciliation_summary: Mapping[str, Any],
    timing: Mapping[str, Any],
    proxy_audit: Mapping[str, Any],
    population: pd.DataFrame,
    population_details: pd.DataFrame,
    market_states: pd.DataFrame,
    panel: pd.DataFrame,
    events: pd.DataFrame,
    event_accounting: pd.DataFrame,
    assessment: pd.DataFrame,
    stress: pd.DataFrame,
    primary: pd.DataFrame,
    ranking: pd.DataFrame,
    quintile_outcomes: pd.DataFrame,
    sign_table: pd.DataFrame,
    gap_diagnostics: pd.DataFrame,
    severe_normal: pd.DataFrame,
    baseline: pd.DataFrame,
    session_bootstrap: pd.DataFrame,
    event_bootstrap: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    leave_month: pd.DataFrame,
    leave_stock: pd.DataFrame,
    leave_event: pd.DataFrame,
    null_draws: pd.DataFrame,
    null_placebo: Mapping[str, Any],
    concentration: pd.DataFrame,
    missingness: pd.DataFrame,
    decision: Mapping[str, Any],
    summary: Mapping[str, Any],
    report: str,
) -> None:
    _write_json(PRIMARY / "frozen_configuration_v1.json", frozen_configuration)
    _write_json(
        PRIMARY / "frozen_2024_opening_threshold_manifest_v1.json",
        threshold_manifest.model_dump(mode="json"),
    )
    _write_csv(PRIMARY / "predictor_only_calibration_v1.csv", calibration)
    _write_csv(
        PRIMARY / "prior_signed_shock_population_reconciliation_v1.csv",
        reconciliation,
    )
    _write_json(
        PRIMARY / "prior_population_reconciliation_summary_v1.json",
        reconciliation_summary,
    )
    _write_json(PRIMARY / "checkpoint_6_timing_audit_v1.json", timing)
    _write_json(PRIMARY / "canonical_vti_source_audit_v1.json", proxy_audit)
    _write_csv(
        PRIMARY / "checkpoint_6_population_accounting_v1.csv",
        population,
    )
    _write_parquet(
        PRIMARY / "checkpoint_6_population_details_v1.parquet",
        population_details,
    )
    _write_parquet(
        PRIMARY / "opening_market_state_surface_v1.parquet",
        market_states,
    )
    _write_parquet(
        PRIMARY / "episode_opening_transition_v1.parquet",
        panel,
    )
    _write_csv(PRIMARY / "unique_opening_transition_events_v1.csv", events)
    _write_csv(PRIMARY / "event_accounting_v1.csv", event_accounting)
    _write_csv(PRIMARY / "assessment_results_v1.csv", assessment)
    _write_csv(PRIMARY / "stress_results_v1.csv", stress)
    _write_csv(
        PRIMARY / "market_following_results_v1.csv",
        primary.loc[primary["mechanism"].eq("market_following")],
    )
    _write_csv(
        PRIMARY / "amplification_results_v1.csv",
        primary.loc[primary["mechanism"].eq("amplification_continuation")],
    )
    _write_csv(
        PRIMARY / "resistance_results_v1.csv",
        primary.loc[primary["mechanism"].eq("resistance_reversal")],
    )
    _write_csv(PRIMARY / "continuous_ranking_results_v1.csv", ranking)
    _write_csv(
        PRIMARY / "response_quintile_outcomes_v1.csv",
        quintile_outcomes,
    )
    _write_csv(
        PRIMARY / "transition_sign_stratification_v1.csv",
        sign_table,
    )
    _write_csv(
        PRIMARY / "gap_open_alignment_diagnostics_v1.csv",
        gap_diagnostics,
    )
    _write_csv(
        PRIMARY / "severe_vs_normal_comparison_v1.csv",
        severe_normal,
    )
    _write_csv(PRIMARY / "baseline_comparisons_v1.csv", baseline)
    _write_parquet(
        PRIMARY / "session_cluster_bootstrap_v1.parquet",
        session_bootstrap,
    )
    _write_parquet(
        PRIMARY / "opening_transition_event_cluster_bootstrap_v1.parquet",
        event_bootstrap,
    )
    _write_csv(
        PRIMARY / "cluster_bootstrap_summary_v1.csv",
        bootstrap_summary,
    )
    _write_csv(PRIMARY / "leave_one_month_out_v1.csv", leave_month)
    _write_csv(PRIMARY / "leave_one_stock_out_v1.csv", leave_stock)
    _write_csv(
        PRIMARY / "leave_one_transition_event_out_v1.csv",
        leave_event,
    )
    _write_parquet(PRIMARY / "primary_null_draws_v1.parquet", null_draws)
    _write_json(
        PRIMARY / "null_and_temporal_placebo_results_v1.json",
        null_placebo,
    )
    _write_csv(PRIMARY / "concentration_report_v1.csv", concentration)
    _write_csv(PRIMARY / "missingness_reasons_v1.csv", missingness)
    _write_json(
        PRIMARY / "target_partition_audit_v1.json",
        {
            "valid_row_count": int(
                panel["primary_outcome_complete_v1"].astype(bool).sum()
            ),
            "strict_material_move_equivalence": True,
            "positive_threshold_equality_partition": "NO_MATERIAL_MOVE",
            "negative_threshold_equality_partition": "NO_MATERIAL_MOVE",
            "stored_m1c_target_changed": False,
        },
    )
    _write_csv(
        PRIMARY / "tail_phase_provenance_v1.csv",
        (
            panel.groupby(
                ["partition", "m1c_tail_phase_v1"],
                as_index=False,
                sort=True,
            )
            .size()
            .rename(columns={"size": "episode_count"})
        ),
    )
    _write_json(PRIMARY / "decision_contract_results_v1.json", decision)
    _write_json(PRIMARY / "summary_v1.json", summary)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "m1c_opening_market_transition_v1.md").write_text(
        report,
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    input_records = _verify_sources()
    (
        inherited_episodes,
        tail_episodes,
        checkpoints,
        prior_market_states,
        stock_bars,
        vti_raw,
    ) = _load_inputs()
    frozen_regression = _frozen_regression(
        tail_episodes,
        inherited_episodes,
    )
    if "pre_entry_broad_market_signed_return_10m_v1" not in inherited_episodes:
        inherited_episodes = inherited_episodes.merge(
            tail_episodes[
                [
                    *IDENTITY,
                    "pre_entry_broad_market_signed_return_10m_v1",
                ]
            ],
            on=IDENTITY,
            how="left",
            validate="one_to_one",
        )
    reconciliation, reconciliation_summary = _prior_population_reconciliation(
        checkpoints,
        tail_episodes,
        inherited_episodes,
        prior_market_states,
    )

    vti_bars, bounded_vti_hash = _prepare_vti_bars(vti_raw)
    market_predictors = _build_market_predictors(vti_bars)
    thresholds = freeze_opening_thresholds_v1(market_predictors)
    market_states = _apply_market_states(market_predictors, thresholds)
    alignment = _proxy_alignment_audit(vti_bars, stock_bars)
    timing = _timing_audit(market_states)
    proxy_audit = _market_proxy_audit(
        vti_raw,
        vti_bars,
        market_states,
        bounded_vti_hash,
        alignment,
    )
    panel = _prepare_episode_panel(
        inherited_episodes,
        market_states,
        stock_bars,
    )
    quintile_source = panel.rename(
        columns={"m1c_tail_phase_v1": "tail_phase_v1"}
    )
    quintiles = freeze_opening_response_quintiles_v1(quintile_source)
    panel["opening_response_quintile_v1"] = [
        assign_opening_response_quintile_v1(
            None if pd.isna(value) else float(value),
            quintiles,
        )
        for value in panel["stock_relative_opening_response_v1"]
    ]
    configuration_hash = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    threshold_manifest = _threshold_manifest(
        thresholds,
        quintiles,
        configuration_hash=configuration_hash,
    )
    calibration = _calibration_table(thresholds)
    population, population_details = _population_accounting(checkpoints, panel)
    events = _unique_transition_events(panel)
    event_accounting = _event_accounting(panel, market_states)
    assessment = _required_outcome_table(panel, period="assessment")
    stress = _required_outcome_table(panel, period="stress")
    gap_diagnostics = _gap_alignment_diagnostics(panel)
    baseline = _baseline_tables(panel)
    primary = _primary_results(panel)
    severe_normal = _severe_vs_normal(panel)
    ranking, quintile_outcomes = _ranking_tables(panel, quintiles)
    sign_table = _transition_sign_stratification(panel)
    concentration = _concentration_report(panel)
    missingness = _missingness_table(panel, market_states)

    session_draws: list[pd.DataFrame] = []
    event_draws: list[pd.DataFrame] = []
    for period in ("assessment", "stress"):
        source = panel.loc[
            panel["partition"].eq(period)
            & panel["final_eligible_v1"].astype(bool)
        ].copy()
        session_draws.append(
            _cluster_bootstrap(
                source,
                period=period,
                cluster_column="session",
                cluster_type="session",
                seed=SESSION_BOOTSTRAP_SEED,
            )
        )
        event_draws.append(
            _cluster_bootstrap(
                source,
                period=period,
                cluster_column="opening_regime_cluster_v1",
                cluster_type="opening_transition_event",
                seed=EVENT_BOOTSTRAP_SEED,
            )
        )
    session_bootstrap = pd.concat(session_draws, ignore_index=True)
    event_bootstrap = pd.concat(event_draws, ignore_index=True)
    bootstrap_summary = _bootstrap_summary(
        pd.concat(
            [session_bootstrap, event_bootstrap],
            ignore_index=True,
        )
    )
    null_draws, null_metadata = _primary_null(panel)
    temporal = {
        period: _temporal_placebo(panel, period=period)
        for period in ("assessment", "stress")
    }
    null_placebo: dict[str, Any] = {
        "primary_null": null_metadata,
        "temporal_placebo": temporal,
        "null_design_selected_after_outcomes": False,
        "alternative_placebo_designs_tested": False,
    }
    leave_month = _leave_one_out(
        panel,
        dependency="month",
        dependency_column="month_v1",
    )
    leave_stock = _leave_one_out(
        panel,
        dependency="stock",
        dependency_column="stock",
    )
    leave_event = _leave_one_out(
        panel,
        dependency="transition_event",
        dependency_column="opening_transition_event_id_v1",
    )
    decision = _decision_contract(
        panel,
        primary,
        bootstrap_summary,
        null_draws,
        sign_table,
        severe_normal,
        ranking,
        leave_month,
        leave_stock,
        leave_event,
    )
    primary = _enrich_primary_results(
        primary,
        bootstrap_summary,
        decision,
    )
    ranking = _attach_ranking_intervals(ranking, bootstrap_summary)
    null_placebo["observed_assessment_null_p_values"] = decision[
        "assessment_null_p_values"
    ]
    null_placebo["holm_adjusted_response_mechanism_p_values"] = decision[
        "holm_adjusted_response_mechanism_p_values"
    ]

    _validate_before_write(
        panel=panel,
        market_states=market_states,
        reconciliation=reconciliation,
        primary=primary,
        null_draws=null_draws,
    )
    frozen_configuration = {
        **cast(
            dict[str, Any],
            json.loads(CONTRACT_PATH.read_text(encoding="utf-8")),
        ),
        "configuration_sha256": configuration_hash,
        "frozen_thresholds": thresholds.model_dump(mode="json"),
        "frozen_response_quintiles": quintiles.model_dump(mode="json"),
    }
    event_records = event_accounting.to_dict(orient="records")
    summary: dict[str, Any] = {
        "schema_version": "m1c-opening-market-transition-summary-v1",
        "experiment": "M1C Opening Market Transition V1",
        "retrospective_status": (
            "previously_opened_2025_not_untouched_confirmation"
        ),
        "prior_population_reconciliation": reconciliation_summary,
        "checkpoint_6_timing": timing,
        "canonical_market_proxy": {
            "identifier": "VTI",
            "available": True,
            "source": proxy_audit["source"],
        },
        "frozen_thresholds": thresholds.model_dump(mode="json"),
        "population_accounting": population.to_dict(orient="records"),
        "event_accounting": event_records,
        "assessment_outcomes": assessment.to_dict(orient="records"),
        "stress_outcomes": stress.to_dict(orient="records"),
        "primary_mechanism_results": primary.to_dict(orient="records"),
        "continuous_ranking": ranking.to_dict(orient="records"),
        "severe_vs_normal": severe_normal.to_dict(orient="records"),
        "decisions": decision,
        "null_and_placebo": null_placebo,
        "operational_blockers": [],
        "option_profitability": "Not tested",
        "execution_realism": {
            "unobservable_from_five_minute_bars": [
                "bid withdrawal",
                "ask withdrawal",
                "replenishment",
                "trade impact",
                "spread changes",
                "queue behaviour",
                "executable option outcomes",
            ]
        },
        "confirmations": {
            "m1c_unchanged": True,
            "tail_phase_v1_unchanged": True,
            "a1_unchanged": True,
            "fresh_episode_definition_unchanged": True,
            "contaminated_fields_used": False,
            "peer_slate_normalisation_used": False,
            "future_bars_used_for_predictors": False,
            "prospective_recorder_logging_only_integrated": True,
            "engineering_transfer_used_for_recalibration": False,
            "protected_2026_outcomes_accessed": False,
            "broker_accessed": False,
            "order_routing_enabled": False,
            "order_placed": False,
            "option_profitability_tested": False,
        },
    }
    report = _report_markdown(
        reconciliation_summary=reconciliation_summary,
        timing=timing,
        thresholds=thresholds,
        population=population,
        event_accounting=event_accounting,
        assessment=assessment,
        stress=stress,
        primary=primary,
        ranking=ranking,
        severe_normal=severe_normal,
        baseline=baseline,
        decision=decision,
        null_placebo=null_placebo,
    )
    _write_outputs(
        frozen_configuration=frozen_configuration,
        threshold_manifest=threshold_manifest,
        calibration=calibration,
        reconciliation=reconciliation,
        reconciliation_summary=reconciliation_summary,
        timing=timing,
        proxy_audit=proxy_audit,
        population=population,
        population_details=population_details,
        market_states=market_states,
        panel=panel,
        events=events,
        event_accounting=event_accounting,
        assessment=assessment,
        stress=stress,
        primary=primary,
        ranking=ranking,
        quintile_outcomes=quintile_outcomes,
        sign_table=sign_table,
        gap_diagnostics=gap_diagnostics,
        severe_normal=severe_normal,
        baseline=baseline,
        session_bootstrap=session_bootstrap,
        event_bootstrap=event_bootstrap,
        bootstrap_summary=bootstrap_summary,
        leave_month=leave_month,
        leave_stock=leave_stock,
        leave_event=leave_event,
        null_draws=null_draws,
        null_placebo=null_placebo,
        concentration=concentration,
        missingness=missingness,
        decision=decision,
        summary=summary,
        report=report,
    )

    input_records.append(
        {
            "path": str(VTI_PATH),
            "sha256": bounded_vti_hash,
            "bytes": None,
            "hash_scope": (
                "bounded Arrow rows timestamp>=2024-01-01 and <2026-01-01"
            ),
        }
    )
    output_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(PRIMARY.iterdir())
        if path.is_file() and path.name != "provenance_manifest_v1.json"
    }
    output_hashes[
        "reports/m1c_opening_market_transition_v1.md"
    ] = _sha256_file(REPORTS / "m1c_opening_market_transition_v1.md")
    provenance = {
        "schema_version": "m1c-opening-market-transition-provenance-v1",
        "generated_at_utc": datetime.now(tz=UTC),
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty_working_tree_status": _git("status", "--short"),
        "input_artifacts": input_records,
        "market_proxy_identity": proxy_audit,
        "data_date_boundaries": {
            "development": {
                "start": DEVELOPMENT_START,
                "end": DEVELOPMENT_END,
            },
            "assessment": {
                "start": ASSESSMENT_START,
                "end": ASSESSMENT_END,
            },
            "stress": {"start": STRESS_START, "end": STRESS_END},
            "protected": {
                "start": PROTECTED_START,
                "read_allowed": False,
            },
        },
        "exact_checkpoint_6_timing": timing,
        "expected_opening_bar_count": 6,
        "calibration_support": thresholds.model_dump(mode="json"),
        "threshold_values": thresholds.model_dump(mode="json"),
        "population_counts": population.to_dict(orient="records"),
        "exclusion_counts_and_reasons": missingness.to_dict(orient="records"),
        "unique_event_counts": event_records,
        "configuration_sha256": configuration_hash,
        "random_seeds": {
            "session_cluster_bootstrap": SESSION_BOOTSTRAP_SEED,
            "opening_transition_event_cluster_bootstrap": EVENT_BOOTSTRAP_SEED,
            "primary_null": NULL_SEED,
        },
        "replication_counts": {
            "session_cluster_bootstrap": BOOTSTRAP_DRAWS,
            "opening_transition_event_cluster_bootstrap": BOOTSTRAP_DRAWS,
            "primary_null": NULL_DRAWS,
        },
        "exact_commands": [
            RUN_COMMAND,
            (
                "rtk uv run pytest "
                "tests/test_m1c_opening_market_transition_v1.py "
                "tests/test_m1c_opening_market_transition_research_v1.py "
                "tests/test_m1c_opening_market_transition_v1_artifacts.py "
                "tests/test_m1c_opening_market_transition_v1_recorder.py -q"
            ),
            (
                "rtk uv run ruff check "
                "packages/stocker_prospective/src/stocker_prospective/"
                "opening_market_transition_v1.py "
                "packages/stocker_research/src/stocker_research/"
                "m1c_opening_market_transition_v1.py "
                "research/directional-readiness/"
                "20260728-m1c-opening-market-transition-v1/run_experiment.py"
            ),
            (
                "rtk uv run mypy "
                "packages/stocker_prospective/src/stocker_prospective/"
                "config.py "
                "packages/stocker_prospective/src/stocker_prospective/"
                "frozen_live_application.py "
                "packages/stocker_prospective/src/stocker_prospective/"
                "live_recorder.py "
                "packages/stocker_prospective/src/stocker_prospective/"
                "recorder_repository.py "
                "packages/stocker_prospective/src/stocker_prospective/"
                "recorder_v0.py "
                "packages/stocker_prospective/src/stocker_prospective/"
                "opening_market_transition_v1.py "
                "packages/stocker_research/src/stocker_research/"
                "m1c_opening_market_transition_v1.py"
            ),
        ],
        "frozen_regression": frozen_regression,
        "causality_confirmations": {
            "entry_bar_excluded": True,
            "partial_bars_excluded": True,
            "future_market_bars_used": False,
            "future_stock_bars_used": False,
            "outcome_driven_thresholds_used": False,
            "cross_sectional_normalisation_used": False,
            "stock_or_month_fitted_inputs_used": False,
            "alternative_windows_tested": False,
            "alternative_thresholds_tested": False,
            "alternative_market_proxies_tested": False,
        },
        "confirmations": summary["confirmations"],
        "protected_data_confirmation": {
            "bounded_reader_upper_limit_exclusive": PROTECTED_START,
            "protected_data_opened": False,
            "protected_outcomes_calculated": False,
            "protected_outcomes_displayed": False,
            "protected_outcomes_inspected": False,
        },
        "execution_confirmation": {
            "broker_access": False,
            "order_routing_enabled": False,
            "orders_submitted": False,
        },
        "output_sha256": output_hashes,
    }
    _write_json(PRIMARY / "provenance_manifest_v1.json", provenance)
    return summary


def main() -> int:
    try:
        summary = run()
    except ExperimentBlocked as error:
        print(json.dumps({"status": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "overall_decision": cast(
                    Mapping[str, Any],
                    summary["decisions"],
                )["overall_decision"],
                "artifact_directory": str(PRIMARY),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
